"""btc_core.py — Shared model and math utilities for Bitcoin Projections.

No Qt or matplotlib dependencies — importable from both the PyQt5 desktop
app (btc_projections.py) and the Plotly Dash web app (btc_web/app.py).

Note: btc_projections.py currently defines these inline for historical reasons.
The web app imports from here directly.
"""

import ast, json, pickle, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress, norm
from statsmodels.regression.quantile_regression import QuantReg

# ── constants ─────────────────────────────────────────────────────────────────

_SETTINGS_PATH = Path.home() / ".config" / "btc-projections" / "ui_settings.json"
_LOTS_PATH     = Path.home() / ".config" / "btc-projections" / "lots.json"

_DEFAULT_QS = [0.001, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

# ── settings persistence ──────────────────────────────────────────────────────

def _load_ui_settings():
    if _SETTINGS_PATH.exists():
        try:
            with open(_SETTINGS_PATH) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_ui_settings(d):
    _SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_SETTINGS_PATH, "w") as f:
        json.dump(d, f, indent=2)


def load_lots():
    """Load lots from ~/.config/btc-projections/lots.json."""
    if _LOTS_PATH.exists():
        try:
            with open(_LOTS_PATH) as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_lots(lots):
    """Persist lots to ~/.config/btc-projections/lots.json."""
    _LOTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_LOTS_PATH, "w") as f:
        json.dump(lots, f, indent=2)


# ── linestyle helpers (shared with desktop) ───────────────────────────────────

def _parse_ls(s):
    """Restore a linestyle spec that was stored as repr()."""
    if isinstance(s, str) and s.startswith("("):
        try:
            return ast.literal_eval(s)
        except Exception:
            return "-"
    return s


# ── price / time helpers ──────────────────────────────────────────────────────

def qr_price(q, t, qr_fits):
    """Return QR model price at years-since-genesis t for quantile q."""
    f = qr_fits[q]
    return 10.0 ** (f["intercept"] + f["slope"] * np.log10(np.asarray(t, float)))


def yr_to_t(cal_year, genesis=pd.Timestamp("2009-07-25")):
    """Calendar year (possibly fractional) → years since genesis (float)."""
    yr = int(cal_year)
    frac = float(cal_year) - yr
    base = (pd.Timestamp(f"{yr}-01-01") - genesis).days / 365.25
    return base + frac


def today_t(genesis=pd.Timestamp("2009-07-25")):
    """Today → years since genesis (float)."""
    return (pd.Timestamp.today() - genesis).days / 365.25


def today_year():
    """Today as a fractional calendar year."""
    return pd.Timestamp.today().year + (pd.Timestamp.today().day_of_year - 1) / 365.25


def fmt_price(p):
    """Format a USD price with comma separators or suffixes for large values."""
    if p >= 1e18:
        return f"${p/1e18:,.1f}Qi"
    if p >= 1e15:
        return f"${p/1e15:,.1f}Q"
    if p >= 1e12:
        return f"${p/1e12:,.1f}T"
    if p >= 1e9:
        return f"${p/1e9:,.1f}B"
    if p >= 1:
        return f"${p:,.0f}"
    return f"${p:.2f}"


def _fmt_btc(v):
    """Format a BTC quantity for axis labels."""
    if v >= 1000: return f"{v:.0f} BTC"
    if v >= 1:    return f"{v:.2f} BTC"
    if v >= 0.01: return f"{v:.4f} BTC"
    return f"{v:.6f} BTC"


# ── lot helpers ───────────────────────────────────────────────────────────────

def _find_lot_percentile(t, price, qr_fits):
    """Interpolate the QR percentile (0–1) for a given time t and price."""
    if not qr_fits:
        return 0.5
    sorted_qs = sorted(qr_fits.keys())
    t_safe = max(float(t), 0.5)
    log_p  = np.log10(max(float(price), 1e-10))
    log_ps = [np.log10(max(float(qr_price(q, t_safe, qr_fits)), 1e-10)) for q in sorted_qs]
    if log_p <= log_ps[0]:
        return sorted_qs[0]
    if log_p >= log_ps[-1]:
        return sorted_qs[-1]
    for i in range(len(sorted_qs) - 1):
        if log_ps[i] <= log_p <= log_ps[i + 1]:
            frac = (log_p - log_ps[i]) / (log_ps[i + 1] - log_ps[i] + 1e-30)
            return sorted_qs[i] + frac * (sorted_qs[i + 1] - sorted_qs[i])
    return sorted_qs[-1]


def leo_weighted_entry(lots):
    """
    Compute weighted-average entry price and time from a list of lot dicts.
    Returns (entry_price, entry_t, avg_pct_q, total_btc) or None if empty.
    """
    if not lots:
        return None
    total_w = sum(l["btc"] for l in lots)
    if total_w <= 0:
        return None
    genesis = pd.Timestamp("2009-07-25")
    ep = sum(l["price"] * l["btc"] for l in lots) / total_w
    et = sum((pd.Timestamp(l["date"]) - genesis).days / 365.25 * l["btc"]
             for l in lots) / total_w
    pct = sum(l["pct_q"] * l["btc"] for l in lots) / total_w
    return ep, et, pct, total_w


# ── model fitting ─────────────────────────────────────────────────────────────

def fit_qr_from_csv(csv_path, quantiles, genesis="2009-07-25", fit_min="2010-01-01"):
    """Re-fit QR model from a price CSV.  Returns (df, qr_fits, ols_intercept, ols_slope)."""
    df = pd.read_csv(csv_path)
    df.columns = ["Date", "Price"]
    df["date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    df = df[df["Price"] > 0].copy()
    gen = pd.Timestamp(genesis)
    df["years"]     = (df["date"] - gen).dt.days / 365.25
    df["log_years"] = np.log10(df["years"])
    df["log_price"] = np.log10(df["Price"])
    df = df.rename(columns={"Price": "price"})
    mask = df["date"] >= pd.Timestamp(fit_min)
    dfit = df[mask].copy()
    X    = np.column_stack([np.ones(len(dfit)), dfit["log_years"].values])
    ols_slope, ols_int, *_ = linregress(dfit["log_years"].values, dfit["log_price"].values)
    qr = {}
    for q in quantiles:
        res = QuantReg(dfit["log_price"].values, X).fit(q=q, max_iter=2000)
        qr[q] = {"intercept": float(res.params[0]), "slope": float(res.params[1]), "r2": 0.0}
    return df, qr, float(ols_int), float(ols_slope)


# ── model data ────────────────────────────────────────────────────────────────

def _find_model_data(explicit_path=None):
    """Search for model_data.pkl: explicit > bundle dir > cwd."""
    if explicit_path and Path(explicit_path).exists():
        return explicit_path
    # PyInstaller bundle
    base = getattr(sys, "_MEIPASS", None) or Path(__file__).parent
    bundled = Path(base) / "model_data.pkl"
    if bundled.exists():
        return str(bundled)
    # cwd (dev / project root)
    for candidate in (Path("model_data.pkl"),
                      Path(__file__).parent / "model_data.pkl",
                      Path(__file__).parent.parent / "model_data.pkl"):
        if candidate.exists():
            return str(candidate)
    return None


class ModelData:
    def __init__(self, path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        self._path = path
        self.qr_fits       = {float(k): v for k, v in d["qr_fits"].items()}
        self.QR_QUANTILES  = [float(q) for q in d["QR_QUANTILES"]]
        self.ols_intercept = d["ols_intercept"]
        self.ols_slope     = d["ols_slope"]
        self.genesis       = pd.Timestamp(d.get("GENESIS_DATE", "2009-07-25"))
        self.years_plot_bm = np.array(d["years_plot_bm"])
        self.support_bm    = np.array(d["support_plot_bm"])
        self.support_intercept = float(d.get("bm_support_intercept", -1.5594))
        self.support_slope     = float(d.get("bm_support_slope", 5.1248))
        self.comp_by_n     = [np.array(c) for c in d["bm_comp_by_n"]]
        self.bm_r2         = d["bm_r2_comp"]
        self.n_future_max  = d["bm_n_future_max"]
        self.price_dates   = d["price_dates"]
        self.price_years   = np.array(d["price_years"])
        self.price_prices  = np.array(d["price_prices"])
        self.qr_colors     = {float(k): v for k, v in d["qr_colors"].items()} if "qr_colors" in d else {}
        raw_ls = d.get("QR_LINESTYLES", {})
        self.qr_linestyles = {float(k): _parse_ls(v) for k, v in raw_ls.items()}
        # Visual config — .get() fallbacks so lean pkls (missing visual keys) don't crash
        _VIS_STR = {
            "PLOT_BG_COLOR": "#FFFFFF", "TEXT_COLOR": "#222222",
            "TITLE_COLOR": "#1A3060", "SPINE_COLOR": "#888888",
            "GRID_MAJOR_COLOR": "#BBBBBB", "GRID_MINOR_COLOR": "#E8E8E8",
            "DATA_COLOR": "#606060",
            "CAGR_SEG_C_LO": "#2166AC", "CAGR_SEG_C_MID1": "#F7F7F7",
            "CAGR_SEG_C_MID2": "#FF8C00", "CAGR_SEG_C_HI": "#CC1100",
        }
        _VIS_INT = {
            "DATA_PT_SIZE": 16, "DATA_PT_SIZE_ZOOM": 32,
            "ZOOM_YEAR_LO": 2025, "ZOOM_YEAR_HI": 2038,
            "CAGR_GRAD_STEPS": 24, "CAGR_HEATMAP_FONTSIZE": 6,
        }
        _VIS_FLOAT = {
            "ZOOM_PRICE_LO": 40000.0, "ZOOM_PRICE_HI": 1750000.0,
            "CAGR_SEG_B1": 5.0, "CAGR_SEG_B2": 16.0,
        }
        for key, default in _VIS_STR.items():
            setattr(self, key, d.get(key, default))
        for key, default in _VIS_INT.items():
            setattr(self, key, int(d.get(key, default)))
        for key, default in _VIS_FLOAT.items():
            setattr(self, key, float(d.get(key, default)))
        self.TABLE_YEARS = d.get("TABLE_YEARS", list(range(2025, 2041)))
        # Shrinking sigma parameters (fitted by tools/fit_sigma.py)
        self.bm_sigma0_up = d.get("bm_sigma0_up", 0.085)
        self.bm_alpha_up = d.get("bm_alpha_up", 0.132)
        self.bm_sigma0_down = d.get("bm_sigma0_down", 0.075)
        self.bm_alpha_down = d.get("bm_alpha_down", 0.218)

    def update_from_csv(self, csv_path):
        df, qr, ols_int, ols_sl = fit_qr_from_csv(
            csv_path, self.QR_QUANTILES, str(self.genesis.date()))
        self.qr_fits       = qr
        self.ols_intercept = ols_int
        self.ols_slope     = ols_sl
        self.price_years   = df["years"].values
        self.price_prices  = df["price"].values
        self.price_dates   = df["date"].dt.strftime("%Y-%m-%d").tolist()

    def save_user_override(self):
        cfg_dir = Path.home() / ".config" / "btc-projections"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        dst = cfg_dir / "model_data.pkl"
        with open(self._path, "rb") as f:
            d = pickle.load(f)
        d["qr_fits"]       = {str(k): v for k, v in self.qr_fits.items()}
        d["ols_intercept"] = self.ols_intercept
        d["ols_slope"]     = self.ols_slope
        d["price_dates"]   = list(self.price_dates)
        d["price_years"]   = list(self.price_years)
        d["price_prices"]  = list(self.price_prices)
        with open(dst, "wb") as f:
            pickle.dump(d, f, protocol=4)
        return str(dst)


def load_model_data(explicit_path=None):
    """Convenience: find model_data.pkl and return a ModelData instance."""
    path = _find_model_data(explicit_path)
    if path is None:
        raise FileNotFoundError(
            "model_data.pkl not found. Run SP.ipynb export cell first, "
            "or pass an explicit path.")
    return ModelData(path)


# ── R² computation ────────────────────────────────────────────────────────────

def _compute_log_r2(actual_prices, predicted_prices):
    """R² in log10 space. Returns float or None if degenerate."""
    log_a = np.log10(np.maximum(np.asarray(actual_prices, float), 1e-10))
    log_p = np.log10(np.maximum(np.asarray(predicted_prices, float), 1e-10))
    ss_res = np.sum((log_a - log_p) ** 2)
    ss_tot = np.sum((log_a - np.mean(log_a)) ** 2)
    if ss_tot == 0:
        return None
    return float(1.0 - ss_res / ss_tot)


def compute_model_r2(mdl, price_years, price_prices):
    """Compute per-quantile R² for any model with price_at() and quantiles."""
    mdl.r2_per_quantile = {}
    mask = price_years >= 1.0
    t = price_years[mask]
    actual = price_prices[mask]
    if hasattr(mdl, 'quantiles') and mdl.quantiles:
        for q in mdl.quantiles:
            try:
                predicted = np.asarray(mdl.price_at(q, t), float)
                r2 = _compute_log_r2(actual, predicted)
                if r2 is not None:
                    mdl.r2_per_quantile[q] = r2
            except Exception:
                pass
    elif hasattr(mdl, 'price_at'):
        try:
            predicted = np.asarray(mdl.price_at(0.5, t), float)
            r2 = _compute_log_r2(actual, predicted)
            if r2 is not None:
                mdl.r2_per_quantile[0.5] = r2
        except Exception:
            pass


# ── PriceModel protocol + implementations ────────────────────────────────────

from typing import Protocol, runtime_checkable


@runtime_checkable
class PriceModel(Protocol):
    """Protocol all price models must satisfy.

    Implement this to add a new model — then register in app.py and the UI
    auto-discovers it via PRICE_MODELS iteration.
    """
    name: str              # human-readable name ("Bubble Model", "Power Law")
    short_name: str        # registry key ("bub", "pl", "s2f")
    quantized: bool        # True → has fits dict → MC-compatible
    quantiles: list        # sorted list of available quantiles
    colors: dict           # {q: "#hex"} for trace coloring
    fits: dict | None      # {q: {"intercept","slope"}} or None
    dash_style: str        # Plotly dash pattern ("solid", "dot", "longdash")

    def price_at(self, q, t): ...
    def interp_price(self, q, t): ...
    def find_percentile(self, t, price): ...


class _FitsBasedModel:
    """Base for models whose quantile bands are log-linear in log10(t).

    Subclasses must set self.fits, self.quantiles, and self.colors.
    """
    quantized = True

    def price_at(self, q, t):
        """Price at quantile q, time t (years since genesis)."""
        f = self.fits[q]
        return 10.0 ** (f["intercept"] + f["slope"] * np.log10(np.asarray(t, float)))

    def interp_price(self, q, t):
        """Log-space interpolated price for arbitrary quantile (e.g. Q7.5%)."""
        if q in self.fits:
            return float(self.price_at(q, t))
        sorted_qs = self.quantiles
        lo = max((qq for qq in sorted_qs if qq <= q), default=sorted_qs[0])
        hi = min((qq for qq in sorted_qs if qq >= q), default=sorted_qs[-1])
        if lo == hi:
            return float(self.price_at(lo, t))
        frac = (q - lo) / (hi - lo)
        p_lo = np.log10(float(self.price_at(lo, t)))
        p_hi = np.log10(float(self.price_at(hi, t)))
        return 10.0 ** (p_lo + frac * (p_hi - p_lo))

    def find_percentile(self, t, price):
        """Interpolate the QR percentile (0–1) for a given time and price."""
        sorted_qs = self.quantiles
        if not sorted_qs:
            return 0.5
        t_safe = max(float(t), 0.5)
        log_p = np.log10(max(float(price), 1e-10))
        log_ps = [np.log10(max(float(self.price_at(q, t_safe)), 1e-10))
                  for q in sorted_qs]
        if log_p <= log_ps[0]:
            return sorted_qs[0]
        if log_p >= log_ps[-1]:
            return sorted_qs[-1]
        for i in range(len(sorted_qs) - 1):
            if log_ps[i] <= log_p <= log_ps[i + 1]:
                frac = (log_p - log_ps[i]) / (log_ps[i + 1] - log_ps[i] + 1e-30)
                return sorted_qs[i] + frac * (sorted_qs[i + 1] - sorted_qs[i])
        return sorted_qs[-1]


class QuantileRegressionModel(_FitsBasedModel):
    """Raw quantile regression fits — model-free, purely empirical.

    Straight lines in log-log space. Each quantile has its own independently
    fitted slope and intercept. This is what BubbleModel used to be before
    the shrinking Gaussian conversion.
    """
    name = "Quantile Regression"
    short_name = "qr"
    legend_name = "QR"
    dash_style = "dash"

    def __init__(self, md):
        self.fits = md.qr_fits
        self.colors = dict(md.qr_colors) if md.qr_colors else {}
        self.quantiles = sorted(md.qr_fits.keys())


class _CompositeModel:
    """Base for models with a shaped composite median and asymmetric shrinking
    Gaussian bands.

    σ_up(t) = σ₀_up × t^(-α_up) for quantiles ≥ 0.5
    σ_down(t) = σ₀_down × t^(-α_down) for quantiles < 0.5

    Subclasses must set self._t_grid, self._log_comp, and call self._init_bands().
    """
    quantized = True

    def _composite_log10(self, t):
        """Interpolate composite curve in log10 space at arbitrary t."""
        t = np.asarray(t, float)
        return np.interp(t, self._t_grid, self._log_comp)

    def _sigma_at(self, t, q):
        """Compute σ at time t for quantile q (asymmetric, shrinking)."""
        t = np.maximum(np.asarray(t, float), 0.5)
        if q >= 0.5:
            return self._sigma0_up * t ** (-self._alpha_up)
        else:
            return self._sigma0_down * t ** (-self._alpha_down)

    def price_at(self, q, t):
        """Price at quantile q, time t (years since genesis)."""
        t_arr = np.asarray(t, float)
        log_median = self._composite_log10(t_arr)
        z = norm.ppf(q)
        sigma = self._sigma_at(t_arr, q)
        return 10.0 ** (log_median + z * sigma)

    def interp_price(self, q, t):
        """Log-space interpolated price for arbitrary quantile."""
        if q in self.fits:
            return float(self.price_at(q, t))
        sorted_qs = self.quantiles
        lo = max((qq for qq in sorted_qs if qq <= q), default=sorted_qs[0])
        hi = min((qq for qq in sorted_qs if qq >= q), default=sorted_qs[-1])
        if lo == hi:
            return float(self.price_at(lo, t))
        frac = (q - lo) / (hi - lo)
        p_lo = np.log10(float(self.price_at(lo, t)))
        p_hi = np.log10(float(self.price_at(hi, t)))
        return 10.0 ** (p_lo + frac * (p_hi - p_lo))

    def find_percentile(self, t, price):
        """Reverse lookup: time + price → quantile."""
        sorted_qs = self.quantiles
        if not sorted_qs:
            return 0.5
        t_safe = max(float(t), 0.5)
        log_p = np.log10(max(float(price), 1e-10))
        log_ps = [np.log10(max(float(self.price_at(q, t_safe)), 1e-10))
                  for q in sorted_qs]
        if log_p <= log_ps[0]:
            return sorted_qs[0]
        if log_p >= log_ps[-1]:
            return sorted_qs[-1]
        for i in range(len(sorted_qs) - 1):
            if log_ps[i] <= log_p <= log_ps[i + 1]:
                frac = (log_p - log_ps[i]) / (log_ps[i + 1] - log_ps[i] + 1e-30)
                return sorted_qs[i] + frac * (sorted_qs[i + 1] - sorted_qs[i])
        return sorted_qs[-1]

    def _init_bands(self, sigma0_up, alpha_up, sigma0_down, alpha_down, quantiles):
        """Initialize σ parameters and build fits dict."""
        self._sigma0_up = sigma0_up
        self._alpha_up = alpha_up
        self._sigma0_down = sigma0_down
        self._alpha_down = alpha_down
        self.fits = {}
        for q in quantiles:
            self.fits[q] = {"z": float(norm.ppf(q))}
        self.quantiles = sorted(self.fits.keys())

    @property
    def comp_by_n(self):
        return self._comp_by_n

    @property
    def support_plot(self):
        return self._support_plot

    @property
    def t_grid(self):
        return self._t_grid

    @property
    def bm_r2(self):
        return self._bm_r2

    @property
    def n_future_max(self):
        return self._n_future_max


class BubbleModel(_CompositeModel):
    """Bubble model with asymmetric shrinking Gaussian bands around composite."""
    name = "Bubble Model"
    short_name = "bub"
    legend_name = "BM"
    dash_style = "solid"

    def __init__(self, md):
        # Composite curve (max future bubbles)
        self._t_grid = np.asarray(md.years_plot_bm, float)
        comp = md.comp_by_n[-1]
        self._log_comp = np.log10(np.maximum(np.asarray(comp, float), 1e-10))

        # Shrinking σ parameters (from pkl, fitted by tools/fit_sigma.py)
        self._init_bands(
            getattr(md, 'bm_sigma0_up', 0.085),
            getattr(md, 'bm_alpha_up', 0.132),
            getattr(md, 'bm_sigma0_down', 0.075),
            getattr(md, 'bm_alpha_down', 0.218),
            md.QR_QUANTILES,
        )

        # Colors: from pkl if present, otherwise generate thermal defaults.
        # app.py overwrites these with the full thermal palette at startup.
        if md.qr_colors:
            self.colors = dict(md.qr_colors)
        else:
            self.colors = {q: f"#{int(255*q):02x}80{int(255*(1-q)):02x}"
                           for q in self.quantiles}


class PowerLawModel(_FitsBasedModel):
    """OLS power law with Gaussian quantile bands.

    All bands share the same slope (OLS slope) but have different intercepts
    shifted by z_q * sigma where sigma is the OLS residual standard deviation.
    This means the bands are parallel lines in log-log space.
    """
    name = "Power Law"
    short_name = "pl"
    legend_name = "PL"
    dash_style = "dot"

    def __init__(self, ols_intercept, ols_slope, price_years, price_prices,
                 genesis, quantiles):
        # Compute OLS residual sigma
        mask = price_years >= 1.0  # skip very early data
        ly = np.log10(price_years[mask])
        lp = np.log10(price_prices[mask])
        predicted = ols_intercept + ols_slope * ly
        residuals = lp - predicted
        sigma = float(np.std(residuals))

        # Build fits: each quantile is the OLS line shifted by z_q * sigma
        self.fits = {}
        for q in quantiles:
            z = norm.ppf(q)
            self.fits[q] = {
                "intercept": ols_intercept + z * sigma,
                "slope": ols_slope,
            }
        self.quantiles = sorted(self.fits.keys())

        # Cool blue/purple palette — visually distinct from Bubble's warm colors
        self._build_colors()

    def _build_colors(self):
        self.colors = {}
        n = len(self.quantiles)
        for i, q in enumerate(self.quantiles):
            frac = i / max(n - 1, 1)
            r = int(40 + 140 * frac)    # 40 → 180
            g = int(60 + 40 * frac)     # 60 → 100
            b = int(200 - 30 * frac)    # 200 → 170
            self.colors[q] = f"#{r:02x}{g:02x}{b:02x}"


class LPPLModel:
    """Damped Log-Periodic Power Law model with Gaussian quantile bands.

    Fits: log10(price) = A + B*log10(t) + C*t^(-d)*cos(w*ln(t) + phi)

    The oscillatory term captures Bitcoin's ~4-year bubble cycles in log-time,
    with damping (d > 0) reflecting decreasing volatility over time. Quantile
    bands are generated by shifting the median curve by z_q * sigma (Gaussian),
    like PowerLawModel — all bands share the same oscillatory shape.
    """
    name = "LPPL"
    short_name = "lppl"
    legend_name = "LPPL"
    dash_style = "dashdot"
    quantized = True

    # Best-fit parameters from differential evolution on full BTC history
    # (genesis = 2009-07-25)
    _A   = -1.153965
    _B   =           5.079504
    _C   =           0.734010
    _W   =           7.558602
    _PHI =           1.376420
    _D   =           0.608070

    def __init__(self, price_years, price_prices, quantiles):
        # Compute residual sigma from historical data
        mask = price_years >= 1.0
        t = price_years[mask]
        lp = np.log10(price_prices[mask])
        predicted = self._lppl_log10(t)
        sigma = float(np.std(lp - predicted))

        # Build quantile fits as median shift
        self.fits = {}
        self._sigma = sigma
        for q in quantiles:
            z = norm.ppf(q)
            self.fits[q] = {"z_shift": z * sigma}
        self.quantiles = sorted(self.fits.keys())
        self._build_colors()

    def _lppl_log10(self, t):
        """Evaluate damped LPPL median in log10 space."""
        t = np.asarray(t, float)
        t_safe = np.maximum(t, 0.1)
        envelope = self._C * t_safe ** (-self._D)
        return self._A + self._B * np.log10(t_safe) + envelope * np.cos(
            self._W * np.log(t_safe) + self._PHI)

    component_names = [
        "A (constant)",
        "B\u00b7log\u2081\u2080(t)",
        "damped osc (\u03c9_log)",
    ]

    def components(self, t):
        """Additive terms in log10 space. sum(values) == _lppl_log10(t)."""
        t = np.asarray(t, float)
        t_safe = np.maximum(t, 0.1)
        return {
            "A (constant)":            np.full_like(t_safe, self._A),
            "B\u00b7log\u2081\u2080(t)": self._B * np.log10(t_safe),
            "damped osc (\u03c9_log)":  self._C * t_safe ** (-self._D) * np.cos(
                self._W * np.log(t_safe) + self._PHI),
        }

    def price_at(self, q, t):
        """Price at quantile q, time t (years since genesis)."""
        t_arr = np.asarray(t, float)
        log_median = self._lppl_log10(t_arr)
        shift = self.fits[q]["z_shift"]
        return 10.0 ** (log_median + shift)

    def interp_price(self, q, t):
        """Log-space interpolated price for arbitrary quantile."""
        if q in self.fits:
            return float(self.price_at(q, t))
        sorted_qs = self.quantiles
        lo = max((qq for qq in sorted_qs if qq <= q), default=sorted_qs[0])
        hi = min((qq for qq in sorted_qs if qq >= q), default=sorted_qs[-1])
        if lo == hi:
            return float(self.price_at(lo, t))
        frac = (q - lo) / (hi - lo)
        p_lo = np.log10(float(self.price_at(lo, t)))
        p_hi = np.log10(float(self.price_at(hi, t)))
        return 10.0 ** (p_lo + frac * (p_hi - p_lo))

    def find_percentile(self, t, price):
        """Reverse lookup: time + price → quantile."""
        sorted_qs = self.quantiles
        if not sorted_qs:
            return 0.5
        t_safe = max(float(t), 0.5)
        log_p = np.log10(max(float(price), 1e-10))
        log_ps = [np.log10(max(float(self.price_at(q, t_safe)), 1e-10))
                  for q in sorted_qs]
        if log_p <= log_ps[0]:
            return sorted_qs[0]
        if log_p >= log_ps[-1]:
            return sorted_qs[-1]
        for i in range(len(sorted_qs) - 1):
            if log_ps[i] <= log_p <= log_ps[i + 1]:
                frac = (log_p - log_ps[i]) / (log_ps[i + 1] - log_ps[i] + 1e-30)
                return sorted_qs[i] + frac * (sorted_qs[i + 1] - sorted_qs[i])
        return sorted_qs[-1]

    def _build_colors(self):
        """Green/teal palette — visually distinct from Bubble and PL."""
        self.colors = {}
        n = len(self.quantiles)
        for i, q in enumerate(self.quantiles):
            frac = i / max(n - 1, 1)
            r = int(20 + 80 * frac)     # 20 → 100
            g = int(160 + 60 * frac)    # 160 → 220
            b = int(120 + 40 * frac)    # 120 → 160
            self.colors[q] = f"#{r:02x}{g:02x}{b:02x}"


class LPPL2Model(LPPLModel):
    """Two-frequency LPPL with independent oscillation frequencies.

    Fits: log10(price) = A + B*log10(t) + C1*t^(-D)*cos(W1*ln(t)+φ1) + C2*cos(W2*ln(t)+φ2)

    Primary oscillation (W1) is damped by t^(-D); secondary oscillation (W2)
    is undamped — it persists as a permanent structural feature. W2 is
    independent of W1 (not constrained to 2×W1).
    """
    name = "LPPL\u2082"
    short_name = "lp2"
    legend_name = "LPPL\u2082"
    dash_style = "dashdot"

    # All 9 params jointly fitted by tools/fit_lppl2.py
    _A   = -1.130788
    _B   =              5.038579
    _C   =              0.705592
    _W   =              7.377563
    _PHI =              1.582787
    _D   =              0.566087
    _C2  =              0.168844
    _W2  =          20.903103
    _PHI2 = -1.155400

    def _lppl_log10(self, t):
        """Evaluate two-frequency LPPL median in log10 space.

        Primary oscillation is damped; secondary oscillation is undamped.
        """
        t = np.asarray(t, float)
        t_safe = np.maximum(t, 0.1)
        term1 = self._C * t_safe ** (-self._D) * np.cos(self._W * np.log(t_safe) + self._PHI)
        term2 = self._C2 * np.cos(self._W2 * np.log(t_safe) + self._PHI2)
        return self._A + self._B * np.log10(t_safe) + term1 + term2

    component_names = [
        "A (constant)",
        "B\u00b7log\u2081\u2080(t)",
        "damped osc (\u03c9\u2081)",
        "undamped osc (\u03c9\u2082)",
    ]

    def components(self, t):
        t = np.asarray(t, float)
        t_safe = np.maximum(t, 0.1)
        return {
            "A (constant)":                np.full_like(t_safe, self._A),
            "B\u00b7log\u2081\u2080(t)":    self._B * np.log10(t_safe),
            "damped osc (\u03c9\u2081)":    self._C * t_safe ** (-self._D) * np.cos(
                self._W * np.log(t_safe) + self._PHI),
            "undamped osc (\u03c9\u2082)":  self._C2 * np.cos(
                self._W2 * np.log(t_safe) + self._PHI2),
        }


class LPPL3Model(LPPL2Model):
    """Three-frequency LPPL: damped primary + two undamped secondary oscillations.

    Fits: log10(price) = A + B*log10(t) + C1*t^(-D)*cos(W1*ln(t)+φ1)
                       + C2*cos(W2*ln(t)+φ2) + C3*cos(W3*ln(t)+φ3)

    Initial search started at W3=13.3 (from FFT residual analysis), but
    the optimizer settled at W3≈10 — the peak at ω≈13 turned out to be
    an intermodulation product (W2 − W1 ≈ 13.5), not a third oscillation.
    W3/W1 ≈ √2, W3/W2 ≈ 0.49.
    """
    name = "LPPL\u2083"
    short_name = "lp3"
    legend_name = "LPPL\u2083"
    dash_style = "dashdot"

    # All 12 params jointly fitted by tools/fit_lppl3.py
    _A   = -1.094406
    _B   =        4.966719
    _C   =        0.614085
    _W   =        7.122128
    _PHI =        1.890711
    _D   =        0.366087
    _C2  =        0.171176
    _W2  =      10.081433
    _PHI2 = -2.164585
    _C3  =        0.178590
    _W3  =      20.804444
    _PHI3 = -0.995478

    def _lppl_log10(self, t):
        """Evaluate three-frequency LPPL median in log10 space."""
        t = np.asarray(t, float)
        t_safe = np.maximum(t, 0.1)
        term1 = self._C * t_safe ** (-self._D) * np.cos(self._W * np.log(t_safe) + self._PHI)
        term2 = self._C2 * np.cos(self._W2 * np.log(t_safe) + self._PHI2)
        term3 = self._C3 * np.cos(self._W3 * np.log(t_safe) + self._PHI3)
        return self._A + self._B * np.log10(t_safe) + term1 + term2 + term3

    component_names = [
        "A (constant)",
        "B\u00b7log\u2081\u2080(t)",
        "damped osc (\u03c9\u2081)",
        "undamped osc (\u03c9\u2082)",
        "undamped osc (\u03c9\u2083)",
    ]

    def components(self, t):
        t = np.asarray(t, float)
        t_safe = np.maximum(t, 0.1)
        return {
            "A (constant)":                np.full_like(t_safe, self._A),
            "B\u00b7log\u2081\u2080(t)":    self._B * np.log10(t_safe),
            "damped osc (\u03c9\u2081)":    self._C * t_safe ** (-self._D) * np.cos(
                self._W * np.log(t_safe) + self._PHI),
            "undamped osc (\u03c9\u2082)":  self._C2 * np.cos(
                self._W2 * np.log(t_safe) + self._PHI2),
            "undamped osc (\u03c9\u2083)":  self._C3 * np.cos(
                self._W3 * np.log(t_safe) + self._PHI3),
        }


class LPPLModelW(LPPLModel):
    """LPPL with log-time-uniform (1/t) weighting.

    Same formula as LPPLModel, but fitted with residuals weighted by 1/t
    so each unit of ln(t) contributes equally to the loss. Emphasizes
    early-history structure over the 2020-2024 bubble era.
    """
    name = "LPPL (weighted)"
    short_name = "lppl_w"
    legend_name = "LPPL\u1d65\u1d65"
    _A   = -1.086800
    _B   =   4.987100
    _C   =   0.552500
    _W   =   7.416000
    _PHI =   1.533100
    _D   =   0.356800


class LPPL2ModelW(LPPL2Model):
    """LPPL\u2082 with log-time-uniform weighting."""
    name = "LPPL\u2082 (weighted)"
    short_name = "lp2_w"
    legend_name = "LPPL\u2082\u1d65\u1d65"
    _A   = -1.102500
    _B   =   4.980300
    _C   =   0.500900
    _W   =   6.857500
    _PHI =   2.196600
    _D   =   0.301900
    _C2  =   0.188000
    _W2  =   9.259900
    _PHI2 = -0.846900


class LPPL3ModelW(LPPL3Model):
    """LPPL\u2083 with log-time-uniform weighting."""
    name = "LPPL\u2083 (weighted)"
    short_name = "lp3_w"
    legend_name = "LPPL\u2083\u1d65\u1d65"
    _A   = -1.103800
    _B   =   4.984500
    _C   =   0.443200
    _W   =   6.760600
    _PHI =   2.284100
    _D   =   0.245900
    _C2  =   0.209000
    _W2  =   9.083400
    _PHI2 = -0.530300
    _C3  =   0.132000
    _W3  =  17.258100
    _PHI3 =  1.054900


class LPPL4Model(LPPL3Model):
    """Four-frequency LPPL: damped primary + three undamped secondary oscillations.

    Fits: log10(price) = A + B*log10(t) + C1*t^(-D)*cos(W1*ln(t)+φ1)
                       + C2*cos(W2*ln(t)+φ2) + C3*cos(W3*ln(t)+φ3) + C4*cos(W4*ln(t)+φ4)

    CAUTION: The 4th frequency is NOT stable under log-time weighting.
    Unweighted fits find W2≈13 (close to 2×W1, looks like 2nd harmonic)
    but weighted fits find ~17 (ratio 2.5, non-harmonic). Use with awareness
    that one of the 4 frequencies may be a recent-era artifact. See the
    "LPPL Weighting & Regime Shifts" section for details.
    """
    name = "LPPL\u2084"
    short_name = "lp4"
    legend_name = "LPPL\u2084"
    dash_style = "dashdot"

    # All 15 params jointly fitted by tools/fit_lppl4.py (unweighted)
    _A   = -1.128637
    _B   =        5.014683
    _C   =        0.576847
    _W   =        6.839028
    _PHI =        2.262803
    _D   =        0.402598
    _C2  =        0.189223
    _W2  =      9.338593
    _PHI2 = -1.140056
    _C3  =        0.134062
    _W3  =       13.318485
    _PHI3 = -2.237722
    _C4  =        0.171483
    _W4  =     20.904759
    _PHI4 = -1.137861

    def _lppl_log10(self, t):
        """Evaluate four-frequency LPPL median in log10 space."""
        t = np.asarray(t, float)
        t_safe = np.maximum(t, 0.1)
        term1 = self._C * t_safe ** (-self._D) * np.cos(self._W * np.log(t_safe) + self._PHI)
        term2 = self._C2 * np.cos(self._W2 * np.log(t_safe) + self._PHI2)
        term3 = self._C3 * np.cos(self._W3 * np.log(t_safe) + self._PHI3)
        term4 = self._C4 * np.cos(self._W4 * np.log(t_safe) + self._PHI4)
        return self._A + self._B * np.log10(t_safe) + term1 + term2 + term3 + term4

    component_names = [
        "A (constant)",
        "B\u00b7log\u2081\u2080(t)",
        "damped osc (\u03c9\u2081)",
        "undamped osc (\u03c9\u2082)",
        "undamped osc (\u03c9\u2083)",
        "undamped osc (\u03c9\u2084)",
    ]

    def components(self, t):
        t = np.asarray(t, float)
        t_safe = np.maximum(t, 0.1)
        return {
            "A (constant)":                np.full_like(t_safe, self._A),
            "B\u00b7log\u2081\u2080(t)":    self._B * np.log10(t_safe),
            "damped osc (\u03c9\u2081)":    self._C * t_safe ** (-self._D) * np.cos(
                self._W * np.log(t_safe) + self._PHI),
            "undamped osc (\u03c9\u2082)":  self._C2 * np.cos(
                self._W2 * np.log(t_safe) + self._PHI2),
            "undamped osc (\u03c9\u2083)":  self._C3 * np.cos(
                self._W3 * np.log(t_safe) + self._PHI3),
            "undamped osc (\u03c9\u2084)":  self._C4 * np.cos(
                self._W4 * np.log(t_safe) + self._PHI4),
        }


class LPPL4ModelW(LPPL4Model):
    """LPPL\u2084 with log-time-uniform weighting."""
    name = "LPPL\u2084 (weighted)"
    short_name = "lp4_w"
    legend_name = "LPPL\u2084\u1d65\u1d65"
    _A   = -1.101500
    _B   =   4.976300
    _C   =   0.491600
    _W   =   6.861300
    _PHI =   2.190700
    _D   =   0.276400
    _C2  =   0.190000
    _W2  =   9.313500
    _PHI2 = -0.889300
    _C3  =   0.119700
    _W3  =  20.921200
    _PHI3 = -1.203000
    _C4  =   0.108400
    _W4  =  17.087700
    _PHI4 =  1.336900


class LPPL4ModelN13(LPPL4Model):
    """LPPL\u2084 with ω=13 intermod band excluded (secondary frequencies constrained to avoid 11.5-14.5).

    Secondaries at ω≈9.9, 17.5, 20.9 (ratios 1.41, 2.48, 2.96). R²=0.9905.
    """
    name = "LPPL\u2084 (no \u03c9\u224813)"
    short_name = "lp4_n13"
    legend_name = "LPPL\u2084-n13"
    _A   = -1.096074
    _B   =   4.966763
    _C   =   0.557848
    _W   =   7.053735
    _PHI =   1.972293
    _D   =   0.306513
    _C2  =   0.146182
    _W2  =  20.900338
    _PHI2 = -1.214332
    _C3  =   0.186290
    _W3  =   9.931483
    _PHI3 = -1.893553
    _C4  =   0.093978
    _W4  =  17.456360
    _PHI4 =  0.827859


class LPPL4ModelWN13(LPPL4ModelN13):
    """LPPL\u2084 weighted + ω=13 excluded.

    Secondaries at ω≈9.3, 17.1, 20.9 (same as LP4 weighted, since
    LP4 weighted's secondaries already avoid the 11.5-14.5 band).
    """
    name = "LPPL\u2084 (weighted, no \u03c9\u224813)"
    short_name = "lp4_w_n13"
    legend_name = "LPPL\u2084\u1d65\u1d65-n13"
    _A   = -1.101466
    _B   =   4.976301
    _C   =   0.491568
    _W   =   6.861297
    _PHI =   2.190663
    _D   =   0.276438
    _C2  =   0.108395
    _W2  =  17.087659
    _PHI2 =  1.336939
    _C3  =   0.119697
    _W3  =  20.921188
    _PHI3 = -1.202956
    _C4  =   0.189996
    _W4  =   9.313489
    _PHI4 = -0.889292


class HybPPLModel(LPPLModel):
    """Hybrid Log+Linear PPL: log-periodic damped + linear-periodic undamped.

    Fits: log10(price) = A + B*log10(t) + C1*t^(-D)*cos(ω_log*ln(t)+φ1)
                       + C2*cos(ω_cal*t+φ2)

    Combines LPPL's log-periodic damped oscillation (captures early-Bitcoin
    self-similarity) with a linear-periodic undamped term (captures the
    halving cycle). 9 parameters — same count as LPPL₂.

    _W is the log-time angular frequency (like LPPL).
    _W2 is the calendar angular frequency in rad/yr (like LinPPL).
    """
    name = "HybPPL"
    short_name = "hybppl"
    legend_name = "HybPPL"
    dash_style = "dashdot"

    # Fitted parameters (will be overwritten by fit_hybppl.py --update)
    _A   = -1.146885  
    _B   =        5.051488  
    _C   =        0.689933  
    _W   =        7.420135  
    _PHI =        1.453241  
    _D   =        0.708316  
    _C2  =        0.233035  
    _W2  =        1.733095  
    _PHI2 = -1.922641  

    def _lppl_log10(self, t):
        """Evaluate hybrid model: log-periodic damped + linear-periodic undamped."""
        t = np.asarray(t, float)
        t_safe = np.maximum(t, 0.1)
        damped = self._C * t_safe ** (-self._D) * np.cos(self._W * np.log(t_safe) + self._PHI)
        undamped = self._C2 * np.cos(self._W2 * t_safe + self._PHI2)
        return self._A + self._B * np.log10(t_safe) + damped + undamped

    component_names = [
        "A (constant)",
        "B\u00b7log\u2081\u2080(t)",
        "damped log osc (\u03c9_log)",
        "undamped cal osc (\u03c9_cal)",
    ]

    def components(self, t):
        """Hybrid: log-periodic damped + linear-periodic undamped."""
        t = np.asarray(t, float)
        t_safe = np.maximum(t, 0.1)
        return {
            "A (constant)":                        np.full_like(t_safe, self._A),
            "B\u00b7log\u2081\u2080(t)":            self._B * np.log10(t_safe),
            "damped log osc (\u03c9_log)":          self._C * t_safe ** (-self._D) * np.cos(
                self._W * np.log(t_safe) + self._PHI),
            "undamped cal osc (\u03c9_cal)":        self._C2 * np.cos(
                self._W2 * t_safe + self._PHI2),
        }


class HybPPLExcessModel(LPPLModel):
    """HybPPL oscillators fit to BM-excess (log_price - BM support).

    Fits: log10(price) = A_sup + B_sup*log10(t)
                       + a0
                       + C1*t^(-D)*cos(W_log*ln(t) + PHI1)
                       + C2*cos(W_cal*t + PHI2)

    A_sup and B_sup are pulled from ModelData at instantiation (dynamic
    trend tracking). The 8 oscillation params are refit daily via
    tools/fit_hybppl_excess.py --update.
    """
    name = "HybPPL (excess)"
    short_name = "hybppl_ex"
    legend_name = "HybPPL (ex)"
    dash_style = "dashdot"

    # Fitted oscillation parameters (will be overwritten by fit_hybppl_excess.py --update)
    _a0    =          0.349890  
    _C1    =          0.642075  
    _W_log =          7.480742  
    _PHI1  =          1.427254  
    _D     =          0.660584  
    _C2    =          0.231473  
    _W_cal =          1.748966  
    _PHI2  = -2.100458  

    def __init__(self, price_years, price_prices, quantiles,
                 a_sup=None, b_sup=None):
        self._A_sup = float(a_sup) if a_sup is not None else 0.0
        self._B_sup = float(b_sup) if b_sup is not None else 0.0
        super().__init__(price_years, price_prices, quantiles)

    def _lppl_log10(self, t):
        """BM support + constant + damped log-periodic + undamped calendar."""
        t_arr = np.asarray(t, float)
        t_safe = np.maximum(t_arr, 0.1)
        support = self._A_sup + self._B_sup * np.log10(t_safe)
        damped = self._C1 * t_safe ** (-self._D) * np.cos(
            self._W_log * np.log(t_safe) + self._PHI1)
        undamped = self._C2 * np.cos(self._W_cal * t_safe + self._PHI2)
        return support + self._a0 + damped + undamped

    component_names = [
        "A_sup",
        "B_sup\u00b7log\u2081\u2080(t)",
        "a\u2080",
        "damped log osc (\u03c9_log)",
        "undamped cal osc (\u03c9_cal)",
    ]

    def components(self, t):
        """BM support + constant + damped log-periodic + undamped calendar."""
        t = np.asarray(t, float)
        t_safe = np.maximum(t, 0.1)
        return {
            "A_sup":                              np.full_like(t_safe, self._A_sup),
            "B_sup\u00b7log\u2081\u2080(t)":      self._B_sup * np.log10(t_safe),
            "a\u2080":                            np.full_like(t_safe, self._a0),
            "damped log osc (\u03c9_log)":        self._C1 * t_safe ** (-self._D) * np.cos(
                self._W_log * np.log(t_safe) + self._PHI1),
            "undamped cal osc (\u03c9_cal)":      self._C2 * np.cos(
                self._W_cal * t_safe + self._PHI2),
        }


class LinPPLModel(LPPLModel):
    """Linear-periodic Power Law: oscillation in CALENDAR time, not log-time.

    Fits: log10(price) = A + B*log10(t) + C*t^(-D)*cos(W_cal*t + φ)

    Unlike LPPL (log-periodic with ω·ln(t)), LinPPL uses ω·t so the cycle
    period is constant in calendar years — matching Bitcoin's ~4-year halving
    cycle rather than LPPL's ever-lengthening log-time cycles.

    W_cal is the angular frequency in radians per year. A halving cycle of
    T calendar years corresponds to W_cal = 2π/T. For T=4yr, W_cal ≈ 1.57.
    """
    name = "LinPPL"
    short_name = "linppl"
    legend_name = "LinPPL"
    dash_style = "dash"

    # Fitted parameters (W_cal in radians/year; T_years = 2π/W_cal)
    _A   = -1.213443  
    _B   =        5.111004  
    _C   =        0.282312  
    _W   =        1.765644  # ≈ 2π/4 (4-year halving cycle, will refit)
    _PHI =  -2.283417  
    _D   =        0.010000  

    def _lppl_log10(self, t):
        """Evaluate LinPPL median in log10 space — oscillation in calendar t, not ln(t)."""
        t = np.asarray(t, float)
        t_safe = np.maximum(t, 0.1)
        envelope = self._C * t_safe ** (-self._D)
        return (self._A + self._B * np.log10(t_safe)
                + envelope * np.cos(self._W * t_safe + self._PHI))

    component_names = [
        "A (constant)",
        "B\u00b7log\u2081\u2080(t)",
        "damped osc (\u03c9_cal\u00b7t)",
    ]

    def components(self, t):
        """Calendar-time oscillation (cos(W_cal * t + phi)), not log-time."""
        t = np.asarray(t, float)
        t_safe = np.maximum(t, 0.1)
        return {
            "A (constant)":                    np.full_like(t_safe, self._A),
            "B\u00b7log\u2081\u2080(t)":        self._B * np.log10(t_safe),
            "damped osc (\u03c9_cal\u00b7t)":   self._C * t_safe ** (-self._D) * np.cos(
                self._W * t_safe + self._PHI),
        }


class ExponentialModel:
    """Exponential growth model with Gaussian quantile bands.

    Fits log10(price) = a + b*t (linear in time, exponential in price).
    Quantile bands shifted by z_q * sigma like PowerLawModel.
    Poor fit (R²~0.87) — included for comparison to show why power law
    is preferred over exponential for Bitcoin.
    """
    name = "Exponential"
    short_name = "exp"
    legend_name = "Exp"
    dash_style = "longdashdot"
    quantized = True

    def __init__(self, price_years, price_prices, quantiles):
        from scipy.stats import linregress as _lr
        mask = price_years >= 1.0
        t = price_years[mask]
        lp = np.log10(price_prices[mask])
        slope, intercept, r, _, _ = _lr(t, lp)
        self._intercept = intercept
        self._slope = slope
        residuals = lp - (intercept + slope * t)
        self._sigma = float(np.std(residuals))

        self.fits = {}
        for q in quantiles:
            z = norm.ppf(q)
            self.fits[q] = {"z_shift": z * self._sigma}
        self.quantiles = sorted(self.fits.keys())
        self._build_colors()

    def price_at(self, q, t):
        t_arr = np.asarray(t, float)
        log_median = self._intercept + self._slope * t_arr
        shift = self.fits[q]["z_shift"]
        return 10.0 ** (log_median + shift)

    def interp_price(self, q, t):
        if q in self.fits:
            return float(self.price_at(q, t))
        sorted_qs = self.quantiles
        lo = max((qq for qq in sorted_qs if qq <= q), default=sorted_qs[0])
        hi = min((qq for qq in sorted_qs if qq >= q), default=sorted_qs[-1])
        if lo == hi:
            return float(self.price_at(lo, t))
        frac = (q - lo) / (hi - lo)
        p_lo = np.log10(float(self.price_at(lo, t)))
        p_hi = np.log10(float(self.price_at(hi, t)))
        return 10.0 ** (p_lo + frac * (p_hi - p_lo))

    def find_percentile(self, t, price):
        sorted_qs = self.quantiles
        if not sorted_qs:
            return 0.5
        t_safe = max(float(t), 0.5)
        log_p = np.log10(max(float(price), 1e-10))
        log_ps = [np.log10(max(float(self.price_at(q, t_safe)), 1e-10))
                  for q in sorted_qs]
        if log_p <= log_ps[0]:
            return sorted_qs[0]
        if log_p >= log_ps[-1]:
            return sorted_qs[-1]
        for i in range(len(sorted_qs) - 1):
            if log_ps[i] <= log_p <= log_ps[i + 1]:
                frac = (log_p - log_ps[i]) / (log_ps[i + 1] - log_ps[i] + 1e-30)
                return sorted_qs[i] + frac * (sorted_qs[i + 1] - sorted_qs[i])
        return sorted_qs[-1]

    def _build_colors(self):
        """Red/pink palette — visually distinct, signals 'caution'."""
        self.colors = {}
        n = len(self.quantiles)
        for i, q in enumerate(self.quantiles):
            frac = i / max(n - 1, 1)
            r = int(200 + 55 * frac)     # 200 → 255
            g = int(60 + 80 * frac)      # 60 → 140
            b = int(80 + 60 * frac)      # 80 → 140
            self.colors[q] = f"#{r:02x}{g:02x}{b:02x}"


class EmpiricalFloorModel(_CompositeModel):
    """BM Empirical Floor with asymmetric shrinking Gaussian bands."""
    name = "BM Empirical Floor"
    short_name = "ef"
    legend_name = "EF"
    dash_style = "longdash"

    def __init__(self, pkl_path):
        import pickle
        with open(pkl_path, "rb") as f:
            d = pickle.load(f)

        self._slope = d["ef_support_slope"]
        self._intercept = d["ef_support_intercept"]
        self._t_grid = np.asarray(d["years_plot"], float)
        self._support_plot = np.asarray(d["support_plot"], float)
        self._comp_by_n = d["comp_by_n"]
        self._bm_r2 = d["bm_r2"]
        self._n_future_max = d["n_future_max"]

        comp = self._comp_by_n[-1]
        self._log_comp = np.log10(np.maximum(np.asarray(comp, float), 1e-10))

        # Shrinking σ parameters
        quantiles = d.get("QR_QUANTILES", [
            0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3,
            0.5, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 0.999])

        self._init_bands(
            d.get("sigma0_up", 0.093),
            d.get("alpha_up", 0.297),
            d.get("sigma0_down", 0.085),
            d.get("alpha_down", 0.295),
            quantiles,
        )
        self._build_colors()

    def _build_colors(self):
        """Amber/warm palette."""
        self.colors = {}
        n = len(self.quantiles)
        for i, q in enumerate(self.quantiles):
            frac = i / max(n - 1, 1)
            r = int(139 + 100 * frac)
            g = int(105 + 87 * frac)
            b = int(20 + 44 * frac)
            self.colors[q] = f"#{r:02x}{g:02x}{b:02x}"


class S2FModel:
    """Stock-to-Flow model — single price trajectory (not quantized).

    Fits log10(price) = a + b * log10(S2F) from historical data, where
    S2F = stock / annual_flow based on the Bitcoin halving schedule.
    """
    name = "Stock-to-Flow"
    short_name = "s2f"
    legend_name = "S2F"
    dash_style = "dot"
    quantized = False
    fits = None
    quantiles = []
    colors = {}

    _HALVING_BLOCKS = 210_000
    _BLOCKS_PER_DAY = 144
    _INITIAL_REWARD = 50.0

    def __init__(self, price_years, price_prices, genesis):
        self.genesis = genesis
        # Fit log10(price) = a + b * log10(S2F) from historical data
        mask = price_years >= 1.0
        yrs = price_years[mask]
        prices = price_prices[mask]

        s2f_vals = np.array([self._s2f_at_t(t) for t in yrs])
        valid = s2f_vals > 0
        log_s2f = np.log10(s2f_vals[valid])
        log_p = np.log10(prices[valid])

        slope, intercept, *_ = linregress(log_s2f, log_p)
        self._s2f_intercept = intercept
        self._s2f_slope = slope

    def _s2f_at_t(self, t):
        """Compute stock-to-flow ratio at years-since-genesis t."""
        days = t * 365.25
        total_blocks = days * self._BLOCKS_PER_DAY
        n_halvings = int(total_blocks // self._HALVING_BLOCKS)
        reward = self._INITIAL_REWARD / (2 ** n_halvings)

        # Cumulative stock
        stock = 0.0
        for h in range(n_halvings):
            stock += self._HALVING_BLOCKS * self._INITIAL_REWARD / (2 ** h)
        remaining = total_blocks - n_halvings * self._HALVING_BLOCKS
        stock += remaining * reward

        # Annual flow
        annual_flow = reward * self._BLOCKS_PER_DAY * 365.25
        if annual_flow <= 0:
            return 1e10  # effectively infinite S2F after all BTC mined
        return stock / annual_flow

    def price_at(self, q, t):
        """S2F model price (ignores quantile — single trajectory)."""
        t_arr = np.asarray(t, float)
        scalar = t_arr.ndim == 0
        t_flat = t_arr.ravel()
        s2f_vals = np.array([self._s2f_at_t(ti) for ti in t_flat])
        log_p = self._s2f_intercept + self._s2f_slope * np.log10(s2f_vals)
        result = 10.0 ** log_p
        return float(result[0]) if scalar else result.reshape(t_arr.shape)

    def interp_price(self, q, t):
        return float(self.price_at(q, t))

    def find_percentile(self, t, price):
        return 0.5  # meaningless for non-quantized model


class UserModel(_FitsBasedModel):
    """User-defined power law model from two clicked points on log-log chart.

    Fully quantized: parallel lines (same slope, shifted intercepts) derived
    from the empirical residual distribution against historical prices.
    """
    name = "User Model"
    short_name = "u1"
    legend_name = "U\u2081"
    dash_style = "solid"
    quantized = True

    def __init__(self, slope, intercept, shifts, quantiles, r2_per_quantile, own_quantile):
        self.fits = {q: {"intercept": intercept + shifts[q], "slope": slope}
                     for q in quantiles}
        self.quantiles = sorted(quantiles)
        self.r2_per_quantile = r2_per_quantile or {}
        self.own_quantile = own_quantile
        self.colors = {q: "#e67e22" for q in self.quantiles}

    @classmethod
    def from_points(cls, t1, p1, t2, p2, price_years, price_prices, quantiles):
        """Factory: two chart points + historical data → fully quantized model."""
        log_t1, log_p1 = np.log10(max(t1, 0.01)), np.log10(max(p1, 1e-10))
        log_t2, log_p2 = np.log10(max(t2, 0.01)), np.log10(max(p2, 1e-10))
        denom = log_t2 - log_t1
        if abs(denom) < 1e-12:
            denom = 1e-12
        slope = (log_p2 - log_p1) / denom
        intercept = log_p1 - slope * log_t1

        mask = price_years >= 0.5
        t_hist = np.asarray(price_years[mask], float)
        p_hist = np.asarray(price_prices[mask], float)
        predicted = intercept + slope * np.log10(np.maximum(t_hist, 0.01))
        residuals = np.log10(np.maximum(p_hist, 1e-10)) - predicted

        own_quantile = float(np.mean(residuals <= 0))
        shifts = {q: float(np.percentile(residuals, q * 100)) for q in quantiles}

        # Ensure own_quantile is in the quantile list with shift=0
        # (the user's drawn line passes exactly through the two points)
        if own_quantile not in shifts:
            shifts[own_quantile] = 0.0
        else:
            shifts[own_quantile] = 0.0  # force exact zero even if percentile is close
        all_quantiles = sorted(set(quantiles) | {own_quantile})

        r2 = {}
        for q in all_quantiles:
            pred_q = 10.0 ** (intercept + shifts.get(q, 0) + slope * np.log10(np.maximum(t_hist, 0.01)))
            r2_val = _compute_log_r2(p_hist, pred_q)
            if r2_val is not None:
                r2[q] = r2_val

        return cls(slope, intercept, shifts, all_quantiles, r2, own_quantile)

    def to_store_dict(self):
        """Serialize to JSON-safe dict for dcc.Store."""
        slope = self.fits[self.quantiles[0]]["slope"]
        # base_intercept: the user's drawn line (shift=0, passes through both points)
        base_intercept = self.fits[self.own_quantile]["intercept"]
        return {
            "slope": slope,
            "base_intercept": base_intercept,
            "intercepts": {str(q): self.fits[q]["intercept"] for q in self.quantiles},
            "r2": {str(q): v for q, v in self.r2_per_quantile.items()},
            "own_quantile": self.own_quantile,
            "quantiles": [float(q) for q in self.quantiles],
        }

    @classmethod
    def from_store_dict(cls, d):
        """Reconstruct from dcc.Store dict."""
        if not d:
            return None
        quantiles = [float(q) for q in d["quantiles"]]
        slope = d["slope"]
        intercepts = {float(q): v for q, v in d["intercepts"].items()}
        r2 = {float(q): v for q, v in d["r2"].items()} if d.get("r2") else {}
        model = cls.__new__(cls)
        model.fits = {q: {"intercept": intercepts[q], "slope": slope} for q in quantiles}
        model.quantiles = sorted(quantiles)
        model.r2_per_quantile = r2
        model.own_quantile = d["own_quantile"]
        model.colors = {q: "#e67e22" for q in quantiles}
        return model
