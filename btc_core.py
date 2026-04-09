"""btc_core.py — Shared model and math utilities for Bitcoin Projections.

No Qt or matplotlib dependencies — importable from both the PyQt5 desktop
app (btc_projections.py) and the Plotly Dash web app (btc_web/app.py).

Note: btc_projections.py currently defines these inline for historical reasons.
The web app imports from here directly.
"""

import ast, pickle, sys
from pathlib import Path

import numpy as np
import pandas as pd

# Lazy imports — scipy and statsmodels add ~2-3s to import time.
# They're needed for model evaluation (norm.ppf) and fitting (linregress,
# QuantReg), but not for unpickling model_data.pkl at startup.
_norm = None
_linregress = None
_QuantReg = None

def _lazy_norm():
    global _norm
    if _norm is None:
        from scipy.stats import norm as _n
        _norm = _n
    return _norm

def _lazy_linregress():
    global _linregress
    if _linregress is None:
        from scipy.stats import linregress as _lr
        _linregress = _lr
    return _linregress

def _lazy_QuantReg():
    global _QuantReg
    if _QuantReg is None:
        from statsmodels.regression.quantile_regression import QuantReg as _QR
        _QuantReg = _QR
    return _QuantReg

# ── constants ─────────────────────────────────────────────────────────────────

_DEFAULT_QS = [0.001, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]


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
    ols_slope, ols_int, *_ = _lazy_linregress()(dfit["log_years"].values, dfit["log_price"].values)
    qr = {}
    for q in quantiles:
        res = _lazy_QuantReg()(dfit["log_price"].values, X).fit(q=q, max_iter=2000)
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
        z = _lazy_norm().ppf(q)
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
            self.fits[q] = {"z": float(_lazy_norm().ppf(q))}
        self.quantiles = sorted(self.fits.keys())

    component_names = ["support", "bubbles"]
    support_component_names = ["support"]
    formula_log10_latex = (
        r"\log_{10}(\text{support}(t)) + \sum_i \text{bubble}_i(t)"
    )
    component_details = {
        "support": ("log\u2081\u2080(support(t))", []),
        "bubbles": ("\u03a3 bubble_i(t)", []),
    }

    def components(self, t):
        """Composite decomposition: support + bubbles (both in log10 space).

        sum(components(t)) == self._composite_log10(t).
        """
        t = np.asarray(t, float)
        log_support = np.interp(t, self._t_grid, self._log_support)
        log_composite = self._composite_log10(t)
        return {
            "support": log_support,
            "bubbles": log_composite - log_support,
        }

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

        # Support line (log10 USD) for component decomposition
        self._log_support = np.log10(np.maximum(
            np.asarray(md.support_bm, float), 1e-10))

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
            z = _lazy_norm().ppf(q)
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
    _A   = -1.153754
    _B   =                    5.079165
    _C   =                    0.733958
    _W   =                    7.557598
    _PHI =                    1.377438
    _D   =                    0.607919

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
            z = _lazy_norm().ppf(q)
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
    # Components that together define the "support" (trend) line.
    # Used by reference/cumulative decomposition rendering modes.
    support_component_names = ["A (constant)", "B\u00b7log\u2081\u2080(t)"]
    formula_log10_latex = (
        r"A + B \log_{10}(t) + C \cdot t^{-D} \cos(\omega \ln t + \varphi)"
    )
    formula_product_latex = (
        r"10^A \cdot t^B \cdot 10^{\,C \cdot t^{-D} \cos(\omega \ln t + \varphi)}"
    )
    # (plain-text formula, [(param_name, attr_name), ...])
    component_details = {
        "A (constant)":           ("A",                         [("A", "_A")]),
        "B\u00b7log\u2081\u2080(t)": ("B\u00b7log\u2081\u2080(t)", [("B", "_B")]),
        "damped osc (\u03c9_log)": ("C\u00b7t^(-D)\u00b7cos(\u03c9\u00b7ln(t)+\u03c6)",
                                    [("C", "_C"), ("D", "_D"),
                                     ("\u03c9", "_W"), ("\u03c6", "_PHI")]),
    }

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
    _A   = -1.130574
    _B   =                       5.038215
    _C   =                       0.705403
    _W   =                       7.376654
    _PHI =                       1.583777
    _D   =                       0.565686
    _C2  =                       0.168834
    _W2  =                20.904698
    _PHI2 = -1.158033

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
    formula_log10_latex = (
        r"A + B \log_{10}(t) + C_1 t^{-D} \cos(\omega_1 \ln t + \varphi_1)"
        r" + C_2 \cos(\omega_2 \ln t + \varphi_2)"
    )
    formula_product_latex = (
        r"10^A \cdot t^B"
        r" \cdot 10^{\,C_1 t^{-D} \cos(\omega_1 \ln t + \varphi_1)}"
        r" \cdot 10^{\,C_2 \cos(\omega_2 \ln t + \varphi_2)}"
    )
    component_details = {
        "A (constant)":           ("A",                         [("A", "_A")]),
        "B\u00b7log\u2081\u2080(t)": ("B\u00b7log\u2081\u2080(t)", [("B", "_B")]),
        "damped osc (\u03c9\u2081)": ("C\u2081\u00b7t^(-D)\u00b7cos(\u03c9\u2081\u00b7ln(t)+\u03c6\u2081)",
                                      [("C\u2081", "_C"), ("D", "_D"),
                                       ("\u03c9\u2081", "_W"), ("\u03c6\u2081", "_PHI")]),
        "undamped osc (\u03c9\u2082)": ("C\u2082\u00b7cos(\u03c9\u2082\u00b7ln(t)+\u03c6\u2082)",
                                        [("C\u2082", "_C2"), ("\u03c9\u2082", "_W2"),
                                         ("\u03c6\u2082", "_PHI2")]),
    }

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
    _A   = -1.094293
    _B   =                 4.966525
    _C   =                 0.613915
    _W   =                 7.122560
    _PHI =                 1.890154
    _D   =                 0.365686
    _C2  =                 0.178607
    _W2  =            20.805576
    _PHI2 = -0.997242
    _C3  =                 0.171206
    _W3  =            10.083261
    _PHI3 = -2.167289

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
    formula_log10_latex = (
        r"A + B \log_{10}(t) + C_1 t^{-D} \cos(\omega_1 \ln t + \varphi_1)"
        r" + C_2 \cos(\omega_2 \ln t + \varphi_2)"
        r" + C_3 \cos(\omega_3 \ln t + \varphi_3)"
    )
    formula_product_latex = (
        r"10^A \cdot t^B"
        r" \cdot 10^{\,C_1 t^{-D} \cos(\omega_1 \ln t + \varphi_1)}"
        r" \cdot 10^{\,C_2 \cos(\omega_2 \ln t + \varphi_2)}"
        r" \cdot 10^{\,C_3 \cos(\omega_3 \ln t + \varphi_3)}"
    )
    component_details = {
        "A (constant)":           ("A",                         [("A", "_A")]),
        "B\u00b7log\u2081\u2080(t)": ("B\u00b7log\u2081\u2080(t)", [("B", "_B")]),
        "damped osc (\u03c9\u2081)": ("C\u2081\u00b7t^(-D)\u00b7cos(\u03c9\u2081\u00b7ln(t)+\u03c6\u2081)",
                                      [("C\u2081", "_C"), ("D", "_D"),
                                       ("\u03c9\u2081", "_W"), ("\u03c6\u2081", "_PHI")]),
        "undamped osc (\u03c9\u2082)": ("C\u2082\u00b7cos(\u03c9\u2082\u00b7ln(t)+\u03c6\u2082)",
                                        [("C\u2082", "_C2"), ("\u03c9\u2082", "_W2"),
                                         ("\u03c6\u2082", "_PHI2")]),
        "undamped osc (\u03c9\u2083)": ("C\u2083\u00b7cos(\u03c9\u2083\u00b7ln(t)+\u03c6\u2083)",
                                        [("C\u2083", "_C3"), ("\u03c9\u2083", "_W3"),
                                         ("\u03c6\u2083", "_PHI3")]),
    }

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
    _A   = -1.099422
    _B   =                 4.971727
    _C   =                 0.581476
    _W   =                 7.103090
    _PHI =                 1.920446
    _D   =                 0.331580
    _C2  =                 0.172529
    _W2  =              10.018587
    _PHI2 = -2.052668
    _C3  =                 0.095754
    _W3  =             30.953441
    _PHI3 =    -2.440706
    _C4  =                 0.170618
    _W4  =           20.754990
    _PHI4 = -0.911211

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
    formula_log10_latex = (
        r"A + B \log_{10}(t) + C_1 t^{-D} \cos(\omega_1 \ln t + \varphi_1)"
        r" + C_2 \cos(\omega_2 \ln t + \varphi_2)"
        r" + C_3 \cos(\omega_3 \ln t + \varphi_3)"
        r" + C_4 \cos(\omega_4 \ln t + \varphi_4)"
    )
    formula_product_latex = (
        r"10^A \cdot t^B"
        r" \cdot 10^{\,C_1 t^{-D} \cos(\omega_1 \ln t + \varphi_1)}"
        r" \cdot 10^{\,C_2 \cos(\omega_2 \ln t + \varphi_2)}"
        r" \cdot 10^{\,C_3 \cos(\omega_3 \ln t + \varphi_3)}"
        r" \cdot 10^{\,C_4 \cos(\omega_4 \ln t + \varphi_4)}"
    )
    component_details = {
        "A (constant)":           ("A",                         [("A", "_A")]),
        "B\u00b7log\u2081\u2080(t)": ("B\u00b7log\u2081\u2080(t)", [("B", "_B")]),
        "damped osc (\u03c9\u2081)": ("C\u2081\u00b7t^(-D)\u00b7cos(\u03c9\u2081\u00b7ln(t)+\u03c6\u2081)",
                                      [("C\u2081", "_C"), ("D", "_D"),
                                       ("\u03c9\u2081", "_W"), ("\u03c6\u2081", "_PHI")]),
        "undamped osc (\u03c9\u2082)": ("C\u2082\u00b7cos(\u03c9\u2082\u00b7ln(t)+\u03c6\u2082)",
                                        [("C\u2082", "_C2"), ("\u03c9\u2082", "_W2"),
                                         ("\u03c6\u2082", "_PHI2")]),
        "undamped osc (\u03c9\u2083)": ("C\u2083\u00b7cos(\u03c9\u2083\u00b7ln(t)+\u03c6\u2083)",
                                        [("C\u2083", "_C3"), ("\u03c9\u2083", "_W3"),
                                         ("\u03c6\u2083", "_PHI3")]),
        "undamped osc (\u03c9\u2084)": ("C\u2084\u00b7cos(\u03c9\u2084\u00b7ln(t)+\u03c6\u2084)",
                                        [("C\u2084", "_C4"), ("\u03c9\u2084", "_W4"),
                                         ("\u03c6\u2084", "_PHI4")]),
    }

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
    _A   = -1.146871  
    _B   =                 5.051440  
    _C   =                 0.689800  
    _W   =                 7.420028  
    _PHI =                 1.453362  
    _D   =                 0.708113  
    _C2  =                 0.233047  
    _W2  =                 1.733178  
    _PHI2 = -1.923186  

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
    formula_log10_latex = (
        r"A + B \log_{10}(t) + C_1 t^{-D} \cos(\omega_{\text{log}} \ln t + \varphi_1)"
        r" + C_2 \cos(\omega_{\text{cal}} t + \varphi_2)"
    )
    formula_product_latex = (
        r"10^A \cdot t^B"
        r" \cdot 10^{\,C_1 t^{-D} \cos(\omega_{\text{log}} \ln t + \varphi_1)}"
        r" \cdot 10^{\,C_2 \cos(\omega_{\text{cal}} t + \varphi_2)}"
    )
    component_details = {
        "A (constant)":           ("A",                         [("A", "_A")]),
        "B\u00b7log\u2081\u2080(t)": ("B\u00b7log\u2081\u2080(t)", [("B", "_B")]),
        "damped log osc (\u03c9_log)": (
            "C\u2081\u00b7t^(-D)\u00b7cos(\u03c9_log\u00b7ln(t)+\u03c6\u2081)",
            [("C\u2081", "_C"), ("D", "_D"),
             ("\u03c9_log", "_W"), ("\u03c6\u2081", "_PHI")]),
        "undamped cal osc (\u03c9_cal)": (
            "C\u2082\u00b7cos(\u03c9_cal\u00b7t+\u03c6\u2082)",
            [("C\u2082", "_C2"), ("\u03c9_cal", "_W2"),
             ("\u03c6\u2082", "_PHI2")]),
    }

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


class HybPPLDDModel(LPPLModel):
    """HybPPL (DD — Double Damped): both oscillators damped, non-excess.

    Fits: log10(price) = A + B*log10(t)
                       + C1*t^(-D1)*cos(W_log*ln(t) + PHI1)
                       + C2*t^(-D2)*cos(W_cal*t + PHI2)

    Like HybPPL but with an independent damping exponent on each oscillator.
    Tests whether the halving cycle is permanent (D2 near 0) or decaying.
    10 parameters — one more than HybPPL's 9.
    """
    name = "HybPPL (DD)"
    short_name = "hybppl_dd"
    legend_name = "HybPPL (DD)"
    dash_style = "dashdot"

    # Fitted parameters (will be overwritten by fit_hybppl_dd.py --update)
    _A     = -1.146940  
    _B     =        5.051521  
    _C1    =        0.690016  
    _W_log =        7.420125  
    _PHI1  =        1.453219  
    _D1    =        0.708418  
    _C2    =        0.233494  
    _W_cal =        1.733171  
    _PHI2  = -1.923130  
    _D2    =        0.001000  

    def _lppl_log10(self, t):
        """Evaluate double-damped hybrid model."""
        t = np.asarray(t, float)
        t_safe = np.maximum(t, 0.1)
        damped_log = self._C1 * t_safe ** (-self._D1) * np.cos(
            self._W_log * np.log(t_safe) + self._PHI1)
        damped_cal = self._C2 * t_safe ** (-self._D2) * np.cos(
            self._W_cal * t_safe + self._PHI2)
        return self._A + self._B * np.log10(t_safe) + damped_log + damped_cal

    component_names = [
        "A (constant)",
        "B\u00b7log\u2081\u2080(t)",
        "damped log osc (\u03c9_log)",
        "damped cal osc (\u03c9_cal)",
    ]
    support_component_names = []
    formula_log10_latex = (
        r"A + B \log_{10}(t)"
        r" + C_1 t^{-D_1} \cos(\omega_{\text{log}} \ln t + \varphi_1)"
        r" + C_2 t^{-D_2} \cos(\omega_{\text{cal}} t + \varphi_2)"
    )
    formula_product_latex = (
        r"10^A \cdot t^B"
        r" \cdot 10^{\,C_1 t^{-D_1} \cos(\omega_{\text{log}} \ln t + \varphi_1)}"
        r" \cdot 10^{\,C_2 t^{-D_2} \cos(\omega_{\text{cal}} t + \varphi_2)}"
    )
    component_details = {
        "A (constant)":           ("A",                         [("A", "_A")]),
        "B\u00b7log\u2081\u2080(t)": ("B\u00b7log\u2081\u2080(t)", [("B", "_B")]),
        "damped log osc (\u03c9_log)": (
            "C\u2081\u00b7t^(-D\u2081)\u00b7cos(\u03c9_log\u00b7ln(t)+\u03c6\u2081)",
            [("C\u2081", "_C1"), ("D\u2081", "_D1"),
             ("\u03c9_log", "_W_log"), ("\u03c6\u2081", "_PHI1")]),
        "damped cal osc (\u03c9_cal)": (
            "C\u2082\u00b7t^(-D\u2082)\u00b7cos(\u03c9_cal\u00b7t+\u03c6\u2082)",
            [("C\u2082", "_C2"), ("D\u2082", "_D2"),
             ("\u03c9_cal", "_W_cal"), ("\u03c6\u2082", "_PHI2")]),
    }

    def components(self, t):
        """Double-damped hybrid: both oscillators have independent damping."""
        t = np.asarray(t, float)
        t_safe = np.maximum(t, 0.1)
        return {
            "A (constant)":                        np.full_like(t_safe, self._A),
            "B\u00b7log\u2081\u2080(t)":            self._B * np.log10(t_safe),
            "damped log osc (\u03c9_log)":          self._C1 * t_safe ** (-self._D1) * np.cos(
                self._W_log * np.log(t_safe) + self._PHI1),
            "damped cal osc (\u03c9_cal)":          self._C2 * t_safe ** (-self._D2) * np.cos(
                self._W_cal * t_safe + self._PHI2),
        }


class Hyb2LModel(LPPLModel):
    """HybPPL + 2nd log-periodic oscillation.

    Fits: log10(price) = A + B*log10(t)
                       + C1*t^(-D1)*cos(W1*ln(t)+PHI1)
                       + C2*cos(Wc*t+PHI2)
                       + C3*t^(-D2)*cos(W2*ln(t)+PHI3)

    Adds a second damped log-periodic harmonic to the baseline HybPPL.
    13 parameters.
    """
    name = "HybPPL +2L"
    short_name = "hyb2l"
    legend_name = "Hyb2L"
    dash_style = "dashdot"
    quantized = True

    # Fitted parameters (will be overwritten by fit_hyb2l.py --update)
    _A    = -1.113051  
    _B    =     5.013919  
    _C1   =     0.765444  
    _W1   =     7.471808  
    _PHI1 =     1.297984  
    _D1   =     0.773452  
    _C2   =     0.257516  
    _Wc   =     1.720228  
    _PHI2 = -1.736955  
    _C3   =     0.392739  
    _W2   =   15.993374  
    _PHI3 =     1.889585  
    _D2   =     0.932751  

    def _lppl_log10(self, t):
        t = np.asarray(t, float)
        t_safe = np.maximum(t, 0.1)
        osc1 = self._C1 * t_safe ** (-self._D1) * np.cos(self._W1 * np.log(t_safe) + self._PHI1)
        cal  = self._C2 * np.cos(self._Wc * t_safe + self._PHI2)
        osc2 = self._C3 * t_safe ** (-self._D2) * np.cos(self._W2 * np.log(t_safe) + self._PHI3)
        return self._A + self._B * np.log10(t_safe) + osc1 + cal + osc2

    component_names = [
        "A (constant)",
        "B\u00b7log\u2081\u2080(t)",
        "damped log osc 1 (\u03c9\u2081)",
        "undamped cal osc (\u03c9_cal)",
        "damped log osc 2 (\u03c9\u2082)",
    ]
    formula_log10_latex = (
        r"A + B \log_{10}(t)"
        r" + C_1 t^{-D_1} \cos(\omega_1 \ln t + \varphi_1)"
        r" + C_2 \cos(\omega_c t + \varphi_2)"
        r" + C_3 t^{-D_2} \cos(\omega_2 \ln t + \varphi_3)"
    )
    formula_product_latex = (
        r"10^A \cdot t^B"
        r" \cdot 10^{\,C_1 t^{-D_1} \cos(\omega_1 \ln t + \varphi_1)}"
        r" \cdot 10^{\,C_2 \cos(\omega_c t + \varphi_2)}"
        r" \cdot 10^{\,C_3 t^{-D_2} \cos(\omega_2 \ln t + \varphi_3)}"
    )
    component_details = {
        "A (constant)":           ("A", [("A", "_A")]),
        "B\u00b7log\u2081\u2080(t)": ("B\u00b7log\u2081\u2080(t)", [("B", "_B")]),
        "damped log osc 1 (\u03c9\u2081)": (
            "C\u2081\u00b7t^(-D\u2081)\u00b7cos(\u03c9\u2081\u00b7ln(t)+\u03c6\u2081)",
            [("C\u2081", "_C1"), ("D\u2081", "_D1"),
             ("\u03c9\u2081", "_W1"), ("\u03c6\u2081", "_PHI1")]),
        "undamped cal osc (\u03c9_cal)": (
            "C\u2082\u00b7cos(\u03c9_c\u00b7t+\u03c6\u2082)",
            [("C\u2082", "_C2"), ("\u03c9_c", "_Wc"), ("\u03c6\u2082", "_PHI2")]),
        "damped log osc 2 (\u03c9\u2082)": (
            "C\u2083\u00b7t^(-D\u2082)\u00b7cos(\u03c9\u2082\u00b7ln(t)+\u03c6\u2083)",
            [("C\u2083", "_C3"), ("D\u2082", "_D2"),
             ("\u03c9\u2082", "_W2"), ("\u03c6\u2083", "_PHI3")]),
    }

    def components(self, t):
        t = np.asarray(t, float)
        t_safe = np.maximum(t, 0.1)
        return {
            "A (constant)":                np.full_like(t_safe, self._A),
            "B\u00b7log\u2081\u2080(t)":    self._B * np.log10(t_safe),
            "damped log osc 1 (\u03c9\u2081)": self._C1 * t_safe ** (-self._D1) * np.cos(
                self._W1 * np.log(t_safe) + self._PHI1),
            "undamped cal osc (\u03c9_cal)": self._C2 * np.cos(
                self._Wc * t_safe + self._PHI2),
            "damped log osc 2 (\u03c9\u2082)": self._C3 * t_safe ** (-self._D2) * np.cos(
                self._W2 * np.log(t_safe) + self._PHI3),
        }


class Hyb2CModel(LPPLModel):
    """HybPPL + 2nd calendar-periodic oscillation.

    Fits: log10(price) = A + B*log10(t)
                       + C1*t^(-D)*cos(W1*ln(t)+PHI1)
                       + C2*cos(Wc1*t+PHI2)
                       + C3*cos(Wc2*t+PHI3)

    Adds a second undamped calendar-periodic term. The 2nd frequency
    (~1.88yr) is roughly half the halving cycle — may capture
    sub-halving market structure.
    12 parameters.
    """
    name = "HybPPL +2C"
    short_name = "hyb2c"
    legend_name = "Hyb2C"
    dash_style = "dashdot"
    quantized = True

    # Fitted parameters (will be overwritten by fit_hyb2c.py --update)
    _A    = -1.135475  
    _B    =     5.037834  
    _C1   =     0.738861  
    _W1   =     7.356028  
    _PHI1 =     1.659079  
    _D    =     0.730244  
    _C2   =     0.235258  
    _Wc1  =     1.750651  
    _PHI2 = -2.086733  
    _C3   =     0.114575  
    _Wc2  =     3.280654  
    _PHI3 = -2.452119  

    def _lppl_log10(self, t):
        t = np.asarray(t, float)
        t_safe = np.maximum(t, 0.1)
        osc  = self._C1 * t_safe ** (-self._D) * np.cos(self._W1 * np.log(t_safe) + self._PHI1)
        cal1 = self._C2 * np.cos(self._Wc1 * t_safe + self._PHI2)
        cal2 = self._C3 * np.cos(self._Wc2 * t_safe + self._PHI3)
        return self._A + self._B * np.log10(t_safe) + osc + cal1 + cal2

    component_names = [
        "A (constant)",
        "B\u00b7log\u2081\u2080(t)",
        "damped log osc (\u03c9_log)",
        "undamped cal osc 1 (\u03c9_c\u2081)",
        "undamped cal osc 2 (\u03c9_c\u2082)",
    ]
    formula_log10_latex = (
        r"A + B \log_{10}(t)"
        r" + C_1 t^{-D} \cos(\omega_1 \ln t + \varphi_1)"
        r" + C_2 \cos(\omega_{c1} t + \varphi_2)"
        r" + C_3 \cos(\omega_{c2} t + \varphi_3)"
    )
    formula_product_latex = (
        r"10^A \cdot t^B"
        r" \cdot 10^{\,C_1 t^{-D} \cos(\omega_1 \ln t + \varphi_1)}"
        r" \cdot 10^{\,C_2 \cos(\omega_{c1} t + \varphi_2)}"
        r" \cdot 10^{\,C_3 \cos(\omega_{c2} t + \varphi_3)}"
    )
    component_details = {
        "A (constant)":           ("A", [("A", "_A")]),
        "B\u00b7log\u2081\u2080(t)": ("B\u00b7log\u2081\u2080(t)", [("B", "_B")]),
        "damped log osc (\u03c9_log)": (
            "C\u2081\u00b7t^(-D)\u00b7cos(\u03c9\u2081\u00b7ln(t)+\u03c6\u2081)",
            [("C\u2081", "_C1"), ("D", "_D"),
             ("\u03c9\u2081", "_W1"), ("\u03c6\u2081", "_PHI1")]),
        "undamped cal osc 1 (\u03c9_c\u2081)": (
            "C\u2082\u00b7cos(\u03c9_c\u2081\u00b7t+\u03c6\u2082)",
            [("C\u2082", "_C2"), ("\u03c9_c\u2081", "_Wc1"), ("\u03c6\u2082", "_PHI2")]),
        "undamped cal osc 2 (\u03c9_c\u2082)": (
            "C\u2083\u00b7cos(\u03c9_c\u2082\u00b7t+\u03c6\u2083)",
            [("C\u2083", "_C3"), ("\u03c9_c\u2082", "_Wc2"), ("\u03c6\u2083", "_PHI3")]),
    }

    def components(self, t):
        t = np.asarray(t, float)
        t_safe = np.maximum(t, 0.1)
        return {
            "A (constant)":                    np.full_like(t_safe, self._A),
            "B\u00b7log\u2081\u2080(t)":        self._B * np.log10(t_safe),
            "damped log osc (\u03c9_log)":      self._C1 * t_safe ** (-self._D) * np.cos(
                self._W1 * np.log(t_safe) + self._PHI1),
            "undamped cal osc 1 (\u03c9_c\u2081)": self._C2 * np.cos(
                self._Wc1 * t_safe + self._PHI2),
            "undamped cal osc 2 (\u03c9_c\u2082)": self._C3 * np.cos(
                self._Wc2 * t_safe + self._PHI3),
        }


class Hyb2BModel(LPPLModel):
    """HybPPL + 2nd log-periodic + 2nd calendar-periodic.

    Fits: log10(price) = A + B*log10(t)
                       + C1*t^(-D1)*cos(W1*ln(t)+PHI1)
                       + C2*cos(Wc1*t+PHI2)
                       + C3*t^(-D2)*cos(W2*ln(t)+PHI3)
                       + C4*cos(Wc2*t+PHI4)

    Full second-frequency model: both log-periodic and calendar-periodic
    get a second harmonic. 16 parameters — highest R² in the family.
    """
    name = "HybPPL +2B"
    short_name = "hyb2b"
    legend_name = "Hyb2B"
    dash_style = "dashdot"
    quantized = True

    # Fitted parameters (will be overwritten by fit_hyb2b.py --update)
    _A    = -1.114180  
    _B    =     5.017427  
    _C1   =     0.890964  
    _W1   =     7.483988  
    _PHI1 =     1.389285  
    _D1   =     0.832962  
    _C2   =     0.242031  
    _Wc1  =     1.739799  
    _PHI2 = -1.918563  
    _C3   =     0.422538  
    _W2   =   16.237963  
    _PHI3 =     1.885419  
    _D2   =     1.166351  
    _C4   =     0.105464  
    _Wc2  =     3.340729  
    _PHI4 =     3.135899  

    def _lppl_log10(self, t):
        t = np.asarray(t, float)
        t_safe = np.maximum(t, 0.1)
        osc1 = self._C1 * t_safe ** (-self._D1) * np.cos(self._W1 * np.log(t_safe) + self._PHI1)
        cal1 = self._C2 * np.cos(self._Wc1 * t_safe + self._PHI2)
        osc2 = self._C3 * t_safe ** (-self._D2) * np.cos(self._W2 * np.log(t_safe) + self._PHI3)
        cal2 = self._C4 * np.cos(self._Wc2 * t_safe + self._PHI4)
        return self._A + self._B * np.log10(t_safe) + osc1 + cal1 + osc2 + cal2

    component_names = [
        "A (constant)",
        "B\u00b7log\u2081\u2080(t)",
        "damped log osc 1 (\u03c9_l\u2081)",
        "undamped cal osc 1 (\u03c9_c\u2081)",
        "damped log osc 2 (\u03c9_l\u2082)",
        "undamped cal osc 2 (\u03c9_c\u2082)",
    ]
    formula_log10_latex = (
        r"A + B \log_{10}(t)"
        r" + C_1 t^{-D_1} \cos(\omega_{l1} \ln t + \varphi_1)"
        r" + C_2 \cos(\omega_{c1} t + \varphi_2)"
        r" + C_3 t^{-D_2} \cos(\omega_{l2} \ln t + \varphi_3)"
        r" + C_4 \cos(\omega_{c2} t + \varphi_4)"
    )
    formula_product_latex = (
        r"10^A \cdot t^B"
        r" \cdot 10^{\,C_1 t^{-D_1} \cos(\omega_{l1} \ln t + \varphi_1)}"
        r" \cdot 10^{\,C_2 \cos(\omega_{c1} t + \varphi_2)}"
        r" \cdot 10^{\,C_3 t^{-D_2} \cos(\omega_{l2} \ln t + \varphi_3)}"
        r" \cdot 10^{\,C_4 \cos(\omega_{c2} t + \varphi_4)}"
    )
    component_details = {
        "A (constant)":           ("A", [("A", "_A")]),
        "B\u00b7log\u2081\u2080(t)": ("B\u00b7log\u2081\u2080(t)", [("B", "_B")]),
        "damped log osc 1 (\u03c9_l\u2081)": (
            "C\u2081\u00b7t^(-D\u2081)\u00b7cos(\u03c9_l\u2081\u00b7ln(t)+\u03c6\u2081)",
            [("C\u2081", "_C1"), ("D\u2081", "_D1"),
             ("\u03c9_l\u2081", "_W1"), ("\u03c6\u2081", "_PHI1")]),
        "undamped cal osc 1 (\u03c9_c\u2081)": (
            "C\u2082\u00b7cos(\u03c9_c\u2081\u00b7t+\u03c6\u2082)",
            [("C\u2082", "_C2"), ("\u03c9_c\u2081", "_Wc1"), ("\u03c6\u2082", "_PHI2")]),
        "damped log osc 2 (\u03c9_l\u2082)": (
            "C\u2083\u00b7t^(-D\u2082)\u00b7cos(\u03c9_l\u2082\u00b7ln(t)+\u03c6\u2083)",
            [("C\u2083", "_C3"), ("D\u2082", "_D2"),
             ("\u03c9_l\u2082", "_W2"), ("\u03c6\u2083", "_PHI3")]),
        "undamped cal osc 2 (\u03c9_c\u2082)": (
            "C\u2084\u00b7cos(\u03c9_c\u2082\u00b7t+\u03c6\u2084)",
            [("C\u2084", "_C4"), ("\u03c9_c\u2082", "_Wc2"), ("\u03c6\u2084", "_PHI4")]),
    }

    def components(self, t):
        t = np.asarray(t, float)
        t_safe = np.maximum(t, 0.1)
        return {
            "A (constant)":                        np.full_like(t_safe, self._A),
            "B\u00b7log\u2081\u2080(t)":            self._B * np.log10(t_safe),
            "damped log osc 1 (\u03c9_l\u2081)":   self._C1 * t_safe ** (-self._D1) * np.cos(
                self._W1 * np.log(t_safe) + self._PHI1),
            "undamped cal osc 1 (\u03c9_c\u2081)": self._C2 * np.cos(
                self._Wc1 * t_safe + self._PHI2),
            "damped log osc 2 (\u03c9_l\u2082)":   self._C3 * t_safe ** (-self._D2) * np.cos(
                self._W2 * np.log(t_safe) + self._PHI3),
            "undamped cal osc 2 (\u03c9_c\u2082)": self._C4 * np.cos(
                self._Wc2 * t_safe + self._PHI4),
        }


class Hyb4DModel(LPPLModel):
    """HybPPL 4D — all 4 oscillatory components damped.

    Fits: log10(price) = A + B*log10(t)
                       + C1*t^(-D1)*cos(W1*ln(t)+PHI1)
                       + C2*t^(-Dc1)*cos(Wc1*t+PHI2)
                       + C3*t^(-D2)*cos(W2*ln(t)+PHI3)
                       + C4*t^(-Dc2)*cos(Wc2*t+PHI4)

    All four oscillators carry damping exponents. 18 parameters.
    Compared to Hyb2B (16 params, R²=0.993), adding 2 extra D params
    yields WORSE fit (R²=0.992, BIC=-22624 vs -23203). The calendar
    terms resist damping — Dc2≈0.076 is near zero, meaning the 2nd
    calendar oscillator WANTS to be undamped.
    """
    name = "HybPPL 4D"
    short_name = "hyb4d"
    legend_name = "Hyb4D"
    dash_style = "dashdot"
    quantized = True

    # Fitted parameters (will be overwritten by fit_hyb4d.py --update)
    _A    = -1.113156  
    _B    =     5.016722  
    _C1   =     0.921541  
    _W1   =     7.482817  
    _PHI1 =     1.403663  
    _D1   =     0.847676  
    _C2   =     0.240589  
    _Wc1  =     1.740838  
    _PHI2 =  -1.935091  
    _Dc1  =     0.000000  
    _C3   =     0.433531  
    _W2   =   16.252921  
    _PHI3 =     1.891523  
    _D2   =     1.205977  
    _C4   =     0.134349  
    _Wc2  =     3.342578  
    _PHI4 =    3.121447  
    _Dc2  =     0.109722  

    def _lppl_log10(self, t):
        t = np.asarray(t, float)
        t_safe = np.maximum(t, 0.1)
        osc1 = self._C1 * t_safe ** (-self._D1) * np.cos(self._W1 * np.log(t_safe) + self._PHI1)
        cal1 = self._C2 * t_safe ** (-self._Dc1) * np.cos(self._Wc1 * t_safe + self._PHI2)
        osc2 = self._C3 * t_safe ** (-self._D2) * np.cos(self._W2 * np.log(t_safe) + self._PHI3)
        cal2 = self._C4 * t_safe ** (-self._Dc2) * np.cos(self._Wc2 * t_safe + self._PHI4)
        return self._A + self._B * np.log10(t_safe) + osc1 + cal1 + osc2 + cal2

    component_names = [
        "A (constant)",
        "B\u00b7log\u2081\u2080(t)",
        "damped log osc 1 (\u03c9_l\u2081)",
        "damped cal osc 1 (\u03c9_c\u2081)",
        "damped log osc 2 (\u03c9_l\u2082)",
        "damped cal osc 2 (\u03c9_c\u2082)",
    ]
    formula_log10_latex = (
        r"A + B \log_{10}(t)"
        r" + C_1 t^{-D_1} \cos(\omega_{l1} \ln t + \varphi_1)"
        r" + C_2 t^{-D_{c1}} \cos(\omega_{c1} t + \varphi_2)"
        r" + C_3 t^{-D_2} \cos(\omega_{l2} \ln t + \varphi_3)"
        r" + C_4 t^{-D_{c2}} \cos(\omega_{c2} t + \varphi_4)"
    )
    formula_product_latex = (
        r"10^A \cdot t^B"
        r" \cdot 10^{\,C_1 t^{-D_1} \cos(\omega_{l1} \ln t + \varphi_1)}"
        r" \cdot 10^{\,C_2 t^{-D_{c1}} \cos(\omega_{c1} t + \varphi_2)}"
        r" \cdot 10^{\,C_3 t^{-D_2} \cos(\omega_{l2} \ln t + \varphi_3)}"
        r" \cdot 10^{\,C_4 t^{-D_{c2}} \cos(\omega_{c2} t + \varphi_4)}"
    )
    component_details = {
        "A (constant)":           ("A", [("A", "_A")]),
        "B\u00b7log\u2081\u2080(t)": ("B\u00b7log\u2081\u2080(t)", [("B", "_B")]),
        "damped log osc 1 (\u03c9_l\u2081)": (
            "C\u2081\u00b7t^(-D\u2081)\u00b7cos(\u03c9_l\u2081\u00b7ln(t)+\u03c6\u2081)",
            [("C\u2081", "_C1"), ("D\u2081", "_D1"),
             ("\u03c9_l\u2081", "_W1"), ("\u03c6\u2081", "_PHI1")]),
        "damped cal osc 1 (\u03c9_c\u2081)": (
            "C\u2082\u00b7t^(-D_c\u2081)\u00b7cos(\u03c9_c\u2081\u00b7t+\u03c6\u2082)",
            [("C\u2082", "_C2"), ("D_c\u2081", "_Dc1"),
             ("\u03c9_c\u2081", "_Wc1"), ("\u03c6\u2082", "_PHI2")]),
        "damped log osc 2 (\u03c9_l\u2082)": (
            "C\u2083\u00b7t^(-D\u2082)\u00b7cos(\u03c9_l\u2082\u00b7ln(t)+\u03c6\u2083)",
            [("C\u2083", "_C3"), ("D\u2082", "_D2"),
             ("\u03c9_l\u2082", "_W2"), ("\u03c6\u2083", "_PHI3")]),
        "damped cal osc 2 (\u03c9_c\u2082)": (
            "C\u2084\u00b7t^(-D_c\u2082)\u00b7cos(\u03c9_c\u2082\u00b7t+\u03c6\u2084)",
            [("C\u2084", "_C4"), ("D_c\u2082", "_Dc2"),
             ("\u03c9_c\u2082", "_Wc2"), ("\u03c6\u2084", "_PHI4")]),
    }

    def components(self, t):
        t = np.asarray(t, float)
        t_safe = np.maximum(t, 0.1)
        return {
            "A (constant)":                        np.full_like(t_safe, self._A),
            "B\u00b7log\u2081\u2080(t)":            self._B * np.log10(t_safe),
            "damped log osc 1 (\u03c9_l\u2081)":   self._C1 * t_safe ** (-self._D1) * np.cos(
                self._W1 * np.log(t_safe) + self._PHI1),
            "damped cal osc 1 (\u03c9_c\u2081)":   self._C2 * t_safe ** (-self._Dc1) * np.cos(
                self._Wc1 * t_safe + self._PHI2),
            "damped log osc 2 (\u03c9_l\u2082)":   self._C3 * t_safe ** (-self._D2) * np.cos(
                self._W2 * np.log(t_safe) + self._PHI3),
            "damped cal osc 2 (\u03c9_c\u2082)":   self._C4 * t_safe ** (-self._Dc2) * np.cos(
                self._Wc2 * t_safe + self._PHI4),
        }


class PCAModel:
    """PCA-based model: principal components from HybPPL-family component basis.

    Takes the ~30 component time series from all HybPPL-family models,
    runs PCA (SVD) to find orthogonal directions, then OLS-regresses
    log10(price) on the top k principal components.

    Result: R²=0.993 with 7 params (6 PCs + intercept) — beats Hyb2B
    (16 params) on BIC. The 30 correlated components collapse into ~6
    orthogonal directions that capture all the signal.

    At prediction time, evaluates all source basis functions at t,
    applies pre-computed weight vector (no matrix ops needed).
    """
    name = "PCA (HybPPL basis)"
    short_name = "pca"
    legend_name = "PCA"
    dash_style = "dot"
    quantized = True

    # Source model keys whose components form the basis
    _SOURCE_KEYS = ("hybppl", "hybppl_dd", "hyb2l", "hyb2c", "hyb2b", "hyb4d")
    _N_PCS = 6  # number of principal components to use

    def __init__(self, price_years, price_prices, quantiles, source_models=None):
        if source_models is None:
            source_models = {}
        mask = price_years >= 1.0
        t = price_years[mask]
        lp = np.log10(price_prices[mask])
        n = len(t)

        # Build component matrix from all source models
        self._basis_info = []  # [(model_key, comp_name), ...] for each column
        columns = []
        for key in self._SOURCE_KEYS:
            mdl = source_models.get(key)
            if mdl is None:
                continue
            comps = mdl.components(t)
            for cname, vals in comps.items():
                columns.append(np.asarray(vals, float))
                self._basis_info.append((key, cname))

        if not columns:
            # Fallback: degenerate model
            self._intercept = float(np.mean(lp))
            self._weights = np.array([])
            self._sigma = float(np.std(lp))
            self._X_mean = np.array([])
            self._V_k = np.array([]).reshape(0, 0)
            self._beta = np.array([self._intercept])
            self._explained = np.array([])
        else:
            X = np.column_stack(columns)
            X_mean = X.mean(axis=0)
            Xc = X - X_mean

            # SVD-based PCA
            U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
            k = min(self._N_PCS, len(S))
            total_var = np.sum(S ** 2)
            self._explained = (S ** 2 / total_var)[:k]

            # PC scores and OLS regression
            scores = (U * S)[:, :k]
            X_reg = np.column_stack([np.ones(n), scores])
            beta = np.linalg.lstsq(X_reg, lp, rcond=None)[0]

            # Collapse PCA + OLS into a single weight vector on components
            V_k = Vt[:k, :].T  # (n_components x k)
            w = V_k @ beta[1:]  # (n_components,)
            intercept = beta[0] - float(X_mean @ w)

            self._intercept = intercept
            self._weights = w
            self._X_mean = X_mean
            self._V_k = V_k
            self._beta = beta
            self._sigma = float(np.std(lp - (intercept + X @ w)))

        # Store source models for component evaluation at prediction time
        self._source_models = {k: source_models[k] for k in self._SOURCE_KEYS
                               if k in source_models}

        # Build quantile bands
        self.fits = {}
        for q in quantiles:
            z = _lazy_norm().ppf(q)
            self.fits[q] = {"z_shift": z * self._sigma}
        self.quantiles = sorted(self.fits.keys())
        self._build_colors()

    def _eval_basis(self, t):
        """Evaluate all source basis functions at time t, return column vector."""
        t = np.asarray(t, float)
        columns = []
        for key, cname in self._basis_info:
            mdl = self._source_models.get(key)
            if mdl is None:
                continue
            comps = mdl.components(t)
            columns.append(np.asarray(comps[cname], float))
        if not columns:
            return np.zeros_like(t)
        return np.column_stack(columns)

    def _model_log10(self, t):
        """Evaluate: intercept + X @ weights."""
        t_arr = np.asarray(t, float)
        scalar = t_arr.ndim == 0
        if scalar:
            t_arr = t_arr.reshape(1)
        X = self._eval_basis(t_arr)
        if X.ndim == 1 or len(self._weights) == 0:
            result = np.full_like(t_arr, self._intercept)
        else:
            result = self._intercept + X @ self._weights
        return float(result[0]) if scalar else result

    def price_at(self, q, t):
        t_arr = np.asarray(t, float)
        log_median = self._model_log10(t_arr)
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

    # Decomposition: group the 30 weighted basis functions by physical role
    _COMP_GROUPS = [
        ("intercept",         "intercept",     None),
        ("power law trend",   "B\u00b7log\u2081\u2080(t)", "B"),
        ("log-periodic osc",  "log osc",       "log"),
        ("calendar-periodic", "cal osc",       "cal"),
    ]

    component_names = [
        "intercept",
        "power law trend",
        "log-periodic osc",
        "calendar-periodic",
    ]

    formula_log10_latex = (
        r"\text{intercept} + \sum_{j} w_j \cdot f_j(t)"
    )
    formula_product_latex = (
        r"10^{\,\text{intercept}} \cdot \prod_{j} 10^{\,w_j \cdot f_j(t)}"
    )

    @property
    def component_details(self):
        return {
            "intercept": (
                "\u03b1 (constant)",
                [("const", "_intercept")],
            ),
            "power law trend": (
                "\u03a3 w\u2c7c\u00b7B\u2c7c\u00b7log\u2081\u2080(t)",
                [],
            ),
            "log-periodic osc": (
                "\u03a3 w\u2c7c\u00b7C\u2c7c\u00b7t^(\u2212D)\u00b7cos(\u03c9\u00b7ln(t)+\u03c6)",
                [],
            ),
            "calendar-periodic": (
                "\u03a3 w\u2c7c\u00b7C\u2c7c\u00b7cos(\u03c9\u00b7t+\u03c6)",
                [],
            ),
        }

    def components(self, t):
        """Decompose into intercept + grouped basis function contributions."""
        t_arr = np.asarray(t, float)
        scalar = t_arr.ndim == 0
        if scalar:
            t_arr = t_arr.reshape(1)
        X = self._eval_basis(t_arr)
        n = len(t_arr)
        intercept = np.full(n, self._intercept)
        trend = np.zeros(n)
        log_osc = np.zeros(n)
        cal_osc = np.zeros(n)

        if X.ndim > 1 and len(self._weights) > 0:
            for i, ((key, cname), w) in enumerate(zip(self._basis_info, self._weights)):
                contrib = w * X[:, i]
                cl = cname.lower()
                if "log" in cl and ("osc" in cl or "cos" in cl):
                    log_osc += contrib
                elif "cal" in cl and ("osc" in cl or "cos" in cl):
                    cal_osc += contrib
                elif "log" in cl and "t" in cl:
                    # B·log₁₀(t) — power law trend
                    trend += contrib
                elif "constant" in cl or cname.startswith("A "):
                    intercept += contrib
                else:
                    trend += contrib  # fallback: lump into trend

        result = {
            "intercept": intercept,
            "power law trend": trend,
            "log-periodic osc": log_osc,
            "calendar-periodic": cal_osc,
        }
        if scalar:
            result = {k: float(v[0]) for k, v in result.items()}
        return result

    def _build_colors(self):
        """Indigo palette — PCA model."""
        self.colors = {}
        n = len(self.quantiles)
        for i, q in enumerate(self.quantiles):
            frac = i / max(n - 1, 1)
            r = int(60 + 50 * frac)
            g = int(40 + 60 * frac)
            b = int(120 + 60 * frac)
            self.colors[q] = f"#{r:02x}{g:02x}{b:02x}"


class GreedyModel:
    """Greedy forward BIC-selected model: 5 oscillatory terms from LPPL/HybPPL.

    Selects components via greedy forward BIC minimisation from the pool
    of individual oscillatory terms in existing LPPL/HybPPL models.
    Result: R²=0.9928, σ=0.130, BIC=-23,319 with only 7 parameters
    (intercept + slope + 5 weighted oscillatory terms).

    All parameters are hardcoded from the greedy search — no runtime
    dependency on other model instances.

    Formula:
        log₁₀(price) = α + β·log₁₀(t) + Σᵢ wᵢ·fᵢ(t)

    where fᵢ are 5 oscillatory basis functions with source-model
    parameters baked in.
    """
    name = "Greedy Select"
    short_name = "grdy"
    legend_name = "Greedy"
    dash_style = "dashdot"
    quantized = True

    # ── OLS intercept and slope ──────────────────────────────────────────
    _alpha = -1.211238
    _beta  =  5.119765
    _sigma =  0.129877

    # ── 5 selected oscillatory terms ─────────────────────────────────────
    # Each term: (weight, amplitude, damping, freq, phase, is_log_periodic)
    #   is_log_periodic=True  → C·t^(-D)·cos(ω·ln(t) + φ)
    #   is_log_periodic=False → C·t^(-D)·cos(ω·t + φ)
    #
    # f₁: halving cycle from LinPPL — undamped calendar oscillation
    _w1 = 0.921198;  _C1 = 0.282344;  _D1 = 0.010000;  _W1 = 1.765746;  _PHI1 = -2.284078;  _LOG1 = False
    # f₂: primary LPPL frequency — damped log-periodic
    _w2 = 0.839768;  _C2 = 0.733975;  _D2 = 0.607967;  _W2 = 7.557911;  _PHI2 =  1.377121;  _LOG2 = True
    # f₃: sub-halving from Hyb2C — undamped calendar oscillation
    _w3 = 0.868695;  _C3 = 0.114588;  _D3 = 0.000000;  _W3 = 3.280720;  _PHI3 = -2.452578;  _LOG3 = False
    # f₄: 2nd log harmonic from Hyb2B — fast-decay log-periodic
    _w4 = 0.783073;  _C4 = 0.422419;  _D4 = 1.165713;  _W4 = 16.238167; _PHI4 =  1.885355;  _LOG4 = True
    # f₅: long calendar from Hyb4D — heavily damped calendar oscillation
    _w5 = 0.546787;  _C5 = 0.586943;  _D5 = 1.062266;  _W5 = 1.116857;  _PHI5 =  3.141119;  _LOG5 = False

    def __init__(self, price_years, price_prices, quantiles):
        # Build quantile bands via Gaussian z·σ shift
        self.fits = {}
        for q in quantiles:
            z = _lazy_norm().ppf(q)
            self.fits[q] = {"z_shift": z * self._sigma}
        self.quantiles = sorted(self.fits.keys())
        self._build_colors()

    @staticmethod
    def _eval_term(t_safe, C, D, W, PHI, is_log):
        """Evaluate a single oscillatory basis function."""
        envelope = C * t_safe ** (-D) if D != 0.0 else np.full_like(t_safe, C)
        arg = W * np.log(t_safe) + PHI if is_log else W * t_safe + PHI
        return envelope * np.cos(arg)

    def _model_log10(self, t):
        """Evaluate: α + β·log₁₀(t) + Σ wᵢ·fᵢ(t)."""
        t_arr = np.asarray(t, float)
        scalar = t_arr.ndim == 0
        if scalar:
            t_arr = t_arr.reshape(1)
        t_safe = np.maximum(t_arr, 0.1)

        result = self._alpha + self._beta * np.log10(t_safe)
        result += self._w1 * self._eval_term(t_safe, self._C1, self._D1, self._W1, self._PHI1, self._LOG1)
        result += self._w2 * self._eval_term(t_safe, self._C2, self._D2, self._W2, self._PHI2, self._LOG2)
        result += self._w3 * self._eval_term(t_safe, self._C3, self._D3, self._W3, self._PHI3, self._LOG3)
        result += self._w4 * self._eval_term(t_safe, self._C4, self._D4, self._W4, self._PHI4, self._LOG4)
        result += self._w5 * self._eval_term(t_safe, self._C5, self._D5, self._W5, self._PHI5, self._LOG5)

        return float(result[0]) if scalar else result

    def price_at(self, q, t):
        t_arr = np.asarray(t, float)
        log_median = self._model_log10(t_arr)
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

    # ── Decomposition ────────────────────────────────────────────────────

    component_names = [
        "\u03b1 (intercept)",
        "\u03b2\u00b7log\u2081\u2080(t)",
        "f\u2081 halving cycle",
        "f\u2082 log-periodic",
        "f\u2083 sub-halving",
        "f\u2084 2nd log harmonic",
        "f\u2085 long calendar",
    ]

    formula_log10_latex = (
        r"\alpha + \beta \log_{10}(t) + \sum_{i=1}^{5} w_i \cdot f_i(t)"
    )
    formula_product_latex = (
        r"10^{\,\alpha} \cdot t^{\beta} \cdot \prod_{i=1}^{5} 10^{\,w_i \cdot f_i(t)}"
    )

    @property
    def component_details(self):
        return {
            "\u03b1 (intercept)": (
                "\u03b1",
                [("\u03b1", "_alpha")],
            ),
            "\u03b2\u00b7log\u2081\u2080(t)": (
                "\u03b2\u00b7log\u2081\u2080(t)",
                [("\u03b2", "_beta")],
            ),
            "f\u2081 halving cycle": (
                "w\u2081\u00b7C\u2081\u00b7t^(\u2212D\u2081)\u00b7cos(\u03c9\u2081\u00b7t+\u03c6\u2081)",
                [("w\u2081", "_w1"), ("C\u2081", "_C1"), ("D\u2081", "_D1"),
                 ("\u03c9\u2081", "_W1"), ("\u03c6\u2081", "_PHI1")],
            ),
            "f\u2082 log-periodic": (
                "w\u2082\u00b7C\u2082\u00b7t^(\u2212D\u2082)\u00b7cos(\u03c9\u2082\u00b7ln(t)+\u03c6\u2082)",
                [("w\u2082", "_w2"), ("C\u2082", "_C2"), ("D\u2082", "_D2"),
                 ("\u03c9\u2082", "_W2"), ("\u03c6\u2082", "_PHI2")],
            ),
            "f\u2083 sub-halving": (
                "w\u2083\u00b7C\u2083\u00b7cos(\u03c9\u2083\u00b7t+\u03c6\u2083)",
                [("w\u2083", "_w3"), ("C\u2083", "_C3"),
                 ("\u03c9\u2083", "_W3"), ("\u03c6\u2083", "_PHI3")],
            ),
            "f\u2084 2nd log harmonic": (
                "w\u2084\u00b7C\u2084\u00b7t^(\u2212D\u2084)\u00b7cos(\u03c9\u2084\u00b7ln(t)+\u03c6\u2084)",
                [("w\u2084", "_w4"), ("C\u2084", "_C4"), ("D\u2084", "_D4"),
                 ("\u03c9\u2084", "_W4"), ("\u03c6\u2084", "_PHI4")],
            ),
            "f\u2085 long calendar": (
                "w\u2085\u00b7C\u2085\u00b7t^(\u2212D\u2085)\u00b7cos(\u03c9\u2085\u00b7t+\u03c6\u2085)",
                [("w\u2085", "_w5"), ("C\u2085", "_C5"), ("D\u2085", "_D5"),
                 ("\u03c9\u2085", "_W5"), ("\u03c6\u2085", "_PHI5")],
            ),
        }

    def components(self, t):
        """Decompose into intercept + trend + 5 individual oscillatory terms."""
        t_arr = np.asarray(t, float)
        scalar = t_arr.ndim == 0
        if scalar:
            t_arr = t_arr.reshape(1)
        t_safe = np.maximum(t_arr, 0.1)

        result = {
            "\u03b1 (intercept)":      np.full_like(t_safe, self._alpha),
            "\u03b2\u00b7log\u2081\u2080(t)": self._beta * np.log10(t_safe),
            "f\u2081 halving cycle":   self._w1 * self._eval_term(t_safe, self._C1, self._D1, self._W1, self._PHI1, self._LOG1),
            "f\u2082 log-periodic":    self._w2 * self._eval_term(t_safe, self._C2, self._D2, self._W2, self._PHI2, self._LOG2),
            "f\u2083 sub-halving":     self._w3 * self._eval_term(t_safe, self._C3, self._D3, self._W3, self._PHI3, self._LOG3),
            "f\u2084 2nd log harmonic": self._w4 * self._eval_term(t_safe, self._C4, self._D4, self._W4, self._PHI4, self._LOG4),
            "f\u2085 long calendar":   self._w5 * self._eval_term(t_safe, self._C5, self._D5, self._W5, self._PHI5, self._LOG5),
        }
        if scalar:
            result = {k: float(v[0]) for k, v in result.items()}
        return result

    def _build_colors(self):
        """Forest green palette — greedy select model."""
        self.colors = {}
        n = len(self.quantiles)
        for i, q in enumerate(self.quantiles):
            frac = i / max(n - 1, 1)
            r = int(30 + 50 * frac)      # 30 → 80
            g = int(120 + 60 * frac)     # 120 → 180
            b = int(50 + 50 * frac)      # 50 → 100
            self.colors[q] = f"#{r:02x}{g:02x}{b:02x}"


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
    _A   = -1.213406  
    _B   =                 5.110908  
    _C   =                 0.282358  
    _W   =                 1.765788  # ≈ 2π/4 (4-year halving cycle, will refit)
    _PHI =  -2.284355  
    _D   =                 0.010000  

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
    formula_log10_latex = (
        r"A + B \log_{10}(t) + C \cdot t^{-D} \cos(\omega_{\text{cal}} \cdot t + \varphi)"
    )
    component_details = {
        "A (constant)":           ("A",                         [("A", "_A")]),
        "B\u00b7log\u2081\u2080(t)": ("B\u00b7log\u2081\u2080(t)", [("B", "_B")]),
        "damped osc (\u03c9_cal\u00b7t)": (
            "C\u00b7t^(-D)\u00b7cos(\u03c9_cal\u00b7t+\u03c6)",
            [("C", "_C"), ("D", "_D"),
             ("\u03c9_cal", "_W"), ("\u03c6", "_PHI")]),
    }

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
        mask = price_years >= 1.0
        t = price_years[mask]
        lp = np.log10(price_prices[mask])
        slope, intercept, r, _, _ = _lazy_linregress()(t, lp)
        self._intercept = intercept
        self._slope = slope
        residuals = lp - (intercept + slope * t)
        self._sigma = float(np.std(residuals))

        self.fits = {}
        for q in quantiles:
            z = _lazy_norm().ppf(q)
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


# ── EPPL config params (auto-generated) ──
_EPPL_CONFIG_PARAMS = {
    "ecfg_0_0": {"n_log": 0, "n_cal": 0, "log_damps": [], "cal_damps": [], "params": {"A": -1.173784, "B": 5.081215}, "r2": 0.962654, "sigma": 0.295620},
    "ecfg_0_1d": {"n_log": 0, "n_cal": 1, "log_damps": [], "cal_damps": ['d'], "params": {"A": -1.185489, "B": 5.092057, "C_cal": 0.376542, "W_cal": 1.739438, "PHI_cal": -2.060581, "w_cal": 0.056790}, "r2": 0.981127, "sigma": 0.210151},
    "ecfg_0_1u": {"n_log": 0, "n_cal": 1, "log_damps": [], "cal_damps": ['u'], "params": {"A": -1.211635, "B": 5.108941, "C_cal": 0.276894, "W_cal": 1.765383, "PHI_cal": -2.280318}, "r2": 0.978941, "sigma": 0.221991},
    "ecfg_0_2dd": {"n_log": 0, "n_cal": 2, "log_damps": [], "cal_damps": ['d', 'd'], "params": {"A": -1.096556, "B": 5.001078, "C_cal1": 0.614571, "W_cal1": 4.147145, "PHI_cal1": -2.025236, "w_cal1": 0.333507, "C_cal2": 0.399038, "W_cal2": 1.711301, "PHI_cal2": -1.800361, "w_cal2": 0.058408}, "r2": 0.986551, "sigma": 0.177400},
    "ecfg_0_2du": {"n_log": 0, "n_cal": 2, "log_damps": [], "cal_damps": ['d', 'u'], "params": {"A": -1.289916, "B": 5.212834, "C_cal1": 0.387360, "W_cal1": 1.753223, "PHI_cal1": -2.186095, "w_cal1": 0.058384, "C_cal2": 0.137250, "W_cal2": 0.834382, "PHI_cal2": -2.702990}, "r2": 0.984474, "sigma": 0.190610},
    "ecfg_0_2uu": {"n_log": 0, "n_cal": 2, "log_damps": [], "cal_damps": ['u', 'u'], "params": {"A": -1.183331, "B": 5.075917, "C_cal1": 0.133621, "W_cal1": 3.119343, "PHI_cal1": -0.727837, "C_cal2": 0.285132, "W_cal2": 1.763443, "PHI_cal2": -2.221823}, "r2": 0.982750, "sigma": 0.200914},
    "ecfg_1d_0": {"n_log": 1, "n_cal": 0, "log_damps": ['d'], "cal_damps": [], "params": {"A": -1.140066, "B": 5.060594, "C_log": 0.542874, "W_log": 7.680129, "PHI_log": 1.224824, "w_log": 0.101156}, "r2": 0.983419, "sigma": 0.196980},
    "ecfg_1d_1d": {"n_log": 1, "n_cal": 1, "log_damps": ['d'], "cal_damps": ['d'], "params": {"A": -1.157398, "B": 5.069443, "C_log": 0.472459, "W_log": 7.769475, "PHI_log": 1.320545, "w_log": 0.106177, "C_cal": 0.219876, "W_cal": 1.843120, "PHI_cal": 2.974032, "w_cal": 0.036873}, "r2": 0.989615, "sigma": 0.155888},
    "ecfg_1d_1u": {"n_log": 1, "n_cal": 1, "log_damps": ['d'], "cal_damps": ['u'], "params": {"A": -1.189358, "B": 5.101294, "C_log": 0.478427, "W_log": 7.794515, "PHI_log": 1.320053, "w_log": 0.105442, "C_cal": 0.193897, "W_cal": 1.857282, "PHI_cal": 2.802240}, "r2": 0.989340, "sigma": 0.157939},
    "ecfg_1d_2dd": {"n_log": 1, "n_cal": 2, "log_damps": ['d'], "cal_damps": ['d', 'd'], "params": {"A": -1.152768, "B": 5.064707, "C_log": 0.484537, "W_log": 7.809597, "PHI_log": 1.264030, "w_log": 0.106425, "C_cal1": 0.219245, "W_cal1": 1.846572, "PHI_cal1": 2.924649, "w_cal1": 0.036499, "C_cal2": 0.236297, "W_cal2": 10.292570, "PHI_cal2": -1.198334, "w_cal2": 0.177048}, "r2": 0.991712, "sigma": 0.139260},
    "ecfg_1d_2du": {"n_log": 1, "n_cal": 2, "log_damps": ['d'], "cal_damps": ['d', 'u'], "params": {"A": -1.189530, "B": 5.101966, "C_log": 0.491673, "W_log": 7.787010, "PHI_log": 1.337354, "w_log": 0.105686, "C_cal1": 0.223446, "W_cal1": 10.000000, "PHI_cal1": -0.385570, "w_cal1": 0.173351, "C_cal2": 0.193789, "W_cal2": 1.862757, "PHI_cal2": 2.731482}, "r2": 0.991245, "sigma": 0.143135},
    "ecfg_1d_2uu": {"n_log": 1, "n_cal": 2, "log_damps": ['d'], "cal_damps": ['u', 'u'], "params": {"A": -1.176691, "B": 5.088978, "C_log": 0.546098, "W_log": 7.726491, "PHI_log": 1.488667, "w_log": 0.106829, "C_cal1": 0.205753, "W_cal1": 1.883481, "PHI_cal1": 2.507911, "C_cal2": 0.116652, "W_cal2": 3.344440, "PHI_cal2": -3.125422}, "r2": 0.991906, "sigma": 0.137621},
    "ecfg_1u_0": {"n_log": 1, "n_cal": 0, "log_damps": ['u'], "cal_damps": [], "params": {"A": -1.232539, "B": 5.182900, "C_log": 0.230726, "W_log": 7.721562, "PHI_log": 1.188723}, "r2": 0.973981, "sigma": 0.246753},
    "ecfg_1u_1d": {"n_log": 1, "n_cal": 1, "log_damps": ['u'], "cal_damps": ['d'], "params": {"A": -1.180404, "B": 5.090632, "C_log": 0.182428, "W_log": 7.252503, "PHI_log": 1.714152, "C_cal": 0.293675, "W_cal": 1.737834, "PHI_cal": -1.997991, "w_cal": 0.041179}, "r2": 0.986943, "sigma": 0.174798},
    "ecfg_1u_1u": {"n_log": 1, "n_cal": 1, "log_damps": ['u'], "cal_damps": ['u'], "params": {"A": -1.234885, "B": 5.154289, "C_log": 0.184747, "W_log": 7.370084, "PHI_log": 1.623173, "C_cal": 0.245727, "W_cal": 1.760606, "PHI_cal": -2.246028}, "r2": 0.986226, "sigma": 0.179535},
    "ecfg_1u_2dd": {"n_log": 1, "n_cal": 2, "log_damps": ['u'], "cal_damps": ['d', 'd'], "params": {"A": -1.168494, "B": 5.071364, "C_log": 0.150704, "W_log": 7.043406, "PHI_log": 2.148976, "C_cal1": 0.300974, "W_cal1": 1.737710, "PHI_cal1": -2.003796, "w_cal1": 0.043205, "C_cal2": 0.212469, "W_cal2": 2.854918, "PHI_cal2": 0.062348, "w_cal2": 0.100288}, "r2": 0.989822, "sigma": 0.154328},
    "ecfg_1u_2du": {"n_log": 1, "n_cal": 2, "log_damps": ['u'], "cal_damps": ['d', 'u'], "params": {"A": -1.089246, "B": 4.968562, "C_log": 0.161827, "W_log": 6.419797, "PHI_log": -2.864354, "C_cal1": 0.805182, "W_cal1": 4.250424, "PHI_cal1": -2.519881, "w_cal1": 0.379993, "C_cal2": 0.286664, "W_cal2": 1.742011, "PHI_cal2": -2.076675}, "r2": 0.989673, "sigma": 0.155450},
    "ecfg_1u_2uu": {"n_log": 1, "n_cal": 2, "log_damps": ['u'], "cal_damps": ['u', 'u'], "params": {"A": -1.195381, "B": 5.101300, "C_log": 0.178136, "W_log": 7.144865, "PHI_log": 2.014151, "C_cal1": 0.256178, "W_cal1": 1.760221, "PHI_cal1": -2.212661, "C_cal2": 0.121642, "W_cal2": 3.167675, "PHI_cal2": -1.212915}, "r2": 0.989170, "sigma": 0.159195},
    "ecfg_2dd_0": {"n_log": 2, "n_cal": 0, "log_damps": ['d', 'd'], "cal_damps": [], "params": {"A": -1.127776, "B": 5.048206, "C_log1": 0.553513, "W_log1": 7.728906, "PHI_log1": 1.158299, "w_log1": 0.101405, "C_log2": 0.277842, "W_log2": 16.516411, "PHI_log2": 1.634666, "w_log2": 0.249992}, "r2": 0.985204, "sigma": 0.186074},
    "ecfg_2dd_1d": {"n_log": 2, "n_cal": 1, "log_damps": ['d', 'd'], "cal_damps": ['d'], "params": {"A": -1.143612, "B": 5.055210, "C_log1": 0.484316, "W_log1": 7.829766, "PHI_log1": 1.229580, "w_log1": 0.106270, "C_log2": 0.266809, "W_log2": 16.646059, "PHI_log2": 1.554957, "w_log2": 0.250686, "C_cal": 0.216854, "W_cal": 1.840976, "PHI_cal": 2.989631, "w_cal": 0.035055}, "r2": 0.991243, "sigma": 0.143154},
    "ecfg_2dd_1u": {"n_log": 2, "n_cal": 1, "log_damps": ['d', 'd'], "cal_damps": ['u'], "params": {"A": -1.191233, "B": 5.102466, "C_log1": 0.473390, "W_log1": 7.807929, "PHI_log1": 1.312068, "w_log1": 0.104927, "C_log2": 0.109168, "W_log2": 37.276563, "PHI_log2": 1.716159, "w_log2": 0.044386, "C_cal": 0.201474, "W_cal": 1.850313, "PHI_cal": 2.844831}, "r2": 0.991176, "sigma": 0.143701},
    "ecfg_2dd_2dd": {"n_log": 2, "n_cal": 2, "log_damps": ['d', 'd'], "cal_damps": ['d', 'd'], "params": {"A": -0.628613, "B": 4.607373, "C_log1": 3.912672, "W_log1": 3.008530, "PHI_log1": 1.981470, "w_log1": 0.500000, "C_log2": 0.831928, "W_log2": 5.804256, "PHI_log2": -1.324804, "w_log2": 0.105199, "C_cal1": 1.516885, "W_cal1": 1.513709, "PHI_cal1": -2.167326, "w_cal1": 0.176575, "C_cal2": 0.205880, "W_cal2": 10.232352, "PHI_cal2": -0.970630, "w_cal2": 0.166852}, "r2": 0.987254, "sigma": 0.172702},
    "ecfg_2dd_2du": {"n_log": 2, "n_cal": 2, "log_damps": ['d', 'd'], "cal_damps": ['d', 'u'], "params": {"A": -1.189596, "B": 5.102710, "C_log1": 0.487108, "W_log1": 7.786044, "PHI_log1": 1.328379, "w_log1": 0.104547, "C_log2": 0.124018, "W_log2": 30.429686, "PHI_log2": -1.408445, "w_log2": 0.065882, "C_cal1": 0.233721, "W_cal1": 10.000000, "PHI_cal1": -0.591390, "w_cal1": 0.172940, "C_cal2": 0.191346, "W_cal2": 1.868342, "PHI_cal2": 2.722372}, "r2": 0.992844, "sigma": 0.129400},
    "ecfg_2dd_2uu": {"n_log": 2, "n_cal": 2, "log_damps": ['d', 'd'], "cal_damps": ['u', 'u'], "params": {"A": -1.167364, "B": 5.079560, "C_log1": 0.250431, "W_log1": 16.823754, "PHI_log1": 1.460423, "w_log1": 0.251550, "C_log2": 0.556269, "W_log2": 7.803554, "PHI_log2": 1.373041, "w_log2": 0.107049, "C_cal1": 0.202747, "W_cal1": 1.881312, "PHI_cal1": 2.520901, "C_cal2": 0.113542, "W_cal2": 3.355482, "PHI_cal2": 3.033229}, "r2": 0.993320, "sigma": 0.125028},
    "ecfg_2du_0": {"n_log": 2, "n_cal": 0, "log_damps": ['d', 'u'], "cal_damps": [], "params": {"A": -1.124220, "B": 5.029270, "C_log1": 0.521293, "W_log1": 7.684369, "PHI_log1": 1.227815, "w_log1": 0.095766, "C_log2": 0.151372, "W_log2": 20.773683, "PHI_log2": -0.896309}, "r2": 0.988298, "sigma": 0.165479},
    "ecfg_2du_1d": {"n_log": 2, "n_cal": 1, "log_damps": ['d', 'u'], "cal_damps": ['d'], "params": {"A": -1.159884, "B": 5.064082, "C_log1": 0.618147, "W_log1": 8.736504, "PHI_log1": -0.566654, "w_log1": 0.104475, "C_log2": 0.163302, "W_log2": 20.724695, "PHI_log2": -0.867348, "C_cal": 0.723176, "W_cal": 4.397533, "PHI_cal": -1.275783, "w_cal": 0.267779}, "r2": 0.991035, "sigma": 0.144844},
    "ecfg_2du_1u": {"n_log": 2, "n_cal": 1, "log_damps": ['d', 'u'], "cal_damps": ['u'], "params": {"A": -1.184380, "B": 5.101305, "C_log1": 0.479821, "W_log1": 7.731005, "PHI_log1": 1.344117, "w_log1": 0.105839, "C_log2": 0.109760, "W_log2": 20.299527, "PHI_log2": -0.436882, "C_cal": 0.157679, "W_cal": 1.902087, "PHI_cal": 2.682141}, "r2": 0.990968, "sigma": 0.145379},
    "ecfg_2du_2dd": {"n_log": 2, "n_cal": 2, "log_damps": ['d', 'u'], "cal_damps": ['d', 'd'], "params": {"A": -1.158361, "B": 5.068582, "C_log1": 0.403762, "W_log1": 7.901512, "PHI_log1": 1.339118, "w_log1": 0.106518, "C_log2": 0.088706, "W_log2": 7.595651, "PHI_log2": 0.660092, "C_cal1": 0.235665, "W_cal1": 1.820294, "PHI_cal1": -3.068852, "w_cal1": 0.033592, "C_cal2": 0.239068, "W_cal2": 10.308671, "PHI_cal2": -1.251776, "w_cal2": 0.177478}, "r2": 0.992519, "sigma": 0.132307},
    "ecfg_2du_2du": {"n_log": 2, "n_cal": 2, "log_damps": ['d', 'u'], "cal_damps": ['d', 'u'], "params": {"A": -1.131272, "B": 5.012643, "C_log1": 0.270433, "W_log1": 7.953887, "PHI_log1": 0.262641, "w_log1": 0.069334, "C_log2": 0.112180, "W_log2": 16.454993, "PHI_log2": 1.452554, "C_cal1": 0.651536, "W_cal1": 2.993211, "PHI_cal1": 0.385461, "w_cal1": 0.330616, "C_cal2": 0.290289, "W_cal2": 1.677997, "PHI_cal2": -1.076457}, "r2": 0.991493, "sigma": 0.141092},
    "ecfg_2du_2uu": {"n_log": 2, "n_cal": 2, "log_damps": ['d', 'u'], "cal_damps": ['u', 'u'], "params": {"A": -1.183241, "B": 5.102046, "C_log1": 0.507664, "W_log1": 7.620463, "PHI_log1": 1.622911, "w_log1": 0.106671, "C_log2": 0.092300, "W_log2": 20.256307, "PHI_log2": -0.722984, "C_cal1": 0.193756, "W_cal1": 1.907184, "PHI_cal1": 2.575084, "C_cal2": 0.106908, "W_cal2": 3.323490, "PHI_cal2": -2.693162}, "r2": 0.992866, "sigma": 0.129202},
    "ecfg_2uu_0": {"n_log": 2, "n_cal": 0, "log_damps": ['u', 'u'], "cal_damps": [], "params": {"A": -1.176813, "B": 5.085663, "C_log1": 0.180520, "W_log1": 20.892431, "PHI_log1": -1.111269, "C_log2": 0.241183, "W_log2": 7.180144, "PHI_log2": 1.870043}, "r2": 0.980460, "sigma": 0.213831},
    "ecfg_2uu_1d": {"n_log": 2, "n_cal": 1, "log_damps": ['u', 'u'], "cal_damps": ['d'], "params": {"A": -1.124459, "B": 5.004040, "C_log1": 0.265000, "W_log1": 7.067953, "PHI_log1": 1.781609, "C_log2": 0.159753, "W_log2": 8.815924, "PHI_log2": 0.461503, "C_cal": 0.247165, "W_cal": 1.763631, "PHI_cal": -2.340795, "w_cal": 0.029502}, "r2": 0.990087, "sigma": 0.152303},
    "ecfg_2uu_1u": {"n_log": 2, "n_cal": 1, "log_damps": ['u', 'u'], "cal_damps": ['u'], "params": {"A": -1.184602, "B": 5.073838, "C_log1": 0.159905, "W_log1": 8.867766, "PHI_log1": 0.497450, "C_log2": 0.257808, "W_log2": 7.160579, "PHI_log2": 1.707531, "C_cal": 0.211546, "W_cal": 1.788972, "PHI_cal": -2.646276}, "r2": 0.989315, "sigma": 0.158123},
    "ecfg_2uu_2dd": {"n_log": 2, "n_cal": 2, "log_damps": ['u', 'u'], "cal_damps": ['d', 'd'], "params": {"A": -1.274573, "B": 5.177999, "C_log1": 0.132538, "W_log1": 21.168160, "PHI_log1": -1.467086, "C_log2": 0.130871, "W_log2": 17.071574, "PHI_log2": 1.327892, "C_cal1": 0.569327, "W_cal1": 2.146082, "PHI_cal1": 2.441430, "w_cal1": 0.130633, "C_cal2": 0.284337, "W_cal2": 1.131472, "PHI_cal2": 1.920439, "w_cal2": 0.085599}, "r2": 0.990184, "sigma": 0.151560},
    "ecfg_2uu_2du": {"n_log": 2, "n_cal": 2, "log_damps": ['u', 'u'], "cal_damps": ['d', 'u'], "params": {"A": -1.087292, "B": 4.964856, "C_log1": 0.161126, "W_log1": 6.347032, "PHI_log1": -2.708822, "C_log2": 0.100394, "W_log2": 37.186575, "PHI_log2": 1.932340, "C_cal1": 0.824688, "W_cal1": 4.212409, "PHI_cal1": -2.462779, "w_cal1": 0.375861, "C_cal2": 0.298069, "W_cal2": 1.742661, "PHI_cal2": -2.090698}, "r2": 0.991786, "sigma": 0.138637},
    "ecfg_2uu_2uu": {"n_log": 2, "n_cal": 2, "log_damps": ['u', 'u'], "cal_damps": ['u', 'u'], "params": {"A": -1.167409, "B": 5.049106, "C_log1": 0.293024, "W_log1": 7.098379, "PHI_log1": 1.931776, "C_log2": 0.185402, "W_log2": 8.846058, "PHI_log2": 0.759028, "C_cal1": 0.224986, "W_cal1": 1.814174, "PHI_cal1": -2.934866, "C_cal2": 0.125693, "W_cal2": 3.283288, "PHI_cal2": -2.543245}, "r2": 0.992145, "sigma": 0.135577},
}


class EntropyPPLModel:
    """Entropy PPL — HybPPL variant with Shannon entropy envelope damping.

    Replaces the t^(-D) power-law damping of HybPPL with a normalized
    Shannon entropy envelope E(w*t) = max(-w*t*ln(w*t), 0) / (1/e).

    The entropy envelope peaks when adoption uncertainty is maximal
    (w*t = 1/e) and decays to zero when adoption is "resolved" (w*t = 1).

    Formula (2+2 version, 16 params):
        log10(price) = A + B*log10(t)
            + C1*E(w1*t)*cos(W1*ln(t)+P1)     # entropy-damped log-periodic 1
            + C3*E(w2*t)*cos(W2*ln(t)+P3)     # entropy-damped log-periodic 2
            + C2*cos(Wc1*t+P2)                 # undamped halving cycle
            + C4*cos(Wc2*t+P4)                 # undamped sub-halving

    R²=0.993320, σ=0.125028
    """
    name = "Entropy PPL"
    short_name = "eppl"
    legend_name = "EPPL"
    dash_style = "dot"
    quantized = True

    # ── Fitted parameters (EPPL 2+2) ────────────────────────────────────
    _A    = -1.167364
    _B    =  5.079560
    _C1   =  0.250431    # log osc 1 amplitude
    _W1   = 16.823756    # log osc 1 frequency
    _P1   =  1.460422    # log osc 1 phase
    _w1   =  0.251550    # log osc 1 entropy rate
    _C3   =  0.556269    # log osc 2 amplitude
    _W2   =  7.803554    # log osc 2 frequency
    _P3   =  1.373041    # log osc 2 phase
    _w2   =  0.107049    # log osc 2 entropy rate
    _C2   =  0.202747    # cal osc 1 amplitude
    _Wc1  =  1.881312    # cal osc 1 frequency (T=3.34yr)
    _P2   =  2.520900    # cal osc 1 phase
    _C4   =  0.113542    # cal osc 2 amplitude
    _Wc2  =  3.355482    # cal osc 2 frequency (T=1.87yr)
    _P4   =  3.033230    # cal osc 2 phase
    _sigma = 0.125028

    def __init__(self, price_years, price_prices, quantiles):
        self.fits = {}
        for q in quantiles:
            z = _lazy_norm().ppf(q)
            self.fits[q] = {"z_shift": z * self._sigma}
        self.quantiles = sorted(self.fits.keys())
        self._build_colors()

    @staticmethod
    def entropy_env(t, w):
        """Normalized Shannon entropy envelope: E(x) = max(-x*ln(x), 0) / (1/e)."""
        x = w * t
        raw = -x * np.log(np.maximum(x, 1e-30))
        return np.maximum(raw, 0.0) / (1.0 / np.e)

    def _model_log10(self, t):
        """Evaluate the 2+2 entropy PPL formula."""
        t_arr = np.asarray(t, float)
        scalar = t_arr.ndim == 0
        if scalar:
            t_arr = t_arr.reshape(1)
        t_safe = np.maximum(t_arr, 0.1)

        result = self._A + self._B * np.log10(t_safe)
        # Entropy-damped log-periodic term 1
        result += self._C1 * self.entropy_env(t_safe, self._w1) * np.cos(
            self._W1 * np.log(t_safe) + self._P1)
        # Entropy-damped log-periodic term 2
        result += self._C3 * self.entropy_env(t_safe, self._w2) * np.cos(
            self._W2 * np.log(t_safe) + self._P3)
        # Undamped halving cycle
        result += self._C2 * np.cos(self._Wc1 * t_safe + self._P2)
        # Undamped sub-halving
        result += self._C4 * np.cos(self._Wc2 * t_safe + self._P4)

        return float(result[0]) if scalar else result

    def price_at(self, q, t):
        t_arr = np.asarray(t, float)
        log_median = self._model_log10(t_arr)
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

    # ── Decomposition ────────────────────────────────────────────────────

    component_names = [
        "A (constant)",
        "B\u00b7log\u2081\u2080(t)",
        "entropy log osc 1 (\u03c9\u2081)",
        "entropy log osc 2 (\u03c9\u2082)",
        "undamped cal osc 1 (\u03c9_c\u2081)",
        "undamped cal osc 2 (\u03c9_c\u2082)",
    ]

    formula_log10_latex = (
        r"A + B \log_{10}(t)"
        r" + C_1 \cdot E(w_1 t) \cos(\omega_1 \ln t + \varphi_1)"
        r" + C_3 \cdot E(w_2 t) \cos(\omega_2 \ln t + \varphi_3)"
        r" + C_2 \cos(\omega_{c1} t + \varphi_2)"
        r" + C_4 \cos(\omega_{c2} t + \varphi_4)"
    )
    formula_product_latex = None  # too complex for product form

    @property
    def component_details(self):
        return {
            "A (constant)": (
                "A",
                [("A", "_A")],
            ),
            "B\u00b7log\u2081\u2080(t)": (
                "B\u00b7log\u2081\u2080(t)",
                [("B", "_B")],
            ),
            "entropy log osc 1 (\u03c9\u2081)": (
                "C\u2081\u00b7E(w\u2081\u00b7t)\u00b7cos(\u03c9\u2081\u00b7ln(t)+\u03c6\u2081)",
                [("C\u2081", "_C1"), ("\u03c9\u2081", "_W1"),
                 ("\u03c6\u2081", "_P1"), ("w\u2081", "_w1")],
            ),
            "entropy log osc 2 (\u03c9\u2082)": (
                "C\u2083\u00b7E(w\u2082\u00b7t)\u00b7cos(\u03c9\u2082\u00b7ln(t)+\u03c6\u2083)",
                [("C\u2083", "_C3"), ("\u03c9\u2082", "_W2"),
                 ("\u03c6\u2083", "_P3"), ("w\u2082", "_w2")],
            ),
            "undamped cal osc 1 (\u03c9_c\u2081)": (
                "C\u2082\u00b7cos(\u03c9_c\u2081\u00b7t+\u03c6\u2082)",
                [("C\u2082", "_C2"), ("\u03c9_c\u2081", "_Wc1"),
                 ("\u03c6\u2082", "_P2")],
            ),
            "undamped cal osc 2 (\u03c9_c\u2082)": (
                "C\u2084\u00b7cos(\u03c9_c\u2082\u00b7t+\u03c6\u2084)",
                [("C\u2084", "_C4"), ("\u03c9_c\u2082", "_Wc2"),
                 ("\u03c6\u2084", "_P4")],
            ),
        }

    def components(self, t):
        """Decompose into constant + trend + 4 oscillatory terms."""
        t_arr = np.asarray(t, float)
        scalar = t_arr.ndim == 0
        if scalar:
            t_arr = t_arr.reshape(1)
        t_safe = np.maximum(t_arr, 0.1)

        result = {
            "A (constant)":                    np.full_like(t_safe, self._A),
            "B\u00b7log\u2081\u2080(t)":        self._B * np.log10(t_safe),
            "entropy log osc 1 (\u03c9\u2081)": self._C1 * self.entropy_env(t_safe, self._w1) * np.cos(
                self._W1 * np.log(t_safe) + self._P1),
            "entropy log osc 2 (\u03c9\u2082)": self._C3 * self.entropy_env(t_safe, self._w2) * np.cos(
                self._W2 * np.log(t_safe) + self._P3),
            "undamped cal osc 1 (\u03c9_c\u2081)": self._C2 * np.cos(
                self._Wc1 * t_safe + self._P2),
            "undamped cal osc 2 (\u03c9_c\u2082)": self._C4 * np.cos(
                self._Wc2 * t_safe + self._P4),
        }
        if scalar:
            result = {k: float(v[0]) for k, v in result.items()}
        return result

    def _build_colors(self):
        """Warm amber/orange palette — entropy PPL model."""
        self.colors = {}
        n = len(self.quantiles)
        for i, q in enumerate(self.quantiles):
            frac = i / max(n - 1, 1)
            r = int(180 + 40 * frac)     # 180 → 220
            g = int(120 + 50 * frac)     # 120 → 170
            b = int(30 + 40 * frac)      # 30 → 70
            self.colors[q] = f"#{r:02x}{g:02x}{b:02x}"


class EPPLConfigModel:
    """Generic EPPL config model -- loads pre-fitted params for any config.

    Config key format: ecfg_{log_spec}_{cal_spec}
    where spec = "0" or "{count}{damps}" e.g. "2du" = 2 freqs, first damped,
    second undamped.

    Model: log10(price) = A + B*log10(t) + sum(log_osc_i) + sum(cal_osc_i)
    where:
      entropy-damped log: C * E(w*t) * cos(W * ln(t) + PHI)
      undamped log:       C * cos(W * ln(t) + PHI)
      entropy-damped cal: C * E(w*t) * cos(W * t + PHI)
      undamped cal:       C * cos(W * t + PHI)
    with E(x) = max(-x*ln(x), 0) / (1/e)   (normalized Shannon entropy envelope)
    """
    quantized = True

    @staticmethod
    def entropy_env(t, w):
        """Normalized Shannon entropy envelope: E(x) = max(-x*ln(x), 0) / (1/e)."""
        x = w * t
        raw = -x * np.log(np.maximum(x, 1e-30))
        return np.maximum(raw, 0.0) / (1.0 / np.e)

    def __init__(self, config_key, price_years, price_prices, quantiles):
        cfg = _EPPL_CONFIG_PARAMS.get(config_key)
        if cfg is None:
            raise ValueError(f"Unknown EPPL config: {config_key}")
        self._config_key = config_key
        self._cfg = cfg
        self._params = cfg["params"]
        self._sigma = cfg["sigma"]
        self._n_log = cfg["n_log"]
        self._n_cal = cfg["n_cal"]
        self._log_damps = cfg["log_damps"]
        self._cal_damps = cfg["cal_damps"]
        self.r2 = cfg["r2"]

        # Readable names
        self.name = config_key
        self.short_name = config_key
        spec = config_key.replace("ecfg_", "")
        self.legend_name = spec.upper()
        self.dash_style = "dot"

        # Build fits dict for quantile bands (Gaussian shift)
        self.fits = {}
        for q in quantiles:
            z = _lazy_norm().ppf(q)
            self.fits[q] = {"z_shift": z * self._sigma}
        self.quantiles = sorted(self.fits.keys())
        self._build_colors()

    def _model_log10(self, t):
        """Evaluate the model at time t using stored params."""
        t = np.asarray(t, float)
        ts = np.maximum(t, 0.1)
        p = self._params
        result = p["A"] + p["B"] * np.log10(ts)

        # Log-periodic terms
        for i in range(self._n_log):
            suffix = str(i + 1) if self._n_log > 1 else ""
            C = p[f"C_log{suffix}"]
            W = p[f"W_log{suffix}"]
            PHI = p[f"PHI_log{suffix}"]
            if self._log_damps[i] == "d":
                w = p[f"w_log{suffix}"]
                result = result + C * self.entropy_env(ts, w) * np.cos(W * np.log(ts) + PHI)
            else:
                result = result + C * np.cos(W * np.log(ts) + PHI)

        # Calendar terms
        for i in range(self._n_cal):
            suffix = str(i + 1) if self._n_cal > 1 else ""
            C = p[f"C_cal{suffix}"]
            W = p[f"W_cal{suffix}"]
            PHI = p[f"PHI_cal{suffix}"]
            if self._cal_damps[i] == "d":
                w = p[f"w_cal{suffix}"]
                result = result + C * self.entropy_env(ts, w) * np.cos(W * ts + PHI)
            else:
                result = result + C * np.cos(W * ts + PHI)

        return result

    def price_at(self, q, t):
        t_arr = np.asarray(t, float)
        log_median = self._model_log10(t_arr)
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

    @property
    def component_names(self):
        names = ["A (constant)", "B\u00b7log\u2081\u2080(t)"]
        for i in range(self._n_log):
            d = self._log_damps[i]
            names.append(f"log osc {i+1} ({'entropy damped' if d == 'd' else 'undamped'})")
        for i in range(self._n_cal):
            d = self._cal_damps[i]
            names.append(f"cal osc {i+1} ({'entropy damped' if d == 'd' else 'undamped'})")
        return names

    @property
    def formula_log10_latex(self):
        parts = [r"A + B \log_{10}(t)"]
        for i in range(self._n_log):
            d = self._log_damps[i]
            idx = i + 1
            if d == "d":
                parts.append(rf"C_{{l{idx}}} E(w_{{l{idx}}} t) \cos(\omega_{{l{idx}}} \ln t + \varphi_{{l{idx}}})")
            else:
                parts.append(rf"C_{{l{idx}}} \cos(\omega_{{l{idx}}} \ln t + \varphi_{{l{idx}}})")
        for i in range(self._n_cal):
            d = self._cal_damps[i]
            idx = i + 1
            if d == "d":
                parts.append(rf"C_{{c{idx}}} E(w_{{c{idx}}} t) \cos(\omega_{{c{idx}}} t + \varphi_{{c{idx}}})")
            else:
                parts.append(rf"C_{{c{idx}}} \cos(\omega_{{c{idx}}} t + \varphi_{{c{idx}}})")
        return " + ".join(parts)

    @property
    def formula_product_latex(self):
        return None  # too complex for product form

    @property
    def component_details(self):
        det = {
            "A (constant)": ("A", [("A", "A")]),
            "B\u00b7log\u2081\u2080(t)": ("B\u00b7log\u2081\u2080(t)", [("B", "B")]),
        }
        for i in range(self._n_log):
            d = self._log_damps[i]
            name = f"log osc {i+1} ({'entropy damped' if d == 'd' else 'undamped'})"
            if d == "d":
                det[name] = (
                    f"C\u00b7E(w\u00b7t)\u00b7cos(\u03c9\u00b7ln(t)+\u03c6)",
                    [],
                )
            else:
                det[name] = ("C\u00b7cos(\u03c9\u00b7ln(t)+\u03c6)", [])
        for i in range(self._n_cal):
            d = self._cal_damps[i]
            name = f"cal osc {i+1} ({'entropy damped' if d == 'd' else 'undamped'})"
            if d == "d":
                det[name] = (
                    f"C\u00b7E(w\u00b7t)\u00b7cos(\u03c9\u00b7t+\u03c6)",
                    [],
                )
            else:
                det[name] = ("C\u00b7cos(\u03c9\u00b7t+\u03c6)", [])
        return det

    def components(self, t):
        """Decompose into individual additive terms."""
        t = np.asarray(t, float)
        ts = np.maximum(t, 0.1)
        p = self._params
        result = {
            "A (constant)": np.full_like(ts, p["A"]),
            "B\u00b7log\u2081\u2080(t)": p["B"] * np.log10(ts),
        }
        for i in range(self._n_log):
            suffix = str(i + 1) if self._n_log > 1 else ""
            d = self._log_damps[i]
            C = p[f"C_log{suffix}"]; W = p[f"W_log{suffix}"]; PHI = p[f"PHI_log{suffix}"]
            name = f"log osc {i+1} ({'entropy damped' if d == 'd' else 'undamped'})"
            if d == "d":
                w = p[f"w_log{suffix}"]
                result[name] = C * self.entropy_env(ts, w) * np.cos(W * np.log(ts) + PHI)
            else:
                result[name] = C * np.cos(W * np.log(ts) + PHI)
        for i in range(self._n_cal):
            suffix = str(i + 1) if self._n_cal > 1 else ""
            d = self._cal_damps[i]
            C = p[f"C_cal{suffix}"]; W = p[f"W_cal{suffix}"]; PHI = p[f"PHI_cal{suffix}"]
            name = f"cal osc {i+1} ({'entropy damped' if d == 'd' else 'undamped'})"
            if d == "d":
                w = p[f"w_cal{suffix}"]
                result[name] = C * self.entropy_env(ts, w) * np.cos(W * ts + PHI)
            else:
                result[name] = C * np.cos(W * ts + PHI)
        return result

    def _build_colors(self):
        """Teal-cyan palette -- distinct from HybPPL's gray-blue."""
        self.colors = {}
        n = len(self.quantiles)
        for i, q in enumerate(self.quantiles):
            frac = i / max(n - 1, 1)
            r = int(20 + 60 * frac)      # 20 -> 80
            g = int(140 + 50 * frac)     # 140 -> 190
            b = int(140 + 40 * frac)     # 140 -> 180
            self.colors[q] = f"#{r:02x}{g:02x}{b:02x}"


# ── HybPPL config params (auto-generated) ──
_HYBPPL_CONFIG_PARAMS = {
    "cfg_0_0": {"n_log": 0, "n_cal": 0, "log_damps": [], "cal_damps": [], "params": {"A": -1.173875, "B": 5.081360}, "r2": 0.962650, "sigma": 0.295634},
    "cfg_0_1d": {"n_log": 0, "n_cal": 1, "log_damps": [], "cal_damps": ['d'], "params": {"A": -1.211645, "B": 5.108969, "C_cal": 0.276881, "W_cal": 1.765340, "PHI_cal": -2.280041, "D_cal": 0.000000}, "r2": 0.978937, "sigma": 0.222008},
    "cfg_0_1u": {"n_log": 0, "n_cal": 1, "log_damps": [], "cal_damps": ['u'], "params": {"A": -1.211646, "B": 5.108970, "C_cal": 0.276881, "W_cal": 1.765341, "PHI_cal": -2.280043}, "r2": 0.978937, "sigma": 0.222008},
    "cfg_0_2dd": {"n_log": 0, "n_cal": 2, "log_damps": [], "cal_damps": ['d', 'd'], "params": {"A": -1.047754, "B": 4.943547, "C_cal1": 1.352713, "W_cal1": 3.066049, "PHI_cal1": -0.612299, "D_cal1": 1.531193, "C_cal2": 0.746118, "W_cal2": 1.716575, "PHI_cal2": -1.783384, "D_cal2": 0.442203}, "r2": 0.986731, "sigma": 0.176210},
    "cfg_0_2du": {"n_log": 0, "n_cal": 2, "log_damps": [], "cal_damps": ['d', 'u'], "params": {"A": -1.087910, "B": 4.978008, "C_cal1": 0.738656, "W_cal1": 3.058029, "PHI_cal1": -0.383182, "D_cal1": 1.035207, "C_cal2": 0.299985, "W_cal2": 1.729902, "PHI_cal2": -1.901978}, "r2": 0.985651, "sigma": 0.183242},
    "cfg_0_2uu": {"n_log": 0, "n_cal": 2, "log_damps": [], "cal_damps": ['u', 'u'], "params": {"A": -1.328861, "B": 5.243841, "C_cal1": 0.292755, "W_cal1": 1.781646, "PHI_cal1": -2.416708, "C_cal2": 0.152796, "W_cal2": 0.813027, "PHI_cal2": -2.741282}, "r2": 0.983187, "sigma": 0.198349},
    "cfg_1d_0": {"n_log": 1, "n_cal": 0, "log_damps": ['d'], "cal_damps": [], "params": {"A": -1.153820, "B": 5.079271, "C_log": 0.733974, "W_log": 7.557911, "PHI_log": 1.377121, "D_log": 0.607966}, "r2": 0.978055, "sigma": 0.226611},
    "cfg_1d_1d": {"n_log": 1, "n_cal": 1, "log_damps": ['d'], "cal_damps": ['d'], "params": {"A": -1.146875, "B": 5.051453, "C_log": 0.689836, "W_log": 7.420057, "PHI_log": 1.453329, "D_log": 0.708168, "C_cal": 0.233044, "W_cal": 1.733155, "PHI_cal": -1.923037, "D_cal": 0.000000}, "r2": 0.988870, "sigma": 0.161384},
    "cfg_1d_1u": {"n_log": 1, "n_cal": 1, "log_damps": ['d'], "cal_damps": ['u'], "params": {"A": -1.146875, "B": 5.051454, "C_log": 0.689837, "W_log": 7.420057, "PHI_log": 1.453329, "D_log": 0.708168, "C_cal": 0.233044, "W_cal": 1.733155, "PHI_cal": -1.923038}, "r2": 0.988870, "sigma": 0.161384},
    "cfg_1d_2dd": {"n_log": 1, "n_cal": 2, "log_damps": ['d'], "cal_damps": ['d', 'd'], "params": {"A": -1.119963, "B": 5.023515, "C_log": 0.894153, "W_log": 7.537078, "PHI_log": 1.247825, "D_log": 0.859867, "C_cal1": 0.575944, "W_cal1": 10.000000, "PHI_cal1": -0.600808, "D_cal1": 1.461041, "C_cal2": 0.234815, "W_cal2": 1.724073, "PHI_cal2": -1.813160, "D_cal2": 0.000000}, "r2": 0.990862, "sigma": 0.146232},
    "cfg_1d_2du": {"n_log": 1, "n_cal": 2, "log_damps": ['d'], "cal_damps": ['d', 'u'], "params": {"A": -1.135472, "B": 5.037822, "C_log": 0.738890, "W_log": 7.356010, "PHI_log": 1.659095, "D_log": 0.730226, "C_cal1": 0.235259, "W_cal1": 1.750674, "PHI_cal1": -2.086894, "D_cal1": 0.000000, "C_cal2": 0.114588, "W_cal2": 3.280720, "PHI_cal2": -2.452578}, "r2": 0.991324, "sigma": 0.142482},
    "cfg_1d_2uu": {"n_log": 1, "n_cal": 2, "log_damps": ['d'], "cal_damps": ['u', 'u'], "params": {"A": -1.135472, "B": 5.037822, "C_log": 0.738890, "W_log": 7.356010, "PHI_log": 1.659096, "D_log": 0.730225, "C_cal1": 0.235259, "W_cal1": 1.750674, "PHI_cal1": -2.086893, "C_cal2": 0.114588, "W_cal2": 3.280720, "PHI_cal2": -2.452577}, "r2": 0.991324, "sigma": 0.142482},
    "cfg_1u_0": {"n_log": 1, "n_cal": 0, "log_damps": ['u'], "cal_damps": [], "params": {"A": -1.232639, "B": 5.183069, "C_log": 0.230735, "W_log": 7.722702, "PHI_log": 1.187301}, "r2": 0.973977, "sigma": 0.246768},
    "cfg_1u_1d": {"n_log": 1, "n_cal": 1, "log_damps": ['u'], "cal_damps": ['d'], "params": {"A": -1.234864, "B": 5.154251, "C_log": 0.184756, "W_log": 7.369940, "PHI_log": 1.623339, "C_cal": 0.245734, "W_cal": 1.760620, "PHI_cal": -2.246105, "D_cal": 0.000000}, "r2": 0.986223, "sigma": 0.179550},
    "cfg_1u_1u": {"n_log": 1, "n_cal": 1, "log_damps": ['u'], "cal_damps": ['u'], "params": {"A": -1.234864, "B": 5.154251, "C_log": 0.184756, "W_log": 7.369940, "PHI_log": 1.623339, "C_cal": 0.245734, "W_cal": 1.760620, "PHI_cal": -2.246106}, "r2": 0.986223, "sigma": 0.179550},
    "cfg_1u_2dd": {"n_log": 1, "n_cal": 2, "log_damps": ['u'], "cal_damps": ['d', 'd'], "params": {"A": -1.006185, "B": 4.865911, "C_log": 0.161247, "W_log": 5.310100, "PHI_log": -0.347192, "C_cal1": 1.772314, "W_cal1": 3.074611, "PHI_cal1": -0.657795, "D_cal1": 1.706594, "C_cal2": 0.848744, "W_cal2": 1.734324, "PHI_cal2": -2.025364, "D_cal2": 0.492016}, "r2": 0.990270, "sigma": 0.150893},
    "cfg_1u_2du": {"n_log": 1, "n_cal": 2, "log_damps": ['u'], "cal_damps": ['d', 'u'], "params": {"A": -1.094170, "B": 4.981448, "C_log": 0.161787, "W_log": 6.664948, "PHI_log": 2.924047, "C_cal1": 0.615604, "W_cal1": 3.121978, "PHI_cal1": -0.910641, "D_cal1": 0.898565, "C_cal2": 0.276107, "W_cal2": 1.740379, "PHI_cal2": -2.038337}, "r2": 0.990144, "sigma": 0.151865},
    "cfg_1u_2uu": {"n_log": 1, "n_cal": 2, "log_damps": ['u'], "cal_damps": ['u', 'u'], "params": {"A": -1.195363, "B": 5.101264, "C_log": 0.178158, "W_log": 7.144729, "PHI_log": 2.014367, "C_cal1": 0.256183, "W_cal1": 1.760251, "PHI_cal1": -2.212877, "C_cal2": 0.121646, "W_cal2": 3.167757, "PHI_cal2": -1.213413}, "r2": 0.989168, "sigma": 0.159207},
    "cfg_2dd_0": {"n_log": 2, "n_cal": 0, "log_damps": ['d', 'd'], "cal_damps": [], "params": {"A": -1.130959, "B": 5.038805, "C_log1": 0.705568, "W_log1": 7.377517, "PHI_log1": 1.583198, "D_log1": 0.566186, "C_log2": 0.171854, "W_log2": 20.903808, "PHI_log2": -1.156632, "D_log2": 0.010000}, "r2": 0.983997, "sigma": 0.193516},
    "cfg_2dd_1d": {"n_log": 2, "n_cal": 1, "log_damps": ['d', 'd'], "cal_damps": ['d'], "params": {"A": -1.113052, "B": 5.013922, "C_log1": 0.765452, "W_log1": 7.471817, "PHI_log1": 1.297976, "D_log1": 0.773463, "C_log2": 0.392752, "W_log2": 15.993349, "PHI_log2": 1.889603, "D_log2": 0.932804, "C_cal": 0.257514, "W_cal": 1.720226, "PHI_cal": -1.736945, "D_cal": 0.000000}, "r2": 0.990843, "sigma": 0.146384},
    "cfg_2dd_1u": {"n_log": 2, "n_cal": 1, "log_damps": ['d', 'd'], "cal_damps": ['u'], "params": {"A": -1.113052, "B": 5.013922, "C_log1": 0.392751, "W_log1": 15.993349, "PHI_log1": 1.889602, "D_log1": 0.932802, "C_log2": 0.765451, "W_log2": 7.471817, "PHI_log2": 1.297977, "D_log2": 0.773462, "C_cal": 0.257514, "W_cal": 1.720226, "PHI_cal": -1.736945}, "r2": 0.990843, "sigma": 0.146384},
    "cfg_2dd_2dd": {"n_log": 2, "n_cal": 2, "log_damps": ['d', 'd'], "cal_damps": ['d', 'd'], "params": {"A": -1.113752, "B": 5.018172, "C_log1": 0.090187, "W_log1": 36.963949, "PHI_log1": 2.332185, "D_log1": 0.010000, "C_log2": 0.872584, "W_log2": 7.574685, "PHI_log2": 1.172171, "D_log2": 0.860020, "C_cal1": 0.245779, "W_cal1": 1.720867, "PHI_cal1": -1.794831, "D_cal1": 0.000000, "C_cal2": 0.538482, "W_cal2": 10.310219, "PHI_cal2": -1.305199, "D_cal2": 1.360139}, "r2": 0.992770, "sigma": 0.130072},
    "cfg_2dd_2du": {"n_log": 2, "n_cal": 2, "log_damps": ['d', 'd'], "cal_damps": ['d', 'u'], "params": {"A": -1.088318, "B": 4.986522, "C_log1": 0.879333, "W_log1": 7.206259, "PHI_log1": 1.974339, "D_log1": 0.804974, "C_log2": 0.366188, "W_log2": 13.038626, "PHI_log2": -2.676943, "D_log2": 0.983059, "C_cal1": 0.394977, "W_cal1": 3.259796, "PHI_cal1": -2.213342, "D_cal1": 0.532655, "C_cal2": 0.230152, "W_cal2": 1.743372, "PHI_cal2": -1.994561}, "r2": 0.992768, "sigma": 0.130093},
    "cfg_2dd_2uu": {"n_log": 2, "n_cal": 2, "log_damps": ['d', 'd'], "cal_damps": ['u', 'u'], "params": {"A": -1.114174, "B": 5.017407, "C_log1": 0.891032, "W_log1": 7.483928, "PHI_log1": 1.389298, "D_log1": 0.832947, "C_log2": 0.422419, "W_log2": 16.238168, "PHI_log2": 1.885353, "D_log2": 1.165713, "C_cal1": 0.242036, "W_cal1": 1.739827, "PHI_cal1": -1.918656, "C_cal2": 0.105488, "W_cal2": 3.340840, "PHI_cal2": 3.134976}, "r2": 0.992744, "sigma": 0.130305},
    "cfg_2du_0": {"n_log": 2, "n_cal": 0, "log_damps": ['d', 'u'], "cal_damps": [], "params": {"A": -1.130641, "B": 5.038329, "C_log1": 0.705462, "W_log1": 7.376939, "PHI_log1": 1.583467, "D_log1": 0.565812, "C_log2": 0.168837, "W_log2": 20.904199, "PHI_log2": -1.157210}, "r2": 0.984022, "sigma": 0.193364},
    "cfg_2du_1d": {"n_log": 2, "n_cal": 1, "log_damps": ['d', 'u'], "cal_damps": ['d'], "params": {"A": -1.163916, "B": 5.073560, "C_log1": 0.670547, "W_log1": 7.443535, "PHI_log1": 1.448851, "D_log1": 0.682897, "C_log2": 0.088414, "W_log2": 20.195363, "PHI_log2": -0.269939, "C_cal": 0.210935, "W_cal": 1.773495, "PHI_cal": -2.132973, "D_cal": 0.000000}, "r2": 0.989885, "sigma": 0.153848},
    "cfg_2du_1u": {"n_log": 2, "n_cal": 1, "log_damps": ['d', 'u'], "cal_damps": ['u'], "params": {"A": -1.144272, "B": 5.049050, "C_log1": 0.701091, "W_log1": 7.420752, "PHI_log1": 1.440933, "D_log1": 0.729824, "C_log2": 0.096745, "W_log2": 37.086656, "PHI_log2": 2.047683, "C_cal": 0.242614, "W_cal": 1.729290, "PHI_cal": -1.897870}, "r2": 0.990826, "sigma": 0.146514},
    "cfg_2du_2dd": {"n_log": 2, "n_cal": 2, "log_damps": ['d', 'u'], "cal_damps": ['d', 'd'], "params": {"A": -1.192589, "B": 5.080377, "C_log1": 0.585018, "W_log1": 7.242775, "PHI_log1": 1.464854, "D_log1": 0.549788, "C_log2": 0.098792, "W_log2": 37.128293, "PHI_log2": 1.998761, "C_cal1": 0.364974, "W_cal1": 1.077836, "PHI_cal1": -3.124365, "D_cal1": 0.690705, "C_cal2": 0.271926, "W_cal2": 1.712807, "PHI_cal2": -1.726335, "D_cal2": 0.000000}, "r2": 0.992295, "sigma": 0.134276},
    "cfg_2du_2du": {"n_log": 2, "n_cal": 2, "log_damps": ['d', 'u'], "cal_damps": ['d', 'u'], "params": {"A": -1.138230, "B": 5.015692, "C_log1": 0.251273, "W_log1": 8.651516, "PHI_log1": 1.092978, "D_log1": 0.130090, "C_log2": 0.292981, "W_log2": 6.957905, "PHI_log2": 2.159159, "C_cal1": 0.320649, "W_cal1": 3.264976, "PHI_cal1": -2.369649, "D_cal1": 0.442434, "C_cal2": 0.233646, "W_cal2": 1.811678, "PHI_cal2": -2.922371}, "r2": 0.992315, "sigma": 0.134099},
    "cfg_2du_2uu": {"n_log": 2, "n_cal": 2, "log_damps": ['d', 'u'], "cal_damps": ['u', 'u'], "params": {"A": -1.135646, "B": 5.040344, "C_log1": 0.778133, "W_log1": 7.385649, "PHI_log1": 1.573670, "D_log1": 0.769954, "C_log2": 0.070961, "W_log2": 36.898377, "PHI_log2": 2.249526, "C_cal1": 0.237345, "W_cal1": 1.735501, "PHI_cal1": -1.970635, "C_cal2": 0.088877, "W_cal2": 3.339989, "PHI_cal2": -2.812188}, "r2": 0.992105, "sigma": 0.135923},
    "cfg_2uu_0": {"n_log": 2, "n_cal": 0, "log_damps": ['u', 'u'], "cal_damps": [], "params": {"A": -1.114110, "B": 4.989677, "C_log1": 0.292255, "W_log1": 6.750135, "PHI_log1": 2.263445, "C_log2": 0.232793, "W_log2": 8.884470, "PHI_log2": -0.115770}, "r2": 0.982124, "sigma": 0.204525},
    "cfg_2uu_1d": {"n_log": 2, "n_cal": 1, "log_damps": ['u', 'u'], "cal_damps": ['d'], "params": {"A": -1.649681, "B": 5.791759, "C_log1": 0.214124, "W_log1": 7.529246, "PHI_log1": 1.453833, "C_log2": 0.296423, "W_log2": 2.000000, "PHI_log2": -1.599010, "C_cal": 0.334831, "W_cal": 1.772901, "PHI_cal": -2.321013, "D_cal": 0.141312}, "r2": 0.988897, "sigma": 0.161190},
    "cfg_2uu_1u": {"n_log": 2, "n_cal": 1, "log_damps": ['u', 'u'], "cal_damps": ['u'], "params": {"A": -1.184615, "B": 5.073845, "C_log1": 0.159895, "W_log1": 8.868180, "PHI_log1": 0.496797, "C_log2": 0.257814, "W_log2": 7.160547, "PHI_log2": 1.707803, "C_cal": 0.211553, "W_cal": 1.789004, "PHI_cal": -2.646637}, "r2": 0.989313, "sigma": 0.158137},
    "cfg_2uu_2dd": {"n_log": 2, "n_cal": 2, "log_damps": ['u', 'u'], "cal_damps": ['d', 'd'], "params": {"A": -1.126108, "B": 5.022372, "C_log1": 0.144599, "W_log1": 6.637244, "PHI_log1": 3.009907, "C_log2": 0.098846, "W_log2": 19.941469, "PHI_log2": -0.463206, "C_cal1": 0.291305, "W_cal1": 1.772604, "PHI_cal1": -2.166151, "D_cal1": 0.000000, "C_cal2": 0.583542, "W_cal2": 3.139937, "PHI_cal2": -0.865760, "D_cal2": 0.776542}, "r2": 0.991229, "sigma": 0.143265},
    "cfg_2uu_2du": {"n_log": 2, "n_cal": 2, "log_damps": ['u', 'u'], "cal_damps": ['d', 'u'], "params": {"A": -1.167439, "B": 5.049097, "C_log1": 0.293113, "W_log1": 7.098274, "PHI_log1": 1.932680, "C_log2": 0.185429, "W_log2": 8.847566, "PHI_log2": 0.756730, "C_cal1": 0.225036, "W_cal1": 1.814325, "PHI_cal1": -2.936559, "D_cal1": 0.000000, "C_cal2": 0.125748, "W_cal2": 3.283480, "PHI_cal2": -2.544719}, "r2": 0.992145, "sigma": 0.135576},
    "cfg_2uu_2uu": {"n_log": 2, "n_cal": 2, "log_damps": ['u', 'u'], "cal_damps": ['u', 'u'], "params": {"A": -1.167439, "B": 5.049097, "C_log1": 0.293113, "W_log1": 7.098275, "PHI_log1": 1.932678, "C_log2": 0.185429, "W_log2": 8.847567, "PHI_log2": 0.756729, "C_cal1": 0.125748, "W_cal1": 3.283480, "PHI_cal1": -2.544720, "C_cal2": 0.225036, "W_cal2": 1.814325, "PHI_cal2": -2.936560}, "r2": 0.992145, "sigma": 0.135576},
}


class HybPPLConfigModel:
    """Generic HybPPL config model -- loads pre-fitted params for any config.

    Config key format: cfg_{log_spec}_{cal_spec}
    where spec = "0" or "{count}{damps}" e.g. "2du" = 2 freqs, first damped,
    second undamped.

    Model: log10(price) = A + B*log10(t) + sum(log_osc_i) + sum(cal_osc_i)
    where:
      damped log:   C * t^(-D) * cos(W * ln(t) + PHI)
      undamped log: C * cos(W * ln(t) + PHI)
      damped cal:   C * t^(-D) * cos(W * t + PHI)
      undamped cal: C * cos(W * t + PHI)
    """
    quantized = True

    def __init__(self, config_key, price_years, price_prices, quantiles):
        cfg = _HYBPPL_CONFIG_PARAMS.get(config_key)
        if cfg is None:
            raise ValueError(f"Unknown HybPPL config: {config_key}")
        self._config_key = config_key
        self._cfg = cfg
        self._params = cfg["params"]
        self._sigma = cfg["sigma"]
        self._n_log = cfg["n_log"]
        self._n_cal = cfg["n_cal"]
        self._log_damps = cfg["log_damps"]
        self._cal_damps = cfg["cal_damps"]
        self.r2 = cfg["r2"]

        # Readable names
        self.name = config_key
        self.short_name = config_key
        spec = config_key.replace("cfg_", "")
        self.legend_name = spec.upper()
        self.dash_style = "solid"

        # Build fits dict for quantile bands (Gaussian shift)
        self.fits = {}
        for q in quantiles:
            z = _lazy_norm().ppf(q)
            self.fits[q] = {"z_shift": z * self._sigma}
        self.quantiles = sorted(self.fits.keys())
        self._build_colors()

    def _model_log10(self, t):
        """Evaluate the model at time t using stored params."""
        t = np.asarray(t, float)
        ts = np.maximum(t, 0.1)
        p = self._params
        result = p["A"] + p["B"] * np.log10(ts)

        # Log-periodic terms
        for i in range(self._n_log):
            suffix = str(i + 1) if self._n_log > 1 else ""
            C = p[f"C_log{suffix}"]
            W = p[f"W_log{suffix}"]
            PHI = p[f"PHI_log{suffix}"]
            if self._log_damps[i] == "d":
                D = p[f"D_log{suffix}"]
                result = result + C * ts**(-D) * np.cos(W * np.log(ts) + PHI)
            else:
                result = result + C * np.cos(W * np.log(ts) + PHI)

        # Calendar terms
        for i in range(self._n_cal):
            suffix = str(i + 1) if self._n_cal > 1 else ""
            C = p[f"C_cal{suffix}"]
            W = p[f"W_cal{suffix}"]
            PHI = p[f"PHI_cal{suffix}"]
            if self._cal_damps[i] == "d":
                D = p[f"D_cal{suffix}"]
                result = result + C * ts**(-D) * np.cos(W * ts + PHI)
            else:
                result = result + C * np.cos(W * ts + PHI)

        return result

    def price_at(self, q, t):
        t_arr = np.asarray(t, float)
        log_median = self._model_log10(t_arr)
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

    @property
    def component_names(self):
        names = ["A (constant)", "B\u00b7log\u2081\u2080(t)"]
        for i in range(self._n_log):
            d = self._log_damps[i]
            names.append(f"log osc {i+1} ({'damped' if d == 'd' else 'undamped'})")
        for i in range(self._n_cal):
            d = self._cal_damps[i]
            names.append(f"cal osc {i+1} ({'damped' if d == 'd' else 'undamped'})")
        return names

    @property
    def formula_log10_latex(self):
        parts = [r"A + B \log_{10}(t)"]
        for i in range(self._n_log):
            d = self._log_damps[i]
            idx = i + 1
            if d == "d":
                parts.append(rf"C_{{l{idx}}} t^{{-D_{{l{idx}}}}} \cos(\omega_{{l{idx}}} \ln t + \varphi_{{l{idx}}})")
            else:
                parts.append(rf"C_{{l{idx}}} \cos(\omega_{{l{idx}}} \ln t + \varphi_{{l{idx}}})")
        for i in range(self._n_cal):
            d = self._cal_damps[i]
            idx = i + 1
            if d == "d":
                parts.append(rf"C_{{c{idx}}} t^{{-D_{{c{idx}}}}} \cos(\omega_{{c{idx}}} t + \varphi_{{c{idx}}})")
            else:
                parts.append(rf"C_{{c{idx}}} \cos(\omega_{{c{idx}}} t + \varphi_{{c{idx}}})")
        return " + ".join(parts)

    @property
    def formula_product_latex(self):
        return None  # too complex for product form

    @property
    def component_details(self):
        det = {
            "A (constant)": ("A", [("A", "A")]),
            "B\u00b7log\u2081\u2080(t)": ("B\u00b7log\u2081\u2080(t)", [("B", "B")]),
        }
        for i in range(self._n_log):
            d = self._log_damps[i]
            name = f"log osc {i+1} ({'damped' if d == 'd' else 'undamped'})"
            if d == "d":
                det[name] = (
                    f"C\u00b7t^(\u2212D)\u00b7cos(\u03c9\u00b7ln(t)+\u03c6)",
                    [],
                )
            else:
                det[name] = ("C\u00b7cos(\u03c9\u00b7ln(t)+\u03c6)", [])
        for i in range(self._n_cal):
            d = self._cal_damps[i]
            name = f"cal osc {i+1} ({'damped' if d == 'd' else 'undamped'})"
            if d == "d":
                det[name] = (
                    f"C\u00b7t^(\u2212D)\u00b7cos(\u03c9\u00b7t+\u03c6)",
                    [],
                )
            else:
                det[name] = ("C\u00b7cos(\u03c9\u00b7t+\u03c6)", [])
        return det

    def components(self, t):
        """Decompose into individual additive terms."""
        t = np.asarray(t, float)
        ts = np.maximum(t, 0.1)
        p = self._params
        result = {
            "A (constant)": np.full_like(ts, p["A"]),
            "B\u00b7log\u2081\u2080(t)": p["B"] * np.log10(ts),
        }
        for i in range(self._n_log):
            suffix = str(i + 1) if self._n_log > 1 else ""
            d = self._log_damps[i]
            C = p[f"C_log{suffix}"]; W = p[f"W_log{suffix}"]; PHI = p[f"PHI_log{suffix}"]
            name = f"log osc {i+1} ({'damped' if d == 'd' else 'undamped'})"
            if d == "d":
                D = p[f"D_log{suffix}"]
                result[name] = C * ts**(-D) * np.cos(W * np.log(ts) + PHI)
            else:
                result[name] = C * np.cos(W * np.log(ts) + PHI)
        for i in range(self._n_cal):
            suffix = str(i + 1) if self._n_cal > 1 else ""
            d = self._cal_damps[i]
            C = p[f"C_cal{suffix}"]; W = p[f"W_cal{suffix}"]; PHI = p[f"PHI_cal{suffix}"]
            name = f"cal osc {i+1} ({'damped' if d == 'd' else 'undamped'})"
            if d == "d":
                D = p[f"D_cal{suffix}"]
                result[name] = C * ts**(-D) * np.cos(W * ts + PHI)
            else:
                result[name] = C * np.cos(W * ts + PHI)
        return result

    def _build_colors(self):
        """Neutral gray-blue palette -- distinct from other model families."""
        self.colors = {}
        n = len(self.quantiles)
        for i, q in enumerate(self.quantiles):
            frac = i / max(n - 1, 1)
            r = int(70 + 80 * frac)
            g = int(100 + 60 * frac)
            b = int(140 + 50 * frac)
            self.colors[q] = f"#{r:02x}{g:02x}{b:02x}"


class LogisticModel:
    """Logistic/Gompertz growth model with Gaussian quantile bands.

    Gompertz: log10(price) = K * exp(-exp(-r * (t - t0)))
    where K = carrying capacity (log10 of max price), r = growth rate,
    t0 = inflection point.

    Provides an upper saturation bound that power law models lack.
    """
    name = "Logistic Growth"
    short_name = "gomp"
    legend_name = "Gomp"
    dash_style = "dot"
    quantized = True

    # Fitted parameters (will be overwritten by fit_logistic.py --update)
    _K  =             4.888545  
    _r  =             0.302367  
    _t0 =             4.373878  

    def __init__(self, price_years, price_prices, quantiles):
        mask = price_years >= 1.0
        t = price_years[mask]
        lp = np.log10(price_prices[mask])
        predicted = self._model_log10(t)
        self._sigma = float(np.std(lp - predicted))
        self.fits = {}
        for q in quantiles:
            z = _lazy_norm().ppf(q)
            self.fits[q] = {"z_shift": z * self._sigma}
        self.quantiles = sorted(self.fits.keys())
        self._build_colors()

    def _model_log10(self, t):
        t = np.asarray(t, float)
        return self._K * np.exp(-np.exp(-self._r * (t - self._t0)))

    def price_at(self, q, t):
        t_arr = np.asarray(t, float)
        log_median = self._model_log10(t_arr)
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
        """Steel blue palette — saturation model."""
        self.colors = {}
        n = len(self.quantiles)
        for i, q in enumerate(self.quantiles):
            frac = i / max(n - 1, 1)
            r = int(50 + 60 * frac)
            g = int(90 + 70 * frac)
            b = int(150 + 50 * frac)
            self.colors[q] = f"#{r:02x}{g:02x}{b:02x}"


class BrokenPowerLawModel:
    """Broken (two-segment) power law with Gaussian quantile bands.

    For t < t_break: log10(price) = a1 + b1 * log10(t)
    For t >= t_break: log10(price) = a2 + b2 * log10(t)
    Continuity constraint: a2 = a1 + (b1 - b2) * log10(t_break)
    """
    name = "Broken Power Law"
    short_name = "bpl"
    legend_name = "BPL"
    dash_style = "longdash"
    quantized = True

    # Fitted parameters (will be overwritten by fit_bpl.py --update)
    _a1      = -1.092244  
    _b1      =             4.920330  
    _t_break =             6.694045  
    _b2      =             5.318074  

    def __init__(self, price_years, price_prices, quantiles):
        mask = price_years >= 1.0
        t = price_years[mask]
        lp = np.log10(price_prices[mask])
        predicted = self._model_log10(t)
        self._sigma = float(np.std(lp - predicted))
        self.fits = {}
        for q in quantiles:
            z = _lazy_norm().ppf(q)
            self.fits[q] = {"z_shift": z * self._sigma}
        self.quantiles = sorted(self.fits.keys())
        self._build_colors()

    @property
    def _a2(self):
        return self._a1 + (self._b1 - self._b2) * np.log10(self._t_break)

    def _model_log10(self, t):
        t = np.asarray(t, float)
        t_safe = np.maximum(t, 0.1)
        lt = np.log10(t_safe)
        return np.where(
            t_safe < self._t_break,
            self._a1 + self._b1 * lt,
            self._a2 + self._b2 * lt,
        )

    def price_at(self, q, t):
        t_arr = np.asarray(t, float)
        log_median = self._model_log10(t_arr)
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
        """Amber/tan palette — regime-shift model."""
        self.colors = {}
        n = len(self.quantiles)
        for i, q in enumerate(self.quantiles):
            frac = i / max(n - 1, 1)
            r = int(160 + 60 * frac)
            g = int(110 + 50 * frac)
            b = int(40 + 40 * frac)
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

        # Support line (log10 USD) for component decomposition
        self._log_support = np.log10(np.maximum(self._support_plot, 1e-10))

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

        slope, intercept, *_ = _lazy_linregress()(log_s2f, log_p)
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
