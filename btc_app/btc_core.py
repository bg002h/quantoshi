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


def yr_to_t(cal_year, genesis=pd.Timestamp("2009-01-03")):
    """Calendar year → years since genesis (float)."""
    return (pd.Timestamp(f"{int(cal_year)}-01-01") - genesis).days / 365.25


def today_t(genesis=pd.Timestamp("2009-01-03")):
    """Today → years since genesis (float)."""
    return (pd.Timestamp.today() - genesis).days / 365.25


def today_year():
    """Today as a fractional calendar year."""
    return pd.Timestamp.today().year + (pd.Timestamp.today().day_of_year - 1) / 365.25


def fmt_price(p):
    """Format a USD price with comma thousands separators."""
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
    genesis = pd.Timestamp("2009-01-03")
    ep = sum(l["price"] * l["btc"] for l in lots) / total_w
    et = sum((pd.Timestamp(l["date"]) - genesis).days / 365.25 * l["btc"]
             for l in lots) / total_w
    pct = sum(l["pct_q"] * l["btc"] for l in lots) / total_w
    return ep, et, pct, total_w


# ── model fitting ─────────────────────────────────────────────────────────────

def fit_qr_from_csv(csv_path, quantiles, genesis="2009-01-03", fit_min="2010-01-01"):
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
    """Search for model_data.pkl: explicit > user config > bundle dir > cwd."""
    if explicit_path and Path(explicit_path).exists():
        return explicit_path
    # ~/.config override (written by "Save Model Override")
    cfg = Path.home() / ".config" / "btc-projections" / "model_data.pkl"
    if cfg.exists():
        return str(cfg)
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
        self.genesis       = pd.Timestamp(d.get("GENESIS_DATE", "2009-01-03"))
        self.years_plot_bm = np.array(d["years_plot_bm"])
        self.support_bm    = np.array(d["support_plot_bm"])
        self.comp_by_n     = [np.array(c) for c in d["bm_comp_by_n"]]
        self.bm_r2         = d["bm_r2_comp"]
        self.n_future_max  = d["bm_n_future_max"]
        self.price_dates   = d["price_dates"]
        self.price_years   = np.array(d["price_years"])
        self.price_prices  = np.array(d["price_prices"])
        self.qr_colors     = {float(k): v for k, v in d["qr_colors"].items()}
        raw_ls = d["QR_LINESTYLES"]
        self.qr_linestyles = {float(k): _parse_ls(v) for k, v in raw_ls.items()}
        # Visual config
        for key in ("PLOT_BG_COLOR", "TEXT_COLOR", "TITLE_COLOR", "SPINE_COLOR",
                    "GRID_MAJOR_COLOR", "GRID_MINOR_COLOR", "DATA_COLOR",
                    "CAGR_SEG_C_LO", "CAGR_SEG_C_MID1", "CAGR_SEG_C_MID2", "CAGR_SEG_C_HI"):
            setattr(self, key, d.get(key, "#888888"))
        for key in ("DATA_PT_SIZE", "DATA_PT_SIZE_ZOOM", "ZOOM_YEAR_LO", "ZOOM_YEAR_HI",
                    "CAGR_GRAD_STEPS", "CAGR_HEATMAP_FONTSIZE"):
            setattr(self, key, int(d.get(key, 8)))
        for key in ("ZOOM_PRICE_LO", "ZOOM_PRICE_HI", "CAGR_SEG_B1", "CAGR_SEG_B2"):
            setattr(self, key, float(d.get(key, 0)))
        self.TABLE_YEARS = d.get("TABLE_YEARS", list(range(2025, 2041)))

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


class BubbleModel(_FitsBasedModel):
    """Wraps existing QR bubble model fits."""
    name = "Bubble Model"
    short_name = "bub"
    dash_style = "solid"

    def __init__(self, md):
        self.fits = md.qr_fits
        self.colors = dict(md.qr_colors)
        self.quantiles = sorted(md.qr_fits.keys())


class PowerLawModel(_FitsBasedModel):
    """OLS power law with Gaussian quantile bands.

    All bands share the same slope (OLS slope) but have different intercepts
    shifted by z_q * sigma where sigma is the OLS residual standard deviation.
    This means the bands are parallel lines in log-log space.
    """
    name = "Power Law"
    short_name = "pl"
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


class S2FModel:
    """Stock-to-Flow model — single price trajectory (not quantized).

    Fits log10(price) = a + b * log10(S2F) from historical data, where
    S2F = stock / annual_flow based on the Bitcoin halving schedule.
    """
    name = "Stock-to-Flow"
    short_name = "s2f"
    dash_style = "longdash"
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
