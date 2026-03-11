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
from scipy.stats import linregress
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
