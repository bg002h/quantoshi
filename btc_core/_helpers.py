"""Top-of-file helpers: lazy imports, shrinking-sigma fitting, resqr
evaluation, time/date helpers, QR/price helpers, R2 computation.

Has no intra-package dependencies — every other submodule imports from here.
"""

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from time_basis import T_MIN


# ══════════════════════════════════════════════════════════════════════════════
# Lazy imports (scipy + statsmodels are slow at module load)
# ══════════════════════════════════════════════════════════════════════════════
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

# ── shrinking-σ band helpers ──────────────────────────────────────────────────

def _fit_shrinking_sigma(t, residuals):
    """Fit σ(t) = σ₀·t^(-α) from model residuals (asymmetric up/down).

    Uses windowed σ estimation (overlapping ~500-point windows) and
    log-linear regression on (log(t), log(σ)) to find σ₀ and α.

    Returns (σ₀_up, α_up, σ₀_dn, α_dn).
    """
    order = np.argsort(t)
    t_s = t[order]
    r_s = residuals[order]
    window = min(500, len(t_s) // 4)
    if window < 50:
        s = float(np.std(residuals))
        return s, 0.0, s, 0.0
    n_windows = max(10, len(t_s) // (window // 2))
    edges = np.linspace(0, len(t_s), n_windows + 1, dtype=int)
    t_mid, s_up, s_dn = [], [], []
    for i in range(n_windows):
        lo, hi = int(edges[i]), int(edges[i + 1])
        if hi - lo < 20:
            continue
        chunk = r_s[lo:hi]
        t_mid.append(np.median(t_s[lo:hi]))
        up = chunk[chunk >= 0]
        dn = chunk[chunk < 0]
        s_up.append(np.std(up) if len(up) > 5 else np.std(chunk))
        s_dn.append(np.std(np.abs(dn)) if len(dn) > 5 else np.std(chunk))
    t_mid = np.array(t_mid)
    s_up = np.array(s_up)
    s_dn = np.array(s_dn)

    def _fit_power(t_arr, s_arr):
        mask = (t_arr > 0) & (s_arr > 0)
        if mask.sum() < 3:
            return float(np.mean(s_arr)), 0.0
        from numpy.polynomial.polynomial import polyfit
        c = polyfit(np.log(t_arr[mask]), np.log(s_arr[mask]), 1)
        return float(np.exp(c[0])), float(-c[1])

    s0u, au = _fit_power(t_mid, s_up)
    s0d, ad = _fit_power(t_mid, s_dn)
    return s0u, au, s0d, ad


def _resqr_price_at(model, q, t_arr, log_median):
    """Evaluate price from stored resqr coefs given a precomputed log10 median.

    Returns a scalar when ``t_arr`` is scalar, else an ndarray shaped like
    ``t_arr``. Returns ``None`` when the model carries no ``_resqr`` bundle —
    callers then fall back to the constant-σ path.
    """
    rq = getattr(model, "_resqr", None)
    if rq is None:
        return None
    try:
        from tools.model_toolkit.resqr_bands import eval_resqr_offsets
    except Exception:
        return None
    t_arr_np = np.asarray(t_arr, dtype=np.float64)
    is_scalar = t_arr_np.ndim == 0
    t_1d = np.atleast_1d(t_arr_np)
    knots = tuple(rq.get("knots", (3.0, 6.0, 9.0, 12.0)))
    offsets = eval_resqr_offsets(t_1d, rq["sorted_qs"], rq["coef_matrix"],
                                  knots=knots)
    qs = np.asarray(rq["sorted_qs"], dtype=np.float64)
    q_f = float(q)
    hits = np.where(qs == q_f)[0]
    if hits.size:
        off = offsets[:, int(hits[0])]
    elif q_f <= qs[0]:
        off = offsets[:, 0]
    elif q_f >= qs[-1]:
        off = offsets[:, -1]
    else:
        lo = int(np.searchsorted(qs, q_f, side="right") - 1)
        hi = lo + 1
        frac = (q_f - qs[lo]) / (qs[hi] - qs[lo])
        off = offsets[:, lo] + frac * (offsets[:, hi] - offsets[:, lo])
    log_median_1d = np.asarray(log_median, dtype=np.float64).reshape(-1)
    log_p = log_median_1d + off
    price = 10.0 ** log_p
    if is_scalar:
        return float(price[0])
    return price.reshape(t_arr_np.shape)



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


def _compute_log_r2(actual_prices, predicted_prices):
    """R² in log10 space. Returns float or None if degenerate."""
    log_a = np.log10(np.maximum(np.asarray(actual_prices, float), 1e-10))
    log_p = np.log10(np.maximum(np.asarray(predicted_prices, float), 1e-10))
    ss_res = np.sum((log_a - log_p) ** 2)
    ss_tot = np.sum((log_a - np.mean(log_a)) ** 2)
    if ss_tot == 0:
        return None
    return float(1.0 - ss_res / ss_tot)


def compute_model_r2(mdl, price_years, price_prices, attr="r2_per_quantile"):
    """Compute per-quantile R² for any model with price_at() and quantiles.

    Stores the result dict on ``mdl.<attr>`` (default ``r2_per_quantile``). Pass a
    different ``attr`` to compute a SECOND measure over a different data window
    without clobbering the first — e.g. the Time Machine "future"/all-data R²
    scores an as-of model against the FULL realized series
    (``attr="r2_all_per_quantile"``) alongside its in-sample ``r2_per_quantile``.
    """
    r2d = {}
    mask = price_years >= T_MIN
    t = price_years[mask]
    actual = price_prices[mask]
    if hasattr(mdl, 'quantiles') and mdl.quantiles:
        for q in mdl.quantiles:
            try:
                predicted = np.asarray(mdl.price_at(q, t), float)
                r2 = _compute_log_r2(actual, predicted)
                if r2 is not None:
                    r2d[q] = r2
            except Exception:
                pass
    elif hasattr(mdl, 'price_at'):
        try:
            predicted = np.asarray(mdl.price_at(0.5, t), float)
            r2 = _compute_log_r2(actual, predicted)
            if r2 is not None:
                r2d[0.5] = r2
        except Exception:
            pass
    setattr(mdl, attr, r2d)


# ── PriceModel protocol + implementations ────────────────────────────────────

from typing import Protocol, runtime_checkable


