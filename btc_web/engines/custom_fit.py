"""Custom Time Axis — fit engine.

Pure functions on cached numpy arrays. No Dash dependencies.
See docs/superpowers/specs/2026-04-13-custom-time-axis-design.md section 3.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import scipy.stats

# Ensure the repo root is on sys.path so `tools.model_toolkit.support` imports
# work when btc_web is launched from its own subdirectory (gunicorn does this).
# Idempotent; runs once at module import under --preload.
import sys as _sys  # noqa: E402
_REPO_ROOT_STR = str(Path(__file__).resolve().parent.parent.parent)
if _REPO_ROOT_STR not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT_STR)

from tools.model_toolkit.support import fit_support  # noqa: E402

_LOG = logging.getLogger("custom_fit")

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PRICE_CSV = _REPO_ROOT / "BitcoinPricesDaily.csv"
_BLOCK_CSV = _REPO_ROOT / "BitcoinBlocksDaily.csv"


# ──────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FitInput:
    t: np.ndarray            # user-chosen scale (years or raw block-units)
    price: np.ndarray        # aligned with t
    weighting: str           # "none" | "inv_t" | "inv_sqrt_t" | "log_density"


@dataclass
class FitResult:
    name: str
    params: dict
    t_plot: np.ndarray
    y_plot: Union[np.ndarray, dict]     # array for PL/Exp/BM-floor, dict for QR
    n_samples: int
    r2: float
    elapsed_ms: float
    note: Optional[str] = None


# ──────────────────────────────────────────────────────────────────────────
# Cached price + block arrays (loaded once per worker at import time)
# ──────────────────────────────────────────────────────────────────────────

_DATES: Optional[pd.DatetimeIndex] = None
_PRICES: Optional[np.ndarray] = None
_BLOCKS: Optional[np.ndarray] = None
_BLOCK_CAP: Optional[int] = None
_BLOCK_MAP_LOADED: bool = False


def _load_price_arrays_once() -> None:
    """Load dates + prices directly from BitcoinPricesDaily.csv.
    Cached for the lifetime of the worker."""
    global _DATES, _PRICES
    if _DATES is not None:
        return
    if not _PRICE_CSV.exists():
        raise RuntimeError(
            f"custom_fit: {_PRICE_CSV} missing. Run update_prices.py first.")
    df = pd.read_csv(_PRICE_CSV)
    # DatetimeIndex (not Series) so (_DATES - t0).days returns a TimedeltaIndex
    _DATES = pd.DatetimeIndex(pd.to_datetime(df["Date"], format="%m/%d/%y"))
    _PRICES = df["Price"].to_numpy(dtype=np.float64)
    # Defense-in-depth: cached arrays are read across every fit request, so
    # make them immutable to catch accidental in-place mutation.
    _PRICES.setflags(write=False)
    _LOG.info("custom_fit: loaded %d price rows", len(_PRICES))


def _load_block_array_once() -> None:
    """Load block heights from BitcoinBlocksDaily.csv and compute the 2016 cap.

    Raises RuntimeError on alignment failure (misaligned = data corruption or
    stale block CSV). The module-level wrapper at the bottom of this file
    catches Exception and leaves `_BLOCKS = None`, so in practice this
    downgrades alignment errors to "block mode unavailable, calendar works".
    That is the safer prod default: a daily price-update row ahead of the
    block CSV shouldn't crash the whole app. Ops: if /health reports
    block_map_loaded=false, rerun `tools/build_block_map.py --append`.
    """
    global _BLOCKS, _BLOCK_CAP, _BLOCK_MAP_LOADED
    if _DATES is None:
        _load_price_arrays_once()
    if not _BLOCK_CSV.exists():
        _LOG.error(
            "custom_fit: block map missing at %s; block mode disabled",
            _BLOCK_CSV,
        )
        return
    df = pd.read_csv(_BLOCK_CSV, parse_dates=["date"])
    # Hard-fail on alignment failure — data corruption per spec §5 case P
    if len(df) != len(_DATES):
        raise RuntimeError(
            f"custom_fit: block/price row count mismatch: "
            f"{len(df)} vs {len(_DATES)} — data corruption, refusing to load."
        )
    if not (df["date"].values == _DATES.values).all():
        raise RuntimeError(
            "custom_fit: block/price dates not aligned row-for-row — "
            "data corruption, refusing to load."
        )
    _BLOCKS = df["blockheight"].to_numpy(dtype=np.int64)
    _BLOCKS.setflags(write=False)  # read-only; prevents accidental mutation
    # Compute _BLOCK_CAP from the last row on or before 2015-12-31
    cap_mask = df["date"] <= pd.Timestamp("2015-12-31")
    if cap_mask.any():
        _BLOCK_CAP = int(df.loc[cap_mask, "blockheight"].iloc[-1])
    _BLOCK_MAP_LOADED = True
    _LOG.info(
        "custom_fit: loaded %d block rows, cap=%s", len(df), _BLOCK_CAP)


# Eager-load at module import so _BLOCK_CAP is available when layout/custom_time.py
# reads it, and so /health reports block_map_loaded:true from the first request.
try:
    _load_price_arrays_once()
    _load_block_array_once()
except Exception as _exc:  # noqa: BLE001
    _LOG.error("custom_fit eager load failed: %s", _exc)


def build_fit_input(scale: str, t0, weighting: str) -> FitInput:
    """Build a FitInput for the given scale + t0 + weighting.

    Calendar: t = (dates - t0).days / 365.25
    Block:    t = blocks - t0_block
    """
    if _DATES is None:
        _load_price_arrays_once()
        _load_block_array_once()
    assert _DATES is not None and _PRICES is not None

    if scale == "calendar":
        t0_ts = pd.Timestamp(t0)
        t_raw = (_DATES - t0_ts).days.values.astype(np.float64) / 365.25
    elif scale == "block":
        if _BLOCKS is None:
            raise RuntimeError(
                "block map unavailable; cannot build block-mode FitInput")
        t_raw = (_BLOCKS - int(t0)).astype(np.float64)
    else:
        raise ValueError(f"unknown scale {scale!r}")
    return FitInput(t=t_raw, price=_PRICES, weighting=weighting)


# ──────────────────────────────────────────────────────────────────────────
# Weight computer
# ──────────────────────────────────────────────────────────────────────────

def _compute_weights(t_positive: np.ndarray,
                      mode: str) -> tuple[np.ndarray, bool]:
    """Return (weights, degraded_flag). Weights normalized to mean=1.0.
    `degraded=True` when KDE fell back to uniform due to small-n or failure."""
    n = len(t_positive)
    if mode == "none":
        return np.ones(n), False
    if n < 30:
        return np.ones(n), True
    try:
        if mode == "inv_t":
            w = 1.0 / t_positive
        elif mode == "inv_sqrt_t":
            w = 1.0 / np.sqrt(t_positive)
        elif mode == "log_density":
            log_t = np.log10(t_positive)
            kde = scipy.stats.gaussian_kde(log_t)  # Scott's rule default
            dens = kde(log_t)
            w = 1.0 / np.maximum(dens, 1e-9)
        else:
            # Unknown mode → uniform fallback with no degradation flag
            return np.ones(n), False
    except Exception as exc:
        _LOG.warning("custom_fit weight compute failed: %s", exc)
        return np.ones(n), True
    # Normalize to mean=1
    w = w * (n / w.sum())
    return w, False


# ──────────────────────────────────────────────────────────────────────────
# Fit functions
# ──────────────────────────────────────────────────────────────────────────

_T_PLOT_POINTS = 400
_QR_FULL = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
_QR_REDUCED = (0.25, 0.50, 0.75)


def fit_pl(fi: FitInput) -> Optional[FitResult]:
    """Power Law — closed-form OLS on log-log. Supports weighting."""
    t0 = time.perf_counter()
    mask = fi.t > 0
    t = fi.t[mask]
    price = fi.price[mask]
    n = len(t)
    if n < 3:
        return None
    log_t = np.log10(t)
    log_p = np.log10(price)
    weights, degraded = _compute_weights(t, fi.weighting)

    if fi.weighting == "none":
        res = scipy.stats.linregress(log_t, log_p)
        slope, intercept = float(res.slope), float(res.intercept)
        r2 = float(res.rvalue ** 2)
    else:
        # np.polyfit uses w as sqrt-of-weights per the loss Σ w[j]² · r[j]²
        slope, intercept = np.polyfit(
            log_t, log_p, 1, w=np.sqrt(weights))
        slope = float(slope)
        intercept = float(intercept)
        pred = slope * log_t + intercept
        wbar = (weights * log_p).sum() / weights.sum()
        ss_res = (weights * (log_p - pred) ** 2).sum()
        ss_tot = (weights * (log_p - wbar) ** 2).sum()
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    t_plot = np.logspace(
        np.log10(max(t.min(), 1e-6)),
        np.log10(t.max() * 1.1), _T_PLOT_POINTS,
    )
    y_plot = slope * np.log10(t_plot) + intercept
    note = "weighting degraded (n<30)" if degraded else None
    return FitResult(
        name="PL",
        params={"slope": slope, "intercept": intercept},
        t_plot=t_plot, y_plot=y_plot,
        n_samples=n, r2=r2,
        elapsed_ms=(time.perf_counter() - t0) * 1000,
        note=note,
    )


def fit_exp(fi: FitInput) -> Optional[FitResult]:
    """Exponential — closed-form OLS on log-linear. Uses ALL samples (no t>0 mask)."""
    t0 = time.perf_counter()
    n = len(fi.t)
    if n < 3:
        return None
    log_p = np.log10(fi.price)
    res = scipy.stats.linregress(fi.t, log_p)
    slope = float(res.slope)
    intercept = float(res.intercept)
    t_plot = np.linspace(fi.t.min(), fi.t.max() * 1.1, _T_PLOT_POINTS)
    y_plot = slope * t_plot + intercept
    return FitResult(
        name="Exp",
        params={"slope": slope, "intercept": intercept},
        t_plot=t_plot, y_plot=y_plot,
        n_samples=n, r2=float(res.rvalue ** 2),
        elapsed_ms=(time.perf_counter() - t0) * 1000,
    )


def fit_qr(fi: FitInput) -> Optional[FitResult]:
    """Quantile Regression — statsmodels QuantReg per-quantile.

    Weighting is applied via multinomial resampling because statsmodels.QuantReg
    does not accept sample weights. Approximate but unbiased in expectation.
    """
    import statsmodels.api as sm
    t0 = time.perf_counter()
    mask = fi.t > 0
    t = fi.t[mask]
    price = fi.price[mask]
    n = len(t)
    if n < 10:
        return None
    quantiles = _QR_FULL if n >= 30 else _QR_REDUCED

    log_t = np.log10(t)
    log_p = np.log10(price)

    if fi.weighting != "none":
        weights, degraded = _compute_weights(t, fi.weighting)
        rng = np.random.default_rng(0)
        probs = weights / weights.sum()
        idx = rng.choice(n, size=5 * n, replace=True, p=probs)
        log_t_fit = log_t[idx]
        log_p_fit = log_p[idx]
        resample_note = "QR weighted via resampling (approximate)"
    else:
        log_t_fit = log_t
        log_p_fit = log_p
        resample_note = None
        degraded = False

    X = sm.add_constant(log_t_fit)
    slopes: dict = {}
    intercepts: dict = {}
    try:
        for q in quantiles:
            mdl = sm.QuantReg(log_p_fit, X).fit(q=q, max_iter=10000)
            intercept, slope = mdl.params
            slopes[q] = float(slope)
            intercepts[q] = float(intercept)
    except Exception as exc:
        elapsed = (time.perf_counter() - t0) * 1000
        return FitResult(
            name="QR", params={}, t_plot=np.array([]), y_plot={},
            n_samples=n, r2=float("nan"), elapsed_ms=elapsed,
            note=f"Fit failed: {type(exc).__name__}",
        )

    t_plot = np.logspace(
        np.log10(max(t.min(), 1e-6)),
        np.log10(t.max() * 1.1), _T_PLOT_POINTS,
    )
    log_t_plot = np.log10(t_plot)
    y_plot: dict = {}
    for q in quantiles:
        y_plot[q] = slopes[q] * log_t_plot + intercepts[q]

    note_parts = [
        resample_note,
        "weighting degraded (n<30)" if degraded else None,
        "reduced quantiles (n<30)" if n < 30 else None,
    ]
    note = " | ".join(p for p in note_parts if p) or None

    return FitResult(
        name="QR",
        params={"slopes": slopes, "intercepts": intercepts},
        t_plot=t_plot, y_plot=y_plot,
        n_samples=n, r2=float("nan"),
        elapsed_ms=(time.perf_counter() - t0) * 1000,
        note=note,
    )


# ──────────────────────────────────────────────────────────────────────────
# BM-floor via fit_support() shim
# ──────────────────────────────────────────────────────────────────────────

class _PriceDataShim:
    """Duck-types the attributes `tools/model_toolkit/support.py::fit_support`
    reads from a real PriceData. In block mode the column is 'log_years' but
    holds log10(block_offset) — misleading name preserved to match upstream."""

    def __init__(self, t_positive: np.ndarray, price_positive: np.ndarray):
        # Filter points with t < 1/365.25 day (log-safe threshold)
        mask = (t_positive > (1.0 / 365.25)) & (price_positive > 0)
        t = t_positive[mask]
        p = price_positive[mask]
        self.log_years = np.log10(t)
        self.log_prices = np.log10(p)
        self.df_full = pd.DataFrame({
            "log_years": self.log_years,
            "log_price": self.log_prices,
        })


def fit_bm_floor(fi: FitInput) -> Optional[FitResult]:
    """BM-floor via tools/model_toolkit/support.py::fit_support on a shim.

    Does NOT reuse BubbleModel (which adds bubble composites + extrapolation).
    Uses only the support-line fitting math. `fit_support` is imported at
    module top (see sys.path setup above).
    """
    t0 = time.perf_counter()
    mask = fi.t > 0
    t = fi.t[mask]
    price = fi.price[mask]
    n = len(t)
    if n < 50:
        return None

    shim = _PriceDataShim(t, price)
    try:
        fit = fit_support(shim, percentile=0.20)
    except Exception as exc:
        return FitResult(
            name="BM floor", params={},
            t_plot=np.array([]), y_plot=np.array([]),
            n_samples=n, r2=float("nan"),
            elapsed_ms=(time.perf_counter() - t0) * 1000,
            note=f"Fit failed: {type(exc).__name__}",
        )

    slope = float(fit.slope)
    intercept = float(fit.intercept)
    # R² is not meaningful for a support line: fit_support fits a bottom-
    # percentile QR line that intentionally sits BELOW the data mean, so the
    # "R² vs full-data mean" formula can go deeply negative and mislead.
    # Report NaN; the UI should show "—" instead of a number.
    r2 = float("nan")

    t_plot = np.logspace(
        np.log10(max(t.min(), 1e-6)),
        np.log10(t.max() * 1.1), _T_PLOT_POINTS,
    )
    y_plot = slope * np.log10(t_plot) + intercept
    return FitResult(
        name="BM floor",
        params={"slope": slope, "intercept": intercept},
        t_plot=t_plot, y_plot=y_plot,
        n_samples=n, r2=r2,
        elapsed_ms=(time.perf_counter() - t0) * 1000,
    )
