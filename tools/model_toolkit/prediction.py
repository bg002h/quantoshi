"""Extrapolate or average fitted bubble parameters to predict future bubbles."""
from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_future(
    fitted_major,
    fitted_minor,
    t_last_data,
    *,
    n_major=3,
    n_minor=1,
    mode="extrap",
    last_n_major=3,
    last_n_minor=4,
    major_weights=None,
    minor_weights=None,
    major_interval_trend=True,
    minor_interval_trend=True,
    major_default_interval=3.8,
    minor_default_interval=3.8,
    major_min_interval=1.4,
    minor_min_interval=0.15,
    plat_pow_range=8.0,
):
    """Predict future bubbles for both major and minor classes.

    Parameters
    ----------
    fitted_major, fitted_minor : list[dict]
        Historical fitted bubble dicts (output of ``classify``).
    t_last_data : float
        Last time value in the price data (years since origin).  Ensures the
        first predicted bubble starts *after* observed data.
    n_major, n_minor : int
        Number of future bubbles to generate for each class.
    mode : ``"extrap"`` | ``"avg"``
        ``"extrap"`` — linear trend extrapolation per parameter (r, d, K
        log-linearly; dur_plateau, plat_pow linearly).
        ``"avg"`` — weighted average of last N historical bubbles, replicated
        with mean spacing.
    last_n_major, last_n_minor : int | None
        Only use the most recent N historical bubbles.  None = use all.
    major_weights, minor_weights : list[float] | None
        Per-bubble, per-parameter importance weights (flat list of length
        N_bubbles * 6).  None = uniform.
    major_interval_trend, minor_interval_trend : bool
        Whether to extrapolate the interval trend (True) or use weighted
        average (False).
    major_default_interval, minor_default_interval : float
        Fallback interval (years) when < 2 historical intervals available.
    major_min_interval, minor_min_interval : float
        Floor on predicted inter-bubble interval.
    plat_pow_range : float
        Clamp range for plateau power parameter.

    Returns
    -------
    (future_major, future_minor) : tuple[list[dict], list[dict]]
    """
    future_major = (
        _predict(
            fitted_major, n_major, mode,
            extrap_weights=major_weights,
            use_interval_trend=major_interval_trend,
            default_interval=major_default_interval,
            min_interval=major_min_interval,
            last_n=last_n_major,
            t_last_data=t_last_data,
            plat_pow_range=plat_pow_range,
        )
        if n_major > 0
        else []
    )
    future_minor = (
        _predict(
            fitted_minor, n_minor, mode,
            extrap_weights=minor_weights,
            use_interval_trend=minor_interval_trend,
            default_interval=minor_default_interval,
            min_interval=minor_min_interval,
            last_n=last_n_minor,
            t_last_data=t_last_data,
            plat_pow_range=plat_pow_range,
        )
        if n_minor > 0
        else []
    )
    return future_major, future_minor


# ---------------------------------------------------------------------------
# Core prediction logic — faithfully ported from Cell 0 lines ~800-954
# ---------------------------------------------------------------------------

def _predict(
    hist_params, n_future, mode,
    *,
    extrap_weights=None,
    use_interval_trend=True,
    default_interval=3.8,
    min_interval=0.15,
    last_n=None,
    t_last_data=None,
    plat_pow_range=8.0,
):
    """Project future bubble parameters from historical bubbles.

    mode='avg'
        Every predicted bubble gets the weighted average of the last ``last_n``
        historical bubbles.  All predictions are identical in shape; only the
        start time advances by the average interval.  Conservative / stable.

    mode='extrap'
        Parameters are linearly extrapolated along their historical trend
        (r and d log-linearly; K, dur_plateau, plat_pow linearly), so each
        successive prediction follows the fitted slope.

    Weights layout per bubble (chronological):
        [r_rise, d_decay, K, interval, dur_plateau, plat_pow]
    """
    future = []
    if last_n is not None:
        hist_params = hist_params[-last_n:]
    n_hist = len(hist_params)
    if n_hist < 1 or n_future < 1:
        return future

    # ── Weights ──────────────────────────────────────────────────────────
    n_params = 6
    expected_w = n_hist * n_params
    use_w = extrap_weights is not None and len(extrap_weights) == expected_w
    w = (
        np.array(extrap_weights, dtype=float).reshape(n_hist, n_params)
        if use_w
        else np.ones((n_hist, n_params))
    )

    # ── Historical arrays ────────────────────────────────────────────────
    starts = [b["t_rise"] for b in hist_params]
    intervals_arr = (
        np.array([starts[i + 1] - starts[i] for i in range(len(starts) - 1)])
        if n_hist >= 2
        else np.array([default_interval])
    )
    intv_weights = w[: len(intervals_arr), 3]

    vals_r = np.array([b["r"] for b in hist_params])
    vals_d = np.array([b["d"] for b in hist_params])
    vals_K = np.array([b["K"] for b in hist_params])
    vals_dp = np.array([b["dur_plateau"] for b in hist_params])
    vals_pp = np.array([b.get("plat_pow", 0.0) for b in hist_params])

    # ── Helpers ──────────────────────────────────────────────────────────
    def wextrap(vals, wi, target):
        x = np.arange(len(vals), dtype=float)
        c = (
            np.polyfit(x, vals, 1, w=wi)
            if len(vals) >= 2
            else [0.0, float(vals[0])]
        )
        return float(np.polyval(c, target)), c

    def wavg(vals, wi):
        return float(np.average(vals, weights=wi))

    # ── Pre-compute avg-mode fixed parameters ────────────────────────────
    if mode == "avg":
        fixed_r = np.exp(wavg(np.log(np.maximum(vals_r, 1e-9)), w[:, 0]))
        fixed_d = np.exp(wavg(np.log(np.maximum(vals_d, 1e-9)), w[:, 1]))
        fixed_K = np.exp(wavg(np.log(np.maximum(vals_K, 1e-9)), w[:, 2]))
        fixed_dp = wavg(vals_dp, w[:, 4])
        fixed_pp = float(
            np.clip(wavg(vals_pp, w[:, 5]), -plat_pow_range, plat_pow_range)
        )

    # ── Generate predictions ─────────────────────────────────────────────
    last_start = starts[-1]
    for j in range(n_future):
        tgt = n_hist + j

        # -- Parameters --
        if mode == "avg":
            pred_r, pred_d, pred_K, pred_dp, pred_pp = (
                fixed_r, fixed_d, fixed_K, fixed_dp, fixed_pp,
            )
        else:
            _lr, _ = wextrap(np.log(np.maximum(vals_r, 1e-9)), w[:, 0], tgt)
            pred_r = np.exp(_lr)
            _ld, _ = wextrap(np.log(np.maximum(vals_d, 1e-9)), w[:, 1], tgt)
            pred_d = np.exp(_ld)
            _lk, _ = wextrap(np.log(np.maximum(vals_K, 1e-9)), w[:, 2], tgt)
            pred_K = np.exp(_lk)
            pred_dp, _ = wextrap(vals_dp, w[:, 4], tgt)
            pred_pp, _ = wextrap(vals_pp, w[:, 5], tgt)
            pred_pp = float(np.clip(pred_pp, -plat_pow_range, plat_pow_range))

        # -- Interval --
        if use_interval_trend and len(intervals_arr) >= 2:
            pred_intv, _ = wextrap(
                intervals_arr, intv_weights, len(intervals_arr) + j
            )
        else:
            pred_intv = float(np.average(intervals_arr, weights=intv_weights))

        # -- Clamp --
        pred_K = max(pred_K, 0.01)
        pred_dp = max(pred_dp, 0.0)
        pred_intv = max(pred_intv, min_interval)

        # -- Timing --
        pred_tr = (last_start if j == 0 else future[-1]["t_rise"]) + pred_intv
        if j == 0 and t_last_data is not None:
            while pred_tr <= t_last_data:
                pred_tr += pred_intv

        pred_dr = max(pred_K / pred_r, 0.02)
        pred_tpl = pred_tr + pred_dr
        pred_tdc = pred_tpl + pred_dp
        pred_K_end = (
            pred_K + pred_pp * np.log10(max(pred_tdc / pred_tpl, 1e-12))
            if pred_dp > 0
            else pred_K
        )
        peak_K_disp = max(pred_K, pred_K_end)

        fb = {
            "t_rise": pred_tr,
            "t_start": pred_tr,
            "r": pred_r,
            "t_plateau": pred_tpl,
            "t_decay": pred_tdc,
            "d": pred_d,
            "K": peak_K_disp,
            "K_peak": pred_K,
            "K_end": pred_K_end,
            "plat_pow": pred_pp,
            "dur_rise": pred_dr,
            "dur_plateau": pred_dp,
            "interval": pred_intv,
        }
        future.append(fb)

    return future
