#!/usr/bin/env python3
"""Task 2 (EPPL family as-of fitter) for the "Time Machine" feature.

Cold-fits ONE EPPL config on data through an as-of horizon (``t <= ymax``)
via differential evolution, mirroring ``tools/fit_all_eppl_configs.py``'s
per-config fit loop but parameterized over an arbitrary right edge instead
of always using the full dataset. Consumed by the Time Machine grid builder
(later task), which calls this once per (config, frame-date) pair.

Bands are CONSTANT sigma, not shrinking: ``sigma = std(residuals)`` is
exactly what the runtime uses (`_ShrinkingBandsMixin._sigma_at`,
btc_core/_base.py:38-40, returns the constant `self._sigma` set at
btc_core/_base.py:32). The shrinking sigma0/alpha fit is dead code for the
constant band path and is intentionally NOT computed here.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, "tools"), ROOT]

from scipy.optimize import differential_evolution  # noqa: E402
from fit_all_eppl_configs import build_model_fn  # noqa: E402


def fit_config_asof(cfg, t, lp, ymax, maxiter):
    """Cold-fit one EPPL config on data through an as-of horizon.

    Parameters
    ----------
    cfg : tuple
        ``(n_log, n_cal, log_damps, cal_damps)`` as consumed by
        ``fit_all_eppl_configs.build_model_fn``.
    t, lp : np.ndarray
        Full time-since-origin (years) and log10(price) arrays.
    ymax : float
        As-of horizon (years since origin). Fit uses ``1 <= t <= ymax``.
    maxiter : int
        ``differential_evolution`` maxiter (caller-controlled — the real
        grid build passes the higher Task 0 ``EPPL_MAXITER``).

    Returns
    -------
    dict
        ``{"params": {name: float}, "sigma": float, "r2": float,
        "n_log": int, "n_cal": int, "log_damps": [...], "cal_damps": [...]}``
    """
    n_log, n_cal, log_damps, cal_damps = cfg

    t = np.asarray(t)
    lp = np.asarray(lp)
    mask = (t >= 1.0) & (t <= ymax)
    t_fit = t[mask]
    lp_fit = lp[mask]

    model_fn, param_names, bounds = build_model_fn(n_log, n_cal, log_damps, cal_damps)

    def obj(params):
        return np.sum((lp_fit - model_fn(t_fit, *params)) ** 2)

    res = differential_evolution(
        obj, bounds, maxiter=maxiter, tol=1e-8, seed=42, workers=1,
    )

    pred = model_fn(t_fit, *res.x)
    resid = lp_fit - pred
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((lp_fit - np.mean(lp_fit)) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    sigma = float(np.std(resid))  # constant band sigma — matches _sigma_at

    return {
        "params": {pn: float(pv) for pn, pv in zip(param_names, res.x)},
        "sigma": sigma,
        "r2": float(r2),
        "n_log": n_log,
        "n_cal": n_cal,
        "log_damps": list(log_damps),
        "cal_damps": list(cal_damps),
    }
