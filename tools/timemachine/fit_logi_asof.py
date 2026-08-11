#!/usr/bin/env python3
"""logi (Logistic S-curve) as-of fitter for the "Time Machine" feature.

Fits the 3 median params (``K``, ``r``, ``t0``) using ONLY data through an
as-of horizon (``years <= ymax``), replicating ``tools/fit_logistic.py::
main()``'s fit EXACTLY -- its bounds, ``differential_evolution`` settings,
and ``curve_fit`` polish (kept unconditionally on success, exactly as
``main()`` does; there is no "improves the SSE" guard in the live tool to
mirror). Unlike spl, ``fit_logistic.py`` has no reusable fit function to
import, so the objective/bounds/DE settings are duplicated here verbatim
rather than re-derived -- see that file for the rationale behind each bound.

sigma is computed as ``std(residuals)`` over the SAME as-of window the fit
ran on -- matching what the runtime uses (``LogisticSCurveModel`` subclasses
``_ShrinkingBandsMixin``, whose ``_sigma_at`` returns the constant
``self._sigma``; see ``tools/timemachine/fit_eppl_asof.py``'s docstring for
the same point re: EPPL). The shrinking sigma0/alpha fit is dead code for the
constant-band path and is intentionally NOT computed here.

Truncation reuses ``fit_bm_asof._truncate`` so every Time Machine model sees
the IDENTICAL as-of data window for a given frame.

Consumed by ``tools/build_timemachine_grid.py`` (``add_series_to_grid`` /
``build_grid``), once per as-of frame date.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, "tools"), ROOT]

from scipy.optimize import curve_fit, differential_evolution  # noqa: E402

from tools.timemachine.fit_bm_asof import _truncate  # noqa: E402

T_MIN = 1.0  # matches btc_web/time_basis.py (calendar basis)

# Exactly tools/fit_logistic.py's bounds:
#   K  in [3, 15]    -- log10 saturation price
#   r  in [0.05, 2]  -- growth rate
#   t0 in [1, 30]    -- inflection year
_BOUNDS_LO = [3.0, 0.05, 1.0]
_BOUNDS_HI = [15.0, 2.0, 30.0]


def logistic_log10(t, K, r, t0):
    """Symmetric logistic: log10(price) = K / (1 + exp(-r * (t - t0)))."""
    return K / (1.0 + np.exp(-r * (t - t0)))


def fit_logi_asof(prices, ymax):
    """Fit the Logistic S-curve's 3 median params on data through an as-of
    horizon.

    Parameters
    ----------
    prices : PriceData
        Full ``PriceData`` from ``load_prices`` (untruncated).
    ymax : float
        As-of horizon in years (same unit as ``prices.df["years"]``).

    Returns
    -------
    dict
        ``{"params": {"K": f, "r": f, "t0": f}, "sigma": f, "r2": f}``
    """
    trunc = _truncate(prices, ymax)
    t = trunc.df_full["years"].values
    lp = trunc.df_full["log_price"].values
    mask = t >= T_MIN
    t_fit = t[mask]
    lp_fit = lp[mask]

    bounds = list(zip(_BOUNDS_LO, _BOUNDS_HI))

    def objective(params):
        pred = logistic_log10(t_fit, *params)
        return np.sum((lp_fit - pred) ** 2)

    res = differential_evolution(objective, bounds, maxiter=5000, seed=42,
                                  tol=1e-14, polish=True, popsize=30,
                                  workers=1)
    try:
        popt, _ = curve_fit(logistic_log10, t_fit, lp_fit, p0=res.x,
                             bounds=(_BOUNDS_LO, _BOUNDS_HI), maxfev=20000)
    except Exception:
        popt = res.x

    K, r, t0 = (float(v) for v in popt)
    pred = logistic_log10(t_fit, K, r, t0)
    resid = lp_fit - pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((lp_fit - lp_fit.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    sigma = float(np.std(resid))  # constant band sigma -- matches _sigma_at

    return {
        "params": {"K": K, "r": r, "t0": t0},
        "sigma": sigma,
        "r2": r2,
    }
