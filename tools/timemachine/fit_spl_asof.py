#!/usr/bin/env python3
"""spl (Saturating Power Law) as-of fitter for the "Time Machine" feature.

Fits the 3 median params (``log10_L``, ``t0``, ``beta``) using ONLY data
through an as-of horizon (``years <= ymax``), reproducing the LIVE fitter
EXACTLY: ``tools/analyze_spl.py::fit_spl`` (``differential_evolution`` over
seeds 0, 1, 2 in the ``(A, beta, log10_t0)`` fit-coordinate system, with the
derived-``L`` range enforced as a penalty inside the objective) followed by
``tools/fit_spl.py``'s ``curve_fit`` polish and the ``(A, beta, log10_t0) ->
(log10_L, t0, beta)`` conversion (``derived``). Two optimizers over one model
drift apart silently, so this imports the live fitter's own routines rather
than re-implementing the objective, bounds, or polish/derived logic.

sigma is computed as ``std(residuals)`` over the SAME as-of window the fit
ran on -- matching what the runtime uses (``SaturatingPowerLawModel``
subclasses ``_ShrinkingBandsMixin``, whose ``_sigma_at`` returns the constant
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

from analyze_spl import CAP_FIT_USD, SUPPLY, T_MIN, fit_spl, spl_log10  # noqa: E402
from fit_spl import derived, polish  # noqa: E402
from tools.timemachine.fit_bm_asof import _truncate  # noqa: E402


def fit_spl_asof(prices, ymax):
    """Fit the Saturating Power Law's 3 median params on data through an
    as-of horizon.

    Parameters
    ----------
    prices : PriceData
        Full ``PriceData`` from ``load_prices`` (untruncated).
    ymax : float
        As-of horizon in years (same unit as ``prices.df["years"]``).

    Returns
    -------
    dict
        ``{"params": {"log10_L": f, "t0": f, "beta": f}, "sigma": f, "r2": f}``
    """
    trunc = _truncate(prices, ymax)
    t = trunc.df_full["years"].values
    lp = trunc.df_full["log_price"].values
    mask = t >= T_MIN
    t_fit = t[mask]
    lp_fit = lp[mask]

    # Same admissible-L range tools/fit_spl.py::main() computes, needed by
    # `polish` (fit_spl's own DE objective re-derives its own copy from
    # whatever (t, lp) it's given, so the DE call itself doesn't need this).
    lo_L = float(np.log10(np.max(10.0 ** lp_fit)))
    hi_L = float(np.log10(CAP_FIT_USD / SUPPLY))

    # differential_evolution over seeds 0, 1, 2 -- exactly tools/fit_spl.py's
    # main(): a global optimum that moves with the seed is not a global
    # optimum, so take the best of three.
    runs = [fit_spl(t_fit, lp_fit, seed=seed) for seed in (0, 1, 2)]
    res = min(runs, key=lambda r: r.fun)

    theta, _note = polish(t_fit, lp_fit, np.asarray(res.x, float), lo_L, hi_L)
    log10_L, t0 = derived(theta)
    beta = float(theta[1])

    pred = spl_log10(t_fit, *theta)
    resid = lp_fit - pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((lp_fit - lp_fit.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    sigma = float(np.std(resid))  # constant band sigma -- matches _sigma_at

    return {
        "params": {
            "log10_L": float(log10_L),
            "t0": float(t0),
            "beta": beta,
        },
        "sigma": sigma,
        "r2": r2,
    }
