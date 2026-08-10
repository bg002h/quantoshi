#!/usr/bin/env python3
"""Task 3 (BM as-of builder) for the "Time Machine" feature.

Builds the Bubble Model (support + fitted/predicted bubbles + composite +
asymmetric shrinking sigma) using ONLY data through an as-of horizon
(``years <= ymax``). Mirrors the pipeline in ``tools/build_bm_model.py``
lines 264-294 exactly, but ``model_toolkit.data.load_prices`` has no cutoff
parameter (tools/model_toolkit/data.py:20), so this module truncates the
already-loaded ``PriceData`` itself via ``_truncate()`` before running the
same sequence of toolkit calls.

Consumed by the Time Machine grid builder (later task), which calls this
once per as-of frame date for the BM flagship model.

BM legitimately uses the 4 ASYMMETRIC shrinking sigma params (sigma0_up/
alpha_up/sigma0_down/alpha_down) -- it is a ``_CompositeModel`` whose
``_sigma_at`` = sigma0 * t^(-alpha) (btc_core/_base.py:198-202) -- unlike
EPPL's constant sigma (tools/timemachine/fit_eppl_asof.py), so all four are
computed and returned here.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, "tools"), ROOT]

from model_toolkit.data import PriceData  # noqa: E402
from model_toolkit.support import fit_support  # noqa: E402
from model_toolkit.fitting import fit_sequential, classify  # noqa: E402
from model_toolkit.prediction import predict_future  # noqa: E402
from model_toolkit.composite import build_composite, build_comp_by_n  # noqa: E402
from model_toolkit.bands import fit_asymmetric_sigma  # noqa: E402


def _truncate(prices, ymax):
    """Truncate a loaded ``PriceData`` to rows with ``years <= ymax``.

    ``load_prices`` has no cutoff parameter, so this rebuilds the same
    filtered/derived fields ``load_prices`` constructs
    (tools/model_toolkit/data.py:103-108) from a row-masked copy of
    ``prices.df`` / ``prices.df_full`` instead of re-reading the CSV.

    Parameters
    ----------
    prices : PriceData
        Full ``PriceData`` from ``load_prices``.
    ymax : float
        As-of horizon (years/blocks since origin, same unit as
        ``prices.df["years"]``).

    Returns
    -------
    PriceData
        Truncated copy with all derived arrays rebuilt from the masked df.
    """
    df = prices.df[prices.df["years"] <= ymax].reset_index(drop=True)
    df_full = prices.df_full[prices.df_full["years"] <= ymax].reset_index(drop=True)

    return PriceData(
        df=df, df_full=df_full,
        years=df["years"].values, log_years=df["log_years"].values,
        prices=df["price"].values, log_prices=df["log_price"].values,
        dates=df["date"].dt.strftime("%Y-%m-%d").tolist(),
    )


def fit_bm_asof(prices, ymax):
    """Build the Bubble Model on data through an as-of horizon.

    Runs the exact ``tools/build_bm_model.py:264-294`` sequence
    (fit_support -> fit_sequential -> classify -> predict_future ->
    build_composite -> build_comp_by_n -> fit_asymmetric_sigma) against a
    truncated copy of ``prices``.

    Parameters
    ----------
    prices : PriceData
        Full ``PriceData`` from ``load_prices`` (untruncated).
    ymax : float
        As-of horizon in years (same unit as ``prices.df["years"]``).

    Returns
    -------
    dict
        ``{"comp_by_n": [[...], ...], "t_grid": [...],
        "support_slope": f, "support_intercept": f,
        "sigma0_up": f, "alpha_up": f, "sigma0_down": f, "alpha_down": f,
        "bm_r2": f}``
    """
    trunc = _truncate(prices, ymax)

    sup = fit_support(trunc)
    fitted = fit_sequential(trunc, sup)
    major, minor = classify(fitted, n_major=5)
    f_maj, f_min = predict_future(major, minor, t_last_data=trunc.years[-1],
                                   n_major=3, n_minor=1)
    all_future = sorted(f_maj + f_min, key=lambda b: b["t_rise"])

    comp = build_composite(sup, fitted, trunc)
    cbn = build_comp_by_n(sup, fitted, all_future, comp.t_grid,
                           comp.hist_K_max, comp.total_plot)

    full_log_prices = np.log10(np.maximum(trunc.df_full["price"].values, 1e-10))
    full_years = trunc.df_full["years"].values
    sigma = fit_asymmetric_sigma(full_log_prices, np.array(cbn[-1]),
                                  comp.t_grid, full_years)

    return {
        "comp_by_n": cbn,
        "t_grid": comp.t_grid.tolist(),
        "support_slope": sup.slope,
        "support_intercept": sup.intercept,
        "sigma0_up": sigma.sigma0_up,
        "alpha_up": sigma.alpha_up,
        "sigma0_down": sigma.sigma0_down,
        "alpha_down": sigma.alpha_down,
        "bm_r2": comp.r2,
    }
