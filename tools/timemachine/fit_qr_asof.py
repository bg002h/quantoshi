#!/usr/bin/env python3
"""QR (quantile regression) as-of builder for the "Time Machine" feature.

Fits quantile-regression channels — ``log10(price) = intercept + slope·log10(t)``
per quantile — using ONLY data through an as-of horizon (``years <= ymax``),
matching the live ``qr`` model's fit exactly (``model_toolkit.bands.fit_qr_channels``
with its default ``BM_QUANTILES``, the same 27-quantile set as ``M.QR_QUANTILES``).

Consumed by ``tools/build_timemachine_grid.py`` once per as-of frame date. Output
is **params-only** (a handful of floats per quantile) — unlike BM there are no
composite arrays, so nothing needs downsampling.

Truncation reuses ``fit_bm_asof._truncate`` so QR and BM see the IDENTICAL as-of
data window for a given frame (a divergence there would silently misalign the two
models' as-of views).
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, "tools"), ROOT]

from model_toolkit.bands import fit_qr_channels  # noqa: E402
from tools.timemachine.fit_bm_asof import _truncate  # noqa: E402


def fit_qr_asof(prices, ymax):
    """Fit QR channels on data through an as-of horizon.

    Parameters
    ----------
    prices : PriceData
        Full ``PriceData`` from ``load_prices`` (untruncated).
    ymax : float
        As-of horizon in years (same unit as ``prices.df["years"]``).

    Returns
    -------
    dict
        ``{"fits": {"<q>": {"intercept": f, "slope": f, "r2": f}}}`` — string
        quantile keys (JSON-safe), matching the stored live-``qr`` fit format
        (``model_toolkit.export`` / ``ModelData.qr_fits``).
    """
    trunc = _truncate(prices, ymax)
    qr = fit_qr_channels(trunc)  # default BM_QUANTILES (27) — matches live qr
    return {"fits": {str(q): dict(f) for q, f in qr.fits.items()}}
