"""Task 5 (runtime grid loader) for the "Time Machine" feature.

Loads the gzip-compressed JSON grid produced by
``tools/build_timemachine_grid.py`` and exposes it as per-request model
objects for the bubble chart's as-of-date view (Task 7).

Grid shape (see ``tools/build_timemachine_grid.py`` module docstring)::

    {"frames": ["YYYY-MM-DD", ...],
     "models": {"bub": [bm_frame_dict | None, ...],
                "ecfg_1d_1u": [ecfg_frame_dict | None, ...], ...}}

A ``null`` entry in ``models[key]`` means that frame's fit FAILED at build
time (logged, not dropped -- see ``build_grid``'s ``failed`` list). Callers
here surface that as ``None`` and never crash; the bubble path (Task 7) is
responsible for skipping a ``None`` frame.

**Hard rails** (mirrors the ``u1`` / User Model escape hatch already used
for click-to-draw models):
  - No matplotlib / kaleido imports -- this module is on the request path.
  - Loaded with ``gzip`` + ``json`` only -- never pickle (untrusted-ish,
    build-time-only artifact; JSON keeps the format inspectable and the
    loader dependency-free).
  - Never mutates ``_app_ctx.PRICE_MODELS`` -- every call builds a FRESH
    per-request object from the loaded grid data. The shared registry of
    "live" models is not touched.
"""
import functools
import gzip
import json
import types
from pathlib import Path

import numpy as np

GRID_PATH = Path(__file__).resolve().parent.parent / "timemachine_grid.json.gz"


def available():
    """True iff the grid file exists on disk (mode disabled otherwise)."""
    return GRID_PATH.exists()


@functools.lru_cache(maxsize=1)
def _load():
    """Load + cache the grid once per worker process.

    ``lru_cache`` is process-local (each gunicorn worker loads its own
    copy), which is fine -- the grid is read-only for the lifetime of the
    worker and re-loading per request would be wasteful.
    """
    with gzip.open(GRID_PATH, "rt") as f:
        return json.load(f)


def frames():
    """As-of frame dates, e.g. ``["2012-01-24", "2012-04-01", ...]``."""
    return _load()["frames"]


def n_frames():
    """Number of as-of frames in the grid."""
    return len(frames())


def _quantiles():
    """QR quantile list -- from the live model if loaded, else a fresh load.

    ``_app_ctx.M`` is only populated once ``app.py``'s registration block
    has run; tests / pre-startup callers fall back to loading model_data.pkl
    directly so this module can be exercised standalone.
    """
    import _app_ctx
    if _app_ctx.M is not None:
        return _app_ctx.M.QR_QUANTILES
    from btc_core import load_model_data
    return load_model_data().QR_QUANTILES


def asof_eppl(config_key, frame_idx):
    """Build a fresh ``EPPLConfigModel`` as-of a grid frame.

    Returns ``None`` if the frame is missing (failed fit at build time,
    stored as JSON ``null``) rather than raising -- callers (Task 7's
    bubble path) must skip a ``None`` frame.

    A NEW object is constructed on every call -- this never reads from or
    writes to ``_app_ctx.PRICE_MODELS``, so the shared/global EPPL config
    models are unaffected regardless of how the returned object is used
    afterward.
    """
    frame = _load()["models"][config_key][frame_idx]
    if frame is None:
        return None
    from btc_core._hybppl_eppl import EPPLConfigModel
    return EPPLConfigModel(
        config_key,
        np.array([1.0, 2.0]), np.array([1.0, 2.0]),  # unused: sigma_override
        _quantiles(),                                 # skips residual-band fit
        cfg_override=frame, sigma_override=frame["sigma"],
    )


def asof_bm(frame_idx):
    """Build an ``m``-shaped shim exposing the Bubble Model as-of a frame.

    Returns ``None`` for a missing/failed frame. The returned
    ``types.SimpleNamespace`` carries only the frame-specific fields the
    bubble BM-primary path needs (Task 7); ``genesis`` / price arrays /
    OLS fields still come from the real (current) ``m``.
    """
    frame = _load()["models"]["bub"][frame_idx]
    if frame is None:
        return None
    t_grid = np.asarray(frame["t_grid"])
    return types.SimpleNamespace(
        years_plot_bm=t_grid,
        comp_by_n=frame["comp_by_n"],
        bm_r2=frame["bm_r2"],
        support_bm=10.0 ** (
            frame["support_intercept"]
            + frame["support_slope"] * np.log10(np.maximum(t_grid, 1e-10))
        ),
        bm_sigma0_up=frame["sigma0_up"],
        bm_alpha_up=frame["alpha_up"],
        bm_sigma0_down=frame["sigma0_down"],
        bm_alpha_down=frame["alpha_down"],
    )
