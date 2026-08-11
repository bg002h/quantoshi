"""Task 5: runtime grid loader (btc_web/timemachine.py).

Verifies ``asof_eppl`` builds a FRESH per-request object on every call and
never mutates the shared ``_app_ctx.PRICE_MODELS`` registry -- the same
escape-hatch contract as the ``u1`` (User Model) click-to-draw path.

Skips (rather than fails) when the grid hasn't been built in this
environment (`timemachine_grid.json.gz` missing) -- the loader itself must
still be importable and ``available()`` must resolve without the file.
"""
import numpy as np

import btc_web.timemachine as tm

# NOT `from btc_web import _app_ctx`: that triggers Python to import
# `btc_web` as a namespace package and load `btc_web._app_ctx` as a SEPARATE
# module object from the bare `_app_ctx` that app.py's registration block
# (and btc_web/timemachine.py itself) populate via plain `import _app_ctx`
# (btc_web/ is on sys.path at runtime, so every in-package module uses the
# bare form -- see cache.py, figures/dca.py, snapshot.py, etc.). Verified
# directly: `from btc_web import _app_ctx` gives a fresh, empty
# PRICE_MODELS == {} in this test process, distinct id() from the populated
# one. `conftest.py` does the bare `import _app_ctx` at collection time
# (same pattern test_model_registration.py uses via `from conftest import
# _app_ctx`), so re-exporting through conftest reaches the SAME populated
# module instance app.py actually wrote to.
from conftest import _app_ctx


def test_available_reflects_grid_file_presence():
    assert tm.available() == tm.GRID_PATH.exists()


def test_asof_is_per_request_and_pure():
    if not tm.available():
        import pytest
        pytest.skip("grid not built in this env")
    before = _app_ctx.PRICE_MODELS["ecfg_1d_1u"]._params.copy()
    a = tm.asof_eppl("ecfg_1d_1u", 0)
    b = tm.asof_eppl("ecfg_1d_1u", 0)
    assert a is not b                                   # fresh object each call
    assert _app_ctx.PRICE_MODELS["ecfg_1d_1u"]._params == before  # no mutation


def test_frames_and_n_frames_agree():
    if not tm.available():
        import pytest
        pytest.skip("grid not built in this env")
    assert tm.n_frames() == len(tm.frames())
    assert tm.n_frames() > 0


def test_asof_eppl_null_frame_returns_none():
    """A failed cold-fit is stored as JSON null (Task 4) -- must surface as
    None, never crash. Simulated here without depending on a real null
    frame existing in the built grid (none may be present)."""
    if not tm.available():
        import pytest
        pytest.skip("grid not built in this env")
    grid = tm._load()
    key = next(k for k in grid["models"] if k.startswith("ecfg_"))
    orig = grid["models"][key][0]
    grid["models"][key][0] = None
    try:
        assert tm.asof_eppl(key, 0) is None
    finally:
        grid["models"][key][0] = orig  # restore the lru_cache'd singleton


def test_asof_bm_null_frame_returns_none():
    if not tm.available():
        import pytest
        pytest.skip("grid not built in this env")
    grid = tm._load()
    orig = grid["models"]["bub"][0]
    grid["models"]["bub"][0] = None
    try:
        assert tm.asof_bm(0) is None
    finally:
        grid["models"]["bub"][0] = orig


def test_asof_bm_shape():
    if not tm.available():
        import pytest
        pytest.skip("grid not built in this env")
    # Find a non-null BM frame to exercise the real shim construction.
    grid = tm._load()
    idx = next(i for i, rec in enumerate(grid["models"]["bub"]) if rec is not None)
    shim = tm.asof_bm(idx)
    assert shim is not None
    assert len(shim.years_plot_bm) == len(shim.support_bm)
    assert isinstance(shim.comp_by_n, list)
    for attr in ("bm_r2", "bm_sigma0_up", "bm_alpha_up", "bm_sigma0_down", "bm_alpha_down"):
        assert hasattr(shim, attr)


def test_asof_bm_is_downsampled():
    """The grid's BM frames were downsampled post-build (tools/build_time
    machine_grid.py::downsample_existing_grid, MAX_BM_POINTS=512) to bring
    the ~19MB grid under the ~5MB RAM budget every gunicorn worker pays to
    load this. asof_bm's shim must reflect that downsampled resolution --
    comp_by_n rows AND years_plot_bm (from t_grid) both <= 512, and
    support_bm (reconstructed from t_grid at load time) must be aligned to
    the SAME downsampled length, not the original 3000-point grid."""
    if not tm.available():
        import pytest
        pytest.skip("grid not built in this env")
    grid = tm._load()
    idx = next(i for i, rec in enumerate(grid["models"]["bub"]) if rec is not None)
    shim = tm.asof_bm(idx)
    assert len(shim.years_plot_bm) <= 512
    assert all(len(row) <= 512 for row in shim.comp_by_n)
    assert len(shim.support_bm) == len(shim.years_plot_bm)  # reconstructed on the downsampled grid


# ── QR (quantile regression) as-of ───────────────────────────────────────────
def test_asof_qr_shape():
    """asof_qr builds a real QuantileRegressionModel from a non-null frame,
    with float-keyed fits carrying intercept/slope/r2 (matching the live qr)."""
    if not tm.available():
        import pytest
        pytest.skip("grid not built in this env")
    from btc_core import QuantileRegressionModel
    grid = tm._load()
    if "qr" not in grid["models"]:
        import pytest
        pytest.skip("grid built before QR support")
    idx = next(i for i, rec in enumerate(grid["models"]["qr"]) if rec is not None)
    mdl = tm.asof_qr(idx)
    assert isinstance(mdl, QuantileRegressionModel)
    assert len(mdl.fits) > 0
    assert all(isinstance(q, float) for q in mdl.fits)     # float-keyed
    for q, f in mdl.fits.items():
        assert set(f) >= {"intercept", "slope", "r2"}


def test_asof_qr_is_per_request_and_pure():
    """Fresh object each call; never mutates the shared live qr model."""
    if not tm.available():
        import pytest
        pytest.skip("grid not built in this env")
    grid = tm._load()
    if "qr" not in grid["models"]:
        import pytest
        pytest.skip("grid built before QR support")
    idx = next(i for i, rec in enumerate(grid["models"]["qr"]) if rec is not None)
    live = _app_ctx.PRICE_MODELS["qr"]
    before = {q: dict(f) for q, f in live.fits.items()}
    a = tm.asof_qr(idx)
    b = tm.asof_qr(idx)
    assert a is not b                                       # fresh object each call
    assert _app_ctx.PRICE_MODELS["qr"] is live              # singleton identity intact
    assert {q: dict(f) for q, f in live.fits.items()} == before  # no mutation


def test_asof_qr_null_frame_returns_none():
    if not tm.available():
        import pytest
        pytest.skip("grid not built in this env")
    grid = tm._load()
    if "qr" not in grid["models"]:
        import pytest
        pytest.skip("grid built before QR support")
    orig = grid["models"]["qr"][0]
    grid["models"]["qr"][0] = None
    try:
        assert tm.asof_qr(0) is None
    finally:
        grid["models"]["qr"][0] = orig  # restore the lru_cache'd singleton


def test_asof_qr_missing_series_returns_none():
    """A grid built before QR support (no ``"qr"`` key) must surface None,
    not KeyError -- the bubble overlay path then simply skips QR."""
    if not tm.available():
        import pytest
        pytest.skip("grid not built in this env")
    grid = tm._load()
    saved = grid["models"].pop("qr", "__absent__")
    try:
        assert tm.asof_qr(0) is None
    finally:
        if saved != "__absent__":
            grid["models"]["qr"] = saved


# ── spl (Saturating Power Law) as-of ──────────────────────────────────────────
def test_asof_spl_shape():
    """asof_spl builds a real SaturatingPowerLawModel from a non-null frame,
    quantized with constant-sigma quantile fits (matching the live spl)."""
    if not tm.available():
        import pytest
        pytest.skip("grid not built in this env")
    from btc_core import SaturatingPowerLawModel
    grid = tm._load()
    if "spl" not in grid["models"]:
        import pytest
        pytest.skip("grid built before spl support")
    idx = next(i for i, rec in enumerate(grid["models"]["spl"]) if rec is not None)
    mdl = tm.asof_spl(idx)
    assert isinstance(mdl, SaturatingPowerLawModel)
    assert mdl.short_name == "spl"
    assert mdl.quantized
    assert len(mdl.fits) > 0


def test_asof_spl_is_per_request_and_pure():
    """Fresh object each call; never mutates the shared live spl model."""
    if not tm.available():
        import pytest
        pytest.skip("grid not built in this env")
    grid = tm._load()
    if "spl" not in grid["models"]:
        import pytest
        pytest.skip("grid built before spl support")
    idx = next(i for i, rec in enumerate(grid["models"]["spl"]) if rec is not None)
    live = _app_ctx.PRICE_MODELS["spl"]
    before = (live._log10_L, live._t0, live._beta)
    a = tm.asof_spl(idx)
    b = tm.asof_spl(idx)
    assert a is not b                                        # fresh object each call
    assert _app_ctx.PRICE_MODELS["spl"] is live               # singleton identity intact
    assert (live._log10_L, live._t0, live._beta) == before    # no mutation


def test_asof_spl_null_frame_returns_none():
    if not tm.available():
        import pytest
        pytest.skip("grid not built in this env")
    grid = tm._load()
    if "spl" not in grid["models"]:
        import pytest
        pytest.skip("grid built before spl support")
    orig = grid["models"]["spl"][0]
    grid["models"]["spl"][0] = None
    try:
        assert tm.asof_spl(0) is None
    finally:
        grid["models"]["spl"][0] = orig  # restore the lru_cache'd singleton


def test_asof_spl_missing_series_returns_none():
    """A grid built before spl support (no ``"spl"`` key) must surface None,
    not KeyError -- the bubble overlay path then simply skips spl."""
    if not tm.available():
        import pytest
        pytest.skip("grid not built in this env")
    grid = tm._load()
    saved = grid["models"].pop("spl", "__absent__")
    try:
        assert tm.asof_spl(0) is None
    finally:
        if saved != "__absent__":
            grid["models"]["spl"] = saved


# ── logi (Logistic S-curve) as-of ─────────────────────────────────────────────
def test_asof_logi_shape():
    """asof_logi builds a real LogisticSCurveModel from a non-null frame,
    quantized with constant-sigma quantile fits (matching the live logi)."""
    if not tm.available():
        import pytest
        pytest.skip("grid not built in this env")
    from btc_core import LogisticSCurveModel
    grid = tm._load()
    if "logi" not in grid["models"]:
        import pytest
        pytest.skip("grid built before logi support")
    idx = next(i for i, rec in enumerate(grid["models"]["logi"]) if rec is not None)
    mdl = tm.asof_logi(idx)
    assert isinstance(mdl, LogisticSCurveModel)
    assert mdl.short_name == "logi"
    assert mdl.quantized
    assert len(mdl.fits) > 0


def test_asof_logi_is_per_request_and_pure():
    """Fresh object each call; never mutates the shared live logi model."""
    if not tm.available():
        import pytest
        pytest.skip("grid not built in this env")
    grid = tm._load()
    if "logi" not in grid["models"]:
        import pytest
        pytest.skip("grid built before logi support")
    idx = next(i for i, rec in enumerate(grid["models"]["logi"]) if rec is not None)
    live = _app_ctx.PRICE_MODELS["logi"]
    before = (live._K, live._r, live._t0)
    a = tm.asof_logi(idx)
    b = tm.asof_logi(idx)
    assert a is not b                                    # fresh object each call
    assert _app_ctx.PRICE_MODELS["logi"] is live          # singleton identity intact
    assert (live._K, live._r, live._t0) == before         # no mutation


def test_asof_logi_null_frame_returns_none():
    if not tm.available():
        import pytest
        pytest.skip("grid not built in this env")
    grid = tm._load()
    if "logi" not in grid["models"]:
        import pytest
        pytest.skip("grid built before logi support")
    orig = grid["models"]["logi"][0]
    grid["models"]["logi"][0] = None
    try:
        assert tm.asof_logi(0) is None
    finally:
        grid["models"]["logi"][0] = orig  # restore the lru_cache'd singleton


def test_asof_logi_missing_series_returns_none():
    """A grid built before logi support (no ``"logi"`` key) must surface
    None, not KeyError -- the bubble overlay path then simply skips logi."""
    if not tm.available():
        import pytest
        pytest.skip("grid not built in this env")
    grid = tm._load()
    saved = grid["models"].pop("logi", "__absent__")
    try:
        assert tm.asof_logi(0) is None
    finally:
        if saved != "__absent__":
            grid["models"]["logi"] = saved


def test_logi_override_does_not_mutate_gompertz_class_attrs():
    """The shared-attr hazard: LogisticSCurveModel and GompertzModel share
    the _K/_r/_t0 attribute NAMES. Constructing LogisticSCurveModel with
    median overrides + sigma_override (the as-of path asof_logi uses) must
    set INSTANCE attrs only -- GompertzModel's class attrs must be
    unchanged afterward. Does not need the grid built (constructs directly)."""
    from btc_core import GompertzModel, LogisticSCurveModel
    before = (GompertzModel._K, GompertzModel._r, GompertzModel._t0)
    LogisticSCurveModel(
        np.array([1.0, 2.0]), np.array([1.0, 2.0]), [0.1, 0.5, 0.9],
        K=999.0, r=999.0, t0=999.0, sigma_override=0.5,
    )
    after = (GompertzModel._K, GompertzModel._r, GompertzModel._t0)
    assert before == after
