"""Task 5: runtime grid loader (btc_web/timemachine.py).

Verifies ``asof_eppl`` builds a FRESH per-request object on every call and
never mutates the shared ``_app_ctx.PRICE_MODELS`` registry -- the same
escape-hatch contract as the ``u1`` (User Model) click-to-draw path.

Skips (rather than fails) when the grid hasn't been built in this
environment (`timemachine_grid.json.gz` missing) -- the loader itself must
still be importable and ``available()`` must resolve without the file.
"""
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
