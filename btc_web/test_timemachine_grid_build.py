# btc_web/test_timemachine_grid_build.py
import gzip, json
import pytest
from tools.build_timemachine_grid import build_grid, continuity_scan

def test_two_frame_build(tmp_path):
    out = tmp_path / "g.json.gz"
    build_grid(frames=["2016-01-01", "2016-02-01"],
               configs=[(1, 1, ["d"], ["u"])], include_bm=True,
               out_path=str(out), maxiter=150, workers=1)
    with gzip.open(out, "rt") as f:
        g = json.load(f)
    assert g["frames"] == ["2016-01-01", "2016-02-01"]
    assert "ecfg_1d_1u" in g["models"] and "bub" in g["models"]
    assert len(g["models"]["ecfg_1d_1u"]) == 2
    assert "params" in g["models"]["ecfg_1d_1u"][0]
    # include_qr defaults True → a params-only "qr" series aligned to frames.
    assert "qr" in g["models"] and len(g["models"]["qr"]) == 2
    assert "fits" in g["models"]["qr"][0]
    assert len(g["models"]["qr"][0]["fits"]) == 27  # default BM_QUANTILES
    # continuity scan runs and returns a (possibly empty) list of suspect strings
    suspects = continuity_scan(g)
    assert isinstance(suspects, list)


def test_add_qr_to_existing_grid(tmp_path):
    """The incremental --add-qr path injects a 'qr' series into an EXISTING
    grid (built here without QR) without touching its bub/ecfg series."""
    from tools.build_timemachine_grid import add_qr_to_grid
    out = tmp_path / "g.json.gz"
    build_grid(frames=["2016-01-01", "2016-02-01"],
               configs=[(1, 1, ["d"], ["u"])], include_bm=True, include_qr=False,
               out_path=str(out), maxiter=150, workers=1)
    with gzip.open(out, "rt") as f:
        before = json.load(f)
    assert "qr" not in before["models"]
    add_qr_to_grid(str(out), workers=1)
    with gzip.open(out, "rt") as f:
        after = json.load(f)
    assert "qr" in after["models"] and len(after["models"]["qr"]) == 2
    assert "fits" in after["models"]["qr"][0]
    # bub / ecfg series untouched by the incremental add.
    assert after["models"]["bub"] == before["models"]["bub"]
    assert after["models"]["ecfg_1d_1u"] == before["models"]["ecfg_1d_1u"]


def test_add_series_to_grid_spl_logi(tmp_path):
    """The generalized incremental-add path injects BOTH 'spl' and 'logi'
    series (params-only) into an EXISTING grid, in ONE call, without
    touching its bub/ecfg/qr series."""
    from tools.build_timemachine_grid import add_series_to_grid
    out = tmp_path / "g.json.gz"
    build_grid(frames=["2016-01-01", "2016-02-01"],
               configs=[(1, 1, ["d"], ["u"])], include_bm=True,
               out_path=str(out), maxiter=150, workers=1)
    with gzip.open(out, "rt") as f:
        before = json.load(f)
    assert "spl" not in before["models"] and "logi" not in before["models"]

    add_series_to_grid(str(out), ["spl", "logi"], 1)

    with gzip.open(out, "rt") as f:
        after = json.load(f)
    assert "spl" in after["models"] and len(after["models"]["spl"]) == 2
    assert "logi" in after["models"] and len(after["models"]["logi"]) == 2
    assert set(after["models"]["spl"][0]["params"]) == {"log10_L", "t0", "beta"}
    assert set(after["models"]["logi"][0]["params"]) == {"K", "r", "t0"}
    for kind in ("spl", "logi"):
        for rec in after["models"][kind]:
            assert rec is None or {"params", "sigma", "r2"} <= set(rec)
    # bub / ecfg / qr series untouched by the incremental add.
    assert after["models"]["bub"] == before["models"]["bub"]
    assert after["models"]["ecfg_1d_1u"] == before["models"]["ecfg_1d_1u"]
    assert after["models"]["qr"] == before["models"]["qr"]


def test_add_series_to_grid_rejects_unknown_kind(tmp_path):
    from tools.build_timemachine_grid import add_series_to_grid
    out = tmp_path / "g.json.gz"
    build_grid(frames=["2016-01-01"], configs=[(1, 1, ["d"], ["u"])],
               include_bm=False, out_path=str(out), maxiter=150, workers=1)
    with pytest.raises(ValueError):
        add_series_to_grid(str(out), ["spl", "bogus"], 1)


def test_add_qr_to_grid_is_now_add_series_to_grid_wrapper(tmp_path):
    """add_qr_to_grid (the pre-existing name --add-qr's CLI branch calls)
    must still work, delegating to add_series_to_grid(path, ["qr"], workers)."""
    from tools.build_timemachine_grid import add_qr_to_grid
    out = tmp_path / "g.json.gz"
    build_grid(frames=["2016-01-01", "2016-02-01"],
               configs=[(1, 1, ["d"], ["u"])], include_bm=True, include_qr=False,
               out_path=str(out), maxiter=150, workers=1)
    with gzip.open(out, "rt") as f:
        before = json.load(f)
    assert "qr" not in before["models"]
    add_qr_to_grid(str(out), workers=1)
    with gzip.open(out, "rt") as f:
        after = json.load(f)
    assert "qr" in after["models"] and len(after["models"]["qr"]) == 2
    assert after["models"]["bub"] == before["models"]["bub"]


def test_continuity_scan_fires_on_bm_median_jump():
    """Hand-crafted grid whose adjacent BM frames jump > 0.5 in log10 median.

    Task 4's review flagged that the existing continuity tests only assert
    ``isinstance(suspects, list)`` -- never that the >0.5 jump-detection
    branch actually fires. This feeds continuity_scan a minimal grid (no
    real fitting/build_grid() involved) where frame 1's composite median is
    flat at $10 and frame 2's is flat at $1e6 (log10 delta = 5.0, well past
    the 0.5 threshold), and asserts the SUSPECT line for the "bub" key
    appears. bm_r2 is set well above the 0.85 r2-threshold on both frames
    so this isolates the jump branch specifically (not the r2 branch).
    """
    t_grid = [1.0, 20.0]
    grid = {
        "frames": ["2016-01-01", "2016-02-01"],
        "models": {
            "bub": [
                {"bm_r2": 0.99, "t_grid": t_grid, "comp_by_n": [[10.0, 10.0]]},
                {"bm_r2": 0.99, "t_grid": t_grid, "comp_by_n": [[1e6, 1e6]]},
            ],
        },
    }
    suspects = continuity_scan(grid)
    assert suspects, "expected the >0.5 log10 median-jump branch to fire"
    assert any(s.startswith("SUSPECT") and "bub" in s and "max median-log10 change"
               in s for s in suspects)
