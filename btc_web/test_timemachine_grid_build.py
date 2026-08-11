# btc_web/test_timemachine_grid_build.py
import gzip, json
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
    # continuity scan runs and returns a (possibly empty) list of suspect strings
    suspects = continuity_scan(g)
    assert isinstance(suspects, list)


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
