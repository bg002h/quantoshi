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
