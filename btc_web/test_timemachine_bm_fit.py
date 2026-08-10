# btc_web/test_timemachine_bm_fit.py
import pickle

# Import fit_bm_asof FIRST: it inserts tools/ onto sys.path as a side effect
# of import (same pattern as tools/timemachine/spike.py /
# tools/timemachine/fit_eppl_asof.py), which is what lets the bare
# `model_toolkit.data` import below resolve. conftest.py only puts
# repo-root and btc_web/ on sys.path, not tools/.
from tools.timemachine.fit_bm_asof import fit_bm_asof
from tools.model_toolkit.data import load_prices


def test_rightedge_bm_matches_shipped_within_tol():
    pr = load_prices("BitcoinPricesDaily.csv")
    r = fit_bm_asof(pr, ymax=float(pr.df["years"].max()))
    # model_data.pkl is a trusted, locally-generated repo artifact (built by
    # tools/build_bm_model.py) -- not an untrusted/external source.
    d = pickle.load(open("model_data.pkl", "rb"))  # noqa: S301 - trusted local file
    assert abs(r["support_slope"] - d["bm_support_slope"]) < 0.05  # most stable BM invariant
    assert r["bm_r2"] > 0.9
    assert len(r["comp_by_n"]) >= 1 and len(r["t_grid"]) > 0
