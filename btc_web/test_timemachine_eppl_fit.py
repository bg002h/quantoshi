# btc_web/test_timemachine_eppl_fit.py
import numpy as np

# Import fit_config_asof FIRST: fit_eppl_asof.py inserts tools/ onto sys.path
# as a side effect of import (same pattern as tools/timemachine/spike.py),
# which is what lets the bare `model_toolkit.data` import below resolve.
# conftest.py only puts repo-root and btc_web/ on sys.path, not tools/.
from tools.timemachine.fit_eppl_asof import fit_config_asof
from model_toolkit.data import load_prices


def test_rightedge_recovers_published_ecfg_1d_1u():
    pr = load_prices("BitcoinPricesDaily.csv")
    t = pr.df_full["years"].values; lp = pr.df_full["log_price"].values
    r = fit_config_asof((1, 1, ["d"], ["u"]), t, lp, ymax=t.max(), maxiter=300)
    assert r["r2"] > 0.97
    assert r["sigma"] > 0                       # constant band σ = std(residuals)
    assert r["params"]["B"] > 3                 # trend slope in a sane range
    assert r["n_log"] == 1 and r["n_cal"] == 1
