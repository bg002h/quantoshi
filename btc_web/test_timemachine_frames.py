# btc_web/test_timemachine_frames.py
from tools.timemachine.frames import frame_dates

def test_quarterly_then_monthly_boundary():
    d = frame_dates("2013-01-01", "2026-07-01")
    assert d == sorted(set(d))                      # sorted, unique
    assert "2013-01-01" in d and "2013-04-01" in d  # quarterly early
    assert "2015-10-01" in d and "2015-11-01" not in d
    assert "2016-01-01" in d and "2016-02-01" in d  # monthly from 2016
    assert d[0] == "2013-01-01" and d[-1] == "2026-07-01"
