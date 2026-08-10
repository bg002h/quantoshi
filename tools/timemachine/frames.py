# tools/timemachine/frames.py
import pandas as pd

def frame_dates(left_edge: str, last_full_month: str) -> list[str]:
    le = pd.Timestamp(left_edge); lm = pd.Timestamp(last_full_month)
    q = pd.date_range(le, "2015-10-01", freq="QS")            # quarter starts
    m = pd.date_range(max(pd.Timestamp("2016-01-01"), le), lm, freq="MS")
    out = sorted({d.strftime("%Y-%m-%d") for d in list(q) + list(m)})
    return out
