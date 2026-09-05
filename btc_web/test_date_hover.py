"""figures.common._add_date_hover / _date_customdata.

The helper stamps a calendar-date string onto ``customdata[0]`` of every trace
whose x looks like t (years since genesis).  It used to REPLACE customdata
outright, which forced any trace with its own per-point values to smuggle them
through ``text`` as a pre-formatted string — several times the bytes of the
numbers behind it.  These tests pin the preserve-and-prepend behaviour and the
fallbacks that must stay byte-identical to the old one.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from conftest import M                                   # noqa: F401  (sys.path)
from figures.common import _add_date_hover, _date_customdata, _t_to_datestr

GENESIS = pd.Timestamp("2009-07-25")
# Three t values well inside the helper's plausible range (0.3 – 120).
T = [1.5, 2.5, 3.5]
DATES = [_t_to_datestr(t, GENESIS) for t in T]


def _rows(trace):
    """customdata as plain lists (plotly may hand back tuples/arrays)."""
    return [list(r) for r in trace.customdata]


def _fig(**kw):
    return go.Figure(data=[go.Scatter(x=T, y=[1, 2, 3], **kw)])


class TestDateCustomdata:
    def test_existing_rows_are_kept_and_date_prepended(self):
        fig = _fig(customdata=[[8.6, 6.4], [1.0, 2.0], [0.0, 100.0]],
                   hovertemplate="%{customdata[0]} %{customdata[1]:.1f}")
        _add_date_hover(fig, GENESIS)
        assert _rows(fig.data[0]) == [
            [DATES[0], 8.6, 6.4], [DATES[1], 1.0, 2.0], [DATES[2], 0.0, 100.0]]

    def test_numeric_rows_stay_numeric(self):
        # A hovertemplate formatting customdata[1] as %{...:.1f} renders NaN if
        # the value arrives as a string, so the numbers must survive as numbers.
        fig = _fig(customdata=np.array([[8.6, 6.4], [1.0, 2.0], [0.0, 100.0]]))
        _add_date_hover(fig, GENESIS)
        for row in fig.data[0].customdata:
            assert isinstance(row[0], str)
            assert all(isinstance(v, float) for v in row[1:])

    def test_no_customdata_gets_plain_date_rows(self):
        fig = _fig()
        _add_date_hover(fig, GENESIS)
        assert _rows(fig.data[0]) == [[d] for d in DATES]

    def test_flat_scalar_customdata_is_replaced(self):
        # supercharge Mode B layout 2 ships a flat list of quantile labels with
        # a "%{customdata}" template; prepending would turn each entry into a
        # 2-element array and change what it renders.  Old behaviour stands.
        fig = _fig(customdata=["Q10%", "Q50%", "Q90%"])
        _add_date_hover(fig, GENESIS)
        assert _rows(fig.data[0]) == [[d] for d in DATES]

    def test_length_mismatch_is_replaced(self):
        fig = _fig(customdata=[[1.0], [2.0]])
        _add_date_hover(fig, GENESIS)
        assert _rows(fig.data[0]) == [[d] for d in DATES]

    def test_hoverinfo_skip_trace_is_untouched(self):
        fig = _fig(customdata=[[1.0], [2.0], [3.0]], hoverinfo="skip")
        _add_date_hover(fig, GENESIS)
        assert _rows(fig.data[0]) == [[1.0], [2.0], [3.0]]

    def test_out_of_range_x_trace_is_untouched(self):
        # Heatmap-style calendar years (> 120) must not be read as t.
        fig = go.Figure(data=[go.Scatter(x=[2025, 2030], y=[1, 2],
                                         customdata=[[1.0], [2.0]])])
        _add_date_hover(fig, GENESIS)
        assert _rows(fig.data[0]) == [[1.0], [2.0]]

    def test_helper_is_pure(self):
        assert _date_customdata(None, ["a", "b"]) == [["a"], ["b"]]
        assert _date_customdata([[1], [2]], ["a", "b"]) == [["a", 1], ["b", 2]]
