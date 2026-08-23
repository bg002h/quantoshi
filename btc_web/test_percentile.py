"""Percentile-vs-time view (Tab 1 'Percentile' mode)."""
import numpy as np
import pytest

from figures.percentile import build_percentile_figure
from btc_core import today_t
from conftest import M, _encode_snapshot, _decode_snapshot


def _p(**kw):
    base = dict(xmin=2011, xmax=2033, active_models=["bub", "qr"],
                palette="default", xscale="log", show_legend=True,
                show_today=True, minor_grid=False, chart_zoom=False,
                legend_pos="outside", user_model=None)
    base.update(kw)
    return base


def _line_names(fig):
    return {t.name for t in fig.data if getattr(t, "mode", None) == "lines"}


class TestPercentileFigure:
    def test_lines_for_quantized_only(self):
        # s2f / s2f_inst are non-quantized (no fan) -> skipped.
        fig = build_percentile_figure(
            M, _p(active_models=["bub", "qr", "pl", "s2f", "s2f_inst"]))
        assert _line_names(fig) == {"BM", "PL", "QR"}

    def test_non_quantized_only_is_empty(self):
        fig = build_percentile_figure(M, _p(active_models=["s2f", "s2f_inst"]))
        assert not _line_names(fig)

    def test_yaxis_is_0_to_100_percent(self):
        fig = build_percentile_figure(M, _p())
        assert list(fig.layout.yaxis.range) == [0, 100]
        assert fig.layout.yaxis.ticksuffix == "%"

    def test_values_within_0_100(self):
        fig = build_percentile_figure(M, _p(active_models=["bub", "qr", "pl"]))
        for t in fig.data:
            if getattr(t, "mode", None) == "lines":
                y = np.asarray(t.y, float)
                assert y.min() >= 0.0 and y.max() <= 100.0

    def test_valuation_zones_and_median_present(self):
        fig = build_percentile_figure(M, _p())
        rects = [s for s in fig.layout.shapes if s.type == "rect"]
        assert len(rects) >= 2, "expected cheap + rich shaded bands"
        median = [s for s in fig.layout.shapes
                  if s.type == "line" and s.y0 == 50 and s.y1 == 50]
        assert median, "expected a 50% median reference line"

    def test_historical_only_no_future(self):
        # Range extends to 2033, but percentile needs a realized price, so lines
        # must stop at ~today.
        fig = build_percentile_figure(M, _p(xmax=2033, active_models=["bub"]))
        td = today_t(M.genesis)
        for t in fig.data:
            if getattr(t, "mode", None) == "lines":
                assert np.asarray(t.x, float).max() <= td + 0.01


class TestPercentileSnapshot:
    def test_view_mode_percentile_roundtrips(self):
        if _encode_snapshot is None:
            pytest.skip("app import failed")
        state = {"bub-view-mode:data": "percentile"}
        decoded = _decode_snapshot(_encode_snapshot(state))
        assert decoded is not None
        assert decoded.get("bub-view-mode:data") == "percentile"
