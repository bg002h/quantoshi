"""Percentile-vs-time view (Tab 1 'Percentile' mode)."""
import numpy as np
import pytest

from figures.percentile import build_percentile_figure, _percentile_series, _bracket_percentile
from btc_core import today_t
from conftest import M, _app_ctx, _encode_snapshot, _decode_snapshot


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

    def test_bracket_percentile_handles_non_monotonic_fan(self):
        # Review finding: a crossed/non-monotonic fan must use find_percentile's
        # first-bracket scan, not np.interp (which silently mis-maps unsorted xp).
        qs = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        col = np.array([1.0, 2.0, 3.0, 2.5, 4.0])  # crossing between q=0.5 and 0.75
        # 2.5 falls in the FIRST bracket [col[1]=2, col[2]=3] -> q in 0.25..0.5
        assert abs(_bracket_percentile(2.5, col, qs) - 0.375) < 1e-9
        assert _bracket_percentile(0.0, col, qs) == 0.1   # below fan -> qs[0]
        assert _bracket_percentile(9.0, col, qs) == 0.9   # above fan -> qs[-1]

    def test_series_matches_find_percentile_qr(self):
        # _percentile_series must equal per-date find_percentile for QR, whose
        # fan is non-monotonic — the consistency guarantee with the navbar ticker.
        m = _app_ctx.PRICE_MODELS["qr"]
        mask = (M.price_years >= 1.0) & (M.price_years <= today_t(M.genesis))
        t = M.price_years[mask][::7]
        px = M.price_prices[mask][::7]
        vec = _percentile_series(m, t, px) / 100.0
        oracle = np.array([m.find_percentile(float(t[i]), float(px[i]))
                           for i in range(len(t))])
        assert np.abs(vec - oracle).max() < 1e-9

    def test_sigma_mode_changes_oscillator(self):
        # resqr (residual quantile bands) vs constant σ give different fans, so
        # the price sits at a different percentile — the toggle must matter.
        yc = np.asarray([t for t in
                         build_percentile_figure(M, _p(active_models=["bub"], sigma_mode="constant")).data
                         if t.mode == "lines"][0].y)
        yr = np.asarray([t for t in
                         build_percentile_figure(M, _p(active_models=["bub"], sigma_mode="resqr")).data
                         if t.mode == "lines"][0].y)
        assert np.abs(yc - yr).max() > 1.0


class TestPercentileSnapshot:
    def test_view_mode_percentile_roundtrips(self):
        if _encode_snapshot is None:
            pytest.skip("app import failed")
        state = {"bub-view-mode:data": "percentile"}
        decoded = _decode_snapshot(_encode_snapshot(state))
        assert decoded is not None
        assert decoded.get("bub-view-mode:data") == "percentile"
