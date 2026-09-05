"""Occupancy view (Tab 1 'Occupancy' mode).

Share of a trailing window that BTC's price spent in the tails of a model's
quantile fan (above Q(100-tail) / below Q(tail)), plotted over time, plus a
"when" strip marking the actual tail days.
"""
import re
from unittest.mock import patch

import numpy as np
import pytest

from figures.occupancy import build_occupancy_figure, _occupancy_series
from btc_core import today_t
from conftest import (M, _app_ctx, _encode_snapshot, _decode_snapshot,
                      _SNAPSHOT_CONTROLS)


def _p(**kw):
    base = dict(xmin=2011, xmax=2033, active_models=["bub", "qr"],
                palette="default", xscale="log", show_legend=True,
                show_today=True, minor_grid=False, chart_zoom=False,
                legend_pos="outside", user_model=None, sigma_mode="constant",
                occ_tail=10, occ_window=4)
    base.update(kw)
    return base


def _lines(fig):
    return [t for t in fig.data if getattr(t, "mode", None) == "lines"
            and not t.name.startswith("nominal")]


def _line_names(fig):
    return {t.name for t in _lines(fig)}


def _strip(fig):
    return [t for t in fig.data if getattr(t, "mode", None) == "markers"]


def _ticks(fig):
    return [t for t in _strip(fig) if t.marker.opacity != 0]


def _hover(fig):
    h = [t for t in _strip(fig) if t.marker.opacity == 0]
    assert len(h) <= 1
    return h[0] if h else None


def _daily_t(years):
    return np.arange(0.0, years, 1.0 / 365.25)


# ═══════════════════════════════════════════════════════════════════════════════
# Series math
# ═══════════════════════════════════════════════════════════════════════════════

class TestOccupancySeries:
    def test_known_fractions_on_step_series(self):
        # 8 years daily; percentile 95 for the first 2 years, 50 afterwards.
        t = _daily_t(8)
        pct = np.where(t < 2.0, 95.0, 50.0)
        t_out, above, below = _occupancy_series(t, pct, tail=10, window=4.0)
        # At t=4 the trailing 4-yr window covers (0, 4]: 2 of 4 years above Q90.
        i4 = int(np.argmin(np.abs(t_out - 4.0)))
        assert abs(above[i4] - 50.0) < 1.0
        assert below[i4] == 0.0
        # By t~8 the window is (4, 8]: nothing above Q90 any more.
        assert above[-1] == 0.0

    def test_below_counts_the_low_tail(self):
        t = _daily_t(6)
        pct = np.full(t.shape, 5.0)
        _, above, below = _occupancy_series(t, pct, tail=10, window=4.0)
        assert np.all(below == 100.0)
        assert np.all(above == 0.0)

    def test_thresholds_are_inclusive(self):
        # Exactly Q90 counts as "above" for tail 10; exactly Q10 counts as "below".
        t = _daily_t(6)
        _, above, _ = _occupancy_series(t, np.full(t.shape, 90.0), tail=10, window=4.0)
        _, _, below = _occupancy_series(t, np.full(t.shape, 10.0), tail=10, window=4.0)
        assert np.all(above == 100.0)
        assert np.all(below == 100.0)

    def test_no_output_before_a_full_window(self):
        t = _daily_t(6)
        pct = np.full(t.shape, 50.0)
        t_out, above, below = _occupancy_series(t, pct, tail=10, window=4.0)
        assert t_out[0] >= t[0] + 4.0 - 1e-9
        assert len(t_out) == len(above) == len(below)
        # Shorter history than the window -> nothing to show.
        t3 = _daily_t(3)
        t_out3, _, _ = _occupancy_series(t3, np.full(t3.shape, 50.0), tail=10, window=4.0)
        assert len(t_out3) == 0

    def test_window_is_trailing_not_centered(self):
        # A single above-Q90 year at t in [2, 3): occupancy must be nonzero at
        # t=3 (window (−1, 3] holds it) and zero again once the window has
        # passed it, i.e. for t > 7.
        t = _daily_t(9)
        pct = np.where((t >= 2.0) & (t < 3.0), 95.0, 50.0)
        t_out, above, _ = _occupancy_series(t, pct, tail=10, window=4.0)
        assert above[int(np.argmin(np.abs(t_out - 4.0)))] > 20.0
        assert above[int(np.argmin(np.abs(t_out - 7.5)))] == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Figure
# ═══════════════════════════════════════════════════════════════════════════════

class TestOccupancyFigure:
    def test_two_lines_per_quantized_model(self):
        fig = build_occupancy_figure(M, _p(active_models=["bub", "qr", "s2f"]))
        assert _line_names(fig) == {"BM ≥Q90", "BM ≤Q10", "QR ≥Q90", "QR ≤Q10"}

    def test_non_quantized_only_is_empty(self):
        fig = build_occupancy_figure(M, _p(active_models=["s2f", "s2f_inst"]))
        assert not _line_names(fig)
        assert not _strip(fig)

    def test_tail_sets_labels_and_changes_values(self):
        f10 = build_occupancy_figure(M, _p(active_models=["bub"], occ_tail=10))
        f25 = build_occupancy_figure(M, _p(active_models=["bub"], occ_tail=25))
        assert _line_names(f25) == {"BM ≥Q75", "BM ≤Q25"}
        y10 = np.asarray([t for t in _lines(f10) if "≥" in t.name][0].y, float)
        y25 = np.asarray([t for t in _lines(f25) if "≥" in t.name][0].y, float)
        assert y25.mean() > y10.mean()

    def test_nominal_reference_line_at_tail(self):
        fig = build_occupancy_figure(M, _p(occ_tail=25))
        nominal = [t for t in fig.data if t.name == "nominal 25%"]
        assert len(nominal) == 1
        assert nominal[0].line.dash == "dot"
        assert set(np.asarray(nominal[0].y, float)) == {25.0}

    def test_values_within_0_100(self):
        fig = build_occupancy_figure(M, _p(active_models=["bub", "qr", "pl"]))
        for t in _lines(fig):
            y = np.asarray(t.y, float)
            assert y.min() >= 0.0 and y.max() <= 100.0

    def test_historical_only_no_future(self):
        fig = build_occupancy_figure(M, _p(xmax=2033, active_models=["bub"]))
        td = today_t(M.genesis)
        for t in list(_lines(fig)) + _strip(fig):
            assert np.asarray(t.x, float).max() <= td + 0.01

    def test_line_starts_one_window_after_first_price(self):
        fig = build_occupancy_figure(M, _p(xmin=2010, active_models=["bub"], occ_window=4))
        first_t = float(M.price_years[M.price_years > 0][0])
        for t in _lines(fig):
            assert np.asarray(t.x, float).min() >= first_t + 4.0 - 1e-6

    def test_xmin_does_not_change_values(self):
        # Occupancy is computed over the full history and only DISPLAYED for
        # the x-range; narrowing xmin must not alter overlapping values.
        wide = build_occupancy_figure(M, _p(xmin=2011, active_models=["bub"]))
        narrow = build_occupancy_figure(M, _p(xmin=2018, active_models=["bub"]))
        w = [t for t in _lines(wide) if "≥" in t.name][0]
        n = [t for t in _lines(narrow) if "≥" in t.name][0]
        wx, wy = np.asarray(w.x, float), np.asarray(w.y, float)
        nx, ny = np.asarray(n.x, float), np.asarray(n.y, float)
        common = np.isin(wx, nx)
        assert common.sum() > 100
        assert np.array_equal(wy[common], ny[np.isin(nx, wx)])

    def test_window_changes_values(self):
        y1 = np.asarray([t for t in _lines(
            build_occupancy_figure(M, _p(active_models=["bub"], occ_window=1))) if "≥" in t.name][0].y)
        y4 = np.asarray([t for t in _lines(
            build_occupancy_figure(M, _p(active_models=["bub"], occ_window=4))) if "≥" in t.name][0].y)
        assert len(y1) > len(y4)          # 1-yr window starts 3 years earlier
        assert np.abs(y1[-len(y4):] - y4).max() > 1.0

    def test_strip_marks_first_model_only(self):
        fig = build_occupancy_figure(M, _p(active_models=["bub", "qr"]))
        strip = _strip(fig)
        assert len(strip) == 3                    # 2 tick rows + 1 hover trace
        assert len(_ticks(fig)) == 2
        assert all(t.yaxis == "y2" for t in strip)
        assert all("BM" in t.name for t in strip)
        assert all(t.showlegend is False for t in strip)

    def test_tick_markers_do_not_hover(self):
        # Hover on the bar must always land on the full-coverage hover trace,
        # so the sparse tick markers must not compete for it.
        fig = build_occupancy_figure(M, _p(active_models=["bub"]))
        assert all(t.hoverinfo == "skip" for t in _ticks(fig))

    def test_strip_hover_trace_covers_every_displayed_day(self):
        fig = build_occupancy_figure(M, _p(xmin=2016, active_models=["bub"]))
        h = _hover(fig)
        assert h is not None and h.yaxis == "y2"
        from btc_core import yr_to_t
        td = today_t(M.genesis)
        t_lo = yr_to_t(2016, M.genesis)
        n_days = int(((M.price_years >= t_lo) & (M.price_years <= td)).sum())
        # covers every displayed day (2016-01-01 .. today), not just tail days
        assert len(h.x) == n_days
        assert len(h.text) == len(h.x)
        assert "%{text}" in h.hovertemplate and "%{customdata[0]}" in h.hovertemplate

    def test_strip_hover_text_matches_line_values(self):
        fig = build_occupancy_figure(M, _p(active_models=["bub"], occ_tail=10))
        h = _hover(fig)
        above = [t for t in _lines(fig) if "≥" in t.name][0]
        below = [t for t in _lines(fig) if "≤" in t.name][0]
        x_last = float(np.asarray(above.x, float)[-1])
        i = int(np.argmin(np.abs(np.asarray(h.x, float) - x_last)))
        txt = h.text[i]
        a = float(re.search(r"≥Q90 ([\d.]+)%", txt).group(1))
        b = float(re.search(r"≤Q10 ([\d.]+)%", txt).group(1))
        assert abs(a - float(above.y[-1])) < 0.06
        assert abs(b - float(below.y[-1])) < 0.06

    def test_strip_hover_before_full_window_says_so(self):
        fig = build_occupancy_figure(M, _p(xmin=2010, active_models=["bub"], occ_window=4))
        h = _hover(fig)
        first_line_t = float(np.asarray([t.x for t in _lines(fig)][0]).min())
        hx = np.asarray(h.x, float)
        early = [h.text[i] for i in range(len(hx)) if hx[i] < first_line_t - 1e-9]
        assert early and all("not yet full" in t for t in early)
        late = [h.text[i] for i in range(len(hx)) if hx[i] >= first_line_t]
        assert late and all("≥Q90" in t for t in late)

    def test_strip_days_are_the_tail_days(self):
        # For QR (quantile regression on ALL data) the share of days above Q90
        # is ~10% by construction — the strip must reflect that, not e.g. the
        # windowed values.
        fig = build_occupancy_figure(M, _p(xmin=2010, active_models=["qr"]))
        above = [t for t in _strip(fig) if "≥" in t.name][0]
        n_days = int(((M.price_years > 0) & (M.price_years <= today_t(M.genesis))).sum())
        share = len(above.x) / n_days
        assert 0.06 < share < 0.14

    def test_strip_axis_is_labelled_for_colorblind_readers(self):
        fig = build_occupancy_figure(M, _p(occ_tail=10))
        assert list(fig.layout.yaxis2.ticktext) == ["≤Q10", "≥Q90"]

    def test_sigma_mode_changes_values(self):
        yc = np.asarray([t for t in _lines(build_occupancy_figure(
            M, _p(active_models=["bub"], sigma_mode="constant"))) if "≥" in t.name][0].y)
        yr = np.asarray([t for t in _lines(build_occupancy_figure(
            M, _p(active_models=["bub"], sigma_mode="resqr"))) if "≥" in t.name][0].y)
        assert np.abs(yc - yr).max() > 1.0

    def test_empty_selection_shows_empty_state(self):
        fig = build_occupancy_figure(M, _p(active_models=[]))
        texts = [a.text for a in fig.layout.annotations]
        assert any("No models selected" in t for t in texts)


# ═══════════════════════════════════════════════════════════════════════════════
# Wiring: pill toggle, clientside sync, snapshot
# ═══════════════════════════════════════════════════════════════════════════════

class _Ctx:
    def __init__(self, triggered_id):
        self.triggered_id = triggered_id


class TestOccupancyWiring:
    def test_pill_toggle_switches_to_occupancy(self):
        from callbacks.charts import toggle_bub_view
        with patch.multiple("callbacks.charts", ctx=_Ctx("bub-view-occ")):
            out = toggle_bub_view(0, 0, 0, 0, 1, [2010, 2033])
        assert out[0] == "occupancy"
        assert out[5] == {}                       # bub-occ-wrap shown
        assert out[4] == {"display": "none"}      # bub-pctile-wrap hidden
        assert out[10] is False                   # bub-view-occ filled (active)
        assert out[9] is True                     # bub-view-pctile outlined
        assert out[14] == {"display": "inline"}   # bub-occ-ctl-wrap shown

    def test_pill_toggle_back_to_price_hides_occupancy(self):
        from callbacks.charts import toggle_bub_view
        with patch.multiple("callbacks.charts", ctx=_Ctx("bub-view-price")):
            out = toggle_bub_view(1, 0, 0, 0, 0, [2010, 2033])
        assert out[0] == "price"
        assert out[5] == {"display": "none"}
        assert out[10] is True
        assert out[14] == {"display": "none"}

    def test_clientside_sync_returns_one_value_per_output(self):
        # The view-mode sync JS returns positional arrays; adding an Output
        # without extending every `return [...]` silently mis-assigns styles.
        scripts = [s for s in _app_ctx.app._inline_scripts
                   if 'mode === "occupancy"' in s and 'mode === "percentile"' in s]
        assert len(scripts) == 1, "view-mode sync clientside callback not found"
        returns = re.findall(r"return \[(.*?)\];", scripts[0], flags=re.S)
        assert len(returns) == 5   # cagr, resid, percentile, occupancy, price
        counts = {len([e for e in r.split(",") if e.strip()]) for r in returns}
        assert counts == {14}, counts

    def test_historical_only_clientside_checks_include_occupancy(self):
        hits = [s for s in _app_ctx.app._inline_scripts
                if "mode === 'occupancy'" in s and "mode === 'percentile'" in s]
        # n-future hide + x-range cap
        assert len(hits) == 2

    def test_update_callback_builds_figure(self):
        from callbacks.charts import update_bub_occ
        fig = update_bub_occ("occupancy", [2010, 2033], ["show_today"], "log",
                             ["bub"], "outside", "default", "constant",
                             10, 4, None, False)
        assert _line_names(fig) == {"BM ≥Q90", "BM ≤Q10"}


class TestOccupancySnapshot:
    def test_mode_tail_window_round_trip(self):
        if _encode_snapshot is None:
            pytest.skip("app import failed")
        state = {"bub-view-mode:data": "occupancy",
                 "bub-occ-tail:value": 25, "bub-occ-window:value": 1}
        dec = _decode_snapshot(_encode_snapshot(state))
        assert dec is not None
        assert dec.get("bub-view-mode:data") == "occupancy"
        assert dec.get("bub-occ-tail:value") == 25
        assert dec.get("bub-occ-window:value") == 1

    def test_new_fields_appended_at_the_absolute_tail(self):
        # v4 links address fields by array index: anything but the tail shifts
        # every later entry and corrupts shipped links (see
        # test_timemachine_snapshot.py for the history). Time Machine's two
        # entries were 334/335 of 336 before this feature and must stay there.
        assert _SNAPSHOT_CONTROLS[334] == ("bub-timemachine-toggle", "value")
        assert _SNAPSHOT_CONTROLS[335] == ("bub-asof-slider", "value")
        assert _SNAPSHOT_CONTROLS[-2] == ("bub-occ-tail", "value")
        assert _SNAPSHOT_CONTROLS[-1] == ("bub-occ-window", "value")

    def test_defaults_registered(self):
        from snapshot_defaults import SNAPSHOT_DEFAULTS
        assert SNAPSHOT_DEFAULTS["bub-occ-tail:value"] == 10
        assert SNAPSHOT_DEFAULTS["bub-occ-window:value"] == 4

    def test_controls_belong_to_bubble_tab(self):
        from callbacks.routing import _TAB_CONTROLS
        assert {"bub-occ-tail", "bub-occ-window"} <= _TAB_CONTROLS["bubble"]
