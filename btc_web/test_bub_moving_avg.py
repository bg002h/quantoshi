"""Tab-1 moving averages — the ``bub-ma`` Display control.

Four selectable simple moving averages of the daily close (200 week / 52 week /
30 day / 7 day) drawn on the Price view of Tab 1.

The tests are grouped the way the feature can break:

* **maths** — the window is a plain arithmetic mean of daily closes and a
  partial window is never emitted;
* **rendering** — one trace per selection, clipped/downsampled/stacked exactly
  like the price scatter beside it;
* **as-of** — a trailing mean must not be drawn past a Time Machine frame date,
  which would leak prices the frozen model was never fit on;
* **registries** — the six plumbing sites that all fail *silently* (the chart
  looks right either way): snapshot control list, checklist bitmask options,
  tab-control set, snapshot defaults, cache-key alignment, restore fast path.
"""
import ast
import pathlib

import numpy as np
import pytest

from bub_ma import MA_WINDOWS, MA_VALUES, MA_BY_VALUE, MA_OPTIONS, rolling_mean
from figures.bubble import build_bubble_figure
from figures.common import _MAX_SCATTER_PTS
from tab_defaults import bubble_defaults
from snapshot_defaults import SNAPSHOT_DEFAULTS
from conftest import (M, _app_ctx, _encode_snapshot, _decode_snapshot,
                      _SNAPSHOT_CONTROLS, _CHECKLIST_OPTIONS,
                      _list_to_mask, _mask_to_list, _TAB_CONTROLS)


_BASE = dict(
    selected_qs=[0.5], active_models=["bub"], sigma_mode="constant",
    xscale="log", yscale="log", xmin=2011, xmax=2030,
    ymin=1.0, ymax=1e8, show_data=True, show_comp=True, show_sup=False,
    n_future=3, palette="default",
)

_LEGEND_NAMES = {w.legend for w in MA_WINDOWS}


def _fig(**kw):
    fig, _ = build_bubble_figure(M, {**_BASE, **kw})
    return fig


def _ma_traces(fig):
    """Every MA trace, in figure order."""
    return [t for t in fig.data if (t.name or "") in _LEGEND_NAMES]


def _ma(fig, legend):
    return next(t for t in fig.data if (t.name or "") == legend)


# ══════════════════════════════════════════════════════════════════════════
# Maths — a full-window arithmetic mean of daily closes
# ══════════════════════════════════════════════════════════════════════════

class TestRollingMean:
    def test_hand_computed_window_of_three(self):
        """[1,2,3,4,5] with w=3 → mean(1,2,3)=2, mean(2,3,4)=3, mean(3,4,5)=4."""
        out = rolling_mean([1.0, 2.0, 3.0, 4.0, 5.0], 3)
        assert np.isnan(out[0]) and np.isnan(out[1])
        assert out[2] == pytest.approx(2.0)
        assert out[3] == pytest.approx(3.0)
        assert out[4] == pytest.approx(4.0)

    def test_hand_computed_uneven_series(self):
        """Deliberately not an arithmetic progression: catches an off-by-one
        window that a straight ramp would hide (every ramp sub-mean equals the
        midpoint, so a window shifted by one still lands on a 'nice' value)."""
        s = [10.0, 1.0, 100.0, 4.0, 25.0]
        out = rolling_mean(s, 2)
        assert np.isnan(out[0])
        assert out[1] == pytest.approx(5.5)      # (10+1)/2
        assert out[2] == pytest.approx(50.5)     # (1+100)/2
        assert out[3] == pytest.approx(52.0)     # (100+4)/2
        assert out[4] == pytest.approx(14.5)     # (4+25)/2

    def test_full_window_rule_nothing_before_day_n(self):
        """A partial window is never emitted — the first N-1 slots are NaN."""
        n, w = 40, 7
        out = rolling_mean(np.arange(1.0, n + 1), w)
        assert np.isnan(out[:w - 1]).all()
        assert np.isfinite(out[w - 1:]).all()
        assert np.count_nonzero(np.isfinite(out)) == n - w + 1

    def test_series_shorter_than_window_is_all_nan(self):
        out = rolling_mean([1.0, 2.0, 3.0], 10)
        assert len(out) == 3
        assert np.isnan(out).all()

    def test_window_of_one_is_the_series(self):
        s = [3.0, 1.0, 4.0, 1.0, 5.0]
        assert rolling_mean(s, 1) == pytest.approx(s)

    def test_matches_naive_loop_on_the_real_price_series(self):
        """Oracle test against an explicit Python mean over the live daily
        closes — guards the cumulative-sum implementation against precision or
        indexing drift on the actual 5.9k-row series."""
        prices = np.asarray(M.price_prices, dtype=float)
        for w in (7, 30, 364, 1400):
            got = rolling_mean(prices, w)
            for i in (w - 1, w + 500, len(prices) - 1):
                want = float(np.mean(prices[i - w + 1:i + 1]))
                assert got[i] == pytest.approx(want, rel=1e-9)

    def test_daily_closes_not_weekly(self):
        """A '200 week' MA is a 1400-DAY mean of daily closes, not a mean of
        200 weekly closes. Window lengths are days, and the price series is
        contiguous daily, so one sample == one day."""
        assert MA_BY_VALUE["ma200w"].days == 200 * 7 == 1400
        assert MA_BY_VALUE["ma52w"].days == 52 * 7 == 364
        assert MA_BY_VALUE["ma30d"].days == 30
        assert MA_BY_VALUE["ma7d"].days == 7
        dates = np.asarray(M.price_dates, dtype="datetime64[D]")
        steps = np.diff(dates).astype(int)
        assert (steps == 1).all(), "price series is no longer contiguous daily"


class TestMATable:
    def test_values_and_order(self):
        assert MA_VALUES == ("ma200w", "ma52w", "ma30d", "ma7d")

    def test_legend_names(self):
        assert [w.legend for w in MA_WINDOWS] == [
            "200W MA", "52W MA", "30D MA", "7D MA"]

    def test_options_match_table(self):
        assert MA_OPTIONS == [{"label": w.label, "value": w.value}
                              for w in MA_WINDOWS]

    def test_windows_are_distinguished_by_dash_not_hue(self):
        """The operator is colourblind: shape carries the encoding, so all four
        share one colour and every dash pattern is distinct."""
        assert len({w.dash for w in MA_WINDOWS}) == len(MA_WINDOWS)
        assert MA_BY_VALUE["ma200w"].dash == "solid"


# ══════════════════════════════════════════════════════════════════════════
# Rendering
# ══════════════════════════════════════════════════════════════════════════

class TestRendering:
    def test_zero_selected_draws_nothing(self):
        assert _ma_traces(_fig(ma=[])) == []
        assert _ma_traces(_fig(ma=None)) == []
        assert _ma_traces(_fig()) == []

    def test_one_selected_draws_one(self):
        fig = _fig(ma=["ma200w"])
        traces = _ma_traces(fig)
        assert [t.name for t in traces] == ["200W MA"]

    def test_n_selected_draws_n_with_the_right_names(self):
        fig = _fig(ma=["ma7d", "ma200w"])
        assert {t.name for t in _ma_traces(fig)} == {"7D MA", "200W MA"}
        fig4 = _fig(ma=list(MA_VALUES))
        assert [t.name for t in _ma_traces(fig4)] == [
            "200W MA", "52W MA", "30D MA", "7D MA"]

    def test_unknown_value_is_ignored(self):
        fig = _fig(ma=["ma7d", "ma999y"])
        assert [t.name for t in _ma_traces(fig)] == ["7D MA"]

    def test_traces_are_lines_with_the_windows_dash(self):
        fig = _fig(ma=list(MA_VALUES))
        for w in MA_WINDOWS:
            tr = _ma(fig, w.legend)
            assert tr.mode == "lines"
            assert tr.line.dash == w.dash
        # one colour for all four — the encoding is shape, not hue
        assert len({_ma(fig, w.legend).line.color for w in MA_WINDOWS}) == 1

    def test_values_equal_the_rolling_mean_at_that_date(self):
        fig = _fig(ma=["ma30d"])
        tr = _ma(fig, "30D MA")
        want = rolling_mean(np.asarray(M.price_prices, float), 30)
        xs = np.asarray(M.price_years, float)
        for x, y in zip(np.asarray(tr.x, float), np.asarray(tr.y, float)):
            i = int(np.argmin(np.abs(xs - x)))
            assert xs[i] == pytest.approx(x)
            assert y == pytest.approx(want[i], rel=1e-9)

    def test_nothing_emitted_before_the_first_full_window(self):
        fig = _fig(ma=["ma200w"], xmin=2011)
        tr = _ma(fig, "200W MA")
        days = MA_BY_VALUE["ma200w"].days
        first_full_t = float(M.price_years[days - 1])
        assert float(np.min(tr.x)) >= first_full_t
        # and there is no NaN hole punched into the middle of the line
        assert np.isfinite(np.asarray(tr.y, float)).all()

    def test_clipped_to_the_x_range(self):
        fig = _fig(ma=["ma7d"], xmin=2018, xmax=2021)
        tr = _ma(fig, "7D MA")
        from btc_core import yr_to_t
        lo, hi = yr_to_t(2018, M.genesis), yr_to_t(2021, M.genesis)
        xs = np.asarray(tr.x, float)
        assert xs.min() >= lo and xs.max() <= hi
        # a narrower window really is narrower (the clip does something)
        wide = np.asarray(_ma(_fig(ma=["ma7d"]), "7D MA").x, float)
        assert wide.max() > xs.max()

    def test_downsampled_like_the_scatter(self):
        """Four full-resolution ~5.9k-point lines is needless wire payload, so
        the MA reuses the price scatter's stride rule verbatim (which, like the
        scatter, can land a little over _MAX_SCATTER_PTS on the last stride)."""
        from btc_core import yr_to_t
        fig = _fig(ma=list(MA_VALUES))
        xs = np.asarray(M.price_years, float)
        lo, hi = yr_to_t(2011, M.genesis), yr_to_t(2030, M.genesis)
        for w in MA_WINDOWS:
            raw = int(np.count_nonzero(
                np.isfinite(rolling_mean(np.asarray(M.price_prices, float),
                                         w.days))
                & (xs >= lo) & (xs <= hi)))
            stride = max(1, raw // _MAX_SCATTER_PTS)
            want = len(np.arange(0, raw, stride))
            assert len(_ma(fig, w.legend).x) == want
            assert want <= raw / 4, "not actually downsampled"

    def test_stack_multiplies_like_the_scatter(self):
        plain = _ma(_fig(ma=["ma7d"]), "7D MA")
        stacked = _ma(_fig(ma=["ma7d"], stack=2.0, show_stack=True), "7D MA")
        assert len(plain.y) == len(stacked.y)
        assert np.asarray(stacked.y, float) == pytest.approx(
            np.asarray(plain.y, float) * 2.0)

    def test_stack_ignored_when_show_stack_is_off(self):
        plain = _ma(_fig(ma=["ma7d"]), "7D MA")
        off = _ma(_fig(ma=["ma7d"], stack=2.0, show_stack=False), "7D MA")
        assert np.asarray(off.y, float) == pytest.approx(
            np.asarray(plain.y, float))

    def test_independent_of_show_data(self):
        """The MA is its own control — hiding the scatter must not hide it."""
        assert [t.name for t in _ma_traces(_fig(ma=["ma7d"], show_data=False))] \
            == ["7D MA"]

    def test_hover_carries_a_date(self):
        fig = _fig(ma=["ma7d"])
        tr = _ma(fig, "7D MA")
        assert tr.customdata is not None
        assert "2" in str(tr.customdata[0][0])   # a date string like 2018-03-04


# ══════════════════════════════════════════════════════════════════════════
# Time Machine — a trailing mean must stop at the frame date
# ══════════════════════════════════════════════════════════════════════════

class TestAsOfTruncation:
    def _t_D(self, idx):
        import pandas as pd
        import timemachine as tm
        return (pd.Timestamp(tm.frames()[idx]) - M.genesis).days / 365.25

    def test_no_ma_point_past_the_frame_date(self):
        import timemachine as tm
        if not tm.available():
            pytest.skip("timemachine grid not built")
        idx = len(tm.frames()) // 2
        t_D = self._t_D(idx)
        fig = _fig(ma=["ma7d", "ma30d"], asof_date=idx)
        traces = _ma_traces(fig)
        assert traces, "as-of view drew no MA at all"
        for tr in traces:
            xs = np.asarray(tr.x, float)
            assert len(xs) > 0
            assert xs.max() <= t_D + 1e-9, (
                f"{tr.name} leaks {xs.max() - t_D:.4f} yr of post-frame prices")

    def test_live_view_does_draw_past_that_date(self):
        """Non-vacuity: without the as-of frame the same MA extends beyond
        t_D, so the previous test is measuring the truncation and not an
        empty/short series."""
        import timemachine as tm
        if not tm.available():
            pytest.skip("timemachine grid not built")
        idx = len(tm.frames()) // 2
        t_D = self._t_D(idx)
        tr = _ma(_fig(ma=["ma7d"]), "7D MA")
        assert np.asarray(tr.x, float).max() > t_D

    def test_values_still_correct_inside_the_frame(self):
        import timemachine as tm
        if not tm.available():
            pytest.skip("timemachine grid not built")
        idx = len(tm.frames()) // 2
        tr = _ma(_fig(ma=["ma30d"], asof_date=idx), "30D MA")
        want = rolling_mean(np.asarray(M.price_prices, float), 30)
        xs = np.asarray(M.price_years, float)
        for x, y in zip(np.asarray(tr.x, float), np.asarray(tr.y, float)):
            i = int(np.argmin(np.abs(xs - x)))
            assert y == pytest.approx(want[i], rel=1e-9)


# ══════════════════════════════════════════════════════════════════════════
# Registries — each of these fails silently
# ══════════════════════════════════════════════════════════════════════════

class TestSnapshotRegistry:
    def test_entry_is_at_the_absolute_tail(self):
        """v4 links are index-addressed: a new field anywhere but the tail
        shifts every later field and corrupts already-shipped links."""
        assert _SNAPSHOT_CONTROLS[-1] == ("bub-ma", "value")

    def test_previous_tail_entries_kept_their_indices(self):
        assert _SNAPSHOT_CONTROLS[-3] == ("bub-occ-tail", "value")
        assert _SNAPSHOT_CONTROLS[-2] == ("bub-occ-window", "value")

    def test_checklist_options_registered_in_table_order(self):
        assert _CHECKLIST_OPTIONS["bub-ma"] == list(MA_VALUES)

    @pytest.mark.parametrize("sel", [
        [], ["ma200w"], ["ma7d"], ["ma52w", "ma30d"], list(MA_VALUES)])
    def test_bitmask_round_trip(self, sel):
        opts = _CHECKLIST_OPTIONS["bub-ma"]
        mask = _list_to_mask(sel, opts)
        assert isinstance(mask, int)
        assert _mask_to_list(mask, opts) == [v for v in opts if v in sel]

    def test_share_link_round_trip(self):
        enc = _encode_snapshot({"bub-ma:value": ["ma52w", "ma7d"]})
        dec = _decode_snapshot(enc)
        assert dec.get("bub-ma:value") == ["ma52w", "ma7d"]

    def test_share_link_round_trip_empty_selection(self):
        """An explicitly empty selection must survive: it differs from the
        default (200 week on), so the sparse diff has to emit it."""
        enc = _encode_snapshot({"bub-ma:value": []})
        assert _decode_snapshot(enc).get("bub-ma:value") == []

    def test_in_bubble_tab_controls(self):
        assert "bub-ma" in _TAB_CONTROLS["bubble"]

    def test_snapshot_default(self):
        """No MA is shown until the user asks for one (operator decision,
        2026-09-07). The chart is already dense; MAs are opt-in."""
        assert SNAPSHOT_DEFAULTS["bub-ma:value"] == []


class TestCacheKeyAlignment:
    def test_defaults_carry_the_key(self):
        assert bubble_defaults()["ma"] == []

    def test_runtime_params_carry_the_same_key(self):
        """The prewarm key and the callback key must have identical key sets or
        every first visit to Tab 1 misses the L1 cache."""
        from test_cache_key_alignment import _extract_kwargs
        assert "ma" in _extract_kwargs("_get_bubble_fig")
        assert "ma" in _extract_kwargs("_get_mc_bubble_fig")


class TestCallbackWiring:
    def _bubble_cb_source(self):
        path = (pathlib.Path(__file__).parent / "callbacks" / "charts"
                / "__init__.py")
        tree = ast.parse(path.read_text())
        return next(n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef)
                    and n.name == "update_bubble")

    def _decorator_ids(self):
        """(kind, component_id) for every Input/State on update_bubble, in
        declaration order — which is the order Dash maps them onto params."""
        fn = self._bubble_cb_source()
        dec = next(d for d in fn.decorator_list if isinstance(d, ast.Call))
        return [(a.func.id, a.args[0].value) for a in dec.args
                if isinstance(a, ast.Call) and isinstance(a.func, ast.Name)
                and a.func.id in ("Input", "State")]

    def test_registered_as_an_input(self):
        seq = self._decorator_ids()
        assert ("Input", "bub-ma") in seq

    def test_positionally_aligned_with_the_signature(self):
        """Dash maps decorator args → params by declaration order, so an Input
        inserted without moving the signature silently feeds every later param
        the previous one's value — and no test of the figure would notice."""
        fn = self._bubble_cb_source()
        seq = self._decorator_ids()
        params = [a.arg for a in fn.args.args]
        assert len(seq) == len(params), (len(seq), len(params))
        idx = [cid for _, cid in seq].index("bub-ma")
        assert params[idx] == "ma_sel"
        # spot-check the immediate neighbours kept their params
        assert params[idx - 1] == "toggles"
        assert params[idx + 1] == "bubble_toggles"

    def test_in_post_restore_triggers(self):
        """The restore short-circuit set must cover EVERY Input on the
        callback; a missing id makes the restore cascade rebuild the figure."""
        fn = self._bubble_cb_source()
        found = None
        for node in ast.walk(fn):
            if (isinstance(node, ast.Assign)
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == "_POST_RESTORE_TRIGGERS"):
                found = {e.value for e in node.value.elts
                         if isinstance(e, ast.Constant)}
        assert found is not None, "_POST_RESTORE_TRIGGERS assignment not found"
        assert "bub-ma" in found


class TestRestoreFastPath:
    def test_share_link_restores_the_moving_average(self):
        from restore_builder import _build_bubble_figure_from_state
        fig = _build_bubble_figure_from_state({"bub-ma:value": ["ma52w"]})
        assert fig is not None
        assert [t.name for t in _ma_traces(fig)] == ["52W MA"]

    def test_restore_without_the_field_uses_the_default(self):
        """A link predating bub-ma restores whatever the default currently is.

        Derived from SNAPSHOT_DEFAULTS rather than hardcoded, so it keeps
        stating the intent when the default changes (it went from
        ["ma200w"] to [] on 2026-09-07). The teeth are in
        test_share_link_restores_the_moving_average above, which proves a
        populated field really does draw its trace.
        """
        from restore_builder import _build_bubble_figure_from_state
        expected = [MA_BY_VALUE[v].legend
                    for v in SNAPSHOT_DEFAULTS["bub-ma:value"]]
        fig = _build_bubble_figure_from_state({})
        assert [t.name for t in _ma_traces(fig)] == expected


class TestLayout:
    def test_control_exists_with_the_agreed_options(self):
        from layout.bubble import _bubble_controls
        import dash
        found = []

        def walk(node):
            if isinstance(node, (list, tuple)):
                for c in node:
                    walk(c)
                return
            if getattr(node, "id", None) == "bub-ma":
                found.append(node)
            ch = getattr(node, "children", None)
            if ch is not None:
                walk(ch)

        walk(_bubble_controls())
        assert len(found) == 1, "expected exactly one bub-ma checklist"
        cl = found[0]
        assert cl.options == MA_OPTIONS
        assert cl.value == []

    def test_bub_toggles_untouched(self):
        """The MA control is deliberately separate: bub-toggles is bitmask
        encoded and order-sensitive.

        Asserted as "the shipped prefix is intact AND no MA value leaked in",
        not as equality with a fixed list. Equality was a stronger claim than
        this test's own intent: it also forbade unrelated legitimate appends,
        and it broke when F-12 appended "show_ucl" (a genuine share-link bug
        fix, with its own proof in test_checklist_option_coverage.py that all
        256 pre-existing masks keep their meaning). The invariant that matters
        here is that the MA windows live in their own control.
        """
        opts = _CHECKLIST_OPTIONS["bub-toggles"]
        assert opts[:8] == [
            "shade", "show_ols", "show_data", "show_today", "show_legend",
            "minor_grid", "chart_zoom", "show_halvings"], (
            "the order-sensitive prefix defines every share link ever issued")
        leaked = [v for v in opts if v.startswith("ma")]
        assert not leaked, f"MA windows leaked into bub-toggles: {leaked}"
