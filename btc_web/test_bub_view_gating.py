"""Tab-1 view gating: a hidden sub-view must not rebuild its figure.

Tab 1 has five view pills (Price / Forward CAGR / Residuals / Percentile /
Occupancy) selected via the ``bub-view-mode`` store.  Each view owns a figure
callback that takes ``Input("bub-view-mode", "data")`` plus the shared bubble
controls (x-range, model ticks, palette, ...).  Without a mode check every
x-range drag in Price view built up to four figures nobody can see.

The gate is safe *because* ``bub-view-mode`` is an Input: any switch into a
view re-fires that view's callback, so the visible figure is always freshly
built.  The test at the bottom pins that wiring so a future edit cannot drop
the trigger and leave a view showing a stale figure.
"""
import dash
import plotly.graph_objects as go
import pytest

from conftest import M, _app_ctx  # noqa: F401 — M/app must be initialised

from callbacks.charts import (update_bub_cagr, update_bub_resid,
                              update_bub_pctile, update_bub_occ)


# ── Positional call helpers (signatures mirror the @callback Input order) ────

def _cagr(view_mode, snapshot_pending=False):
    #  view_mode, first_render, sel_qs, adv_qs, xrange, toggles, xscale,
    #  yscale, model_show, legend_pos, fwd_yrs, palette, user_model,
    #  qs_mode (State), snapshot_pending (State)
    return update_bub_cagr(view_mode, 1, None, [0.5], [2015, 2025], [], "log",
                           "log", ["bub"], "outside", 1, "default", None,
                           ["advanced"], snapshot_pending)


def _resid(view_mode, snapshot_pending=False):
    #  view_mode, xrange, toggles, xscale, model_show, bub_toggles, n_future,
    #  bm_commit, legend_pos, palette, decomp_model, decomp_components,
    #  lppl_n_freqs, lppl_weighted, lppl_no_13, user_model, snapshot_pending
    #  (BM only draws a residual trace when a bubble-toggle selects one)
    return update_bub_resid(view_mode, [2015, 2025], [], "log", ["bub"],
                            ["show_comp"], 3, 0, "outside", "default",
                            "", [], [], [], [], None, snapshot_pending)


def _pctile(view_mode, snapshot_pending=False):
    #  view_mode, xrange, toggles, xscale, model_show, legend_pos, palette,
    #  sigma_mode, user_model, snapshot_pending
    return update_bub_pctile(view_mode, [2015, 2025], [], "log", ["bub"],
                             "outside", "default", "constant", None,
                             snapshot_pending)


def _occ(view_mode, snapshot_pending=False):
    #  view_mode, xrange, toggles, xscale, model_show, legend_pos, palette,
    #  sigma_mode, occ_tail, occ_window, user_model, snapshot_pending
    return update_bub_occ(view_mode, [2015, 2025], [], "log", ["bub"],
                          "outside", "default", "constant", 10, 4, None,
                          snapshot_pending)


# builder, own mode, a couple of modes it must refuse to build in
_VIEWS = [
    pytest.param(_cagr, "cagr", id="cagr"),
    pytest.param(_resid, "resid", id="resid"),
    pytest.param(_pctile, "percentile", id="percentile"),
    pytest.param(_occ, "occupancy", id="occupancy"),
]

_ALL_MODES = ("price", "cagr", "resid", "percentile", "occupancy")


class TestHiddenViewsSkipTheBuild:
    @pytest.mark.parametrize("call,mode", _VIEWS)
    def test_snapshot_pending_still_wins(self, call, mode):
        # Pre-existing gate — must stay ahead of the view-mode check
        # (spec 2026-04-24-single-redraw-per-snapshot).
        assert call(mode, snapshot_pending=True) is dash.no_update

    @pytest.mark.parametrize("call,mode", _VIEWS)
    def test_other_view_modes_return_no_update(self, call, mode):
        for other in _ALL_MODES:
            if other == mode:
                continue
            assert call(other) is dash.no_update, other

    @pytest.mark.parametrize("call,mode", _VIEWS)
    def test_own_view_mode_builds_a_figure(self, call, mode):
        fig = call(mode)
        assert isinstance(fig, go.Figure), type(fig)
        assert len(fig.data) > 0


class TestViewModeStaysAnInput:
    """The gate is only safe while bub-view-mode triggers each callback.

    Demote it to a State (or drop it) and a hidden view would keep its stale
    figure after the user switches into it — silently, on every chart.
    """

    _GRAPHS = ("bub-cagr-graph", "bub-resid-graph", "bub-pctile-graph",
               "bub-occ-graph")

    def test_all_four_view_callbacks_take_view_mode_as_input(self):
        # Module-level @callback registrations live in dash's
        # GLOBAL_CALLBACK_MAP until the server starts; app.callback_map only
        # holds app.* registrations.  (Same lookup as test_bub_deep_links.)
        from dash._callback import GLOBAL_CALLBACK_MAP
        cm = {**GLOBAL_CALLBACK_MAP, **_app_ctx.app.callback_map}
        for gid in self._GRAPHS:
            key = f"{gid}.figure"
            assert key in cm, f"{key} not registered"
            inputs = [(i["id"], i["property"]) for i in cm[key]["inputs"]]
            assert ("bub-view-mode", "data") in inputs, (gid, inputs)
