"""Tab 1: the price-regime filter visibly changes the MC picture.

Free-tier MC on Tab 1 draws from 200 cached paths that every one visits every
regime, so the filter cannot re-simulate. Before 2026-11-10 it only re-ranked
the example lines (invisible at sims=200) and greyed the fan. Now (operator
request, 2026-11-10):

  * with a regime box unchecked, the fan is the percentiles of the
    MC_REGIME_FAN_PATHS best-matching cached paths (ranked by
    filter_paths_by_regime), labelled as such, and the example lines come
    from the same subset;
  * a one-line note under the regime checklist (Tab 1 only) says so, shown
    exactly when the server applies the filter (0 < checked < bins).
"""
import json
import shutil
import subprocess

import numpy as np
import pytest

from conftest import _app_ctx  # noqa: F401  (registers every callback)

from test_callbacks import TestUpdateBubbleCallbackMC as _MC
from test_mc_paths_toggle import _find

_NODE = shutil.which("node")
needs_node = pytest.mark.skipif(_NODE is None, reason="node not installed")


# ── figure builder ──────────────────────────────────────────────────────────

def test_fan_label_suffix_names_every_legend_entry():
    import plotly.graph_objects as go
    from figures.bubble import _add_mc_fan
    from colors import MC_AMBER, MC_PRICE_BAND_OUTER_ALPHA
    from colors import _hex_alpha
    t = np.linspace(20.0, 25.0, 12)
    paths = np.exp(np.random.default_rng(1).normal(11, 0.3, (40, 12)))
    fig = go.Figure()
    _add_mc_fan(fig, paths, t, label_suffix=" · best 50 of 200")
    named = {tr.name: tr for tr in fig.data if tr.name and tr.showlegend is not False}
    assert set(named) == {"MC 5–95% · best 50 of 200",
                          "MC 25–75% · best 50 of 200",
                          "MC median · best 50 of 200"}
    # amber: it is the (subset) result, not the gray all-regimes reference
    assert named["MC 5–95% · best 50 of 200"].fillcolor == \
        _hex_alpha(MC_AMBER, MC_PRICE_BAND_OUTER_ALPHA)


def test_fan_without_suffix_is_unchanged():
    import plotly.graph_objects as go
    from figures.bubble import _add_mc_fan
    t = np.linspace(20.0, 25.0, 12)
    paths = np.exp(np.random.default_rng(1).normal(11, 0.3, (40, 12)))
    fig = go.Figure()
    _add_mc_fan(fig, paths, t)
    assert {tr.name for tr in fig.data if tr.name} == {
        "MC 5–95%", "MC 25–75%", "MC median"}


# ── callback ────────────────────────────────────────────────────────────────

class TestCallback(_MC):
    """Reuses TestUpdateBubbleCallbackMC's _run / _named / _lines helpers.
    Its own tests are collected there, not re-run here (see below)."""

    def _ranked(self, blocked):
        from btc_core import yr_to_t
        from mc_overlay import filter_paths_by_regime
        allp = self._all_cached()
        t_axis = np.linspace(yr_to_t(self._START_YR, _app_ctx.M.genesis),
                             yr_to_t(self._START_YR + self._YEARS,
                                     _app_ctx.M.genesis),
                             allp.shape[1])
        return filter_paths_by_regime(allp, t_axis,
                                      _app_ctx.PRICE_MODELS["bub"], blocked,
                                      n_bins=5)

    @staticmethod
    def _suffix():
        from figures.bubble import MC_REGIME_FAN_PATHS
        return f" · best {MC_REGIME_FAN_PATHS} of 200"

    def test_subset_size_is_fifty(self):
        from figures.bubble import MC_REGIME_FAN_PATHS
        assert MC_REGIME_FAN_PATHS == 50

    def test_filter_draws_the_fan_from_the_best_matching_subset(self):
        from figures.bubble import MC_REGIME_FAN_PATHS
        fig = self._run(mc_sims=8, mc_regime=[0, 1])
        sub = self._ranked([2, 3, 4])[:MC_REGIME_FAN_PATHS]
        for base, pct in (("MC median", 50), ("MC 5–95%", 5),
                          ("MC 25–75%", 25)):
            y = np.asarray(self._named(fig, base + self._suffix()).y, dtype=float)
            np.testing.assert_allclose(
                y, np.percentile(sub, pct, axis=0)[:len(y)], rtol=1e-5)
        names = {t.name for t in fig.data if t.name}
        assert not any(n.startswith("MC ref") for n in names)
        assert not {"MC 5–95%", "MC 25–75%", "MC median"} & names

    def test_the_filter_moves_the_fan(self):
        """The reported bug: unchecking regimes changed nothing visible."""
        plain = self._run(mc_sims=8)
        filt = self._run(mc_sims=8, mc_regime=[0, 1])
        a = np.asarray(self._named(plain, "MC median").y, dtype=float)
        b = np.asarray(self._named(filt, "MC median" + self._suffix()).y,
                       dtype=float)
        assert np.max(np.abs(np.log(b / a))) > 0.05

    def test_example_lines_come_from_the_same_subset(self):
        """sims=200 with a filter draws the 50 subset paths, not all 200."""
        from figures.bubble import MC_REGIME_FAN_PATHS
        fig = self._run(mc_sims=200, mc_regime=[0, 1])
        lines = self._lines(fig)
        assert len(lines) == MC_REGIME_FAN_PATHS
        sub = self._ranked([2, 3, 4])[:MC_REGIME_FAN_PATHS]
        first = np.asarray(lines[0].y, dtype=float)
        # rounded to 2 sig figs, every other step: within 5% of a subset path
        from figures.bubble import _mc_path_steps
        steps = _mc_path_steps(sub.shape[1])[:len(first)]
        err = np.abs(np.log(sub[:, steps] / first)).max(axis=1)
        assert err.min() < 0.06

    def test_fewer_sims_than_the_subset_draws_the_best_of_it(self):
        fig = self._run(mc_sims=8, mc_regime=[0, 1])
        assert len(self._lines(fig)) == 8

    def test_paths_off_with_a_filter_draws_the_subset_fan_alone(self):
        fig = self._run(mc_sims=8, mc_regime=[0, 1], mc_paths=[])
        assert self._lines(fig) == []
        self._named(fig, "MC median" + self._suffix())

    @pytest.mark.parametrize("regime", [[0, 1, 2, 3, 4], []])
    def test_no_effective_filter_keeps_the_plain_fan(self, regime):
        """All checked, or all unchecked (treated as all allowed)."""
        fig = self._run(mc_sims=8, mc_regime=regime)
        self._named(fig, "MC median")


# Keep pytest from re-running every inherited TestUpdateBubbleCallbackMC test.
for _name in list(vars(_MC)):
    if _name.startswith("test_") and _name not in vars(TestCallback):
        setattr(TestCallback, _name, None)


# ── layout: the note ────────────────────────────────────────────────────────

_NOTE = "bub-mc-regime-note"


class TestNoteLayout:
    def test_tab_1_has_one_hidden_note_in_the_advanced_body(self):
        from layout.bubble import _bubble_controls
        adv = _find(_bubble_controls(), "bub-mc-adv-body", [])
        assert len(adv) == 1
        hits = _find(adv[0], _NOTE, [])
        assert len(hits) == 1
        assert hits[0].style == {"display": "none"}

    def test_note_says_what_the_filter_does(self):
        from layout.bubble import _bubble_controls
        from figures.bubble import MC_REGIME_FAN_PATHS
        note = _find(_bubble_controls(), _NOTE, [])[0]
        text = str(note.children)
        assert f"{MC_REGIME_FAN_PATHS} of 200" in text
        assert "not a new simulation" in text

    def test_tab_1_drops_the_generic_note_it_contradicts(self):
        """The shared note says 'percentiles unchanged' -- false on Tab 1 now."""
        from layout.bubble import _bubble_controls
        assert "percentiles unchanged" not in str(_bubble_controls())

    @pytest.mark.parametrize("prefix", ["dca", "ret", "hm", "sc", "cp"])
    def test_other_tabs_keep_the_generic_note_and_get_no_new_one(self, prefix):
        from layout.mc_controls import _mc_controls
        panel = _mc_controls(prefix)
        assert _find(panel, f"{prefix}-mc-regime-note", []) == []
        assert "percentiles unchanged" in str(panel)

    def test_placeholder_exists_when_mc_is_unavailable(self, monkeypatch):
        from layout import mc_controls
        monkeypatch.setattr(mc_controls._app_ctx, "_HAS_MARKOV", False)
        assert len(_find(mc_controls._mc_controls("bub", show_paths_toggle=True),
                         _NOTE, [])) == 1


# ── the note's clientside callback ──────────────────────────────────────────

def _entry():
    hits = [e for e in _app_ctx.app._callback_list
            if f"{_NOTE}.style" in str(e["output"])]
    assert len(hits) == 1, hits
    return hits[0]


class TestNoteCallback:
    def test_inputs_are_the_checklist_and_the_bin_count(self):
        e = _entry()
        assert [(i["id"], i["property"]) for i in e["inputs"]] == [
            ("bub-mc-regime", "value"), ("bub-mc-bins", "value")]

    def test_is_clientside(self):
        assert _entry()["clientside_function"]


def _js(cases):
    from layout.mc_controls import MC_REGIME_NOTE_JS
    script = ("const fn = (%s);\nconsole.log(JSON.stringify(%s.map("
              "c => fn(c[0], c[1]))));" % (MC_REGIME_NOTE_JS, json.dumps(cases)))
    res = subprocess.run([_NODE, "-e", script], capture_output=True,
                         text=True, timeout=30)
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout)


SHOW, HIDE = {}, {"display": "none"}


@needs_node
class TestNoteDecision:
    """Mirrors callbacks/mc_helpers.py: blocked only when 0 < checked < bins."""

    def test_some_unchecked_shows(self):
        assert _js([[[0, 1], 5], [[0, 1, 2, 3], 5], [[3], 7]]) == [SHOW] * 3

    def test_all_checked_hides(self):
        assert _js([[[0, 1, 2, 3, 4], 5], [list(range(7)), 7]]) == [HIDE] * 2

    def test_none_checked_hides(self):
        """All unchecked is treated as all allowed by the server."""
        assert _js([[[], 5]]) == [HIDE]

    def test_missing_values_hide(self):
        assert _js([[None, 5], [[0, 1], None], [None, None]]) == [HIDE] * 3

    def test_bin_count_may_arrive_as_a_string(self):
        assert _js([[[0, 1], "5"]]) == [SHOW]
