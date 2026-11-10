"""Raw MC, Task 4: the example-line sample and the two extreme paths.

``select_line_paths`` ranks the simulated paths by their price at the last
plotted step and picks evenly spaced ranks between the 2nd and 98th
percentile, so the drawn lines span the spread without the freak tails.
``_add_mc_extremes`` draws the highest and lowest of ALL the paths as dashed,
slightly bolder lines in the same dark neutral -- told apart from the solid
example lines by dash pattern and weight, never by hue.
"""
import numpy as np
import plotly.graph_objects as go
import pytest

import colors
from colors import (MC_EXTREME_ALPHA, MC_EXTREME_WIDTH, MC_PATH_COLOR,
                    MC_PATH_WIDTH, _hex_alpha)
from figures.bubble import (_add_mc_extremes, _add_mc_spaghetti,
                            _sig_round_list, _mc_path_steps,
                            select_line_paths)
from raw_mc import RAW_LINES_DEFAULT, RAW_LINES_MAX, RAW_LINES_MIN


def _paths(n, n_steps=12, shuffle=False, seed=0):
    """(n, n_steps) paths whose LAST column is arange(n) (optionally permuted).

    Earlier columns are random so a selector that looked at the wrong column
    (or at row order) would pick a visibly different set.
    """
    rng = np.random.default_rng(seed)
    p = rng.uniform(1.0, 1e6, size=(n, n_steps))
    last = np.arange(n, dtype=float) + 1.0
    if shuffle:
        last = rng.permutation(last)
    p[:, -1] = last
    return p


# ── select_line_paths ───────────────────────────────────────────────────────

@pytest.mark.parametrize("shuffle", [False, True])
def test_select_spreads_across_end_prices(shuffle):
    paths = _paths(1000, shuffle=shuffle)
    idx = select_line_paths(paths, 20)
    assert len(idx) == 20 and len(set(idx.tolist())) == 20
    ends = paths[idx, -1]
    ranks = ends - 1.0                      # last column is rank + 1
    assert np.all(np.diff(ranks) > 0), "picks run low -> high end price"
    assert 15 <= ranks[0] <= 25, f"lowest pick ~ the 2nd percentile, got {ranks[0]}"
    assert 975 <= ranks[-1] <= 985, f"highest pick ~ the 98th, got {ranks[-1]}"
    # evenly spaced: every gap within one rank of the mean gap
    gaps = np.diff(ranks)
    assert gaps.max() - gaps.min() <= 1.0 + 1e-9


def test_select_excludes_the_tails():
    paths = _paths(1000, shuffle=True)
    idx = select_line_paths(paths, 50)
    ends = np.sort(paths[:, -1])
    assert paths[idx, -1].min() > ends[9], "nothing from the bottom 1 %"
    assert paths[idx, -1].max() < ends[-10], "nothing from the top 1 %"
    assert paths[:, -1].argmax() not in idx and paths[:, -1].argmin() not in idx


def test_select_ranks_by_last_column_only():
    """Row order and the earlier columns must not matter."""
    a = _paths(300, shuffle=True, seed=1)
    b = a.copy()
    b[:, :-1] = np.random.default_rng(9).uniform(1, 9, size=b[:, :-1].shape)
    assert np.array_equal(a[select_line_paths(a, 20), -1],
                          b[select_line_paths(b, 20), -1])


def test_lines_exceeding_sims_draws_all():
    paths = _paths(10)
    idx = select_line_paths(paths, 20)
    assert sorted(idx.tolist()) == list(range(10))
    assert sorted(select_line_paths(paths, 10).tolist()) == list(range(10))


@pytest.mark.parametrize("n, k", [(100, 99), (50, 49), (25, 24), (11, 10),
                                  (100, 97), (100, 96), (100, 5), (7, 6)])
def test_select_returns_exactly_k_distinct_paths(n, k):
    """Just below the sim count the 2nd-98th window is narrower than k ranks;
    the selector must still hand back k DIFFERENT paths (a duplicate would be
    one line drawn twice and the line count would not match the control)."""
    idx = select_line_paths(_paths(n, shuffle=True), k)
    assert len(idx) == k
    assert len(set(idx.tolist())) == k
    assert idx.min() >= 0 and idx.max() < n


def test_select_degenerate_counts():
    paths = _paths(100, shuffle=True)
    assert len(select_line_paths(paths, 0)) == 0
    assert len(select_line_paths(paths, -3)) == 0
    one = select_line_paths(paths, 1)
    assert len(one) == 1
    assert 45 <= paths[one[0], -1] - 1.0 <= 55, "a lone line is the median path"
    assert len(select_line_paths(paths[:0], 20)) == 0
    assert len(select_line_paths(None, 20)) == 0


def test_select_is_deterministic():
    paths = _paths(500, shuffle=True)
    assert np.array_equal(select_line_paths(paths, 20),
                          select_line_paths(paths, 20))


def test_select_ties_are_stable():
    paths = np.ones((50, 4))                # every path ends at the same price
    idx = select_line_paths(paths, 10)
    assert len(set(idx.tolist())) == 10


def test_line_count_constants():
    assert (RAW_LINES_MIN, RAW_LINES_MAX, RAW_LINES_DEFAULT) == (5, 200, 20)


# ── _add_mc_extremes ────────────────────────────────────────────────────────

def test_extremes_traces():
    paths = _paths(200, shuffle=True)
    t = np.linspace(2025.0, 2036.0, paths.shape[1])
    fig = go.Figure()
    _add_mc_extremes(fig, paths, t)
    assert [tr.name for tr in fig.data] == ["Highest end", "Lowest end"]
    for tr in fig.data:
        assert tr.mode == "lines"
        assert tr.line.dash == "dash"
        assert tr.line.width == 1.6 == MC_EXTREME_WIDTH
        assert tr.line.color == _hex_alpha(MC_PATH_COLOR, MC_EXTREME_ALPHA)
        assert tr.hoverinfo == "skip"
        assert tr.meta == {"qs_keep_width": True}
        assert tr.showlegend is not False
    hi, lo = fig.data
    assert hi.y[-1] == pytest.approx(paths[:, -1].max(), rel=0.05)
    assert lo.y[-1] == pytest.approx(paths[:, -1].min(), rel=0.05)
    # and they ARE the rows with the max / min last column
    steps = _mc_path_steps(paths.shape[1])
    assert list(hi.y) == _sig_round_list(paths[paths[:, -1].argmax()][steps])
    assert list(lo.y) == _sig_round_list(paths[paths[:, -1].argmin()][steps])
    # same x grid as the example paths (x is shared, so it compresses away)
    assert np.allclose(np.asarray(hi.x), np.asarray(t, dtype=np.float32)[steps])
    assert np.array_equal(np.asarray(hi.x), np.asarray(lo.x))


def test_extremes_come_from_all_paths_not_the_selection():
    """The extremes are the most extreme of the whole set, i.e. precisely the
    paths select_line_paths leaves out."""
    paths = _paths(500, shuffle=True)
    fig = go.Figure()
    _add_mc_extremes(fig, paths, np.arange(paths.shape[1]))
    hi, lo = fig.data
    assert hi.y[-1] == pytest.approx(500.0, rel=0.01)    # last col max = 500
    assert lo.y[-1] == pytest.approx(1.0, rel=0.01)      # last col min = 1


def test_extremes_are_not_told_apart_by_hue():
    """Operator is colour-blind: one dark neutral, differing by dash + weight
    from the example lines."""
    r, g, b = colors._hex_to_rgb(MC_PATH_COLOR)
    assert max(r, g, b) - min(r, g, b) <= 8 and max(r, g, b) <= 0x60
    fig = go.Figure()
    _add_mc_spaghetti(fig, np.ones((3, 6)), np.arange(6), n_display=3)
    _add_mc_extremes(fig, _paths(10), np.arange(12))
    example, ext = fig.data[0], fig.data[-1]
    assert ext.line.dash == "dash" and example.line.dash in (None, "solid")
    assert ext.line.width > example.line.width == MC_PATH_WIDTH
    # the extremes are stronger than any example line, but the same hue
    assert float(ext.line.color.rstrip(")").split(",")[-1]) == MC_EXTREME_ALPHA
    assert ext.line.color.split(",")[:3] == example.line.color.split(",")[:3]


def test_extremes_noop_on_missing_or_empty_paths():
    for bad in (None, np.empty((0, 5)), np.empty((0,))):
        fig = go.Figure()
        _add_mc_extremes(fig, bad, np.arange(5))
        assert len(fig.data) == 0


def test_extremes_leave_existing_traces_alone():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2], y=[3, 4], name="keep"))
    before = fig.data[0].to_plotly_json()
    _add_mc_extremes(fig, _paths(20), np.arange(12))
    assert len(fig.data) == 3
    assert fig.data[0].to_plotly_json() == before


def test_extreme_constants():
    assert MC_EXTREME_WIDTH == 1.6
    assert MC_EXTREME_ALPHA == 0.6
    assert MC_EXTREME_WIDTH > MC_PATH_WIDTH


# ── hand-off to _add_mc_spaghetti ───────────────────────────────────────────

@pytest.mark.parametrize("n, k", [(1000, 20), (1000, 200), (200, 5),
                                  (10, 20), (60, 59)])
def test_spaghetti_receives_exact_selection(n, k):
    """Passing the chosen rows with n_display=len(idx) makes _add_mc_spaghetti's
    stride 1, so it draws exactly the selection, in rank order."""
    paths = _paths(n, shuffle=True)
    t = np.linspace(2025.0, 2036.0, paths.shape[1])
    idx = select_line_paths(paths, k)
    fig = go.Figure()
    _add_mc_spaghetti(fig, paths[idx], t, n_display=len(idx))
    assert len(fig.data) == len(idx) == min(k, n)
    steps = _mc_path_steps(paths.shape[1])
    for tr, i in zip(fig.data, idx):
        assert list(tr.y) == _sig_round_list(paths[i][steps])
