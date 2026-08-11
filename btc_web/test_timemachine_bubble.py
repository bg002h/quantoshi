"""Task 7: as-of (Time Machine) substitution in the bubble figure.

The bubble builder resolves EPPL config overlays (``ecfg_*``) and the BM
primary from the precomputed as-of grid when ``p["asof_date"]`` is an int
frame index. When it is ``None`` (the default) behaviour is unchanged.

Imports are bare (``import timemachine``, ``from figures.bubble import ...``)
to match the runtime request path + the rest of the suite: bubble.py itself
does ``import timemachine`` / ``import _app_ctx`` bare, and ``btc_web.<mod>``
is a DISTINCT (empty) module object from the bare one the app populates.
"""
import pytest

import timemachine as tm
from figures.bubble import build_bubble_figure
from conftest import M


_BASE = dict(
    selected_qs=[0.5], active_models=["ecfg_1d_1u"], sigma_mode="constant",
    xscale="log", yscale="log", xmin=2012, xmax=2030,
    ymin=1.0, ymax=1e8, show_data=True,
)

# BM primary path: bub active, composite + support drawn from the grid.
_BM_BASE = dict(
    selected_qs=[0.5], active_models=["bub"], sigma_mode="constant",
    xscale="log", yscale="log", xmin=2011, xmax=2030,
    ymin=1.0, ymax=1e8, show_data=True, show_comp=True, show_sup=True, n_future=3,
)


def _ecfg_yvals(fig):
    """y-values of the ecfg_1d_1u overlay trace (legend_name == '1D_1U')."""
    return next(tr.y for tr in fig.data
                if "ECFG" in (tr.name or "").upper()
                or "1D_1U" in (tr.name or "").upper())


def _trace_by(fig, needle):
    """First trace whose name contains `needle` (case-insensitive)."""
    return next(tr for tr in fig.data if needle in (tr.name or "").lower())


def test_asof_changes_the_eppl_fan():
    """As-of frame 0 (fit on data ≤ 2012-04-01) must yield a different EPPL
    fan than the live full-data fit."""
    if not tm.available():
        pytest.skip("timemachine grid not built")
    live, _ = build_bubble_figure(M, {**_BASE, "asof_date": None})
    past, _ = build_bubble_figure(M, {**_BASE, "asof_date": 0})  # earliest frame
    live_y = _ecfg_yvals(live)
    past_y = _ecfg_yvals(past)
    assert live_y is not None and past_y is not None
    assert list(live_y) != list(past_y)


def test_asof_none_is_unchanged():
    """asof_date=None must be byte-identical to omitting the key entirely."""
    if not tm.available():
        pytest.skip("timemachine grid not built")
    omitted, _ = build_bubble_figure(M, {**_BASE})
    explicit_none, _ = build_bubble_figure(M, {**_BASE, "asof_date": None})
    assert list(_ecfg_yvals(omitted)) == list(_ecfg_yvals(explicit_none))


def test_asof_splits_realized_price_scatter():
    """In as-of mode the realized-price scatter is split into a solid segment
    (≤ frame date) and a faded, greyed 'Price after …' reveal segment."""
    if not tm.available():
        pytest.skip("timemachine grid not built")
    fig, _ = build_bubble_figure(M, {**_BASE, "asof_date": 0})
    names = [tr.name for tr in fig.data]
    assert "Price data" in names
    after = [tr for tr in fig.data if (tr.name or "").startswith("Price after")]
    assert len(after) == 1
    # faded segment date matches the frame date
    assert tm.frames()[0] in after[0].name
    # reveal must be visibly distinct: greyed marker + clearly lower opacity
    from colors import FALLBACK_MODEL_GRAY, SCATTER_POINT
    solid = _trace_by(fig, "price data")
    assert solid.marker.color == SCATTER_POINT
    assert after[0].marker.color == FALLBACK_MODEL_GRAY
    assert after[0].marker.color != solid.marker.color
    assert after[0].marker.opacity < solid.marker.opacity


def test_asof_bm_composite_and_support_come_from_grid():
    """BM primary as-of path (bub active): the composite + support are rebuilt
    from the grid frame, so they build without crashing and DIFFER (both y and
    the composite R² label) from the live full-data build."""
    if not tm.available():
        pytest.skip("timemachine grid not built")
    live, _ = build_bubble_figure(M, {**_BM_BASE, "asof_date": None})
    past, _ = build_bubble_figure(M, {**_BM_BASE, "asof_date": 0})  # earliest frame
    live_comp = _trace_by(live, "composite")
    past_comp = _trace_by(past, "composite")
    # built without crashing + non-empty
    assert live_comp.y is not None and len(past_comp.y) > 0
    # composite curve differs (comes from the as-of grid, not full data)
    assert list(live_comp.y) != list(past_comp.y)
    # R² label carries the frame's bm_r2, so the trace name differs too
    assert live_comp.name != past_comp.name
    # support line is also grid-sourced → differs
    live_sup = _trace_by(live, "support")
    past_sup = _trace_by(past, "support")
    assert list(live_sup.y) != list(past_sup.y)
