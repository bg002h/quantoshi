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


def _ecfg_yvals(fig):
    """y-values of the ecfg_1d_1u overlay trace (legend_name == '1D_1U')."""
    return next(tr.y for tr in fig.data
                if "ECFG" in (tr.name or "").upper()
                or "1D_1U" in (tr.name or "").upper())


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
    (≤ frame date) and a faded 'Price after …' reveal segment."""
    if not tm.available():
        pytest.skip("timemachine grid not built")
    fig, _ = build_bubble_figure(M, {**_BASE, "asof_date": 0})
    names = [tr.name for tr in fig.data]
    assert "Price data" in names
    after = [tr for tr in fig.data if (tr.name or "").startswith("Price after")]
    assert len(after) == 1
    # faded segment date matches the frame date
    assert tm.frames()[0] in after[0].name
