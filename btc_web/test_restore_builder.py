"""Unit tests for restore_builder._build_bubble_figure_from_state.

Verifies the helper produces correct figures for representative
snapshot state dicts. The helper is the load-bearing component for
fast share-link restore (see memory/restore_callback_architecture.md).
"""
from __future__ import annotations

import sys
from pathlib import Path

import plotly.graph_objects as go
import pytest

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))


@pytest.fixture(scope="module", autouse=True)
def _init_app():
    """Initialize Dash app context so lazy imports inside the helper work."""
    import os
    os.environ["DEV"] = "1"
    import app  # noqa: F401 — registers callbacks; populates _app_ctx.app


def test_default_state_returns_figure():
    """An empty state dict (all defaults) should produce a valid go.Figure."""
    from restore_builder import _build_bubble_figure_from_state
    fig = _build_bubble_figure_from_state({})
    assert isinstance(fig, go.Figure)
    # Default sigma_mode and config produce traces.
    assert len(fig.data) > 0, "default-state figure has no traces"


def test_cta_active_returns_none():
    """When snapshot encodes CTA on, helper falls back to existing path."""
    from restore_builder import _build_bubble_figure_from_state
    state = {"cta-active:value": ["yes"]}
    assert _build_bubble_figure_from_state(state) is None


def test_lots_resolution_from_state():
    """Helper reads lots from state['_lots'], not from effective-lots:data
    (which doesn't exist in a snapshot dict)."""
    from restore_builder import _build_bubble_figure_from_state
    state = {
        "bub-use-lots:value": ["yes"],
        "_lots": [
            {"date": "2020-01-01", "btc": 0.5, "price": 8000.0, "notes": ""},
        ],
    }
    fig = _build_bubble_figure_from_state(state)
    assert isinstance(fig, go.Figure)
    # If lots resolution worked, the lot marker traces should be present.
    # The figure builder draws lot markers when use_lots=True and lots
    # is non-empty. We don't assert specific trace count (depends on
    # other defaults), just that a figure is produced.


def test_non_default_state_produces_figure():
    """Non-default widget values should propagate through the helper
    without crashing or returning None (when CTA isn't active)."""
    from restore_builder import _build_bubble_figure_from_state
    state = {
        "bub-xscale:value": "linear",
        "bub-yscale:value": "log",
        "bub-toggles:value": ["shade", "show_data"],
        "bub-n-future:value": 5,
        "bub-stack:value": 0.5,
    }
    fig = _build_bubble_figure_from_state(state)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0


def test_palette_from_state():
    """palette-store:data is read from the decoded state dict directly."""
    from restore_builder import _build_bubble_figure_from_state
    state = {"palette-store:data": "cb_brian"}
    fig = _build_bubble_figure_from_state(state)
    assert isinstance(fig, go.Figure)


def test_helper_is_pure():
    """Calling the helper twice with the same input produces the same figure."""
    from restore_builder import _build_bubble_figure_from_state
    state = {"bub-xscale:value": "linear", "bub-yscale:value": "log"}
    fig1 = _build_bubble_figure_from_state(state)
    fig2 = _build_bubble_figure_from_state(state)
    assert isinstance(fig1, go.Figure) and isinstance(fig2, go.Figure)
    # Same number of traces.
    assert len(fig1.data) == len(fig2.data)
