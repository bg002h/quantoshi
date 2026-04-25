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


# ════════════════════════════════════════════════════════════════════════════
# Phase 2 (2026-04-25): _build_dca_figure_from_state tests.
# ════════════════════════════════════════════════════════════════════════════

class TestBuildDcaFigureFromState:
    def test_dca_basic_returns_figure(self):
        """Minimal state with bub model enabled — builder returns a figure with
        ≥1 quantile trace. (DCA only draws traces when "bub" is in
        active_models; defaults are empty until user toggles model show.)"""
        from restore_builder import _build_dca_figure_from_state
        state = {
            "main-tabs:active_tab": "dca",
            "dca-amount:value": 100,
            "dca-yr-range:value": [2024, 2034],
            "dca-qs:value": [0.5],
            "dca-model-show:value": ["bub"],
        }
        fig = _build_dca_figure_from_state(state)
        assert fig is not None, "builder returned None for non-MC non-SC-live state"
        # Convert to dict if Figure object
        fig_dict = fig.to_dict() if hasattr(fig, "to_dict") else fig
        assert "data" in fig_dict
        # At least one trace whose name matches quantile pattern (e.g., "Q50%", "Q1%")
        import re
        q_re = re.compile(r"Q\d")
        has_q = any(q_re.search(str(t.get("name", ""))) for t in fig_dict["data"])
        assert has_q, (
            f"no quantile trace in figure data: "
            f"{[t.get('name') for t in fig_dict['data']]}"
        )

    def test_dca_mc_enabled_returns_none(self):
        """dca-mc-enable=['yes'] (decoded list, not bitmask int) — return None."""
        from restore_builder import _build_dca_figure_from_state
        state = {
            "main-tabs:active_tab": "dca",
            "dca-mc-enable:value": ["yes"],
            "dca-amount:value": 100,
        }
        fig = _build_dca_figure_from_state(state)
        assert fig is None, "MC-enabled snapshot must fall back to cascade path"

    def test_dca_sc_live_returns_none(self):
        """sc-enable + sc-entry-mode=live — return None, no exception, no
        attempt to resolve btc-price-store from state."""
        from restore_builder import _build_dca_figure_from_state
        state = {
            "main-tabs:active_tab": "dca",
            "dca-sc-enable:value": ["yes"],
            "dca-sc-entry-mode:value": "live",
            "dca-amount:value": 100,
        }
        # Must not raise — even though btc-price-store key is absent.
        fig = _build_dca_figure_from_state(state)
        assert fig is None, "Saylor-live snapshot must fall back to cascade path"

    def test_dca_sc_custom_returns_figure(self):
        """sc-enable + sc-entry-mode=custom + custom price — returns figure."""
        from restore_builder import _build_dca_figure_from_state
        state = {
            "main-tabs:active_tab": "dca",
            "dca-sc-enable:value": ["yes"],
            "dca-sc-entry-mode:value": "custom",
            "dca-sc-custom-price:value": 50000.0,
            "dca-sc-loan:value": 100000.0,
            "dca-amount:value": 100,
        }
        fig = _build_dca_figure_from_state(state)
        assert fig is not None, "SC-custom snapshot should produce a figure"

    def test_dca_with_lots_returns_figure(self):
        """Snapshot with _lots + dca-use-lots=['yes'] — figure builds without
        error (lots-resolution mirrors bubble builder)."""
        from restore_builder import _build_dca_figure_from_state
        state = {
            "main-tabs:active_tab": "dca",
            "dca-use-lots:value": ["yes"],
            "dca-amount:value": 100,
            "dca-model-show:value": ["bub"],  # required for traces to be drawn
            "_lots": [
                {"date": "2020-01-01", "btc": 0.5, "price": 7000.0,
                 "pct_q": 0.5, "label": "test"},
            ],
        }
        fig = _build_dca_figure_from_state(state)
        assert fig is not None
        fig_dict = fig.to_dict() if hasattr(fig, "to_dict") else fig
        # At least one trace expected when bub model is shown
        assert len(fig_dict.get("data", [])) > 0, "expected ≥1 trace with bub model"


# ════════════════════════════════════════════════════════════════════════════
# Phase 2 ship 2 (2026-04-25): _build_retire_figure_from_state tests.
# ════════════════════════════════════════════════════════════════════════════

class TestBuildRetireFigureFromState:
    def test_retire_basic_returns_figure(self):
        """Minimal state with bub model — returns a figure with ≥1 quantile trace."""
        from restore_builder import _build_retire_figure_from_state
        state = {
            "main-tabs:active_tab": "retire",
            "ret-wd:value": 5000,
            "ret-yr-range:value": [2031, 2050],
            "ret-qs:value": ["median"],
            "ret-model-show:value": ["bub"],
        }
        fig = _build_retire_figure_from_state(state)
        assert fig is not None
        fig_dict = fig.to_dict() if hasattr(fig, "to_dict") else fig
        import re
        q_re = re.compile(r"Q\d")
        has_q = any(q_re.search(str(t.get("name", ""))) for t in fig_dict["data"])
        assert has_q, (
            f"no quantile trace: {[t.get('name') for t in fig_dict['data']]}"
        )

    def test_retire_mc_enabled_returns_none(self):
        """ret-mc-enable=['yes'] — return None, fall back to cascade."""
        from restore_builder import _build_retire_figure_from_state
        state = {
            "main-tabs:active_tab": "retire",
            "ret-mc-enable:value": ["yes"],
            "ret-wd:value": 5000,
        }
        fig = _build_retire_figure_from_state(state)
        assert fig is None, "MC-enabled retire snapshot must fall back"

    def test_retire_with_lots_returns_figure(self):
        """Snapshot with _lots + ret-use-lots=['yes'] — figure builds."""
        from restore_builder import _build_retire_figure_from_state
        state = {
            "main-tabs:active_tab": "retire",
            "ret-use-lots:value": ["yes"],
            "ret-wd:value": 5000,
            "ret-model-show:value": ["bub"],
            "_lots": [
                {"date": "2020-01-01", "btc": 0.5, "price": 7000.0,
                 "pct_q": 0.5, "label": "test"},
            ],
        }
        fig = _build_retire_figure_from_state(state)
        assert fig is not None
        fig_dict = fig.to_dict() if hasattr(fig, "to_dict") else fig
        assert len(fig_dict.get("data", [])) > 0

    def test_retire_default_quantiles(self):
        """No ret-qs set — uses default quantile resolution without crashing."""
        from restore_builder import _build_retire_figure_from_state
        state = {
            "main-tabs:active_tab": "retire",
            "ret-wd:value": 5000,
            "ret-model-show:value": ["bub"],
        }
        fig = _build_retire_figure_from_state(state)
        assert fig is not None
