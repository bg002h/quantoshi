"""Tests for the Tab 1 halving-line toggle (show_halvings).

Covers the pure shape helper and its integration into build_bubble_figure.
"""
import pandas as pd

from conftest import M
from figures import build_bubble_figure
from figures.common import (
    _halving_line_shapes, _halving_annotations,
    _HALVING_EPOCHS, _build_halving_epochs,
)
from colors import (
    HALVING_LINE_COLOR, HALVING_PAST_OPACITY, HALVING_FUTURE_OPACITY,
)

_GENESIS = pd.Timestamp("2009-07-25")


def _bubble_p(show_halvings):
    return {
        "selected_qs": [0.5] if 0.5 in M.qr_fits else [],
        "xscale": "log", "yscale": "log",
        "xmin": 2012, "xmax": 2030, "ymin": -2, "ymax": 8,
        "shade": False, "show_ols": False, "show_data": False,
        "show_today": True, "show_halvings": show_halvings,
        "show_legend": False, "show_comp": False, "show_sup": False,
        "n_future": 0, "ptsize": 2, "ptalpha": 0.2,
        "stack": 1.0, "show_stack": False, "lots": [], "use_lots": False,
        "auto_y": False,
    }


class TestHalvingEpochs:
    def test_known_halvings_are_not_estimated(self):
        epochs = _build_halving_epochs()
        known = [dt for dt, est in epochs if not est]
        assert [dt.strftime("%Y-%m-%d") for dt in known] == [
            "2012-11-28", "2016-07-09", "2020-05-11", "2024-04-20"]

    def test_future_halvings_are_estimated_and_4yr_spaced(self):
        epochs = _build_halving_epochs()
        future = [dt for dt, est in epochs if est]
        assert future, "expected at least one estimated future halving"
        # First future halving = 2024 anchor + 4 years.
        assert future[0].strftime("%Y-%m-%d") == "2028-04-20"
        # Cadence is exactly 4 calendar years.
        assert future[1].year - future[0].year == 4
        # Generated out to the chart's max x-range, not past it.
        assert future[-1].year <= 2080


class TestHalvingLineShapes:
    def test_all_within_range(self):
        shapes = _halving_line_shapes(
            _HALVING_EPOCHS, _GENESIS, 0.0, 100.0, 1.0, 1e6,
            HALVING_LINE_COLOR, HALVING_PAST_OPACITY, HALVING_FUTURE_OPACITY,
            1.4)
        assert len(shapes) == len(_HALVING_EPOCHS)
        # Every shape is a vertical line spanning the given y-range.
        for s in shapes:
            assert s["type"] == "line"
            assert s["x0"] == s["x1"]
            assert s["y0"] == 1.0 and s["y1"] == 1e6
            assert s["line"]["color"] == HALVING_LINE_COLOR

    def test_past_solid_future_dashed(self):
        shapes = _halving_line_shapes(
            _HALVING_EPOCHS, _GENESIS, 0.0, 100.0, 1.0, 1e6,
            HALVING_LINE_COLOR, HALVING_PAST_OPACITY, HALVING_FUTURE_OPACITY,
            1.4)
        solids = [s for s in shapes if s["line"]["dash"] == "solid"]
        dashes = [s for s in shapes if s["line"]["dash"] == "dash"]
        # 4 known halvings are solid; the rest (estimated) are dashed.
        assert len(solids) == 4
        assert len(dashes) == len(shapes) - 4
        assert all(s["opacity"] == HALVING_PAST_OPACITY for s in solids)
        assert all(s["opacity"] == HALVING_FUTURE_OPACITY for s in dashes)

    def test_range_clipping_drops_offscreen(self):
        # Window covering only the 2020 + 2024 halvings (t ~ 10.8 .. 14.7).
        shapes = _halving_line_shapes(
            _HALVING_EPOCHS, _GENESIS, 10.0, 15.0, 1.0, 1e6,
            HALVING_LINE_COLOR, HALVING_PAST_OPACITY, HALVING_FUTURE_OPACITY,
            1.4)
        assert len(shapes) == 2


class TestHalvingAnnotations:
    def test_labels_are_years(self):
        annots = _halving_annotations(
            _HALVING_EPOCHS, _GENESIS, 0.5, 100.0, HALVING_LINE_COLOR)
        labels = [a["text"] for a in annots]
        assert "2024" in labels
        assert "2028" in labels
        # One label per in-range halving.
        assert len(annots) == len(_HALVING_EPOCHS)
        # Positioned in paper space so they render on log + linear x-axes.
        assert all(a["xref"] == "paper" and a["yref"] == "paper"
                   and a["showarrow"] is False for a in annots)
        assert all(0.0 <= a["x"] <= 1.0 for a in annots)

    def test_log_x_positions_differ_from_linear(self):
        lin = _halving_annotations(
            _HALVING_EPOCHS, _GENESIS, 0.5, 100.0, HALVING_LINE_COLOR,
            xlog=False)
        log = _halving_annotations(
            _HALVING_EPOCHS, _GENESIS, 0.5, 100.0, HALVING_LINE_COLOR,
            xlog=True)
        assert [a["text"] for a in lin] == [a["text"] for a in log]
        # Log axis compresses the x positions, so fractions differ.
        assert any(abs(a["x"] - b["x"]) > 1e-6 for a, b in zip(lin, log))
        assert all(0.0 <= a["x"] <= 1.0 for a in log)

    def test_range_clipping_drops_offscreen(self):
        annots = _halving_annotations(
            _HALVING_EPOCHS, _GENESIS, 10.0, 15.0, HALVING_LINE_COLOR)
        years = {a["text"] for a in annots}
        assert years == {"2020", "2024"}


class TestHalvingInBubbleFigure:
    @staticmethod
    def _halving_lines(fig):
        return [s for s in (fig.layout.shapes or [])
                if getattr(s.line, "color", None) == HALVING_LINE_COLOR]

    @staticmethod
    def _halving_labels(fig):
        # Halving labels are the only annotations drawn in the halving color.
        return [a.text for a in (fig.layout.annotations or [])
                if getattr(a.font, "color", None) == HALVING_LINE_COLOR]

    def test_off_adds_no_halving_shapes_or_labels(self):
        fig, _ = build_bubble_figure(M, _bubble_p(False))
        assert self._halving_lines(fig) == []
        assert self._halving_labels(fig) == []

    def test_on_adds_halving_lines_and_labels(self):
        fig, _ = build_bubble_figure(M, _bubble_p(True))
        assert self._halving_lines(fig), "show_halvings=True should add lines"
        labels = set(self._halving_labels(fig))
        assert "2024" in labels   # known halving in the 2012-2030 window
        assert "2028" in labels   # estimated future halving

    def test_coexists_with_today_line(self):
        # Today line (orange) and halving lines (indigo) both present.
        fig, _ = build_bubble_figure(M, _bubble_p(True))
        all_shapes = fig.layout.shapes or []
        halving = self._halving_lines(fig)
        assert len(all_shapes) > len(halving)  # today line(s) also present
