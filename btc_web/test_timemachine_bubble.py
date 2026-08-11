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


# ── QR (quantile regression) as-of, via the same overlay path as EPPL ─────────
_QR_BASE = dict(
    selected_qs=[0.001, 0.1, 0.25, 0.5], active_models=["qr"], sigma_mode="constant",
    xscale="log", yscale="log", xmin=2012, xmax=2030,
    ymin=1.0, ymax=1e8, show_data=True, shade=True,
)


def _qr_line_yvals(fig):
    """Concatenated y of all QR overlay LINE traces (legend_name 'QR')."""
    ys = []
    for tr in fig.data:
        if (tr.name or "").startswith("QR") and getattr(tr, "mode", None) == "lines":
            ys.extend(list(tr.y))
    return ys


def _qr_available():
    return tm.available() and "qr" in tm._load()["models"]


def test_asof_changes_the_qr_fan():
    """As-of QR (fit on data ≤ the earliest frame) yields a different quantile
    fan than the live full-data QR fit."""
    if not _qr_available():
        pytest.skip("timemachine grid without QR")
    live, _ = build_bubble_figure(M, {**_QR_BASE, "asof_date": None})
    past, _ = build_bubble_figure(M, {**_QR_BASE, "asof_date": 0})
    live_y = _qr_line_yvals(live)
    past_y = _qr_line_yvals(past)
    assert live_y and past_y
    assert live_y != past_y


def test_asof_qr_none_is_unchanged():
    """asof_date=None must match omitting the key for QR too (default path intact)."""
    if not _qr_available():
        pytest.skip("timemachine grid without QR")
    omitted, _ = build_bubble_figure(M, {**_QR_BASE})
    explicit_none, _ = build_bubble_figure(M, {**_QR_BASE, "asof_date": None})
    assert _qr_line_yvals(omitted) == _qr_line_yvals(explicit_none)


def test_asof_qr_draws_shaded_bands():
    """QR as-of draws its quantile lines AND shaded bands (the overlay path's
    _build_symmetric_bands), so 'qr + quantile bands' actually renders."""
    if not _qr_available():
        pytest.skip("timemachine grid without QR")
    fig, _ = build_bubble_figure(M, {**_QR_BASE, "asof_date": 0})
    qr_lines = [tr for tr in fig.data
                if (tr.name or "").startswith("QR") and getattr(tr, "mode", None) == "lines"]
    band_fills = [tr for tr in fig.data if getattr(tr, "fill", None) == "tonexty"]
    assert len(qr_lines) >= 2          # multiple quantile channels
    assert len(band_fills) >= 1        # at least one shaded band between them


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


# ── spl (Saturating Power Law) as-of, via the same overlay path as EPPL/QR ────
_SPL_BASE = dict(
    selected_qs=[0.1, 0.5, 0.9], active_models=["spl"], sigma_mode="constant",
    xscale="log", yscale="log", xmin=2012, xmax=2030,
    ymin=1.0, ymax=1e8, show_data=True, shade=True,
)


def _spl_line_yvals(fig):
    """Concatenated y of all spl overlay LINE traces (legend_name 'SatPL')."""
    ys = []
    for tr in fig.data:
        if (tr.name or "").startswith("SatPL") and getattr(tr, "mode", None) == "lines":
            ys.extend(list(tr.y))
    return ys


def _spl_available():
    return tm.available() and "spl" in tm._load()["models"]


def test_asof_changes_the_spl_fan():
    """As-of spl (fit on data <= the earliest frame) yields a different
    quantile fan than the live full-data spl fit."""
    if not _spl_available():
        pytest.skip("timemachine grid without spl")
    live, _ = build_bubble_figure(M, {**_SPL_BASE, "asof_date": None})
    past, _ = build_bubble_figure(M, {**_SPL_BASE, "asof_date": 0})
    live_y = _spl_line_yvals(live)
    past_y = _spl_line_yvals(past)
    assert live_y and past_y
    assert live_y != past_y


def test_asof_spl_none_is_unchanged():
    """asof_date=None must match omitting the key for spl too (default path intact)."""
    if not _spl_available():
        pytest.skip("timemachine grid without spl")
    omitted, _ = build_bubble_figure(M, {**_SPL_BASE})
    explicit_none, _ = build_bubble_figure(M, {**_SPL_BASE, "asof_date": None})
    assert _spl_line_yvals(omitted) == _spl_line_yvals(explicit_none)


def test_asof_spl_draws_shaded_bands():
    """spl as-of draws its quantile lines AND shaded bands (the overlay
    path's _build_symmetric_bands), so 'spl + quantile bands' actually
    renders."""
    if not _spl_available():
        pytest.skip("timemachine grid without spl")
    fig, _ = build_bubble_figure(M, {**_SPL_BASE, "asof_date": 0})
    spl_lines = [tr for tr in fig.data
                 if (tr.name or "").startswith("SatPL") and getattr(tr, "mode", None) == "lines"]
    band_fills = [tr for tr in fig.data if getattr(tr, "fill", None) == "tonexty"]
    assert len(spl_lines) >= 2          # multiple quantile channels
    assert len(band_fills) >= 1         # at least one shaded band between them


# ── logi (Logistic S-curve) as-of, via the same overlay path as EPPL/QR ───────
_LOGI_BASE = dict(
    selected_qs=[0.1, 0.5, 0.9], active_models=["logi"], sigma_mode="constant",
    xscale="log", yscale="log", xmin=2012, xmax=2030,
    ymin=1.0, ymax=1e8, show_data=True, shade=True,
)


def _logi_line_yvals(fig):
    """Concatenated y of all logi overlay LINE traces (legend_name 'Logi')."""
    ys = []
    for tr in fig.data:
        if (tr.name or "").startswith("Logi") and getattr(tr, "mode", None) == "lines":
            ys.extend(list(tr.y))
    return ys


def _logi_available():
    return tm.available() and "logi" in tm._load()["models"]


def test_asof_changes_the_logi_fan():
    """As-of logi (fit on data <= the earliest frame) yields a different
    quantile fan than the live full-data logi fit."""
    if not _logi_available():
        pytest.skip("timemachine grid without logi")
    live, _ = build_bubble_figure(M, {**_LOGI_BASE, "asof_date": None})
    past, _ = build_bubble_figure(M, {**_LOGI_BASE, "asof_date": 0})
    live_y = _logi_line_yvals(live)
    past_y = _logi_line_yvals(past)
    assert live_y and past_y
    assert live_y != past_y


def test_asof_logi_none_is_unchanged():
    """asof_date=None must match omitting the key for logi too (default path intact)."""
    if not _logi_available():
        pytest.skip("timemachine grid without logi")
    omitted, _ = build_bubble_figure(M, {**_LOGI_BASE})
    explicit_none, _ = build_bubble_figure(M, {**_LOGI_BASE, "asof_date": None})
    assert _logi_line_yvals(omitted) == _logi_line_yvals(explicit_none)


def test_asof_logi_draws_shaded_bands():
    """logi as-of draws its quantile lines AND shaded bands (the overlay
    path's _build_symmetric_bands), so 'logi + quantile bands' actually
    renders."""
    if not _logi_available():
        pytest.skip("timemachine grid without logi")
    fig, _ = build_bubble_figure(M, {**_LOGI_BASE, "asof_date": 0})
    logi_lines = [tr for tr in fig.data
                  if (tr.name or "").startswith("Logi") and getattr(tr, "mode", None) == "lines"]
    band_fills = [tr for tr in fig.data if getattr(tr, "fill", None) == "tonexty"]
    assert len(logi_lines) >= 2          # multiple quantile channels
    assert len(band_fills) >= 1          # at least one shaded band between them


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


def test_asof_r2_in_legend_and_evolves():
    """Time Machine legend shows per-frame R² for as-of models (computed vs data
    ≤ D — the same compute_model_r2 path as outside-TM), and it EVOLVES as the
    date sweeps. Matches the live R² at the right edge (D = today)."""
    if not _qr_available():
        pytest.skip("timemachine grid without QR")
    import re

    def qr_q50_r2(frame):
        fig, _ = build_bubble_figure(M, {**_QR_BASE, "asof_date": frame, "selected_qs": [0.5]})
        for t in fig.data:
            n = t.name or ""
            if n.startswith("QR") and "²" in n:
                return float(re.search("R²=([0-9.]+)", n).group(1))
        return None

    mid = qr_q50_r2(min(40, tm.n_frames() - 1))
    last = qr_q50_r2(tm.n_frames() - 1)
    assert mid is not None and last is not None    # R² present in the TM legend
    assert mid != last                              # evolves with the as-of date


def test_asof_future_r2_in_legend():
    """TM legend shows the 'future'/all-data R² (as-of model scored vs the FULL
    realized series) alongside the in-sample R², and the two converge at the
    right edge (D = today, where ≤D == all data)."""
    if not _qr_available():
        pytest.skip("timemachine grid without QR")
    import re

    def qr_group_title(frame):
        fig, _ = build_bubble_figure(M, {**_QR_BASE, "asof_date": frame, "selected_qs": [0.5]})
        for t in fig.data:
            g = getattr(t, "legendgrouptitle", None)
            if g and g.text and g.text.startswith("QR"):
                return g.text
        return None

    last = qr_group_title(tm.n_frames() - 1)
    assert last is not None and "all=" in last          # dual R² present
    fit, allr2 = (float(x) for x in re.findall(r"=([0-9.]+)", last)[:2])
    assert abs(fit - allr2) < 0.02                        # converge at the right edge
