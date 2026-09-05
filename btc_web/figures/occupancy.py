"""Occupancy view for Tab 1.

Share of a trailing window that BTC's actual price spent in the *tails* of a
model's quantile fan — above Q(100-tail) and below Q(tail) — plotted over time,
against the nominal share. Answers "where in time did the tail days sit?" —
e.g. is BTC spending more of the recent halving cycle below Q10 than it did six
years ago? A thin "when" strip at the top marks the individual tail days.

Builds on the per-day percentile series the Percentile view already computes
(same model fan, same sigma mode). Historical only.

Calibration note: for a model fit on ALL data (QR exactly, BM/resqr bands
approximately) the all-history share above Q90 is ~10% by construction, so a
trailing window running above nominal now must be balanced by a window below
nominal elsewhere. The chart is about *where* the tail days are, not whether
they exist.
"""
from __future__ import annotations

import math
import numpy as np
import plotly.graph_objects as go
from typing import Any

import _app_ctx
from btc_core import yr_to_t, today_t, ModelData, UserModel
from colors import (_hex_alpha, PLOT_BG_COLOR, FALLBACK_MODEL_GRAY,
                    MC_LEGEND_BG_ALPHA, PCTILE_CHEAP_COLOR, PCTILE_RICH_COLOR)

from figures.common import (
    TRACE_WIDTH_OVERLAY,
    _TODAY_LINE_COLOR,
    _get_palette, _get_model_color,
    _base_layout, _year_ticks,
    _apply_sans_typography,
    _apply_final_steps, _today_line_shapes, _empty_state_annotation,
    _round_trace_data, _MC_LEGEND_POS,
)
from figures.percentile import _percentile_series

# Paper-fraction vertical domains: main panel below, "when" strip on top. The
# strip sits at the TOP because the watermark + x tick labels own the bottom.
_MAIN_DOMAIN = (0.0, 0.90)
_STRIP_DOMAIN = (0.93, 1.0)
_STRIP_BELOW_Y = 1.0   # y2 row for days <= Q(tail)
_STRIP_ABOVE_Y = 2.0   # y2 row for days >= Q(100-tail)
_STRIP_MARKER_SIZE = 9


def _occupancy_series(t, pct, tail, window):
    """Trailing-window share (percent) of samples in each tail.

    For every sample i, the window is (t[i] - window, t[i]]. Returns
    ``(t_out, above, below)`` restricted to samples that have a FULL window
    behind them (t >= t[0] + window), so early values are not computed over a
    shorter, noisier span. ``above`` counts pct >= 100 - tail, ``below`` counts
    pct <= tail (both inclusive). ``t`` must be ascending.
    """
    t = np.asarray(t, dtype=float)
    pct = np.asarray(pct, dtype=float)
    n = t.shape[0]
    empty = np.empty(0, dtype=float)
    if n == 0:
        return empty, empty, empty
    hi = pct >= (100.0 - tail)
    lo = pct <= tail
    cum_hi = np.concatenate(([0], np.cumsum(hi)))
    cum_lo = np.concatenate(([0], np.cumsum(lo)))
    idx = np.arange(n)
    j = np.searchsorted(t, t - window, side="right")   # first index inside window
    count = (idx - j + 1).astype(float)
    above = (cum_hi[idx + 1] - cum_hi[j]) / count * 100.0
    below = (cum_lo[idx + 1] - cum_lo[j]) / count * 100.0
    full = t >= t[0] + window - 1e-9
    return t[full], above[full], below[full]


def build_occupancy_figure(m: ModelData, p: dict[str, Any]) -> go.Figure:
    """Trailing-window tail occupancy per active quantized model + when-strip."""
    palette = _get_palette(p)
    tail = int(p.get("occ_tail", 10) or 10)
    tail = min(max(tail, 1), 49)
    window = float(p.get("occ_window", 4) or 4)
    q_hi = 100 - tail

    _xmin = float(p["xmin"])
    if _xmin <= 2010:
        _xmin = 2010.5
    t_lo = max(yr_to_t(_xmin, m.genesis), 0.01)
    td = today_t(m.genesis)
    # Historical only: occupancy needs realized prices.
    t_hi = min(yr_to_t(p["xmax"], m.genesis), td)

    # Occupancy is computed over the FULL history and only displayed for the
    # x-range, so narrowing the range slider never changes the values.
    hist = (m.price_years > 0) & (m.price_years <= td)
    t_all = np.asarray(m.price_years[hist], dtype=float)
    px_all = np.asarray(m.price_prices[hist], dtype=float)
    order = np.argsort(t_all, kind="stable")
    t_all, px_all = t_all[order], px_all[order]
    disp = (t_all >= t_lo) & (t_all <= t_hi)

    traces = []
    strip_drawn = False
    for model_key in p.get("active_models", []):
        if model_key == "u1":
            um = p.get("user_model")
            mdl = UserModel.from_store_dict(um) if um else None
            if not mdl:
                continue
        else:
            mdl = _app_ctx.PRICE_MODELS.get(model_key)
            if not mdl:
                continue
        if not getattr(mdl, "quantized", False):
            continue
        try:
            pct = _percentile_series(mdl, t_all, px_all,
                                     sigma_mode=p.get("sigma_mode", "constant"))
            if pct is None:
                continue
        except Exception:
            continue
        t_out, above, below = _occupancy_series(t_all, pct, tail, window)
        sel = (t_out >= t_lo) & (t_out <= t_hi)
        color = _get_model_color(model_key, p)
        name = mdl.legend_name
        traces.append(go.Scatter(
            x=t_out[sel], y=_round_trace_data(above[sel]),
            mode="lines", name=f"{name} ≥Q{q_hi}",
            line=dict(color=color, width=TRACE_WIDTH_OVERLAY),
        ))
        traces.append(go.Scatter(
            x=t_out[sel], y=_round_trace_data(below[sel]),
            mode="lines", name=f"{name} ≤Q{tail}",
            line=dict(color=color, width=TRACE_WIDTH_OVERLAY, dash="dash"),
        ))
        if not strip_drawn:
            # "When" strip for the first model only (keeps it one strip tall).
            strip_drawn = True
            for cond, y_row, col, lbl in (
                (pct >= q_hi, _STRIP_ABOVE_Y, PCTILE_RICH_COLOR, f"≥Q{q_hi}"),
                (pct <= tail, _STRIP_BELOW_Y, PCTILE_CHEAP_COLOR, f"≤Q{tail}"),
            ):
                days = t_all[cond & disp]
                traces.append(go.Scatter(
                    x=days, y=np.full(days.shape, y_row),
                    mode="markers", yaxis="y2",
                    name=f"{name} days {lbl}", showlegend=False,
                    marker=dict(symbol="line-ns", size=_STRIP_MARKER_SIZE,
                                color=col, line=dict(width=1, color=col)),
                    hovertemplate=("<b>%{fullData.name}</b><br>%{customdata[0]}"
                                   "<extra></extra>"),
                ))

    if traces:
        traces.append(go.Scatter(
            x=[t_lo, t_hi], y=[tail, tail], mode="lines",
            name=f"nominal {tail}%",
            line=dict(color=FALLBACK_MODEL_GRAY, width=1.5, dash="dot"),
            hovertemplate=(f"nominal {tail}%: a model fit on all data puts "
                           f"~{tail}% of ALL days above Q{q_hi} (and below "
                           f"Q{tail}) by construction<extra></extra>"),
        ))

    shapes = []
    if p.get("show_today") and t_lo <= td <= t_hi:
        shapes.extend(_today_line_shapes(
            td, 0, 1, palette.get("today_line", _TODAY_LINE_COLOR),
            glow=False, yref="paper"))

    tick_ts, tick_lbls = _year_ticks(p["xmin"], p["xmax"], m.genesis,
                                     minor_grid=p.get("minor_grid"))
    filtered = [(t, lbl) for t, lbl in zip(tick_ts, tick_lbls) if t_lo <= t <= t_hi]
    tick_ts, tick_lbls = (list(x) for x in zip(*filtered)) if filtered else ([], [])

    layout = _base_layout(
        title=(f"Share of trailing {window:g} yr spent above Q{q_hi} / "
               f"below Q{tail} of the model fan"),
        xlabel="", ylabel="")
    layout["xaxis"].update(range=[t_lo, t_hi], tickvals=tick_ts,
                           ticktext=tick_lbls, tickangle=45)
    layout["yaxis"].update(rangemode="tozero", ticksuffix="%",
                           ticklabelposition="inside", ticklabelshift=-5,
                           domain=list(_MAIN_DOMAIN))
    layout["yaxis2"] = dict(
        domain=list(_STRIP_DOMAIN), range=[0.4, 2.6], anchor="x",
        tickvals=[_STRIP_BELOW_Y, _STRIP_ABOVE_Y],
        ticktext=[f"≤Q{tail}", f"≥Q{q_hi}"],
        ticklabelposition="inside", ticklabelshift=-5,
        showgrid=False, zeroline=False, showline=False, fixedrange=True,
        tickfont=dict(size=9), ticks="",
    )
    layout["shapes"] = shapes
    layout["showlegend"] = bool(p.get("show_legend", True))
    leg_pos = p.get("legend_pos", "outside")
    if leg_pos != "outside" and leg_pos in _MC_LEGEND_POS:
        pos = _MC_LEGEND_POS[leg_pos]
        layout.setdefault("legend", {})
        layout["legend"].update(
            x=pos["x"], y=pos["y"], xanchor=pos["xanchor"], yanchor=pos["yanchor"],
            bgcolor=_hex_alpha(PLOT_BG_COLOR, MC_LEGEND_BG_ALPHA))
    if p.get("xscale") == "log":
        layout["xaxis"].update(
            type="log",
            range=[math.log10(max(t_lo, 1e-10)), math.log10(max(t_hi, 1e-10))])

    if not traces:
        _empty_state_annotation(layout)

    _apply_sans_typography(layout)
    from figures.common import _uirevision_key
    layout.setdefault("uirevision", _uirevision_key(p, "occ"))
    fig = go.Figure(data=traces, layout=go.Layout(**layout))
    _apply_final_steps(
        fig, p, "bub",
        hover_fmt=("<b>%{fullData.name}</b><br>%{customdata[0]}<br>"
                   f"%{{y:.1f}}% of trailing {window:g} yr<extra></extra>"),
        show_qr=False, show_mc=False, wm_pos="bottom-right",
    )
    return fig
