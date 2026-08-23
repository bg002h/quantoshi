"""Percentile-vs-time view for Tab 1.

For each quantized model, plot over time the percentile at which the actual BTC
price sat within that model's quantile fan (0-100%) — a model-relative
valuation / mean-reversion oscillator. Historical only (percentile needs a
realized price). Colorblind-safe shaded valuation zones sit behind the lines.
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

_CHEAP_MAX = 25.0   # below this percentile = "cheap" band
_RICH_MIN = 75.0    # above this percentile = "rich" band
_ZONE_ALPHA = 0.10  # background band opacity


def _percentile_series(mdl, t_data, px_data, sigma_mode="constant"):
    """Vectorized percentile of each actual price within the model's fan.

    Same log-space interpolation as ``mdl.find_percentile`` (edge-clamped to the
    outer quantiles), but computes each quantile's price for ALL dates at once
    (price_at is vectorized over t) instead of calling find_percentile per date
    — ~n_quantiles model evals total rather than n_dates × n_quantiles scalar
    ones. ``sigma_mode`` selects the band model (resqr = residual quantile bands,
    constant = legacy), so the oscillator matches the fan shown on the price
    chart. Returns percentiles in [0, 100], or None if the model has no fan.
    """
    qs = mdl.quantiles
    if not qs:
        return None
    qs_arr = np.asarray(qs, dtype=float)              # increasing
    t_safe = np.maximum(np.asarray(t_data, float), 0.5)
    # (n_q, n_t) log-price fan
    logfan = np.array([
        np.log10(np.maximum(
            np.asarray(mdl.price_at(q, t_safe, sigma_mode=sigma_mode), float), 1e-10))
        for q in qs])
    logpx = np.log10(np.maximum(np.asarray(px_data, float), 1e-10))
    pct = np.empty(logpx.shape[0], dtype=float)
    for j in range(pct.shape[0]):
        # np.interp clamps to qs_arr[0]/qs_arr[-1] outside the fan — matching
        # find_percentile's edge behavior.
        pct[j] = np.interp(logpx[j], logfan[:, j], qs_arr)
    return pct * 100.0


def build_percentile_figure(m: ModelData, p: dict[str, Any]) -> go.Figure:
    """Percentile of the actual price within each active quantized model's fan,
    over time. Non-quantized models (no fan) are skipped."""
    palette = _get_palette(p)

    _xmin = float(p["xmin"])
    if _xmin <= 2010:
        _xmin = 2010.5
    t_lo = max(yr_to_t(_xmin, m.genesis), 0.01)
    td = today_t(m.genesis)
    # Historical only: percentile needs a realized price, so cap the window at
    # today regardless of how far the range slider extends into the future.
    t_hi = min(yr_to_t(p["xmax"], m.genesis), td)

    mask = (m.price_years >= t_lo) & (m.price_years <= t_hi)
    t_data = m.price_years[mask]
    px_data = m.price_prices[mask]

    traces = []
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
        # Percentile is only meaningful for quantized (fan) models; skip s2f etc.
        if not getattr(mdl, "quantized", False):
            continue
        try:
            pct = _percentile_series(mdl, t_data, px_data,
                                     sigma_mode=p.get("sigma_mode", "constant"))
            if pct is None:
                continue
        except Exception:
            continue
        traces.append(go.Scatter(
            x=t_data, y=_round_trace_data(pct),
            mode="lines", name=mdl.legend_name,
            line=dict(color=_get_model_color(model_key, p),
                      width=TRACE_WIDTH_OVERLAY,
                      dash=getattr(mdl, "dash_style", "solid")),
        ))

    # ── colorblind-safe valuation zones + 50% median reference ────────────────
    shapes = [
        dict(type="rect", xref="paper", yref="y", x0=0, x1=1,
             y0=0, y1=_CHEAP_MAX, layer="below", line=dict(width=0),
             fillcolor=_hex_alpha(PCTILE_CHEAP_COLOR, _ZONE_ALPHA)),
        dict(type="rect", xref="paper", yref="y", x0=0, x1=1,
             y0=_RICH_MIN, y1=100, layer="below", line=dict(width=0),
             fillcolor=_hex_alpha(PCTILE_RICH_COLOR, _ZONE_ALPHA)),
        dict(type="line", xref="paper", yref="y", x0=0, x1=1, y0=50, y1=50,
             line=dict(color=FALLBACK_MODEL_GRAY, width=1.5, dash="dot")),
    ]
    if p.get("show_today") and t_lo <= td <= t_hi:
        shapes.extend(_today_line_shapes(
            td, 0, 1, palette.get("today_line", _TODAY_LINE_COLOR),
            glow=False, yref="paper"))

    tick_ts, tick_lbls = _year_ticks(p["xmin"], p["xmax"], m.genesis,
                                     minor_grid=p.get("minor_grid"))
    filtered = [(t, lbl) for t, lbl in zip(tick_ts, tick_lbls) if t_lo <= t <= t_hi]
    tick_ts, tick_lbls = (list(x) for x in zip(*filtered)) if filtered else ([], [])

    layout = _base_layout(
        title="Price percentile within model fan (valuation oscillator)",
        xlabel="", ylabel="")
    layout["xaxis"].update(range=[t_lo, t_hi], tickvals=tick_ts,
                           ticktext=tick_lbls, tickangle=45)
    layout["yaxis"].update(range=[0, 100], ticksuffix="%",
                           ticklabelposition="inside", ticklabelshift=-5)
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
    layout.setdefault("uirevision", _uirevision_key(p, "pctile"))
    fig = go.Figure(data=traces, layout=go.Layout(**layout))
    _apply_final_steps(
        fig, p, "bub",
        hover_fmt="<b>%{fullData.name}</b><br>percentile %{y:.1f}%<extra></extra>",
        show_qr=False, show_mc=False, wm_pos="bottom-right",
    )
    return fig
