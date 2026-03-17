"""Bubble + QR Overlay chart builder."""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Any

import _app_ctx
from btc_core import yr_to_t, today_t, fmt_price

from figures.common import (
    _INTERP_POINTS, _MAX_SCATTER_PTS, _QR_LINE_WIDTH, _SHADE_ALPHA,
    _NON_QUANTIZED_MODEL_COLOR, _OVERLAY_LINE_WIDTH,
    _TODAY_LINE_COLOR, _TODAY_LINE_WIDTH, _TODAY_LINE_OPACITY,
    _TODAY_GLOW_WIDTH, _TODAY_GLOW_OPACITY,
    _FONT_LEGEND, _FONT_TITLE, _FONT_SUBTITLE,
    _SANS_FONT, _FONT_TITLE_LG, _FONT_BODY_LG, _FONT_TICK_LG, _FONT_LEGEND_LG,
    _LOG_MINOR, _MC_LEGEND_POS,
    _get_palette, _build_thermal_colors, _add_glow_trace, _fmt_q_label,
    _dark_layout, _year_ticks, _price_tickvals,
    _apply_sans_typography, _apply_config_annotation, _apply_watermark,
    _lerp_hex, _hex_alpha,
)


def build_bubble_figure(m: ModelData, p: dict[str, Any]) -> go.Figure:
    """
    p keys: selected_qs, shade, xscale, yscale, xmin, xmax, ymin, ymax,
            n_future, show_comp, comp_color, comp_lw,
            show_sup, sup_color, sup_lw,
            show_ols, show_data, show_today, pt_size, pt_alpha,
            stack, show_stack, lots (list of lot dicts), use_lots
    """
    model = _app_ctx.DEFAULT_MODEL
    palette = _get_palette(p)
    t_lo = max(yr_to_t(p["xmin"], m.genesis), 0.01)
    t_hi = yr_to_t(p["xmax"], m.genesis)
    y_lo = float(p["ymin"])
    y_hi = float(p["ymax"])
    t_arr = np.linspace(max(t_lo, 0.1), t_hi, _INTERP_POINTS)

    stack = float(p.get("stack", 0)) if p.get("show_stack") else 0.0

    traces = []

    # ── shading between adjacent quantiles ───────────────────────────────────
    sel_qs = sorted([float(q) for q in (p.get("selected_qs") or [])])

    # Pre-compute prices for all selected quantiles
    _price_cache = {}
    for q in sel_qs:
        if q in model.fits:
            _price_cache[q] = model.price_at(q, t_arr) * (stack if stack > 0 else 1)

    if p.get("shade") and len(sel_qs) >= 2:
        for j in range(len(sel_qs) - 1):
            if sel_qs[j] not in _price_cache or sel_qs[j+1] not in _price_cache:
                continue
            lo_p = _price_cache[sel_qs[j]]
            hi_p = _price_cache[sel_qs[j+1]]
            col  = model.colors.get(sel_qs[j], "#888888")
            traces.append(go.Scatter(
                x=list(t_arr), y=list(lo_p),
                mode="lines", line=dict(width=0),
                showlegend=False, hoverinfo="skip",
            ))
            traces.append(go.Scatter(
                x=list(t_arr), y=list(hi_p),
                mode="lines", line=dict(width=0), fill="tonexty",
                fillcolor=_hex_alpha(col, _SHADE_ALPHA),
                showlegend=False, hoverinfo="skip",
            ))

    # ── quantile lines (thermal palette + neon glow) ─────────────────────────
    _thermal = _build_thermal_colors(sel_qs, palette)
    for q in sel_qs:
        if q not in _price_cache:
            continue
        prices = _price_cache[q]
        lbl = _fmt_q_label(q)
        if stack > 0:
            lbl += f"  \u2192  {fmt_price(float(prices[-1]))}"
        col = _thermal.get(q, model.colors.get(q, "#888888"))
        _add_glow_trace(traces, t_arr, prices, col)
        traces.append(go.Scatter(
            x=list(t_arr), y=list(prices),
            mode="lines", name=lbl,
            line=dict(color=col, width=_QR_LINE_WIDTH),
        ))

    # ── alternative model overlays ────────────────────────────────────────────
    for model_key in p.get("active_models", []):
        mdl = _app_ctx.PRICE_MODELS.get(model_key)
        if not mdl:
            continue
        if mdl.quantized:
            for q in sel_qs:
                if q not in mdl.fits:
                    continue
                prices = mdl.price_at(q, t_arr) * (stack if stack > 0 else 1)
                col = mdl.colors.get(q, "#888888")
                lbl = f"{mdl.name} {_fmt_q_label(q, '')}"
                if stack > 0:
                    lbl += f"  \u2192  {fmt_price(float(prices[-1]))}"
                traces.append(go.Scatter(
                    x=list(t_arr), y=list(prices),
                    mode="lines", name=lbl,
                    line=dict(color=col, width=_OVERLAY_LINE_WIDTH, dash=mdl.dash_style),
                    legendgroup=mdl.short_name,
                    legendgrouptitle_text=mdl.name,
                ))
        else:
            # Non-quantized model: single trajectory
            prices = mdl.price_at(0.5, t_arr)
            if stack > 0:
                prices = prices * stack
            lbl = mdl.name
            if stack > 0:
                lbl += f"  \u2192  {fmt_price(float(np.asarray(prices)[-1]))}"
            traces.append(go.Scatter(
                x=list(t_arr), y=list(np.asarray(prices)),
                mode="lines", name=lbl,
                line=dict(color=_NON_QUANTIZED_MODEL_COLOR, width=_OVERLAY_LINE_WIDTH, dash=mdl.dash_style),
                legendgroup=mdl.short_name,
            ))

    # ── OLS line ──────────────────────────────────────────────────────────────
    if p.get("show_ols"):
        p_ols = 10.0 ** (m.ols_intercept + m.ols_slope * np.log10(t_arr))
        if stack > 0:
            p_ols = p_ols * stack
        traces.append(go.Scatter(
            x=list(t_arr), y=list(p_ols),
            mode="lines", name="OLS",
            line=dict(color="#888888", dash="dash", width=1.3),
            opacity=0.8,
        ))

    # ── bubble support ────────────────────────────────────────────────────────
    if p.get("show_sup"):
        mask = (m.years_plot_bm >= t_lo) & (m.years_plot_bm <= t_hi)
        traces.append(go.Scatter(
            x=list(m.years_plot_bm[mask]), y=list(m.support_bm[mask]),
            mode="lines", name="Bubble support",
            line=dict(color=p.get("sup_color", "#888888"),
                      dash="dash", width=float(p.get("sup_lw", 1.5))),
            opacity=0.9,
        ))

    # ── bubble composite ──────────────────────────────────────────────────────
    if p.get("show_comp"):
        n = int(p.get("n_future", 0))
        n = min(n, len(m.comp_by_n) - 1)
        mask = (m.years_plot_bm >= t_lo) & (m.years_plot_bm <= t_hi)
        traces.append(go.Scatter(
            x=list(m.years_plot_bm[mask]), y=list(m.comp_by_n[n][mask]),
            mode="lines",
            name=f"Bubble composite (N={n})  R\u00b2={m.bm_r2:.4f}",
            line=dict(color=p.get("comp_color", "#FFD700"),
                      width=float(p.get("comp_lw", 2.0))),
        ))

    # ── historical price data ─────────────────────────────────────────────────
    if p.get("show_data"):
        mask  = (m.price_years >= t_lo) & (m.price_years <= t_hi)
        x_sc  = m.price_years[mask]
        y_sc  = m.price_prices[mask] * (stack if stack > 0 else 1)
        d_sc  = [m.price_dates[i] for i in range(len(m.price_dates)) if mask[i]]
        # Downsample to <=1 200 points — imperceptible on log scale but cuts
        # figure JSON ~50 % and serialisation time meaningfully.
        _MAX_PTS = _MAX_SCATTER_PTS
        n_pts = len(x_sc)
        if n_pts > _MAX_PTS:
            stride = max(1, n_pts // _MAX_PTS)
            idx   = np.arange(0, n_pts, stride)
            x_sc  = x_sc[idx]
            y_sc  = y_sc[idx]
            d_sc  = [d_sc[i] for i in idx]
        # Temporal gradient: old data muted gray -> recent data warm amber
        n_sc = len(x_sc)
        scatter_colors = [_lerp_hex("#4a5568", "#f7931a", i / max(n_sc - 1, 1))
                          for i in range(n_sc)]
        traces.append(go.Scatter(
            x=list(x_sc), y=list(y_sc),
            mode="markers", name="Price data",
            marker=dict(color=scatter_colors, size=max(2, int(p.get("pt_size", 3))),
                        opacity=float(p.get("pt_alpha", 0.6))),
            text=d_sc, hovertemplate="%{text}<br>%{y:$,.0f}<extra></extra>",
        ))

    # ── LEO lot markers ───────────────────────────────────────────────────────
    lots = p.get("lots") or []
    if p.get("use_lots") and lots:
        lt_vals, lp_vals, lhover = [], [], []
        for lot in lots:
            try:
                lt = (pd.Timestamp(lot["date"]) - m.genesis).days / 365.25
                lp = float(lot["price"]) * (stack if stack > 0 else 1)
                if t_lo <= lt <= t_hi:
                    lt_vals.append(lt)
                    lp_vals.append(lp)
                    lhover.append(
                        f"{lot['date']}<br>Q{lot['pct_q']*100:.1f}%<br>"
                        f"{lot['btc']:.4f} BTC @ {fmt_price(lot['price'])}")
            except Exception:
                pass
        if lt_vals:
            traces.append(go.Scatter(
                x=lt_vals, y=lp_vals, mode="markers", name="Lots",
                marker=dict(color="#FFD700", size=10,
                            line=dict(color="#333333", width=0.7)),
                text=lhover,
                hovertemplate="%{text}<extra></extra>",
            ))

    # ── today line (with glow shadow) ─────────────────────────────────────────
    shapes = []
    if p.get("show_today"):
        td = today_t(m.genesis)
        today_color = palette.get("today_line", _TODAY_LINE_COLOR)
        if t_lo <= td <= t_hi:
            # Glow shadow behind the today line
            shapes.append(dict(
                type="line", x0=td, x1=td, y0=y_lo, y1=y_hi,
                line=dict(color=today_color, width=_TODAY_GLOW_WIDTH),
                opacity=_TODAY_GLOW_OPACITY, yref="y",
            ))
            shapes.append(dict(
                type="line", x0=td, x1=td, y0=y_lo, y1=y_hi,
                line=dict(color=today_color, dash="dash", width=_TODAY_LINE_WIDTH),
                opacity=_TODAY_LINE_OPACITY, yref="y",
            ))

    # ── x-axis ticks (calendar years) ─────────────────────────────────────────
    tick_ts, tick_lbls = _year_ticks(p["xmin"], p["xmax"], m.genesis,
                                     minor_grid=p.get("minor_grid"))
    filtered = [(t, lbl) for t, lbl in zip(tick_ts, tick_lbls) if t_lo <= t <= t_hi]
    tick_ts, tick_lbls = (list(x) for x in zip(*filtered)) if filtered else ([], [])

    # ── y-axis ticks (log price) ──────────────────────────────────────────────
    maj = _price_tickvals(y_lo, y_hi)

    def _fmt_y(price_val):
        v = price_val * stack if stack > 0 else price_val
        return fmt_price(v)

    ylabel = "Stack Value (USD)" if stack > 0 else "Bitcoin Price (USD)"

    layout = _dark_layout(
        m,
        title="Bitcoin Bubble Model + Quantile Regression Channels",
        xlabel="Years since genesis (2009-01-03)",
        ylabel=ylabel,
    )
    layout["xaxis"].update(
        range=[t_lo, t_hi],
        tickvals=tick_ts, ticktext=tick_lbls, tickangle=-45,
    )
    if p.get("yscale", "log") == "log":
        y_log_update = dict(
            type="log",
            range=[math.log10(max(y_lo, 1e-10)), math.log10(max(y_hi, 1e-10))],
        )
        if p.get("minor_grid"):
            # Plotly.js crashes when minor + explicit tickvals are combined.
            # Use auto-ticks so minor gridlines render safely.
            y_log_update["dtick"] = 1  # decade labels only (drop 2x and 5x)
            y_log_update["minor"] = _LOG_MINOR
        else:
            y_log_update["tickvals"] = maj
            y_log_update["ticktext"] = [_fmt_y(v) for v in maj]
        layout["yaxis"].update(y_log_update)
    else:
        layout["yaxis"].update(range=[y_lo, y_hi])

    if p.get("xscale", "linear") == "log":
        layout["xaxis"].update(
            type="log",
            range=[math.log10(max(t_lo, 1e-10)), math.log10(max(t_hi, 1e-10))],
        )
        # X-axis uses explicit tickvals (year labels); skip minor to avoid crash.

    layout["showlegend"] = bool(p.get("show_legend", True))
    leg_pos = p.get("legend_pos", "outside")
    if leg_pos != "outside" and leg_pos in _MC_LEGEND_POS:
        pos = _MC_LEGEND_POS[leg_pos]
        layout["legend"].update(
            x=pos["x"], y=pos["y"],
            xanchor=pos["xanchor"], yanchor=pos["yanchor"],
            bgcolor="rgba(255,255,255,0.7)",
        )
    layout["shapes"] = shapes

    if stack > 0:
        layout["annotations"] = [dict(
            text=f"Stack: {p['stack']:.6g} BTC",
            xref="paper", yref="paper", x=0.99, y=0.01,
            xanchor="right", yanchor="bottom",
            showarrow=False, font=dict(size=_FONT_LEGEND, color=m.TEXT_COLOR),
            bgcolor=m.PLOT_BG_COLOR, bordercolor=m.SPINE_COLOR, borderwidth=1,
        )]

    _apply_sans_typography(layout)
    fig = go.Figure(data=traces, layout=go.Layout(**layout))
    _apply_config_annotation(fig, p, "bub", show_qr=True, show_mc=False)
    wm_pos = "bottom-left" if leg_pos == "bottom-right" else "bottom-right"
    _apply_watermark(fig, pos=wm_pos)
    return fig
