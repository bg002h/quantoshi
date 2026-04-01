"""Bubble + QR Overlay chart builder."""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Any

import _app_ctx
import theme
from btc_core import yr_to_t, today_t, fmt_price, UserModel
from tab_defaults import BUBBLE

from figures.common import (
    _INTERP_POINTS, _MAX_SCATTER_PTS, _QR_LINE_WIDTH, _SHADE_ALPHA,
    _OVERLAY_LINE_WIDTH,
    _TODAY_LINE_COLOR, _TODAY_LINE_WIDTH, _TODAY_LINE_OPACITY,
    _TODAY_GLOW_WIDTH, _TODAY_GLOW_OPACITY,
    _FONT_LEGEND, _FONT_TITLE, _FONT_SUBTITLE,
    _SANS_FONT, _FONT_TITLE_LG, _FONT_BODY_LG, _FONT_TICK_LG, _FONT_LEGEND_LG,
    _LOG_MINOR, _MC_LEGEND_POS,
    _get_palette, _build_thermal_colors, _fmt_q_label,
    _base_layout, _year_ticks, _price_tickvals,
    _apply_sans_typography, _apply_config_annotation, _apply_watermark, _add_date_hover,
    _HOVER_FMT_USD,
    _lerp_hex, _hex_alpha,
    _round_trace_data,
)


def _r2_suffix(mdl, q):
    """Return '  R²=X.XXXX' suffix if R² available for model at quantile q, else ''."""
    r2 = getattr(mdl, 'r2_per_quantile', {}).get(q)
    if r2 is not None:
        return f"  R\u00b2={r2:.4f}"
    return ""


def _build_symmetric_bands(sel_qs, price_cache, t_arr, model_color="#000000",
                            max_bands=2):
    """Build shaded band traces from symmetric quantile pairs.

    Pairs from outside in: (lowest, highest), (2nd lowest, 2nd highest).
    Outer band = lighter opacity, inner = darker.
    """
    if len(sel_qs) < 2:
        return []

    n = len(sel_qs)
    pairs = []
    for i in range(n // 2):
        lo_q = sel_qs[i]
        hi_q = sel_qs[n - 1 - i]
        if lo_q != hi_q and lo_q in price_cache and hi_q in price_cache:
            pairs.append((lo_q, hi_q))
    pairs = pairs[:max_bands]

    if not pairs:
        return []

    opacities = [0.08, 0.15] if len(pairs) >= 2 else [0.10]

    traces = []
    x = list(t_arr)
    for i, (lo_q, hi_q) in enumerate(pairs):
        alpha = opacities[i] if i < len(opacities) else opacities[-1]
        lo_p = price_cache[lo_q]
        hi_p = price_cache[hi_q]
        traces.append(go.Scatter(
            x=x, y=list(lo_p), mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        ))
        traces.append(go.Scatter(
            x=x, y=list(hi_p), mode="lines", line=dict(width=0),
            fill="tonexty",
            fillcolor=_hex_alpha(model_color, alpha),
            showlegend=False, hoverinfo="skip",
        ))

    return traces


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
    _xmin = float(p["xmin"])
    # When starting at/before 2010, begin at mid-2010 (first BTC price data)
    if _xmin <= 2010:
        _xmin = 2010.5
    t_lo = max(yr_to_t(_xmin, m.genesis), 0.01)
    t_hi = yr_to_t(p["xmax"], m.genesis)
    y_lo = float(p["ymin"])
    y_hi = float(p["ymax"])
    t_arr = np.linspace(max(t_lo, 0.1), t_hi, _INTERP_POINTS)
    t_arr = _round_trace_data(t_arr)

    stack = float(p.get("stack", 0)) if p.get("show_stack") else 0.0

    traces = []

    # ── shading between adjacent quantiles ───────────────────────────────────
    sel_qs = sorted([float(q) for q in (p.get("selected_qs") or [])])
    _thermal = _build_thermal_colors(sel_qs, palette)

    bub_active = "bub" in p.get("active_models", ["bub"])
    # If no quantiles selected but models are active, fall back to Q50%
    _fallback_q50 = not sel_qs and p.get("active_models")
    _default_mode = "advanced" not in (p.get("qs_mode") or [])
    if _fallback_q50:
        sel_qs = [0.5]
        _thermal = _build_thermal_colors(sel_qs, palette)

    if bub_active:
        # Pre-compute prices for all selected quantiles
        _price_cache = {}
        for q in sel_qs:
            if q in model.fits:
                _price_cache[q] = _round_trace_data(model.price_at(q, t_arr) * (stack if stack > 0 else 1))

        if p.get("shade") and len(sel_qs) >= 2:
            _model_color = _app_ctx.MODEL_TRACE_COLORS.get("bub", "#000000")
            traces.extend(_build_symmetric_bands(
                sel_qs, _price_cache, t_arr, model_color=_model_color))

    # ── historical price data (behind all lines) ───────────────────────────
    if p.get("show_data"):
        mask  = (m.price_years >= t_lo) & (m.price_years <= t_hi)
        x_sc  = m.price_years[mask]
        y_sc  = m.price_prices[mask] * (stack if stack > 0 else 1)
        d_sc  = [m.price_dates[i] for i in range(len(m.price_dates)) if mask[i]]
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
            marker=dict(color=scatter_colors, size=max(2, int(p.get("pt_size", BUBBLE["pt_size"]))),
                        opacity=float(p.get("pt_alpha", BUBBLE["pt_alpha"]))),
            hovertemplate=_HOVER_FMT_USD,
        ))

    if bub_active:
        # ── quantile lines (thermal palette + neon glow) ─────────────────────
        _bub_color = _app_ctx.MODEL_TRACE_COLORS.get("bub", "#DAA520")
        for q in sel_qs:
            if q not in _price_cache:
                continue
            prices = _price_cache[q]
            lbl = _fmt_q_label(q) + _r2_suffix(model, q)
            if stack > 0:
                lbl += f"  \u2192  {fmt_price(float(prices[-1]))}"
            # Opacity: Q50% = 1.0, Q5%/Q95% = 0.25, extrapolated beyond
            _dist = abs(q - 0.5) / 0.45  # 0 at Q50%, 1.0 at Q5%/Q95%
            _q_opacity = max(0.1, 1.0 - _dist * 0.75)  # floor at 0.1
            if _fallback_q50 and _default_mode:
                _q_opacity = 0.5
            traces.append(go.Scatter(
                x=list(t_arr), y=list(prices),
                mode="lines", name=lbl,
                line=dict(color=_bub_color, width=_QR_LINE_WIDTH),
                opacity=_q_opacity,
            ))

    # ── U1 user model: direct line through two points ───────────────────────
    um_data = p.get("user_model")
    if um_data and "u1" in p.get("active_models", []):
        um_slope = um_data["slope"]
        um_oq = um_data.get("own_quantile", 0.5)
        um_intercept = um_data["base_intercept"]  # exact, no float key lookup
        um_prices = _round_trace_data(10.0 ** (um_intercept + um_slope * np.log10(np.maximum(t_arr, 0.01))))
        if stack > 0:
            um_prices = _round_trace_data(np.asarray(um_prices) * stack)
        um_lbl = f"U\u2081  m={um_slope:.3f}  Q{um_oq*100:.2f}%"
        traces.append(go.Scatter(
            x=list(t_arr), y=list(um_prices),
            mode="lines", name=um_lbl,
            line=dict(color="#e67e22", width=3.0, dash="solid"),
            legendgroup="u1", legendgrouptitle_text=f"U\u2081  m={um_slope:.3f}",
        ))
        # Also draw standard quantile bands from the UserModel overlay loop below
        # (they'll be drawn at 1.5px via the normal overlay path)

    # ── alternative model overlays ────────────────────────────────────────────
    for model_key in p.get("active_models", []):
        if model_key == "bub":
            continue  # main BM drawn above, not as overlay
        if model_key == "u1":
            um_data2 = p.get("user_model")
            if not um_data2:
                continue
            mdl = UserModel.from_store_dict(um_data2)
            if not mdl:
                continue
        else:
            mdl = _app_ctx.PRICE_MODELS.get(model_key)
            if not mdl:
                continue
        if mdl.quantized:
            # For U1, only draw user-selected quantiles (same as other models).
            # The user's own line is already drawn above at 3px.
            overlay_qs = list(sel_qs)
            if model_key == "u1":
                # Skip own_quantile — already drawn as direct line above
                overlay_qs = [q for q in overlay_qs
                              if not (hasattr(mdl, 'own_quantile') and abs(q - mdl.own_quantile) < 0.005)]
            for q in overlay_qs:
                if q not in mdl.fits:
                    continue
                prices = _round_trace_data(mdl.price_at(q, t_arr) * (stack if stack > 0 else 1))
                col = mdl.colors.get(q, "#888888")
                lbl = f"{mdl.legend_name} {_fmt_q_label(q, '')}" + _r2_suffix(mdl, q)
                if stack > 0:
                    lbl += f"  \u2192  {fmt_price(float(prices[-1]))}"
                traces.append(go.Scatter(
                    x=list(t_arr), y=list(prices),
                    mode="lines", name=lbl,
                    line=dict(color=col, width=3.0 if (model_key == "u1" and hasattr(mdl, 'own_quantile') and abs(q - mdl.own_quantile) < 0.005) else _OVERLAY_LINE_WIDTH, dash=mdl.dash_style),
                    legendgroup=mdl.short_name,
                    legendgrouptitle_text=(
                        f"{mdl.legend_name}  m={mdl.fits[mdl.quantiles[0]]['slope']:.3f}"
                        if model_key == "u1" else mdl.legend_name
                    ),
                ))
            # Symmetric band shading for overlay model
            if p.get("shade") and len(overlay_qs) >= 2:
                _overlay_prices = {}
                for q in overlay_qs:
                    if q in mdl.fits:
                        _overlay_prices[q] = _round_trace_data(
                            mdl.price_at(q, t_arr) * (stack if stack > 0 else 1))
                _overlay_color = _app_ctx.MODEL_TRACE_COLORS.get(model_key, "#888888")
                traces.extend(_build_symmetric_bands(
                    sorted(_overlay_prices.keys()), _overlay_prices, t_arr,
                    model_color=_overlay_color))
        else:
            # Non-quantized model: single trajectory
            prices = mdl.price_at(0.5, t_arr)
            if stack > 0:
                prices = prices * stack
            lbl = mdl.legend_name + _r2_suffix(mdl, 0.5)
            if stack > 0:
                lbl += f"  \u2192  {fmt_price(float(np.asarray(prices)[-1]))}"
            prices = _round_trace_data(np.asarray(prices))
            traces.append(go.Scatter(
                x=list(t_arr), y=list(prices),
                mode="lines", name=lbl,
                line=dict(color=palette["non_quantized_model"], width=_OVERLAY_LINE_WIDTH, dash=mdl.dash_style),
                legendgroup=mdl.short_name,
            ))

        # ── composite/support/future for _CompositeModel overlays ─────
        if hasattr(mdl, "comp_by_n") and hasattr(mdl, "t_grid"):
            _EF_COMP_COLOR = "#D4A017"
            _EF_SUP_COLOR  = "#8B6914"
            mdl_t = np.asarray(mdl.t_grid)
            mdl_mask = (mdl_t >= t_lo) & (mdl_t <= t_hi)

            if p.get("show_sup") and hasattr(mdl, "support_plot"):
                sup_y = np.asarray(mdl.support_plot)[mdl_mask] * (stack if stack > 0 else 1)
                traces.append(go.Scatter(
                    x=list(mdl_t[mdl_mask]), y=list(sup_y),
                    mode="lines", name=f"{mdl.legend_name} support",
                    line=dict(color=_EF_SUP_COLOR, dash="dash", width=1.5),
                    opacity=0.9,
                    legendgroup=mdl.short_name,
                ))

            if p.get("show_comp"):
                n = int(p.get("n_future", BUBBLE["n_future"]))
                n = min(n, len(mdl.comp_by_n) - 1)
                comp_y = np.asarray(mdl.comp_by_n[n])[mdl_mask] * (stack if stack > 0 else 1)
                traces.append(go.Scatter(
                    x=list(mdl_t[mdl_mask]), y=list(comp_y),
                    mode="lines",
                    name=f"{mdl.legend_name} composite (N={n})  R\u00b2={mdl.bm_r2:.4f}",
                    line=dict(color=_EF_COMP_COLOR, width=2.0),
                    legendgroup=mdl.short_name,
                ))

    # ── scanner quantile lines ───────────────────────────────────────────────
    for sl in p.get("scanner_lines", []):
        mdl = _app_ctx.PRICE_MODELS.get(sl["model"])
        if not mdl:
            continue
        q = sl["q"]
        scan_prices = _round_trace_data(np.array([
            float(mdl.price_at(q, t)) for t in t_arr]))
        if stack > 0:
            scan_prices = scan_prices * stack
        nearest_q = min(mdl.quantiles, key=lambda qq: abs(qq - q)) if mdl.quantiles else q
        col = mdl.colors.get(nearest_q, "#ffd93d")
        traces.append(go.Scatter(
            x=list(t_arr), y=list(scan_prices),
            mode="lines",
            name=f"{mdl.legend_name} Q{q*100:.1f}%",
            line=dict(color=col, width=2, dash=mdl.dash_style),
            legendgroup=f"scan-{mdl.short_name}",
        ))

    # ── Unfairly Cheap Line ──────────────────────────────────────────────────
    if p.get("show_ucl"):
        p_ucl = 10.0 ** (_app_ctx.UCL_INTERCEPT + _app_ctx.UCL_SLOPE * np.log10(t_arr))
        if stack > 0:
            p_ucl = p_ucl * stack
        traces.append(go.Scatter(
            x=list(t_arr), y=list(p_ucl),
            mode="lines", name="Unfairly Cheap Line",
            line=dict(color="#ff6b6b", dash="dot", width=1.8),
            opacity=0.9,
        ))

    # ── OLS line ──────────────────────────────────────────────────────────────
    if p.get("show_ols"):
        p_ols = 10.0 ** (m.ols_intercept + m.ols_slope * np.log10(t_arr))
        if stack > 0:
            p_ols = p_ols * stack
        traces.append(go.Scatter(
            x=list(t_arr), y=list(p_ols),
            mode="lines", name=f"OLS  R\u00b2={m.ols_r2:.4f}" if hasattr(m, 'ols_r2') and m.ols_r2 else "OLS",
            line=dict(color="#888888", dash="dash", width=1.3),
            opacity=0.8,
        ))

    # ── bubble support ────────────────────────────────────────────────────────
    if bub_active and p.get("show_sup"):
        mask = (m.years_plot_bm >= t_lo) & (m.years_plot_bm <= t_hi)
        sup_y = m.support_bm[mask] * (stack if stack > 0 else 1)
        traces.append(go.Scatter(
            x=list(m.years_plot_bm[mask]), y=list(sup_y),
            mode="lines", name="Bubble support",
            line=dict(color=p.get("sup_color", BUBBLE["sup_color"]),
                      dash="dash", width=float(p.get("sup_lw", BUBBLE["sup_lw"]))),
            opacity=0.9,
        ))

    # ── bubble composite ──────────────────────────────────────────────────────
    if bub_active and p.get("show_comp"):
        n = int(p.get("n_future", BUBBLE["n_future"]))
        n = min(n, len(m.comp_by_n) - 1)
        mask = (m.years_plot_bm >= t_lo) & (m.years_plot_bm <= t_hi)
        comp_y = m.comp_by_n[n][mask] * (stack if stack > 0 else 1)
        traces.append(go.Scatter(
            x=list(m.years_plot_bm[mask]), y=list(comp_y),
            mode="lines",
            name=f"Bubble composite (N={n})  R\u00b2={m.bm_r2:.4f}",
            line=dict(color=p.get("comp_color", BUBBLE["comp_color"]),
                      width=float(p.get("comp_lw", BUBBLE["comp_lw"]))),
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
        if v >= 1e18:  return f"${v/1e18:.0f}Qi"
        if v >= 1e15:  return f"${v/1e15:.0f}Q"
        if v >= 1e12:  return f"${v/1e12:.0f}T"
        if v >= 1e9:   return f"${v/1e9:.0f}B"
        if v >= 1e6:   return f"${v/1e6:.0f}M"
        if v >= 1e3:   return f"${v/1e3:.0f}k"
        if v >= 1:     return f"${v:.0f}"
        if v >= 0.01:  return f"{v*100:.0f}\u00a2"
        return f"${v:.2f}"

    layout = _base_layout(
        title="Bitcoin Bubble Model + Quantile Regression Channels",
        xlabel="",
        ylabel="",
    )
    layout["xaxis"].update(
        range=[t_lo, t_hi],
        tickvals=tick_ts, ticktext=tick_lbls, tickangle=45,
        tickfont=dict(family="Arial Narrow, sans-serif-condensed, sans-serif"),
    )
    if p.get("yscale", BUBBLE["yscale"]) == "log":
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

    if p.get("xscale", BUBBLE["xscale"]) == "log":
        layout["xaxis"].update(
            type="log",
            range=[math.log10(max(t_lo, 1e-10)), math.log10(max(t_hi, 1e-10))],
        )
        # X-axis uses explicit tickvals (year labels); skip minor to avoid crash.

    layout["showlegend"] = bool(p.get("show_legend", BUBBLE["show_legend"]))
    leg_pos = p.get("legend_pos", BUBBLE["legend_pos"])
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
            showarrow=False, font=dict(size=_FONT_LEGEND, color=theme.TEXT_COLOR),
            bgcolor=theme.PLOT_BG_COLOR, bordercolor=theme.SPINE_COLOR, borderwidth=1,
        )]

    _apply_sans_typography(layout)
    fig = go.Figure(data=traces, layout=go.Layout(**layout))
    _add_date_hover(fig, m.genesis)
    _apply_config_annotation(fig, p, "bub", show_qr=True, show_mc=False)
    wm_pos = "bottom-left" if leg_pos == "bottom-right" else "bottom-right"
    _apply_watermark(fig, pos=wm_pos)
    return fig
