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
from colors import (
    SCATTER_POINT, USER_MODEL_TRACE, UCL_LINE_COLOR,
    FALLBACK_MODEL_GRAY, LOT_MARKER_COLOR, LOT_MARKER_OUTLINE,
    SCAN_LINE_FALLBACK, _hex_alpha, PLOT_BG_COLOR,
    SUPPORT_LINE_OPACITY, UCL_LINE_OPACITY, OLS_LINE_OPACITY,
    MC_LEGEND_BG_ALPHA,
)

from figures.common import (
    _INTERP_POINTS, _MAX_SCATTER_PTS, TRACE_WIDTH, SHADE_ALPHA,
    TRACE_WIDTH_OVERLAY,
    _TODAY_LINE_COLOR,
    CHART_FONT_LEGEND, CHART_FONT_TITLE, CHART_FONT_SUBTITLE,
    FONT_SANS, CHART_FONT_TITLE_LG, CHART_FONT_BODY_LG, CHART_FONT_TICK_LG, CHART_FONT_LEGEND_LG,
    _LOG_MINOR, _MC_LEGEND_POS,
    _get_palette, _get_model_color, _fmt_q_label,
    _base_layout, _year_ticks, _price_tickvals,
    _apply_sans_typography, _apply_config_annotation, _apply_watermark, _add_date_hover,
    _HOVER_FMT_USD,
    _lerp_hex, _hex_alpha, _build_symmetric_bands,
    _round_trace_data,
)
from colors import quantile_shade, config_b_shade
from figures.common import (quantile_opacity, _parse_quantiles,
                             _today_line_shapes)


def _r2_suffix(mdl, q):
    """Return '  R²=X.XXXX' suffix if R² available for model at quantile q, else ''."""
    r2 = getattr(mdl, 'r2_per_quantile', {}).get(q)
    if r2 is not None:
        return f"  R\u00b2={r2:.4f}"
    return ""


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
    sigma_mode = p.get("sigma_mode", "constant") or "constant"

    traces = []

    # ── shading between adjacent quantiles ───────────────────────────────────
    sel_qs = _parse_quantiles(p)

    bub_active = "bub" in p.get("active_models", ["bub"])
    # If no quantiles selected and non-BM overlay models are active, fall
    # back to Q50%. BM is excluded: composite/support already provide a
    # median reference, and a ghost Q50 line is visually indistinguishable
    # from the composite — hiding composite would appear to do nothing.
    # U1 is also exempt from the Q50 fallback: its direct P1→P2 line
    # already provides a visible trace, so there's no trace-free-plot risk
    # when the user deselects Q50 with only U1 active.
    _non_bub_active = [k for k in p.get("active_models", []) if k not in ("bub", "u1")]
    _fallback_q50 = not sel_qs and _non_bub_active
    _default_mode = "advanced" not in (p.get("qs_mode") or [])
    if _fallback_q50:
        sel_qs = [0.5]

    if bub_active:
        # Pre-compute prices for all selected quantiles
        _price_cache = {}
        for q in sel_qs:
            if q in model.fits:
                _price_cache[q] = _round_trace_data(model.price_at(q, t_arr, sigma_mode=sigma_mode) * (stack if stack > 0 else 1))

        if p.get("shade") and len(sel_qs) >= 2:
            _model_color = _get_model_color("bub", p)
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
        # Dark slate gray — distinct from BM gold so data points read clearly
        traces.append(go.Scatter(
            x=list(x_sc), y=y_sc,
            mode="markers", name="Price data",
            marker=dict(color=SCATTER_POINT,
                        size=max(2, int(p.get("pt_size", BUBBLE["pt_size"]))),
                        opacity=float(p.get("pt_alpha", BUBBLE["pt_alpha"]))),
            hovertemplate=_HOVER_FMT_USD,
        ))

    if bub_active:
        _bub_color = _get_model_color("bub", p)
        for q in sel_qs:
            if abs(q - 0.5) < 0.001:
                continue
            if q not in _price_cache:
                continue
            prices = _price_cache[q]
            lbl = _fmt_q_label(q) + _r2_suffix(model, q)
            if stack > 0:
                lbl += f"  \u2192  {fmt_price(float(prices[-1]))}"
            _shade = quantile_shade(_bub_color, q)
            _trace_opacity = quantile_opacity(q)
            if _fallback_q50 and _default_mode:
                _trace_opacity = _app_ctx.FALLBACK_Q50_OPACITY
            traces.append(go.Scatter(
                x=t_arr, y=prices,
                mode="lines", name=lbl,
                line=dict(color=_shade, width=TRACE_WIDTH),
                opacity=_trace_opacity,
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
            x=t_arr, y=um_prices,
            mode="lines", name=um_lbl,
            line=dict(color=USER_MODEL_TRACE, width=3.0, dash="solid"),
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
            _ovl_color_raw = _get_model_color(model_key, p)
            _b_keys = set(p.get("config_b_keys") or [])
            _is_b = model_key in _b_keys
            _has_b_sibling = bool(_b_keys)
            _ovl_color = (config_b_shade(_ovl_color_raw, _is_b)
                          if _has_b_sibling else _ovl_color_raw)
            _ovl_lines = []
            _ovl_band_prices = {}
            for q in overlay_qs:
                if q not in mdl.fits:
                    continue
                prices = _round_trace_data(mdl.price_at(q, t_arr, sigma_mode=sigma_mode) * (stack if stack > 0 else 1))
                _ovl_band_prices[q] = prices
                lbl = f"{mdl.legend_name} {_fmt_q_label(q, '')}" + _r2_suffix(mdl, q)
                if stack > 0:
                    lbl += f"  \u2192  {fmt_price(float(prices[-1]))}"
                _shade = quantile_shade(_ovl_color, q)
                _lw = 3.0 if (model_key == "u1" and hasattr(mdl, 'own_quantile') and abs(q - mdl.own_quantile) < 0.005) else TRACE_WIDTH_OVERLAY
                # Include Q50% R² in the group title so it's always visible
                # (Plotly shows the group title separately from each trace's
                # name; on narrow viewports the trace name — which carries the
                # R² suffix — can be truncated, so embedding R² in the title
                # guarantees the user sees it.)
                _group_title = mdl.legend_name
                if model_key == "u1":
                    _group_title = f"{mdl.legend_name}  m={mdl.fits[mdl.quantiles[0]]['slope']:.3f}"
                else:
                    _median_r2 = getattr(mdl, "r2_per_quantile", {}).get(0.5)
                    if _median_r2 is not None:
                        _group_title = f"{mdl.legend_name}  R\u00b2={_median_r2:.4f}"
                _ovl_lines.append(go.Scatter(
                    x=t_arr, y=prices,
                    mode="lines", name=lbl,
                    line=dict(color=_shade, width=_lw, dash=mdl.dash_style),
                    opacity=quantile_opacity(q),
                    legendgroup=mdl.short_name,
                    legendgrouptitle_text=_group_title,
                ))
            # Band fills FIRST so lines render on top (matching BM pattern).
            if p.get("shade") and len(_ovl_band_prices) >= 2:
                traces.extend(_build_symmetric_bands(
                    sorted(_ovl_band_prices.keys()), _ovl_band_prices, t_arr,
                    model_color=_ovl_color))
            traces.extend(_ovl_lines)
        else:
            # Non-quantized model: single trajectory
            prices = mdl.price_at(0.5, t_arr, sigma_mode=sigma_mode)
            if stack > 0:
                prices = prices * stack
            lbl = mdl.legend_name + _r2_suffix(mdl, 0.5)
            if stack > 0:
                lbl += f"  \u2192  {fmt_price(float(np.asarray(prices)[-1]))}"
            prices = _round_trace_data(np.asarray(prices))
            traces.append(go.Scatter(
                x=t_arr, y=prices,
                mode="lines", name=lbl,
                line=dict(color=_get_model_color(model_key, p), width=TRACE_WIDTH_OVERLAY, dash=mdl.dash_style),
                legendgroup=mdl.short_name,
            ))

        # ── composite/support/future for _CompositeModel overlays ─────
        if hasattr(mdl, "comp_by_n") and hasattr(mdl, "t_grid"):
            _mdl_color = _get_model_color(model_key, p)
            mdl_t = np.asarray(mdl.t_grid)
            mdl_mask = (mdl_t >= t_lo) & (mdl_t <= t_hi)

            if p.get("show_sup") and hasattr(mdl, "support_plot"):
                sup_y = np.asarray(mdl.support_plot)[mdl_mask] * (stack if stack > 0 else 1)
                traces.append(go.Scatter(
                    x=list(mdl_t[mdl_mask]), y=sup_y,
                    mode="lines", name=f"{mdl.legend_name} support",
                    line=dict(color=_mdl_color, dash="dash", width=1.5),
                    opacity=SUPPORT_LINE_OPACITY,
                    legendgroup=mdl.short_name,
                ))

            if p.get("show_comp"):
                n = int(p.get("n_future", BUBBLE["n_future"]))
                n = min(n, len(mdl.comp_by_n) - 1)
                comp_y = np.asarray(mdl.comp_by_n[n])[mdl_mask] * (stack if stack > 0 else 1)
                traces.append(go.Scatter(
                    x=list(mdl_t[mdl_mask]), y=comp_y,
                    mode="lines",
                    name=f"{mdl.legend_name} composite (N={n})  R\u00b2={mdl.bm_r2:.4f}",
                    line=dict(color=_mdl_color, width=2.0),
                    legendgroup=mdl.short_name,
                ))

    # ── scanner quantile lines ───────────────────────────────────────────────
    for sl in p.get("scanner_lines", []):
        mdl = _app_ctx.PRICE_MODELS.get(sl["model"])
        if not mdl:
            continue
        q = sl["q"]
        scan_prices = _round_trace_data(np.array([
            float(mdl.price_at(q, t, sigma_mode=sigma_mode)) for t in t_arr]))
        if stack > 0:
            scan_prices = scan_prices * stack
        col = _get_model_color(sl["model"], p)
        _scan_shade = quantile_shade(col, q)
        traces.append(go.Scatter(
            x=t_arr, y=scan_prices,
            mode="lines",
            name=f"{mdl.legend_name} Q{q*100:.1f}%",
            line=dict(color=_scan_shade, width=2, dash=mdl.dash_style),
            legendgroup=f"scan-{mdl.short_name}",
        ))

    # ── Unfairly Cheap Line ──────────────────────────────────────────────────
    if p.get("show_ucl"):
        p_ucl = 10.0 ** (_app_ctx.UCL_INTERCEPT + _app_ctx.UCL_SLOPE * np.log10(t_arr))
        if stack > 0:
            p_ucl = p_ucl * stack
        traces.append(go.Scatter(
            x=t_arr, y=p_ucl,
            mode="lines", name="Unfairly Cheap Line",
            line=dict(color=UCL_LINE_COLOR, dash="dot", width=1.8),
            opacity=UCL_LINE_OPACITY,
        ))

    # ── OLS line ──────────────────────────────────────────────────────────────
    if p.get("show_ols"):
        p_ols = 10.0 ** (m.ols_intercept + m.ols_slope * np.log10(t_arr))
        if stack > 0:
            p_ols = p_ols * stack
        traces.append(go.Scatter(
            x=t_arr, y=p_ols,
            mode="lines", name=f"OLS  R\u00b2={m.ols_r2:.4f}" if hasattr(m, 'ols_r2') and m.ols_r2 else "OLS",
            line=dict(color=FALLBACK_MODEL_GRAY, dash="dash", width=1.3),
            opacity=OLS_LINE_OPACITY,
        ))

    # Downsample BM curves to ~400 points (plenty for screen)
    def _downsample_bm(mask_arr, y_arr, target=400):
        x = m.years_plot_bm[mask_arr]
        y = y_arr
        n = len(x)
        if n > target:
            # Use linspace for evenly-spaced sampling — more reliable than stride division
            idx = np.linspace(0, n - 1, target, dtype=int)
            return x[idx], y[idx]
        return x, y

    # ── bubble support ────────────────────────────────────────────────────────
    _bm_color = _get_model_color("bub", p)
    if bub_active and p.get("show_sup"):
        mask = (m.years_plot_bm >= t_lo) & (m.years_plot_bm <= t_hi)
        sup_y = m.support_bm[mask] * (stack if stack > 0 else 1)
        x_sup, sup_y = _downsample_bm(mask, sup_y)
        traces.append(go.Scatter(
            x=x_sup, y=sup_y,
            mode="lines", name="Bubble support",
            line=dict(color=_bm_color,
                      dash="dash", width=float(p.get("sup_lw", BUBBLE["sup_lw"]))),
            opacity=SUPPORT_LINE_OPACITY,
        ))

    # ── bubble composite ──────────────────────────────────────────────────────
    if bub_active and p.get("show_comp"):
        n = int(p.get("n_future", BUBBLE["n_future"]))
        n = min(n, len(m.comp_by_n) - 1)
        mask = (m.years_plot_bm >= t_lo) & (m.years_plot_bm <= t_hi)
        comp_y = m.comp_by_n[n][mask] * (stack if stack > 0 else 1)
        x_comp, comp_y = _downsample_bm(mask, comp_y)
        traces.append(go.Scatter(
            x=x_comp, y=comp_y,
            mode="lines",
            name=f"Bubble composite (N={n})  R\u00b2={m.bm_r2:.4f}",
            line=dict(color=_bm_color,
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
                marker=dict(color=LOT_MARKER_COLOR, size=10,
                            line=dict(color=LOT_MARKER_OUTLINE, width=0.7)),
                text=lhover,
                hovertemplate="%{text}<extra></extra>",
            ))

    # ── today line (with glow shadow) ─────────────────────────────────────────
    shapes = []
    if p.get("show_today"):
        td = today_t(m.genesis)
        today_color = palette.get("today_line", _TODAY_LINE_COLOR)
        if t_lo <= td <= t_hi:
            shapes.extend(_today_line_shapes(td, y_lo, y_hi, today_color,
                                             glow=True, yref="y"))

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
        title="Bitcoin Price + Model Overlay",
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
            bgcolor=_hex_alpha(PLOT_BG_COLOR, MC_LEGEND_BG_ALPHA),
        )
    layout["shapes"] = shapes

    if stack > 0:
        layout["annotations"] = [dict(
            text=f"Stack: {p['stack']:.6g} BTC",
            xref="paper", yref="paper", x=0.99, y=0.01,
            xanchor="right", yanchor="bottom",
            showarrow=False, font=dict(size=CHART_FONT_LEGEND, color=theme.TEXT_COLOR),
            bgcolor=theme.PLOT_BG_COLOR, bordercolor=theme.SPINE_COLOR, borderwidth=1,
        )]

    # ── component decomposition traces (dotted individual + solid Σ sum) ─────
    _add_decomposition_traces(traces, t_arr, m, p)

    _apply_sans_typography(layout)
    fig = go.Figure(data=traces, layout=go.Layout(**layout))
    _add_date_hover(fig, m.genesis, recovery=True)
    _apply_config_annotation(fig, p, "bub", show_qr=True, show_mc=False)
    wm_pos = "bottom-left" if leg_pos == "bottom-right" else "bottom-right"
    _apply_watermark(fig, pos=wm_pos)
    return fig


def _add_decomposition_traces(traces, t_arr, m, p):
    """Append ONE decomposition trace = 10^(sum of checked components).

    Each checkbox is a 0/1 switch:
      log₁₀(price) = c₁·[on?] + c₂·[on?] + c₃·[on?] + ...
    All checked → full model. Subset → partial model. None checked → nothing.
    """
    import _app_ctx
    from callbacks.charts import _resolve_decomp_model_key

    family = p.get("decomp_model", "") or ""
    selected = list(p.get("decomp_components", []) or [])
    # Ignore legacy __sum__ pseudo-entry from old snapshots
    selected = [s for s in selected if s != "__sum__"]
    if not family or not selected:
        return

    key = _resolve_decomp_model_key(
        family,
        p.get("lppl_n_freqs", []),
        p.get("lppl_weighted", []),
        p.get("lppl_no_13", []),
    )
    if key is None:
        return
    model = _app_ctx.PRICE_MODELS.get(key)
    if model is None:
        return

    palette = p.get("palette", "default")
    sum_color = _app_ctx.DECOMP_SUM_COLOR.get(
        palette, _app_ctx.DECOMP_SUM_COLOR["default"])

    comps = model.components(t_arr)
    canonical = [n for n in model.component_names if n in selected]
    if not canonical:
        return

    # log₁₀(price) = sum of checked component values
    log_vals = comps[canonical[0]].copy()
    for n in canonical[1:]:
        log_vals = log_vals + comps[n]

    total = len(model.component_names)
    n_checked = len(canonical)
    base_label = (f"{model.legend_name} | full model" if n_checked == total
                  else f"{model.legend_name} | {n_checked}/{total} components")

    # R² vs actual price data at historical points
    r2_str = ""
    try:
        t_price = np.asarray(m.price_years, float)
        mask = t_price >= 1.0
        t_price = t_price[mask]
        log_actual = np.log10(np.asarray(m.price_prices, float)[mask])
        comps_at_price = model.components(t_price)
        partial_log = comps_at_price[canonical[0]].copy()
        for n in canonical[1:]:
            partial_log = partial_log + comps_at_price[n]
        residuals = log_actual - partial_log
        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((log_actual - np.mean(log_actual)) ** 2))
        if ss_tot > 0:
            r2 = 1.0 - ss_res / ss_tot
            r2_str = f"  R\u00b2={r2:.4f}"
    except Exception:
        pass

    traces.append(go.Scatter(
        x=t_arr, y=10.0 ** log_vals, mode="lines",
        line=dict(dash="solid", width=2.5, color=sum_color),
        name=base_label + r2_str,
    ))
