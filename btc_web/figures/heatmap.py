"""CAGR Heatmap chart builders."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from typing import Any

import _app_ctx
import theme
from tab_defaults import HEATMAP
from btc_core import ModelData, yr_to_t, fmt_price, leo_weighted_entry
from mc_overlay import _mc_heatmap_overlay

from figures.common import (
    _FONT_SUBTITLE, _FONT_TITLE_LG, _FONT_BODY_LG, _FONT_TICK_LG,
    _SANS_FONT,
    _get_palette, _error_figure, _fmt_q_label,
    _lerp_hex, _dense_colorscale, _hex_alpha,
    _apply_config_annotation, _apply_watermark,
    _apply_mc_premium, _apply_mc_xlabel,
)


def _seg_colorscale(mc, b1, b2, c_lo, c_mid1, c_mid2, c_hi):
    """Build a dense 256-point colorscale from the segmented colour config.

    Breakpoints b1/b2 (raw CAGR %) are mapped to normalised [0,1] positions.
    Returns (colorscale, zmin, zmax).
    """
    mn, mx = float(mc.min()), float(mc.max())
    if mx - mn < 1e-9:
        return [[0.0, c_mid1], [1.0, c_mid1]], mn, mx
    # Build piecewise-linear color ramp: anchor points at breakpoints b1/b2,
    # normalised to [0,1] within [mn, mx]. _lerp_hex interpolates between
    # adjacent anchors; _dense_colorscale samples the resulting ramp at 256 pts.
    anchors = [(0.0, c_lo if mn <= b1 else (c_mid1 if mn <= b2 else c_mid2))]
    if mn < b1 < mx:
        anchors.append(((b1 - mn) / (mx - mn), c_mid1))
    if mn < b2 < mx:
        anchors.append(((b2 - mn) / (mx - mn), c_mid2))
    anchors.append((1.0, c_hi if mx > b2 else (c_mid2 if mx > b1 else c_mid1)))

    def color_at(t):
        col = anchors[-1][1]
        for i in range(len(anchors) - 1):
            if anchors[i][0] <= t <= anchors[i + 1][0]:
                span = anchors[i + 1][0] - anchors[i][0]
                col = (anchors[i][1] if span < 1e-9
                       else _lerp_hex(anchors[i][1], anchors[i + 1][1],
                                      (t - anchors[i][0]) / span))
                break
        return col

    return _dense_colorscale(color_at), mn, mx


def _heatmap_colorscale(m, p, mc):
    """Compute colorscale, zmin, zmax from heatmap params and CAGR matrix."""
    mode   = int(p.get("color_mode", 0))
    c_lo   = p.get("c_lo",   m.CAGR_SEG_C_LO)
    c_mid1 = p.get("c_mid1", m.CAGR_SEG_C_MID1)
    c_mid2 = p.get("c_mid2", m.CAGR_SEG_C_MID2)
    c_hi   = p.get("c_hi",   m.CAGR_SEG_C_HI)
    b1     = float(p.get("b1", m.CAGR_SEG_B1))
    b2     = float(p.get("b2", m.CAGR_SEG_B2))

    valid = mc[~np.isnan(mc)] if np.any(np.isnan(mc)) else mc
    mn, mx = float(valid.min()), float(valid.max())

    if mode == 0:
        return _seg_colorscale(valid, b1, b2, c_lo, c_mid1, c_mid2, c_hi)
    elif mode == 1:
        return _dense_colorscale(lambda t: _lerp_hex(c_lo, c_hi, t)), mn, mx
    else:
        abs_max = max(abs(mn), abs(mx), 1e-6)
        def _div_color(t):
            if t < 0.5:
                return _lerp_hex(c_lo, c_mid1, t * 2.0)
            return _lerp_hex(c_mid2, c_hi, (t - 0.5) * 2.0)
        return _dense_colorscale(_div_color), -abs_max, abs_max


def _heatmap_cell_annots(mc, mp, mm, vfmt, hm_stk, zmin, zmax, cell_fs, colorscale=None, palette=None):
    """Build cell text annotation dicts for a CAGR heatmap."""
    if not colorscale:
        # Fallback: use simple threshold-based approach
        colorscale = [[0.0, "rgb(27,10,46)"], [1.0, "rgb(255,215,0)"]]
    annots = []
    for ri in range(mc.shape[0]):
        for ci in range(mc.shape[1]):
            vc2 = mc[ri, ci]
            if np.isnan(vc2):
                continue
            vp2 = mp[ri, ci]
            vm  = mm[ri, ci]
            if vfmt == "cagr":
                tx = f"{vc2:+.0f}%"
            elif vfmt == "price":
                tx = fmt_price(vp2)
            elif vfmt == "both":
                tx = f"{vc2:+.0f}%\n{fmt_price(vp2)}"
            elif vfmt == "stack":
                pv = fmt_price(vp2 * hm_stk) if hm_stk > 0 else fmt_price(vp2)
                tx = f"{vc2:+.0f}%\n{pv}"
            elif vfmt == "port_only":
                tx = fmt_price(vp2 * hm_stk) if hm_stk > 0 else fmt_price(vp2)
            elif vfmt == "mult_only":
                tx = f"{vm:.2f}\u00d7"
            elif vfmt == "cagr_mult":
                tx = f"{vc2:+.0f}%\n{vm:.2f}\u00d7"
            elif vfmt == "mult_port":
                pv = fmt_price(vp2 * hm_stk) if hm_stk > 0 else fmt_price(vp2)
                tx = f"{vm:.2f}\u00d7\n{pv}"
            elif vfmt == "none":
                tx = ""
            else:
                import logging as _log
                _log.getLogger(__name__).warning("Unknown heatmap vfmt: %s", vfmt)
                tx = ""

            if tx:
                # Compute actual cell background brightness from the colorscale
                cell_norm = (vc2 - zmin) / max(zmax - zmin, 1e-6)
                cell_norm = max(0.0, min(1.0, cell_norm))
                # Sample the colorscale to get the actual cell RGB
                cs_idx = int(cell_norm * (len(colorscale) - 1))
                cs_idx = max(0, min(len(colorscale) - 1, cs_idx))
                _cs_rgb = colorscale[cs_idx][1]  # "rgb(r,g,b)"
                _rgb = [int(x) for x in _cs_rgb.replace("rgb(", "").replace(")", "").split(",")]
                _lum = (0.299 * _rgb[0] + 0.587 * _rgb[1] + 0.114 * _rgb[2]) / 255.0
                # Signed cell text colors: red for loss, gold for exceptional
                _bg = None  # optional text background for low-contrast cells
                _loss_color = palette.get("hm_loss_text", "#ff8a80") if palette else "#ff8a80"
                _exc_color = palette.get("hm_exceptional_text", "#ffd700") if palette else "#ffd700"
                if vc2 < 0:
                    txt_col = _loss_color     # soft red — loss
                elif vc2 > 50:
                    txt_col = _exc_color     # gold — exceptional
                elif _lum < 0.45:
                    txt_col = "#ffffff"     # white on dark cells
                else:
                    txt_col = "#111111"     # dark on light cells
                # Low-contrast guard: if text and background are too close,
                # add a semi-transparent backdrop so text is always readable
                _txt_lum = 1.0 if txt_col in ("#ffffff", _loss_color, _exc_color) else 0.07
                if abs(_txt_lum - _lum) < 0.25:
                    _bg = "rgba(0,0,0,0.55)" if _txt_lum > 0.5 else "rgba(255,255,255,0.6)"
                    txt_col = "#ffffff" if _txt_lum > 0.5 else "#111111"
                ann = dict(
                    x=ci, y=ri,
                    text=tx.replace("\n", "<br>"),
                    showarrow=False,
                    font=dict(size=cell_fs, color=txt_col,
                              family=_SANS_FONT, weight="bold"),
                    xref="x", yref="y",
                )
                if _bg:
                    ann["bgcolor"] = _bg
                    ann["borderpad"] = 2
                annots.append(ann)
    return annots


def build_heatmap_figure(m: ModelData, p: dict[str, Any]) -> go.Figure:
    """
    p keys: entry_yr, entry_q, exit_yr_lo, exit_yr_hi, exit_qs (list),
            color_mode (0=Segmented,1=DataScaled,2=Diverging),
            b1, b2, c_lo, c_mid1, c_mid2, c_hi, n_disc,
            vfmt, show_colorbar, stack,
            lots (list), use_lots, hm_model (str, default "bub")
    """
    hm_model_key = p.get("hm_model", "bub")
    model = _app_ctx.PRICE_MODELS.get(hm_model_key, _app_ctx.DEFAULT_MODEL)
    palette = _get_palette(p)
    eyr = int(p.get("entry_yr", 2020))
    eq  = float(p.get("entry_q", 50)) / 100.0   # stored as percentage (e.g. 7.5 -> 0.075)
    entry_t = yr_to_t(eyr, m.genesis)
    live_price = p.get("live_price")
    is_quantized = getattr(model, "quantized", True)

    # For non-quantized models, entry price uses model median (0.5) or interp
    if is_quantized:
        ep = float(live_price) if live_price else model.interp_price(eq, entry_t)
    else:
        ep = float(live_price) if live_price else float(model.price_at(0.5, entry_t))

    # LOT ENTRY OVERRIDE
    lots = p.get("lots") or []
    if p.get("use_lots") and lots:
        result = leo_weighted_entry(lots)
        if result:
            ep, entry_t, _pct, _tw = result

    xlo = int(p.get("exit_yr_lo", eyr))
    xhi = int(p.get("exit_yr_hi", eyr + 10))
    eyrs = list(range(xlo, xhi + 1))

    if is_quantized:
        # Quantized model: full quantile x year matrix
        xqs_raw = p.get("exit_qs") or []
        xqs = sorted([float(q) for q in xqs_raw if float(q) in model.fits], reverse=True)

        if not eyrs or not xqs:
            return _error_figure("No data \u2014 adjust Entry / Exit settings")

        mc = np.zeros((len(xqs), len(eyrs)))
        mp = np.zeros((len(xqs), len(eyrs)))
        mm = np.zeros((len(xqs), len(eyrs)))
        for ci, ey in enumerate(eyrs):
            et = yr_to_t(ey, m.genesis)
            nyr = et - entry_t if p.get("use_lots") and lots else float(ey - eyr)
            for ri, xq in enumerate(xqs):
                xpp = float(model.price_at(xq, et))
                mp[ri, ci] = xpp
                mm[ri, ci] = xpp / ep if ep > 0 else 0.0
                if nyr <= 0:
                    mc[ri, ci] = (xpp / ep - 1.0) * 100.0
                else:
                    mc[ri, ci] = ((xpp / ep) ** (1.0 / nyr) - 1.0) * 100.0

        ylabels = [_fmt_q_label(q) for q in xqs]
        y_title = "Exit Quantile"
        hover_y_label = "Quantile"
        n_rows = len(xqs)
    else:
        # Non-quantized model: single-row heatmap
        if not eyrs:
            return _error_figure("No data \u2014 adjust Exit year range")

        mc = np.zeros((1, len(eyrs)))
        mp = np.zeros((1, len(eyrs)))
        mm = np.zeros((1, len(eyrs)))
        for ci, ey in enumerate(eyrs):
            et = yr_to_t(ey, m.genesis)
            nyr = et - entry_t if p.get("use_lots") and lots else float(ey - eyr)
            xpp = float(model.price_at(0.5, et))
            mp[0, ci] = xpp
            mm[0, ci] = xpp / ep if ep > 0 else 0.0
            if nyr <= 0:
                mc[0, ci] = (xpp / ep - 1.0) * 100.0
            else:
                mc[0, ci] = ((xpp / ep) ** (1.0 / nyr) - 1.0) * 100.0

        ylabels = [model.name]
        y_title = "Model"
        hover_y_label = "Model"
        n_rows = 1

    colorscale, zmin, zmax = _heatmap_colorscale(m, p, mc)

    # ── cell text ─────────────────────────────────────────────────────────────
    vfmt    = p.get("vfmt", HEATMAP["vfmt"])
    hm_stk  = float(p.get("stack", HEATMAP["stack"]))

    # ── cell annotations ──────────────────────────────────────────────────────
    annots = _heatmap_cell_annots(mc, mp, mm, vfmt, hm_stk, zmin, zmax,
                                   int(p.get("cell_font_size", HEATMAP["cell_font_size"])),
                                   colorscale=colorscale, palette=palette)

    fig = go.Figure(data=go.Heatmap(
        z=mc, x=[str(y) for y in eyrs], y=ylabels,
        colorscale=colorscale, zmin=zmin, zmax=zmax,
        showscale=bool(p.get("show_colorbar", True)),
        colorbar=dict(
            title=dict(text="CAGR %", font=dict(color=theme.TEXT_COLOR)),
            tickfont=dict(color=theme.TEXT_COLOR),
            bgcolor=theme.PLOT_BG_COLOR,
            outlinecolor=theme.SPINE_COLOR,
        ),
        hovertemplate=f"Exit: %{{x}}<br>{hover_y_label}: %{{y}}<br>CAGR: %{{z:.1f}}%<extra></extra>",
    ))

    entry_lbl = (f"Entry: {eyr}  {fmt_price(ep)}  \u00b7  Q{eq*100:.4g}%"
                 if not (p.get("use_lots") and lots)
                 else f"Entry: lots weighted avg  {fmt_price(ep)}")

    # Title varies by model type
    is_default = (hm_model_key == "bub")
    if not is_quantized:
        title_text = f"{model.name} (non-quantized) \u2014 {entry_lbl}"
    elif not is_default:
        title_text = f"{model.name} \u2014 {entry_lbl}"
    else:
        title_text = f"CAGR Heatmap \u2014 {entry_lbl}"

    fig.update_layout(
        title=dict(text=title_text,
                   font=dict(color=theme.TITLE_COLOR, size=_FONT_SUBTITLE)),
        paper_bgcolor=theme.PLOT_BG_COLOR,
        plot_bgcolor=theme.PLOT_BG_COLOR,
        font=dict(color=theme.TEXT_COLOR),
        xaxis=dict(title="Exit Year", gridcolor=theme.GRID_MAJOR_COLOR,
                   linecolor=theme.SPINE_COLOR, tickcolor=theme.TEXT_COLOR,
                   fixedrange=True),
        yaxis=dict(title=y_title, gridcolor=theme.GRID_MAJOR_COLOR,
                   linecolor=theme.SPINE_COLOR, tickcolor=theme.TEXT_COLOR,
                   fixedrange=True),
        annotations=annots,
        margin=dict(l=70, r=20, t=60, b=50),
    )
    fig.layout.title.font.update(family=_SANS_FONT, size=_FONT_TITLE_LG)
    fig.layout.font.update(family=_SANS_FONT, size=_FONT_TICK_LG, weight="bold")
    fig.layout.xaxis.title.font.update(family=_SANS_FONT, size=_FONT_BODY_LG)
    fig.layout.yaxis.title.font.update(family=_SANS_FONT, size=_FONT_BODY_LG)
    # Cell font family/size/weight set in _heatmap_cell_annots; no override here.
    # Global font.weight="bold" ensures iOS Safari renders bold on first paint

    # ── Entry year column highlight (orange border around the entry column) ──
    if str(eyr) in [str(y) for y in eyrs]:
        entry_ci = [str(y) for y in eyrs].index(str(eyr))
        fig.add_shape(type="rect",
            x0=entry_ci - 0.5, x1=entry_ci + 0.5,
            y0=-0.5, y1=n_rows - 0.5,
            line=dict(color="#f7931a", width=2),
            fillcolor="rgba(247,147,26,0.06)",
            xref="x", yref="y",
        )

    # (per-annotation weight is unreliable on initial mobile render).
    _apply_config_annotation(fig, p, "hm", show_qr=True, show_mc=False)
    _apply_watermark(fig)
    return fig


def build_mc_heatmap_figure(m: ModelData, p: dict[str, Any]) -> tuple[go.Figure, dict | None]:
    """Build a standalone MC heatmap figure from MC-simulated CAGR percentiles.
    Returns (fig, mc_result) or (empty_fig, None).
    """
    model = _app_ctx.DEFAULT_MODEL
    palette = _get_palette(p)
    eyr = int(p.get("mc_start_yr", p.get("entry_yr", 2020)))
    eq  = float(p.get("mc_entry_q", p.get("entry_q", 50))) / 100.0
    entry_t = yr_to_t(eyr, m.genesis)
    live_price = p.get("live_price")
    ep  = float(live_price) if live_price else model.interp_price(eq, entry_t)

    lots = p.get("lots") or []
    if p.get("use_lots") and lots:
        result = leo_weighted_entry(lots)
        if result:
            ep, entry_t, _pct, _tw = result

    mc_years = int(p.get("mc_years", 10))
    eyrs = list(range(eyr, eyr + mc_years + 1))

    if not eyrs:
        return _error_figure("No data \u2014 adjust Entry / Exit settings"), None

    mc_data = _mc_heatmap_overlay(m, p, ep, entry_t, eyrs)
    mc_cagr, mc_prices, mc_mults, mc_labels, mc_result = mc_data
    if mc_cagr is None:
        return _error_figure("MC simulation error"), None

    mc = mc_cagr
    mp = mc_prices
    mm = mc_mults

    valid = mc[~np.isnan(mc)]
    if len(valid) == 0:
        return _error_figure("MC: no valid data in range"), mc_result

    colorscale, zmin, zmax = _heatmap_colorscale(m, p, mc)

    # ── cell text ────────────────────────────────────────────────────────────
    vfmt   = p.get("vfmt", "cagr")
    hm_stk = float(p.get("stack", 0))

    annots = _heatmap_cell_annots(mc, mp, mm, vfmt, hm_stk, zmin, zmax,
                                   int(p.get("cell_font_size", HEATMAP["cell_font_size"])),
                                   colorscale=colorscale, palette=palette)

    fig = go.Figure(data=go.Heatmap(
        z=mc, x=[str(y) for y in eyrs], y=mc_labels,
        colorscale=colorscale, zmin=zmin, zmax=zmax,
        showscale=bool(p.get("show_colorbar", True)),
        colorbar=dict(
            title=dict(text="CAGR %", font=dict(color=theme.TEXT_COLOR)),
            tickfont=dict(color=theme.TEXT_COLOR),
            bgcolor=theme.PLOT_BG_COLOR,
            outlinecolor=theme.SPINE_COLOR,
        ),
        hovertemplate="Exit: %{x}<br>Percentile: %{y}<br>CAGR: %{z:.1f}%<extra></extra>",
    ))

    entry_lbl = (f"Entry: {eyr}  {fmt_price(ep)}  \u00b7  Q{eq*100:.4g}%"
                 if not (p.get("use_lots") and lots)
                 else f"Entry: lots weighted avg  {fmt_price(ep)}")

    fig.update_layout(
        title=dict(text=f"Monte Carlo CAGR \u2014 {entry_lbl}",
                   font=dict(color=theme.TITLE_COLOR, size=_FONT_SUBTITLE)),
        paper_bgcolor=theme.PLOT_BG_COLOR,
        plot_bgcolor=theme.PLOT_BG_COLOR,
        font=dict(color=theme.TEXT_COLOR),
        xaxis=dict(title="Exit Year", gridcolor=theme.GRID_MAJOR_COLOR,
                   linecolor=theme.SPINE_COLOR, tickcolor=theme.TEXT_COLOR,
                   fixedrange=True),
        yaxis=dict(title="MC Percentile", gridcolor=theme.GRID_MAJOR_COLOR,
                   linecolor=theme.SPINE_COLOR, tickcolor=theme.TEXT_COLOR,
                   fixedrange=True),
        annotations=annots,
        margin=dict(l=70, r=20, t=60, b=50),
    )
    fig.layout.title.font.update(family=_SANS_FONT, size=_FONT_TITLE_LG)
    fig.layout.font.update(family=_SANS_FONT, size=_FONT_TICK_LG)
    fig.layout.xaxis.title.font.update(family=_SANS_FONT, size=_FONT_BODY_LG)
    fig.layout.yaxis.title.font.update(family=_SANS_FONT, size=_FONT_BODY_LG)
    # Cell font family/size/weight set in _heatmap_cell_annots; no override here.
    if p.get("mc_enabled"):
        _apply_mc_premium(fig, legend_pos=None)
        _apply_mc_xlabel(fig, p, "hm")
    _apply_watermark(fig)
    return fig, mc_result


def build_cagr_line_figure(m: ModelData, p: dict[str, Any]) -> go.Figure:
    """Build a 1-year forward CAGR vs time chart for multiple models.

    At each year t, plots: (model_price(t+1) / model_price(t) - 1) * 100
    This shows how the implied annual growth rate evolves over time for
    each model at the selected quantile.
    """
    from figures.common import _get_model_color, _base_layout
    from layout.common import _bands_to_qs

    eq = float(p.get("entry_q", 50)) / 100.0
    xlo = int(p.get("exit_yr_lo", 2010))
    xhi = int(p.get("exit_yr_hi", 2040))

    # Monthly resolution for smooth curves
    steps_per_year = 12
    n_steps = (xhi - xlo) * steps_per_year
    if n_steps < 1:
        return _error_figure("No years \u2014 adjust range")
    years = [xlo + i / steps_per_year for i in range(n_steps)]

    active_models = p.get("cagr_models") or ["bub"]

    # Quantiles to plot
    raw_qs = p.get("cagr_qs") or [eq]
    if raw_qs and isinstance(raw_qs[0], str):
        raw_qs = _bands_to_qs(raw_qs)
    sel_qs = sorted(set([float(q) for q in raw_qs]))
    if not sel_qs:
        sel_qs = [eq]

    traces = []
    for model_key in active_models:
        model = _app_ctx.PRICE_MODELS.get(model_key)
        if not model:
            continue

        is_q = getattr(model, "quantized", True)
        color = _get_model_color(model_key, p)

        qs_to_plot = sel_qs if is_q else [0.5]
        for q in qs_to_plot:
            cagrs = []
            for yr in years:
                t0 = yr_to_t(yr, m.genesis)
                t1 = yr_to_t(yr + 1, m.genesis)  # always 1-year forward
                if is_q:
                    p0 = model.interp_price(q, max(t0, 0.5))
                    p1 = model.interp_price(q, max(t1, 0.5))
                else:
                    p0 = float(model.price_at(0.5, max(t0, 0.5)))
                    p1 = float(model.price_at(0.5, max(t1, 0.5)))
                if p0 > 0:
                    cagrs.append((p1 / p0 - 1.0) * 100.0)
                else:
                    cagrs.append(0.0)

            _dist = abs(q - 0.5) / 0.45
            _q_opacity = max(0.1, 1.0 - _dist * 0.5)
            q_pct = q * 100
            q_lbl = f"Q{q_pct:.4g}%" if q_pct >= 1 else f"Q{q_pct:.3g}%"
            lbl = f"{model.legend_name} {q_lbl}" if len(qs_to_plot) > 1 else model.legend_name
            traces.append(go.Scatter(
                x=years,
                y=cagrs,
                mode="lines",
                name=lbl,
                line=dict(color=color, width=2),
                opacity=_q_opacity,
                legendgroup=model_key,
            ))

    # Zero line
    traces.append(go.Scatter(
        x=[years[0], years[-1]],
        y=[0, 0],
        mode="lines",
        line=dict(color="#888", width=0.5, dash="dot"),
        showlegend=False, hoverinfo="skip",
    ))

    # Today line + beacon markers
    import pandas as pd
    yr_now = pd.Timestamp.today().year + pd.Timestamp.today().day_of_year / 365.25
    today_shapes = []
    if years[0] <= yr_now <= years[-1]:
        palette = _get_palette(p)
        today_color = palette.get("today_line", "#FF6600")
        today_shapes.append(dict(
            type="line", xref="x", yref="paper",
            x0=yr_now, x1=yr_now, y0=0, y1=1,
            line=dict(color=today_color, width=1.5, dash="dash"),
        ))
        # Beacon dots — find nearest data point to today
        idx = min(range(len(years)), key=lambda i: abs(years[i] - yr_now))
        if abs(years[idx] - yr_now) < 0.5:
            for tr in traces:
                if not hasattr(tr, 'y') or not tr.y or idx >= len(tr.y):
                    continue
                if tr.showlegend is False:
                    continue
                y_val = tr.y[idx]
                tr_color = tr.line.color if tr.line and tr.line.color else today_color
                # Glow ring
                traces.append(go.Scatter(
                    x=[yr_now], y=[y_val], mode="markers",
                    marker=dict(size=16, color=tr_color, opacity=0.2,
                                line=dict(width=0)),
                    showlegend=False, hoverinfo="skip",
                ))
                # Core dot
                traces.append(go.Scatter(
                    x=[yr_now], y=[y_val], mode="markers",
                    marker=dict(size=6, color=tr_color,
                                line=dict(color="#fff", width=1)),
                    showlegend=False,
                    hovertemplate=f"Today: {y_val:.1f}%<extra></extra>",
                ))

    # Build title from quantiles shown
    if len(sel_qs) == 1:
        q_lbl = f"Q{sel_qs[0]*100:.4g}%"
    elif len(sel_qs) <= 3:
        q_lbl = "/".join(f"Q{q*100:.4g}%" for q in sel_qs)
    else:
        q_lbl = f"{len(sel_qs)} quantiles"

    layout = _base_layout(
        title=f"1-Year Forward Growth Rate \u2014 {q_lbl}",
        xlabel="",
        ylabel="",
    )
    layout["margin"] = dict(l=5, r=20, t=50, b=30, autoexpand=False)
    layout["yaxis"]["automargin"] = False
    layout["yaxis"]["side"] = "left"
    layout["yaxis"]["ticklabelposition"] = "inside"
    layout["yaxis"]["ticklabelshift"] = -5
    layout["yaxis"]["ticklabeloverflow"] = "allow"
    layout["yaxis"]["ticksuffix"] = "%"
    layout["yaxis"]["ticks"] = ""

    # X axis tick labels — aligned to round years, step by 1/2/5
    xlo = int(p.get("exit_yr_lo", 2010))
    xhi = int(p.get("exit_yr_hi", 2040))
    span = xhi - xlo
    step = 1 if span <= 15 else (2 if span <= 30 else (5 if span <= 50 else 10))
    # Align ticks to round multiples (2010, 2020, 2030...)
    first_tick = xlo + (-xlo % step) if xlo % step else xlo
    tick_yrs = list(range(first_tick, xhi + 1, step))
    layout["xaxis"].update(
        tickmode="array",
        tickvals=tick_yrs,
        ticktext=[f"'{y % 100:02d}" for y in tick_yrs],
        tickangle=0,
        ticks="",
        tickfont=dict(family="Arial Narrow, sans-serif-condensed, sans-serif"),
        range=[xlo - 0.5, xhi + 0.5],
        autorange=False,
    )

    # Legend
    show_legend = p.get("show_legend", True)
    legend_pos = p.get("legend_pos", "outside")
    if not show_legend:
        layout["showlegend"] = False
    elif legend_pos != "outside":
        from figures.common import _MC_LEGEND_POS
        lp = _MC_LEGEND_POS.get(legend_pos, {})
        layout["legend"] = dict(**lp, font=dict(size=10),
                                bgcolor="rgba(255,255,255,0.85)")
    else:
        layout["legend"] = dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(size=10),
        )

    # CAGR always uses linear axes (negative values possible)

    if today_shapes:
        layout["shapes"] = today_shapes

    fig = go.Figure(data=traces, layout=go.Layout(**layout))

    # Zoom + click control
    if not p.get("chart_zoom"):
        fig.update_layout(dragmode=False)
    fig.update_layout(clickmode="none")

    _apply_watermark(fig)
    return fig
