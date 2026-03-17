"""CAGR Heatmap chart builders."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from typing import Any

import _app_ctx
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
            lots (list), use_lots
    """
    model = _app_ctx.DEFAULT_MODEL
    palette = _get_palette(p)
    eyr = int(p.get("entry_yr", 2020))
    eq  = float(p.get("entry_q", 50)) / 100.0   # stored as percentage (e.g. 7.5 -> 0.075)
    entry_t = yr_to_t(eyr, m.genesis)
    live_price = p.get("live_price")
    ep  = float(live_price) if live_price else model.interp_price(eq, entry_t)

    # LOT ENTRY OVERRIDE
    lots = p.get("lots") or []
    if p.get("use_lots") and lots:
        result = leo_weighted_entry(lots)
        if result:
            ep, entry_t, _pct, _tw = result

    xlo = int(p.get("exit_yr_lo", eyr))
    xhi = int(p.get("exit_yr_hi", eyr + 10))
    eyrs = list(range(xlo, xhi + 1))

    xqs_raw = p.get("exit_qs") or []
    xqs = sorted([float(q) for q in xqs_raw if float(q) in model.fits], reverse=True)

    if not eyrs or not xqs:
        return _error_figure(m, "No data \u2014 adjust Entry / Exit settings")

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

    colorscale, zmin, zmax = _heatmap_colorscale(m, p, mc)

    # ── cell text ─────────────────────────────────────────────────────────────
    vfmt    = p.get("vfmt", "cagr")
    hm_stk  = float(p.get("stack", 0))

    # ── quantile y-axis labels ────────────────────────────────────────────────
    ylabels = []
    for q in xqs:
        ylabels.append(_fmt_q_label(q))

    # ── cell annotations ──────────────────────────────────────────────────────
    annots = _heatmap_cell_annots(mc, mp, mm, vfmt, hm_stk, zmin, zmax,
                                   int(p.get("cell_font_size", 9)),
                                   colorscale=colorscale, palette=palette)

    fig = go.Figure(data=go.Heatmap(
        z=mc, x=[str(y) for y in eyrs], y=ylabels,
        colorscale=colorscale, zmin=zmin, zmax=zmax,
        showscale=bool(p.get("show_colorbar", True)),
        colorbar=dict(
            title=dict(text="CAGR %", font=dict(color=m.TEXT_COLOR)),
            tickfont=dict(color=m.TEXT_COLOR),
            bgcolor=m.PLOT_BG_COLOR,
            outlinecolor=m.SPINE_COLOR,
        ),
        hovertemplate="Exit: %{x}<br>Quantile: %{y}<br>CAGR: %{z:.1f}%<extra></extra>",
    ))

    entry_lbl = (f"Entry: {eyr}  {fmt_price(ep)}  \u00b7  Q{eq*100:.4g}%"
                 if not (p.get("use_lots") and lots)
                 else f"Entry: lots weighted avg  {fmt_price(ep)}")

    fig.update_layout(
        title=dict(text=f"CAGR Heatmap \u2014 {entry_lbl}",
                   font=dict(color=m.TITLE_COLOR, size=_FONT_SUBTITLE)),
        paper_bgcolor=m.PLOT_BG_COLOR,
        plot_bgcolor=m.PLOT_BG_COLOR,
        font=dict(color=m.TEXT_COLOR),
        xaxis=dict(title="Exit Year", gridcolor=m.GRID_MAJOR_COLOR,
                   linecolor=m.SPINE_COLOR, tickcolor=m.TEXT_COLOR,
                   fixedrange=True),
        yaxis=dict(title="Exit Quantile", gridcolor=m.GRID_MAJOR_COLOR,
                   linecolor=m.SPINE_COLOR, tickcolor=m.TEXT_COLOR,
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
            y0=-0.5, y1=len(xqs) - 0.5,
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
        return _error_figure(m, "No data \u2014 adjust Entry / Exit settings"), None

    mc_data = _mc_heatmap_overlay(m, p, ep, entry_t, eyrs)
    mc_cagr, mc_prices, mc_mults, mc_labels, mc_result = mc_data
    if mc_cagr is None:
        return _error_figure(m, "MC simulation error"), None

    mc = mc_cagr
    mp = mc_prices
    mm = mc_mults

    valid = mc[~np.isnan(mc)]
    if len(valid) == 0:
        return _error_figure(m, "MC: no valid data in range"), mc_result

    colorscale, zmin, zmax = _heatmap_colorscale(m, p, mc)

    # ── cell text ────────────────────────────────────────────────────────────
    vfmt   = p.get("vfmt", "cagr")
    hm_stk = float(p.get("stack", 0))

    annots = _heatmap_cell_annots(mc, mp, mm, vfmt, hm_stk, zmin, zmax,
                                   int(p.get("cell_font_size", 9)),
                                   colorscale=colorscale, palette=palette)

    fig = go.Figure(data=go.Heatmap(
        z=mc, x=[str(y) for y in eyrs], y=mc_labels,
        colorscale=colorscale, zmin=zmin, zmax=zmax,
        showscale=bool(p.get("show_colorbar", True)),
        colorbar=dict(
            title=dict(text="CAGR %", font=dict(color=m.TEXT_COLOR)),
            tickfont=dict(color=m.TEXT_COLOR),
            bgcolor=m.PLOT_BG_COLOR,
            outlinecolor=m.SPINE_COLOR,
        ),
        hovertemplate="Exit: %{x}<br>Percentile: %{y}<br>CAGR: %{z:.1f}%<extra></extra>",
    ))

    entry_lbl = (f"Entry: {eyr}  {fmt_price(ep)}  \u00b7  Q{eq*100:.4g}%"
                 if not (p.get("use_lots") and lots)
                 else f"Entry: lots weighted avg  {fmt_price(ep)}")

    fig.update_layout(
        title=dict(text=f"Monte Carlo CAGR \u2014 {entry_lbl}",
                   font=dict(color=m.TITLE_COLOR, size=_FONT_SUBTITLE)),
        paper_bgcolor=m.PLOT_BG_COLOR,
        plot_bgcolor=m.PLOT_BG_COLOR,
        font=dict(color=m.TEXT_COLOR),
        xaxis=dict(title="Exit Year", gridcolor=m.GRID_MAJOR_COLOR,
                   linecolor=m.SPINE_COLOR, tickcolor=m.TEXT_COLOR,
                   fixedrange=True),
        yaxis=dict(title="MC Percentile", gridcolor=m.GRID_MAJOR_COLOR,
                   linecolor=m.SPINE_COLOR, tickcolor=m.TEXT_COLOR,
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
