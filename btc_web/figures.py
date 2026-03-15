"""figures.py — Plotly chart builders for the Bitcoin Projections web app.

Each function takes a ModelData instance and a params dict of control values
and returns a go.Figure ready for dcc.Graph.
"""
#
# Sections:
#   Imports & QR interpolation ..... ~1-48
#   Shared constants ............... ~49-75
#   Theme helpers .................. ~77-220
#   Watermark & MC premium ......... ~270-340
#   Color helpers .................. ~340-400
#   Bubble + QR Overlay ............ ~600-830
#   CAGR Heatmap ................... ~833-1100
#   DCA + Stack-celerator .......... ~1103-1260
#   Retirement ..................... ~1658-1775
#   HODL Supercharger .............. ~1776-2127

from __future__ import annotations

import math
import base64
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# btc_app/ is added to sys.path by app.py before this import
from typing import Any
import _app_ctx
from btc_core import ModelData, yr_to_t, today_t, fmt_price, leo_weighted_entry
try:
    from markov import build_transition_matrix  # noqa: F401 — presence check
    _HAS_MARKOV = True
except ImportError:
    _HAS_MARKOV = False

# MC overlay logic lives in mc_overlay.py
from mc_overlay import (
    _mc_build_traces, _mc_depletion_annots,
    _mc_dca_overlay, _mc_withdraw_overlay,
    _mc_retire_overlay, _mc_supercharge_overlay,
    _mc_heatmap_overlay,
    ghost_traces_from_params,
)


# ── shared constants ─────────────────────────────────────────────────────────

_ANNOT_STAGGER_Y = _app_ctx.ANNOT_STAGGER_Y

_BTC_ORANGE       = _app_ctx.BTC_ORANGE
_TODAY_LINE_COLOR  = "#FF6600"
_TODAY_LINE_WIDTH  = 1.5
_TODAY_LINE_OPACITY = 0.85
_QR_LINE_WIDTH    = 1.8
_INTERP_POINTS    = 1500     # sample points for QR interpolation curves
_MAX_SCATTER_PTS  = 1200     # max data points before downsampling
_FONT_TITLE       = 14
_FONT_SUBTITLE    = 13
_FONT_BODY        = 11
_FONT_LEGEND      = _app_ctx.FONT_LEGEND
_FONT_WATERMARK   = 9
_FONT_ANNOT       = 11       # depletion / edge annotation text

_SHADE_ALPHA      = 0.08     # fill opacity between adjacent quantile lines
_WM_OPACITY       = 0.55     # watermark logo opacity
_WM_SIZE_X        = 0.09     # watermark logo width (fraction of paper)
_WM_SIZE_Y        = 0.12     # watermark logo height (fraction of paper)
_COLORSCALE_STEPS = 256      # dense colorscale points (avoids browser interpolation bugs)
_BISECT_ITERS     = 60       # binary search iterations for Mode B max-withdrawal
_HM_TEXT_THRESHOLD = 0.55    # cell brightness threshold: white text below, dark above

# ── Enhanced font stack (sans-serif base, serif for premium/MC) ──────────
_SANS_FONT = "Inter, Segoe UI, Roboto, Helvetica Neue, Arial, sans-serif"
_FONT_TITLE_LG    = 17
_FONT_BODY_LG     = 13
_FONT_TICK_LG     = 12
_FONT_LEGEND_LG   = 11
_FONT_ANNOT_LG    = 12
_FONT_WATERMARK_LG = 10


def _apply_sans_typography(layout: dict) -> None:
    """Upgrade layout fonts to enhanced sans-serif stack with larger sizes."""
    layout["title"]["font"].update(family=_SANS_FONT, size=_FONT_TITLE_LG)
    layout["font"].update(family=_SANS_FONT, size=_FONT_TICK_LG)
    layout["xaxis"]["title"]["font"].update(family=_SANS_FONT, size=_FONT_BODY_LG)
    layout["yaxis"]["title"]["font"].update(family=_SANS_FONT, size=_FONT_BODY_LG)
    layout["legend"]["font"] = dict(family=_SANS_FONT, size=_FONT_LEGEND_LG)
    for ann in layout.get("annotations", []):
        ann.setdefault("font", {}).update(family=_SANS_FONT, size=_FONT_ANNOT_LG)

# ── shared small helpers ──────────────────────────────────────────────────────


def _fmt_q_label(q: float, prefix: str = "BM") -> str:
    """Format quantile as '{prefix} Q{pct}%' with appropriate precision."""
    pct = q * 100
    ql = f"Q{pct:.4g}%" if pct >= 1 else f"Q{pct:.3g}%"
    return f"{prefix} {ql}" if prefix else ql


def _error_figure(m, title):
    """Return a blank figure with a message title, styled for dark theme."""
    fig = go.Figure()
    fig.update_layout(
        title=title,
        paper_bgcolor=m.PLOT_BG_COLOR,
        plot_bgcolor=m.PLOT_BG_COLOR,
        font=dict(color=m.TEXT_COLOR),
    )
    return fig


# ── shared theme helpers ──────────────────────────────────────────────────────

_LOG_MINOR = dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)",
                  griddash="dot", gridwidth=0.5, dtick="D1")


def _apply_log_y(layout, p):
    """Apply log-Y axis settings when enabled."""
    if p.get("log_y"):
        layout["yaxis"]["type"] = "log"
        layout["yaxis"]["dtick"] = 1
        if p.get("minor_grid"):
            layout["yaxis"]["minor"] = _LOG_MINOR


def _stagger_depletion_annots(deplete_annots, layout):
    """Sort depletion annotations by x and reassign stagger heights."""
    if len(deplete_annots) > 1:
        deplete_annots.sort(key=lambda a: a["x"])
        for i, a in enumerate(deplete_annots):
            a["ay"] = _ANNOT_STAGGER_Y[i % 3]
        layout["annotations"] = deplete_annots


def _build_freq_config(p):
    """Extract frequency string, periods-per-year, and dt from params."""
    freq_str = p.get("freq", "Monthly")
    ppy = FREQ_PPY.get(freq_str, 12)
    dt = 1.0 / ppy
    return freq_str, ppy, dt


def _build_time_array(p, m, default_syr, default_eyr):
    """Extract freq config, build time series, validate year range.

    Returns (syr, eyr, t_start, t_end, ts, dt, freq_str, ppy) or
    (fig, None) tuple when year range is invalid.
    """
    freq_str, ppy, dt = _build_freq_config(p)
    syr = int(p.get("start_yr", default_syr))
    eyr = int(p.get("end_yr", default_eyr))
    if eyr <= syr:
        return _error_figure(m, "Set end year > start year"), None
    t_start = max(yr_to_t(syr, m.genesis), 1.0)
    t_end = yr_to_t(eyr, m.genesis)
    ts = np.arange(t_start, t_end + dt * 0.5, dt)
    if len(ts) == 0:
        return go.Figure(), None
    return syr, eyr, t_start, t_end, ts, dt, freq_str, ppy


def _get_starting_stack(p, default=0.0):
    """Extract starting stack, applying lots override if enabled."""
    start_stack = float(p.get("start_stack", default))
    lots = p.get("lots") or []
    if p.get("use_lots") and lots:
        result = leo_weighted_entry(lots)
        if result:
            start_stack = result[3]  # total_btc
    return start_stack


def _sim_layout(m, p, title, ylabel, ts, t_start, t_end, dt, syr, eyr, shapes=None):
    """Build dark layout with time-series axis, log_y, tick labels."""
    tick_ts, tick_lbls = _year_ticks(syr, eyr, m.genesis,
                                     minor_grid=p.get("minor_grid"))
    layout = _dark_layout(m, title=title, xlabel="Year", ylabel=ylabel)
    layout["yaxis"]["title"]["standoff"] = 5
    _x_end = max(float(ts[-1]), t_end) + dt * 0.15
    layout["xaxis"].update(
        tickvals=tick_ts, ticktext=tick_lbls, tickangle=-45,
        range=[t_start, _x_end],
    )
    _apply_log_y(layout, p)
    layout["shapes"] = shapes or []
    return layout, _x_end


def _apply_mc_overlay(m, p, overlay_fn, overlay_args, traces,
                      deplete_annots, layout, x_end, disp_mode):
    """Integrate MC overlay traces and annotations into chart.
    Returns (mc_traces_list, mc_result)."""
    mc_traces_list, mc_annots, mc_result = overlay_fn(*overlay_args)
    mc_traces_list = _post_mc_overlay(mc_traces_list, p, x_end, disp_mode)
    traces.extend(mc_traces_list)
    if mc_annots:
        mc_annots = [a for a in mc_annots if a["x"] <= x_end]
        deplete_annots.extend(mc_annots)
        layout["annotations"] = deplete_annots
    return mc_traces_list, mc_result


def _dark_layout(m, title, xlabel, ylabel, **kwargs):
    """Base dark-theme layout dict."""
    return dict(
        title=dict(text=title, font=dict(color=m.TITLE_COLOR, size=_FONT_TITLE)),
        paper_bgcolor=m.PLOT_BG_COLOR,
        plot_bgcolor=m.PLOT_BG_COLOR,
        font=dict(color=m.TEXT_COLOR, size=_FONT_BODY),
        xaxis=dict(
            title=dict(text=xlabel, font=dict(color=m.TEXT_COLOR)),
            gridcolor=m.GRID_MAJOR_COLOR, gridwidth=0.6,
            linecolor=m.SPINE_COLOR, tickcolor=m.TEXT_COLOR,
            zerolinecolor=m.GRID_MAJOR_COLOR,
        ),
        yaxis=dict(
            title=dict(text=ylabel, font=dict(color=m.TEXT_COLOR)),
            gridcolor=m.GRID_MAJOR_COLOR, gridwidth=0.6,
            linecolor=m.SPINE_COLOR, tickcolor=m.TEXT_COLOR,
            zerolinecolor=m.GRID_MAJOR_COLOR,
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.85)", bordercolor=m.GRID_MAJOR_COLOR,
            borderwidth=1, font=dict(size=_FONT_LEGEND),
        ),
        margin=dict(l=60, r=20, t=50, b=60),
        **kwargs,
    )


def _year_ticks(start_yr, end_yr, genesis, minor_grid=False):
    """Return (tick_t_values, tick_year_labels) for a year-based x-axis.

    When *minor_grid* is True and the label step > 1, return tickvals for
    every integer year (so each gets a vertical gridline) but only label
    the years at the major step interval.
    """
    span = end_yr - start_yr
    step = 1 if span <= 15 else (2 if span <= 30 else 5)
    if minor_grid and step > 1:
        all_yrs = list(range(start_yr, end_yr + 1))
        ts   = [yr_to_t(y, genesis) for y in all_yrs]
        lbls = [str(y) if (y - start_yr) % step == 0 else "" for y in all_yrs]
    else:
        yrs  = list(range(start_yr, end_yr + 1, step))
        ts   = [yr_to_t(y, genesis) for y in yrs]
        lbls = [str(y) for y in yrs]
    return ts, lbls


# ── Watermark (logo + URL) ────────────────────────────────────────────────────────────────────────────

_LOGO_B64 = None       # 1x — used for on-screen display
_LOGO_B64_ALL = {}     # {1: b64, 2: b64, 3: b64, 4: b64} — for resolution-matched exports
_ASSETS = Path(__file__).parent / "assets"
_WM_FILES = {
    1: _ASSETS / "quantoshi_logo_wm.png",
    2: _ASSETS / "quantoshi_logo_wm_2x.png",
    3: _ASSETS / "quantoshi_logo_wm_3x.png",
    4: _ASSETS / "quantoshi_logo_wm_4x.png",
}
for _scale, _wm_path in _WM_FILES.items():
    try:
        with open(_wm_path, "rb") as _f:
            _b64 = "data:image/png;base64," + base64.b64encode(_f.read()).decode()
            _LOGO_B64_ALL[_scale] = _b64
            if _scale == 1:
                _LOGO_B64 = _b64
    except Exception:
        pass


# ── MC premium figure styling ────────────────────────────────────────────────

_MC_FONT_FAMILY = "Palatino Linotype, Palatino, Georgia, serif"
_MC_TITLE_COLOR = "#996515"          # dark burnished gold — readable on light bg
_MC_LEGEND_BORDER = "#c9a227"        # legend border gold


_MC_LEGEND_POS = {
    "top-left":     dict(x=0.02, y=0.98, xanchor="left",  yanchor="top"),
    "top-right":    dict(x=0.98, y=0.98, xanchor="right", yanchor="top"),
    "bottom-left":  dict(x=0.02, y=0.02, xanchor="left",  yanchor="bottom"),
    "bottom-right": dict(x=0.98, y=0.02, xanchor="right", yanchor="bottom"),
}


def _build_qr_config_text(p: dict, tab: str) -> str:
    """Build compact QR parameter summary for chart annotation.

    Format: QR: Q10%/Q50%/Q85% · $100/mo · 2026–2036 · 1.0 BTC · Log Y
    """
    sel_qs = sorted([float(q) for q in (p.get("selected_qs") or [])])
    if sel_qs:
        qs_str = "/".join(_fmt_q_label(q) for q in sel_qs)
    else:
        qs_str = "none"

    parts = ["QR: " + qs_str]

    # Amount + frequency (DCA/Retire/SC)
    if tab == "dca":
        amt = p.get("amount")
        if amt is not None:
            parts.append(f"${float(amt):,.0f}")
    elif tab in ("ret", "sc"):
        amt = p.get("wd_amount")
        if amt is not None:
            parts.append(f"${float(amt):,.0f}")
    freq = p.get("freq")
    freq_short = {"Daily": "/day", "Weekly": "/wk", "Monthly": "/mo",
                  "Quarterly": "/qtr", "Annually": "/yr"}.get(freq, "")
    if freq_short and tab in ("dca", "ret", "sc"):
        parts[-1] = parts[-1] + freq_short if len(parts) > 1 else parts[0]

    # Year range
    syr = p.get("start_yr")
    eyr = p.get("end_yr")
    if syr and eyr:
        parts.append(f"{int(syr)}\u2013{int(eyr)}")
    elif syr:
        parts.append(f"{int(syr)}")

    # Inflation (Retire/SC)
    infl = p.get("inflation")
    if infl is not None and float(infl) > 0 and tab in ("dca", "ret", "sc"):
        parts.append(f"{float(infl):g}% infl")

    # Stack
    stack = p.get("start_stack")
    if stack is not None and float(stack) > 0 and tab in ("dca", "ret", "sc"):
        parts.append(f"{float(stack):g} BTC")

    # Display toggles
    if p.get("log_y"):
        parts.append("Log Y")

    return " \u00b7 ".join(parts)


def _build_mc_config_text(p: dict, tab: str) -> str:
    """Build compact MC parameter summary for chart annotation.

    Format: MC: 800 sims · 5 bins · Monthly · 2031 · 10yr · Q50% entry · $100/mo · 4% infl · 1.0 BTC
    Also appends a matching JSON download filename.
    """
    start_yr = int(p.get("mc_start_yr", 2031))
    years    = int(p.get("mc_years", 10))
    entry_q  = float(p.get("mc_entry_q", 50))
    sims     = int(p.get("mc_sims", 800))
    freq     = p.get("mc_freq", "Monthly")
    amount   = p.get("mc_amount")
    infl     = p.get("mc_infl")
    stack    = p.get("mc_start_stack")

    parts = [f"MC {tab.upper()}", f"{start_yr}", f"{years}yr",
             f"Q{entry_q:g}%", f"{sims} sims", freq]
    if amount is not None:
        parts.append(f"${float(amount):,.0f}")
    if infl is not None and float(infl) > 0:
        parts.append(f"{float(infl):g}% infl")
    if stack is not None and float(stack) > 0:
        parts.append(f"{float(stack):g} BTC")

    # Filename mirrors the JS _mcFilename() convention
    _eq = round(float(entry_q), 1)
    fn_parts = ["mc", tab, f"yr{start_yr}", f"{years}y", f"q{_eq:g}"]
    if amount is not None:
        fn_parts.append(f"${int(float(amount))}")
    if infl is not None and float(infl) > 0:
        fn_parts.append(f"{float(infl):g}pctInfl")
    if stack is not None and float(stack) > 0:
        fn_parts.append(f"{float(stack):g}btc")
    filename = "_".join(fn_parts) + ".json"

    return " \u00b7 ".join(parts) + "  |  " + filename


def _apply_config_annotation(fig: go.Figure, p: dict, tab: str,
                              show_qr: bool = True, show_mc: bool = False) -> None:
    """Set x-axis title to model config summary — self-documenting exports.

    Builds one or two lines of config text (QR and/or MC) and places them
    as the x-axis title in monospace font below the chart.
    """
    lines = []
    if show_qr:
        lines.append(_build_qr_config_text(p, tab))
    if show_mc:
        lines.append(_build_mc_config_text(p, tab))
    if not lines:
        return
    text = "<br>".join(lines)
    fig.layout.xaxis.title.text = text
    fig.layout.xaxis.title.font.update(
        family="'Courier New', Courier, monospace",
        size=9,
        color="rgba(100,100,100,0.8)",
    )


def _apply_mc_xlabel(fig: go.Figure, p: dict, tab: str) -> None:
    """Set x-axis title to MC simulation config in small monospace font.

    Legacy wrapper — calls unified annotation with MC only.
    """
    _apply_config_annotation(fig, p, tab, show_qr=False, show_mc=True)


def _apply_mc_premium(fig: go.Figure, legend_pos: str = "top-left", hide_xlabel: bool = False) -> None:
    """Upgrade figure fonts / colours for premium MC-rendered charts.

    *legend_pos*: move legend inside the plot area at the named corner.
    Pass ``None`` to keep the default (outside) position.
    """
    # Title: serif, gold, bold, centered, +4px
    fig.layout.title.font.family = _MC_FONT_FAMILY
    fig.layout.title.font.size = _FONT_TITLE + 4
    fig.layout.title.font.color = _MC_TITLE_COLOR
    fig.layout.title.font.weight = "bold"
    fig.layout.title.x = 0.5
    fig.layout.title.xanchor = "center"
    # Global font (tick labels): serif, bold, +3px
    fig.layout.font.family = _MC_FONT_FAMILY
    fig.layout.font.size = _FONT_BODY + 3
    fig.layout.font.weight = "bold"
    # Axis titles: serif, bold, +4px
    if hide_xlabel:
        fig.layout.xaxis.title.text = ""
    fig.layout.xaxis.title.font.family = _MC_FONT_FAMILY
    fig.layout.xaxis.title.font.size = _FONT_BODY + 4
    fig.layout.xaxis.title.font.weight = "bold"
    fig.layout.yaxis.title.font.family = _MC_FONT_FAMILY
    fig.layout.yaxis.title.font.size = _FONT_BODY + 4
    fig.layout.yaxis.title.font.weight = "bold"
    # Legend — bold but not enlarged
    fig.layout.legend.font.weight = "bold"
    fig.layout.legend.font.size = _FONT_LEGEND
    fig.layout.legend.bordercolor = _MC_LEGEND_BORDER
    if legend_pos and legend_pos in _MC_LEGEND_POS:
        pos = _MC_LEGEND_POS[legend_pos]
        fig.layout.legend.x = pos["x"]
        fig.layout.legend.y = pos["y"]
        fig.layout.legend.xanchor = pos["xanchor"]
        fig.layout.legend.yanchor = pos["yanchor"]
        fig.layout.legend.bgcolor = "rgba(255,255,255,0.7)"
    # Top border line to close the plot area
    fig.add_shape(
        type="line", xref="paper", yref="paper",
        x0=0, x1=1, y0=1, y1=1,
        line=dict(color=fig.layout.yaxis.linecolor or "#999", width=1),
    )


def _apply_watermark(fig: go.Figure, pos: str = "bottom-right") -> None:
    """Stamp Quantoshi logo + URL onto a go.Figure.

    pos: 'bottom-right' (default) or 'bottom-left'.
    """
    if pos == "bottom-left":
        img_x, img_xa = 0.0, "left"
        txt_x, txt_xa = 0.075, "left"
    else:
        img_x, img_xa = 1.0, "right"
        txt_x, txt_xa = 0.925, "right"
    if _LOGO_B64:
        fig.add_layout_image(dict(
            source=_LOGO_B64,
            xref="paper", yref="paper",
            x=img_x, y=0.0,
            sizex=_WM_SIZE_X, sizey=_WM_SIZE_Y,
            xanchor=img_xa, yanchor="bottom",
            opacity=_WM_OPACITY,
            layer="above",
        ))
    fig.add_annotation(dict(
        text="quantoshi.xyz",
        xref="paper", yref="paper",
        x=txt_x, y=0.015,
        xanchor=txt_xa, yanchor="bottom",
        showarrow=False,
        font=dict(size=_FONT_WATERMARK, color="rgba(180,180,180,0.65)"),
    ))
    return fig


def _finalize_chart(traces: list, layout: dict, p: dict, tab: str,
                    mc_result: dict | None = None, mc_premium: bool = True
                    ) -> tuple[go.Figure, dict | None]:
    """Shared chart finalization: legend, typography, MC premium, annotations, watermark."""
    layout["showlegend"] = bool(p.get("show_legend", True))
    leg_pos = p.get("legend_pos", "outside")
    if leg_pos != "outside" and leg_pos in _MC_LEGEND_POS:
        pos = _MC_LEGEND_POS[leg_pos]
        layout["legend"].update(
            x=pos["x"], y=pos["y"],
            xanchor=pos["xanchor"], yanchor=pos["yanchor"],
            bgcolor="rgba(255,255,255,0.7)",
        )
    _apply_sans_typography(layout)
    fig = go.Figure(data=traces, layout=go.Layout(**layout))
    show_qr = p.get("show_qr", True)
    show_mc = p.get("show_mc", bool(p.get("mc_enabled")))
    if mc_premium and p.get("mc_enabled"):
        _apply_mc_premium(fig, legend_pos=None, hide_xlabel=True)
    _apply_config_annotation(fig, p, tab, show_qr=show_qr, show_mc=show_mc)
    wm_pos = "bottom-left" if leg_pos == "bottom-right" else "bottom-right"
    _apply_watermark(fig, pos=wm_pos)
    return fig, mc_result


def _price_tickvals(y_lo, y_hi):
    """Decade tick values for a log price y-axis."""
    decades = [0.01, 0.1, 1, 10, 100, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8]
    return [p for p in decades if y_lo <= p <= y_hi]


def _lerp_hex(c1, c2, f):
    """Linearly interpolate between two hex colours, returns hex string."""
    def h2rgb(h):
        h = h.lstrip("#")
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    r1, g1, b1 = h2rgb(c1)
    r2, g2, b2 = h2rgb(c2)
    r = int(r1 + (r2 - r1) * f)
    g = int(g1 + (g2 - g1) * f)
    b = int(b1 + (b2 - b1) * f)
    return f"#{r:02x}{g:02x}{b:02x}"


def _dense_colorscale(color_fn, n=_COLORSCALE_STEPS):
    """Sample color_fn(t) at n uniform points and return an rgb() colorscale.

    Using 256 rgb() entries avoids browser-specific colorscale interpolation
    issues (e.g. Tor Browser canvas rendering) — each cell's colour is
    effectively a direct lookup with sub-1% granularity.
    """
    cs = []
    for k in range(n):
        t = k / (n - 1)
        h = color_fn(t).lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        cs.append([t, f"rgb({r},{g},{b})"])
    return cs


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


# ── Bubble + QR Overlay ───────────────────────────────────────────────────────

def build_bubble_figure(m: ModelData, p: dict[str, Any]) -> go.Figure:
    """
    p keys: selected_qs, shade, xscale, yscale, xmin, xmax, ymin, ymax,
            n_future, show_comp, comp_color, comp_lw,
            show_sup, sup_color, sup_lw,
            show_ols, show_data, show_today, pt_size, pt_alpha,
            stack, show_stack, lots (list of lot dicts), use_lots
    """
    model = _app_ctx.DEFAULT_MODEL
    t_lo = max(yr_to_t(p["xmin"], m.genesis), 0.01)
    t_hi = yr_to_t(p["xmax"], m.genesis)
    y_lo = float(p["ymin"])
    y_hi = float(p["ymax"])
    t_arr = np.linspace(max(t_lo, 0.1), t_hi, _INTERP_POINTS)

    stack = float(p.get("stack", 0)) if p.get("show_stack") else 0.0

    traces = []

    # ── shading between adjacent quantiles ───────────────────────────────────
    sel_qs = sorted([float(q) for q in (p.get("selected_qs") or [])])
    if p.get("shade") and len(sel_qs) >= 2:
        for j in range(len(sel_qs) - 1):
            lo_p = model.price_at(sel_qs[j], t_arr) * (stack if stack > 0 else 1)
            hi_p = model.price_at(sel_qs[j+1], t_arr) * (stack if stack > 0 else 1)
            col  = model.colors.get(sel_qs[j], "#888888")
            traces.append(go.Scatter(
                x=list(t_arr), y=list(lo_p),
                mode="lines", line=dict(width=0),
                showlegend=False, hoverinfo="skip",
            ))
            traces.append(go.Scatter(
                x=list(t_arr), y=list(hi_p),
                mode="lines", line=dict(width=0), fill="tonexty",
                fillcolor=col.replace("#", "rgba(").rstrip(")") if False else _hex_alpha(col, _SHADE_ALPHA),
                showlegend=False, hoverinfo="skip",
            ))

    # ── quantile lines ────────────────────────────────────────────────────────
    for q in sel_qs:
        if q not in model.fits:
            continue
        prices = model.price_at(q, t_arr) * (stack if stack > 0 else 1)
        lbl = _fmt_q_label(q)
        if stack > 0:
            lbl += f"  \u2192  {fmt_price(float(prices[-1]))}"
        col = model.colors.get(q, "#888888")
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
                    line=dict(color=col, width=_QR_LINE_WIDTH * 0.8, dash=mdl.dash_style),
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
                line=dict(color="#8B4513", width=_QR_LINE_WIDTH * 0.8, dash=mdl.dash_style),
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
            name=f"Bubble composite (N={n})  R²={m.bm_r2:.4f}",
            line=dict(color=p.get("comp_color", "#FFD700"),
                      width=float(p.get("comp_lw", 2.0))),
        ))

    # ── historical price data ─────────────────────────────────────────────────
    if p.get("show_data"):
        mask  = (m.price_years >= t_lo) & (m.price_years <= t_hi)
        x_sc  = m.price_years[mask]
        y_sc  = m.price_prices[mask] * (stack if stack > 0 else 1)
        d_sc  = [m.price_dates[i] for i in range(len(m.price_dates)) if mask[i]]
        # Downsample to ≤1 200 points — imperceptible on log scale but cuts
        # figure JSON ~50 % and serialisation time meaningfully.
        _MAX_PTS = _MAX_SCATTER_PTS
        n_pts = len(x_sc)
        if n_pts > _MAX_PTS:
            stride = max(1, n_pts // _MAX_PTS)
            idx   = np.arange(0, n_pts, stride)
            x_sc  = x_sc[idx]
            y_sc  = y_sc[idx]
            d_sc  = [d_sc[i] for i in idx]
        traces.append(go.Scatter(
            x=list(x_sc), y=list(y_sc),
            mode="markers", name="Price data",
            marker=dict(color=m.DATA_COLOR, size=max(2, int(p.get("pt_size", 3))),
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

    # ── today line ────────────────────────────────────────────────────────────
    shapes = []
    if p.get("show_today"):
        td = today_t(m.genesis)
        if t_lo <= td <= t_hi:
            shapes.append(dict(
                type="line", x0=td, x1=td, y0=y_lo, y1=y_hi,
                line=dict(color=_TODAY_LINE_COLOR, dash="dash", width=_TODAY_LINE_WIDTH),
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
            y_log_update["dtick"] = 1  # decade labels only (drop 2× and 5×)
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


def _hex_alpha(hex_color, alpha):
    """Convert hex color + alpha float to an rgba() CSS string."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ── CAGR Heatmap — shared helpers ─────────────────────────────────────────────

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


def _heatmap_cell_annots(mc, mp, mm, vfmt, hm_stk, zmin, zmax, cell_fs):
    """Build cell text annotation dicts for a CAGR heatmap."""
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
                cell_norm = (vc2 - zmin) / max(zmax - zmin, 1e-6)
                txt_col = "#ffffff" if cell_norm < _HM_TEXT_THRESHOLD else "#111111"
                annots.append(dict(
                    x=ci, y=ri,
                    text=tx.replace("\n", "<br>"),
                    showarrow=False,
                    font=dict(size=cell_fs, color=txt_col,
                              family=_SANS_FONT, weight="bold"),
                    xref="x", yref="y",
                ))
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
    eyr = int(p.get("entry_yr", 2020))
    eq  = float(p.get("entry_q", 50)) / 100.0   # stored as percentage (e.g. 7.5 → 0.075)
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
        return _error_figure(m, "No data — adjust Entry / Exit settings")

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
                                   int(p.get("cell_font_size", 9)))

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
        title=dict(text=f"CAGR Heatmap — {entry_lbl}",
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
    # (per-annotation weight is unreliable on initial mobile render).
    _apply_config_annotation(fig, p, "hm", show_qr=True, show_mc=False)
    _apply_watermark(fig)
    return fig


def build_mc_heatmap_figure(m: ModelData, p: dict[str, Any]) -> tuple[go.Figure, dict | None]:
    """Build a standalone MC heatmap figure from MC-simulated CAGR percentiles.
    Returns (fig, mc_result) or (empty_fig, None).
    """
    model = _app_ctx.DEFAULT_MODEL
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
        return _error_figure(m, "No data — adjust Entry / Exit settings"), None

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
                                   int(p.get("cell_font_size", 9)))

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
        title=dict(text=f"Monte Carlo CAGR — {entry_lbl}",
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


# ── DCA helpers ──────────────────────────────────────────────────────────────

def _dca_sc_overlay(m, p, ts, sel_qs, start_stack, all_prices, disp_mode, ppy):
    """Run Stack-celerator overlay simulation for DCA tab.

    Returns (sc_traces, all_sc_usd_vals, all_sc_btc_vals).
    """
    model = _app_ctx.DEFAULT_MODEL
    from _app_ctx import _compute_sc_loan

    principal    = float(p.get("sc_loan_amount", 0))
    sc_rate      = float(p.get("sc_rate", 13.0))
    sc_live      = float(p.get("sc_live_price", 0))
    loan_type    = p.get("sc_loan_type", "interest_only")
    term_months  = float(p.get("sc_term_months", 12))
    sc_repeats   = int(p.get("sc_repeats", 0))
    entry_mode   = p.get("sc_entry_mode", "live")
    custom_price = float(p.get("sc_custom_price", 0))
    tax_rate     = max(0.0, min(float(p.get("sc_tax_rate", 0.33)), 0.9999))
    sc_rollover  = bool(p.get("sc_rollover", False)) and loan_type == "interest_only"
    amount       = float(p.get("amount", 100))

    term_periods = max(1, round(term_months * ppy / 12))
    n_cycles     = 1 + sc_repeats
    r            = sc_rate / 100.0 / ppy

    principal, pmt, _ = _compute_sc_loan(principal, amount, r, term_periods, loan_type)
    sc_dca_amt = amount - pmt

    sc_traces       = []
    all_sc_usd_vals = {}
    all_sc_btc_vals = {}

    if principal <= 0:
        return sc_traces, all_sc_usd_vals, all_sc_btc_vals

    for q in sel_qs:
        if q not in model.fits:
            continue
        sc_stack    = start_stack
        outstanding = 0.0
        sc_vals     = np.empty(len(ts))
        sc_prices   = np.empty(len(ts))
        _prices_q   = all_prices[q]
        for i, t in enumerate(ts):
            price           = _prices_q[i]
            cycle_idx       = i // term_periods
            period_in_cycle = i % term_periods

            if cycle_idx < n_cycles:
                if period_in_cycle == 0:
                    if cycle_idx == 0:
                        if entry_mode == "live" and sc_live > 0:
                            ep = sc_live
                        elif entry_mode == "custom" and custom_price > 0:
                            ep = custom_price
                        else:
                            ep = price
                        sc_stack += principal / ep
                    elif not sc_rollover:
                        ep = price
                        sc_stack += principal / ep
                    outstanding = principal

                sc_stack += sc_dca_amt / price

                # ── Loan repayment logic ──────────────────────────────────
                # Amortizing: each period pays interest + principal in fiat
                #   (no BTC sold, tax has no effect on amortizing loans).
                # Interest-only: at cycle end, sell BTC to repay principal.
                #   Tax applies ONLY to capital gain (sell_price - buy_price),
                #   not the full proceeds. If selling at a loss (price <= ep),
                #   no tax is owed. Rollover defers all repayment to sim end.
                if loan_type == "amortizing":
                    interest_p  = outstanding * r
                    principal_p = pmt - interest_p
                    outstanding = max(outstanding - principal_p, 0.0)

                if loan_type == "interest_only" and period_in_cycle == term_periods - 1:
                    if sc_rollover:
                        pass
                    else:
                        # Sell BTC to repay principal. Tax is on capital gain
                        # only: (sell_price - buy_price) per BTC. net_per_btc
                        # is what you keep per BTC sold after tax on the gain.
                        gain_per_btc = max(price - ep, 0.0)
                        net_per_btc  = price - tax_rate * gain_per_btc
                        sc_stack    -= principal / net_per_btc
                        sc_stack     = max(sc_stack, 0.0)
                        outstanding  = 0.0
            else:
                sc_stack += amount / price

            sc_vals[i]   = sc_stack
            sc_prices[i] = price

        # Deduct outstanding balance at simulation end
        if outstanding > 1e-8 and sc_prices[-1] > 0:
            final_price  = sc_prices[-1]
            gain_per_btc = max(final_price - ep, 0.0)
            net_per_btc  = final_price - tax_rate * gain_per_btc
            sc_vals[-1]  = max(sc_vals[-1] - outstanding / net_per_btc, 0.0)

        all_sc_usd_vals[q] = sc_vals * sc_prices
        all_sc_btc_vals[q] = sc_vals.copy()

        if disp_mode == "usd":
            y_sc     = sc_vals * sc_prices
            final_sc = fmt_price(float(y_sc[-1]))
        else:
            y_sc      = sc_vals
            final_usd = fmt_price(float(all_sc_usd_vals[q][-1]))
            final_sc  = f"{float(sc_vals[-1]):.4f} BTC  ({final_usd})"

        lbl_sc = f"SC {_fmt_q_label(q)}" + f"  \u2192  {final_sc}"
        col = model.colors.get(q, "#888888")
        sc_traces.append(go.Scatter(
            x=list(ts), y=list(y_sc), mode="lines", name=lbl_sc,
            line=dict(color=col, width=_QR_LINE_WIDTH, dash="dash"),
        ))

    return sc_traces, all_sc_usd_vals, all_sc_btc_vals


def _clip_mc_traces(mc_traces, x_max):
    """Clip MC traces to x ≤ x_max so they don't extend beyond the visible chart.

    Traces whose data is entirely beyond x_max are dropped.
    Fill='tonexty' alignment is preserved because paired traces are
    clipped to the same cutoff index (x arrays are identical).
    """
    clipped = []
    for tr in mc_traces:
        x = tr.x
        if x is None or len(x) == 0:
            clipped.append(tr)
            continue
        x_arr = np.asarray(x, dtype=float)
        n = int(np.searchsorted(x_arr, x_max, side='right'))
        if n == 0:
            continue
        if n < len(x_arr):
            tr.x = list(x_arr[:n])
            tr.y = list(np.asarray(tr.y, dtype=float)[:n])
        clipped.append(tr)
    return clipped


def _post_mc_overlay(mc_traces, p, x_end, disp_mode):
    """Prepend ghost fan traces and clip to visible chart range."""
    ghost = ghost_traces_from_params(p, x_end, disp_mode)
    return _clip_mc_traces(ghost + mc_traces, x_end)


def _find_mc_median_trace(mc_traces):
    """Find the dotted MC median trace. Returns (x_list, y_list) or (None, None)."""
    for tr in mc_traces:
        if getattr(getattr(tr, "line", None), "dash", None) != "dot":
            continue
        mx = list(tr.x) if tr.x is not None else []
        my = list(tr.y) if tr.y is not None else []
        if mx and my:
            return mx, my
    return None, None


def _mc_median_annot(mc_traces, disp_mode, m, ts_end, t_start, t_end,
                     syr, eyr, btc_fmt=".2f", estimate_usd=True):
    """Build MC median edge annotation for Retire/SC tabs, or return None.

    btc_fmt: format spec for BTC value (e.g. ".2f" for Retire, ".4f" for SC).
    estimate_usd: if True (Retire), estimate USD from BTC via model.price_at(Q50%).
    """
    model = _app_ctx.DEFAULT_MODEL
    mx, my = _find_mc_median_trace(mc_traces)
    if mx is None:
        return None
    mc_y_final = float(my[-1])
    if mc_y_final <= 0:
        return None
    if disp_mode == "usd":
        mc_lbl = fmt_price(mc_y_final)
        _mc_btc, _mc_usd = 0, mc_y_final
    else:
        if estimate_usd:
            mc_t = np.array([max(float(mx[-1]), 0.5)])
            _mc_usd = mc_y_final * float(model.price_at(0.5, mc_t)[0])
            mc_lbl = f"{mc_y_final:{btc_fmt}} \u20bf  {fmt_price(_mc_usd)}"
        else:
            mc_lbl = f"{mc_y_final:{btc_fmt}} \u20bf"
            _mc_usd = 0
        _mc_btc = mc_y_final
    mc_x_last = float(mx[-1])
    if mc_x_last < ts_end:
        ann_yr = int(syr + (mc_x_last - t_start)
                     / max(t_end - t_start, 1e-6) * (eyr - syr))
        mc_lbl = f"\u2248{ann_yr}  {mc_lbl}"
    return dict(
        x_arr=mx, y_arr=my,
        label=f"MC {mc_lbl}",
        short_label=_fmt_short(_mc_btc, _mc_usd),
        color="#F7931A", y_last=mc_y_final)


def _fmt_short(btc, usd):
    """Compact annotation label: B0.32/$1.23M"""
    if usd >= 1e9:
        u = f"${usd/1e9:.2f}B"
    elif usd >= 1e6:
        u = f"${usd/1e6:.2f}M"
    elif usd >= 1e3:
        u = f"${usd/1e3:.1f}K"
    else:
        u = f"${usd:.0f}"
    return f"B{btc:.2f}/{u}"


def _edge_text_trace(x_arr, y_arr, label, color, *, log_y=False,
                     textpos_override=None):
    """Place a text-trace annotation at the last data point.

    Automatically positions text based on the trace slope at the
    endpoint so the label sits on the opposite side of the line's
    approach direction, avoiding overlap:
      ascending  → "bottom left"  (text below the rising line)
      descending → "top left"     (text above the falling line)
      flat       → "middle left"

    If textpos_override is given, it takes precedence (used by
    _resolve_edge_overlaps to spread clustered labels).
    """
    y_last = float(y_arr[-1])
    if textpos_override:
        textpos = textpos_override
    else:
        # Slope direction from the last two data points
        if len(y_arr) >= 2:
            y_prev = float(y_arr[-2])
            if log_y and y_last > 0 and y_prev > 0:
                slope_sign = np.sign(np.log10(y_last) - np.log10(y_prev))
            else:
                slope_sign = np.sign(y_last - y_prev)
        else:
            slope_sign = 0
        if slope_sign > 0:
            textpos = "bottom left"
        elif slope_sign < 0:
            textpos = "top left"
        else:
            textpos = "middle left"

    return go.Scatter(
        x=[float(x_arr[-1])], y=[y_last],
        mode="markers+text",
        marker=dict(size=7, color=color, symbol="circle"),
        text=[f"{label}  "],
        textposition=textpos,
        textfont=dict(size=_FONT_ANNOT, color=color),
        showlegend=False, hoverinfo="skip",
        cliponaxis=False,
    )


# Overlap threshold: annotations closer than this fraction of the y-axis
# visual range (or 0.08 log-decades in log scale) are considered overlapping.
_OVERLAP_FRAC = 0.06       # 6% of linear axis range
_OVERLAP_LOG  = 0.12       # log-decades

# When 4+ annotations overlap in a cluster, consolidate into one label.
_CONSOLIDATE_THRESHOLD = 4


def _resolve_edge_annotations(pending, log_y):
    """Take a list of pending edge annotations and return go.Scatter traces
    with overlaps resolved.

    Each entry in *pending* is a dict with keys:
        x_arr, y_arr, label, color, y_last (float), short_label (str)

    Strategy:
    1. Sort by y_last (ascending in linear, log-ascending in log).
    2. Walk sorted list, grouping consecutive items whose y values are
       within the overlap threshold.
    3. Clusters of 1: emit normally (slope-based textposition).
    4. Clusters of 2–3: alternate textposition top/bottom to spread apart.
    5. Clusters of 4+: consolidate into a single merged label at the
       median y position.
    """
    if not pending:
        return []

    # Sort by y_last (log-space if log scale)
    def sort_key(item):
        y = item["y_last"]
        if log_y and y > 0:
            return np.log10(y)
        return y

    pending.sort(key=sort_key)

    # Determine axis range for threshold calculation
    y_vals = [item["y_last"] for item in pending]
    if log_y:
        pos_vals = [v for v in y_vals if v > 0]
        if len(pos_vals) >= 2:
            threshold = _OVERLAP_LOG
        else:
            threshold = _OVERLAP_LOG
    else:
        y_min, y_max = min(y_vals), max(y_vals)
        y_span = y_max - y_min if y_max > y_min else abs(y_max) * 0.1 or 1.0
        threshold = y_span * _OVERLAP_FRAC

    # Group into clusters of nearby annotations
    clusters = []
    current_cluster = [pending[0]]
    for item in pending[1:]:
        prev = current_cluster[-1]
        if log_y:
            y_cur = np.log10(item["y_last"]) if item["y_last"] > 0 else -99
            y_prv = np.log10(prev["y_last"]) if prev["y_last"] > 0 else -99
            gap = abs(y_cur - y_prv)
        else:
            gap = abs(item["y_last"] - prev["y_last"])
        if gap <= threshold:
            current_cluster.append(item)
        else:
            clusters.append(current_cluster)
            current_cluster = [item]
    clusters.append(current_cluster)

    # Emit traces for each cluster.
    # Items are sorted ascending by y_last. To avoid visual crossing,
    # lower annotations get "bottom left" (text below point) and upper
    # annotations get "top left" (text above point). For singletons
    # between two clusters we pick the side with more room.
    traces = []
    n_total = sum(len(c) for c in clusters)
    flat_idx = 0  # running index across all items
    for cluster in clusters:
        if len(cluster) == 1:
            item = cluster[0]
            # Position based on rank: bottom half → "bottom left",
            # top half → "top left", middle → slope-based default
            if n_total == 1:
                pos = None  # let slope decide
            elif flat_idx < n_total / 2:
                pos = "bottom left"
            else:
                pos = "top left"
            traces.append(_edge_text_trace(
                item["x_arr"], item["y_arr"], item["label"],
                item["color"], log_y=log_y, textpos_override=pos))
            flat_idx += 1
        elif len(cluster) < _CONSOLIDATE_THRESHOLD:
            # Spread: lowest gets bottom, highest gets top
            for i, item in enumerate(cluster):
                if i == 0:
                    pos = "bottom left"
                elif i == len(cluster) - 1:
                    pos = "top left"
                else:
                    pos = "middle left"
                traces.append(_edge_text_trace(
                    item["x_arr"], item["y_arr"], item["label"],
                    item["color"], log_y=log_y, textpos_override=pos))
                flat_idx += 1
        else:
            # Consolidate: merge into single label at median position
            mid_idx = len(cluster) // 2
            anchor = cluster[mid_idx]
            parts = [item["short_label"] for item in cluster]
            merged_label = " · ".join(parts)
            # Use first item's color (or neutral gray for mixed)
            colors = {item["color"] for item in cluster}
            merged_color = cluster[0]["color"] if len(colors) == 1 else "#AAAAAA"
            traces.append(_edge_text_trace(
                anchor["x_arr"], anchor["y_arr"], merged_label,
                merged_color, log_y=log_y, textpos_override="top left"))
            # Still place dot markers at each original position (no text)
            for item in cluster:
                if item is anchor:
                    continue
                traces.append(go.Scatter(
                    x=[float(item["x_arr"][-1])],
                    y=[item["y_last"]],
                    mode="markers",
                    marker=dict(size=7, color=item["color"], symbol="circle"),
                    showlegend=False, hoverinfo="skip",
                    cliponaxis=False,
                ))
            flat_idx += len(cluster)
    return traces


# ── DCA Accumulator ───────────────────────────────────────────────────────────

def build_dca_figure(m: ModelData, p: dict[str, Any]) -> tuple[go.Figure, dict | None]:
    """
    p keys: start_yr, end_yr, start_stack, amount, freq, disp_mode,
            selected_qs, log_y, show_today,
            lots, use_lots
    """
    model = _app_ctx.DEFAULT_MODEL
    ta = _build_time_array(p, m, 2024, 2035)
    if ta[1] is None:
        return ta[0], None
    syr, eyr, t_start, t_end, ts, dt, freq_str, ppy = ta

    start_stack = _get_starting_stack(p, default=0)

    amount    = float(p.get("amount", 100))
    inflation = float(p.get("inflation", 0)) / 100.0
    disp_mode = p.get("disp_mode", "btc")
    sel_qs    = sorted([float(q) for q in (p.get("selected_qs") or [])])

    traces = []
    all_btc_vals = {}  # q -> BTC balance array
    all_usd_vals = {}  # q -> USD value array (for annotations + title)
    all_prices   = {}  # q -> price array — reused by SC loop to avoid redundant qr_price calls
    ts_clamped = np.maximum(ts, 0.5)
    adj_amount_arr = amount * ((1 + inflation) ** (ts - t_start))
    for q in sel_qs:
        if q not in model.fits:
            continue
        prices_q = model.price_at(q, ts_clamped)
        vals = start_stack + np.cumsum(adj_amount_arr / prices_q)
        all_btc_vals[q] = vals
        all_usd_vals[q] = vals * prices_q
        all_prices[q]   = prices_q          # save for SC loop below

        if disp_mode == "usd":
            y_vals    = vals * prices_q
            final_lbl = fmt_price(float(y_vals[-1]))
        else:
            y_vals    = vals
            final_usd = fmt_price(float(all_usd_vals[q][-1]))
            final_lbl = f"{float(vals[-1]):.4f} BTC  ({final_usd})"

        lbl = _fmt_q_label(q) + f"  →  {final_lbl}"
        col = model.colors.get(q, "#888888")
        traces.append(go.Scatter(
            x=list(ts), y=list(y_vals), mode="lines", name=lbl,
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
                prices_q = mdl.price_at(q, ts_clamped)
                vals = start_stack + np.cumsum(adj_amount_arr / prices_q)
                if disp_mode == "usd":
                    y_vals = vals * prices_q
                    final_lbl = fmt_price(float(y_vals[-1]))
                else:
                    y_vals = vals
                    final_usd = fmt_price(float(vals[-1] * prices_q[-1]))
                    final_lbl = f"{float(vals[-1]):.4f} BTC  ({final_usd})"
                col = mdl.colors.get(q, "#888888")
                traces.append(go.Scatter(
                    x=list(ts), y=list(y_vals), mode="lines",
                    name=f"{mdl.name} {_fmt_q_label(q, '')}  \u2192  {final_lbl}",
                    line=dict(color=col, width=_QR_LINE_WIDTH * 0.8, dash=mdl.dash_style),
                    legendgroup=mdl.short_name,
                    legendgrouptitle_text=mdl.name,
                ))
        else:
            # Non-quantized: single trajectory DCA simulation
            prices_q = mdl.price_at(0.5, ts_clamped)
            vals = start_stack + np.cumsum(adj_amount_arr / prices_q)
            if disp_mode == "usd":
                y_vals = vals * prices_q
                final_lbl = fmt_price(float(y_vals[-1]))
            else:
                y_vals = vals
                final_usd = fmt_price(float(vals[-1] * prices_q[-1]))
                final_lbl = f"{float(vals[-1]):.4f} BTC  ({final_usd})"
            traces.append(go.Scatter(
                x=list(ts), y=list(y_vals), mode="lines",
                name=f"{mdl.name}  \u2192  {final_lbl}",
                line=dict(color="#8B4513", width=_QR_LINE_WIDTH * 0.8, dash=mdl.dash_style),
                legendgroup=mdl.short_name,
            ))

    shapes = []
    if p.get("show_today"):
        td = today_t(m.genesis)
        if t_start <= td <= t_end:
            shapes.append(dict(
                type="line", x0=td, x1=td, y0=0, y1=1,
                yref="paper", line=dict(color=_TODAY_LINE_COLOR, dash="dash", width=_TODAY_LINE_WIDTH),
                opacity=_TODAY_LINE_OPACITY,
            ))

    # ── Total cost & value ratio ────────────────────────────────────────────
    n_periods = len(ts)
    total_spent = amount * n_periods
    freq_short = freq_str.lower()[:-2] if freq_str.endswith("ly") else freq_str

    # Build title — add cost info and median ROI if we have quantile data
    title_line = f"Bitcoin DCA — {fmt_price(amount)}/{freq_short}"
    title_line += f"  ·  {fmt_price(total_spent)} invested over {n_periods} periods"
    _qr_med_final = None
    if all_usd_vals:
        _qr_med_final = float(np.median([v[-1] for v in all_usd_vals.values()]))
        roi = _qr_med_final / total_spent if total_spent > 0 else 0
        title_line += f"<br>QR median {fmt_price(_qr_med_final)}  ·  {roi:.1f}\u00d7"

    ylabel = "USD Value" if disp_mode == "usd" else "BTC Balance"
    layout, _x_end = _sim_layout(m, p, title_line, ylabel, ts, t_start, t_end, dt, syr, eyr, shapes)

    # ── Stack-celerator overlay ─────────────────────────────────────────────
    all_sc_usd_vals = {}
    all_sc_btc_vals = {}
    if p.get("sc_enabled") and sel_qs:
        sc_traces, all_sc_usd_vals, all_sc_btc_vals = _dca_sc_overlay(
            m, p, ts, sel_qs, start_stack, all_prices, disp_mode, ppy)
        traces.extend(sc_traces)

    # ── SC factor (ratio of median SC to median DCA at end date) ─────────────
    sc_factor_val = None
    if all_sc_usd_vals and all_usd_vals:
        _sc_end  = float(np.median([v[-1] for v in all_sc_usd_vals.values()]))
        _dca_end = float(np.median([v[-1] for v in all_usd_vals.values()]))
        if _dca_end > 0:
            sc_factor_val = _sc_end / _dca_end

    # ── Stack-celeration factor → append to title ────────────────────────────
    if sc_factor_val is not None:
        layout["title"]["text"] += (
            f"<br><b>Stack-celeration: {sc_factor_val:.2f}\u00d7</b>"
        )

    # ── Monte Carlo fan overlay ─────────────────────────────────────────────
    _is_log = bool(p.get("log_y"))
    mc_result = None
    mc_fan_usd = {}
    mc_traces = []
    if _HAS_MARKOV and p.get("mc_enabled"):
        mc_traces, mc_result, mc_fan_usd = _mc_dca_overlay(m, p, ts, t_start, dt, start_stack, disp_mode)
        mc_traces = _post_mc_overlay(mc_traces, p, _x_end, disp_mode)
        traces.extend(mc_traces)
        # MC median text trace annotation — collected into pending below
        # MC median final value + multiplier → append to title
        if mc_fan_usd and 0.50 in mc_fan_usd and len(mc_fan_usd[0.50]) > 0:
            mc_med_final = float(mc_fan_usd[0.50][-1])
            mc_roi = mc_med_final / total_spent if total_spent > 0 else 0
            layout["title"]["text"] += f"  ·  MC median {fmt_price(mc_med_final)}  ·  {mc_roi:.1f}\u00d7"

    # ── Right-edge annotations (text traces for alignment stability) ─────────
    _pending_annots = []
    if p.get("annotate") and all_usd_vals:
        for q in sel_qs:
            if q not in all_usd_vals:
                continue
            col = model.colors.get(q, "#888888")
            y_arr = all_btc_vals[q] if disp_mode == "btc" else all_usd_vals[q]
            _btc_f = float(all_btc_vals[q][-1])
            _usd_f = float(all_usd_vals[q][-1])
            _pending_annots.append(dict(
                x_arr=ts, y_arr=y_arr,
                label=f"Q{q*100:g}% {fmt_price(_usd_f)}",
                short_label=_fmt_short(_btc_f, _usd_f),
                color=col, y_last=float(y_arr[-1])))
        for q in all_sc_usd_vals:
            col = model.colors.get(q, "#888888")
            sc_y = all_sc_btc_vals[q] if disp_mode == "btc" else all_sc_usd_vals[q]
            _btc_f = float(all_sc_btc_vals[q][-1])
            _usd_f = float(all_sc_usd_vals[q][-1])
            _pending_annots.append(dict(
                x_arr=ts, y_arr=sc_y,
                label=f"SC Q{q*100:g}% {fmt_price(_usd_f)}",
                short_label=_fmt_short(_btc_f, _usd_f),
                color=col, y_last=float(sc_y[-1])))
    if p.get("annotate") and mc_fan_usd and 0.50 in mc_fan_usd:
        mc_med_usd = mc_fan_usd[0.50]
        if len(mc_med_usd) > 0:
            _mx, _my = _find_mc_median_trace(mc_traces)
            if _mx is not None:
                _mc_usd_f = float(mc_med_usd[-1])
                _mc_btc_f = float(_my[-1]) if disp_mode == "btc" else 0
                _pending_annots.append(dict(
                    x_arr=_mx, y_arr=_my,
                    label=f"MC {fmt_price(_mc_usd_f)}",
                    short_label=_fmt_short(_mc_btc_f, _mc_usd_f),
                    color=_BTC_ORANGE, y_last=float(_my[-1])))
    traces.extend(_resolve_edge_annotations(_pending_annots, _is_log))

    return _finalize_chart(traces, layout, p, "dca", mc_result)


# Re-export from _app_ctx for backward compat (used by chart builders and callbacks)
FREQ_PPY = _app_ctx.FREQ_PPY
_FREQ_STEP_DAYS = _app_ctx.FREQ_STEP_DAYS


# ── BTC Retireator ────────────────────────────────────────────────────────────

def build_retire_figure(m: ModelData, p: dict[str, Any]) -> tuple[go.Figure, dict | None]:
    """
    p keys: start_yr, end_yr, start_stack, wd_amount, freq, inflation,
            disp_mode, selected_qs, log_y, annotate,
            lots, use_lots
    """
    model = _app_ctx.DEFAULT_MODEL
    ta = _build_time_array(p, m, 2025, 2045)
    if ta[1] is None:
        return ta[0], None
    syr, eyr, t_start, t_end, ts, dt, freq_str, ppy = ta

    start_stack = _get_starting_stack(p, default=1.0)

    wd_amount = float(p.get("wd_amount", 5000))
    inflation = float(p.get("inflation", 0)) / 100.0
    disp_mode = p.get("disp_mode", "btc")
    sel_qs    = sorted([float(q) for q in (p.get("selected_qs") or [])])

    traces   = []
    deplete_annots = []
    all_btc_vals = {}  # q -> BTC balance array
    all_y_vals   = {}  # q -> plotted y-values (for text trace annotations)

    ts_clamped = np.maximum(ts, 0.5)
    adj_wd_arr = wd_amount * ((1 + inflation) ** (ts - t_start))
    for q in sel_qs:
        if q not in model.fits:
            continue
        prices = model.price_at(q, ts_clamped)
        vals = np.maximum(start_stack - np.cumsum(adj_wd_arr / prices), 0.0)
        all_btc_vals[q] = vals

        if disp_mode == "usd":
            y_vals = vals * prices
            final_lbl = fmt_price(float(y_vals[-1]))
        else:
            y_vals = vals
            final_usd = fmt_price(float(vals[-1]) * float(prices[-1]))
            final_lbl = f"{float(vals[-1]):.4f} BTC  ({final_usd})"
        all_y_vals[q] = y_vals

        lbl = _fmt_q_label(q) + f"  \u2192  {final_lbl}"
        col = model.colors.get(q, "#888888")
        traces.append(go.Scatter(
            x=list(ts), y=list(y_vals), mode="lines", name=lbl,
            line=dict(color=col, width=_QR_LINE_WIDTH),
        ))

        # depletion annotation — always shown
        depl_i = next((i for i, v in enumerate(vals) if v <= 0), None)
        if depl_i is not None:
            depl_t = ts[depl_i]
            depl_yr = int(syr + (depl_t - t_start) * (eyr - syr) / max(t_end - t_start, 1e-6))
            _ay = _ANNOT_STAGGER_Y[len(deplete_annots) % 3]
            deplete_annots.append(dict(
                x=depl_t, xref="x",
                y=0, yref="paper",
                ax=28, ay=_ay,
                text=f"≈{depl_yr}",
                showarrow=True, arrowhead=2, arrowsize=1,
                arrowcolor=col,
                font=dict(size=_FONT_ANNOT, color=col),
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
                prices = mdl.price_at(q, ts_clamped)
                vals = np.maximum(start_stack - np.cumsum(adj_wd_arr / prices), 0.0)
                if disp_mode == "usd":
                    y_vals = vals * prices
                    final_lbl = fmt_price(float(y_vals[-1]))
                else:
                    y_vals = vals
                    final_usd = fmt_price(float(vals[-1]) * float(prices[-1]))
                    final_lbl = f"{float(vals[-1]):.4f} BTC  ({final_usd})"
                col = mdl.colors.get(q, "#888888")
                traces.append(go.Scatter(
                    x=list(ts), y=list(y_vals), mode="lines",
                    name=f"{mdl.name} {_fmt_q_label(q, '')}  \u2192  {final_lbl}",
                    line=dict(color=col, width=_QR_LINE_WIDTH * 0.8, dash=mdl.dash_style),
                    legendgroup=mdl.short_name,
                    legendgrouptitle_text=mdl.name,
                ))
        else:
            # Non-quantized: single trajectory withdrawal simulation
            prices = mdl.price_at(0.5, ts_clamped)
            vals = np.maximum(start_stack - np.cumsum(adj_wd_arr / prices), 0.0)
            if disp_mode == "usd":
                y_vals = vals * prices
                final_lbl = fmt_price(float(y_vals[-1]))
            else:
                y_vals = vals
                final_usd = fmt_price(float(vals[-1]) * float(prices[-1]))
                final_lbl = f"{float(vals[-1]):.4f} BTC  ({final_usd})"
            traces.append(go.Scatter(
                x=list(ts), y=list(y_vals), mode="lines",
                name=f"{mdl.name}  \u2192  {final_lbl}",
                line=dict(color="#8B4513", width=_QR_LINE_WIDTH * 0.8, dash=mdl.dash_style),
                legendgroup=mdl.short_name,
            ))

    shapes = []

    ylabel = "USD Value" if disp_mode == "usd" else "BTC Remaining"
    title = f"Bitcoin Retireator — {fmt_price(wd_amount)}/{freq_str.lower()[:-2] if freq_str.endswith('ly') else freq_str}"
    layout, _x_end = _sim_layout(m, p, title, ylabel, ts, t_start, t_end, dt, syr, eyr, shapes)
    layout["annotations"] = deplete_annots

    # ── Monte Carlo fan overlay ─────────────────────────────────────────────
    mc_traces_list = []
    mc_result = None
    if _HAS_MARKOV and p.get("mc_enabled"):
        mc_traces_list, mc_result = _apply_mc_overlay(
            m, p, _mc_retire_overlay,
            (m, p, ts, t_start, t_end, dt, start_stack, disp_mode, len(deplete_annots)),
            traces, deplete_annots, layout, _x_end, disp_mode)

    _stagger_depletion_annots(deplete_annots, layout)

    # ── Right-edge annotations (text traces for alignment stability) ─────────
    _is_log = bool(p.get("log_y"))
    _pending_annots = []
    if p.get("annotate") and all_y_vals:
        ts_end_arr = np.maximum(np.array([ts[-1]]), 0.5)
        for q in sel_qs:
            if q not in all_y_vals:
                continue
            btc_final = float(all_btc_vals[q][-1])
            if btc_final <= 0:
                continue
            col = model.colors.get(q, "#888888")
            usd_final = btc_final * float(model.price_at(q, ts_end_arr)[0])
            qpfx = f"Q{q*100:g}%"
            if disp_mode == "usd":
                lbl = f"{qpfx} {fmt_price(usd_final)}"
            else:
                lbl = f"{qpfx} {btc_final:.2f} \u20bf  {fmt_price(usd_final)}"
            short = _fmt_short(btc_final, usd_final)
            _pending_annots.append(dict(
                x_arr=ts, y_arr=all_y_vals[q],
                label=lbl, short_label=short,
                color=col, y_last=float(all_y_vals[q][-1])))

        # MC median endpoint
        _mc_ann = _mc_median_annot(mc_traces_list, disp_mode, m,
                                   float(ts[-1]), t_start, t_end, syr, eyr)
        if _mc_ann:
            _pending_annots.append(_mc_ann)
    traces.extend(_resolve_edge_annotations(_pending_annots, _is_log))

    return _finalize_chart(traces, layout, p, "ret", mc_result)


# ── HODL Supercharger ─────────────────────────────────────────────────────────

_DELAY_COLORS = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#E07000']
_ANNOT_COLORS = ['#636EFA', '#EF553B', '#1D8348', '#AB63FA', '#E07000']
_DASH_STYLES  = ['solid', 'dash', 'dot', 'dashdot', 'longdash']


def build_supercharge_figure(m: ModelData, p: dict[str, Any]) -> tuple[go.Figure, dict | None]:
    """
    p keys: mode ('a'/'b'), start_stack, start_yr, delays (list), freq, inflation,
            selected_qs, chart_layout (0/1/2), display_q,
            wd_amount (Mode A), end_yr (Mode A), disp_mode (Mode A),
            log_y, annotate, show_legend,
            target_yr (Mode B), lots, use_lots
    """
    model = _app_ctx.DEFAULT_MODEL

    mode         = p.get("mode", "a")
    freq_str, ppy, dt = _build_freq_config(p)
    syr          = int(p.get("start_yr", pd.Timestamp.today().year))
    inflation    = float(p.get("inflation", 4)) / 100.0
    chart_layout = int(p.get("chart_layout", 0))
    display_q    = float(p.get("display_q", 0.5))
    show_legend  = bool(p.get("show_legend", True))

    # Starting stack (lots override)
    start_stack = _get_starting_stack(p, default=1.0)

    # Quantiles
    sel_qs = sorted([float(q) for q in (p.get("selected_qs") or [])
                     if float(q) in model.fits])
    if not sel_qs:
        return go.Figure(layout=dict(
            title="Select at least one quantile",
            paper_bgcolor=m.PLOT_BG_COLOR,
            font=dict(color=m.TEXT_COLOR))), None

    # Delays: filter None/negative, sort, deduplicate
    raw_delays = p.get("delays") or [0, 1, 2, 4, 8]
    delays = sorted(set(float(d) for d in raw_delays if d is not None and float(d) >= 0))
    if not delays:
        delays = [0.0]

    freq_label = {"Daily": "/day", "Weekly": "/wk", "Monthly": "/mo",
                  "Quarterly": "/qtr", "Annually": "/yr"}.get(freq_str, "/mo")

    # ── MODE A: fixed spending \u2192 show how long savings last ───────────────────
    if mode == "a":
        eyr       = int(p.get("end_yr", 2075))
        wd_amount = float(p.get("wd_amount", 5000))
        disp_mode = p.get("disp_mode", "usd")
        t_end     = yr_to_t(eyr, m.genesis)

        # Simulate all (delay, quantile) combos
        results = {}
        for d in delays:
            t_start_d = max(yr_to_t(syr + d, m.genesis), 1.0)
            if t_start_d >= t_end:
                continue
            ts_d = np.arange(t_start_d, t_end + dt * 0.5, dt)
            if len(ts_d) == 0:
                continue
            ts_d_clamped = np.maximum(ts_d, 0.5)
            adj_wd_d = wd_amount * ((1 + inflation) ** (ts_d - t_start_d))
            for q in sel_qs:
                prices = model.price_at(q, ts_d_clamped)
                vals = np.maximum(start_stack - np.cumsum(adj_wd_d / prices), 0.0)
                depl_mask = vals == 0.0
                depl_t = float(ts_d[np.argmax(depl_mask)]) if depl_mask.any() else None
                if disp_mode == "usd":
                    y_vals = vals * prices
                else:
                    y_vals = vals
                results[(d, q)] = (ts_d, y_vals, depl_t, t_start_d, vals, prices)

        traces         = []
        deplete_annots = []

        _AY_LEVELS = _ANNOT_STAGGER_Y

        def _depl_annot(depl_t, t_start_d, d, col, stagger=0):
            depl_yr = int((syr + d) + (depl_t - t_start_d) *
                          (eyr - (syr + d)) / max(t_end - t_start_d, 1e-6))
            return dict(
                x=depl_t - dt, xref="x",   # last nonzero step, aligns with band end
                y=0, yref="paper",
                ax=28, ay=_AY_LEVELS[stagger % 3],
                text=f"\u2248{depl_yr}",
                showarrow=True, arrowhead=2, arrowsize=1,
                arrowcolor=col, font=dict(size=_FONT_ANNOT, color=col),
            )

        if chart_layout == 0:
            # Color = delay, show quantile closest to display_q
            q_show = min(sel_qs, key=lambda q: abs(q - display_q))
            for di, d in enumerate(delays):
                key = (d, q_show)
                if key not in results:
                    continue
                ts_d, y_vals, depl_t, t_start_d, *_ = results[key]
                col   = _DELAY_COLORS[di % len(_DELAY_COLORS)]
                d_lbl = f"+{int(d)}yr" if d == int(d) else f"+{d:.1f}yr"
                if disp_mode == "usd":
                    final = fmt_price(float(y_vals[-1]))
                else:
                    _vals, _prices = results[key][4], results[key][5]
                    final_usd = fmt_price(float(_vals[-1]) * float(_prices[-1]))
                    final = f"{float(y_vals[-1]):.4f} BTC  ({final_usd})"
                traces.append(go.Scatter(
                    x=list(ts_d), y=list(y_vals), mode="lines",
                    name=f"Delay {d_lbl}  \u2192  {final}",
                    line=dict(color=col, width=2),
                ))
                if depl_t is not None:
                    deplete_annots.append(_depl_annot(depl_t, t_start_d, d,
                                                      _ANNOT_COLORS[di % len(_ANNOT_COLORS)],
                                                      len(deplete_annots)))

        elif chart_layout == 1:
            # Color = quantile, line style = delay
            for q in sel_qs:
                col   = model.colors.get(q, "#888888")
                q_lbl = _fmt_q_label(q)
                for di, d in enumerate(delays):
                    key = (d, q)
                    if key not in results:
                        continue
                    ts_d, y_vals, depl_t, t_start_d, *_ = results[key]
                    d_lbl = f"+{int(d)}yr" if d == int(d) else f"+{d:.1f}yr"
                    traces.append(go.Scatter(
                        x=list(ts_d), y=list(y_vals), mode="lines",
                        name=f"{q_lbl} delay={d_lbl}",
                        line=dict(color=col, width=_QR_LINE_WIDTH,
                                  dash=_DASH_STYLES[di % len(_DASH_STYLES)]),
                    ))
                    if depl_t is not None:
                        deplete_annots.append(_depl_annot(depl_t, t_start_d, d, col,
                                                          len(deplete_annots)))

        else:
            # Layout 2: shaded band per delay (min/max across quantiles)
            for di, d in enumerate(delays):
                t_start_d = max(yr_to_t(syr + d, m.genesis), 1.0)
                ts_d = np.arange(t_start_d, t_end + dt * 0.5, dt)
                if len(ts_d) == 0:
                    continue
                all_y = [results[(d, q)][1] for q in sel_qs if (d, q) in results]
                if not all_y:
                    continue
                all_y  = np.array(all_y)
                y_min  = all_y.min(axis=0)
                y_max  = all_y.max(axis=0)
                col    = _DELAY_COLORS[di % len(_DELAY_COLORS)]
                d_lbl  = f"+{int(d)}yr" if d == int(d) else f"+{d:.1f}yr"
                traces.append(go.Scatter(
                    x=list(ts_d), y=list(y_max), mode="lines",
                    line=dict(color=col, width=0), showlegend=False, hoverinfo="skip",
                ))
                max_final = (fmt_price(float(y_max[-1])) if disp_mode == "usd"
                             else f"{float(y_max[-1]):.4f} BTC")
                traces.append(go.Scatter(
                    x=list(ts_d), y=list(y_min), mode="lines",
                    fill="tonexty", fillcolor=_hex_alpha(col, 0.2),
                    line=dict(color=col, width=0),
                    name=f"Delay {d_lbl}  \u2192  {max_final}",
                    hoverinfo="skip",
                ))
                for q in sel_qs:
                    key = (d, q)
                    if key not in results:
                        continue
                    _, _, depl_t, t_start_d, *_ = results[key]
                    if depl_t is not None:
                        deplete_annots.append(_depl_annot(depl_t, t_start_d, d,
                                                          _ANNOT_COLORS[di % len(_ANNOT_COLORS)],
                                                          len(deplete_annots)))

        # ── alternative model overlays ────────────────────────────────────────
        for model_key in p.get("active_models", []):
            mdl = _app_ctx.PRICE_MODELS.get(model_key)
            if not mdl:
                continue
            _sc_overlay_qs = sel_qs if mdl.quantized else [0.5]
            for q in _sc_overlay_qs:
                if mdl.quantized and q not in mdl.fits:
                    continue
                for di, d in enumerate(delays):
                    t_start_d = max(yr_to_t(syr + d, m.genesis), 1.0)
                    if t_start_d >= t_end:
                        continue
                    ts_d = np.arange(t_start_d, t_end + dt * 0.5, dt)
                    if len(ts_d) == 0:
                        continue
                    ts_d_clamped = np.maximum(ts_d, 0.5)
                    adj_wd_d = wd_amount * ((1 + inflation) ** (ts_d - t_start_d))
                    prices = mdl.price_at(q, ts_d_clamped)
                    vals = np.maximum(start_stack - np.cumsum(adj_wd_d / prices), 0.0)
                    y_vals = vals * prices if disp_mode == "usd" else vals
                    col = mdl.colors.get(q, "#888888") if mdl.quantized else "#8B4513"
                    d_lbl = f"+{int(d)}yr" if d == int(d) else f"+{d:.1f}yr"
                    q_lbl = f" {_fmt_q_label(q, '')}" if mdl.quantized else ""
                    if disp_mode == "usd":
                        final = fmt_price(float(y_vals[-1]))
                    else:
                        final_usd = fmt_price(float(vals[-1]) * float(prices[-1]))
                        final = f"{float(vals[-1]):.4f} BTC  ({final_usd})"
                    traces.append(go.Scatter(
                        x=list(ts_d), y=list(y_vals), mode="lines",
                        name=f"{mdl.name}{q_lbl} {d_lbl}  \u2192  {final}",
                        line=dict(color=col, width=1.2, dash=mdl.dash_style),
                        legendgroup=mdl.short_name,
                        legendgrouptitle_text=mdl.name,
                        showlegend=(di == 0),  # show legend only for first delay
                    ))

        t_start_base = max(yr_to_t(syr, m.genesis), 1.0)
        ylabel = "USD Value" if disp_mode == "usd" else "BTC Remaining"
        sc_title = (f"HODL Supercharger \u2014 {fmt_price(wd_amount)}{freq_label} \u00b7 "
                    f"Retire {syr}+ \u00b7 to {eyr}")
        layout, _ = _sim_layout(m, p, sc_title, ylabel, np.array([t_end]),
                                t_start_base, t_end, dt, syr, eyr)
        layout["annotations"] = deplete_annots
        # ── Monte Carlo fan overlay ───────────────────────────────────────────
        mc_traces_list = []
        mc_result = None
        if _HAS_MARKOV and p.get("mc_enabled"):
            t_start_base = max(yr_to_t(syr, m.genesis), 1.0)
            _sc_x_end = layout["xaxis"]["range"][1]
            mc_traces_list, mc_result = _apply_mc_overlay(
                m, p, _mc_supercharge_overlay,
                (m, p, np.arange(t_start_base, t_end + dt * 0.5, dt),
                 t_start_base, t_end, dt, start_stack, disp_mode, len(deplete_annots)),
                traces, deplete_annots, layout, _sc_x_end, disp_mode)

        _stagger_depletion_annots(deplete_annots, layout)

        # ── Right-edge / endpoint value labels ─────────────────────────────
        # Use text traces (go.Scatter mode="markers+text") instead of
        # annotations — consistent with Retire tab; avoids paper-x arrowhead
        # misalignment on declining traces.
        _sc_log = bool(p.get("log_y"))
        _pending_annots = []
        if p.get("annotate"):
            if chart_layout == 2:
                # Band endpoint labels: one per delay, upper-bound value
                for di, d in enumerate(delays):
                    band = [(q, results[(d, q)]) for q in sel_qs
                            if (d, q) in results]
                    surviving = [(q, r) for q, r in band
                                 if r[2] is None and float(r[1][-1]) > 0]
                    if not surviving:
                        continue
                    col = _DELAY_COLORS[di % len(_DELAY_COLORS)]
                    y_max = max(float(r[1][-1]) for _, r in surviving)
                    lbl = (fmt_price(y_max) if disp_mode == "usd"
                           else f"{y_max:.4f} \u20bf")
                    y_arr = np.maximum.reduce(
                        [r[1] for _, r in surviving])
                    ts_d_r = surviving[0][1][0]
                    # Get BTC/USD for short label from best surviving entry
                    _best_q, _best_r = max(surviving, key=lambda x: float(x[1][1][-1]))
                    _sc_btc = float(_best_r[4][-1])  # raw BTC vals
                    _sc_usd = float(_best_r[4][-1] * _best_r[5][-1])  # BTC * price
                    _pending_annots.append(dict(
                        x_arr=ts_d_r, y_arr=y_arr,
                        label=lbl, short_label=_fmt_short(_sc_btc, _sc_usd),
                        color=col, y_last=float(y_arr[-1])))
            else:
                for (d, q), res in results.items():
                    ts_d_r, y_vals_r, depl_t_r, _, btc_vals_r, prices_r = res
                    if depl_t_r is not None:
                        continue  # depleted — already has year annotation
                    y_final = float(y_vals_r[-1])
                    if y_final <= 0:
                        continue
                    if chart_layout == 0:
                        di = delays.index(d) if d in delays else 0
                        col = _DELAY_COLORS[di % len(_DELAY_COLORS)]
                    else:
                        col = model.colors.get(q, "#888888")
                    lbl = (fmt_price(y_final) if disp_mode == "usd"
                           else f"{y_final:.4f} \u20bf")
                    _sc_btc = float(btc_vals_r[-1])
                    _sc_usd = float(btc_vals_r[-1] * prices_r[-1])
                    _pending_annots.append(dict(
                        x_arr=ts_d_r, y_arr=y_vals_r,
                        label=lbl, short_label=_fmt_short(_sc_btc, _sc_usd),
                        color=col, y_last=y_final))
            # MC median endpoint
            _mc_ann = _mc_median_annot(
                mc_traces_list, disp_mode, m, t_end, t_start_base, t_end,
                syr, eyr, btc_fmt=".4f", estimate_usd=False)
            if _mc_ann:
                _pending_annots.append(_mc_ann)
        traces.extend(_resolve_edge_annotations(_pending_annots, _sc_log))

        return _finalize_chart(traces, layout, p, "sc", mc_result, mc_premium=False)

    # ── MODE B: fixed depletion date → max withdrawal per period ──────────────
    else:
        return _sc_mode_b(m, p, syr, delays, sel_qs, start_stack, ppy, dt,
                          inflation, chart_layout, display_q, show_legend, freq_label)


def _sc_mode_b(m, p, syr, delays, sel_qs, start_stack, ppy, dt,
               inflation, chart_layout, display_q, show_legend, freq_label):
    """HODL Supercharger Mode B: binary-search max withdrawal per period."""
    model = _app_ctx.DEFAULT_MODEL
    target_yr = int(p.get("target_yr", 2060))

    def _max_wd_for(d, q):
        t_start_d = max(yr_to_t(syr + d, m.genesis), 1.0)
        t_end_b   = yr_to_t(target_yr, m.genesis)
        if t_end_b <= t_start_d:
            return 0.0
        first_price = float(model.price_at(q, max(t_start_d, 0.5)))
        # Binary search: find max withdrawal where stack survives to target_yr.
        # Upper bound = 4x annual stack value (generous overestimate).
        # 60 iterations gives precision to ~1e-18 of the range (more than enough).
        lo, hi = 0.0, start_stack * first_price * ppy * 4
        for _ in range(_BISECT_ITERS):
            mid = (lo + hi) / 2.0
            s   = start_stack
            survived = True
            for t in np.arange(t_start_d, t_end_b + dt * 0.5, dt):
                adj = mid * ((1 + inflation) ** (t - t_start_d))
                s  -= adj / float(model.price_at(q, max(t, 0.5)))
                if s <= 0:
                    survived = False
                    break
            if survived:
                lo = mid
            else:
                hi = mid
        return lo

    max_wd = {(d, q): _max_wd_for(d, q) for d in delays for q in sel_qs}
    traces = []

    if chart_layout == 0:
        q_show = min(sel_qs, key=lambda q: abs(q - display_q))
        y_line = [max_wd.get((d, q_show), 0) for d in delays]
        traces.append(go.Scatter(
            x=delays, y=y_line, mode="lines",
            line=dict(color="#888888", width=1, dash="dot"),
            showlegend=False, hoverinfo="skip",
        ))
        for di, d in enumerate(delays):
            col   = _DELAY_COLORS[di % len(_DELAY_COLORS)]
            val   = max_wd.get((d, q_show), 0)
            d_lbl = f"+{int(d)}yr" if d == int(d) else f"+{d:.1f}yr"
            traces.append(go.Scatter(
                x=[d], y=[val], mode="markers+text",
                marker=dict(color=col, size=12),
                text=[fmt_price(val) + freq_label],
                textposition="top center",
                name=f"Delay {d_lbl}",
                hovertemplate=f"Delay {d_lbl}<br>{fmt_price(val)}{freq_label}<extra></extra>",
            ))

    elif chart_layout == 1:
        for q in sel_qs:
            col   = model.colors.get(q, "#888888")
            q_lbl = _fmt_q_label(q)
            y_q   = [max_wd.get((d, q), 0) for d in delays]
            traces.append(go.Scatter(
                x=delays, y=y_q, mode="lines+markers",
                name=q_lbl,
                line=dict(color=col, width=2),
                marker=dict(color=col, size=7),
            ))

    else:
        for di, d in enumerate(delays):
            col   = _DELAY_COLORS[di % len(_DELAY_COLORS)]
            d_lbl = f"+{int(d)}yr" if d == int(d) else f"+{d:.1f}yr"
            y_d   = [max_wd.get((d, q), 0) for q in sel_qs]
            qlbls = [_fmt_q_label(q) for q in sel_qs]
            med_val = y_d[len(y_d) // 2] if y_d else 0
            traces.append(go.Scatter(
                x=list(sel_qs), y=y_d, mode="lines+markers",
                name=f"Delay {d_lbl}  \u2192  {fmt_price(med_val)}{freq_label} (med)",
                line=dict(color=col, width=2),
                marker=dict(color=col, size=6),
                customdata=qlbls,
                hovertemplate="%{customdata}: %{y:,.0f}<extra></extra>",
            ))

    xlabel = "Delay (years)" if chart_layout in (0, 1) else "Quantile"
    layout = _dark_layout(
        m,
        title=f"HODL Supercharger \u2014 Max spend{freq_label} to deplete by {target_yr}",
        xlabel=xlabel,
        ylabel=f"Max withdrawal{freq_label}",
    )
    return _finalize_chart(traces, layout, p, "sc", mc_premium=False)
