"""Shared constants, helpers, and theme functions for all chart builders."""

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
from btc_core import yr_to_t, fmt_price, leo_weighted_entry
try:
    from markov import build_transition_matrix  # noqa: F401 — presence check
    _HAS_MARKOV = True
except ImportError:
    _HAS_MARKOV = False

# MC overlay logic lives in mc_overlay.py
from mc_overlay import (
    _mc_build_traces, _mc_depletion_annots,
    _mc_dca_overlay,
    _mc_retire_overlay, _mc_supercharge_overlay,
    _mc_heatmap_overlay,
    ghost_traces_from_params,
)


# ── shared constants ─────────────────────────────────────────────────────────

_ANNOT_STAGGER_Y = _app_ctx.ANNOT_STAGGER_Y

_BTC_ORANGE       = _app_ctx.BTC_ORANGE
_TODAY_LINE_COLOR  = "#FF6600"
_TODAY_LINE_WIDTH  = 2.0
_TODAY_LINE_OPACITY = 0.85
_TODAY_GLOW_WIDTH  = 6
_TODAY_GLOW_OPACITY = 0.12
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
_NON_QUANTIZED_MODEL_COLOR = "#8B4513"  # saddlebrown — single-trajectory models
_OVERLAY_LINE_WIDTH = _QR_LINE_WIDTH * 0.8  # alt-model overlay lines
_GLOW_WIDTH       = 6        # neon wire glow shadow width
_GLOW_ALPHA       = 0.15     # neon wire glow opacity
_WM_OPACITY       = 0.55     # watermark logo opacity
_WM_SIZE_X        = 0.09     # watermark logo width (fraction of paper)
_WM_SIZE_Y        = 0.12     # watermark logo height (fraction of paper)
_COLORSCALE_STEPS = 256      # dense colorscale points (avoids browser interpolation bugs)
_BISECT_ITERS     = 60       # binary search iterations for Mode B max-withdrawal
_HM_TEXT_THRESHOLD = 0.55    # cell brightness threshold: white text below, dark above

# ── Enhanced font stack (sans-serif base, serif for premium/MC) ──────────
_SANS_FONT = "Avenir Next, Avenir, Segoe UI, system-ui, -apple-system, sans-serif"
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

# ── Bitcoin Thermal palette — quantile → temperature color ────────────────────
# Low percentiles (value zone) = cool blue, median = silver, high = hot orange/red
_THERMAL_STOPS = [
    (0.001, "#0d47a1"),   # deep sapphire
    (0.01,  "#1565c0"),   # royal blue
    (0.015, "#1976d2"),   # blue
    (0.05,  "#42a5f5"),   # sky blue
    (0.10,  "#80deea"),   # light cyan
    (0.25,  "#b2dfdb"),   # pale teal
    (0.50,  "#bdbdbd"),   # silver — the pivot
    (0.75,  "#ffcc80"),   # light amber
    (0.90,  "#f7931a"),   # Bitcoin orange
    (0.95,  "#e65100"),   # deep orange
    (0.99,  "#c62828"),   # crimson
    (0.999, "#7f0000"),   # deep blood red
]


def _thermal_color(q: float) -> str:
    """Map a quantile (0–1) to a temperature color via the thermal palette."""
    if q <= _THERMAL_STOPS[0][0]:
        return _THERMAL_STOPS[0][1]
    if q >= _THERMAL_STOPS[-1][0]:
        return _THERMAL_STOPS[-1][1]
    for i in range(len(_THERMAL_STOPS) - 1):
        q0, c0 = _THERMAL_STOPS[i]
        q1, c1 = _THERMAL_STOPS[i + 1]
        if q0 <= q <= q1:
            f = (q - q0) / (q1 - q0) if q1 > q0 else 0
            return _lerp_hex(c0, c1, f)
    return "#bdbdbd"


def _build_thermal_colors(quantiles: list) -> dict:
    """Build a {quantile: hex_color} dict using the thermal palette."""
    return {q: _thermal_color(q) for q in quantiles}


def _add_glow_trace(traces: list, x, y, color: str, width: float = _GLOW_WIDTH,
                    opacity: float = _GLOW_ALPHA) -> None:
    """Add a wider semi-transparent 'neon wire' shadow trace behind a line."""
    traces.append(go.Scatter(
        x=list(x) if not isinstance(x, list) else x,
        y=list(y) if not isinstance(y, list) else y,
        mode="lines",
        line=dict(color=color, width=width),
        opacity=opacity,
        showlegend=False, hoverinfo="skip",
    ))


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
    """Base dark-theme layout dict — shared Quantoshi chart template.

    Includes consistent font family, sizes, colors, and grid styling
    so all charts have a cohesive look.
    """
    return dict(
        title=dict(text=title, font=dict(family=_SANS_FONT, color=m.TITLE_COLOR, size=_FONT_TITLE)),
        paper_bgcolor=m.PLOT_BG_COLOR,
        plot_bgcolor=m.PLOT_BG_COLOR,
        font=dict(family=_SANS_FONT, color=m.TEXT_COLOR, size=_FONT_BODY),
        xaxis=dict(
            title=dict(text=xlabel, font=dict(family=_SANS_FONT, color=m.TEXT_COLOR)),
            gridcolor=m.GRID_MAJOR_COLOR, gridwidth=0.6,
            linecolor=m.SPINE_COLOR, tickcolor=m.TEXT_COLOR,
            zerolinecolor=m.GRID_MAJOR_COLOR,
        ),
        yaxis=dict(
            title=dict(text=ylabel, font=dict(family=_SANS_FONT, color=m.TEXT_COLOR)),
            gridcolor=m.GRID_MAJOR_COLOR, gridwidth=0.6,
            linecolor=m.SPINE_COLOR, tickcolor=m.TEXT_COLOR,
            zerolinecolor=m.GRID_MAJOR_COLOR,
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.85)", bordercolor=m.GRID_MAJOR_COLOR,
            borderwidth=1, font=dict(family=_SANS_FONT, size=_FONT_LEGEND),
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
_ASSETS = Path(__file__).resolve().parent.parent / "assets"
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


def _hex_alpha(hex_color, alpha):
    """Convert hex color + alpha float to an rgba() CSS string."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _clip_mc_traces(mc_traces, x_max):
    """Clip MC traces to x <= x_max so they don't extend beyond the visible chart.

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
      ascending  -> "bottom left"  (text below the rising line)
      descending -> "top left"     (text above the falling line)
      flat       -> "middle left"

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
    4. Clusters of 2-3: alternate textposition top/bottom to spread apart.
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
            # Position based on rank: bottom half -> "bottom left",
            # top half -> "top left", middle -> slope-based default
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
            merged_label = " \u00b7 ".join(parts)
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


# Re-export from _app_ctx for backward compat (used by chart builders and callbacks)
FREQ_PPY = _app_ctx.FREQ_PPY
_FREQ_STEP_DAYS = _app_ctx.FREQ_STEP_DAYS

