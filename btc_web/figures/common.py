"""Shared constants, helpers, and theme functions for all chart builders."""

from __future__ import annotations

import base64
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# btc_core/ package lives at project root, added to sys.path by app.py
from typing import Any
import _app_ctx
import theme
from btc_core import yr_to_t, fmt_price, leo_weighted_entry
from colors import (
    FALLBACK_MODEL_GRAY, BTC_ORANGE as _COLORS_BTC_ORANGE,
    TODAY_LINE_COLOR as _COLORS_TODAY_LINE,
    NON_QUANTIZED_MODEL_COLOR as _COLORS_NON_Q,
    MC_TITLE_COLOR as _COLORS_MC_TITLE,
    MC_LEGEND_BORDER as _COLORS_MC_BORDER,
    SPINE_COLOR_FALLBACK,
    CLUSTER_MERGE_GRAY,
    BLACK, BLACK_A0,
    THERMAL_NEUTRAL,
    PLOT_BG_COLOR as _COLORS_PLOT_BG,
    LOG_MINOR_GRID_GRAY,
    WATERMARK_TEXT_COLOR,
    _hex_alpha,
    LOG_MINOR_GRID_ALPHA, ANNOT_TEXT_ALPHA,
    WATERMARK_TEXT_ALPHA, MC_LEGEND_BG_ALPHA,
    # ── Section 5: appearance constants (Phase 3 consolidation) ──
    FONT_SANS, FONT_BRAND,
    CHART_FONT_TITLE, CHART_FONT_SUBTITLE, CHART_FONT_BODY,
    CHART_FONT_LEGEND, CHART_FONT_WATERMARK, CHART_FONT_ANNOT,
    CHART_FONT_TITLE_LG, CHART_FONT_BODY_LG, CHART_FONT_TICK_LG,
    CHART_FONT_LEGEND_LG, CHART_FONT_ANNOT_LG, CHART_FONT_WATERMARK_LG,
    TRACE_WIDTH, TRACE_WIDTH_OVERLAY, TRACE_WIDTH_TODAY,
    GRID_MAJOR_WIDTH, GRID_MINOR_WIDTH, LEGEND_BG_OPACITY,
    TODAY_GLOW_WIDTH, TODAY_LINE_OPACITY, TODAY_GLOW_OPACITY,
    SHADE_ALPHA, WM_OPACITY, WM_SIZE_X, WM_SIZE_Y,
    CHART_MARGIN,
    Q_OPACITY_FLOOR, Q_OPACITY_RANGE, Q_OPACITY_DECAY,
    quantile_shade, BAND_FILL_MODE, BAND_PASTEL_ALPHA,
)
_HAS_MARKOV = _app_ctx._HAS_MARKOV

# MC overlay logic lives in mc_overlay.py
from mc_overlay import (
    _mc_build_traces, _mc_depletion_annots,
    _mc_dca_overlay,
    _mc_retire_overlay, _mc_supercharge_overlay,
    _mc_heatmap_overlay,
    ghost_traces_from_params,
)


_q3_trace = _app_ctx._q3

def _round_trace_data(arr):
    """Round array to 3 sig figs for bandwidth savings. Passes through 0/None/NaN.

    Vectorised when ``arr`` is a numpy array or anything convertible to one
    without losing a mixed-dtype sentinel; falls back to an element-wise
    Python loop for lists containing ``None`` values (which don't round-trip
    through numpy cleanly).
    """
    if isinstance(arr, np.ndarray):
        a = arr.astype(np.float64, copy=False)
        sign = np.sign(a)
        absv = np.abs(a)
        # mask of values safe to scale (non-zero, finite)
        safe = np.isfinite(absv) & (absv > 0)
        out = a.astype(np.float64, copy=True)
        if safe.any():
            exp = np.floor(np.log10(absv, where=safe, out=np.zeros_like(absv)))
            factor = 10.0 ** (exp - 2)
            rounded = np.where(safe, np.round(a / factor) * factor, a)
            out = rounded
        # Preserve NaN / +-inf untouched (np.round already does).
        return out.tolist()
    # Slow path: list may contain None sentinels.
    return [_q3_trace(v) if v and v == v else v for v in arr]


def quantile_opacity(q: float) -> float:
    """Quantile-dependent line opacity: 1.0 at Q50, fading toward floor at extremes."""
    return max(Q_OPACITY_FLOOR, 1.0 - abs(q - 0.5) / Q_OPACITY_RANGE * Q_OPACITY_DECAY)


def _parse_quantiles(p: dict, key: str = "selected_qs") -> list[float]:
    """Parse and sort quantile list from params dict.

    Basic form only -- callers needing model.fits filtering, reverse sort,
    or custom defaults apply post-processing themselves.
    """
    return sorted(float(q) for q in (p.get(key) or []))


def _format_final_value(vals, prices, disp_mode: str, show_usd_parens: bool = True):
    """Format final simulation value for legend label.

    Returns (y_vals, final_label). show_usd_parens=False used by overlay
    traces where USD would duplicate.
    """
    if disp_mode == "usd":
        y_vals = vals * prices
        return y_vals, fmt_price(float(y_vals[-1]))
    y_vals = vals
    btc_final = float(vals[-1])
    if show_usd_parens:
        usd_final = fmt_price(float(btc_final * float(prices[-1])))
        return y_vals, f"{btc_final:.4f} BTC  ({usd_final})"
    return y_vals, f"{btc_final:.4f} BTC"


def _quantile_trace(ts, y_vals, q: float, color: str, label: str,
                    width: float | None = None, shape: str = "linear",
                    **kw) -> go.Scatter:
    """Build a quantile-colored Scatter trace with standard shade + opacity."""
    _shade = quantile_shade(color, q)
    return go.Scatter(
        x=ts, y=y_vals, mode="lines", name=label,
        line=dict(color=_shade, width=width or TRACE_WIDTH, shape=shape),
        opacity=quantile_opacity(q),
        **kw,
    )


def _empty_state_annotation(layout: dict) -> None:
    """Set the 'No models selected' fallback annotation on layout."""
    layout["annotations"] = [dict(
        text="No models selected \u2014 check Display Models",
        xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16, color=FALLBACK_MODEL_GRAY),
    )]


def _today_line_shapes(t_today: float, y_lo, y_hi, color: str,
                       glow: bool = True, yref: str = "y") -> list[dict]:
    """Build today-line shape(s). bubble uses glow+yref='y'; residuals uses no glow+yref='paper'."""
    shapes = []
    if glow:
        shapes.append(dict(
            type="line", x0=t_today, x1=t_today, y0=y_lo, y1=y_hi,
            line=dict(color=color, width=TODAY_GLOW_WIDTH),
            opacity=TODAY_GLOW_OPACITY, yref=yref,
        ))
    shapes.append(dict(
        type="line", x0=t_today, x1=t_today, y0=y_lo, y1=y_hi,
        line=dict(color=color, dash="dash", width=TRACE_WIDTH_TODAY),
        opacity=TODAY_LINE_OPACITY, yref=yref,
    ))
    return shapes


# ── shared constants ─────────────────────────────────────────────────────────

_ANNOT_STAGGER_Y = _app_ctx.ANNOT_STAGGER_Y

_BTC_ORANGE       = _app_ctx.BTC_ORANGE
_TODAY_LINE_COLOR  = _COLORS_TODAY_LINE

# Appearance constants (FONT_*, CHART_FONT_*, TRACE_WIDTH*, TODAY_*,
# SHADE_ALPHA, WM_*) are imported directly from colors.py at the top of
# this module and re-exposed to figure modules by their canonical names.
# The underscore-prefix aliases that formerly lived here were removed
# 2026-04-16 — all figure modules now import the canonical names.

_NON_QUANTIZED_MODEL_COLOR = _COLORS_NON_Q  # saddlebrown — single-trajectory models

# ── Algorithmic constants (not appearance — stay local) ──────────────────
_INTERP_POINTS    = 400      # sample points for QR interpolation curves (400 > max screen px)
_MAX_SCATTER_PTS  = 800      # max data points before downsampling (plenty for screen)
_COLORSCALE_STEPS = 256      # dense colorscale points (avoids browser interpolation bugs)
_BISECT_ITERS     = 60       # binary search iterations for Mode B max-withdrawal
_HM_TEXT_THRESHOLD = 0.55    # cell brightness threshold: white text below, dark above

# ── Hover format templates ──────────────────────────────────────────────────
# All chart traces (data points + model lines) use these via _add_date_hover.
# customdata[0] = calendar date string.  %{fullData.name} = trace name.
# Depletion/terminal annotations and heatmap hovers are separate.
_HOVER_FMT_USD = "<b>%{fullData.name}</b><br>%{customdata[0]}<br>$%{y:,.0f}<extra></extra>"
_HOVER_FMT_BTC = "<b>%{fullData.name}</b><br>%{customdata[0]}<br>%{y:,.4f} BTC<extra></extra>"


def _apply_sans_typography(layout: dict) -> None:
    """Upgrade layout fonts to enhanced sans-serif stack with larger sizes."""
    layout["title"]["font"].update(family=FONT_SANS, size=CHART_FONT_TITLE_LG)
    layout["font"].update(family=FONT_SANS, size=CHART_FONT_TICK_LG)
    layout["xaxis"]["title"]["font"].update(family=FONT_SANS, size=CHART_FONT_BODY_LG)
    layout["yaxis"]["title"]["font"].update(family=FONT_SANS, size=CHART_FONT_BODY_LG)
    layout["legend"]["font"] = dict(family=FONT_SANS, size=CHART_FONT_LEGEND_LG)
    for ann in layout.get("annotations", []):
        ann.setdefault("font", {}).update(family=FONT_SANS, size=CHART_FONT_ANNOT_LG)

# ── Bitcoin Thermal palette — quantile → temperature color ────────────────────
# Low percentiles (value zone) = cool blue, median = silver, high = hot orange/red
_THERMAL_STOPS = _app_ctx.PALETTES["default"]["thermal_stops"]


def _get_palette(p):
    """Return active palette dict from params, defaulting to 'default'."""
    key = p.get("palette", "default")
    return _app_ctx.PALETTES.get(key, _app_ctx.PALETTES["default"])


def _get_model_color(model_key, p=None):
    """Return the palette-aware color for a model key.

    Family fallbacks: the ``lppl`` / ``hybppl`` / ``eppl`` checkboxes in
    the UI are master gates that get resolved to concrete variant keys
    (e.g. ``lp3_w``, ``cfg_1d_1u``, ``ecfg_1d_1u``, ``hybppl_cfg_a``)
    before they reach the figure builder. Many of those variant keys
    are NOT individually registered in the palette's ``model_colors``
    dict, so a direct lookup falls back to ``#888888`` gray — which is
    why Entropy PPL / Hybrid PPL / weighted-LPPL traces were all
    rendering gray instead of their family colors. Resolve the family
    prefix and inherit the master's color.

    Directly-registered variant keys (``lp2``, ``lp3``, ``lp4``,
    ``hybppl_dd``, ``hyb2l``, ``hyb2c``, ``hyb2b``, ``hyb4d``) keep
    their own distinct colors — the ``model_key not in mc`` guard
    prevents the fallback from overriding them.
    """
    palette = _get_palette(p) if p else _app_ctx.PALETTES["default"]
    mc = palette.get("model_colors", _app_ctx.MODEL_TRACE_COLORS)
    if model_key and model_key not in mc:
        if model_key.startswith("ecfg_"):
            return mc.get("eppl", FALLBACK_MODEL_GRAY)
        if model_key.startswith("cfg_") or model_key.startswith("hyb"):
            return mc.get("hybppl", FALLBACK_MODEL_GRAY)
        if model_key.startswith("lp"):
            return mc.get("lppl", FALLBACK_MODEL_GRAY)
    return mc.get(model_key, FALLBACK_MODEL_GRAY)


def _thermal_color(q: float, palette=None) -> str:
    """Map a quantile (0–1) to a temperature color via the thermal palette."""
    stops = palette["thermal_stops"] if palette is not None else _THERMAL_STOPS
    if q <= stops[0][0]:
        return stops[0][1]
    if q >= stops[-1][0]:
        return stops[-1][1]
    for i in range(len(stops) - 1):
        q0, c0 = stops[i]
        q1, c1 = stops[i + 1]
        if q0 <= q <= q1:
            f = (q - q0) / (q1 - q0) if q1 > q0 else 0
            return _lerp_hex(c0, c1, f)
    return THERMAL_NEUTRAL


def _build_thermal_colors(quantiles: list, palette=None) -> dict:
    """Build a {quantile: hex_color} dict using the thermal palette."""
    return {q: _thermal_color(q, palette) for q in quantiles}


def _symmetric_thermal_color(q: float, palette=None) -> str:
    """Map quantile to symmetric thermal color (mirrors about 0.5)."""
    mirror = min(q, 1.0 - q)
    return _thermal_color(mirror, palette)


def _build_symmetric_thermal_colors(quantiles: list, palette=None) -> dict:
    """Build {quantile: hex_color} with symmetric colors about Q50%."""
    return {q: _symmetric_thermal_color(q, palette) for q in quantiles}


# ── shared small helpers ──────────────────────────────────────────────────────


def _fmt_q_label(q: float, prefix: str = "BM") -> str:
    """Format quantile as '{prefix} Q{pct}%' with appropriate precision."""
    pct = q * 100
    ql = f"Q{pct:.4g}%" if pct >= 1 else f"Q{pct:.3g}%"
    return f"{prefix} {ql}" if prefix else ql


def _fmt_q_range(qs: list) -> str:
    """Format a list of quantiles as a compact range, e.g. 'Q1\u201310%'."""
    if not qs:
        return ""
    lo, hi = min(qs), max(qs)
    def _pct(q):
        p = q * 100
        return f"{p:.4g}%" if p >= 1 else f"{p:.3g}%"
    if lo == hi:
        return f"Q{_pct(lo)}"
    return f"Q{_pct(lo)}\u2013{_pct(hi)}"


def _error_figure(title):
    """Return a blank figure with a message title, styled for dark theme."""
    fig = go.Figure()
    fig.update_layout(
        title=title,
        paper_bgcolor=theme.PLOT_BG_COLOR,
        plot_bgcolor=theme.PLOT_BG_COLOR,
        font=dict(color=theme.TEXT_COLOR),
    )
    return fig


# ── shared theme helpers ──────────────────────────────────────────────────────

_LOG_MINOR = dict(showgrid=True, gridcolor=_hex_alpha(LOG_MINOR_GRID_GRAY, LOG_MINOR_GRID_ALPHA),
                  griddash="dot", gridwidth=GRID_MINOR_WIDTH, dtick="D1")


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
            a["ay"] = _ANNOT_STAGGER_Y[i % len(_ANNOT_STAGGER_Y)]
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
        return _error_figure("Set end year > start year"), None
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
    layout = _base_layout(title=title, xlabel="Year", ylabel=ylabel)
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


def _uirevision_key(p: dict, tab: str) -> str:
    """Stable uirevision key that preserves Plotly pan/zoom/hover state
    across redundant figure rebuilds (e.g. late hydration fires during
    snapshot restore, background prefetch side-effects).

    Includes palette — a palette change is a legitimate visual reset.
    Includes tab — each tab has its own revision so cross-tab state
    doesn't bleed.
    See spec 2026-04-24-single-redraw-per-snapshot-design.md."""
    palette = (p.get("palette") or "default") if p else "default"
    return f"{tab}:{palette}"


def apply_zoom_lock(fig, zoom_on: bool, axes: bool = True):
    """Sync a figure's drag interactivity with the "Enable chart zoom" toggle.

    Writes BOTH states explicitly. `chart_zoom` is not part of the figure
    cache key for most tabs, so every zoom state shares one cached
    go.Figure object; a one-way `dragmode=False` write sticks to that shared
    object and re-checking the box could never restore zoom.

    axes=False for the heatmap, whose builder pins `fixedrange=True` on both
    axes by design — there only `dragmode` follows the toggle.
    """
    fig.update_layout(dragmode="zoom" if zoom_on else False)
    if axes:
        fig.update_xaxes(fixedrange=not zoom_on)
        fig.update_yaxes(fixedrange=not zoom_on)
    return fig


def _base_layout(title, xlabel, ylabel, **kwargs):
    """Base layout dict — shared Quantoshi chart template.

    Includes consistent font family, sizes, colors, and grid styling
    so all charts have a cohesive look.
    """
    return dict(
        title=dict(
            text=title,
            font=dict(family=FONT_BRAND, color=theme.TITLE_COLOR, size=CHART_FONT_TITLE),
            x=0.02, xanchor="left",
        ),
        paper_bgcolor=theme.PLOT_BG_COLOR,
        plot_bgcolor=theme.PLOT_BG_COLOR,
        font=dict(family=FONT_SANS, color=theme.TEXT_COLOR, size=CHART_FONT_BODY),
        xaxis=dict(
            title=dict(text=xlabel, font=dict(family=FONT_SANS, color=theme.TEXT_COLOR)),
            gridcolor=theme.GRID_MAJOR_COLOR, gridwidth=GRID_MAJOR_WIDTH,
            linecolor=theme.SPINE_COLOR, tickcolor=theme.TEXT_COLOR,
            zerolinecolor=theme.GRID_MAJOR_COLOR,
            ticks="",
            ticklabelstandoff=-8,
            ticklabelposition="outside top",
            ticklabelshift=0,
        ),
        yaxis=dict(
            title=dict(text=ylabel, font=dict(family=FONT_SANS, color=theme.TEXT_COLOR)),
            gridcolor=theme.GRID_MAJOR_COLOR, gridwidth=GRID_MAJOR_WIDTH,
            linecolor=theme.SPINE_COLOR, tickcolor=theme.TEXT_COLOR,
            zerolinecolor=theme.GRID_MAJOR_COLOR,
            automargin=False,
            side="left", ticklabelposition="inside",
            ticklabelshift=-5,
            ticklabeloverflow="allow",
        ),
        legend=dict(
            bgcolor=_hex_alpha(_COLORS_PLOT_BG, LEGEND_BG_OPACITY),
            bordercolor=BLACK_A0,
            borderwidth=0, font=dict(family=FONT_SANS, size=CHART_FONT_LEGEND),
        ),
        margin=dict(**CHART_MARGIN),
        **kwargs,
    )


def _year_ticks(start_yr, end_yr, genesis, minor_grid=False):
    """Return (tick_t_values, tick_year_labels) for a year-based x-axis.

    When *minor_grid* is True and the label step > 1, return tickvals for
    every integer year (so each gets a vertical gridline) but only label
    the years at the major step interval.
    """
    s = int(start_yr)
    e = int(end_yr)
    span = e - s
    step = 1 if span <= 15 else (2 if span <= 30 else 5)
    if minor_grid and step > 1:
        all_yrs = list(range(s, e + 1))
        ts   = [yr_to_t(y, genesis) for y in all_yrs]
        lbls = [f"'{y % 100:02d}" if (y - s) % step == 0 else "" for y in all_yrs]
    else:
        yrs  = list(range(s, e + 1, step))
        ts   = [yr_to_t(y, genesis) for y in yrs]
        lbls = [f"'{y % 100:02d}" for y in yrs]
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
_MC_TITLE_COLOR = _COLORS_MC_TITLE          # dark burnished gold — readable on light bg
_MC_LEGEND_BORDER = _COLORS_MC_BORDER      # legend border gold


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
    sel_qs = _parse_quantiles(p)
    qs_str = "/".join(_fmt_q_label(q, "") for q in sel_qs) if sel_qs else "Q50%"

    # Show active models and quantiles as separate labeled lists
    active = p.get("active_models", [])
    _MODEL_LABELS = {
        "bub": "BM", "qr": "QR", "pl": "PL", "lppl": "LPPL",
        "lp2": "LPPL\u2082", "lp3": "LPPL\u2083", "lp4": "LPPL\u2084",
        "lppl_w": "LPPL (w)", "lp2_w": "LPPL\u2082 (w)",
        "lp3_w": "LPPL\u2083 (w)", "lp4_w": "LPPL\u2084 (w)",
        "lp4_n13": "LPPL\u2084 (no \u03c9\u224813)",
        "lp4_w_n13": "LPPL\u2084 (w, no \u03c9\u224813)",
        "linppl": "LinPPL", "hybppl": "HybPPL", "hybppl_dd": "HybPPL (DD)",
        "exp": "Exp", "s2f": "S2F", "ef": "EF", "u1": "U\u2081",
    }
    if active:
        model_str = ", ".join(_MODEL_LABELS.get(m, m) for m in active)
        parts = [f"Model: {model_str}", f"Quantiles: {qs_str}"]
    else:
        parts = [f"Quantiles: {qs_str}"]

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
        color=_hex_alpha(LOG_MINOR_GRID_GRAY, ANNOT_TEXT_ALPHA),
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
    fig.layout.title.font.size = CHART_FONT_TITLE + 4
    fig.layout.title.font.color = _MC_TITLE_COLOR
    fig.layout.title.font.weight = "bold"
    fig.layout.title.x = 0.5
    fig.layout.title.xanchor = "center"
    # Global font (tick labels): serif, bold, +3px
    fig.layout.font.family = _MC_FONT_FAMILY
    fig.layout.font.size = CHART_FONT_BODY + 3
    fig.layout.font.weight = "bold"
    # Axis titles: serif, bold, +4px
    if hide_xlabel:
        fig.layout.xaxis.title.text = ""
    fig.layout.xaxis.title.font.family = _MC_FONT_FAMILY
    fig.layout.xaxis.title.font.size = CHART_FONT_BODY + 4
    fig.layout.xaxis.title.font.weight = "bold"
    fig.layout.yaxis.title.font.family = _MC_FONT_FAMILY
    fig.layout.yaxis.title.font.size = CHART_FONT_BODY + 4
    fig.layout.yaxis.title.font.weight = "bold"
    # Legend — bold but not enlarged
    fig.layout.legend.font.weight = "bold"
    fig.layout.legend.font.size = CHART_FONT_LEGEND
    fig.layout.legend.bordercolor = _MC_LEGEND_BORDER
    if legend_pos and legend_pos in _MC_LEGEND_POS:
        pos = _MC_LEGEND_POS[legend_pos]
        fig.layout.legend.x = pos["x"]
        fig.layout.legend.y = pos["y"]
        fig.layout.legend.xanchor = pos["xanchor"]
        fig.layout.legend.yanchor = pos["yanchor"]
        fig.layout.legend.bgcolor = _hex_alpha(_COLORS_PLOT_BG, MC_LEGEND_BG_ALPHA)
    # Top border line to close the plot area
    fig.add_shape(
        type="line", xref="paper", yref="paper",
        x0=0, x1=1, y0=1, y1=1,
        line=dict(color=fig.layout.yaxis.linecolor or SPINE_COLOR_FALLBACK, width=1),
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
            sizex=WM_SIZE_X, sizey=WM_SIZE_Y,
            xanchor=img_xa, yanchor="bottom",
            opacity=WM_OPACITY,
            layer="above",
        ))
    fig.add_annotation(dict(
        text="quantoshi.xyz",
        xref="paper", yref="paper",
        x=txt_x, y=0.015,
        xanchor=txt_xa, yanchor="bottom",
        showarrow=False,
        font=dict(size=CHART_FONT_WATERMARK, color=_hex_alpha(WATERMARK_TEXT_COLOR, WATERMARK_TEXT_ALPHA)),
    ))
    return fig


def _t_to_datestr(t_val, genesis):
    """Convert a single t (years since genesis) to a calendar date string."""
    return (genesis + pd.Timedelta(days=float(t_val) * 365.25)).strftime("%b %d, %Y")


def _compute_recovery(x_arr, y_arr, genesis=None):
    """For each point, find time until price recovers (first future point >= current).

    Returns a list of recovery strings: "Recovery: X.X yr (Mon YYYY)" or "" if monotonic/last.
    """
    import numpy as np
    x = np.asarray(x_arr, dtype=float)
    y = np.asarray(y_arr, dtype=float)
    n = len(y)
    if n == 0:
        return []
    result = [""] * n
    for i in range(n - 1):
        if y[i] <= 0:
            continue
        if y[i + 1] >= y[i]:
            continue
        for j in range(i + 1, n):
            if y[j] >= y[i]:
                dt = float(x[j] - x[i])
                if dt >= 0.1:
                    if genesis is not None:
                        rec_date = (genesis + pd.Timedelta(days=float(x[j]) * 365.25)).strftime("%b %Y")
                        result[i] = f"Recovery: {dt:.1f} yr ({rec_date})"
                    else:
                        result[i] = f"Recovery: {dt:.1f} yr"
                break
        else:
            result[i] = "No recovery in range"
    return result


def _add_date_hover(fig, genesis, fmt=None, recovery=False):
    """Add calendar-date hover to all traces whose x-axis is t (years since genesis).

    Traces with hoverinfo="skip" are left alone.  Traces whose x values fall
    outside the plausible t range (0.5–100) are left alone (e.g. Mode B
    supercharge uses delay-years or quantile fractions as x).

    recovery=True: compute price recovery time and append to hover.
    """
    if fmt is None:
        fmt = _HOVER_FMT_USD
    _rec_suffix = "<br>%{customdata[1]}" if recovery else ""
    _fmt_with_rec = fmt.replace("<extra>", _rec_suffix + "<extra>") if recovery else fmt
    for trace in fig.data:
        if getattr(trace, "hoverinfo", None) == "skip":
            continue
        x = trace.x
        if x is None or len(x) == 0:
            continue
        # Sanity check: x should be years-since-genesis (roughly 0.5–100).
        # Skip traces whose x-range doesn't look like t values.
        try:
            x_min, x_max = float(min(x)), float(max(x))
        except (TypeError, ValueError):
            continue
        if x_min < 0.3 or x_max > 120:
            continue
        try:
            dates = [_t_to_datestr(t, genesis) for t in x]
        except (TypeError, ValueError):
            continue
        if recovery:
            y = trace.y
            if y is not None and len(y) == len(x):
                rec = _compute_recovery(x, y, genesis=genesis)
                trace.customdata = [[d, r] for d, r in zip(dates, rec)]
            else:
                trace.customdata = [[d, ""] for d in dates]
            if not getattr(trace, "hovertemplate", None):
                trace.hovertemplate = _fmt_with_rec
            elif getattr(trace, "hovertemplate", None) == fmt:
                trace.hovertemplate = _fmt_with_rec
        else:
            trace.customdata = [[d] for d in dates]
            if not getattr(trace, "hovertemplate", None):
                trace.hovertemplate = fmt


def _apply_final_steps(fig: go.Figure, p: dict, tab: str,
                       recovery: bool = False, hover_fmt: str | None = None,
                       show_qr: bool = True, show_mc: bool = False,
                       wm_pos: str = "bottom-right") -> None:
    """Lower-level finalization: date hover, config annotation, watermark.

    Caller must already have applied typography (on the layout dict) and
    constructed the go.Figure. Used by bubble/residuals which finalize
    without the legend+MC premium steps _finalize_chart handles.
    """
    fmt = hover_fmt or (_HOVER_FMT_BTC if p.get("disp_mode") == "btc" else _HOVER_FMT_USD)
    _add_date_hover(fig, _app_ctx.M.genesis, fmt=fmt, recovery=recovery)
    _apply_config_annotation(fig, p, tab, show_qr=show_qr, show_mc=show_mc)
    _apply_watermark(fig, pos=wm_pos)


def _finalize_chart(traces: list, layout: dict, p: dict, tab: str,
                    mc_result: dict | None = None, mc_premium: bool = True
                    ) -> tuple[go.Figure, dict | None]:
    """Shared chart finalization: legend, typography, MC premium, annotations, watermark."""
    layout["showlegend"] = bool(p.get("show_legend", True))
    leg_pos = p.get("legend_pos", "outside")
    if leg_pos == "outside":
        # Horizontal legend below chart — prevents stealing width on mobile
        layout["legend"].update(
            orientation="h",
            x=0.5, y=-0.15,
            xanchor="center", yanchor="top",
        )
    elif leg_pos in _MC_LEGEND_POS:
        pos = _MC_LEGEND_POS[leg_pos]
        layout["legend"].update(
            x=pos["x"], y=pos["y"],
            xanchor=pos["xanchor"], yanchor=pos["yanchor"],
            bgcolor=_hex_alpha(_COLORS_PLOT_BG, MC_LEGEND_BG_ALPHA),
        )
    _apply_sans_typography(layout)
    layout.setdefault("uirevision", _uirevision_key(p, tab))
    fig = go.Figure(data=traces, layout=go.Layout(**layout))
    if mc_premium and p.get("mc_enabled"):
        _apply_mc_premium(fig, legend_pos=None, hide_xlabel=True)
    wm_pos = "bottom-left" if leg_pos == "bottom-right" else "bottom-right"
    _apply_final_steps(
        fig, p, tab,
        show_qr=p.get("show_qr", True),
        show_mc=p.get("show_mc", bool(p.get("mc_enabled"))),
        wm_pos=wm_pos,
    )
    return fig, mc_result


def _price_tickvals(y_lo, y_hi):
    """Decade tick values for a log price y-axis."""
    decades = [0.01, 0.1, 1, 10, 100, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8,
               1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20]
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


from colors import _hex_alpha  # noqa: F401 — re-exported for backward compat


def _build_symmetric_bands(sel_qs, y_cache, x_arr, model_color=BLACK,
                            max_bands=2, fill_mode=None):
    """Build shaded band traces from symmetric quantile pairs.

    Pairs from outside in: (lowest, highest), (2nd lowest, 2nd highest).
    Outer band = lighter opacity, inner = darker.

    Parameters
    ----------
    sel_qs : sorted list of floats (quantile values)
    y_cache : dict[q -> y array] (price or any y-axis value)
    x_arr : x-axis array (shared across all quantiles)
    model_color : hex color for the fill
    max_bands : max number of symmetric pairs to shade
    """
    if len(sel_qs) < 2:
        return []

    n = len(sel_qs)
    pairs = []
    for i in range(n // 2):
        lo_q = sel_qs[i]
        hi_q = sel_qs[n - 1 - i]
        if lo_q != hi_q and lo_q in y_cache and hi_q in y_cache:
            pairs.append((lo_q, hi_q))
    pairs = pairs[:max_bands]

    if not pairs:
        return []

    if fill_mode is None:
        fill_mode = BAND_FILL_MODE

    opacities = [0.08, 0.15] if len(pairs) >= 2 else [0.10]

    traces = []
    x = list(x_arr)
    for i, (lo_q, hi_q) in enumerate(pairs):
        alpha = opacities[i] if i < len(opacities) else opacities[-1]
        lo_y = y_cache[lo_q]
        hi_y = y_cache[hi_q]
        _invis = BLACK_A0
        traces.append(go.Scatter(
            x=x, y=lo_y, mode="lines",
            line=dict(width=0, color=_invis),
            showlegend=False, hoverinfo="skip",
        ))
        traces.append(go.Scatter(
            x=x, y=hi_y, mode="lines",
            line=dict(width=0, color=_invis),
            fill="tonexty",
            fillcolor=(_hex_alpha(quantile_shade(model_color, (lo_q + hi_q) / 2),
                                   BAND_PASTEL_ALPHA)
                       if fill_mode == "pastel"
                       else _hex_alpha(model_color, alpha)),
            showlegend=False, hoverinfo="skip",
        ))

    return traces


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
        color=_COLORS_BTC_ORANGE, y_last=mc_y_final)


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
        textfont=dict(size=CHART_FONT_ANNOT, color=color),
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
            merged_color = cluster[0]["color"] if len(colors) == 1 else CLUSTER_MERGE_GRAY
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


# ── overlay model helpers ────────────────────────────────────────────────────

def _resolve_model(model_key: str, p: dict):
    """Resolve a model key to a model object.

    Returns ``None`` when the key cannot be resolved (unknown key, missing
    user-model data, etc.).  The caller should skip the model in that case.
    """
    if model_key == "bub":
        return None  # primary model — callers draw it themselves
    if model_key == "u1":
        from btc_core import UserModel
        um_data = p.get("user_model")
        if not um_data:
            return None
        return UserModel.from_store_dict(um_data)
    return _app_ctx.PRICE_MODELS.get(model_key)


def build_overlay_traces(
    p: dict,
    ts: np.ndarray,
    ts_clamped: np.ndarray,
    sel_qs: list[float],
    disp_mode: str,
    palette: dict,
    sim_fn,
    *,
    line_shape: str = "linear",
) -> list[go.Scatter]:
    """Build alternative-model overlay traces for simulation tabs (DCA / Retire).

    Parameters
    ----------
    p : dict
        Full params dict (must contain ``active_models`` list).
    ts : array
        Time array (x-axis values).
    ts_clamped : array
        ``np.maximum(ts, 0.5)`` — passed to ``mdl.price_at``.
    sel_qs : list[float]
        Selected quantiles from the primary model.
    disp_mode : str
        ``"usd"`` or ``"btc"``.
    palette : dict
        Active colour palette.
    sim_fn : callable
        ``sim_fn(prices_q) -> vals``  where *prices_q* is a 1-D price array
        and *vals* is the resulting 1-D simulation output (e.g. accumulated
        BTC for DCA, remaining BTC for Retire).
    line_shape : str
        Plotly line shape (``"linear"`` or ``"hv"``).

    Returns
    -------
    list[go.Scatter]
        Traces for every resolved overlay model × quantile.
    """
    traces: list[go.Scatter] = []
    shade_on = bool(p.get("shade"))
    for model_key in p.get("active_models", []):
        mdl = _resolve_model(model_key, p)
        if not mdl:
            continue

        _mdl_color = _get_model_color(model_key, p)
        if mdl.quantized:
            # Collect the per-quantile line traces for this model into a
            # local list so shade bands can be emitted BEFORE them (bands
            # need to render beneath the lines). y_cache is keyed by
            # quantile so we can hand it directly to _build_symmetric_bands.
            _model_lines: list[go.Scatter] = []
            _y_cache: dict[float, np.ndarray] = {}
            for q in sel_qs:
                if q not in mdl.fits:
                    continue
                prices_q = mdl.price_at(q, ts_clamped)
                vals = sim_fn(prices_q)
                if disp_mode == "usd":
                    y_vals = vals * prices_q
                    final_lbl = fmt_price(float(y_vals[-1]))
                else:
                    # BTC mode: show only the final BTC amount. The
                    # terminal USD value (vals[-1] * prices_q[-1]) is
                    # mathematically constant across scalar-quantized
                    # bands — showing it per-quantile would duplicate
                    # the same number on every line. BM has
                    # non-scalar quantile spread so it still shows USD
                    # in its own builder.
                    y_vals = vals
                    final_lbl = f"{float(vals[-1]):.4f} BTC"
                _y_cache[q] = y_vals
                _shade = quantile_shade(_mdl_color, q)
                _model_lines.append(go.Scatter(
                    x=ts, y=y_vals, mode="lines",
                    name=f"{mdl.legend_name} {_fmt_q_label(q, '')}  \u2192  {final_lbl}",
                    line=dict(color=_shade, width=TRACE_WIDTH_OVERLAY,
                              dash=mdl.dash_style, shape=line_shape),
                    opacity=quantile_opacity(q),
                    legendgroup=mdl.short_name,
                    legendgrouptitle_text=mdl.legend_name,
                ))
            # Symmetric band shading per-model — same treatment as the
            # primary BM model on these tabs, so every quantized overlay
            # with shade enabled gets its own translucent band fan.
            if shade_on and len(_y_cache) >= 2:
                traces.extend(_build_symmetric_bands(
                    sorted(_y_cache.keys()), _y_cache, ts,
                    model_color=_mdl_color))
            traces.extend(_model_lines)
        else:
            # Non-quantized: single trajectory at Q50%
            prices_q = mdl.price_at(0.5, ts_clamped)
            vals = sim_fn(prices_q)
            if disp_mode == "usd":
                y_vals = vals * prices_q
                final_lbl = fmt_price(float(y_vals[-1]))
            else:
                y_vals = vals
                final_usd = fmt_price(float(vals[-1]) * float(prices_q[-1]))
                final_lbl = f"{float(vals[-1]):.4f} BTC  ({final_usd})"
            traces.append(go.Scatter(
                x=ts, y=y_vals, mode="lines",
                name=f"{mdl.legend_name}  \u2192  {final_lbl}",
                line=dict(color=_mdl_color,
                          width=TRACE_WIDTH_OVERLAY, dash=mdl.dash_style,
                          shape=line_shape),
                legendgroup=mdl.short_name,
            ))
    return traces


# Re-export from _app_ctx for backward compat (used by chart builders and callbacks)
FREQ_PPY = _app_ctx.FREQ_PPY
_FREQ_STEP_DAYS = _app_ctx.FREQ_STEP_DAYS

