"""figures.py — Plotly chart builders for the Bitcoin Projections web app.

Each function takes a ModelData instance and a params dict of control values
and returns a go.Figure ready for dcc.Graph.
"""
from __future__ import annotations

import math
import base64
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# btc_app/ is added to sys.path by app.py before this import
import bisect
from typing import Any
import _app_ctx
from btc_core import ModelData, qr_price, yr_to_t, today_t, fmt_price, leo_weighted_entry
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
)


def _interp_qr_price(q: float, t: float, qr_fits: dict) -> float:
    """Return QR price for arbitrary quantile q in (0,1), interpolating in log space."""
    qs = sorted(qr_fits.keys())
    if q <= qs[0]:
        return float(qr_price(qs[0], t, qr_fits))
    if q >= qs[-1]:
        return float(qr_price(qs[-1], t, qr_fits))
    idx = bisect.bisect_left(qs, q)
    q_lo, q_hi = qs[idx - 1], qs[idx]
    p_lo = float(qr_price(q_lo, t, qr_fits))
    p_hi = float(qr_price(q_hi, t, qr_fits))
    frac = (q - q_lo) / (q_hi - q_lo)
    return float(np.exp(np.log(p_lo) + frac * (np.log(p_hi) - np.log(p_lo))))

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

# ── shared small helpers ──────────────────────────────────────────────────────


def _fmt_q_label(q: float) -> str:
    """Format quantile as 'Q{pct}%' with appropriate precision."""
    pct = q * 100
    return f"Q{pct:.4g}%" if pct >= 1 else f"Q{pct:.3g}%"


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


def _apply_mc_premium(fig: go.Figure, legend_pos: str = "top-left") -> None:
    """Upgrade figure fonts / colours for premium MC-rendered charts.

    *legend_pos*: move legend inside the plot area at the named corner.
    Pass ``None`` to keep the default (outside) position.
    """
    # Title: serif, gold, centered, +2px
    fig.layout.title.font.family = _MC_FONT_FAMILY
    fig.layout.title.font.size = _FONT_TITLE + 2
    fig.layout.title.font.color = _MC_TITLE_COLOR
    fig.layout.title.x = 0.5
    fig.layout.title.xanchor = "center"
    # Global font (tick labels): serif, +1px
    fig.layout.font.family = _MC_FONT_FAMILY
    fig.layout.font.size = _FONT_BODY + 1
    # Axis titles: serif, +2px
    fig.layout.xaxis.title.font.family = _MC_FONT_FAMILY
    fig.layout.xaxis.title.font.size = _FONT_BODY + 2
    fig.layout.yaxis.title.font.family = _MC_FONT_FAMILY
    fig.layout.yaxis.title.font.size = _FONT_BODY + 2
    # Legend — inside the plot at the specified corner
    fig.layout.legend.bordercolor = _MC_LEGEND_BORDER
    if legend_pos and legend_pos in _MC_LEGEND_POS:
        pos = _MC_LEGEND_POS[legend_pos]
        fig.layout.legend.x = pos["x"]
        fig.layout.legend.y = pos["y"]
        fig.layout.legend.xanchor = pos["xanchor"]
        fig.layout.legend.yanchor = pos["yanchor"]
        fig.layout.legend.bgcolor = "rgba(255,255,255,0.7)"
    # yaxis2 (dual-y) if present
    if hasattr(fig.layout, "yaxis2") and fig.layout.yaxis2.title is not None:
        fig.layout.yaxis2.title.font.family = _MC_FONT_FAMILY
        fig.layout.yaxis2.title.font.size = _FONT_BODY + 2


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
            sizex=0.09, sizey=0.12,
            xanchor=img_xa, yanchor="bottom",
            opacity=0.55,
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


def _dense_colorscale(color_fn, n=256):
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
    # anchor points at breakpoints (normalised 0-1)
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
            lo_p = qr_price(sel_qs[j],   t_arr, m.qr_fits) * (stack if stack > 0 else 1)
            hi_p = qr_price(sel_qs[j+1], t_arr, m.qr_fits) * (stack if stack > 0 else 1)
            col  = m.qr_colors.get(sel_qs[j], "#888888")
            traces.append(go.Scatter(
                x=list(t_arr), y=list(lo_p),
                mode="lines", line=dict(width=0),
                showlegend=False, hoverinfo="skip",
            ))
            traces.append(go.Scatter(
                x=list(t_arr), y=list(hi_p),
                mode="lines", line=dict(width=0), fill="tonexty",
                fillcolor=col.replace("#", "rgba(").rstrip(")") if False else _hex_alpha(col, 0.08),
                showlegend=False, hoverinfo="skip",
            ))

    # ── quantile lines ────────────────────────────────────────────────────────
    for q in sel_qs:
        if q not in m.qr_fits:
            continue
        prices = qr_price(q, t_arr, m.qr_fits) * (stack if stack > 0 else 1)
        lbl = _fmt_q_label(q)
        if stack > 0:
            lbl += f"  \u2192  {fmt_price(float(prices[-1]))}"
        col = m.qr_colors.get(q, "#888888")
        traces.append(go.Scatter(
            x=list(t_arr), y=list(prices),
            mode="lines", name=lbl,
            line=dict(color=col, width=_QR_LINE_WIDTH),
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
    layout["shapes"] = shapes

    if stack > 0:
        layout["annotations"] = [dict(
            text=f"Stack: {p['stack']:.6g} BTC",
            xref="paper", yref="paper", x=0.99, y=0.01,
            xanchor="right", yanchor="bottom",
            showarrow=False, font=dict(size=_FONT_LEGEND, color=m.TEXT_COLOR),
            bgcolor=m.PLOT_BG_COLOR, bordercolor=m.SPINE_COLOR, borderwidth=1,
        )]

    fig = go.Figure(data=traces, layout=go.Layout(**layout))
    _apply_watermark(fig)
    return fig


def _hex_alpha(hex_color, alpha):
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
            else:
                tx = ""

            if tx:
                cell_norm = (vc2 - zmin) / max(zmax - zmin, 1e-6)
                txt_col = "#ffffff" if cell_norm < 0.55 else "#111111"
                annots.append(dict(
                    x=ci, y=ri,
                    text=tx.replace("\n", "<br>"),
                    showarrow=False,
                    font=dict(size=cell_fs, color=txt_col),
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
    eyr = int(p.get("entry_yr", 2020))
    eq  = float(p.get("entry_q", 50)) / 100.0   # stored as percentage (e.g. 7.5 → 0.075)
    entry_t = yr_to_t(eyr, m.genesis)
    live_price = p.get("live_price")
    ep  = float(live_price) if live_price else _interp_qr_price(eq, entry_t, m.qr_fits)

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
    xqs = sorted([float(q) for q in xqs_raw if float(q) in m.qr_fits], reverse=True)

    if not eyrs or not xqs:
        return _error_figure(m, "No data — adjust Entry / Exit settings")

    mc = np.zeros((len(xqs), len(eyrs)))
    mp = np.zeros((len(xqs), len(eyrs)))
    mm = np.zeros((len(xqs), len(eyrs)))
    for ci, ey in enumerate(eyrs):
        et = yr_to_t(ey, m.genesis)
        nyr = et - entry_t if p.get("use_lots") and lots else float(ey - eyr)
        for ri, xq in enumerate(xqs):
            xpp = float(qr_price(xq, et, m.qr_fits))
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
    _apply_watermark(fig)
    return fig


def build_mc_heatmap_figure(m: ModelData, p: dict[str, Any]) -> tuple[go.Figure, dict | None]:
    """Build a standalone MC heatmap figure from MC-simulated CAGR percentiles.
    Returns (fig, mc_result) or (empty_fig, None).
    """
    eyr = int(p.get("mc_start_yr", p.get("entry_yr", 2020)))
    eq  = float(p.get("mc_entry_q", p.get("entry_q", 50))) / 100.0
    entry_t = yr_to_t(eyr, m.genesis)
    live_price = p.get("live_price")
    ep  = float(live_price) if live_price else _interp_qr_price(eq, entry_t, m.qr_fits)

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
    if p.get("mc_enabled"):
        _apply_mc_premium(fig, legend_pos=None)
    _apply_watermark(fig)
    return fig, mc_result


# ── DCA helpers ──────────────────────────────────────────────────────────────

def _dca_sc_overlay(m, p, ts, sel_qs, start_stack, all_prices, disp_mode, ppy):
    """Run Stack-celerator overlay simulation for DCA tab.

    Returns (sc_traces, all_sc_usd_vals, all_sc_btc_vals).
    """
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
        if q not in m.qr_fits:
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

                if loan_type == "amortizing":
                    interest_p  = outstanding * r
                    principal_p = pmt - interest_p
                    outstanding = max(outstanding - principal_p, 0.0)

                if loan_type == "interest_only" and period_in_cycle == term_periods - 1:
                    if sc_rollover:
                        pass
                    else:
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
        col = m.qr_colors.get(q, "#888888")
        sc_traces.append(go.Scatter(
            x=list(ts), y=list(y_sc), mode="lines", name=lbl_sc,
            line=dict(color=col, width=_QR_LINE_WIDTH, dash="dash"),
        ))

    return sc_traces, all_sc_usd_vals, all_sc_btc_vals


def _render_edge_annots(edge_items, x_positions=None, yref="y"):
    """Build staggered right-edge value annotations.

    Each item: (x, final_y, label, color, prefix).
    *label* may be a number (formatted via fmt_price) or a pre-formatted string.
    *x_positions*: paper-x offsets for stagger columns (default [0.97, 0.87, 0.77]).
    *yref*: y-axis reference for arrow targets (default "y").
    Returns list of annotation dicts.
    """
    if not edge_items:
        return []
    edge_items.sort(key=lambda it: it[1])
    _AY_STEP = 18
    xp_list = x_positions or [0.97, 0.87, 0.77]
    annots = []
    n = len(edge_items)
    for idx, (ax_x, fy, lbl, col, pfx) in enumerate(edge_items):
        ay = -35 - int((n - 1 - idx) * _AY_STEP)
        xp = xp_list[idx % len(xp_list)]
        lbl_text = lbl if isinstance(lbl, str) else fmt_price(lbl)
        annots.append(dict(
            x=xp, xref="paper",
            y=fy, yref=yref,
            text=f"<b>{pfx}{lbl_text}</b>",
            showarrow=True, arrowhead=2, arrowsize=1,
            arrowcolor=col, arrowwidth=1.5,
            ax=0, ay=ay,
            font=dict(size=_FONT_ANNOT, color=col),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor=col, borderwidth=1, borderpad=2,
        ))
    return annots


# ── DCA Accumulator ───────────────────────────────────────────────────────────

def build_dca_figure(m: ModelData, p: dict[str, Any]) -> tuple[go.Figure, dict | None]:
    """
    p keys: start_yr, end_yr, start_stack, amount, freq, disp_mode,
            selected_qs, log_y, show_today, dual_y,
            lots, use_lots
    """
    freq_str = p.get("freq", "Monthly")
    ppy  = FREQ_PPY.get(freq_str, 12)
    dt   = 1.0 / ppy
    syr  = int(p.get("start_yr", 2024))
    eyr  = int(p.get("end_yr",   2035))
    if eyr <= syr:
        return _error_figure(m, "Set end year > start year"), None

    t_start = max(yr_to_t(syr, m.genesis), 1.0)
    t_end   = yr_to_t(eyr, m.genesis)
    ts      = np.arange(t_start, t_end + dt * 0.5, dt)
    if len(ts) == 0:
        return go.Figure(), None

    start_stack = float(p.get("start_stack", 0))
    lots = p.get("lots") or []
    if p.get("use_lots") and lots:
        result = leo_weighted_entry(lots)
        if result:
            start_stack = result[3]  # total_btc

    amount    = float(p.get("amount", 100))
    disp_mode = p.get("disp_mode", "btc")
    sel_qs    = sorted([float(q) for q in (p.get("selected_qs") or [])])

    traces = []
    all_btc_vals = {}  # q -> BTC balance array
    all_usd_vals = {}  # q -> USD value array (always tracked for dual-y)
    all_prices   = {}  # q -> price array — reused by SC loop to avoid redundant qr_price calls
    ts_clamped = np.maximum(ts, 0.5)
    for q in sel_qs:
        if q not in m.qr_fits:
            continue
        prices_q = qr_price(q, ts_clamped, m.qr_fits)
        vals = start_stack + np.cumsum(amount / prices_q)
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
        col = m.qr_colors.get(q, "#888888")
        traces.append(go.Scatter(
            x=list(ts), y=list(y_vals), mode="lines", name=lbl,
            line=dict(color=col, width=_QR_LINE_WIDTH),
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

    tick_ts, tick_lbls = _year_ticks(syr, eyr, m.genesis,
                                     minor_grid=p.get("minor_grid"))

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
    layout = _dark_layout(
        m,
        title=title_line,
        xlabel="Year",
        ylabel=ylabel,
    )
    layout["xaxis"].update(
        tickvals=tick_ts, ticktext=tick_lbls, tickangle=-45,
        range=[t_start, t_end],
    )
    if p.get("log_y"):
        layout["yaxis"]["type"] = "log"
        if p.get("minor_grid"):
            layout["yaxis"]["minor"] = _LOG_MINOR
    layout["shapes"] = shapes

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

    # ── Right-edge USD annotations (replaces dual-y axis) ──────────────────
    # Collect (x, final_y, final_usd, color, prefix) then stagger to avoid overlap.
    x_end = float(ts[-1]) if len(ts) > 0 else t_end
    _edge_items = []
    if p.get("dual_y") and all_usd_vals:
        # QR quantile final USD values
        for q in sel_qs:
            if q not in all_usd_vals:
                continue
            final_usd = float(all_usd_vals[q][-1])
            final_y = float(all_btc_vals[q][-1]) if disp_mode == "btc" else final_usd
            col = m.qr_colors.get(q, "#888888")
            _edge_items.append((x_end, final_y, final_usd, col, ""))
        # SC quantile final USD values
        for q, sc_usd in all_sc_usd_vals.items():
            final_usd = float(sc_usd[-1])
            final_y = final_usd if disp_mode == "usd" else float(all_sc_btc_vals[q][-1])
            col = m.qr_colors.get(q, "#888888")
            _edge_items.append((x_end, final_y, final_usd, col, "SC "))

    # ── Stack-celeration factor → append to title ────────────────────────────
    if sc_factor_val is not None:
        layout["title"]["text"] += (
            f"<br><b>Stack-celeration: {sc_factor_val:.2f}\u00d7</b>"
        )

    # ── Monte Carlo fan overlay ─────────────────────────────────────────────
    mc_result = None
    if _HAS_MARKOV and p.get("mc_enabled"):
        mc_traces, mc_result, mc_fan_usd = _mc_dca_overlay(m, p, ts, t_start, dt, start_stack, disp_mode)
        traces.extend(mc_traces)
        # MC median → add to edge items for staggered annotation
        if p.get("dual_y") and mc_fan_usd and 0.50 in mc_fan_usd:
            mc_med_usd = mc_fan_usd[0.50]
            if len(mc_med_usd) > 0:
                final_usd = float(mc_med_usd[-1])
                mc_med_y = float(mc_traces[-1].y[-1]) if mc_traces else final_usd
                mc_x = float(mc_traces[-1].x[-1]) if mc_traces else x_end
                _edge_items.append((mc_x, mc_med_y, final_usd, _BTC_ORANGE, "MC "))
        # MC median final value + multiplier → append to title
        if mc_fan_usd and 0.50 in mc_fan_usd and len(mc_fan_usd[0.50]) > 0:
            mc_med_final = float(mc_fan_usd[0.50][-1])
            mc_roi = mc_med_final / total_spent if total_spent > 0 else 0
            layout["title"]["text"] += f"  ·  MC median {fmt_price(mc_med_final)}  ·  {mc_roi:.1f}\u00d7"

    # ── Render staggered right-edge annotations ─────────────────────────────
    if _edge_items:
        layout.setdefault("annotations", []).extend(_render_edge_annots(_edge_items))

    layout["showlegend"] = bool(p.get("show_legend", True))
    fig = go.Figure(data=traces, layout=go.Layout(**layout))
    if p.get("mc_enabled"):
        _apply_mc_premium(fig, legend_pos="top-left")
    _apply_watermark(fig)
    return fig, mc_result


# Re-export from _app_ctx for backward compat (used by chart builders and callbacks)
FREQ_PPY = _app_ctx.FREQ_PPY
_FREQ_STEP_DAYS = _app_ctx.FREQ_STEP_DAYS


# ── BTC Retireator ────────────────────────────────────────────────────────────

def build_retire_figure(m: ModelData, p: dict[str, Any]) -> tuple[go.Figure, dict | None]:
    """
    p keys: start_yr, end_yr, start_stack, wd_amount, freq, inflation,
            disp_mode, selected_qs, log_y, show_today, dual_y, annotate,
            lots, use_lots
    """
    freq_str = p.get("freq", "Monthly")
    ppy  = FREQ_PPY.get(freq_str, 12)
    dt   = 1.0 / ppy
    syr  = int(p.get("start_yr", 2025))
    eyr  = int(p.get("end_yr",   2045))
    if eyr <= syr:
        return _error_figure(m, "Set end year > start year"), None

    t_start = max(yr_to_t(syr, m.genesis), 1.0)
    t_end   = yr_to_t(eyr, m.genesis)
    ts      = np.arange(t_start, t_end + dt * 0.5, dt)
    if len(ts) == 0:
        return go.Figure(), None

    start_stack = float(p.get("start_stack", 1.0))
    lots = p.get("lots") or []
    if p.get("use_lots") and lots:
        result = leo_weighted_entry(lots)
        if result:
            start_stack = result[3]  # total_btc

    wd_amount = float(p.get("wd_amount", 5000))
    inflation = float(p.get("inflation", 0)) / 100.0
    disp_mode = p.get("disp_mode", "btc")
    sel_qs    = sorted([float(q) for q in (p.get("selected_qs") or [])])

    traces   = []
    deplete_annots = []
    all_btc_vals = {}  # q -> BTC balance array (for dual-y median)

    ts_clamped = np.maximum(ts, 0.5)
    adj_wd_arr = wd_amount * ((1 + inflation) ** (ts - t_start))
    for q in sel_qs:
        if q not in m.qr_fits:
            continue
        prices = qr_price(q, ts_clamped, m.qr_fits)
        vals = np.maximum(start_stack - np.cumsum(adj_wd_arr / prices), 0.0)
        all_btc_vals[q] = vals

        if disp_mode == "usd":
            prices_arr = prices
            y_vals = vals * prices_arr
            final_lbl = fmt_price(float(y_vals[-1]))
        else:
            y_vals = vals
            final_lbl = f"{float(vals[-1]):.4f} BTC"

        lbl = _fmt_q_label(q) + f"  →  {final_lbl}"
        col = m.qr_colors.get(q, "#888888")
        traces.append(go.Scatter(
            x=list(ts), y=list(y_vals), mode="lines", name=lbl,
            line=dict(color=col, width=_QR_LINE_WIDTH),
        ))

        # depletion annotation
        if p.get("annotate"):
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

    shapes = []
    if p.get("show_today"):
        td = today_t(m.genesis)
        if t_start <= td <= t_end:
            shapes.append(dict(
                type="line", x0=td, x1=td, y0=0, y1=1,
                yref="paper", line=dict(color=_TODAY_LINE_COLOR, dash="dash", width=_TODAY_LINE_WIDTH),
                opacity=_TODAY_LINE_OPACITY,
            ))

    tick_ts, tick_lbls = _year_ticks(syr, eyr, m.genesis,
                                     minor_grid=p.get("minor_grid"))
    ylabel = "USD Value" if disp_mode == "usd" else "BTC Remaining"
    layout = _dark_layout(
        m,
        title=f"Bitcoin Retireator — {fmt_price(wd_amount)}/{freq_str.lower()[:-2] if freq_str.endswith('ly') else freq_str}",
        xlabel="Year",
        ylabel=ylabel,
    )
    layout["xaxis"].update(
        tickvals=tick_ts, ticktext=tick_lbls, tickangle=-45,
        range=[t_start, t_end],
    )
    if p.get("log_y"):
        layout["yaxis"]["type"] = "log"
        layout["yaxis"]["dtick"] = 1  # decades only (drop 2× and 5×)
        if p.get("minor_grid"):
            layout["yaxis"]["minor"] = _LOG_MINOR
    layout["shapes"]      = shapes
    layout["annotations"] = deplete_annots

    # ── dual Y-axis — one dotted trace per quantile on the opposite unit ────
    if p.get("dual_y") and traces and all_btc_vals:
        if disp_mode == "usd":
            y2_lbl = "BTC Remaining"
        else:
            y2_lbl = "USD Value"
        for q in sel_qs:
            if q not in all_btc_vals:
                continue
            if disp_mode == "usd":
                y2_vals = all_btc_vals[q]
            else:
                y2_vals = all_btc_vals[q] * qr_price(q, ts_clamped, m.qr_fits)
            col = m.qr_colors.get(q, "#888888")
            traces.append(go.Scatter(
                x=list(ts), y=list(y2_vals),
                mode="lines", name=f"{_fmt_q_label(q)} {y2_lbl}",
                line=dict(color=col, dash="dot", width=1),
                yaxis="y2", showlegend=False,
            ))
        # $ prefix only when secondary axis shows USD
        y2_tick_prefix = "$" if y2_lbl == "USD Value" else ""
        layout["yaxis2"] = dict(
            title=dict(text=y2_lbl, font=dict(color=m.TEXT_COLOR)),
            overlaying="y", side="right",
            showgrid=False, linecolor=m.SPINE_COLOR,
            tickcolor=m.TEXT_COLOR,
            ticks="outside", ticklen=8, tickwidth=1.5,
            tickprefix=y2_tick_prefix,
        )
        if p.get("log_y"):
            layout["yaxis2"]["type"] = "log"
            layout["yaxis2"]["dtick"] = 1

    # ── Monte Carlo fan overlay ─────────────────────────────────────────────
    mc_traces_list = []
    mc_result = None
    if _HAS_MARKOV and p.get("mc_enabled"):
        mc_traces_list, mc_annots, mc_result = _mc_retire_overlay(
            m, p, ts, t_start, t_end, dt,
            start_stack, disp_mode, len(deplete_annots))
        traces.extend(mc_traces_list)
        if mc_annots:
            deplete_annots.extend(mc_annots)
            layout["annotations"] = deplete_annots

    # Sort all depletion annotations by x and reassign stagger levels
    if len(deplete_annots) > 1:
        deplete_annots.sort(key=lambda a: a["x"])
        for i, a in enumerate(deplete_annots):
            a["ay"] = _ANNOT_STAGGER_Y[i % 3]
        layout["annotations"] = deplete_annots

    # ── Right-edge value annotations (arrow style, paper coords) ────────────
    # Use yref="paper" to avoid dual-y / log-scale annotation positioning bug.
    # Mirrors depleted annotation style: arrow from text to the right axis edge.
    if p.get("annotate"):
        is_log = bool(p.get("log_y"))

        def _axis_paper_fn(axis_key):
            """Compute paper_y function from explicit axis range."""
            ys = []
            for tr in traces:
                ya = getattr(tr, 'yaxis', None) or 'y'
                if ya != axis_key:
                    continue
                if tr.y is not None:
                    ys.extend(float(v) for v in tr.y if float(v) > 0)
            if not ys:
                return None, None
            if is_log:
                lmin = float(np.log10(min(ys)))
                lmax = float(np.log10(max(ys)))
                span = max(lmax - lmin, 0.3)
                pad = 0.05 * span
                r0, r1 = lmin - pad, lmax + pad
                def fn(v):
                    if v <= 0:
                        return 0.0
                    return (float(np.log10(v)) - r0) / (r1 - r0)
                return fn, [r0, r1]
            else:
                ylo, yhi = 0.0, max(ys)
                span = max(yhi, 1e-6)
                pad = 0.05 * span
                r0, r1 = ylo - pad, yhi + pad
                def fn(v):
                    return (v - r0) / (r1 - r0)
                return fn, [r0, r1]

        to_py1, y1_range = _axis_paper_fn('y')
        if y1_range:
            layout["yaxis"]["range"] = y1_range

        to_py2, y2_range = None, None
        if p.get("dual_y") and "yaxis2" in layout:
            to_py2, y2_range = _axis_paper_fn('y2')
            if y2_range:
                layout["yaxis2"]["range"] = y2_range

        edge_annots = []   # (x_data, paper_y, label, color)
        ts_end_arr = np.maximum(np.array([ts[-1]]), 0.5)
        x_right = float(ts[-1])

        for q in sel_qs:
            if q not in all_btc_vals:
                continue
            btc_final = float(all_btc_vals[q][-1])
            if btc_final <= 0:
                continue
            col = m.qr_colors.get(q, "#888888")
            usd_final = btc_final * float(qr_price(q, ts_end_arr, m.qr_fits)[0])

            # Primary (solid) trace annotation
            if to_py1:
                if disp_mode == "usd":
                    lbl1 = fmt_price(usd_final)
                    py1 = to_py1(usd_final)
                else:
                    lbl1 = f"{btc_final:.4f} \u20bf"
                    py1 = to_py1(btc_final)
                edge_annots.append((x_right, py1, lbl1, col))

            # Secondary (dashed) trace annotation
            if to_py2:
                if disp_mode == "usd":
                    lbl2 = f"{btc_final:.4f} \u20bf"
                    py2 = to_py2(btc_final)
                else:
                    lbl2 = fmt_price(usd_final)
                    py2 = to_py2(usd_final)
                edge_annots.append((x_right, py2, lbl2, col))

        # MC median endpoint — annotate at trace terminus, not right edge
        mc_col = "#F7931A"
        if to_py1:
            for tr in mc_traces_list:
                if not (getattr(getattr(tr, "line", None), "dash", None) == "dot"):
                    continue
                x_data = list(tr.x) if tr.x is not None else []
                y_data = list(tr.y) if tr.y is not None else []
                if not x_data or not y_data:
                    continue
                mc_x_last = float(x_data[-1])
                # Use the y-value at the visible right edge if MC extends beyond
                if mc_x_last > x_right:
                    ann_x = x_right
                    ann_y = float(np.interp(x_right, x_data, y_data))
                else:
                    ann_x = mc_x_last
                    ann_y = float(y_data[-1])
                if ann_y <= 0:
                    continue
                if disp_mode == "usd":
                    mc_lbl = fmt_price(ann_y)
                else:
                    mc_lbl = f"{ann_y:.4f} \u20bf"
                if mc_x_last < x_right:
                    ann_yr = int(syr + (mc_x_last - t_start)
                                 / max(t_end - t_start, 1e-6) * (eyr - syr))
                    mc_lbl = f"\u2248{ann_yr}  {mc_lbl}"
                py = to_py1(ann_y)
                edge_annots.append((ann_x, py, f"MC {mc_lbl}", mc_col))

        # Sort by paper_y, stagger overlapping labels
        edge_annots.sort(key=lambda x: x[1])
        _EDGE_AY = [-25, -40, -55]
        for i, (ax_x, py, lbl, col) in enumerate(edge_annots):
            ay = _EDGE_AY[i % 3]
            layout.setdefault("annotations", []).append(dict(
                x=ax_x, xref="x",
                y=py, yref="paper",
                text=lbl,
                showarrow=True, arrowhead=2, arrowsize=1,
                arrowcolor=col, arrowwidth=1.5,
                ax=-40, ay=ay,
                font=dict(size=_FONT_ANNOT, color=col),
            ))

    layout["showlegend"] = bool(p.get("show_legend", True))
    # Legend position — user-selectable
    leg_pos = p.get("legend_pos", "outside")
    if leg_pos != "outside" and leg_pos in _MC_LEGEND_POS:
        pos = _MC_LEGEND_POS[leg_pos]
        layout["legend"].update(
            x=pos["x"], y=pos["y"],
            xanchor=pos["xanchor"], yanchor=pos["yanchor"],
            bgcolor="rgba(255,255,255,0.7)",
        )
    fig = go.Figure(data=traces, layout=go.Layout(**layout))
    if p.get("mc_enabled"):
        # legend_pos=None so MC premium doesn't override user's legend choice
        _apply_mc_premium(fig, legend_pos=None)
    wm_pos = "bottom-left" if leg_pos == "bottom-right" else "bottom-right"
    _apply_watermark(fig, pos=wm_pos)
    return fig, mc_result


# ── HODL Supercharger ─────────────────────────────────────────────────────────

_DELAY_COLORS = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#E07000']
_ANNOT_COLORS = ['#636EFA', '#EF553B', '#1D8348', '#AB63FA', '#E07000']
_DASH_STYLES  = ['solid', 'dash', 'dot', 'dashdot', 'longdash']


def build_supercharge_figure(m: ModelData, p: dict[str, Any]) -> tuple[go.Figure, dict | None]:
    """
    p keys: mode ('a'/'b'), start_stack, start_yr, delays (list), freq, inflation,
            selected_qs, chart_layout (0/1/2), display_q,
            wd_amount (Mode A), end_yr (Mode A), disp_mode (Mode A),
            log_y, annotate, show_today, show_legend,
            target_yr (Mode B), lots, use_lots
    """

    mode         = p.get("mode", "a")
    freq_str     = p.get("freq", "Monthly")
    ppy          = FREQ_PPY.get(freq_str, 12)
    dt           = 1.0 / ppy
    syr          = int(p.get("start_yr", pd.Timestamp.today().year))
    inflation    = float(p.get("inflation", 4)) / 100.0
    chart_layout = int(p.get("chart_layout", 0))
    display_q    = float(p.get("display_q", 0.5))
    show_legend  = bool(p.get("show_legend", True))

    # Starting stack (lots override)
    start_stack = float(p.get("start_stack", 1.0))
    lots = p.get("lots") or []
    if p.get("use_lots") and lots:
        result = leo_weighted_entry(lots)
        if result:
            start_stack = result[3]

    # Quantiles
    sel_qs = sorted([float(q) for q in (p.get("selected_qs") or [])
                     if float(q) in m.qr_fits])
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
                prices = qr_price(q, ts_d_clamped, m.qr_fits)
                vals = np.maximum(start_stack - np.cumsum(adj_wd_d / prices), 0.0)
                depl_mask = vals == 0.0
                depl_t = float(ts_d[np.argmax(depl_mask)]) if depl_mask.any() else None
                if disp_mode == "usd":
                    y_vals = vals * prices
                else:
                    y_vals = vals
                results[(d, q)] = (ts_d, y_vals, depl_t, t_start_d)

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
                ts_d, y_vals, depl_t, t_start_d = results[key]
                col   = _DELAY_COLORS[di % len(_DELAY_COLORS)]
                d_lbl = f"+{int(d)}yr" if d == int(d) else f"+{d:.1f}yr"
                final = (fmt_price(float(y_vals[-1])) if disp_mode == "usd"
                         else f"{float(y_vals[-1]):.4f} BTC")
                traces.append(go.Scatter(
                    x=list(ts_d), y=list(y_vals), mode="lines",
                    name=f"Delay {d_lbl}  \u2192  {final}",
                    line=dict(color=col, width=2),
                ))
                if p.get("annotate") and depl_t is not None:
                    deplete_annots.append(_depl_annot(depl_t, t_start_d, d,
                                                      _ANNOT_COLORS[di % len(_ANNOT_COLORS)],
                                                      len(deplete_annots)))

        elif chart_layout == 1:
            # Color = quantile, line style = delay
            for q in sel_qs:
                col   = m.qr_colors.get(q, "#888888")
                q_lbl = _fmt_q_label(q)
                for di, d in enumerate(delays):
                    key = (d, q)
                    if key not in results:
                        continue
                    ts_d, y_vals, depl_t, t_start_d = results[key]
                    d_lbl = f"+{int(d)}yr" if d == int(d) else f"+{d:.1f}yr"
                    traces.append(go.Scatter(
                        x=list(ts_d), y=list(y_vals), mode="lines",
                        name=f"{q_lbl} delay={d_lbl}",
                        line=dict(color=col, width=_QR_LINE_WIDTH,
                                  dash=_DASH_STYLES[di % len(_DASH_STYLES)]),
                    ))
                    if p.get("annotate") and depl_t is not None:
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
                if p.get("annotate"):
                    for q in sel_qs:
                        key = (d, q)
                        if key not in results:
                            continue
                        _, _, depl_t, t_start_d = results[key]
                        if depl_t is not None:
                            deplete_annots.append(_depl_annot(depl_t, t_start_d, d,
                                                              _ANNOT_COLORS[di % len(_ANNOT_COLORS)],
                                                              len(deplete_annots)))

        t_start_base = max(yr_to_t(syr, m.genesis), 1.0)
        tick_ts, tick_lbls = _year_ticks(syr, eyr, m.genesis,
                                         minor_grid=p.get("minor_grid"))
        ylabel = "USD Value" if disp_mode == "usd" else "BTC Remaining"
        layout = _dark_layout(
            m,
            title=(f"HODL Supercharger \u2014 {fmt_price(wd_amount)}{freq_label} \u00b7 "
                   f"Retire {syr}+ \u00b7 to {eyr}"),
            xlabel="Year", ylabel=ylabel,
        )
        layout["xaxis"].update(
            tickvals=tick_ts, ticktext=tick_lbls, tickangle=-45,
            range=[t_start_base, t_end],
        )
        if p.get("log_y"):
            layout["yaxis"]["type"] = "log"
            if p.get("minor_grid"):
                layout["yaxis"]["minor"] = _LOG_MINOR
        layout["annotations"] = deplete_annots
        shapes = []
        if p.get("show_today"):
            td = today_t(m.genesis)
            if t_start_base <= td <= t_end:
                shapes.append(dict(
                    type="line", x0=td, x1=td, y0=0, y1=1,
                    yref="paper",
                    line=dict(color=_TODAY_LINE_COLOR, dash="dash", width=_TODAY_LINE_WIDTH),
                    opacity=_TODAY_LINE_OPACITY,
                ))
        # ── Monte Carlo fan overlay ───────────────────────────────────────────
        mc_traces_list = []
        mc_result = None
        if _HAS_MARKOV and p.get("mc_enabled"):
            t_start_base = max(yr_to_t(syr, m.genesis), 1.0)
            mc_traces_list, mc_annots, mc_result = _mc_supercharge_overlay(
                m, p, ts_d if delays == [0.0] else np.arange(t_start_base, t_end + dt * 0.5, dt),
                t_start_base, t_end, dt, start_stack, disp_mode,
                len(deplete_annots))
            traces.extend(mc_traces_list)
            if mc_annots:
                deplete_annots.extend(mc_annots)
                layout["annotations"] = deplete_annots

        # Sort all depletion annotations by x and reassign stagger levels
        if len(deplete_annots) > 1:
            deplete_annots.sort(key=lambda a: a["x"])
            for i, a in enumerate(deplete_annots):
                a["ay"] = _ANNOT_STAGGER_Y[i % 3]
            layout["annotations"] = deplete_annots

        # ── Right-edge / endpoint value labels ─────────────────────────────
        # Use text traces (go.Scatter mode="markers+text") instead of
        # annotations — consistent with Retire tab; avoids paper-x arrowhead
        # misalignment on declining traces.
        if p.get("annotate"):
            for (d, q), (ts_d_r, y_vals_r, depl_t_r, _) in results.items():
                if depl_t_r is not None:
                    continue  # depleted — already has year annotation
                y_final = float(y_vals_r[-1])
                if y_final <= 0:
                    continue
                if chart_layout == 0:
                    di = delays.index(d) if d in delays else 0
                    col = _DELAY_COLORS[di % len(_DELAY_COLORS)]
                elif chart_layout == 2:
                    continue  # bands don't get individual labels
                else:
                    col = m.qr_colors.get(q, "#888888")
                if disp_mode == "usd":
                    lbl = fmt_price(y_final)
                else:
                    lbl = f"{y_final:.4f} \u20bf"
                n_pts = len(ts_d_r)
                idx_ann = max(0, n_pts - max(3, n_pts // 20))
                traces.append(go.Scatter(
                    x=[float(ts_d_r[idx_ann])], y=[float(y_vals_r[idx_ann])],
                    mode="markers+text",
                    marker=dict(size=7, color=col, symbol="circle"),
                    text=[f"  \u2192 {lbl}"],
                    textposition="middle right",
                    textfont=dict(size=_FONT_ANNOT, color=col),
                    showlegend=False, hoverinfo="skip",
                ))
            # MC median endpoint
            mc_col = "#F7931A"
            for tr in mc_traces_list:
                if not (getattr(getattr(tr, "line", None), "dash", None) == "dot"):
                    continue
                x_data = list(tr.x) if tr.x is not None else []
                y_data = list(tr.y) if tr.y is not None else []
                if not x_data or not y_data:
                    continue
                mc_n = len(x_data)
                mc_idx = max(0, mc_n - max(3, mc_n // 20))
                final_y = float(y_data[-1])
                if final_y <= 0:
                    continue
                if disp_mode == "usd":
                    mc_lbl = fmt_price(final_y)
                else:
                    mc_lbl = f"{final_y:.4f} \u20bf"
                mc_x_last = float(x_data[-1])
                if mc_x_last < t_end:
                    ann_yr = int(syr + (mc_x_last - t_start_base)
                                 / max(t_end - t_start_base, 1e-6) * (eyr - syr))
                    mc_lbl = f"\u2248{ann_yr}  {mc_lbl}"
                traces.append(go.Scatter(
                    x=[float(x_data[mc_idx])], y=[float(y_data[mc_idx])],
                    mode="markers+text",
                    marker=dict(size=7, color=mc_col, symbol="circle"),
                    text=[f"  MC \u2192 {mc_lbl}"],
                    textposition="middle right",
                    textfont=dict(size=_FONT_ANNOT, color=mc_col),
                    showlegend=False, hoverinfo="skip",
                ))

        layout["shapes"]     = shapes
        layout["showlegend"] = show_legend
        fig = go.Figure(data=traces, layout=go.Layout(**layout))
        _apply_watermark(fig)
        return fig, mc_result

    # ── MODE B: fixed depletion date → max withdrawal per period ──────────────
    else:
        return _sc_mode_b(m, p, syr, delays, sel_qs, start_stack, ppy, dt,
                          inflation, chart_layout, display_q, show_legend, freq_label)


def _sc_mode_b(m, p, syr, delays, sel_qs, start_stack, ppy, dt,
               inflation, chart_layout, display_q, show_legend, freq_label):
    """HODL Supercharger Mode B: binary-search max withdrawal per period."""
    target_yr = int(p.get("target_yr", 2060))

    def _max_wd_for(d, q):
        t_start_d = max(yr_to_t(syr + d, m.genesis), 1.0)
        t_end_b   = yr_to_t(target_yr, m.genesis)
        if t_end_b <= t_start_d:
            return 0.0
        first_price = float(qr_price(q, max(t_start_d, 0.5), m.qr_fits))
        lo, hi = 0.0, start_stack * first_price * ppy * 4
        for _ in range(60):
            mid = (lo + hi) / 2.0
            s   = start_stack
            survived = True
            for t in np.arange(t_start_d, t_end_b + dt * 0.5, dt):
                adj = mid * ((1 + inflation) ** (t - t_start_d))
                s  -= adj / float(qr_price(q, max(t, 0.5), m.qr_fits))
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
            col   = m.qr_colors.get(q, "#888888")
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
    layout["showlegend"] = show_legend
    fig = go.Figure(data=traces, layout=go.Layout(**layout))
    _apply_watermark(fig)
    return fig, None
