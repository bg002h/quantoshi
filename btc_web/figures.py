"""figures.py — Plotly chart builders for the Bitcoin Projections web app.

Each function takes a ModelData instance and a params dict of control values
and returns a go.Figure ready for dcc.Graph.
"""

import math
import base64
import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# btc_app/ is added to sys.path by app.py before this import
import bisect
from btc_core import qr_price, yr_to_t, today_t, fmt_price, _fmt_btc, leo_weighted_entry, _find_lot_percentile
try:
    from markov import (build_transition_matrix, monte_carlo_prices,
                        mc_dca, mc_retire, compute_fan_percentiles, depletion_stats,
                        max_bins_for_window)
    _HAS_MARKOV = True
except ImportError:
    _HAS_MARKOV = False

try:
    from mc_cache import (load_caches as _load_mc_caches,
                          get_cached_paths, get_cached_overlay,
                          snap_to_bin, is_cached_year, FAN_PCTS as _MC_CACHE_FAN_PCTS)
    _load_mc_caches()
    _HAS_MC_CACHE = True
except ImportError:
    _HAS_MC_CACHE = False

# Server-side cache for transition matrices.
# Key: (n_bins, step_days, window_start_yr, window_end_yr) → (trans, bin_edges)
# Persisted to disk and lazily populated.
_TRANS_MATRIX_CACHE = {}
_TRANS_CACHE_PATH = Path(__file__).parent / ".trans_matrix_cache.pkl"
_TRANS_CACHE_DIRTY = False  # track whether cache has new entries to flush


def _load_trans_cache_from_disk():
    """Load cached transition matrices from disk if they match current model data."""
    global _TRANS_MATRIX_CACHE
    if not _TRANS_CACHE_PATH.exists():
        return
    try:
        import pickle
        with open(_TRANS_CACHE_PATH, "rb") as f:
            saved = pickle.load(f)
        # Invalidate if model_data.pkl has changed since cache was written
        pkl_path = Path(__file__).parent.parent / "btc_app" / "model_data.pkl"
        if pkl_path.exists():
            pkl_mtime = pkl_path.stat().st_mtime
            if saved.get("_pkl_mtime") != pkl_mtime:
                return  # stale cache
        _TRANS_MATRIX_CACHE = saved.get("matrices", {})
    except Exception:
        pass  # corrupt cache file — start fresh


def save_trans_cache_to_disk():
    """Flush transition matrix cache to disk for reuse across restarts."""
    global _TRANS_CACHE_DIRTY
    if not _TRANS_CACHE_DIRTY:
        return
    try:
        import pickle
        pkl_path = Path(__file__).parent.parent / "btc_app" / "model_data.pkl"
        pkl_mtime = pkl_path.stat().st_mtime if pkl_path.exists() else None
        with open(_TRANS_CACHE_PATH, "wb") as f:
            pickle.dump({"matrices": _TRANS_MATRIX_CACHE, "_pkl_mtime": pkl_mtime}, f)
        _TRANS_CACHE_DIRTY = False
    except Exception:
        pass


# Load from disk at import time
_load_trans_cache_from_disk()


def _get_transition_matrix(m, n_bins, step_days, mc_window):
    """Get transition matrix from cache or build on the fly."""
    global _TRANS_CACHE_DIRTY
    window_start_yr = None
    window_end_yr   = None
    ws_cal = we_cal = None
    if mc_window:
        ws_cal = int(mc_window[0])
        we_cal = int(mc_window[1])
        window_years = max(1, we_cal - ws_cal)
        n_bins = min(n_bins, max_bins_for_window(window_years, step_days))
        window_start_yr = yr_to_t(ws_cal, m.genesis)
        window_end_yr   = yr_to_t(we_cal, m.genesis)

    cache_key = (n_bins, step_days, ws_cal, we_cal)
    cached = _TRANS_MATRIX_CACHE.get(cache_key)
    if cached is not None:
        return cached[0], cached[1], n_bins

    trans, bin_edges, _ = build_transition_matrix(
        m.price_prices, m.price_years, m.qr_fits,
        n_bins=n_bins,
        window_start_yr=window_start_yr,
        window_end_yr=window_end_yr,
        step_days=step_days,
    )
    _TRANS_MATRIX_CACHE[cache_key] = (trans, bin_edges)
    _TRANS_CACHE_DIRTY = True
    return trans, bin_edges, n_bins


def _interp_qr_price(q, t, qr_fits):
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

# ── shared theme helpers ──────────────────────────────────────────────────────

def _dark_layout(m, title, xlabel, ylabel, **kwargs):
    """Base dark-theme layout dict."""
    return dict(
        title=dict(text=title, font=dict(color=m.TITLE_COLOR, size=14)),
        paper_bgcolor=m.PLOT_BG_COLOR,
        plot_bgcolor=m.PLOT_BG_COLOR,
        font=dict(color=m.TEXT_COLOR, size=11),
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
            borderwidth=1, font=dict(size=10),
        ),
        margin=dict(l=60, r=20, t=50, b=60),
        **kwargs,
    )


def _year_ticks(start_yr, end_yr, genesis):
    """Return (tick_t_values, tick_year_labels) for a year-based x-axis."""
    span = end_yr - start_yr
    step = 1 if span <= 15 else (2 if span <= 30 else 5)
    yrs  = list(range(start_yr, end_yr + 1, step))
    ts   = [yr_to_t(y, genesis) for y in yrs]
    return ts, [str(y) for y in yrs]


# ── Watermark (logo + URL) ────────────────────────────────────────────────────────────────────────────

_LOGO_B64 = None
try:
    _logo_path = Path(__file__).parent / "assets" / "quantoshi_logo_wm.png"
    with open(_logo_path, "rb") as _f:
        _LOGO_B64 = "data:image/png;base64," + base64.b64encode(_f.read()).decode()
except Exception:
    pass


def _apply_watermark(fig):
    """Stamp Quantoshi logo + URL onto a go.Figure (bottom-right corner)."""
    if _LOGO_B64:
        fig.add_layout_image(dict(
            source=_LOGO_B64,
            xref="paper", yref="paper",
            x=1.0, y=0.0,
            sizex=0.07, sizey=0.12,
            xanchor="right", yanchor="bottom",
            opacity=0.55,
            layer="above",
        ))
    fig.add_annotation(dict(
        text="quantoshi.xyz",
        xref="paper", yref="paper",
        x=0.925, y=0.015,
        xanchor="right", yanchor="bottom",
        showarrow=False,
        font=dict(size=9, color="rgba(180,180,180,0.65)"),
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

def build_bubble_figure(m, p):
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
    t_arr = np.linspace(max(t_lo, 0.1), t_hi, 1500)

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
        pct = q * 100
        lbl = f"Q{pct:.4g}%" if pct >= 1 else f"Q{pct:.3g}%"
        if stack > 0:
            lbl += f"  \u2192  {fmt_price(float(prices[-1]))}"
        col = m.qr_colors.get(q, "#888888")
        traces.append(go.Scatter(
            x=list(t_arr), y=list(prices),
            mode="lines", name=lbl,
            line=dict(color=col, width=1.8),
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
        _MAX_PTS = 1200
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
                line=dict(color="#FF6600", dash="dash", width=1.5),
                opacity=0.85, yref="y",
            ))

    # ── x-axis ticks (calendar years) ─────────────────────────────────────────
    tick_ts, tick_lbls = _year_ticks(p["xmin"], p["xmax"], m.genesis)
    tick_ts  = [t for t in tick_ts if t_lo <= t <= t_hi]
    tick_lbls = tick_lbls[:len(tick_ts)]

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
        layout["yaxis"].update(
            type="log",
            range=[math.log10(max(y_lo, 1e-10)), math.log10(max(y_hi, 1e-10))],
            tickvals=maj, ticktext=[_fmt_y(v) for v in maj],
        )
    else:
        layout["yaxis"].update(range=[y_lo, y_hi])

    if p.get("xscale", "linear") == "log":
        layout["xaxis"].update(
            type="log",
            range=[math.log10(max(t_lo, 1e-10)), math.log10(max(t_hi, 1e-10))],
        )

    layout["showlegend"] = bool(p.get("show_legend", True))
    layout["shapes"] = shapes

    if stack > 0:
        layout["annotations"] = [dict(
            text=f"Stack: {p['stack']:.6g} BTC",
            xref="paper", yref="paper", x=0.99, y=0.01,
            xanchor="right", yanchor="bottom",
            showarrow=False, font=dict(size=10, color=m.TEXT_COLOR),
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


def build_heatmap_figure(m, p):
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
        fig = go.Figure()
        fig.update_layout(
            title="No data — adjust Entry / Exit settings",
            paper_bgcolor=m.PLOT_BG_COLOR, plot_bgcolor=m.PLOT_BG_COLOR,
            font=dict(color=m.TEXT_COLOR),
        )
        return fig

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
        pct = q * 100
        ylabels.append(f"Q{pct:.4g}%" if pct >= 1 else f"Q{pct:.3g}%")

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
                   font=dict(color=m.TITLE_COLOR, size=13)),
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


def build_mc_heatmap_figure(m, p):
    """Build a standalone MC heatmap figure from MC-simulated CAGR percentiles.
    Returns (fig, mc_result) or (empty_fig, None).
    """
    eyr = int(p.get("entry_yr", 2020))
    eq  = float(p.get("entry_q", 50)) / 100.0
    entry_t = yr_to_t(eyr, m.genesis)
    live_price = p.get("live_price")
    ep  = float(live_price) if live_price else _interp_qr_price(eq, entry_t, m.qr_fits)

    lots = p.get("lots") or []
    if p.get("use_lots") and lots:
        result = leo_weighted_entry(lots)
        if result:
            ep, entry_t, _pct, _tw = result

    mc_years = int(p.get("mc_years", 10))
    xlo = max(int(p.get("exit_yr_lo", eyr)), eyr)
    xhi = min(int(p.get("exit_yr_hi", eyr + mc_years)), eyr + mc_years)
    eyrs = list(range(xlo, xhi + 1))

    if not eyrs:
        fig = go.Figure()
        fig.update_layout(
            title="No data — adjust Entry / Exit settings",
            paper_bgcolor=m.PLOT_BG_COLOR, plot_bgcolor=m.PLOT_BG_COLOR,
            font=dict(color=m.TEXT_COLOR),
        )
        return fig, None

    mc_data = _mc_heatmap_overlay(m, p, ep, entry_t, eyrs)
    mc_cagr, mc_prices, mc_mults, mc_labels, mc_result = mc_data
    if mc_cagr is None:
        fig = go.Figure()
        fig.update_layout(
            title="MC simulation error",
            paper_bgcolor=m.PLOT_BG_COLOR, plot_bgcolor=m.PLOT_BG_COLOR,
            font=dict(color=m.TEXT_COLOR),
        )
        return fig, None

    mc = mc_cagr
    mp = mc_prices
    mm = mc_mults

    valid = mc[~np.isnan(mc)]
    if len(valid) == 0:
        fig = go.Figure()
        fig.update_layout(
            title="MC: no valid data in range",
            paper_bgcolor=m.PLOT_BG_COLOR, plot_bgcolor=m.PLOT_BG_COLOR,
            font=dict(color=m.TEXT_COLOR),
        )
        return fig, mc_result

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
                   font=dict(color=m.TITLE_COLOR, size=13)),
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
    _apply_watermark(fig)
    return fig, mc_result


# ── DCA Accumulator ───────────────────────────────────────────────────────────

def build_dca_figure(m, p):
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
        return go.Figure(layout=dict(
            title="Set end year > start year",
            paper_bgcolor=m.PLOT_BG_COLOR, font=dict(color=m.TEXT_COLOR))), None

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
    for q in sel_qs:
        if q not in m.qr_fits:
            continue
        stack    = start_stack
        vals     = np.empty(len(ts))
        prices_q = np.empty(len(ts))
        for i, t in enumerate(ts):
            price      = float(qr_price(q, max(t, 0.5), m.qr_fits))
            stack     += amount / price
            vals[i]    = stack
            prices_q[i] = price
        all_btc_vals[q] = vals.copy()
        all_usd_vals[q]  = vals * prices_q
        all_prices[q]    = prices_q          # save for SC loop below

        if disp_mode == "usd":
            y_vals    = vals * prices_q
            final_lbl = fmt_price(float(y_vals[-1]))
        else:
            y_vals    = vals
            final_usd = fmt_price(float(all_usd_vals[q][-1]))
            final_lbl = f"{float(vals[-1]):.4f} BTC  ({final_usd})"

        pct = q * 100
        lbl = (f"Q{pct:.4g}%" if pct >= 1 else f"Q{pct:.3g}%") + f"  →  {final_lbl}"
        col = m.qr_colors.get(q, "#888888")
        traces.append(go.Scatter(
            x=list(ts), y=list(y_vals), mode="lines", name=lbl,
            line=dict(color=col, width=1.8),
        ))

    shapes = []
    if p.get("show_today"):
        td = today_t(m.genesis)
        if t_start <= td <= t_end:
            shapes.append(dict(
                type="line", x0=td, x1=td, y0=0, y1=1,
                yref="paper", line=dict(color="#FF6600", dash="dash", width=1.5),
                opacity=0.85,
            ))

    tick_ts, tick_lbls = _year_ticks(syr, eyr, m.genesis)

    # ── Total cost & value ratio ────────────────────────────────────────────
    n_periods = len(ts)
    total_spent = amount * n_periods
    freq_short = freq_str.lower()[:-2] if freq_str.endswith("ly") else freq_str

    # Build title — add cost info and median ROI if we have quantile data
    title_line = f"Bitcoin DCA — {fmt_price(amount)}/{freq_short}"
    title_line += f"  ·  {fmt_price(total_spent)} invested over {n_periods} periods"
    if all_usd_vals:
        med_final_usd = float(np.median([v[-1] for v in all_usd_vals.values()]))
        roi = med_final_usd / total_spent if total_spent > 0 else 0
        title_line += f"<br>Median final value {fmt_price(med_final_usd)}  ·  {roi:.1f}× return on capital"

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
    layout["shapes"] = shapes

    # ── Stack-cellerator overlay ─────────────────────────────────────────────
    all_sc_usd_vals = {}  # q -> SC USD value array
    all_sc_btc_vals = {}  # q -> SC BTC value array (for right-edge annotations)
    if p.get("sc_enabled") and sel_qs:
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

        term_periods = max(1, round(term_months * ppy / 12))
        n_cycles     = 1 + sc_repeats
        r            = sc_rate / 100.0 / ppy

        # Cap principal so payment never exceeds DCA amount
        if r > 0:
            if loan_type == "amortizing":
                max_principal = amount * (1 - (1 + r) ** (-term_periods)) / r
            else:
                max_principal = amount / r
            principal = min(principal, max_principal)

        if loan_type == "amortizing":
            if r > 0:
                pmt = principal * r / (1 - (1 + r) ** (-term_periods))
            else:
                pmt = principal / term_periods
        else:  # interest_only
            pmt = principal * r

        sc_dca_amt = amount - pmt  # guaranteed >= 0 after cap

        if principal > 0:
            for q in sel_qs:
                if q not in m.qr_fits:
                    continue
                sc_stack    = start_stack
                outstanding = 0.0
                sc_vals     = np.empty(len(ts))
                sc_prices   = np.empty(len(ts))
                _prices_q   = all_prices[q]  # precomputed — same as qr_price(q, t, ...) for each t
                for i, t in enumerate(ts):
                    price           = _prices_q[i]
                    cycle_idx       = i // term_periods
                    period_in_cycle = i % term_periods

                    if cycle_idx < n_cycles:
                        if period_in_cycle == 0:        # start of new cycle
                            if cycle_idx == 0:          # first cycle — use entry mode
                                if entry_mode == "live" and sc_live > 0:
                                    ep = sc_live
                                elif entry_mode == "custom" and custom_price > 0:
                                    ep = custom_price
                                else:
                                    ep = price          # model price
                                sc_stack    += principal / ep  # buy BTC with loan proceeds
                            elif not sc_rollover:
                                ep = price              # repeat cycle, no rollover: new independent loan
                                sc_stack    += principal / ep
                            # rollover repeat: new loan pays off old — no net BTC movement
                            outstanding  = principal

                        sc_stack += sc_dca_amt / price  # DCA minus loan payment

                        if loan_type == "amortizing":   # track amortizing balance
                            interest_p  = outstanding * r
                            principal_p = pmt - interest_p
                            outstanding = max(outstanding - principal_p, 0.0)

                        if loan_type == "interest_only" and period_in_cycle == term_periods - 1:
                            if sc_rollover:
                                pass                    # rollover: no BTC sold; post-loop handles final repayment
                            else:
                                # Sell BTC to repay principal.  Tax applies only to the gain
                                # (sell price − buy price); if selling at a loss there is no tax.
                                gain_per_btc = max(price - ep, 0.0)
                                net_per_btc  = price - tax_rate * gain_per_btc
                                sc_stack    -= principal / net_per_btc
                                sc_stack     = max(sc_stack, 0.0)
                                outstanding  = 0.0
                    else:                               # cycles exhausted → plain DCA
                        sc_stack += amount / price

                    sc_vals[i]   = sc_stack
                    sc_prices[i] = price

                # Deduct any outstanding balance at simulation end
                # (incomplete final cycle or rollover — sell BTC at final price to repay,
                #  tax on gain only: sell_price − buy_price, zero if selling at a loss)
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

                pct = q * 100
                lbl_sc = (f"SC Q{pct:.4g}%" if pct >= 1 else f"SC Q{pct:.3g}%") + f"  \u2192  {final_sc}"
                col = m.qr_colors.get(q, "#888888")
                traces.append(go.Scatter(
                    x=list(ts), y=list(y_sc), mode="lines", name=lbl_sc,
                    line=dict(color=col, width=1.8, dash="dash"),
                ))

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
                _edge_items.append((mc_x, mc_med_y, final_usd, "#f7931a", "MC "))

    # ── Render staggered right-edge annotations ─────────────────────────────
    if _edge_items:
        # Sort by final_y so we can stagger close labels
        _edge_items.sort(key=lambda it: it[1])
        _AY_STEP = 18  # px between staggered labels
        _X_POSITIONS = [0.97, 0.87, 0.77]  # 3 horizontal positions (paper coords)
        usd_annots = []
        n = len(_edge_items)
        for idx, (ax_x, fy, fusd, col, pfx) in enumerate(_edge_items):
            # Stagger: spread labels vertically, base offset -35px (above trace)
            ay = -35 - int((n - 1 - idx) * _AY_STEP)
            xp = _X_POSITIONS[idx % len(_X_POSITIONS)]
            # Arrow points straight down (ax=0), text above trace
            usd_annots.append(dict(
                x=xp, xref="paper",
                y=fy, yref="y",
                text=f"<b>{pfx}{fmt_price(fusd)}</b>",
                showarrow=True, arrowhead=2, arrowsize=1,
                arrowcolor=col, arrowwidth=1.5,
                ax=0, ay=ay,
                font=dict(size=11, color=col),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor=col, borderwidth=1, borderpad=2,
            ))
        layout.setdefault("annotations", []).extend(usd_annots)

    layout["showlegend"] = bool(p.get("show_legend", True))
    fig = go.Figure(data=traces, layout=go.Layout(**layout))
    _apply_watermark(fig)
    return fig, mc_result


_FREQ_STEP_DAYS = {"Daily": 1, "Weekly": 7, "Monthly": 30, "Quarterly": 91, "Annually": 365}
FREQ_PPY = {"Daily": 365, "Weekly": 52, "Monthly": 12, "Quarterly": 4, "Annually": 1}

_MC_FAN_PCTS = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95)

def _mc_path_key(p, tab):
    """Build dict of params that determine price paths (expensive MC sampling).

    If these match, cached price_paths can be reused — only the overlay
    (DCA amount / withdrawal / inflation / start_stack) needs recomputing.
    """
    key = {
        "tab": tab,
        "mc_bins": int(p.get("mc_bins", 5)),
        "mc_sims": int(p.get("mc_sims", 800)),
        "mc_years": int(p.get("mc_years", 10)),
        "mc_freq": p.get("mc_freq", "Monthly"),
        "mc_window": p.get("mc_window"),
        "start_yr": int(p.get("start_yr", 2024)),
    }
    # entry_q determines the starting bin (affects paths)
    if tab == "hm":
        key["entry_q"] = float(p.get("entry_q", 50))
    elif tab == "dca":
        key["mc_entry_q"] = float(p.get("mc_entry_q", 50))
        key["mc_start_yr"] = int(p.get("mc_start_yr", 2026))
    elif tab in ("ret", "sc"):
        key["mc_entry_q"] = float(p.get("mc_entry_q", 50))
        key["mc_start_yr"] = int(p.get("mc_start_yr", 2031))
    return key


def _mc_overlay_key(p, tab, start_stack):
    """Build dict of overlay-specific params (cheap to recompute from paths)."""
    key = {
        "mc_amount": float(p.get("mc_amount", 100)),
        "start_stack": float(start_stack),
    }
    if tab in ("ret", "sc"):
        key["mc_infl"] = float(p.get("mc_infl", 0))
    return key


def _mc_fan_to_lists(fan):
    """Convert fan dict {pct: ndarray} to JSON-serializable {str: list}."""
    return {str(k): [round(float(v), 4) for v in arr] for k, arr in fan.items()}


def _mc_fan_from_lists(d):
    """Restore fan dict from JSON-serialized form."""
    return {float(k): np.array(v) for k, v in d.items()}


def _mc_paths_to_lists(paths):
    """Convert price_paths ndarray (n_sims, n_steps) to compact JSON list (float32)."""
    return np.asarray(paths, dtype=np.float32).tolist()


def _mc_paths_from_lists(lst):
    """Restore price_paths ndarray from JSON-serialized list."""
    return np.array(lst, dtype=np.float32)


def _mc_build_traces(mc_ts, fan, extra_label="", show_median=True,
                     show_final_values=False, fan_usd=None):
    """Build standard MC fan band traces from precomputed fan percentiles.

    fan_usd: if provided, use these values for legend final-value labels
             (always show USD regardless of display mode).
    """
    lf = fan_usd if fan_usd is not None else fan  # legend fan

    traces = []
    _MC_BANDS = [
        (0.01, 0.95, "rgba(247,147,26,0.04)", "MC 1\u201395%"),
        (0.05, 0.95, "rgba(247,147,26,0.08)", "MC 5\u201395%"),
        (0.25, 0.75, "rgba(247,147,26,0.15)", "MC 25\u201375%"),
    ]
    for p_lo, p_hi, fill_color, label in _MC_BANDS:
        if show_final_values:
            lo_final = fmt_price(float(lf[p_lo][-1])) if len(lf[p_lo]) > 0 else ""
            hi_final = fmt_price(float(lf[p_hi][-1])) if len(lf[p_hi]) > 0 else ""
            label = f"{label}  ({lo_final} \u2013 {hi_final})"
        traces.append(go.Scatter(
            x=list(mc_ts), y=list(fan[p_hi]),
            mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))
        traces.append(go.Scatter(
            x=list(mc_ts), y=list(fan[p_lo]),
            mode="lines", line=dict(width=0), fill="tonexty",
            fillcolor=fill_color, name=label, showlegend=True, hoverinfo="skip",
        ))
    if show_median:
        med_label = "MC median" + extra_label
        if show_final_values and 0.50 in lf and len(lf[0.50]) > 0:
            med_label += f"  \u2192  {fmt_price(float(lf[0.50][-1]))}"
        traces.append(go.Scatter(
            x=list(mc_ts), y=list(fan[0.50]),
            mode="lines", name=med_label,
            line=dict(color="rgba(247,147,26,0.7)", width=1.5, dash="dot"),
        ))
    return traces


def _mc_depletion_annots(mc_ts, fan, mc_start_yr, mc_years, existing_count=0):
    """Detect depletion on MC fan percentiles and return annotations.

    Matches the QR depletion annotation style (arrow to y=0 on paper coords).
    Annotates median, P25, P75 if they deplete within the horizon.
    """
    annots = []
    mc_col = "#f7931a"  # BTC orange
    for pct in [0.01, 0.50, 0.95]:
        vals = fan.get(pct)
        if vals is None:
            continue
        depl_i = next((i for i, v in enumerate(vals) if v <= 0.0001), None)
        if depl_i is not None:
            depl_t = mc_ts[depl_i]
            t0 = mc_ts[0]
            t_span = mc_ts[-1] - t0 if mc_ts[-1] > t0 else 1.0
            depl_yr = int(mc_start_yr + (depl_t - t0) / t_span * mc_years)
            _ay = [-20, -33, -46][(existing_count + len(annots)) % 3]
            annots.append(dict(
                x=depl_t, xref="x",
                y=0, yref="paper",
                ax=28, ay=_ay,
                text=f"\u2248{depl_yr}",
                showarrow=True, arrowhead=2, arrowsize=1,
                arrowcolor=mc_col,
                font=dict(size=10, color=mc_col),
            ))
    return annots


def _mc_dca_overlay(m, p, ts, t_start, dt, start_stack, disp_mode):
    """Build Monte Carlo fan band traces for DCA overlay.

    Returns (traces, result_dict).  result_dict is JSON-serializable and
    contains price_paths + fan bands so overlays can be recomputed cheaply.

    Cache logic:
    - If path_key matches cached data, reuse price_paths (skip expensive MC).
    - If overlay_key also matches, reuse fan bands directly (no recompute).
    - Otherwise run full simulation.
    """
    amount     = float(p.get("mc_amount", 100))
    n_bins     = int(p.get("mc_bins", 5))
    n_sims    = int(p.get("mc_sims", 800))
    mc_window = p.get("mc_window")
    mc_freq   = p.get("mc_freq", "Monthly")
    mc_ppy    = FREQ_PPY.get(mc_freq, 12)
    mc_dt     = 1.0 / mc_ppy
    step_days = _FREQ_STEP_DAYS.get(mc_freq, 30)

    mc_years = int(p.get("mc_years", 10))
    mc_start_yr = int(p.get("mc_start_yr", 0))
    mc_t_start = yr_to_t(mc_start_yr, m.genesis) if mc_start_yr else t_start
    mc_t_end = mc_t_start + mc_years
    mc_ts    = np.arange(mc_t_start, mc_t_end + mc_dt * 0.5, mc_dt)

    # Clip MC fan to DCA year range so off-screen points don't distort y-axis
    dca_t_end = ts[-1] if len(ts) > 0 else mc_t_end
    clip_n = int(np.searchsorted(mc_ts, dca_t_end + mc_dt * 0.5))
    clip_n = max(clip_n, 1)

    def _clip(mc_ts_full, fan_full):
        ct = mc_ts_full[:clip_n]
        cf = {k: v[:clip_n] for k, v in fan_full.items()}
        return ct, cf

    path_key    = _mc_path_key(p, "dca")
    overlay_key = _mc_overlay_key(p, "dca", start_stack)

    # ── Check cache ──────────────────────────────────────────────────────
    cached = p.get("mc_cached")
    if cached and cached.get("path_key") == path_key:
        if cached.get("overlay_key") == overlay_key:
            # Full cache hit — reuse fan bands as-is
            fan_btc = _mc_fan_from_lists(cached["fan_btc"])
            fan_usd = _mc_fan_from_lists(cached["fan_usd"])
            fan = fan_usd if disp_mode == "usd" else fan_btc
            ct, cf = _clip(mc_ts, fan)
            _, cf_usd = _clip(mc_ts, fan_usd)
            return _mc_build_traces(ct, cf, show_final_values=True, fan_usd=cf_usd), None, cf_usd

        # Path hit, overlay miss — recompute DCA from cached price paths
        price_paths = _mc_paths_from_lists(cached["price_paths"])
        btc_paths, usd_paths = mc_dca(price_paths, amount, start_stack)
        fan_btc = compute_fan_percentiles(btc_paths, _MC_FAN_PCTS)
        fan_usd = compute_fan_percentiles(usd_paths, _MC_FAN_PCTS)
        fan = fan_usd if disp_mode == "usd" else fan_btc

        result = {
            "tab": "dca",
            "path_key": path_key,
            "overlay_key": overlay_key,
            "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "ts": cached["ts"],
            "price_paths": cached["price_paths"],
            "fan_btc": _mc_fan_to_lists(fan_btc),
            "fan_usd": _mc_fan_to_lists(fan_usd),
        }
        ct, cf = _clip(mc_ts, fan)
        _, cf_usd = _clip(mc_ts, fan_usd)
        return _mc_build_traces(ct, cf, show_final_values=True, fan_usd=cf_usd), result, cf_usd

    # ── Check pre-computed path cache ────────────────────────────────────
    if _HAS_MC_CACHE:
        _syr = int(p.get("mc_start_yr", p.get("start_yr", 2024)))
        raw_pctile = float(p.get("mc_entry_q", 50)) / 100.0
        pct_bin = snap_to_bin(raw_pctile)
        cached_paths = get_cached_paths(_syr, pct_bin, mc_years)
        if cached_paths is not None:
            btc_paths, usd_paths = mc_dca(cached_paths, amount, start_stack)
            fan_btc = compute_fan_percentiles(btc_paths, _MC_FAN_PCTS)
            fan_usd = compute_fan_percentiles(usd_paths, _MC_FAN_PCTS)
            fan = fan_usd if disp_mode == "usd" else fan_btc
            ct, cf = _clip(mc_ts, fan)
            _, cf_usd = _clip(mc_ts, fan_usd)
            return _mc_build_traces(ct, cf, show_final_values=True, fan_usd=cf_usd), None, cf_usd

    # ── Run full simulation ──────────────────────────────────────────────
    trans, bin_edges, n_bins = _get_transition_matrix(m, n_bins, step_days, mc_window)
    n_steps = len(mc_ts)

    # Start percentile from mc_entry_q param (user-selected, cache-aligned 10% steps)
    start_pctile = float(p.get("mc_entry_q", 50)) / 100.0
    start_pctile = max(0.05, min(start_pctile, 0.95))

    price_paths, _ = monte_carlo_prices(
        trans, bin_edges, start_pctile, n_steps, n_sims,
        m.qr_fits, m.genesis, mc_t_start, mc_dt,
    )
    btc_paths, usd_paths = mc_dca(price_paths, amount, start_stack)

    fan_btc = compute_fan_percentiles(btc_paths, _MC_FAN_PCTS)
    fan_usd = compute_fan_percentiles(usd_paths, _MC_FAN_PCTS)
    fan = fan_usd if disp_mode == "usd" else fan_btc

    result = {
        "tab": "dca",
        "path_key": path_key,
        "overlay_key": overlay_key,
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "ts": [round(float(t), 6) for t in mc_ts],
        "price_paths": _mc_paths_to_lists(price_paths),
        "fan_btc": _mc_fan_to_lists(fan_btc),
        "fan_usd": _mc_fan_to_lists(fan_usd),
    }

    ct, cf = _clip(mc_ts, fan)
    _, cf_usd = _clip(mc_ts, fan_usd)
    return _mc_build_traces(ct, cf, show_final_values=True, fan_usd=cf_usd), result, cf_usd


# ── BTC Retireator ────────────────────────────────────────────────────────────

def build_retire_figure(m, p):
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
        return go.Figure(layout=dict(
            title="Set end year > start year",
            paper_bgcolor=m.PLOT_BG_COLOR, font=dict(color=m.TEXT_COLOR))), None

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

    for q in sel_qs:
        if q not in m.qr_fits:
            continue
        stack = start_stack
        vals  = np.empty(len(ts))
        for i, t in enumerate(ts):
            years_elapsed = t - t_start
            adj_wd = wd_amount * ((1 + inflation) ** years_elapsed)
            price  = float(qr_price(q, max(t, 0.5), m.qr_fits))
            stack -= adj_wd / price
            stack  = max(stack, 0.0)
            vals[i] = stack
        all_btc_vals[q] = vals.copy()

        if disp_mode == "usd":
            prices_arr = np.array([float(qr_price(q, max(t, 0.5), m.qr_fits)) for t in ts])
            y_vals = vals * prices_arr
            final_lbl = fmt_price(float(y_vals[-1]))
        else:
            y_vals = vals
            final_lbl = f"{float(vals[-1]):.4f} BTC"

        pct = q * 100
        lbl = (f"Q{pct:.4g}%" if pct >= 1 else f"Q{pct:.3g}%") + f"  →  {final_lbl}"
        col = m.qr_colors.get(q, "#888888")
        traces.append(go.Scatter(
            x=list(ts), y=list(y_vals), mode="lines", name=lbl,
            line=dict(color=col, width=1.8),
        ))

        # depletion annotation
        if p.get("annotate"):
            depl_i = next((i for i, v in enumerate(vals) if v <= 0), None)
            if depl_i is not None:
                depl_t = ts[depl_i]
                depl_yr = int(syr + (depl_t - t_start) * (eyr - syr) / max(t_end - t_start, 1e-6))
                _ay = [-20, -33, -46][len(deplete_annots) % 3]
                deplete_annots.append(dict(
                    x=depl_t, xref="x",
                    y=0, yref="paper",
                    ax=28, ay=_ay,
                    text=f"≈{depl_yr}",
                    showarrow=True, arrowhead=2, arrowsize=1,
                    arrowcolor=col,
                    font=dict(size=11, color=col),
                ))

    shapes = []
    if p.get("show_today"):
        td = today_t(m.genesis)
        if t_start <= td <= t_end:
            shapes.append(dict(
                type="line", x0=td, x1=td, y0=0, y1=1,
                yref="paper", line=dict(color="#FF6600", dash="dash", width=1.5),
                opacity=0.85,
            ))

    tick_ts, tick_lbls = _year_ticks(syr, eyr, m.genesis)
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
    layout["shapes"]      = shapes
    layout["annotations"] = deplete_annots

    # ── dual Y-axis ───────────────────────────────────────────────────────────
    if p.get("dual_y") and traces and all_btc_vals:
        # median across all selected quantiles (not just the middle one)
        stacked_btc = np.array(list(all_btc_vals.values()))
        btc_med = np.median(stacked_btc, axis=0)
        if disp_mode == "usd":
            y2_vals = btc_med
            y2_lbl  = "BTC Remaining"
        else:
            all_usd = np.array([
                all_btc_vals[q] * np.array([float(qr_price(q, max(t, 0.5), m.qr_fits))
                                             for t in ts])
                for q in all_btc_vals
            ])
            y2_vals = np.median(all_usd, axis=0)
            y2_lbl  = "USD Value"
        traces.append(go.Scatter(
            x=list(ts), y=list(y2_vals),
            mode="lines", name=f"{y2_lbl} (median)",
            line=dict(color="#aaaaaa", dash="dot", width=1),
            yaxis="y2", showlegend=True,
        ))
        layout["yaxis2"] = dict(
            title=dict(text=y2_lbl, font=dict(color=m.TEXT_COLOR)),
            overlaying="y", side="right",
            gridcolor=m.GRID_MAJOR_COLOR, linecolor=m.SPINE_COLOR,
            tickcolor=m.TEXT_COLOR,
        )
        if p.get("log_y"):
            layout["yaxis2"]["type"] = "log"

    # ── Monte Carlo fan overlay ─────────────────────────────────────────────
    mc_result = None
    if _HAS_MARKOV and p.get("mc_enabled"):
        mc_traces, mc_annots, mc_result = _mc_retire_overlay(m, p, ts, t_start, t_end, dt,
                                                              start_stack, disp_mode,
                                                              len(deplete_annots))
        traces.extend(mc_traces)
        if mc_annots:
            deplete_annots.extend(mc_annots)
            layout["annotations"] = deplete_annots

    # Sort all depletion annotations by x and reassign stagger levels
    if len(deplete_annots) > 1:
        deplete_annots.sort(key=lambda a: a["x"])
        _AY = [-20, -33, -46]
        for i, a in enumerate(deplete_annots):
            a["ay"] = _AY[i % 3]
        layout["annotations"] = deplete_annots

    layout["showlegend"] = bool(p.get("show_legend", True))
    fig = go.Figure(data=traces, layout=go.Layout(**layout))
    _apply_watermark(fig)
    return fig, mc_result


def _mc_withdraw_overlay(m, p, ts, t_start, t_end, dt,
                          start_stack, disp_mode, tab,
                          existing_annot_count=0):
    """Build Monte Carlo fan band traces for withdrawal-based overlays (Retire/SC).

    Args:
        tab: "ret" or "sc" — determines cache keys and result metadata.

    Returns (traces, annots, result) — annots is a list of depletion
    annotation dicts (empty if annotate is off or no depletion detected).
    """
    wd_amount  = float(p.get("mc_amount", 5000))
    inflation  = float(p.get("mc_infl", 4)) / 100.0
    n_bins     = int(p.get("mc_bins", 5))
    n_sims     = int(p.get("mc_sims", 800))
    mc_window  = p.get("mc_window")
    mc_freq    = p.get("mc_freq", "Monthly")
    mc_ppy     = FREQ_PPY.get(mc_freq, 12)
    mc_dt      = 1.0 / mc_ppy
    step_days  = _FREQ_STEP_DAYS.get(mc_freq, 30)

    mc_years = int(p.get("mc_years", 10))
    mc_start_yr = int(p.get("mc_start_yr", p.get("start_yr", 2031)))
    mc_t_end = t_start + mc_years
    mc_ts    = np.arange(t_start, mc_t_end + mc_dt * 0.5, mc_dt)

    path_key    = _mc_path_key(p, tab)
    overlay_key = _mc_overlay_key(p, tab, start_stack)
    do_annot    = bool(p.get("annotate"))

    def _depl_extra(dstats):
        if dstats.get("pct_depleted", 0) > 0:
            return f"  ({dstats['pct_depleted']:.0%} depleted)"
        return ""

    def _build_return(fan_btc, fan, extra, result=None):
        traces = _mc_build_traces(mc_ts, fan, extra)
        annots = _mc_depletion_annots(mc_ts, fan_btc, mc_start_yr, mc_years,
                                       existing_annot_count) if do_annot else []
        return traces, annots, result

    # ── Check cache ──────────────────────────────────────────────────────
    cached = p.get("mc_cached")
    if cached and cached.get("path_key") == path_key:
        if cached.get("overlay_key") == overlay_key:
            fan_btc = _mc_fan_from_lists(cached["fan_btc"])
            fan_usd = _mc_fan_from_lists(cached["fan_usd"])
            fan = fan_usd if disp_mode == "usd" else fan_btc
            return _build_return(fan_btc, fan, _depl_extra(cached.get("depletion", {})))

        price_paths = _mc_paths_from_lists(cached["price_paths"])
        btc_paths, usd_paths, depl_steps = mc_retire(
            price_paths, start_stack, wd_amount, inflation, mc_dt,
        )
        fan_btc = compute_fan_percentiles(btc_paths, _MC_FAN_PCTS)
        fan_usd = compute_fan_percentiles(usd_paths, _MC_FAN_PCTS)
        dstats = depletion_stats(depl_steps, len(mc_ts), mc_dt, t_start)
        fan = fan_usd if disp_mode == "usd" else fan_btc

        result = {
            "tab": tab,
            "path_key": path_key,
            "overlay_key": overlay_key,
            "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "ts": cached["ts"],
            "price_paths": cached["price_paths"],
            "fan_btc": _mc_fan_to_lists(fan_btc),
            "fan_usd": _mc_fan_to_lists(fan_usd),
            "depletion": dstats,
        }
        return _build_return(fan_btc, fan, _depl_extra(dstats), result)

    # ── Check pre-computed cache ─────────────────────────────────────────
    mc_stack = float(p.get("mc_start_stack", start_stack))
    if _HAS_MC_CACHE:
        raw_pctile = float(p.get("mc_entry_q", 50)) / 100.0
        pct_bin = snap_to_bin(raw_pctile)
        infl_pct = int(round(inflation * 100))

        fan_btc, fan_usd = get_cached_overlay(
            mc_start_yr, pct_bin, mc_years,
            int(wd_amount), infl_pct, mc_stack)
        if fan_btc is not None:
            fan = fan_usd if disp_mode == "usd" else fan_btc
            return _build_return(fan_btc, fan, "")

        cached_paths = get_cached_paths(mc_start_yr, pct_bin, mc_years)
        if cached_paths is not None:
            btc_paths, usd_paths, depl_steps = mc_retire(
                cached_paths, mc_stack, wd_amount, inflation, mc_dt,
            )
            fan_btc = compute_fan_percentiles(btc_paths, _MC_FAN_PCTS)
            fan_usd = compute_fan_percentiles(usd_paths, _MC_FAN_PCTS)
            dstats = depletion_stats(depl_steps, len(mc_ts), mc_dt, t_start)
            fan = fan_usd if disp_mode == "usd" else fan_btc
            return _build_return(fan_btc, fan, _depl_extra(dstats))

    # ── Run full simulation ──────────────────────────────────────────────
    trans, bin_edges, n_bins = _get_transition_matrix(m, n_bins, step_days, mc_window)
    n_steps = len(mc_ts)

    start_pctile = float(p.get("mc_entry_q", 50)) / 100.0
    start_pctile = round(start_pctile * 20) / 20
    start_pctile = max(0.05, min(start_pctile, 0.95))

    price_paths, _ = monte_carlo_prices(
        trans, bin_edges, start_pctile, n_steps, n_sims,
        m.qr_fits, m.genesis, t_start, mc_dt,
    )
    btc_paths, usd_paths, depl_steps = mc_retire(
        price_paths, mc_stack, wd_amount, inflation, mc_dt,
    )

    fan_btc = compute_fan_percentiles(btc_paths, _MC_FAN_PCTS)
    fan_usd = compute_fan_percentiles(usd_paths, _MC_FAN_PCTS)
    dstats = depletion_stats(depl_steps, n_steps, mc_dt, t_start)
    fan = fan_usd if disp_mode == "usd" else fan_btc

    result = {
        "tab": tab,
        "path_key": path_key,
        "overlay_key": overlay_key,
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "ts": [round(float(t), 6) for t in mc_ts],
        "price_paths": _mc_paths_to_lists(price_paths),
        "fan_btc": _mc_fan_to_lists(fan_btc),
        "fan_usd": _mc_fan_to_lists(fan_usd),
        "depletion": dstats,
    }

    return _build_return(fan_btc, fan, _depl_extra(dstats), result)


# Backward-compatible aliases
def _mc_retire_overlay(m, p, ts, t_start, t_end, dt,
                        start_stack, disp_mode, existing_annot_count=0):
    return _mc_withdraw_overlay(m, p, ts, t_start, t_end, dt,
                                 start_stack, disp_mode, "ret", existing_annot_count)


def _mc_supercharge_overlay(m, p, ts, t_start, t_end, dt,
                             start_stack, disp_mode, existing_annot_count=0):
    return _mc_withdraw_overlay(m, p, ts, t_start, t_end, dt,
                                 start_stack, disp_mode, "sc", existing_annot_count)


def _mc_heatmap_overlay(m, p, ep, entry_t, eyrs):
    """Compute MC-derived CAGR percentile rows for the heatmap.

    Returns (mc_cagr, mc_prices, mc_mults, mc_labels, mc_result) or
    (None, None, None, None, None) on cache hit / error.
    mc_cagr:  (n_pcts, n_exit_years) CAGR array
    mc_prices: (n_pcts, n_exit_years) exit price array
    mc_mults:  (n_pcts, n_exit_years) price multiple array
    mc_labels: row labels e.g. ['MC P5%', ...]
    mc_result: JSON dict for localStorage (or None if cache hit)
    """
    n_bins    = int(p.get("mc_bins", 5))
    n_sims    = int(p.get("mc_sims", 800))
    mc_window = p.get("mc_window")
    mc_freq   = p.get("mc_freq", "Monthly")
    mc_ppy    = FREQ_PPY.get(mc_freq, 12)
    mc_dt     = 1.0 / mc_ppy
    step_days = _FREQ_STEP_DAYS.get(mc_freq, 30)

    mc_years  = int(p.get("mc_years", 10))
    t_start   = entry_t
    mc_t_end  = t_start + mc_years
    mc_ts     = np.arange(t_start, mc_t_end + mc_dt * 0.5, mc_dt)

    # Ensure start_yr is set for path_key (heatmap uses entry_yr, not start_yr)
    p_for_key = dict(p, start_yr=int(p.get("entry_yr", 2024)))
    path_key    = _mc_path_key(p_for_key, "hm")
    overlay_key = {"entry_price": float(ep)}

    def _compute_cagr_rows(price_paths, mc_ts):
        """From MC price paths, compute CAGR percentile rows for each exit year."""
        n_pcts = len(_MC_FAN_PCTS)
        n_eyrs = len(eyrs)
        mc_cagr  = np.full((n_pcts, n_eyrs), np.nan)
        mc_prices = np.full((n_pcts, n_eyrs), np.nan)
        mc_mults  = np.full((n_pcts, n_eyrs), np.nan)

        for ci, ey in enumerate(eyrs):
            et = yr_to_t(ey, m.genesis)
            nyr = et - entry_t
            if et > mc_ts[-1]:
                continue
            if nyr <= 0:
                # Same-year exit: use half-year offset for MC price lookup,
                # show simple return (not annualized) like QR heatmap
                et_eff = entry_t + 0.5
                if et_eff > mc_ts[-1]:
                    continue
                idx = int(np.argmin(np.abs(mc_ts - et_eff)))
                if idx == 0:
                    idx = min(1, len(mc_ts) - 1)
            else:
                idx = int(np.argmin(np.abs(mc_ts - et)))
            prices_at_exit = price_paths[:, idx]  # one per simulation
            pct_prices = np.percentile(prices_at_exit, [pf * 100 for pf in _MC_FAN_PCTS])
            for ri, pp in enumerate(pct_prices):
                mc_prices[ri, ci] = pp
                mc_mults[ri, ci] = pp / ep if ep > 0 else 0.0
                if nyr <= 0:
                    mc_cagr[ri, ci] = (pp / ep - 1.0) * 100.0 if ep > 0 else 0.0
                else:
                    mc_cagr[ri, ci] = ((pp / ep) ** (1.0 / nyr) - 1.0) * 100.0 if ep > 0 else 0.0

        return mc_cagr, mc_prices, mc_mults

    mc_labels = [f"MC P{int(pf*100)}%" for pf in _MC_FAN_PCTS]

    # ── Check cache ──────────────────────────────────────────────────────
    cached = p.get("mc_cached")
    if cached and cached.get("path_key") == path_key:
        price_paths = _mc_paths_from_lists(cached["price_paths"])
        cached_ts = np.array(cached["ts"])
        mc_cagr, mc_prices_arr, mc_mults = _compute_cagr_rows(price_paths, cached_ts)
        if cached.get("overlay_key") == overlay_key:
            return mc_cagr, mc_prices_arr, mc_mults, mc_labels, None
        # Entry price changed — recompute CAGRs from cached paths
        result = {
            "tab": "hm",
            "path_key": path_key,
            "overlay_key": overlay_key,
            "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "ts": cached["ts"],
            "price_paths": cached["price_paths"],
        }
        return mc_cagr, mc_prices_arr, mc_mults, mc_labels, result

    # ── Run full simulation ──────────────────────────────────────────────
    trans, bin_edges, n_bins = _get_transition_matrix(m, n_bins, step_days, mc_window)
    n_steps = len(mc_ts)

    # Starting percentile bin: for historical entry, use the user's entry
    # percentile directly; for current-year entry, derive from live price.
    eyr = int(p.get("entry_yr", 2024))
    yr_now_hm = pd.Timestamp.today().year
    if eyr < yr_now_hm:
        # Historical: entry percentile IS the starting bin
        raw_pctile = float(p.get("entry_q", 50)) / 100.0
    else:
        live_price = float(p.get("mc_live_price", 0))
        t_now = max(t_start, 0.5)
        if live_price > 0:
            raw_pctile = _find_lot_percentile(t_now, live_price, m.qr_fits)
        else:
            raw_pctile = _find_lot_percentile(t_now,
                                               float(qr_price(0.5, t_now, m.qr_fits)),
                                               m.qr_fits)
    start_pctile = round(raw_pctile * 20) / 20
    start_pctile = max(0.05, min(start_pctile, 0.95))

    price_paths, _ = monte_carlo_prices(
        trans, bin_edges, start_pctile, n_steps, n_sims,
        m.qr_fits, m.genesis, t_start, mc_dt,
    )

    mc_cagr, mc_prices_arr, mc_mults = _compute_cagr_rows(price_paths, mc_ts)

    result = {
        "tab": "hm",
        "path_key": path_key,
        "overlay_key": overlay_key,
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "ts": [round(float(t), 6) for t in mc_ts],
        "price_paths": _mc_paths_to_lists(price_paths),
    }

    return mc_cagr, mc_prices_arr, mc_mults, mc_labels, result


# ── HODL Supercharger ─────────────────────────────────────────────────────────

_DELAY_COLORS = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#E07000']
_ANNOT_COLORS = ['#636EFA', '#EF553B', '#1D8348', '#AB63FA', '#E07000']
_DASH_STYLES  = ['solid', 'dash', 'dot', 'dashdot', 'longdash']


def build_supercharge_figure(m, p):
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
            for q in sel_qs:
                stack  = start_stack
                vals   = np.empty(len(ts_d))
                depl_t = None
                for i, t in enumerate(ts_d):
                    years_elapsed = t - t_start_d
                    adj_wd = wd_amount * ((1 + inflation) ** years_elapsed)
                    price  = float(qr_price(q, max(t, 0.5), m.qr_fits))
                    stack -= adj_wd / price
                    stack  = max(stack, 0.0)
                    vals[i] = stack
                    if stack == 0.0 and depl_t is None:
                        depl_t = t
                if disp_mode == "usd":
                    prices_arr = np.array([float(qr_price(q, max(t, 0.5), m.qr_fits))
                                           for t in ts_d])
                    y_vals = vals * prices_arr
                else:
                    y_vals = vals
                results[(d, q)] = (ts_d, y_vals, depl_t, t_start_d)

        traces         = []
        deplete_annots = []

        _AY_LEVELS = [-20, -33, -46]   # stagger by ~1 font-height (13 px) each level

        def _depl_annot(depl_t, t_start_d, d, col, stagger=0):
            depl_yr = int((syr + d) + (depl_t - t_start_d) *
                          (eyr - (syr + d)) / max(t_end - t_start_d, 1e-6))
            return dict(
                x=depl_t - dt, xref="x",   # last nonzero step, aligns with band end
                y=0, yref="paper",
                ax=28, ay=_AY_LEVELS[stagger % 3],
                text=f"\u2248{depl_yr}",
                showarrow=True, arrowhead=2, arrowsize=1,
                arrowcolor=col, font=dict(size=11, color=col),
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
                pct   = q * 100
                q_lbl = f"Q{pct:.4g}%" if pct >= 1 else f"Q{pct:.3g}%"
                for di, d in enumerate(delays):
                    key = (d, q)
                    if key not in results:
                        continue
                    ts_d, y_vals, depl_t, t_start_d = results[key]
                    d_lbl = f"+{int(d)}yr" if d == int(d) else f"+{d:.1f}yr"
                    traces.append(go.Scatter(
                        x=list(ts_d), y=list(y_vals), mode="lines",
                        name=f"{q_lbl} delay={d_lbl}",
                        line=dict(color=col, width=1.8,
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
                y_med  = np.median(all_y, axis=0)
                col    = _DELAY_COLORS[di % len(_DELAY_COLORS)]
                d_lbl  = f"+{int(d)}yr" if d == int(d) else f"+{d:.1f}yr"
                traces.append(go.Scatter(
                    x=list(ts_d), y=list(y_max), mode="lines",
                    line=dict(color=col, width=0), showlegend=False, hoverinfo="skip",
                ))
                traces.append(go.Scatter(
                    x=list(ts_d), y=list(y_min), mode="lines",
                    fill="tonexty", fillcolor=_hex_alpha(col, 0.2),
                    line=dict(color=col, width=0), showlegend=False, hoverinfo="skip",
                ))
                med_final = (fmt_price(float(y_med[-1])) if disp_mode == "usd"
                             else f"{float(y_med[-1]):.4f} BTC")
                traces.append(go.Scatter(
                    x=list(ts_d), y=list(y_med), mode="lines",
                    name=f"Delay {d_lbl}  \u2192  {med_final}",
                    line=dict(color=col, width=2),
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
        tick_ts, tick_lbls = _year_ticks(syr, eyr, m.genesis)
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
        layout["annotations"] = deplete_annots
        shapes = []
        if p.get("show_today"):
            td = today_t(m.genesis)
            if t_start_base <= td <= t_end:
                shapes.append(dict(
                    type="line", x0=td, x1=td, y0=0, y1=1,
                    yref="paper",
                    line=dict(color="#FF6600", dash="dash", width=1.5),
                    opacity=0.85,
                ))
        # ── Monte Carlo fan overlay ───────────────────────────────────────────
        mc_result = None
        if _HAS_MARKOV and p.get("mc_enabled"):
            t_start_base = max(yr_to_t(syr, m.genesis), 1.0)
            mc_traces, mc_annots, mc_result = _mc_supercharge_overlay(
                m, p, ts_d if delays == [0.0] else np.arange(t_start_base, t_end + dt * 0.5, dt),
                t_start_base, t_end, dt, start_stack, disp_mode,
                len(deplete_annots))
            traces.extend(mc_traces)
            if mc_annots:
                deplete_annots.extend(mc_annots)
                layout["annotations"] = deplete_annots

        # Sort all depletion annotations by x and reassign stagger levels
        if len(deplete_annots) > 1:
            deplete_annots.sort(key=lambda a: a["x"])
            _AY = [-20, -33, -46]
            for i, a in enumerate(deplete_annots):
                a["ay"] = _AY[i % 3]
            layout["annotations"] = deplete_annots

        layout["shapes"]     = shapes
        layout["showlegend"] = show_legend
        fig = go.Figure(data=traces, layout=go.Layout(**layout))
        _apply_watermark(fig)
        return fig, mc_result

    # ── MODE B: fixed depletion date \u2192 max withdrawal per period ──────────────
    else:
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
            # Color = delay, one quantile closest to display_q
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
            # Color = quantile, X = delay years
            for q in sel_qs:
                col   = m.qr_colors.get(q, "#888888")
                pct   = q * 100
                q_lbl = f"Q{pct:.4g}%" if pct >= 1 else f"Q{pct:.3g}%"
                y_q   = [max_wd.get((d, q), 0) for d in delays]
                traces.append(go.Scatter(
                    x=delays, y=y_q, mode="lines+markers",
                    name=q_lbl,
                    line=dict(color=col, width=2),
                    marker=dict(color=col, size=7),
                ))

        else:
            # Layout 2: color = delay, X = quantile fraction
            for di, d in enumerate(delays):
                col   = _DELAY_COLORS[di % len(_DELAY_COLORS)]
                d_lbl = f"+{int(d)}yr" if d == int(d) else f"+{d:.1f}yr"
                y_d   = [max_wd.get((d, q), 0) for q in sel_qs]
                qlbls = [f"Q{q*100:.4g}%" if q*100 >= 1 else f"Q{q*100:.3g}%"
                         for q in sel_qs]
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
