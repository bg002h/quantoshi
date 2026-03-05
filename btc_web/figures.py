"""figures.py — Plotly chart builders for the Bitcoin Projections web app.

Each function takes a ModelData instance and a params dict of control values
and returns a go.Figure ready for dcc.Graph.
"""

import math
import base64
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# btc_app/ is added to sys.path by app.py before this import
import bisect
from btc_core import qr_price, yr_to_t, today_t, fmt_price, _fmt_btc, leo_weighted_entry


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
    _logo_path = Path(__file__).parent / "assets" / "quantoshi_logo.png"
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
        mask = (m.price_years >= t_lo) & (m.price_years <= t_hi)
        y_data = m.price_prices[mask] * (stack if stack > 0 else 1)
        dates  = [m.price_dates[i] for i in range(len(m.price_dates)) if mask[i]]
        traces.append(go.Scatter(
            x=list(m.price_years[mask]), y=list(y_data),
            mode="markers", name="Price data",
            marker=dict(color=m.DATA_COLOR, size=max(2, int(p.get("pt_size", 3))),
                        opacity=float(p.get("pt_alpha", 0.6))),
            text=dates, hovertemplate="%{text}<br>%{y:$,.0f}<extra></extra>",
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


# ── CAGR Heatmap ──────────────────────────────────────────────────────────────

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

    # ── colorscale ────────────────────────────────────────────────────────────
    mode   = int(p.get("color_mode", 0))
    c_lo   = p.get("c_lo",   m.CAGR_SEG_C_LO)
    c_mid1 = p.get("c_mid1", m.CAGR_SEG_C_MID1)
    c_mid2 = p.get("c_mid2", m.CAGR_SEG_C_MID2)
    c_hi   = p.get("c_hi",   m.CAGR_SEG_C_HI)
    b1     = float(p.get("b1", m.CAGR_SEG_B1))
    b2     = float(p.get("b2", m.CAGR_SEG_B2))
    mn, mx = float(mc.min()), float(mc.max())

    if mode == 0:  # Segmented — breakpoints anchored at b1/b2 CAGR values
        colorscale, zmin, zmax = _seg_colorscale(mc, b1, b2, c_lo, c_mid1, c_mid2, c_hi)
    elif mode == 1:  # Data-Scaled — simple linear gradient across data range
        colorscale = _dense_colorscale(lambda t: _lerp_hex(c_lo, c_hi, t))
        zmin, zmax = mn, mx
    else:  # Diverging — centred at 0% CAGR; c_mid1/c_mid2 mark the zero crossing
        abs_max = max(abs(mn), abs(mx), 1e-6)
        zmin, zmax = -abs_max, abs_max

        def _div_color(t):
            # t=0 → c_lo, t=0.5 → c_mid1/c_mid2 boundary (0% CAGR), t=1 → c_hi
            if t < 0.5:
                return _lerp_hex(c_lo, c_mid1, t * 2.0)
            else:
                return _lerp_hex(c_mid2, c_hi, (t - 0.5) * 2.0)

        colorscale = _dense_colorscale(_div_color)

    # ── cell text ─────────────────────────────────────────────────────────────
    vfmt    = p.get("vfmt", "cagr")
    hm_stk  = float(p.get("stack", 0))
    annots  = []
    for ri in range(len(xqs)):
        for ci in range(len(eyrs)):
            vc2 = mc[ri, ci]
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
                tx = f"{vm:.2f}×"
            elif vfmt == "cagr_mult":
                tx = f"{vc2:+.0f}%\n{vm:.2f}×"
            elif vfmt == "mult_port":
                pv = fmt_price(vp2 * hm_stk) if hm_stk > 0 else fmt_price(vp2)
                tx = f"{vm:.2f}×\n{pv}"
            else:
                tx = ""

            if tx:
                # brightness of cell to pick text colour
                cell_norm = (vc2 - zmin) / max(zmax - zmin, 1e-6)
                txt_col = "#ffffff" if cell_norm < 0.55 else "#111111"
                annots.append(dict(
                    x=ci, y=ri,
                    text=tx.replace("\n", "<br>"),
                    showarrow=False,
                    font=dict(size=int(p.get("cell_font_size", 9)), color=txt_col),
                    xref="x", yref="y",
                ))

    # ── quantile y-axis labels ────────────────────────────────────────────────
    ylabels = []
    for q in xqs:
        pct = q * 100
        ylabels.append(f"Q{pct:.4g}%" if pct >= 1 else f"Q{pct:.3g}%")

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

    entry_lbl = (f"Entry: {eyr}  {fmt_price(ep)}  ·  Q{eq*100:.4g}%"
                 if not (p.get("use_lots") and lots)
                 else f"Entry: lots weighted avg  {fmt_price(ep)}")

    fig.update_layout(
        title=dict(text=f"CAGR Heatmap — {entry_lbl}",
                   font=dict(color=m.TITLE_COLOR, size=13)),
        paper_bgcolor=m.PLOT_BG_COLOR,
        plot_bgcolor=m.PLOT_BG_COLOR,
        font=dict(color=m.TEXT_COLOR),
        xaxis=dict(title="Exit Year", gridcolor=m.GRID_MAJOR_COLOR,
                   linecolor=m.SPINE_COLOR, tickcolor=m.TEXT_COLOR),
        yaxis=dict(title="Exit Quantile", gridcolor=m.GRID_MAJOR_COLOR,
                   linecolor=m.SPINE_COLOR, tickcolor=m.TEXT_COLOR),
        annotations=annots,
        margin=dict(l=70, r=20, t=60, b=50),
    )
    _apply_watermark(fig)
    return fig


# ── DCA Accumulator ───────────────────────────────────────────────────────────

def build_dca_figure(m, p):
    """
    p keys: start_yr, end_yr, start_stack, amount, freq, disp_mode,
            selected_qs, log_y, show_today, dual_y,
            lots, use_lots
    """
    FREQ_PPY = {"Daily": 365, "Weekly": 52, "Monthly": 12, "Quarterly": 4, "Annually": 1}
    freq_str = p.get("freq", "Monthly")
    ppy  = FREQ_PPY.get(freq_str, 12)
    dt   = 1.0 / ppy
    syr  = int(p.get("start_yr", 2024))
    eyr  = int(p.get("end_yr",   2035))
    if eyr <= syr:
        return go.Figure(layout=dict(
            title="Set end year > start year",
            paper_bgcolor=m.PLOT_BG_COLOR, font=dict(color=m.TEXT_COLOR)))

    t_start = max(yr_to_t(syr, m.genesis), 1.0)
    t_end   = yr_to_t(eyr, m.genesis)
    ts      = np.arange(t_start, t_end + dt * 0.5, dt)
    if len(ts) == 0:
        return go.Figure()

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

    ylabel = "USD Value" if disp_mode == "usd" else "BTC Balance"
    layout = _dark_layout(
        m,
        title=f"Bitcoin DCA — {fmt_price(amount)}/{freq_str.lower()[:-2] if freq_str.endswith('ly') else freq_str}",
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
    all_sc_usd_vals = {}  # q -> SC USD value array (for dual-y median)
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
                for i, t in enumerate(ts):
                    price           = float(qr_price(q, max(t, 0.5), m.qr_fits))
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
                                # Sell BTC to repay; tax means we must sell more to net principal
                                sc_stack   -= principal / (price * (1.0 - tax_rate))
                                sc_stack    = max(sc_stack, 0.0)
                                outstanding = 0.0
                    else:                               # cycles exhausted → plain DCA
                        sc_stack += amount / price

                    sc_vals[i]   = sc_stack
                    sc_prices[i] = price

                # Deduct any outstanding balance at simulation end
                # (incomplete final cycle — sell BTC at final price to repay, tax-adjusted)
                if outstanding > 1e-8 and sc_prices[-1] > 0:
                    sc_vals[-1] = max(
                        sc_vals[-1] - outstanding / (sc_prices[-1] * (1.0 - tax_rate)),
                        0.0)

                all_sc_usd_vals[q] = sc_vals * sc_prices

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

    # ── dual Y-axis — always USD Value (median) ───────────────────────────────
    if p.get("dual_y") and all_usd_vals:
        usd_axis = "y" if disp_mode == "usd" else "y2"
        # DCA USD median
        usd_med = np.median(np.array(list(all_usd_vals.values())), axis=0)
        dca_med_final = fmt_price(float(usd_med[-1]))
        traces.append(go.Scatter(
            x=list(ts), y=list(usd_med),
            mode="lines", name=f"USD Value (median)  \u2192  {dca_med_final}",
            line=dict(color="#aaaaaa", dash="dot", width=1),
            yaxis=usd_axis, showlegend=True,
        ))
        # SC USD median (only when SC is active and ran successfully)
        if all_sc_usd_vals:
            sc_usd_med = np.median(np.array(list(all_sc_usd_vals.values())), axis=0)
            sc_med_final = fmt_price(float(sc_usd_med[-1]))
            sc_lbl = f"SC USD (median)  \u2192  {sc_med_final}"
            if sc_factor_val is not None:
                sc_lbl += f"  \u00b7  {sc_factor_val:.2f}\u00d7 DCA"
            traces.append(go.Scatter(
                x=list(ts), y=list(sc_usd_med),
                mode="lines", name=sc_lbl,
                line=dict(color="#888888", dash="dashdot", width=1),
                yaxis=usd_axis, showlegend=True,
            ))
        # y2 axis definition (BTC mode only — USD mode plots on y1)
        if disp_mode == "btc":
            layout["yaxis2"] = dict(
                title=dict(text="USD Value", font=dict(color=m.TEXT_COLOR)),
                overlaying="y", side="right",
                gridcolor=m.GRID_MAJOR_COLOR, linecolor=m.SPINE_COLOR,
                tickcolor=m.TEXT_COLOR,
            )
            if p.get("log_y"):
                layout["yaxis2"]["type"] = "log"

    # ── Stack-celeration factor → append to title ────────────────────────────
    if sc_factor_val is not None:
        layout["title"]["text"] += (
            f"<br><b>Stack-celeration: {sc_factor_val:.2f}\u00d7</b>"
        )

    layout["showlegend"] = bool(p.get("show_legend", True))
    fig = go.Figure(data=traces, layout=go.Layout(**layout))
    _apply_watermark(fig)
    return fig


# ── BTC Retireator ────────────────────────────────────────────────────────────

def build_retire_figure(m, p):
    """
    p keys: start_yr, end_yr, start_stack, wd_amount, freq, inflation,
            disp_mode, selected_qs, log_y, show_today, dual_y, annotate,
            lots, use_lots
    """
    FREQ_PPY = {"Daily": 365, "Weekly": 52, "Monthly": 12, "Quarterly": 4, "Annually": 1}
    freq_str = p.get("freq", "Monthly")
    ppy  = FREQ_PPY.get(freq_str, 12)
    dt   = 1.0 / ppy
    syr  = int(p.get("start_yr", 2025))
    eyr  = int(p.get("end_yr",   2045))
    if eyr <= syr:
        return go.Figure(layout=dict(
            title="Set end year > start year",
            paper_bgcolor=m.PLOT_BG_COLOR, font=dict(color=m.TEXT_COLOR)))

    t_start = max(yr_to_t(syr, m.genesis), 1.0)
    t_end   = yr_to_t(eyr, m.genesis)
    ts      = np.arange(t_start, t_end + dt * 0.5, dt)
    if len(ts) == 0:
        return go.Figure()

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
                deplete_annots.append(dict(
                    x=depl_t, y=0,
                    text=f"≈{depl_yr}",
                    showarrow=True, arrowhead=2, arrowsize=1,
                    arrowcolor=col,
                    font=dict(size=9, color=col),
                    xref="x", yref="y",
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

    layout["showlegend"] = bool(p.get("show_legend", True))
    fig = go.Figure(data=traces, layout=go.Layout(**layout))
    _apply_watermark(fig)
    return fig


# ── HODL Supercharger ─────────────────────────────────────────────────────────

_DELAY_COLORS = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A']
_DASH_STYLES  = ['solid', 'dash', 'dot', 'dashdot', 'longdash']


def build_supercharge_figure(m, p):
    """
    p keys: mode ('a'/'b'), start_stack, start_yr, delays (list), freq, inflation,
            selected_qs, chart_layout (0/1/2), display_q,
            wd_amount (Mode A), end_yr (Mode A), disp_mode (Mode A),
            log_y, annotate, show_today, show_legend,
            target_yr (Mode B), lots, use_lots
    """
    FREQ_PPY = {"Daily": 365, "Weekly": 52, "Monthly": 12, "Quarterly": 4, "Annually": 1}

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
            font=dict(color=m.TEXT_COLOR)))

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

        def _depl_annot(depl_t, t_start_d, d, col):
            depl_yr = int((syr + d) + (depl_t - t_start_d) *
                          (eyr - (syr + d)) / max(t_end - t_start_d, 1e-6))
            return dict(
                x=depl_t, y=0, text=f"\u2248{depl_yr}",
                showarrow=True, arrowhead=2, arrowsize=1,
                arrowcolor=col, font=dict(size=9, color=col),
                xref="x", yref="y",
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
                    deplete_annots.append(_depl_annot(depl_t, t_start_d, d, col))

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
                        deplete_annots.append(_depl_annot(depl_t, t_start_d, d, col))

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
        layout["shapes"]     = shapes
        layout["showlegend"] = show_legend
        fig = go.Figure(data=traces, layout=go.Layout(**layout))
        _apply_watermark(fig)
        return fig

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
        return fig
