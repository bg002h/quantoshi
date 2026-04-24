"""Leverage calculator — figure builder and math helpers.

Design spec: docs/superpowers/specs/2026-04-18-leverage-calculator-design.md
"""
from __future__ import annotations

import datetime as _dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import _app_ctx
from colors import TRACE_WIDTH
from figures.common import _base_layout, _apply_watermark

# Shared project genesis — every PriceModel in btc_core uses this as t=0
# (CLAUDE.md: "All models use 2009-07-25 as their time origin").
_GENESIS = pd.Timestamp("2009-07-25")


def _t_yr(target_date) -> float:
    return (pd.Timestamp(target_date) - _GENESIS).days / 365.25


def _bm_support_log10(t_yr: float) -> float:
    """Log10(support_price) for the Bubble Model at time t (years since genesis).

    Uses the BM support line (`support_intercept + support_slope * log10(t)`)
    rather than the bubble composite. This is the "floor" of the BM —
    the underlying power-law base that bubbles oscillate above.
    """
    import math
    md = _app_ctx.M
    return md.support_intercept + md.support_slope * math.log10(max(t_yr, 0.01))


def _bm_sigma_down(t_yr: float) -> float:
    md = _app_ctx.M
    return float(md.bm_sigma0_down * max(t_yr, 0.5) ** (-md.bm_alpha_down))


def _bm_sigma_up(t_yr: float) -> float:
    md = _app_ctx.M
    return float(md.bm_sigma0_up * max(t_yr, 0.5) ** (-md.bm_alpha_up))


def floor_price(model_short: str, q: float, target_date) -> float:
    """Return the `model_short`-q floor price at `target_date` in USD.

    Bubble Model special case: floor is the support power-law line
    (not the composite-quantile), log-shifted downward by |z|·σ_down for q < 0.5.
    This matches the user's intent for "BM floor" — the support line, not
    the composite's lower band.

    Other models: `model.interp_price(q, t)` — log-space interpolation between
    adjacent QR fits.

    Args:
        model_short: key into _app_ctx.PRICE_MODELS (e.g. "bub", "pl", "lppl").
        q: quantile in (0, 1), e.g. 0.01 for Q1%.
        target_date: datetime.date or datetime.datetime.
    """
    t_yr = _t_yr(target_date)
    if model_short == "bub":
        log_support = _bm_support_log10(t_yr)
        if q != 0.5:
            from scipy.stats import norm
            z = float(norm.ppf(q))
            sigma = _bm_sigma_down(t_yr) if q < 0.5 else _bm_sigma_up(t_yr)
            log_support = log_support + z * sigma
        return float(10.0 ** log_support)
    model = _app_ctx.PRICE_MODELS[model_short]
    return float(model.interp_price(q, t_yr))


def implied_quantile(model_short: str, price: float, target_date) -> float:
    """Return the model's implied quantile of `price` at `target_date`.

    Useful for the table "Sell quantile" column — tells the user what
    probability the sell-price corresponds to. For BM this reflects the
    support line's implied quantile (typically around Q5–15%); for other
    models it will echo the pill-bar `q` (modulo small interpolation noise).
    """
    if price <= 0:
        return 0.5
    t_yr = _t_yr(target_date)
    if model_short == "bub":
        import math
        from scipy.stats import norm
        log_support = _bm_support_log10(t_yr)
        log_p = math.log10(price)
        sigma = _bm_sigma_up(t_yr) if log_p >= log_support else _bm_sigma_down(t_yr)
        if sigma < 1e-9:
            return 0.5
        z = (log_p - log_support) / sigma
        return float(norm.cdf(z))
    model = _app_ctx.PRICE_MODELS[model_short]
    try:
        return float(model.find_percentile(t_yr, price))
    except (AttributeError, NotImplementedError):
        return 0.5


def P_max(sell_price: float, H_yr: float, target_cagr: float) -> float:
    """Max rational pay-price today for a target CAGR c over horizon H."""
    return sell_price / (1.0 + target_cagr) ** H_yr


def implied_cagr(sell_price: float, P_now: float, H_yr: float):
    """CAGR implied by buying at P_now today and selling at sell_price in H years."""
    if P_now <= 0 or H_yr <= 0:
        return None
    try:
        return (sell_price / P_now) ** (1.0 / H_yr) - 1.0
    except OverflowError:
        return None


def _parse_date(s):
    """Accept ISO date string, date, or datetime; return datetime.date."""
    if isinstance(s, _dt.datetime):
        return s.date()
    if isinstance(s, _dt.date):
        return s
    return _dt.date.fromisoformat(str(s)[:10])


def build_leverage_figure(p: dict) -> go.Figure:
    """Build the max-pay-price plot."""
    H_slider = max(float(p["lev_horizon"]), 0.01)
    P_now    = max(float(p["lev_price"]), 1.0)
    c        = float(p["lev_cagr"]) / 100.0
    r_b      = float(p["lev_rb"]) / 100.0
    r_l      = float(p["lev_rl"]) / 100.0
    q        = float(p["lev_floor_q"])
    model    = str(p["lev_model"])
    buy_date = _parse_date(p["lev_date"])

    H_grid = np.linspace(0.25, 20.0, 400)
    dates = [buy_date + _dt.timedelta(days=int(round(H * 365.25))) for H in H_grid]
    try:
        sell_grid = np.array([floor_price(model, q, d) for d in dates])
    except (KeyError, AttributeError, ValueError):
        sell_grid = np.full_like(H_grid, np.nan)

    def _curve(target_c):
        return sell_grid / (1.0 + target_c) ** H_grid

    curve_0   = _curve(0.0)
    curve_rl  = _curve(r_l)
    curve_rb  = _curve(r_b)
    curve_c   = _curve(c)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=H_grid, y=curve_0, name="Nominal breakeven (0%)",
        line=dict(width=TRACE_WIDTH, dash="dot"),
        hovertemplate="H=%{x:.2f} yr<br>Max pay=%{y:$,.0f}<extra>0%</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=H_grid, y=curve_rl, name=f"Opp cost ({r_l*100:.2f}%)",
        line=dict(width=TRACE_WIDTH),
        hovertemplate="H=%{x:.2f} yr<br>Max pay=%{y:$,.0f}<extra>r_l</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=H_grid, y=curve_rb, name=f"Borrow cost ({r_b*100:.2f}%)",
        line=dict(width=TRACE_WIDTH),
        hovertemplate="H=%{x:.2f} yr<br>Max pay=%{y:$,.0f}<extra>r_b</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=H_grid, y=curve_c, name=f"Your target ({c*100:.1f}%)",
        line=dict(width=TRACE_WIDTH * 1.8),
        hovertemplate="H=%{x:.2f} yr<br>Max pay=%{y:$,.0f}<extra>target</extra>",
    ))

    fig.add_hline(y=P_now, line=dict(dash="dash"),
                  annotation_text=f"Current: ${P_now:,.0f}",
                  annotation_position="right")

    try:
        sell_at_slider = floor_price(model, q, buy_date + _dt.timedelta(days=int(round(H_slider * 365.25))))
        y_dot = sell_at_slider / (1.0 + c) ** H_slider
        fig.add_vline(x=H_slider, line=dict(dash="dash"))
        fig.add_trace(go.Scatter(
            x=[H_slider], y=[y_dot], mode="markers",
            marker=dict(size=12, symbol="circle"),
            name="Your max pay today", showlegend=False,
            hovertemplate="H=%{x:.2f} yr<br>Max pay=%{y:$,.0f}<extra></extra>",
        ))
    except (KeyError, AttributeError, ValueError):
        pass

    q_label = f"Q{q*100:g}%"
    title = (
        f"<b>Max rational pay-price — reversion to "
        f"{model} {q_label} floor</b><br>"
        f"<span style='font-size:0.85em'>"
        f"Current date: {buy_date.isoformat()}  ·  "
        f"Current price: ${P_now:,.0f}</span>"
    )
    base = _base_layout(title=title, xlabel="Horizon H (years)",
                        ylabel="Max pay-price today ($)")
    base["xaxis"].update(range=[0.25, 20])
    base["yaxis"].update(type="log", tickformat="$,.0f")
    base["margin"] = dict(l=60, r=40, t=90, b=60)
    fig.update_layout(**base)
    show_legend = "legend" in (p.get("lev_toggles") or ())
    fig.update_layout(showlegend=show_legend)
    # Preserve Plotly pan/zoom/hover across redundant figure rebuilds.
    # See spec 2026-04-24-single-redraw-per-snapshot-design.md.
    from figures.common import _uirevision_key
    fig.update_layout(uirevision=_uirevision_key(p, "lev"))
    _apply_watermark(fig)
    return fig
