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


def floor_price(model_short: str, q: float, target_date) -> float:
    """Return the `model_short`-q floor price at `target_date` in USD.

    Args:
        model_short: key into _app_ctx.PRICE_MODELS (e.g. "bub", "pl", "lppl").
        q: quantile in (0, 1), e.g. 0.01 for Q1%.
        target_date: datetime.date or datetime.datetime.

    Returns:
        Floor price in USD (positive float).
    """
    model = _app_ctx.PRICE_MODELS[model_short]
    t_yr = (pd.Timestamp(target_date) - _GENESIS).days / 365.25
    return float(model.interp_price(q, t_yr))


def P_max(sell_price: float, H_yr: float, target_cagr: float) -> float:
    """Max rational pay-price today for a target CAGR c over horizon H."""
    return sell_price / (1.0 + target_cagr) ** H_yr


def implied_cagr(sell_price: float, P_now: float, H_yr: float):
    """CAGR implied by buying at P_now today and selling at sell_price in H years."""
    if P_now <= 0 or H_yr <= 0:
        return None
    return (sell_price / P_now) ** (1.0 / H_yr) - 1.0


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
    _apply_watermark(fig)
    return fig
