"""Citadel Planner chart builder — multi-line portfolio visualization."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from typing import Any

import _app_ctx
from btc_core import ModelData, yr_to_t, fmt_price
from engines.citadel import SimConfig, simulate, PriceModel

from figures.common import (
    _QR_LINE_WIDTH, _ANNOT_STAGGER_Y,
    _FONT_ANNOT,
    _get_palette, _build_thermal_colors, _fmt_q_label,
    _build_time_array, _get_starting_stack,
    _sim_layout, _finalize_chart, _error_figure,
    _stagger_depletion_annots,
)


# ── ModelData → PriceModel adapter ────────────────────────────────────────────

class _ModelAdapter:
    """Adapts _app_ctx price model + ModelData to engines.citadel.PriceModel protocol."""

    def __init__(self, m: ModelData, model_key: str = "bub"):
        self._model = _app_ctx.PRICE_MODELS.get(model_key, _app_ctx.DEFAULT_MODEL)
        self.fits = self._model.fits if hasattr(self._model, "fits") else {}
        self.genesis = m.genesis

    def price_at(self, q: float, t: float) -> float:
        return float(self._model.price_at(q, max(t, 0.5)))

    def quantile_at(self, price: float, t: float) -> float:
        """Bisection search: find q such that price_at(q, t) ~ price."""
        qs = sorted(self.fits.keys())
        if not qs:
            return 0.5
        lo, hi = qs[0], qs[-1]
        for _ in range(50):
            mid = (lo + hi) / 2
            if self.price_at(mid, t) < price:
                lo = mid
            else:
                hi = mid
        return max(0.001, min((lo + hi) / 2, 0.999))


# ── Trace colors ──────────────────────────────────────────────────────────────

_C_TOTAL      = "#000000"       # black — total portfolio
_C_BTC        = "#F7931A"       # bitcoin orange — BTC holdings USD
_C_CASH       = "#C0C0C0"       # silver — cash
_C_RESERVES   = "#4A90D9"       # blue — reserves total
_C_INVEST     = "#27AE60"       # green — investments total
_C_SPEND      = "#E74C3C"       # red — monthly spending


def _build_sim_config(p: dict) -> SimConfig:
    """Convert callback params dict → SimConfig dataclass."""

    # Quantiles
    sel_qs = sorted([float(q) for q in (p.get("selected_qs") or [0.01, 0.10, 0.25])])

    # Reserve bins
    reserve_bins = [
        {"label": "Short (T-Bills)",
         "initial": float(p.get("res_short_init", 50000)),
         "rate": float(p.get("res_short_rate", 5.0)),
         "volatility": float(p.get("res_short_vol", 2.0))},
        {"label": "Medium (T-Notes)",
         "initial": float(p.get("res_med_init", 100000)),
         "rate": float(p.get("res_med_rate", 4.5)),
         "volatility": float(p.get("res_med_vol", 8.0))},
        {"label": "Long (T-Bonds)",
         "initial": float(p.get("res_long_init", 50000)),
         "rate": float(p.get("res_long_rate", 4.0)),
         "volatility": float(p.get("res_long_vol", 15.0))},
    ]

    # Investment bins
    invest_bins = [
        {"label": "Equities",
         "initial": float(p.get("inv_eq_init", 200000)),
         "return_rate": float(p.get("inv_eq_rate", 10.0)),
         "volatility": float(p.get("inv_eq_vol", 16.0))},
        {"label": "Bonds",
         "initial": float(p.get("inv_bd_init", 100000)),
         "return_rate": float(p.get("inv_bd_rate", 5.0)),
         "volatility": float(p.get("inv_bd_vol", 7.0))},
    ]

    # High-Q trigger
    high_q_split = {
        "cash": float(p.get("high_q_split_cash", 20)) / 100,
        "res_short": float(p.get("high_q_split_rs", 20)) / 100,
        "res_med": float(p.get("high_q_split_rm", 20)) / 100,
        "res_long": float(p.get("high_q_split_rl", 10)) / 100,
        "inv_eq": float(p.get("high_q_split_eq", 20)) / 100,
        "inv_bd": float(p.get("high_q_split_bd", 10)) / 100,
    }
    high_q_action = {
        "mode": p.get("high_q_mode", "gradual"),
        "rate": float(p.get("high_q_rate", 2.0)),
        "duration": int(p.get("high_q_dur", 6)),
        "split": high_q_split,
    }

    # Low-Q trigger
    low_q_split = {
        "cash": float(p.get("low_q_split_cash", 10)) / 100,
        "res_short": float(p.get("low_q_split_rs", 10)) / 100,
        "res_med": float(p.get("low_q_split_rm", 10)) / 100,
        "res_long": float(p.get("low_q_split_rl", 10)) / 100,
        "inv_eq": float(p.get("low_q_split_eq", 40)) / 100,
        "inv_bd": float(p.get("low_q_split_bd", 20)) / 100,
    }
    low_q_action = {
        "mode": p.get("low_q_mode", "lump"),
        "rate": float(p.get("low_q_rate", 10.0)),
        "duration": int(p.get("low_q_dur", 1)),
        "split": low_q_split,
    }

    return SimConfig(
        price_model=p.get("price_model", "bub"),
        start_stack=float(p.get("start_stack", 1.0)),
        selected_qs=sel_qs,
        cash_initial=float(p.get("cash_initial", 50000)),
        cash_rate=float(p.get("cash_rate", 4.0)),
        reserve_bins=reserve_bins,
        invest_bins=invest_bins,
        monthly_spend=float(p.get("monthly_spend", 5000)),
        inflation=float(p.get("inflation", 4.0)),
        spend_growth=float(p.get("spend_growth", 0.0)),
        high_q_trigger=float(p.get("high_q_trigger", 80)) / 100,
        high_q_action=high_q_action,
        low_q_trigger=float(p.get("low_q_trigger", 20)) / 100,
        low_q_action=low_q_action,
        lump_cooldown=int(p.get("lump_cooldown", 12)),
        cash_floor=float(p.get("cash_floor", 0)),
        reserve_floors=[
            float(p.get("res_short_floor", 0)),
            float(p.get("res_med_floor", 0)),
            float(p.get("res_long_floor", 0)),
        ],
        scf_enabled=bool(p.get("scf_enabled")),
        scf_amount=float(p.get("scf_amount", 0)),
        scf_type=p.get("scf_type", "term"),
        scf_rate=float(p.get("scf_rate", 8.0)),
        scf_term=int(p.get("scf_term", 60)),
        scf_repay_trigger=float(p.get("scf_repay_trigger", 1.0)),
        start_yr=int(p.get("start_yr", 2031)),
        end_yr=int(p.get("end_yr", 2075)),
        freq=p.get("freq", "Monthly"),
        n_sims=int(p.get("n_sims", 1)),
        tax_rate=0.0,
    )


def build_citadel_figure(m: ModelData, p: dict[str, Any]) -> tuple[go.Figure, dict | None]:
    """Build multi-line portfolio chart from Citadel Planner simulation.

    p keys: start_stack, cash_initial, cash_rate, monthly_spend, inflation,
            spend_growth, reserve/invest bins, rebalancing triggers, floors,
            scf params, start_yr, end_yr, freq, selected_qs, disp_mode,
            log_y, annotate, show_legend, legend_pos, palette, ...
    """
    disp_mode = p.get("disp_mode", "usd_per_asset")

    # Build SimConfig and run simulation
    try:
        config = _build_sim_config(p)
    except (ValueError, TypeError) as e:
        return _error_figure(m, f"Config error: {e}"), None

    model_key = p.get("price_model", "bub")
    model = _ModelAdapter(m, model_key=model_key)

    # Deterministic always runs with n_sims=1; MC overlay runs separately
    config.n_sims = 1

    try:
        result = simulate(config, model)
    except (ValueError, NotImplementedError) as e:
        return _error_figure(m, f"Simulation error: {e}"), None

    # Extract time axis and median data
    ts = result.time_axis
    med = result.median
    n_periods = len(ts)
    if n_periods == 0:
        return _error_figure(m, "No simulation periods"), None

    # Spending per period (cumulative -> per-period via diff)
    cum_spend = result.cumulative_spend[0]  # sim 0 (deterministic)
    period_spend = np.diff(cum_spend, prepend=0)

    # Build traces
    traces = []

    if disp_mode in ("usd_total", "usd_per_asset"):
        # Total portfolio
        total_y = med["total"]
        traces.append(go.Scatter(
            x=list(ts), y=list(total_y), mode="lines",
            name=f"Total Portfolio  \u2192  {fmt_price(float(total_y[-1]))}",
            line=dict(color=_C_TOTAL, width=_QR_LINE_WIDTH + 0.5),
        ))

    if disp_mode == "usd_per_asset":
        # BTC Holdings (USD)
        btc_usd = med["btc_usd"]
        traces.append(go.Scatter(
            x=list(ts), y=list(btc_usd), mode="lines",
            name=f"BTC Holdings  \u2192  {fmt_price(float(btc_usd[-1]))}",
            line=dict(color=_C_BTC, width=_QR_LINE_WIDTH),
        ))

        # Cash
        cash_y = med["cash"]
        traces.append(go.Scatter(
            x=list(ts), y=list(cash_y), mode="lines",
            name=f"Cash  \u2192  {fmt_price(float(cash_y[-1]))}",
            line=dict(color=_C_CASH, width=_QR_LINE_WIDTH, dash="dash"),
        ))

        # Reserves total
        res_y = med["reserves_total"]
        traces.append(go.Scatter(
            x=list(ts), y=list(res_y), mode="lines",
            name=f"Reserves  \u2192  {fmt_price(float(res_y[-1]))}",
            line=dict(color=_C_RESERVES, width=_QR_LINE_WIDTH),
        ))

        # Investments total
        inv_y = med["investments_total"]
        traces.append(go.Scatter(
            x=list(ts), y=list(inv_y), mode="lines",
            name=f"Investments  \u2192  {fmt_price(float(inv_y[-1]))}",
            line=dict(color=_C_INVEST, width=_QR_LINE_WIDTH),
        ))

        # Monthly spending
        traces.append(go.Scatter(
            x=list(ts), y=list(period_spend), mode="lines",
            name=f"Spending/period  \u2192  {fmt_price(float(period_spend[-1]))}",
            line=dict(color=_C_SPEND, width=_QR_LINE_WIDTH * 0.7, dash="dot"),
        ))

    elif disp_mode == "btc":
        # BTC holdings only
        btc_stack = result.btc_holdings[0]  # sim 0
        traces.append(go.Scatter(
            x=list(ts), y=list(btc_stack), mode="lines",
            name=f"BTC Stack  \u2192  {float(btc_stack[-1]):.4f} BTC",
            line=dict(color=_C_BTC, width=_QR_LINE_WIDTH),
        ))

    elif disp_mode == "usd_total":
        pass  # total already added above

    # Depletion annotations
    deplete_annots = []
    depl_period = result.depletion_period[0]  # sim 0
    if depl_period is not None and depl_period < n_periods:
        depl_t = ts[depl_period]
        syr = config.start_yr
        eyr = config.end_yr
        t_start = ts[0]
        t_end = ts[-1]
        depl_yr = int(syr + (depl_t - t_start) * (eyr - syr) / max(t_end - t_start, 1e-6))
        deplete_annots.append(dict(
            x=depl_t, xref="x",
            y=0, yref="paper",
            ax=28, ay=_ANNOT_STAGGER_Y[0],
            text=f"\u2248{depl_yr}",
            showarrow=True, arrowhead=2, arrowsize=1,
            arrowcolor=_C_SPEND,
            font=dict(size=_FONT_ANNOT, color=_C_SPEND),
        ))

    # Build layout
    syr = config.start_yr
    eyr = config.end_yr
    t_start = ts[0]
    t_end = ts[-1]
    from engines.citadel import FREQ_PPY as _CITADEL_FREQ_PPY
    ppy = _CITADEL_FREQ_PPY.get(config.freq, 12)
    dt = 1.0 / ppy

    # MC overlay — generate fan band traces from Markov price paths
    mc_result = None
    try:
        from mc_overlay import _mc_citadel_overlay, _HAS_MARKOV
        if _HAS_MARKOV and p.get("mc_enabled"):
            mc_traces, mc_result = _mc_citadel_overlay(m, p, config, model)
            if mc_traces:
                traces.extend(mc_traces)
    except ImportError:
        pass

    q_label = f"Q{config.selected_qs[0]*100:g}%" if config.selected_qs else "Q25%"
    ylabel = "BTC Remaining" if disp_mode == "btc" else "USD Value"
    title = f"Citadel Planner \u2014 {fmt_price(config.monthly_spend)}/mo \u00b7 {q_label}"

    layout, _x_end = _sim_layout(m, p, title, ylabel, ts, t_start, t_end, dt, syr, eyr)
    layout["annotations"] = deplete_annots

    _stagger_depletion_annots(deplete_annots, layout)

    return _finalize_chart(traces, layout, p, "cp", mc_result)
