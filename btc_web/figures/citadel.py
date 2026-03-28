"""Citadel Planner chart builder — multi-line portfolio visualization."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from typing import Any

import _app_ctx
from btc_core import ModelData, yr_to_t, fmt_price
from tab_defaults import CITADEL
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
    """Adapts _app_ctx price model + ModelData to engines.citadel.PriceModel protocol.

    Precomputes a quantile lookup grid for fast quantile_at() calls during
    MC simulations. Uses the model's own quantile keys (fits dict) to avoid
    KeyError on models that don't interpolate (PL, QR, LPPL, EXP).
    BUB and EF interpolate internally; others require exact key matches."""

    def __init__(self, m: ModelData, model_key: str = "bub", user_model=None):
        if model_key == "u1" and user_model is not None:
            from btc_core import UserModel
            mdl = UserModel.from_store_dict(user_model) if isinstance(user_model, dict) else user_model
            self._model = mdl if mdl else _app_ctx.DEFAULT_MODEL
        else:
            self._model = _app_ctx.PRICE_MODELS.get(model_key, _app_ctx.DEFAULT_MODEL)
        self.fits = self._model.fits if hasattr(self._model, "fits") else {}
        self.genesis = m.genesis
        self._price_grid_cache = {}  # t_key -> (q_grid, prices)
        self._quantized = getattr(self._model, 'quantized', True)
        # Build the quantile grid from model's actual fits keys
        if self.fits:
            self._q_grid = np.array(sorted(self.fits.keys()))
        else:
            # Non-quantized model (e.g., S2F) — use single median point
            self._q_grid = np.array([0.5])

    def price_at(self, q: float, t: float) -> float:
        return float(self._model.price_at(q, max(t, 0.5)))

    def _get_price_grid(self, t: float):
        """Get or build price grid for time t using model's quantile keys."""
        t_key = round(t * 12) / 12  # cache at monthly granularity
        if t_key not in self._price_grid_cache:
            prices = np.array([self.price_at(q, t) for q in self._q_grid])
            self._price_grid_cache[t_key] = prices
        return self._price_grid_cache[t_key]

    def quantile_at(self, price: float, t: float) -> float:
        """Fast numpy interpolation using precomputed price grid."""
        if len(self._q_grid) < 2:
            return 0.5  # non-quantized model, no inversion possible
        prices = self._get_price_grid(t)
        q = float(np.interp(price, prices, self._q_grid))
        return max(0.001, min(q, 0.999))


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
         "initial": float(p.get("res_short_init", CITADEL["res_short_init"])),
         "rate": float(p.get("res_short_rate", CITADEL["res_short_rate"])),
         "volatility": float(p.get("res_short_vol", CITADEL["res_short_vol"]))},
        {"label": "Medium (T-Notes)",
         "initial": float(p.get("res_med_init", CITADEL["res_med_init"])),
         "rate": float(p.get("res_med_rate", CITADEL["res_med_rate"])),
         "volatility": float(p.get("res_med_vol", CITADEL["res_med_vol"]))},
        {"label": "Long (T-Bonds)",
         "initial": float(p.get("res_long_init", CITADEL["res_long_init"])),
         "rate": float(p.get("res_long_rate", CITADEL["res_long_rate"])),
         "volatility": float(p.get("res_long_vol", CITADEL["res_long_vol"]))},
    ]

    # Investment bins
    invest_bins = [
        {"label": "Equities",
         "initial": float(p.get("inv_eq_init", CITADEL["inv_eq_init"])),
         "return_rate": float(p.get("inv_eq_rate", CITADEL["inv_eq_rate"])),
         "volatility": float(p.get("inv_eq_vol", CITADEL["inv_eq_vol"]))},
        {"label": "Bonds",
         "initial": float(p.get("inv_bd_init", CITADEL["inv_bd_init"])),
         "return_rate": float(p.get("inv_bd_rate", CITADEL["inv_bd_rate"])),
         "volatility": float(p.get("inv_bd_vol", CITADEL["inv_bd_vol"]))},
    ]

    # High-Q trigger
    high_q_split = {
        "cash": float(p.get("high_q_split_cash", CITADEL["high_q_split_cash"])) / 100,
        "res_short": float(p.get("high_q_split_rs", CITADEL["high_q_split_rs"])) / 100,
        "res_med": float(p.get("high_q_split_rm", CITADEL["high_q_split_rm"])) / 100,
        "res_long": float(p.get("high_q_split_rl", CITADEL["high_q_split_rl"])) / 100,
        "inv_eq": float(p.get("high_q_split_eq", CITADEL["high_q_split_eq"])) / 100,
        "inv_bd": float(p.get("high_q_split_bd", CITADEL["high_q_split_bd"])) / 100,
    }
    high_q_action = {
        "mode": p.get("high_q_mode", CITADEL["high_q_mode"]),
        "rate": float(p.get("high_q_rate", CITADEL["high_q_rate"])),
        "duration": int(p.get("high_q_dur", CITADEL["high_q_dur"])),
        "split": high_q_split,
    }

    # Low-Q trigger
    low_q_split = {
        "cash": float(p.get("low_q_split_cash", CITADEL["low_q_split_cash"])) / 100,
        "res_short": float(p.get("low_q_split_rs", CITADEL["low_q_split_rs"])) / 100,
        "res_med": float(p.get("low_q_split_rm", CITADEL["low_q_split_rm"])) / 100,
        "res_long": float(p.get("low_q_split_rl", CITADEL["low_q_split_rl"])) / 100,
        "inv_eq": float(p.get("low_q_split_eq", CITADEL["low_q_split_eq"])) / 100,
        "inv_bd": float(p.get("low_q_split_bd", CITADEL["low_q_split_bd"])) / 100,
    }
    low_q_action = {
        "mode": p.get("low_q_mode", CITADEL["low_q_mode"]),
        "rate": float(p.get("low_q_rate", CITADEL["low_q_rate"])),
        "duration": int(p.get("low_q_dur", CITADEL["low_q_dur"])),
        "split": low_q_split,
    }

    cfg = SimConfig(
        price_model=p.get("price_model", CITADEL["price_model"]),
        start_stack=float(p.get("start_stack", CITADEL["start_stack"])),
        selected_qs=sel_qs,
        cash_initial=float(p.get("cash_initial", CITADEL["cash_initial"])),
        cash_rate=float(p.get("cash_rate", CITADEL["cash_rate"])),
        reserve_bins=reserve_bins,
        invest_bins=invest_bins,
        monthly_spend=float(p.get("monthly_spend", CITADEL["monthly_spend"])),
        inflation=float(p.get("inflation", CITADEL["inflation"])),
        spend_growth=float(p.get("spend_growth", CITADEL["spend_growth"])),
        high_q_trigger=float(p.get("high_q_trigger", CITADEL["high_q_trigger"])) / 100,
        high_q_action=high_q_action,
        low_q_trigger=float(p.get("low_q_trigger", CITADEL["low_q_trigger"])) / 100,
        low_q_action=low_q_action,
        lump_cooldown=int(p.get("lump_cooldown", CITADEL["lump_cooldown"])),
        cash_floor=float(p.get("cash_floor", CITADEL["cash_floor"])),
        cash_floor_growth=float(p.get("cash_floor_growth", CITADEL["cash_floor_growth"])),
        reserve_floors=[
            float(p.get("res_short_floor", CITADEL["res_short_floor"])),
            float(p.get("res_med_floor", CITADEL["res_med_floor"])),
            float(p.get("res_long_floor", CITADEL["res_long_floor"])),
        ],
        reserve_floor_growth=float(p.get("reserve_floor_growth", CITADEL["reserve_floor_growth"])),
        scf_enabled=bool(p.get("scf_enabled")),
        scf_amount=float(p.get("scf_amount", CITADEL["scf_amount"])),
        scf_type=p.get("scf_type", CITADEL["scf_type"]),
        scf_rate=float(p.get("scf_rate", CITADEL["scf_rate"])),
        scf_term=int(p.get("scf_term", CITADEL["scf_term"])),
        scf_repay_trigger=float(p.get("scf_repay_trigger", CITADEL["scf_repay_trigger"])),
        start_yr=int(p.get("start_yr", CITADEL["start_yr"])),
        end_yr=int(p.get("end_yr", CITADEL["end_yr"])),
        freq=p.get("freq", CITADEL["freq"]),
        n_sims=int(p.get("n_sims", 1)),
        tax_rate=0.0,
        asset_return_model=p.get("asset_return_model", CITADEL["asset_return_model"]),
    )

    # Load asset transition matrices when Markov mode selected
    if cfg.asset_return_model == "markov":
        try:
            from data.asset_matrices import load_asset_matrices
            cfg.asset_matrices = load_asset_matrices(n_bins=5)
        except Exception:
            cfg.asset_return_model = "lognormal"  # fallback

    return cfg


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
    model = _ModelAdapter(m, model_key=model_key, user_model=p.get("user_model"))

    # Deterministic always runs with n_sims=1; MC overlay runs separately
    import time as _time
    print(f"[CITADEL] build: model={model_key}, n_sims={p.get('n_sims', '?')} (forced→1), "
          f"qs={config.selected_qs}, yr={config.start_yr}-{config.end_yr}, freq={config.freq}",
          flush=True)
    config.n_sims = 1

    _t0 = _time.time()
    try:
        result = simulate(config, model)
    except (ValueError, NotImplementedError) as e:
        return _error_figure(m, f"Simulation error: {e}"), None
    print(f"[CITADEL] simulate: {(_time.time()-_t0)*1000:.1f}ms (n_sims=1, {len(result.time_axis)} periods)",
          flush=True)

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
    mc_pending = False
    try:
        from mc_overlay import _mc_citadel_overlay, _HAS_MARKOV
        if _HAS_MARKOV and p.get("mc_enabled"):
            mc_traces, mc_result = _mc_citadel_overlay(m, p, config, model)
            if mc_result and isinstance(mc_result, dict) and mc_result.get("_pending"):
                mc_pending = True  # Celery task submitted, not done yet
            elif mc_traces:
                traces.extend(mc_traces)
    except ImportError:
        pass

    q_label = f"Q{config.selected_qs[0]*100:g}%" if config.selected_qs else "Q25%"
    ylabel = "BTC Remaining" if disp_mode == "btc" else "USD Value"
    title = f"Citadel Planner \u2014 {fmt_price(config.monthly_spend)}/mo \u00b7 {q_label}"
    print(f"[CITADEL] title: mc_enabled={p.get('mc_enabled')}, mc_result={type(mc_result).__name__}({bool(mc_result)}), mc_pending={mc_pending}", flush=True)
    if p.get("mc_enabled") and mc_result and not mc_pending:
        mc_eq = p.get("mc_entry_q", "")
        # Show actual sim count from result, not requested count
        actual_sims = mc_result.get("n_sims", p.get("mc_sims", "?")) if isinstance(mc_result, dict) else p.get("mc_sims", "?")
        title += f"  \u00b7  MC entry Q{mc_eq}% ({actual_sims} sims)"
    if mc_pending:
        title += "  \u00b7  MC computing..."

    layout, _x_end = _sim_layout(m, p, title, ylabel, ts, t_start, t_end, dt, syr, eyr)
    layout["annotations"] = deplete_annots

    _stagger_depletion_annots(deplete_annots, layout)

    return _finalize_chart(traces, layout, p, "cp", mc_result)
