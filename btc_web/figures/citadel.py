"""Citadel Planner chart builder — multi-line portfolio visualization."""

from __future__ import annotations

import logging
import numpy as np
import plotly.graph_objects as go
from typing import Any

import _app_ctx

logger = logging.getLogger(__name__)
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


def _build_band_traces(bands, time_axis, series_key="total", color="#000000",
                       name_prefix="MC spread"):
    """Build shaded band traces from percentile band data.

    Returns 4 traces: P5-P95 (light fill) lower/upper + P25-P75 (dark fill) lower/upper.
    """
    if not bands or not time_axis:
        return []

    def _hex_alpha(hex_color, alpha):
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        return f"rgba({r},{g},{b},{alpha})"

    # Keys may be string (from JSON store) or int
    def _get(pct):
        return bands.get(pct) or bands.get(str(pct)) or {}

    p5 = _get(5).get(series_key, [])
    p25 = _get(25).get(series_key, [])
    p75 = _get(75).get(series_key, [])
    p95 = _get(95).get(series_key, [])

    if not p5 or not p95:
        return []

    x = list(time_axis)

    traces = []
    # P5-P95 band (light fill, opacity 0.15)
    traces.append(go.Scatter(
        x=x, y=list(p5), mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip",
    ))
    traces.append(go.Scatter(
        x=x, y=list(p95), mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor=_hex_alpha(color, 0.15),
        name=f"{name_prefix} (P5\u2013P95)",
        legendgroup="mc-bands",
    ))
    # P25-P75 band (dark fill, opacity 0.30)
    traces.append(go.Scatter(
        x=x, y=list(p25), mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip",
    ))
    traces.append(go.Scatter(
        x=x, y=list(p75), mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor=_hex_alpha(color, 0.30),
        name=f"{name_prefix} (P25\u2013P75)",
        legendgroup="mc-bands",
    ))
    return traces


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
    eq_init = float(p.get("inv_eq_init", CITADEL["inv_eq_init"]))
    bd_init = float(p.get("inv_bd_init", CITADEL["inv_bd_init"]))
    invest_bins = [
        {"label": "Equities",
         "initial": eq_init,
         "return_rate": float(p.get("inv_eq_rate", CITADEL["inv_eq_rate"])),
         "volatility": float(p.get("inv_eq_vol", CITADEL["inv_eq_vol"]))},
        {"label": "Bonds",
         "initial": bd_init,
         "return_rate": float(p.get("inv_bd_rate", CITADEL["inv_bd_rate"])),
         "volatility": float(p.get("inv_bd_vol", CITADEL["inv_bd_vol"]))},
    ]

    # Investment cost basis (what user originally paid; defaults to initial value)
    eq_basis = float(p.get("inv_eq_basis", eq_init))
    bd_basis = float(p.get("inv_bd_basis", bd_init))
    invest_cost_basis = [eq_basis, bd_basis]

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
        invest_cost_basis_initial=invest_cost_basis,
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
        # Tax configuration
        tax_enabled=bool(p.get("tax_enabled", False)),
        filing_status=str(p.get("filing_status", CITADEL["filing_status"])),
        state_code=str(p.get("state_code", CITADEL["state_code"])),
        state_rate_override=p.get("state_rate_override"),
        tcja_sunset=bool(p.get("tcja_sunset", False)),
        birth_year=p.get("birth_year"),
        cost_basis_method=str(p.get("cost_basis_method", CITADEL["cost_basis_method"])),
        other_income=float(p.get("other_income") or 0),
        other_income_growth=float(p.get("other_income_growth") or 0),
        td_btc_stack=float(p.get("td_btc") or 0),
        td_cash_initial=float(p.get("td_cash") or 0),
        td_reserve_bins=[
            {"label": "Short", "initial": float(p.get("td_res_short") or 0)},
            {"label": "Medium", "initial": float(p.get("td_res_med") or 0)},
            {"label": "Long", "initial": float(p.get("td_res_long") or 0)},
        ],
        td_invest_bins=[
            {"label": "Equities", "initial": float(p.get("td_inv_eq") or 0)},
            {"label": "Bonds", "initial": float(p.get("td_inv_bd") or 0)},
        ],
        tf_btc_stack=float(p.get("tf_btc") or 0),
        tf_cash_initial=float(p.get("tf_cash") or 0),
        tf_reserve_bins=[
            {"label": "Short", "initial": float(p.get("tf_res_short") or 0)},
            {"label": "Medium", "initial": float(p.get("tf_res_med") or 0)},
            {"label": "Long", "initial": float(p.get("tf_res_long") or 0)},
        ],
        tf_invest_bins=[
            {"label": "Equities", "initial": float(p.get("tf_inv_eq") or 0)},
            {"label": "Bonds", "initial": float(p.get("tf_inv_bd") or 0)},
        ],
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
        return _error_figure(f"Config error: {e}"), None

    model_key = p.get("price_model", "bub")
    model = _ModelAdapter(m, model_key=model_key, user_model=p.get("user_model"))

    # Deterministic always runs with n_sims=1; MC overlay runs separately
    import time as _time
    logger.debug("[CITADEL] build: model=%s, n_sims=%s (forced->1), qs=%s, yr=%s-%s, freq=%s",
                 model_key, p.get('n_sims', '?'), config.selected_qs,
                 config.start_yr, config.end_yr, config.freq)
    config.n_sims = 1

    _t0 = _time.time()
    try:
        result = simulate(config, model)
    except (ValueError, NotImplementedError) as e:
        return _error_figure(f"Simulation error: {e}"), None
    logger.debug("[CITADEL] simulate: %.1fms (n_sims=1, %d periods)",
                 (_time.time()-_t0)*1000, len(result.time_axis))

    # Extract time axis and median data
    ts = result.time_axis
    med = result.median
    n_periods = len(ts)
    if n_periods == 0:
        return _error_figure("No simulation periods"), None

    # Spending per period (cumulative -> per-period via diff)
    cum_spend = result.cumulative_spend[0]  # sim 0 (deterministic)
    period_spend = np.diff(cum_spend, prepend=0)

    # Build traces — tag with quantile so deterministic is distinguishable from MC
    traces = []
    _q_str = f"Q{config.selected_qs[0]*100:g}%" if config.selected_qs else "Q25%"
    _qtag = f" ({_q_str})" if p.get("mc_enabled") else ""

    if disp_mode in ("usd_total", "usd_per_asset"):
        total_y = med["total"]
        traces.append(go.Scatter(
            x=list(ts), y=list(total_y), mode="lines",
            name=f"Total Portfolio{_qtag}  \u2192  {fmt_price(float(total_y[-1]))}",
            line=dict(color=_C_TOTAL, width=_QR_LINE_WIDTH + 0.5),
        ))

    if disp_mode == "usd_per_asset":
        btc_usd = med["btc_usd"]
        traces.append(go.Scatter(
            x=list(ts), y=list(btc_usd), mode="lines",
            name=f"BTC Holdings{_qtag}  \u2192  {fmt_price(float(btc_usd[-1]))}",
            line=dict(color=_C_BTC, width=_QR_LINE_WIDTH),
        ))

        cash_y = med["cash"]
        traces.append(go.Scatter(
            x=list(ts), y=list(cash_y), mode="lines",
            name=f"Cash{_qtag}  \u2192  {fmt_price(float(cash_y[-1]))}",
            line=dict(color=_C_CASH, width=_QR_LINE_WIDTH, dash="dash"),
        ))

        res_y = med["reserves_total"]
        traces.append(go.Scatter(
            x=list(ts), y=list(res_y), mode="lines",
            name=f"Reserves{_qtag}  \u2192  {fmt_price(float(res_y[-1]))}",
            line=dict(color=_C_RESERVES, width=_QR_LINE_WIDTH),
        ))

        inv_y = med["investments_total"]
        traces.append(go.Scatter(
            x=list(ts), y=list(inv_y), mode="lines",
            name=f"Investments{_qtag}  \u2192  {fmt_price(float(inv_y[-1]))}",
            line=dict(color=_C_INVEST, width=_QR_LINE_WIDTH),
        ))

        traces.append(go.Scatter(
            x=list(ts), y=list(period_spend), mode="lines",
            name=f"Spending/period  \u2192  {fmt_price(float(period_spend[-1]))}",
            line=dict(color=_C_SPEND, width=_QR_LINE_WIDTH * 0.7, dash="dot"),
        ))

        # TD/TF wrapper traces (only when tax is on)
        if p.get("tax_enabled") and result.td_total is not None:
            td_y = np.median(result.td_total, axis=0)
            traces.append(go.Scatter(
                x=list(ts), y=list(td_y), mode="lines",
                name=f"Tax-Deferred  \u2192  {fmt_price(float(td_y[-1]))}",
                line=dict(color="#8B4513", width=_QR_LINE_WIDTH, dash="dashdot"),
            ))
        if p.get("tax_enabled") and result.tf_total is not None:
            tf_y = np.median(result.tf_total, axis=0)
            traces.append(go.Scatter(
                x=list(ts), y=list(tf_y), mode="lines",
                name=f"Tax-Free (Roth)  \u2192  {fmt_price(float(tf_y[-1]))}",
                line=dict(color="#228B22", width=_QR_LINE_WIDTH, dash="dashdot"),
            ))

    elif disp_mode == "btc":
        btc_stack = result.btc_holdings[0]  # sim 0
        traces.append(go.Scatter(
            x=list(ts), y=list(btc_stack), mode="lines",
            name=f"BTC Stack{_qtag}  \u2192  {float(btc_stack[-1]):.4f} BTC",
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

    # Quick Scenario bands from cached presets
    scenario_bands = p.get("scenario_bands")
    if scenario_bands:
        _band_series = "btc_stack" if disp_mode == "btc" else "total"
        _band_color = "#F7931A" if disp_mode == "btc" else "#000000"
        band_traces = _build_band_traces(
            scenario_bands, ts.tolist(),
            series_key=_band_series, color=_band_color)
        traces.extend(band_traces)

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
    logger.debug("[CITADEL] title: mc_enabled=%s, mc_result=%s(%s), mc_pending=%s",
                 p.get('mc_enabled'), type(mc_result).__name__, bool(mc_result), mc_pending)
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

    # ── Tax ghost traces + annotations ────────────────────────────────────────
    extra_tax = {}
    if p.get("tax_enabled"):
        # Run a parallel no-tax simulation for comparison.
        # Fold TD/TF balances into taxable accounts so starting capital
        # is identical — the difference is purely tax drag.
        try:
            p_notax = {**p, "tax_enabled": False}
            cfg_notax = _build_sim_config(p_notax)
            cfg_notax.start_stack += cfg_notax.td_btc_stack + cfg_notax.tf_btc_stack
            cfg_notax.cash_initial += cfg_notax.td_cash_initial + cfg_notax.tf_cash_initial
            for i, rb in enumerate(cfg_notax.reserve_bins):
                rb["initial"] += (cfg_notax.td_reserve_bins[i]["initial"]
                                  + cfg_notax.tf_reserve_bins[i]["initial"])
            for i, ib in enumerate(cfg_notax.invest_bins):
                ib["initial"] += (cfg_notax.td_invest_bins[i]["initial"]
                                  + cfg_notax.tf_invest_bins[i]["initial"])
            cfg_notax.n_sims = 1
            result_notax = simulate(cfg_notax, model)
        except Exception:
            result_notax = None

        if result_notax is not None and len(result_notax.time_axis) > 0:
            # Ghost "no-tax" total portfolio trace
            notax_total = result_notax.median["total"]
            traces.append(go.Scatter(
                x=list(ts), y=list(notax_total), mode="lines",
                name=f"Total Portfolio (no tax){_qtag}",
                line=dict(dash="dash", color="rgba(100,100,100,0.5)"),
                showlegend=True,
            ))

            # Ghost "no-tax" BTC holdings trace (per-asset mode only)
            if disp_mode == "usd_per_asset":
                notax_btc = result_notax.median["btc_usd"]
                traces.append(go.Scatter(
                    x=list(ts), y=list(notax_btc), mode="lines",
                    name=f"BTC Holdings (no tax){_qtag}",
                    line=dict(dash="dash", color="rgba(247,147,26,0.5)"),
                    showlegend=True,
                ))

            # Tax-drag annotation at the final period
            final_tax = float(med["total"][-1])
            final_notax = float(notax_total[-1])
            drag = final_notax - final_tax
            drag_pct = (drag / final_notax * 100) if final_notax > 0 else 0
            if drag > 0:
                deplete_annots.append(dict(
                    x=ts[-1], xref="x",
                    y=final_tax, yref="y",
                    ax=0, ay=-40,
                    text=f"Tax drag: \u2212{fmt_price(drag)} ({drag_pct:.1f}%)",
                    showarrow=True, arrowhead=2, arrowsize=1,
                    arrowcolor="#E74C3C",
                    font=dict(size=_FONT_ANNOT, color="#E74C3C"),
                ))

        # Append tax info to chart title
        state = p.get("state_code", "TX")
        title += f"  (Federal + {state} Tax)"
        layout["title"]["text"] = title

        # Cumulative "Taxes Paid" trace
        taxes_paid = getattr(result, 'taxes_paid', None)
        if taxes_paid is not None:
            traces.append(go.Scatter(
                x=list(ts), y=list(taxes_paid[0]),
                name=f"Cumulative Taxes Paid  \u2192  {fmt_price(float(taxes_paid[0, -1]))}",
                fill="tozeroy",
                line=dict(color="rgba(220,50,50,0.6)"),
            ))

        # Store annual tax data in extra dict
        annual_taxes = getattr(result, 'annual_taxes', None)
        if annual_taxes is not None:
            extra_tax["annual_taxes"] = annual_taxes

    fig, extra = _finalize_chart(traces, layout, p, "cp", mc_result)

    # Merge tax data into extra dict
    if extra_tax:
        if extra is None:
            extra = extra_tax
        else:
            extra.update(extra_tax)

    return fig, extra
