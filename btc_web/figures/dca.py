"""DCA Accumulator + Stack-celerator chart builder."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from typing import Any

import _app_ctx
from btc_core import ModelData, yr_to_t, fmt_price
from tab_defaults import DCA

from figures.common import (
    _QR_LINE_WIDTH, _BTC_ORANGE,
    _NON_QUANTIZED_MODEL_COLOR, _OVERLAY_LINE_WIDTH,
    _FONT_ANNOT,
    _HAS_MARKOV,
    _get_palette, _get_model_color, _fmt_q_label, _error_figure,
    _build_freq_config, _build_time_array, _get_starting_stack,
    _sim_layout, _apply_mc_overlay,
    _finalize_chart, _fmt_short, _find_mc_median_trace,
    _resolve_edge_annotations,
    _post_mc_overlay,
    _mc_dca_overlay,
    build_overlay_traces,
    _build_symmetric_bands,

    FREQ_PPY,
)


def _dca_sc_overlay(m, p, ts, sel_qs, start_stack, all_prices, disp_mode, ppy, line_shape="linear"):
    """Run Stack-celerator overlay simulation for DCA tab.

    Returns (sc_traces, all_sc_usd_vals, all_sc_btc_vals).

    SC traces render in the BM model color with opacity varying by
    distance from Q50% — matching the BM lines themselves and every
    other overlay model. Visually differentiated by their dashed style.
    """
    model = _app_ctx.DEFAULT_MODEL
    from _app_ctx import _compute_sc_loan
    _bm_color = _get_model_color("bub", p)

    principal    = float(p.get("sc_loan_amount", DCA["sc_loan_amount"]))
    sc_rate      = float(p.get("sc_rate", DCA["sc_rate"]))
    sc_live      = float(p.get("sc_live_price", 0))
    loan_type    = p.get("sc_loan_type", "interest_only")
    term_months  = float(p.get("sc_term_months", DCA["sc_term_months"]))
    sc_repeats   = int(p.get("sc_repeats", 0))
    entry_mode   = p.get("sc_entry_mode", "live")
    custom_price = float(p.get("sc_custom_price", DCA["sc_custom_price"]))
    tax_rate     = max(0.0, min(float(p.get("sc_tax_rate", DCA["sc_tax_rate"])), 0.9999))
    sc_rollover  = bool(p.get("sc_rollover", False)) and loan_type == "interest_only"
    amount       = float(p.get("amount", DCA["amount"]))

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
        if q not in model.fits:
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

                # ── Loan repayment logic ──────────────────────────────────
                # Amortizing: each period pays interest + principal in fiat
                #   (no BTC sold, tax has no effect on amortizing loans).
                # Interest-only: at cycle end, sell BTC to repay principal.
                #   Tax applies ONLY to capital gain (sell_price - buy_price),
                #   not the full proceeds. If selling at a loss (price <= ep),
                #   no tax is owed. Rollover defers all repayment to sim end.
                if loan_type == "amortizing":
                    interest_p  = outstanding * r
                    principal_p = pmt - interest_p
                    outstanding = max(outstanding - principal_p, 0.0)

                if loan_type == "interest_only" and period_in_cycle == term_periods - 1:
                    if sc_rollover:
                        pass
                    else:
                        # Sell BTC to repay principal. Tax is on capital gain
                        # only: (sell_price - buy_price) per BTC. net_per_btc
                        # is what you keep per BTC sold after tax on the gain.
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

        lbl_sc = f"{model.legend_name} SC {_fmt_q_label(q)}" + f"  \u2192  {final_sc}"
        _dist = abs(q - 0.5) / 0.45
        _q_opacity = max(0.1, 1.0 - _dist * 0.5)
        sc_traces.append(go.Scatter(
            x=list(ts), y=list(y_sc), mode="lines", name=lbl_sc,
            line=dict(color=_bm_color, width=_QR_LINE_WIDTH, dash="dash", shape=line_shape),
            opacity=_q_opacity,
        ))

    return sc_traces, all_sc_usd_vals, all_sc_btc_vals


def build_dca_figure(m: ModelData, p: dict[str, Any]) -> tuple[go.Figure, dict | None]:
    """
    p keys: start_yr, end_yr, start_stack, amount, freq, disp_mode,
            selected_qs, log_y, show_today,
            lots, use_lots
    """
    model = _app_ctx.DEFAULT_MODEL
    palette = _get_palette(p)
    _line_shape = "hv" if p.get("discrete") else "linear"
    sel_qs_raw = sorted([float(q) for q in (p.get("selected_qs") or [])])
    ta = _build_time_array(p, m, 2024, 2035)
    if ta[1] is None:
        return ta[0], None
    syr, eyr, t_start, t_end, ts, dt, freq_str, ppy = ta

    start_stack = _get_starting_stack(p, default=0)

    amount    = float(p.get("amount", DCA["amount"]))
    inflation = float(p.get("inflation", DCA["inflation"])) / 100.0
    disp_mode = p.get("disp_mode", "btc")
    sel_qs    = sorted([float(q) for q in (p.get("selected_qs") or [])])
    show_bm   = "bub" in (p.get("active_models") or ["bub"])

    traces = []
    all_btc_vals = {}  # q -> BTC balance array
    all_usd_vals = {}  # q -> USD value array (for annotations + title)
    all_prices   = {}  # q -> price array — reused by SC loop to avoid redundant qr_price calls
    _bm_line_traces = []  # collected first so bands can render beneath them
    _y_for_bands = {}     # q -> y array in current disp_mode (for shading)
    ts_clamped = np.maximum(ts, 0.5)
    adj_amount_arr = amount * ((1 + inflation) ** (ts - t_start))
    # BM lines on DCA use a single model color (like every other overlay
    # model), varying only opacity with distance from Q50%. The old
    # thermal-per-quantile mapping read as a rainbow on stack-domain
    # charts because Q10% (best outcome for accumulation) ended up cyan
    # at the top of the chart and Q99% (worst) ended up red at the
    # bottom — visually inverted from how bubble tab reads.
    _bm_color = _get_model_color("bub", p)
    for q in sel_qs:
        if q not in model.fits:
            continue
        prices_q = model.price_at(q, ts_clamped)
        vals = start_stack + np.cumsum(adj_amount_arr / prices_q)
        all_btc_vals[q] = vals
        all_usd_vals[q] = vals * prices_q
        all_prices[q]   = prices_q          # save for SC loop below

        if show_bm:
            if disp_mode == "usd":
                y_vals    = vals * prices_q
                final_lbl = fmt_price(float(y_vals[-1]))
            else:
                y_vals    = vals
                final_usd = fmt_price(float(all_usd_vals[q][-1]))
                final_lbl = f"{float(vals[-1]):.4f} BTC  ({final_usd})"

            _y_for_bands[q] = y_vals
            lbl = f"{model.legend_name} {_fmt_q_label(q)}" + f"  \u2192  {final_lbl}"
            _dist = abs(q - 0.5) / 0.45
            _q_opacity = max(0.1, 1.0 - _dist * 0.5)
            _bm_line_traces.append(go.Scatter(
                x=list(ts), y=list(y_vals), mode="lines", name=lbl,
                line=dict(color=_bm_color, width=_QR_LINE_WIDTH, shape=_line_shape),
                opacity=_q_opacity,
            ))

    # ── Symmetric band shading (added before line traces so lines render on top) ──
    if show_bm and p.get("shade") and len(_y_for_bands) >= 2:
        _bm_color = _get_model_color("bub", p)
        traces.extend(_build_symmetric_bands(
            sorted(_y_for_bands.keys()), _y_for_bands, ts, model_color=_bm_color))
    traces.extend(_bm_line_traces)

    # ── alternative model overlays ────────────────────────────────────────────
    _dca_sim = lambda prices_q: start_stack + np.cumsum(adj_amount_arr / prices_q)
    traces.extend(build_overlay_traces(
        p, ts, ts_clamped, sel_qs, disp_mode, palette, _dca_sim,
        line_shape=_line_shape,
    ))

    shapes = []

    # ── Total cost & value ratio ────────────────────────────────────────────
    n_periods = len(ts)
    total_spent = amount * n_periods
    freq_short = _app_ctx.FREQ_LABEL.get(freq_str, freq_str)

    # Build title — add cost info and median ROI if we have quantile data
    title_line = f"Bitcoin DCA \u2014 {fmt_price(amount)}/{freq_short}"
    title_line += f"  \u00b7  {fmt_price(total_spent)} invested over {n_periods} periods"
    _qr_med_final = None
    if all_usd_vals:
        _qr_med_final = float(np.median([v[-1] for v in all_usd_vals.values()]))
        roi = _qr_med_final / total_spent if total_spent > 0 else 0
        title_line += f"<br>QR median {fmt_price(_qr_med_final)}  \u00b7  {roi:.1f}\u00d7"

    ylabel = ""
    layout, _x_end = _sim_layout(m, p, title_line, ylabel, ts, t_start, t_end, dt, syr, eyr, shapes)

    # ── Stack-celerator overlay ─────────────────────────────────────────────
    all_sc_usd_vals = {}
    all_sc_btc_vals = {}
    if p.get("sc_enabled") and sel_qs:
        sc_traces, all_sc_usd_vals, all_sc_btc_vals = _dca_sc_overlay(
            m, p, ts, sel_qs, start_stack, all_prices, disp_mode, ppy, line_shape=_line_shape)
        traces.extend(sc_traces)

    # ── SC factor (ratio of median SC to median DCA at end date) ─────────────
    sc_factor_val = None
    if all_sc_usd_vals and all_usd_vals:
        _sc_end  = float(np.median([v[-1] for v in all_sc_usd_vals.values()]))
        _dca_end = float(np.median([v[-1] for v in all_usd_vals.values()]))
        if _dca_end > 0:
            sc_factor_val = _sc_end / _dca_end

    # ── Stack-celeration factor -> append to title ────────────────────────────
    if sc_factor_val is not None:
        layout["title"]["text"] += (
            f"<br><b>Stack-celeration: {sc_factor_val:.2f}\u00d7</b>"
        )

    # ── Monte Carlo fan overlay ─────────────────────────────────────────────
    _is_log = bool(p.get("log_y"))
    mc_result = None
    mc_fan_usd = {}
    mc_traces = []
    if _HAS_MARKOV and p.get("mc_enabled"):
        mc_traces, mc_result, mc_fan_usd = _mc_dca_overlay(m, p, ts, t_start, dt, start_stack, disp_mode)
        mc_traces = _post_mc_overlay(mc_traces, p, _x_end, disp_mode)
        traces.extend(mc_traces)
        # MC median text trace annotation — collected into pending below
        # MC median final value + multiplier -> append to title
        if mc_fan_usd and 0.50 in mc_fan_usd and len(mc_fan_usd[0.50]) > 0:
            mc_med_final = float(mc_fan_usd[0.50][-1])
            mc_roi = mc_med_final / total_spent if total_spent > 0 else 0
            layout["title"]["text"] += f"  \u00b7  MC median {fmt_price(mc_med_final)}  \u00b7  {mc_roi:.1f}\u00d7"

    # ── Right-edge annotations (text traces for alignment stability) ─────────
    _pending_annots = []
    if p.get("annotate") and all_usd_vals:
        for q in sel_qs:
            if q not in all_usd_vals:
                continue
            col = _bm_color  # matches the trace line for a unified look
            y_arr = all_btc_vals[q] if disp_mode == "btc" else all_usd_vals[q]
            _btc_f = float(all_btc_vals[q][-1])
            _usd_f = float(all_usd_vals[q][-1])
            _pending_annots.append(dict(
                x_arr=ts, y_arr=y_arr,
                label=f"Q{q*100:g}% {fmt_price(_usd_f)}",
                short_label=_fmt_short(_btc_f, _usd_f),
                color=col, y_last=float(y_arr[-1])))
        for q in all_sc_usd_vals:
            col = _bm_color  # SC annotations match SC trace line
            sc_y = all_sc_btc_vals[q] if disp_mode == "btc" else all_sc_usd_vals[q]
            _btc_f = float(all_sc_btc_vals[q][-1])
            _usd_f = float(all_sc_usd_vals[q][-1])
            _pending_annots.append(dict(
                x_arr=ts, y_arr=sc_y,
                label=f"SC Q{q*100:g}% {fmt_price(_usd_f)}",
                short_label=_fmt_short(_btc_f, _usd_f),
                color=col, y_last=float(sc_y[-1])))
    if p.get("annotate") and mc_fan_usd and 0.50 in mc_fan_usd:
        mc_med_usd = mc_fan_usd[0.50]
        if len(mc_med_usd) > 0:
            _mx, _my = _find_mc_median_trace(mc_traces)
            if _mx is not None:
                _mc_usd_f = float(mc_med_usd[-1])
                _mc_btc_f = float(_my[-1]) if disp_mode == "btc" else 0
                _pending_annots.append(dict(
                    x_arr=_mx, y_arr=_my,
                    label=f"MC {fmt_price(_mc_usd_f)}",
                    short_label=_fmt_short(_mc_btc_f, _mc_usd_f),
                    color=_BTC_ORANGE, y_last=float(_my[-1])))
    traces.extend(_resolve_edge_annotations(_pending_annots, _is_log))

    # Handle empty plot (all models unchecked)
    if not traces:
        layout["annotations"] = [dict(
            text="No models selected \u2014 check Display Models",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16, color="#888"),
        )]

    return _finalize_chart(traces, layout, p, "dca", mc_result)
