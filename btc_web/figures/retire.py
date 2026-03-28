"""Bitcoin Retireator chart builder."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from typing import Any

import _app_ctx
from btc_core import ModelData, yr_to_t, fmt_price
from mc_overlay import _mc_retire_overlay
from tab_defaults import RETIRE

from figures.common import (
    _QR_LINE_WIDTH, _ANNOT_STAGGER_Y,
    _NON_QUANTIZED_MODEL_COLOR, _OVERLAY_LINE_WIDTH,
    _FONT_ANNOT,
    _HAS_MARKOV,
    _get_palette, _build_thermal_colors, _fmt_q_label,
    _build_time_array, _get_starting_stack,
    _sim_layout, _apply_mc_overlay,
    _stagger_depletion_annots,
    _finalize_chart, _fmt_short, _mc_median_annot,
    _resolve_edge_annotations,
    build_overlay_traces,
)


def build_retire_figure(m: ModelData, p: dict[str, Any]) -> tuple[go.Figure, dict | None]:
    """
    p keys: start_yr, end_yr, start_stack, wd_amount, freq, inflation,
            disp_mode, selected_qs, log_y, annotate,
            lots, use_lots
    """
    model = _app_ctx.DEFAULT_MODEL
    palette = _get_palette(p)
    _line_shape = "hv" if p.get("discrete") else "linear"
    sel_qs_raw = sorted([float(q) for q in (p.get("selected_qs") or [])])
    _thermal = _build_thermal_colors(sel_qs_raw, palette)
    ta = _build_time_array(p, m, 2025, 2045)
    if ta[1] is None:
        return ta[0], None
    syr, eyr, t_start, t_end, ts, dt, freq_str, ppy = ta

    start_stack = _get_starting_stack(p, default=1.0)

    wd_amount = float(p.get("wd_amount", RETIRE["wd_amount"]))
    inflation = float(p.get("inflation", RETIRE["inflation"])) / 100.0
    disp_mode = p.get("disp_mode", "btc")
    sel_qs    = sorted([float(q) for q in (p.get("selected_qs") or [])])

    traces   = []
    deplete_annots = []
    all_btc_vals = {}  # q -> BTC balance array
    all_y_vals   = {}  # q -> plotted y-values (for text trace annotations)
    all_prices   = {}  # q -> price array — reused by annotation loop

    ts_clamped = np.maximum(ts, 0.5)
    adj_wd_arr = wd_amount * ((1 + inflation) ** (ts - t_start))
    for q in sel_qs:
        if q not in model.fits:
            continue
        prices = model.price_at(q, ts_clamped)
        vals = np.maximum(start_stack - np.cumsum(adj_wd_arr / prices), 0.0)
        all_btc_vals[q] = vals
        all_prices[q]   = prices

        if disp_mode == "usd":
            y_vals = vals * prices
            final_lbl = fmt_price(float(y_vals[-1]))
        else:
            y_vals = vals
            final_usd = fmt_price(float(vals[-1]) * float(prices[-1]))
            final_lbl = f"{float(vals[-1]):.4f} BTC  ({final_usd})"
        all_y_vals[q] = y_vals

        lbl = f"{model.legend_name} {_fmt_q_label(q)}" + f"  \u2192  {final_lbl}"
        col = _thermal.get(q, model.colors.get(q, "#888888"))
        traces.append(go.Scatter(
            x=list(ts), y=list(y_vals), mode="lines", name=lbl,
            line=dict(color=col, width=_QR_LINE_WIDTH, shape=_line_shape),
        ))

        # depletion annotation — always shown
        depl_i = next((i for i, v in enumerate(vals) if v <= 0), None)
        if depl_i is not None:
            depl_t = ts[depl_i]
            depl_yr = int(syr + (depl_t - t_start) * (eyr - syr) / max(t_end - t_start, 1e-6))
            _ay = _ANNOT_STAGGER_Y[len(deplete_annots) % 3]
            deplete_annots.append(dict(
                x=depl_t, xref="x",
                y=0, yref="paper",
                ax=28, ay=_ay,
                text=f"\u2248{depl_yr}",
                showarrow=True, arrowhead=2, arrowsize=1,
                arrowcolor=col,
                font=dict(size=_FONT_ANNOT, color=col),
            ))

    # ── alternative model overlays ────────────────────────────────────────────
    _ret_sim = lambda prices: np.maximum(start_stack - np.cumsum(adj_wd_arr / prices), 0.0)
    traces.extend(build_overlay_traces(
        p, ts, ts_clamped, sel_qs, disp_mode, palette, _ret_sim,
        line_shape=_line_shape,
    ))

    shapes = []

    ylabel = "USD Value" if disp_mode == "usd" else "BTC Remaining"
    title = f"Bitcoin Retireator \u2014 {fmt_price(wd_amount)}/{freq_str.lower()[:-2] if freq_str.endswith('ly') else freq_str}"
    layout, _x_end = _sim_layout(m, p, title, ylabel, ts, t_start, t_end, dt, syr, eyr, shapes)
    layout["annotations"] = deplete_annots

    # ── Monte Carlo fan overlay ─────────────────────────────────────────────
    mc_traces_list = []
    mc_result = None
    if _HAS_MARKOV and p.get("mc_enabled"):
        mc_traces_list, mc_result = _apply_mc_overlay(
            m, p, _mc_retire_overlay,
            (m, p, ts, t_start, t_end, dt, start_stack, disp_mode, len(deplete_annots)),
            traces, deplete_annots, layout, _x_end, disp_mode)

    _stagger_depletion_annots(deplete_annots, layout)

    # ── Right-edge annotations (text traces for alignment stability) ─────────
    _is_log = bool(p.get("log_y"))
    _pending_annots = []
    if p.get("annotate") and all_y_vals:
        for q in sel_qs:
            if q not in all_y_vals:
                continue
            btc_final = float(all_btc_vals[q][-1])
            if btc_final <= 0:
                continue
            col = _thermal.get(q, model.colors.get(q, "#888888"))
            usd_final = btc_final * float(all_prices[q][-1])
            qpfx = f"Q{q*100:g}%"
            if disp_mode == "usd":
                lbl = f"{qpfx} {fmt_price(usd_final)}"
            else:
                lbl = f"{qpfx} {btc_final:.2f} \u20bf  {fmt_price(usd_final)}"
            short = _fmt_short(btc_final, usd_final)
            _pending_annots.append(dict(
                x_arr=ts, y_arr=all_y_vals[q],
                label=lbl, short_label=short,
                color=col, y_last=float(all_y_vals[q][-1])))

        # MC median endpoint
        _mc_ann = _mc_median_annot(mc_traces_list, disp_mode, m,
                                   float(ts[-1]), t_start, t_end, syr, eyr)
        if _mc_ann:
            _pending_annots.append(_mc_ann)
    traces.extend(_resolve_edge_annotations(_pending_annots, _is_log))

    return _finalize_chart(traces, layout, p, "ret", mc_result)
