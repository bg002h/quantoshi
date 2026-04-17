"""HODL Supercharger chart builder."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Any

import _app_ctx
import theme
from btc_core import ModelData, yr_to_t, fmt_price
from tab_defaults import SUPERCHARGE
from mc_overlay import _mc_supercharge_overlay
from colors import (FALLBACK_MODEL_GRAY, BLACK, LIGHT_GRAY,
                    SC_ENVELOPE_ALPHA, SC_OVERLAY_ENVELOPE_ALPHA)

from figures.common import (
    TRACE_WIDTH, _ANNOT_STAGGER_Y, _BISECT_ITERS,
    CHART_FONT_ANNOT,
    _HAS_MARKOV,
    _get_palette, _get_model_color, _fmt_q_label, _fmt_q_range, _error_figure,
    _build_freq_config, _get_starting_stack,
    _sim_layout, _apply_mc_overlay,
    _stagger_depletion_annots,
    _base_layout, _finalize_chart,
    _fmt_short, _mc_median_annot,
    _resolve_edge_annotations,
    _hex_alpha,
    _resolve_model,
    quantile_opacity, _parse_quantiles, _empty_state_annotation,
)
from colors import quantile_shade

_DASH_STYLES  = ['solid', 'dash', 'dot', 'dashdot', 'longdash']


def _resolve_sc_model(p):
    """Primary model is always DEFAULT_MODEL (Bubble Model).

    Overlay models from active_models are drawn as additional traces.
    """
    return _app_ctx.DEFAULT_MODEL


def build_supercharge_figure(m: ModelData, p: dict[str, Any]) -> tuple[go.Figure, dict | None]:
    """
    p keys: mode ('a'/'b'), start_stack, start_yr, delays (list), freq, inflation,
            selected_qs, chart_layout (0/1/2), display_q,
            wd_amount (Mode A), end_yr (Mode A), disp_mode (Mode A),
            log_y, annotate, show_legend,
            target_yr (Mode B), lots, use_lots
    """
    model = _resolve_sc_model(p)
    palette = _get_palette(p)
    delay_colors = palette["delay_colors"]
    annot_colors = palette["annot_colors"]
    sel_qs_raw = _parse_quantiles(p)
    _bm_color = _get_model_color("bub", p)

    mode         = p.get("mode", SUPERCHARGE["mode"])
    freq_str, ppy, dt = _build_freq_config(p)
    syr          = int(p.get("start_yr", SUPERCHARGE["start_yr"]))
    inflation    = float(p.get("inflation", SUPERCHARGE["inflation"])) / 100.0
    chart_layout = int(p.get("chart_layout", SUPERCHARGE["chart_layout"]))
    display_q    = float(p.get("display_q", SUPERCHARGE["display_q"]))
    show_legend  = bool(p.get("show_legend", SUPERCHARGE["show_legend"]))

    # Starting stack (lots override)
    start_stack = _get_starting_stack(p, default=1.0)

    # Quantiles
    sel_qs = sorted([float(q) for q in (p.get("selected_qs") or [])
                     if float(q) in model.fits])
    mc_enabled = _HAS_MARKOV and p.get("mc_enabled") and p.get("show_mc", True)
    if not sel_qs and not mc_enabled:
        return go.Figure(layout=dict(
            title="Select at least one quantile",
            paper_bgcolor=theme.PLOT_BG_COLOR,
            font=dict(color=theme.TEXT_COLOR))), None
    if not sel_qs:
        sel_qs = [0.5]  # need at least one for MC simulation grid

    # Delays: filter None/negative, sort, deduplicate
    raw_delays = p.get("delays") or list(SUPERCHARGE["delays"])
    delays = sorted(set(float(d) for d in raw_delays if d is not None and float(d) >= 0))
    if not delays:
        delays = [0.0]

    freq_label = _app_ctx.FREQ_LABEL.get(freq_str, "/mo")

    # ── MODE A: fixed spending -> show how long savings last ───────────────────
    if mode == "a":
        eyr       = int(p.get("end_yr", SUPERCHARGE["end_yr"]))
        wd_amount = float(p.get("wd_amount", SUPERCHARGE["wd_amount"]))
        disp_mode = p.get("disp_mode", "usd")
        t_end     = yr_to_t(eyr, m.genesis)
        _line_shape = "hv" if p.get("discrete") else "linear"

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
                prices = model.price_at(q, ts_d_clamped)
                vals = np.maximum(start_stack - np.cumsum(adj_wd_d / prices), 0.0)
                depl_mask = vals == 0.0
                depl_t = float(ts_d[np.argmax(depl_mask)]) if depl_mask.any() else None
                if disp_mode == "usd":
                    y_vals = vals * prices
                else:
                    y_vals = vals
                results[(d, q)] = (ts_d, y_vals, depl_t, t_start_d, vals, prices)

        traces         = []
        deplete_annots = []

        _AY_LEVELS = _ANNOT_STAGGER_Y

        def _depl_annot(depl_t, t_start_d, d, arrow_col, text_col,
                        legendgroup, model_prefix="", stagger=0):
            depl_yr = int((syr + d) + (depl_t - t_start_d) *
                          (eyr - (syr + d)) / max(t_end - t_start_d, 1e-6))
            prefix = f"{model_prefix} " if model_prefix else ""
            return dict(
                x=depl_t - dt, xref="x",   # last nonzero step, aligns with band end
                y=0, yref="paper",
                ax=28, ay=_AY_LEVELS[stagger % len(_AY_LEVELS)],
                text=f"{prefix}\u2248{depl_yr}",
                showarrow=True, arrowhead=2, arrowsize=1,
                arrowcolor=arrow_col,
                font=dict(size=CHART_FONT_ANNOT, color=text_col),
                name=legendgroup,
            )

        show_bm = "bub" in (p.get("active_models") or [])

        q_range = _fmt_q_range(sel_qs)
        grp_model = f"sc-{model.short_name}"
        _tcol_annot = _app_ctx.MODEL_TRACE_COLORS.get(model.short_name, BLACK)
        _first_legend = True  # only first trace gets showlegend=True

        # Representative terminal value for legend label
        _med_q = sel_qs[len(sel_qs) // 2] if sel_qs else 0.5
        _rep_key = (delays[0], _med_q)
        if _rep_key in results:
            _rep_final = float(results[_rep_key][1][-1])
            _rep_lbl = fmt_price(_rep_final) if disp_mode == "usd" else f"{_rep_final:.4f}"
            _legend_name = f"{model.legend_name} {q_range} \u2192 {_rep_lbl}"
        else:
            _legend_name = f"{model.legend_name} {q_range}"

        def _delay_suffix(d):
            if d == 0:
                return ""
            return f" +{int(d)}yr" if d == int(d) else f" +{d:.1f}yr"

        if not show_bm:
            pass  # skip QR traces, keep results for MC/annotations
        elif chart_layout == 0:
            # Color = delay, show quantile closest to display_q
            q_show = min(sel_qs, key=lambda q: abs(q - display_q))
            for di, d in enumerate(delays):
                key = (d, q_show)
                if key not in results:
                    continue
                ts_d, y_vals, depl_t, t_start_d, *_ = results[key]
                col   = delay_colors[di % len(delay_colors)]
                if disp_mode == "usd":
                    final = fmt_price(float(y_vals[-1]))
                else:
                    _vals, _prices = results[key][4], results[key][5]
                    final_usd = fmt_price(float(_vals[-1]) * float(_prices[-1]))
                    final = f"{float(y_vals[-1]):.4f} BTC  ({final_usd})"
                traces.append(go.Scatter(
                    x=list(ts_d), y=list(y_vals), mode="lines",
                    name=_legend_name,
                    legendgroup=grp_model,
                    showlegend=_first_legend,
                    line=dict(color=col, width=2, shape=_line_shape),
                ))
                _first_legend = False
                if depl_t is not None:
                    deplete_annots.append(_depl_annot(depl_t, t_start_d, d,
                        arrow_col=annot_colors[di % len(annot_colors)],
                        text_col=_tcol_annot, legendgroup=grp_model,
                        model_prefix=model.legend_name,
                        stagger=len(deplete_annots)))

        elif chart_layout == 1:
            # Color = quantile, line style = delay
            for di, d in enumerate(delays):
                for qi, q in enumerate(sel_qs):
                    key = (d, q)
                    if key not in results:
                        continue
                    ts_d, y_vals, depl_t, t_start_d, *_ = results[key]
                    _shade = quantile_shade(_bm_color, q)
                    traces.append(go.Scatter(
                        x=list(ts_d), y=list(y_vals), mode="lines",
                        name=_legend_name,
                        legendgroup=grp_model,
                        showlegend=_first_legend,
                        opacity=quantile_opacity(q),
                        line=dict(color=_shade, width=TRACE_WIDTH,
                                  dash=_DASH_STYLES[di % len(_DASH_STYLES)], shape=_line_shape),
                    ))
                    _first_legend = False
                    if depl_t is not None:
                        deplete_annots.append(_depl_annot(depl_t, t_start_d, d,
                            arrow_col=_bm_color, text_col=_tcol_annot,
                            legendgroup=grp_model,
                            model_prefix=model.legend_name,
                            stagger=len(deplete_annots)))

        else:
            # Layout 2: shaded band + individual traces per delay
            _tcol = _app_ctx.MODEL_TRACE_COLORS.get(model.short_name, BLACK)
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
                col    = delay_colors[di % len(delay_colors)]
                # Shade band (never in legend)
                traces.append(go.Scatter(
                    x=list(ts_d), y=list(y_max), mode="lines",
                    line=dict(color=col, width=0, shape=_line_shape), showlegend=False, hoverinfo="skip",
                    legendgroup=grp_model,
                ))
                traces.append(go.Scatter(
                    x=list(ts_d), y=list(y_min), mode="lines",
                    fill="tonexty", fillcolor=_hex_alpha(col, SC_ENVELOPE_ALPHA),
                    line=dict(color=col, width=0, shape=_line_shape),
                    legendgroup=grp_model,
                    showlegend=False,
                    hoverinfo="skip",
                ))
                # Individual quantile traces on top — single color per model
                for q in sel_qs:
                    key = (d, q)
                    if key not in results:
                        continue
                    ts_d_q, y_vals_q, depl_t, t_start_d, *_ = results[key]
                    traces.append(go.Scatter(
                        x=list(ts_d_q), y=list(y_vals_q), mode="lines",
                        name=_legend_name,
                        line=dict(color=_tcol, width=TRACE_WIDTH, dash=model.dash_style, shape=_line_shape),
                        legendgroup=grp_model, showlegend=_first_legend,
                    ))
                    _first_legend = False
                    if depl_t is not None:
                        deplete_annots.append(_depl_annot(depl_t, t_start_d, d,
                            arrow_col=annot_colors[di % len(annot_colors)],
                            text_col=_tcol_annot, legendgroup=grp_model,
                            model_prefix=model.legend_name,
                            stagger=len(deplete_annots)))

        # ── alternative model overlays (same layout logic as primary) ─────────
        _pending_annots = []  # initialized here so overlays can append
        for model_key in p.get("active_models", []):
            mdl = _resolve_model(model_key, p)
            if not mdl:
                continue
            _sc_overlay_qs = [q for q in sel_qs if not mdl.quantized or q in mdl.fits] if mdl.quantized else [0.5]
            if not _sc_overlay_qs:
                continue
            _ov_q_range = _fmt_q_range(_sc_overlay_qs) if mdl.quantized else ""
            _ov_grp = f"sc-{mdl.short_name}"
            _ov_tcol = _app_ctx.MODEL_TRACE_COLORS.get(mdl.short_name, LIGHT_GRAY)
            _ov_first = True
            _ov_lbl = f"{mdl.legend_name} {_ov_q_range}" if _ov_q_range else mdl.legend_name

            # Compute all (delay, quantile) results for this overlay model
            ov_results = {}
            for d in delays:
                t_start_d = max(yr_to_t(syr + d, m.genesis), 1.0)
                if t_start_d >= t_end:
                    continue
                ts_d = np.arange(t_start_d, t_end + dt * 0.5, dt)
                if len(ts_d) == 0:
                    continue
                ts_d_clamped = np.maximum(ts_d, 0.5)
                adj_wd_d = wd_amount * ((1 + inflation) ** (ts_d - t_start_d))
                for q in _sc_overlay_qs:
                    prices = mdl.price_at(q, ts_d_clamped) if mdl.quantized else np.full_like(ts_d_clamped, mdl.price_at(0.5, ts_d_clamped))
                    vals = np.maximum(start_stack - np.cumsum(adj_wd_d / prices), 0.0)
                    depl_mask = vals == 0.0
                    depl_t_ov = float(ts_d[np.argmax(depl_mask)]) if depl_mask.any() else None
                    y_vals = vals * prices if disp_mode == "usd" else vals
                    ov_results[(d, q)] = (ts_d, y_vals, depl_t_ov, t_start_d, vals, prices)

            # Representative terminal value for overlay legend
            _ov_med_q = _sc_overlay_qs[len(_sc_overlay_qs) // 2]
            _ov_rep_key = (delays[0], _ov_med_q)
            if _ov_rep_key in ov_results:
                _ov_rep = float(ov_results[_ov_rep_key][1][-1])
                _ov_rep_lbl = fmt_price(_ov_rep) if disp_mode == "usd" else f"{_ov_rep:.4f}"
                _ov_lbl = f"{_ov_lbl} \u2192 {_ov_rep_lbl}"

            if chart_layout == 2 and len(_sc_overlay_qs) >= 2:
                # Shade bands + traces for overlay model
                for di, d in enumerate(delays):
                    all_y = [ov_results[(d, q)][1] for q in _sc_overlay_qs if (d, q) in ov_results]
                    if not all_y:
                        continue
                    ts_d = ov_results[(d, _sc_overlay_qs[0])][0]
                    all_y_arr = np.array(all_y)
                    y_min, y_max = all_y_arr.min(axis=0), all_y_arr.max(axis=0)
                    col = delay_colors[di % len(delay_colors)]
                    # Shade band (never in legend)
                    traces.append(go.Scatter(
                        x=list(ts_d), y=list(y_max), mode="lines",
                        line=dict(color=col, width=0, shape=_line_shape), showlegend=False, hoverinfo="skip",
                        legendgroup=_ov_grp,
                    ))
                    traces.append(go.Scatter(
                        x=list(ts_d), y=list(y_min), mode="lines",
                        fill="tonexty", fillcolor=_hex_alpha(col, SC_OVERLAY_ENVELOPE_ALPHA),
                        line=dict(color=col, width=0, shape=_line_shape),
                        legendgroup=_ov_grp,
                        showlegend=False, hoverinfo="skip",
                    ))
                    # Quantile traces — legend shows dash + color
                    for q in _sc_overlay_qs:
                        if (d, q) not in ov_results:
                            continue
                        ts_d_q, y_vals_q, *_ = ov_results[(d, q)]
                        traces.append(go.Scatter(
                            x=list(ts_d_q), y=list(y_vals_q), mode="lines",
                            name=_ov_lbl,
                            line=dict(color=_ov_tcol, width=1, dash=mdl.dash_style, shape=_line_shape),
                            legendgroup=_ov_grp, showlegend=_ov_first,
                        ))
                        _ov_first = False
                    # Depletion arrows for overlay model at this delay
                    for q in _sc_overlay_qs:
                        if (d, q) not in ov_results:
                            continue
                        _, _, depl_t_ov, t_start_d_ov, *_ = ov_results[(d, q)]
                        if depl_t_ov is not None:
                            deplete_annots.append(_depl_annot(depl_t_ov, t_start_d_ov, d,
                                arrow_col=annot_colors[di % len(annot_colors)],
                                text_col=_ov_tcol, legendgroup=_ov_grp,
                                model_prefix=mdl.legend_name,
                                stagger=len(deplete_annots)))
            else:
                # Individual lines for overlay model
                _ov_depl_seen = set()  # track (delay) to emit one arrow per delay
                for (d, q), (ts_d, y_vals, depl_t_ov, t_start_d_ov, *_) in ov_results.items():
                    col = _get_model_color(model_key, p) if mdl.quantized else palette["non_quantized_model"]
                    traces.append(go.Scatter(
                        x=list(ts_d), y=list(y_vals), mode="lines",
                        name=_ov_lbl,
                        line=dict(color=col, width=1.2, dash=mdl.dash_style, shape=_line_shape),
                        legendgroup=_ov_grp,
                        showlegend=_ov_first,
                    ))
                    _ov_first = False
                    if depl_t_ov is not None and d not in _ov_depl_seen:
                        _ov_depl_seen.add(d)
                        di = delays.index(d) if d in delays else 0
                        deplete_annots.append(_depl_annot(depl_t_ov, t_start_d_ov, d,
                            arrow_col=annot_colors[di % len(annot_colors)],
                            text_col=_ov_tcol, legendgroup=_ov_grp,
                            model_prefix=mdl.legend_name,
                            stagger=len(deplete_annots)))

            # Overlay endpoint annotations (best surviving per delay)
            if p.get("annotate") and not p.get("is_mobile"):
                # Overlay endpoint annotations use _ov_tcol (computed at loop top)
                for di, d in enumerate(delays):
                    surviving = [(q, ov_results[(d, q)]) for q in _sc_overlay_qs
                                 if (d, q) in ov_results and ov_results[(d, q)][2] is None]
                    if not surviving:
                        continue
                    best_q, best_r = max(surviving, key=lambda x: float(x[1][1][-1]))
                    y_final = float(best_r[1][-1])
                    if y_final <= 0:
                        continue
                    lbl = fmt_price(y_final) if disp_mode == "usd" else f"{y_final:.4f}"
                    _pending_annots.append(dict(
                        x_arr=best_r[0], y_arr=best_r[1],
                        label=f"{mdl.legend_name} {lbl}",
                        short_label=f"{mdl.legend_name} {lbl}",
                        color=_ov_tcol, y_last=y_final))

        t_start_base = max(yr_to_t(syr, m.genesis), 1.0)
        ylabel = ""
        sc_title = (f"HODL Supercharger \u2014 {fmt_price(wd_amount)}{freq_label} \u00b7 "
                    f"Retire {syr}+ \u00b7 to {eyr}")
        layout, _ = _sim_layout(m, p, sc_title, ylabel, np.array([t_end]),
                                t_start_base, t_end, dt, syr, eyr)
        layout["annotations"] = deplete_annots
        # ── Monte Carlo fan overlay ───────────────────────────────────────────
        mc_traces_list = []
        mc_result = None
        if _HAS_MARKOV and p.get("mc_enabled") and p.get("show_mc", True):
            t_start_base = max(yr_to_t(syr, m.genesis), 1.0)
            _sc_x_end = layout["xaxis"]["range"][1]
            mc_traces_list, mc_result = _apply_mc_overlay(
                m, p, _mc_supercharge_overlay,
                (m, p, np.arange(t_start_base, t_end + dt * 0.5, dt),
                 t_start_base, t_end, dt, start_stack, disp_mode, len(deplete_annots)),
                traces, deplete_annots, layout, _sc_x_end, disp_mode)

        _stagger_depletion_annots(deplete_annots, layout)

        # ── Right-edge / endpoint value labels ─────────────────────────────
        # Use text traces (go.Scatter mode="markers+text") instead of
        # annotations — consistent with Retire tab; avoids paper-x arrowhead
        # misalignment on declining traces.
        _sc_log = bool(p.get("log_y"))
        if p.get("annotate") and not p.get("is_mobile"):
            if show_bm and chart_layout == 2:
                # Band endpoint labels: one per delay, upper-bound value
                for di, d in enumerate(delays):
                    band = [(q, results[(d, q)]) for q in sel_qs
                            if (d, q) in results]
                    surviving = [(q, r) for q, r in band
                                 if r[2] is None and float(r[1][-1]) > 0]
                    if not surviving:
                        continue
                    col = delay_colors[di % len(delay_colors)]
                    y_max = max(float(r[1][-1]) for _, r in surviving)
                    lbl = (fmt_price(y_max) if disp_mode == "usd"
                           else f"{y_max:.4f} \u20bf")
                    y_arr = np.maximum.reduce(
                        [r[1] for _, r in surviving])
                    ts_d_r = surviving[0][1][0]
                    # Get BTC/USD for short label from best surviving entry
                    _best_q, _best_r = max(surviving, key=lambda x: float(x[1][1][-1]))
                    _sc_btc = float(_best_r[4][-1])  # raw BTC vals
                    _sc_usd = float(_best_r[4][-1] * _best_r[5][-1])  # BTC * price
                    _pending_annots.append(dict(
                        x_arr=ts_d_r, y_arr=y_arr,
                        label=lbl, short_label=_fmt_short(_sc_btc, _sc_usd),
                        color=col, y_last=float(y_arr[-1])))
            elif show_bm:
                for (d, q), res in results.items():
                    ts_d_r, y_vals_r, depl_t_r, _, btc_vals_r, prices_r = res
                    if depl_t_r is not None:
                        continue  # depleted — already has year annotation
                    y_final = float(y_vals_r[-1])
                    if y_final <= 0:
                        continue
                    if chart_layout == 0:
                        di = delays.index(d) if d in delays else 0
                        col = delay_colors[di % len(delay_colors)]
                    else:
                        col = _bm_color  # uniform per-model, matches the line
                    lbl = (fmt_price(y_final) if disp_mode == "usd"
                           else f"{y_final:.4f} \u20bf")
                    _sc_btc = float(btc_vals_r[-1])
                    _sc_usd = float(btc_vals_r[-1] * prices_r[-1])
                    _pending_annots.append(dict(
                        x_arr=ts_d_r, y_arr=y_vals_r,
                        label=lbl, short_label=_fmt_short(_sc_btc, _sc_usd),
                        color=col, y_last=y_final))
            # MC median endpoint
            _mc_ann = _mc_median_annot(
                mc_traces_list, disp_mode, m, t_end, t_start_base, t_end,
                syr, eyr, btc_fmt=".4f", estimate_usd=False)
            if _mc_ann:
                _pending_annots.append(_mc_ann)
        traces.extend(_resolve_edge_annotations(_pending_annots, _sc_log))

        # Handle empty plot (all models unchecked)
        if not traces:
            _empty_state_annotation(layout)

        return _finalize_chart(traces, layout, p, "sc", mc_result, mc_premium=False)

    # ── MODE B: fixed depletion date -> max withdrawal per period ──────────────
    else:
        return _sc_mode_b(m, p, syr, delays, sel_qs, start_stack, ppy, dt,
                          inflation, chart_layout, display_q, show_legend, freq_label)


def _sc_mode_b(m, p, syr, delays, sel_qs, start_stack, ppy, dt,
               inflation, chart_layout, display_q, show_legend, freq_label):
    """HODL Supercharger Mode B: binary-search max withdrawal per period."""
    model = _resolve_sc_model(p)
    palette = _get_palette(p)
    delay_colors = palette["delay_colors"]
    _bm_color = _get_model_color("bub", p)
    target_yr = int(p.get("target_yr", SUPERCHARGE["target_yr"]))

    def _max_wd_for(d, q):
        t_start_d = max(yr_to_t(syr + d, m.genesis), 1.0)
        t_end_b   = yr_to_t(target_yr, m.genesis)
        if t_end_b <= t_start_d:
            return 0.0
        first_price = float(model.price_at(q, max(t_start_d, 0.5)))
        # Binary search: find max withdrawal where stack survives to target_yr.
        # Upper bound = 4x annual stack value (generous overestimate).
        # 60 iterations gives precision to ~1e-18 of the range (more than enough).
        lo, hi = 0.0, start_stack * first_price * ppy * 4
        for _ in range(_BISECT_ITERS):
            mid = (lo + hi) / 2.0
            s   = start_stack
            survived = True
            for t in np.arange(t_start_d, t_end_b + dt * 0.5, dt):
                adj = mid * ((1 + inflation) ** (t - t_start_d))
                s  -= adj / float(model.price_at(q, max(t, 0.5)))
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
    show_bm = "bub" in (p.get("active_models") or [])

    if not show_bm:
        pass  # skip all BM traces
    elif chart_layout == 0:
        q_show = min(sel_qs, key=lambda q: abs(q - display_q))
        y_line = [max_wd.get((d, q_show), 0) for d in delays]
        traces.append(go.Scatter(
            x=delays, y=y_line, mode="lines",
            line=dict(color=FALLBACK_MODEL_GRAY, width=1, dash="dot"),
            showlegend=False, hoverinfo="skip",
        ))
        for di, d in enumerate(delays):
            col   = delay_colors[di % len(delay_colors)]
            val   = max_wd.get((d, q_show), 0)
            d_lbl = f"+{int(d)}yr" if d == int(d) else f"+{d:.1f}yr"
            traces.append(go.Scatter(
                x=[d], y=[val], mode="markers+text",
                marker=dict(color=col, size=12),
                text=[fmt_price(val) + freq_label],
                textposition="top center",
                name=f"{model.legend_name} Delay {d_lbl}",
                hovertemplate=f"{model.legend_name} Delay {d_lbl}<br>{fmt_price(val)}{freq_label}<extra></extra>",
            ))

    elif chart_layout == 1:
        q_range = _fmt_q_range(sel_qs)
        grp = f"{model.short_name}-b1"
        for qi, q in enumerate(sel_qs):
            _shade = quantile_shade(_bm_color, q)
            y_q   = [max_wd.get((d, q), 0) for d in delays]
            traces.append(go.Scatter(
                x=delays, y=y_q, mode="lines+markers",
                name=f"{model.legend_name} {q_range}",
                legendgroup=grp,
                showlegend=(qi == 0),
                opacity=quantile_opacity(q),
                line=dict(color=_shade, width=2),
                marker=dict(color=_shade, size=7),
            ))

    else:
        for di, d in enumerate(delays):
            col   = delay_colors[di % len(delay_colors)]
            d_lbl = f"+{int(d)}yr" if d == int(d) else f"+{d:.1f}yr"
            y_d   = [max_wd.get((d, q), 0) for q in sel_qs]
            qlbls = [_fmt_q_label(q) for q in sel_qs]
            med_val = y_d[len(y_d) // 2] if y_d else 0
            traces.append(go.Scatter(
                x=list(sel_qs), y=y_d, mode="lines+markers",
                name=f"{model.legend_name} Delay {d_lbl}  \u2192  {fmt_price(med_val)}{freq_label} (med)",
                line=dict(color=col, width=2),
                marker=dict(color=col, size=6),
                customdata=qlbls,
                hovertemplate="%{customdata}: %{y:,.0f}<extra></extra>",
            ))

    xlabel = "Delay (years)" if chart_layout in (0, 1) else "Quantile"
    layout = _base_layout(
        title=f"HODL Supercharger \u2014 Max spend{freq_label} to deplete by {target_yr}  ({model.name})",
        xlabel=xlabel,
        ylabel="",
    )

    # Handle empty plot (all models unchecked)
    if not traces:
        _empty_state_annotation(layout)

    return _finalize_chart(traces, layout, p, "sc", mc_premium=False)
