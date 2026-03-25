"""HODL Supercharger chart builder."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Any

import _app_ctx
from btc_core import ModelData, yr_to_t, fmt_price
from mc_overlay import _mc_supercharge_overlay

from figures.common import (
    _QR_LINE_WIDTH, _ANNOT_STAGGER_Y, _BISECT_ITERS,
    _FONT_ANNOT,
    _HAS_MARKOV,
    _get_palette, _build_thermal_colors, _fmt_q_label, _fmt_q_range, _error_figure,
    _build_freq_config, _get_starting_stack,
    _sim_layout, _apply_mc_overlay,
    _stagger_depletion_annots,
    _dark_layout, _finalize_chart,
    _fmt_short, _mc_median_annot,
    _resolve_edge_annotations,
    _hex_alpha,

)

_DASH_STYLES  = ['solid', 'dash', 'dot', 'dashdot', 'longdash']


def _resolve_sc_model(p):
    """Pick the primary quantized model from active_models, or fall back to default."""
    for key in (p.get("active_models") or []):
        mdl = _app_ctx.PRICE_MODELS.get(key)
        if mdl and mdl.quantized:
            return mdl
    if p.get("show_qr"):
        return _app_ctx.DEFAULT_MODEL
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
    sel_qs_raw = sorted([float(q) for q in (p.get("selected_qs") or [])])
    _thermal = _build_thermal_colors(sel_qs_raw, palette)

    mode         = p.get("mode", "a")
    freq_str, ppy, dt = _build_freq_config(p)
    syr          = int(p.get("start_yr", pd.Timestamp.today().year))
    inflation    = float(p.get("inflation", 4)) / 100.0
    chart_layout = int(p.get("chart_layout", 0))
    display_q    = float(p.get("display_q", 0.5))
    show_legend  = bool(p.get("show_legend", True))

    # Starting stack (lots override)
    start_stack = _get_starting_stack(p, default=1.0)

    # Quantiles
    sel_qs = sorted([float(q) for q in (p.get("selected_qs") or [])
                     if float(q) in model.fits])
    mc_enabled = _HAS_MARKOV and p.get("mc_enabled")
    if not sel_qs and not mc_enabled:
        return go.Figure(layout=dict(
            title="Select at least one quantile",
            paper_bgcolor=m.PLOT_BG_COLOR,
            font=dict(color=m.TEXT_COLOR))), None
    if not sel_qs:
        sel_qs = [0.5]  # need at least one for MC simulation grid

    # Delays: filter None/negative, sort, deduplicate
    raw_delays = p.get("delays") or [0, 1, 2, 4, 8]
    delays = sorted(set(float(d) for d in raw_delays if d is not None and float(d) >= 0))
    if not delays:
        delays = [0.0]

    freq_label = {"Daily": "/day", "Weekly": "/wk", "Monthly": "/mo",
                  "Quarterly": "/qtr", "Annually": "/yr"}.get(freq_str, "/mo")

    # ── MODE A: fixed spending -> show how long savings last ───────────────────
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

        def _depl_annot(depl_t, t_start_d, d, col, stagger=0):
            depl_yr = int((syr + d) + (depl_t - t_start_d) *
                          (eyr - (syr + d)) / max(t_end - t_start_d, 1e-6))
            return dict(
                x=depl_t - dt, xref="x",   # last nonzero step, aligns with band end
                y=0, yref="paper",
                ax=28, ay=_AY_LEVELS[stagger % 3],
                text=f"\u2248{depl_yr}",
                showarrow=True, arrowhead=2, arrowsize=1,
                arrowcolor=col, font=dict(size=_FONT_ANNOT, color=col),
            )

        show_qr = p.get("show_qr", True)

        if not show_qr:
            pass  # skip QR traces, keep results for MC/annotations
        elif chart_layout == 0:
            # Color = delay, show quantile closest to display_q
            q_show = min(sel_qs, key=lambda q: abs(q - display_q))
            q_range = _fmt_q_range(sel_qs)
            for di, d in enumerate(delays):
                key = (d, q_show)
                if key not in results:
                    continue
                ts_d, y_vals, depl_t, t_start_d, *_ = results[key]
                col   = delay_colors[di % len(delay_colors)]
                d_lbl = f"+{int(d)}yr" if d == int(d) else f"+{d:.1f}yr"
                if disp_mode == "usd":
                    final = fmt_price(float(y_vals[-1]))
                else:
                    _vals, _prices = results[key][4], results[key][5]
                    final_usd = fmt_price(float(_vals[-1]) * float(_prices[-1]))
                    final = f"{float(y_vals[-1]):.4f} BTC  ({final_usd})"
                traces.append(go.Scatter(
                    x=list(ts_d), y=list(y_vals), mode="lines",
                    name=f"{model.name} {q_range} Delay {d_lbl}  \u2192  {final}",
                    line=dict(color=col, width=2),
                ))
                if depl_t is not None:
                    deplete_annots.append(_depl_annot(depl_t, t_start_d, d,
                                                      annot_colors[di % len(annot_colors)],
                                                      len(deplete_annots)))

        elif chart_layout == 1:
            # Color = quantile, line style = delay — group by delay
            q_range = _fmt_q_range(sel_qs)
            for di, d in enumerate(delays):
                d_lbl = f"+{int(d)}yr" if d == int(d) else f"+{d:.1f}yr"
                grp = f"{model.short_name}-d{d}"
                for qi, q in enumerate(sel_qs):
                    key = (d, q)
                    if key not in results:
                        continue
                    ts_d, y_vals, depl_t, t_start_d, *_ = results[key]
                    col = _thermal.get(q, model.colors.get(q, "#888888"))
                    traces.append(go.Scatter(
                        x=list(ts_d), y=list(y_vals), mode="lines",
                        name=f"{model.name} {q_range} delay={d_lbl}",
                        legendgroup=grp,
                        showlegend=(qi == 0),
                        line=dict(color=col, width=_QR_LINE_WIDTH,
                                  dash=_DASH_STYLES[di % len(_DASH_STYLES)]),
                    ))
                    if depl_t is not None:
                        deplete_annots.append(_depl_annot(depl_t, t_start_d, d, col,
                                                          len(deplete_annots)))

        else:
            # Layout 2: shaded band per delay (min/max across quantiles)
            q_range = _fmt_q_range(sel_qs)
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
                d_lbl  = f"+{int(d)}yr" if d == int(d) else f"+{d:.1f}yr"
                traces.append(go.Scatter(
                    x=list(ts_d), y=list(y_max), mode="lines",
                    line=dict(color=col, width=0), showlegend=False, hoverinfo="skip",
                ))
                max_final = (fmt_price(float(y_max[-1])) if disp_mode == "usd"
                             else f"{float(y_max[-1]):.4f} BTC")
                traces.append(go.Scatter(
                    x=list(ts_d), y=list(y_min), mode="lines",
                    fill="tonexty", fillcolor=_hex_alpha(col, 0.2),
                    line=dict(color=col, width=0),
                    name=f"{model.name} {q_range} Delay {d_lbl}  \u2192  {max_final}",
                    hoverinfo="skip",
                ))
                for q in sel_qs:
                    key = (d, q)
                    if key not in results:
                        continue
                    _, _, depl_t, t_start_d, *_ = results[key]
                    if depl_t is not None:
                        deplete_annots.append(_depl_annot(depl_t, t_start_d, d,
                                                          annot_colors[di % len(annot_colors)],
                                                          len(deplete_annots)))

        # ── alternative model overlays ────────────────────────────────────────
        for model_key in p.get("active_models", []):
            mdl = _app_ctx.PRICE_MODELS.get(model_key)
            if not mdl:
                continue
            _sc_overlay_qs = sel_qs if mdl.quantized else [0.5]
            for di, d in enumerate(delays):
                t_start_d = max(yr_to_t(syr + d, m.genesis), 1.0)
                if t_start_d >= t_end:
                    continue
                ts_d = np.arange(t_start_d, t_end + dt * 0.5, dt)
                if len(ts_d) == 0:
                    continue
                ts_d_clamped = np.maximum(ts_d, 0.5)
                adj_wd_d = wd_amount * ((1 + inflation) ** (ts_d - t_start_d))
                for q in _sc_overlay_qs:
                    if mdl.quantized and q not in mdl.fits:
                        continue
                    prices = mdl.price_at(q, ts_d_clamped)
                    vals = np.maximum(start_stack - np.cumsum(adj_wd_d / prices), 0.0)
                    y_vals = vals * prices if disp_mode == "usd" else vals
                    col = mdl.colors.get(q, "#888888") if mdl.quantized else palette["non_quantized_model"]
                    d_lbl = f"+{int(d)}yr" if d == int(d) else f"+{d:.1f}yr"
                    q_lbl = f" {_fmt_q_label(q, '')}" if mdl.quantized else ""
                    if disp_mode == "usd":
                        final = fmt_price(float(y_vals[-1]))
                    else:
                        final_usd = fmt_price(float(vals[-1]) * float(prices[-1]))
                        final = f"{float(vals[-1]):.4f} BTC  ({final_usd})"
                    traces.append(go.Scatter(
                        x=list(ts_d), y=list(y_vals), mode="lines",
                        name=f"{mdl.name}{q_lbl} {d_lbl}  \u2192  {final}",
                        line=dict(color=col, width=1.2, dash=mdl.dash_style),  # intentional: 1.2 not _OVERLAY_LINE_WIDTH
                        legendgroup=mdl.short_name,
                        legendgrouptitle_text=mdl.name,
                        showlegend=(di == 0),  # show legend only for first delay
                    ))

        t_start_base = max(yr_to_t(syr, m.genesis), 1.0)
        ylabel = "USD Value" if disp_mode == "usd" else "BTC Remaining"
        sc_title = (f"HODL Supercharger \u2014 {fmt_price(wd_amount)}{freq_label} \u00b7 "
                    f"Retire {syr}+ \u00b7 to {eyr}")
        layout, _ = _sim_layout(m, p, sc_title, ylabel, np.array([t_end]),
                                t_start_base, t_end, dt, syr, eyr)
        layout["annotations"] = deplete_annots
        # ── Monte Carlo fan overlay ───────────────────────────────────────────
        mc_traces_list = []
        mc_result = None
        if _HAS_MARKOV and p.get("mc_enabled"):
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
        _pending_annots = []
        if p.get("annotate"):
            if chart_layout == 2:
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
            else:
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
                        col = _thermal.get(q, model.colors.get(q, "#888888"))
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
    _thermal = _build_thermal_colors(sel_qs, palette)
    target_yr = int(p.get("target_yr", 2060))

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

    if chart_layout == 0:
        q_show = min(sel_qs, key=lambda q: abs(q - display_q))
        y_line = [max_wd.get((d, q_show), 0) for d in delays]
        traces.append(go.Scatter(
            x=delays, y=y_line, mode="lines",
            line=dict(color="#888888", width=1, dash="dot"),
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
                name=f"{model.name} Delay {d_lbl}",
                hovertemplate=f"{model.name} Delay {d_lbl}<br>{fmt_price(val)}{freq_label}<extra></extra>",
            ))

    elif chart_layout == 1:
        q_range = _fmt_q_range(sel_qs)
        grp = f"{model.short_name}-b1"
        for qi, q in enumerate(sel_qs):
            col   = _thermal.get(q, model.colors.get(q, "#888888"))
            y_q   = [max_wd.get((d, q), 0) for d in delays]
            traces.append(go.Scatter(
                x=delays, y=y_q, mode="lines+markers",
                name=f"{model.name} {q_range}",
                legendgroup=grp,
                showlegend=(qi == 0),
                line=dict(color=col, width=2),
                marker=dict(color=col, size=7),
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
                name=f"{model.name} Delay {d_lbl}  \u2192  {fmt_price(med_val)}{freq_label} (med)",
                line=dict(color=col, width=2),
                marker=dict(color=col, size=6),
                customdata=qlbls,
                hovertemplate="%{customdata}: %{y:,.0f}<extra></extra>",
            ))

    xlabel = "Delay (years)" if chart_layout in (0, 1) else "Quantile"
    layout = _dark_layout(
        m,
        title=f"HODL Supercharger \u2014 Max spend{freq_label} to deplete by {target_yr}  ({model.name})",
        xlabel=xlabel,
        ylabel=f"Max withdrawal{freq_label}",
    )
    return _finalize_chart(traces, layout, p, "sc", mc_premium=False)
