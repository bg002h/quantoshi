"""Residuals chart for tab 1 — price deviation from model reference lines."""

from __future__ import annotations

import math
import numpy as np
import plotly.graph_objects as go
from typing import Any

import _app_ctx
from btc_core import yr_to_t, today_t, ModelData, UserModel

from figures.common import (
    _OVERLAY_LINE_WIDTH, _QR_LINE_WIDTH,
    _TODAY_LINE_COLOR, _TODAY_LINE_WIDTH, _TODAY_LINE_OPACITY,
    _get_palette, _get_model_color,
    _base_layout, _year_ticks,
    _apply_sans_typography, _apply_watermark, _add_date_hover,
    _apply_config_annotation,
    _round_trace_data,
    _INTERP_POINTS, _MC_LEGEND_POS,
)


def build_residuals_figure(m: ModelData, p: dict[str, Any]) -> go.Figure:
    """Build residuals figure: log10(price) - log10(model_reference(t)).

    For each active model in p['active_models']:
      - BM ('bub'): uses support if 'show_sup' in bub_toggles, composite if 'show_comp'
      - Others (lppl, lp2, pl, exp, s2f, ef, u1): always uses median (Q50%)

    Residuals are plotted at actual price data timestamps.
    """
    model = _app_ctx.DEFAULT_MODEL
    palette = _get_palette(p)

    _xmin = float(p["xmin"])
    if _xmin <= 2010:
        _xmin = 2010.5
    t_lo = max(yr_to_t(_xmin, m.genesis), 0.01)
    t_hi = yr_to_t(p["xmax"], m.genesis)

    # Use actual daily price data (timestamps + log-prices)
    mask = (m.price_years >= t_lo) & (m.price_years <= t_hi)
    t_data = m.price_years[mask]
    log_p_data = np.log10(np.maximum(m.price_prices[mask], 1e-10))

    traces = []
    active_models = p.get("active_models", [])
    bub_toggles = p.get("bub_toggles", [])
    show_comp = "show_comp" in bub_toggles
    show_sup = "show_sup" in bub_toggles

    # ── BM residuals: support and/or composite ────────────────────────────────
    if "bub" in active_models:
        bm_color = _get_model_color("bub", p)
        # Support residual
        if show_sup and hasattr(m, "support_bm") and hasattr(m, "years_plot_bm"):
            sup_log = np.log10(np.maximum(
                np.interp(t_data, m.years_plot_bm, m.support_bm), 1e-10))
            resid = log_p_data - sup_log
            traces.append(go.Scatter(
                x=list(t_data), y=_round_trace_data(resid),
                mode="lines", name="BM support residual",
                line=dict(color=bm_color, width=_OVERLAY_LINE_WIDTH, dash="dash"),
                opacity=0.8,
            ))
        # Composite residual
        if show_comp and hasattr(m, "comp_by_n") and hasattr(m, "years_plot_bm"):
            n_fut = min(int(p.get("n_future", 3)), len(m.comp_by_n) - 1)
            comp_vals = np.asarray(m.comp_by_n[n_fut])
            comp_log = np.log10(np.maximum(
                np.interp(t_data, m.years_plot_bm, comp_vals), 1e-10))
            resid = log_p_data - comp_log
            traces.append(go.Scatter(
                x=list(t_data), y=_round_trace_data(resid),
                mode="lines", name="BM composite residual",
                line=dict(color=bm_color, width=_OVERLAY_LINE_WIDTH),
                opacity=1.0,
            ))

    # ── Other models: median (Q50%) residuals ────────────────────────────────
    for model_key in active_models:
        if model_key == "bub":
            continue
        if model_key == "u1":
            um_data = p.get("user_model")
            if not um_data:
                continue
            mdl = UserModel.from_store_dict(um_data)
            if not mdl:
                continue
        else:
            mdl = _app_ctx.PRICE_MODELS.get(model_key)
            if not mdl:
                continue
        # Use median (Q50%) if quantized, else the single trajectory
        q_ref = 0.5
        if mdl.quantized and q_ref not in mdl.fits:
            # Pick nearest available quantile
            q_ref = min(mdl.quantiles, key=lambda q: abs(q - 0.5))
        try:
            pred_prices = mdl.price_at(q_ref, t_data)
        except Exception:
            continue
        pred_log = np.log10(np.maximum(np.asarray(pred_prices), 1e-10))
        resid = log_p_data - pred_log
        color = _get_model_color(model_key, p)
        traces.append(go.Scatter(
            x=list(t_data), y=_round_trace_data(resid),
            mode="lines", name=f"{mdl.legend_name} median residual",
            line=dict(color=color, width=_OVERLAY_LINE_WIDTH,
                      dash=getattr(mdl, "dash_style", "solid")),
            opacity=1.0,
        ))

    # ── Component Decomposition partial-model residual ───────────────────────
    decomp_family = p.get("decomp_model", "") or ""
    decomp_selected = list(p.get("decomp_components", []) or [])
    if decomp_family and decomp_selected:
        from callbacks.charts import _resolve_decomp_model_key
        dkey = _resolve_decomp_model_key(
            decomp_family,
            p.get("lppl_n_freqs", []),
            p.get("lppl_weighted", []),
            p.get("lppl_no_13", []),
        )
        dmodel = _app_ctx.PRICE_MODELS.get(dkey) if dkey else None
        if dmodel is not None:
            canonical = [n for n in dmodel.component_names
                         if n in decomp_selected]
            if canonical:
                dcomps = dmodel.components(t_data)
                partial_log = dcomps[canonical[0]].copy()
                for n in canonical[1:]:
                    partial_log = partial_log + dcomps[n]
                resid = log_p_data - partial_log
                total = len(dmodel.component_names)
                n_checked = len(canonical)
                label = (f"{dmodel.legend_name} decomp ({n_checked}/{total})"
                         if n_checked < total
                         else f"{dmodel.legend_name} decomp (full)")
                traces.append(go.Scatter(
                    x=list(t_data), y=_round_trace_data(resid),
                    mode="lines", name=label,
                    line=dict(color="#000000", width=2, dash="solid"),
                    opacity=1.0,
                ))

    # ── Today line + zero line ────────────────────────────────────────────────
    shapes = []
    today_color = palette.get("today_line", _TODAY_LINE_COLOR)
    # Zero reference
    shapes.append(dict(
        type="line", xref="x", yref="y",
        x0=t_lo, x1=t_hi, y0=0, y1=0,
        line=dict(color="#888", width=1.5, dash="dot"),
        opacity=0.6,
    ))
    if p.get("show_today"):
        td = today_t(m.genesis)
        if t_lo <= td <= t_hi:
            shapes.append(dict(
                type="line", xref="x", yref="paper",
                x0=td, x1=td, y0=0, y1=1,
                line=dict(color=today_color, dash="dash", width=_TODAY_LINE_WIDTH),
                opacity=_TODAY_LINE_OPACITY,
            ))

    # ── Axis ticks ────────────────────────────────────────────────────────────
    tick_ts, tick_lbls = _year_ticks(p["xmin"], p["xmax"], m.genesis,
                                     minor_grid=p.get("minor_grid"))
    filtered = [(t, lbl) for t, lbl in zip(tick_ts, tick_lbls) if t_lo <= t <= t_hi]
    tick_ts, tick_lbls = (list(x) for x in zip(*filtered)) if filtered else ([], [])

    layout = _base_layout(
        title="Model Residuals — log\u2081\u2080(price) \u2212 log\u2081\u2080(model)",
        xlabel="", ylabel="",
    )
    layout["xaxis"].update(
        range=[t_lo, t_hi],
        tickvals=tick_ts, ticktext=tick_lbls, tickangle=45,
    )
    layout["yaxis"].update(
        ticksuffix="",
        ticklabelposition="inside",
        ticklabelshift=-5,
    )
    # Residuals can be negative, so log Y doesn't apply — always linear.
    # (bub-yrange is in log10 USD space, doesn't translate to residual space.)
    layout["shapes"] = shapes
    layout["showlegend"] = bool(p.get("show_legend", True))
    leg_pos = p.get("legend_pos", "outside")
    if leg_pos != "outside" and leg_pos in _MC_LEGEND_POS:
        pos = _MC_LEGEND_POS[leg_pos]
        layout.setdefault("legend", {})
        layout["legend"].update(
            x=pos["x"], y=pos["y"],
            xanchor=pos["xanchor"], yanchor=pos["yanchor"],
            bgcolor="rgba(255,255,255,0.7)",
        )
    if p.get("xscale") == "log":
        layout["xaxis"].update(
            type="log",
            range=[math.log10(max(t_lo, 1e-10)), math.log10(max(t_hi, 1e-10))],
        )

    # Empty state
    if not traces:
        layout["annotations"] = [dict(
            text="No models selected \u2014 check Display Models",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16, color="#888"),
        )]

    _apply_sans_typography(layout)
    fig = go.Figure(data=traces, layout=go.Layout(**layout))
    _add_date_hover(fig, m.genesis, fmt="<b>%{fullData.name}</b><br>%{customdata[0]}<br>Δlog₁₀ = %{y:.3f}<extra></extra>")
    _apply_config_annotation(fig, p, "bub", show_qr=False, show_mc=False)
    _apply_watermark(fig, pos="bottom-right")
    return fig
