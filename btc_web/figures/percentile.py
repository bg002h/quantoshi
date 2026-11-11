"""Percentile-vs-time view for Tab 1.

For each quantized model, plot over time the percentile at which the actual BTC
price sat within that model's quantile fan (0-100%) — a model-relative
valuation / mean-reversion oscillator. Historical only (percentile needs a
realized price). Colorblind-safe shaded valuation zones sit behind the lines.
"""
from __future__ import annotations

import functools
import math
from datetime import date
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Any

import _app_ctx
import pctile_cycle as _pc
from btc_core import yr_to_t, today_t, ModelData, UserModel
from colors import (_hex_alpha, PLOT_BG_COLOR, FALLBACK_MODEL_GRAY,
                    MC_LEGEND_BG_ALPHA, PCTILE_CHEAP_COLOR, PCTILE_RICH_COLOR,
                    PCTILE_FIT_DATA_COLOR, PCTILE_FIT_BAND_ALPHA,
                    PCTILE_FIT_CONTEXT_OPACITY, PCTILE_FORECAST_SHADE,
                    PCTILE_FORECAST_SHADE_ALPHA, PCTILE_CENSOR_ALPHA,
                    CHART_FONT_LEGEND, BLACK_A0)

from figures.common import (
    TRACE_WIDTH_OVERLAY,
    _TODAY_LINE_COLOR,
    _get_palette, _get_model_color,
    _base_layout, _year_ticks,
    _apply_sans_typography,
    _apply_final_steps, _today_line_shapes, _empty_state_annotation,
    _round_trace_data, _MC_LEGEND_POS,
)

_CHEAP_MAX = 25.0   # below this percentile = "cheap" band
_RICH_MIN = 75.0    # above this percentile = "rich" band
_ZONE_ALPHA = 0.10  # background band opacity


def _bracket_percentile(log_p, col, qs_arr):
    """Percentile of log_p within one date's per-quantile log-price column,
    using the SAME first-bracket linear scan as PriceModel.find_percentile.

    Robust to a non-monotonic fan (quantile crossing — the QR model's
    independently-fit channels cross in the extreme tails). np.interp assumes a
    sorted xp and would silently return a wrong value from its binary search;
    this mirrors find_percentile's scan exactly, so the oscillator matches the
    navbar ticker for every model.
    """
    if log_p <= col[0]:
        return qs_arr[0]
    if log_p >= col[-1]:
        return qs_arr[-1]
    for i in range(qs_arr.shape[0] - 1):
        a = col[i]
        b = col[i + 1]
        if a <= log_p <= b:
            frac = (log_p - a) / (b - a + 1e-30)
            return qs_arr[i] + frac * (qs_arr[i + 1] - qs_arr[i])
    return qs_arr[-1]


def _percentile_series(mdl, t_data, px_data, sigma_mode="constant"):
    """Vectorized percentile of each actual price within the model's fan.

    Same log-space interpolation as ``mdl.find_percentile`` (edge-clamped to the
    outer quantiles), but computes each quantile's price for ALL dates at once
    (price_at is vectorized over t) instead of calling find_percentile per date
    — ~n_quantiles model evals total rather than n_dates × n_quantiles scalar
    ones. ``sigma_mode`` selects the band model (resqr = residual quantile bands,
    constant = legacy), so the oscillator matches the fan shown on the price
    chart. Returns percentiles in [0, 100], or None if the model has no fan.
    """
    qs = mdl.quantiles
    if not qs:
        return None
    qs_arr = np.asarray(qs, dtype=float)              # increasing
    t_safe = np.maximum(np.asarray(t_data, float), 0.5)
    # (n_q, n_t) log-price fan
    logfan = np.array([
        np.log10(np.maximum(
            np.asarray(mdl.price_at(q, t_safe, sigma_mode=sigma_mode), float), 1e-10))
        for q in qs])
    logpx = np.log10(np.maximum(np.asarray(px_data, float), 1e-10))
    pct = np.empty(logpx.shape[0], dtype=float)
    for j in range(pct.shape[0]):
        # First-bracket scan (not np.interp) so a non-monotonic fan can't
        # silently mis-map — matches find_percentile for every model.
        pct[j] = _bracket_percentile(logpx[j], logfan[:, j], qs_arr)
    return pct * 100.0


# ── Cycle fit (one calendar-time sinusoid through ONE model's percentile) ─────
# Spec: docs/superpowers/specs/2026-11-10-percentile-cycle-fit-design.md
# The math lives in pctile_cycle.py; this section caches it per model / sigma
# mode / censor / data version and turns it into traces and labels.

_WEEK = 7.0 / 365.25
_TRANSPARENT = BLACK_A0   # width-0 band edges need an explicit transparent colour
_FIT_LINE_WIDTH = 2.4
_TURN_SIZE = 9
_LABEL_SHIFT = 24        # px between a turn marker and its label


def _t_of(datestr: str, genesis) -> float:
    return (pd.Timestamp(datestr) - genesis).days / 365.25


def _mon_yr(t: float, genesis) -> str:
    return (genesis + pd.Timedelta(days=float(t) * 365.25)).strftime("%b %Y")


def _money(v: float) -> str:
    if v >= 1e12:
        return f"${v/1e12:.2f}T"
    if v >= 1e9:
        return f"${v/1e9:.2f}B"
    if v >= 1e6:
        return f"${v/1e6:.2f}M"
    if v >= 1e3:
        return f"${v/1e3:,.0f}k"
    if v >= 1:
        return f"${v:,.0f}"
    return f"${v:.2f}"


def _sig4(a: np.ndarray) -> list[float]:
    """Round to 4 significant figures for the wire (prices span 9 decades)."""
    return [float(f"{v:.4g}") for v in np.asarray(a, float)]


def _cycle_model(key: str):
    """The registered quantized model for a picker key, or None."""
    if key not in _pc.CYCLE_MODEL_KEYS:
        return None
    mdl = _app_ctx.PRICE_MODELS.get(key)
    if mdl is None or not getattr(mdl, "quantized", False) or not mdl.quantiles:
        return None
    return mdl


def _cycle_label(key: str, mdl) -> str:
    return _pc.CYCLE_MODEL_LABELS.get(key) or getattr(mdl, "legend_name", key)


@functools.lru_cache(maxsize=16)
def _cycle_result(model_key: str, sigma_mode: str, censor: bool, fp: str):
    """Fit, bootstrap and hindcast for one model's FULL-record percentile.

    ``fp`` (the model-data fingerprint) is only a cache key: a rebuilt
    model_data.pkl changes it, so a worker never serves a stale fit. The
    x-window never enters — the slider cannot change the fit. Returns None
    when the model is unavailable or has no fan.
    """
    del fp
    mdl = _cycle_model(model_key)
    if mdl is None:
        return None
    M = _app_ctx.M
    t_all = np.asarray(M.price_years, float)
    keep = t_all >= 1.0                     # log-log anchor is t = 1
    t = t_all[keep]
    px = np.asarray(M.price_prices, float)[keep]
    pct = _percentile_series(mdl, t, px, sigma_mode=sigma_mode)
    if pct is None:
        return None
    pct = np.asarray(pct, float)
    ok = np.isfinite(pct)
    if censor:
        a = _t_of(_pc.CENSOR_WINDOW[0], M.genesis)
        b = _t_of(_pc.CENSOR_WINDOW[1], M.genesis)
        ok &= ~((t >= a) & (t <= b))
    t, pct = t[ok], pct[ok]
    if t.size < 2 * 365:
        return None
    fit = _pc.fit_cycle(t, pct)
    return dict(model_key=model_key, t=t, pct=pct, fit=fit,
                params=_pc.bootstrap_params(t, pct, fit),
                hind=_pc.hindcast(t, pct))


def _logfan(mdl, tt, sigma_mode):
    """(qs, log10 fan of shape (n_q, len(tt))) for the model at times tt."""
    qs = np.asarray(mdl.quantiles, float)
    tt = np.maximum(np.asarray(tt, float), 0.5)
    lf = np.array([
        np.log10(np.maximum(np.broadcast_to(
            np.asarray(mdl.price_at(q, tt, sigma_mode=sigma_mode), float),
            tt.shape), 1e-10))
        for q in qs])
    return qs, lf


def _cycle_overlay(cyc, mdl, p, m, t_lo, t_hi, xlog):
    """Traces (back layer, front layer), shapes and annotations for the fit."""
    fit, params, hind = cyc["fit"], cyc["params"], cyc["hind"]
    key = cyc["model_key"]
    label = _cycle_label(key, mdl)
    color = _get_model_color(key, p)
    sm = p.get("sigma_mode", "constant")
    t_last = float(m.price_years[-1])
    weak = fit.r2 < _pc.WEAK_R2

    def ax_x(t):
        return math.log10(t) if xlog else t

    def rows(tt):
        """Wave value + hover rows [band lo, band hi, $, $lo, $hi] at tt."""
        yc = _pc.wave(fit.c, fit.A, fit.f, fit.phi, tt)
        blo, bhi = _pc.band(params, tt)
        qs, lf = _logfan(mdl, tt, sm)
        pr = _pc.fan_prices(lf, qs, np.clip(yc, 0.0, 100.0))
        prl = _pc.fan_prices(lf, qs, blo)
        prh = _pc.fan_prices(lf, qs, bhi)
        cd = [[round(float(a), 1), round(float(b), 1), c, d, e]
              for a, b, c, d, e in zip(blo, bhi, _sig4(pr), _sig4(prl), _sig4(prh))]
        return yc, blo, bhi, cd

    grid = np.append(np.arange(t_lo, t_hi, _WEEK), t_hi)
    _, blo, bhi, _ = rows(grid)
    back = [
        go.Scatter(x=grid, y=_round_trace_data(bhi), mode="lines",
                   line=dict(width=0, color=_TRANSPARENT), hoverinfo="skip",
                   showlegend=False, name="band 90%"),
        go.Scatter(x=grid, y=_round_trace_data(blo), mode="lines",
                   line=dict(width=0, color=_TRANSPARENT), hoverinfo="skip",
                   fill="tonexty",
                   fillcolor=_hex_alpha(color, PCTILE_FIT_BAND_ALPHA),
                   name="10–90% band (bootstrap)"),
    ]

    hover = ("<b>%{fullData.name}</b><br>%{customdata[0]}<br>"
             "fit Q%{y:.0f} (band Q%{customdata[1]:.0f}–Q%{customdata[2]:.0f})<br>"
             "≈ $%{customdata[3]:.3s} "
             "($%{customdata[4]:.3s}–$%{customdata[5]:.3s})<extra></extra>")
    front = []
    ins = grid[grid <= t_last]
    if ins.size == 0 or ins[-1] < min(t_last, t_hi):
        ins = np.append(ins, min(t_last, t_hi))
    yc, _, _, cd = rows(ins)
    front.append(go.Scatter(
        x=ins, y=_round_trace_data(yc), mode="lines", customdata=cd,
        name=f"{label} cycle fit ({fit.period:.2f} yr)", hovertemplate=hover,
        line=dict(color=color, width=_FIT_LINE_WIDTH)))
    forecast = t_hi > t_last
    if forecast:
        fc = np.insert(grid[grid > t_last], 0, t_last)
        yc, _, _, cd = rows(fc)
        front.append(go.Scatter(
            x=fc, y=_round_trace_data(yc), mode="lines", customdata=cd,
            name=f"{label} cycle extrapolation", hovertemplate=hover,
            line=dict(color=color, width=_FIT_LINE_WIDTH, dash="dash")))

    anns, shapes = [], []
    tturns = _pc.turns(fit, t_lo, t_hi)
    for past, tag in ((True, "fitted turns"), (False, "forecast turns")):
        sel = [r for r in tturns if (r[0] <= t_last) == past]
        if not sel:
            continue
        xs = np.array([r[0] for r in sel])
        vs = np.array([r[1] for r in sel])
        qs, lf = _logfan(mdl, xs, sm)
        prices = _pc.fan_prices(lf, qs, vs)
        kinds = [r[2] for r in sel]
        sym = ["triangle-up" if k == "peak" else "triangle-down" for k in kinds]
        if past:
            cd = [[k, float(f"{pr:.4g}")] for k, pr in zip(kinds, prices)]
            tmpl = ("<b>fitted %{customdata[1]}</b> %{customdata[0]}<br>"
                    "Q%{y:.0f} · ≈ $%{customdata[2]:.3s}<extra></extra>")
            err = None
        else:
            timing = [_pc.turn_timing(params, x, k) for x, k in zip(xs, kinds)]
            cd = [[k, float(f"{pr:.4g}"), _mon_yr(a, m.genesis), _mon_yr(b, m.genesis)]
                  for k, pr, (a, b) in zip(kinds, prices, timing)]
            tmpl = ("<b>forecast %{customdata[1]}</b> %{customdata[0]}<br>"
                    "Q%{y:.0f} · ≈ $%{customdata[2]:.3s}<br>"
                    "turn date, 10–90% of refits: "
                    "%{customdata[3]} – %{customdata[4]}<extra></extra>")
            err = dict(type="data", symmetric=False,
                       array=[b - x for x, (a, b) in zip(xs, timing)],
                       arrayminus=[x - a for x, (a, b) in zip(xs, timing)],
                       color=color, thickness=1.5, width=5)
        front.append(go.Scatter(
            x=xs, y=vs, mode="markers", name=f"{label} {tag}",
            showlegend=False, customdata=cd, hovertemplate=tmpl,
            error_x=err,
            marker=dict(symbol=sym, size=_TURN_SIZE, color=color,
                        line=dict(color=PLOT_BG_COLOR, width=1))))
        if not past and not weak:
            for x, v, k, pr in zip(xs, vs, kinds, prices):
                anns.append(dict(
                    x=ax_x(float(x)), y=float(v), xref="x", yref="y",
                    text=f"{_mon_yr(x, m.genesis)}<br>Q{v:.0f} · {_money(pr)}",
                    showarrow=False, yshift=_LABEL_SHIFT if k == "peak" else -_LABEL_SHIFT,
                    font=dict(size=CHART_FONT_LEGEND, color=color),
                    bgcolor=_hex_alpha(PLOT_BG_COLOR, MC_LEGEND_BG_ALPHA),
                    bordercolor=color, borderwidth=1, borderpad=2))

    if forecast:
        shapes.append(dict(
            type="rect", xref="x", yref="paper", x0=t_last, x1=t_hi, y0=0, y1=1,
            layer="below", line=dict(width=0),
            fillcolor=_hex_alpha(PCTILE_FORECAST_SHADE, PCTILE_FORECAST_SHADE_ALPHA)))
    if p.get("cycle_censor"):
        a = _t_of(_pc.CENSOR_WINDOW[0], m.genesis)
        b = _t_of(_pc.CENSOR_WINDOW[1], m.genesis)
        if b > t_lo and a < t_hi:
            shapes.append(dict(
                type="rect", xref="x", yref="paper", x0=max(a, t_lo),
                x1=min(b, t_hi), y0=0, y1=1, layer="below", line=dict(width=0),
                fillcolor=_hex_alpha(PCTILE_RICH_COLOR, PCTILE_CENSOR_ALPHA)))
            anns.append(dict(
                x=ax_x((max(a, t_lo) + min(b, t_hi)) / 2), y=0.02, xref="x",
                yref="paper", text="withheld<br>from fit", showarrow=False,
                font=dict(size=CHART_FONT_LEGEND)))

    per = 1.0 / params[:, 2]
    p_lo, p_hi = np.percentile(per, [10, 90])
    if hind["n"]:
        hline = (f"hindcast: the wave beats a flat forecast in "
                 f"{hind['win_rate']:.0%} of {hind['n']} past 2-yr windows "
                 f"(RMSE {hind['rmse_wave']:.0f} vs {hind['rmse_flat']:.0f} pp)")
    else:
        hline = "hindcast: not enough history to score"
    if weak:
        head = (f"⚠ weak cycle in {label} (R² {fit.r2:.2f}): this model "
                f"already explains most of the cycle — the wave is not meaningful")
        box = _hex_alpha(PCTILE_RICH_COLOR, 0.30)
    else:
        head = (f"<b>{label}</b> · period {fit.period:.2f} yr "
                f"({p_lo:.2f}–{p_hi:.2f}) · amplitude {fit.A:.0f} pp "
                f"· R² {fit.r2:.2f}")
        box = _hex_alpha(PLOT_BG_COLOR, MC_LEGEND_BG_ALPHA)
    anns.append(dict(
        x=0.01, y=0.99, xref="paper", yref="paper", xanchor="left",
        yanchor="top", align="left", showarrow=False, text=f"{head}<br>{hline}",
        font=dict(size=CHART_FONT_LEGEND), bgcolor=box, borderpad=3))
    note = (f"band = 10–90% of {params.shape[0]} refits on 1-yr-block "
            f"resamples (periods 2–8 yr searched): where the wave is, not "
            f"the {hind['rmse_wave']:.0f} pp the percentile typically strays from it"
            if hind["n"] else
            f"band = 10–90% of {params.shape[0]} refits on 1-yr-block resamples")
    return back, front, shapes, anns, note + " · descriptive fit, not a forecast"


def build_percentile_figure(m: ModelData, p: dict[str, Any]) -> go.Figure:
    """Percentile of the actual price within each active quantized model's fan,
    over time. Non-quantized models (no fan) are skipped."""
    palette = _get_palette(p)

    _xmin = float(p["xmin"])
    if _xmin <= 2010:
        _xmin = 2010.5
    t_lo = max(yr_to_t(_xmin, m.genesis), 0.01)
    td = today_t(m.genesis)
    # Historical only: percentile needs a realized price, so cap the window at
    # today regardless of how far the range slider extends into the future.
    t_hi = min(yr_to_t(p["xmax"], m.genesis), td)

    mask = (m.price_years >= t_lo) & (m.price_years <= t_hi)
    t_data = m.price_years[mask]
    px_data = m.price_prices[mask]

    # Cycle fit: one model's wave + band. With the slider's right handle at or
    # past this calendar year the axis runs to today + horizon (the slider
    # itself stays capped and is never written); zoomed into the past, no
    # forecast region is drawn.
    cyc = cyc_mdl = None
    sigma_mode = p.get("sigma_mode", "constant")
    if p.get("cycle_fit"):
        cyc_key = p.get("cycle_model") or "qr"
        cyc_mdl = _cycle_model(cyc_key)
        if cyc_mdl is not None:
            cyc = _cycle_result(cyc_key, sigma_mode, bool(p.get("cycle_censor")),
                                str(getattr(_app_ctx, "_MODEL_FP", "")))
    t_edge = t_hi
    horizon = int(p.get("cycle_horizon") or 20)
    if cyc is not None and int(p["xmax"]) >= date.today().year:
        t_edge = td + horizon

    traces = []
    for model_key in p.get("active_models", []):
        if cyc is not None and model_key == cyc["model_key"]:
            continue        # drawn below, in the data role
        if model_key == "u1":
            um = p.get("user_model")
            mdl = UserModel.from_store_dict(um) if um else None
            if not mdl:
                continue
        else:
            mdl = _app_ctx.PRICE_MODELS.get(model_key)
            if not mdl:
                continue
        # Percentile is only meaningful for quantized (fan) models; skip s2f etc.
        if not getattr(mdl, "quantized", False):
            continue
        try:
            pct = _percentile_series(mdl, t_data, px_data,
                                     sigma_mode=p.get("sigma_mode", "constant"))
            if pct is None:
                continue
        except Exception:
            continue
        traces.append(go.Scatter(
            x=t_data, y=_round_trace_data(pct),
            mode="lines", name=mdl.legend_name,
            line=dict(color=_get_model_color(model_key, p),
                      width=TRACE_WIDTH_OVERLAY,
                      dash=getattr(mdl, "dash_style", "solid")),
            opacity=PCTILE_FIT_CONTEXT_OPACITY if cyc is not None else None,
        ))

    cyc_shapes, cyc_anns, cyc_note = [], [], ""
    if cyc is not None:
        actual = _percentile_series(cyc_mdl, t_data, px_data, sigma_mode=sigma_mode)
        back, front, cyc_shapes, cyc_anns, cyc_note = _cycle_overlay(
            cyc, cyc_mdl, p, m, t_lo, t_edge, p.get("xscale") == "log")
        mid = []
        if actual is not None and len(t_data):
            mid.append(go.Scatter(
                x=t_data, y=_round_trace_data(actual), mode="lines",
                name=f"{_cycle_label(cyc['model_key'], cyc_mdl)} percentile (actual)",
                line=dict(color=PCTILE_FIT_DATA_COLOR, width=1.0)))
        traces = back + traces + mid + front

    # ── colorblind-safe valuation zones + 50% median reference ────────────────
    shapes = [
        dict(type="rect", xref="paper", yref="y", x0=0, x1=1,
             y0=0, y1=_CHEAP_MAX, layer="below", line=dict(width=0),
             fillcolor=_hex_alpha(PCTILE_CHEAP_COLOR, _ZONE_ALPHA)),
        dict(type="rect", xref="paper", yref="y", x0=0, x1=1,
             y0=_RICH_MIN, y1=100, layer="below", line=dict(width=0),
             fillcolor=_hex_alpha(PCTILE_RICH_COLOR, _ZONE_ALPHA)),
        dict(type="line", xref="paper", yref="y", x0=0, x1=1, y0=50, y1=50,
             line=dict(color=FALLBACK_MODEL_GRAY, width=1.5, dash="dot")),
    ] + cyc_shapes
    if p.get("show_today") and t_lo <= td <= t_edge:
        shapes.extend(_today_line_shapes(
            td, 0, 1, palette.get("today_line", _TODAY_LINE_COLOR),
            glow=False, yref="paper"))

    end_yr = (date.today().year + horizon + 1) if t_edge > t_hi else p["xmax"]
    tick_ts, tick_lbls = _year_ticks(p["xmin"], end_yr, m.genesis,
                                     minor_grid=p.get("minor_grid"))
    filtered = [(t, lbl) for t, lbl in zip(tick_ts, tick_lbls) if t_lo <= t <= t_edge]
    tick_ts, tick_lbls = (list(x) for x in zip(*filtered)) if filtered else ([], [])

    title = "Price percentile within model fan (valuation oscillator)"
    if cyc is not None:
        title = ("Price percentile within model fan + cycle fit"
                 f"<br><sup>{cyc_note}</sup>")
    layout = _base_layout(title=title, xlabel="", ylabel="")
    layout["xaxis"].update(range=[t_lo, t_edge], tickvals=tick_ts,
                           ticktext=tick_lbls, tickangle=45)
    layout["yaxis"].update(range=[0, 100], ticksuffix="%",
                           ticklabelposition="inside", ticklabelshift=-5)
    if cyc is not None:
        # headroom for the turn labels above Q-peaks / below Q-troughs
        layout["yaxis"].update(range=[-12, 112], tickvals=[0, 25, 50, 75, 100])
    layout["shapes"] = shapes
    layout["showlegend"] = bool(p.get("show_legend", True))
    leg_pos = p.get("legend_pos", "outside")
    if leg_pos != "outside" and leg_pos in _MC_LEGEND_POS:
        pos = _MC_LEGEND_POS[leg_pos]
        layout.setdefault("legend", {})
        layout["legend"].update(
            x=pos["x"], y=pos["y"], xanchor=pos["xanchor"], yanchor=pos["yanchor"],
            bgcolor=_hex_alpha(PLOT_BG_COLOR, MC_LEGEND_BG_ALPHA))
    if p.get("xscale") == "log":
        layout["xaxis"].update(
            type="log",
            range=[math.log10(max(t_lo, 1e-10)), math.log10(max(t_edge, 1e-10))])

    if not traces:
        _empty_state_annotation(layout)
    elif cyc_anns:
        layout["annotations"] = cyc_anns

    _apply_sans_typography(layout)
    from figures.common import _uirevision_key
    layout.setdefault("uirevision", _uirevision_key(p, "pctile"))
    fig = go.Figure(data=traces, layout=go.Layout(**layout))
    _apply_final_steps(
        fig, p, "bub",
        hover_fmt="<b>%{fullData.name}</b><br>percentile %{y:.1f}%<extra></extra>",
        show_qr=False, show_mc=False, wm_pos="bottom-right",
    )
    return fig
