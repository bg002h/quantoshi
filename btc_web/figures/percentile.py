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
                    PCTILE_LIKELY_BAND_ALPHA, PCTILE_WEAK_BOX_ALPHA,
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


# ── Cycle fit (one basis element through ONE model's percentile) ─────────────
# Spec: docs/superpowers/specs/2026-11-10-percentile-cycle-fit-design.md
# The math lives in pctile_cycle.py; this section caches it per model / sigma
# mode / censor / element / data version and turns it into traces, labels and
# the summary lines that the info div under the chart shows (they ride in
# layout.meta.cycle_info, so every writer of the figure carries them).

_WEEK = 7.0 / 365.25
_TRANSPARENT = BLACK_A0   # width-0 band edges need an explicit transparent colour
_FIT_LINE_WIDTH = 2.4
_TURN_SIZE = 9
_LABEL_SHIFT = 24          # px between a turn marker and its label
_GRID_PAST_LAST = 30.0     # fixed band grid runs this far past the last close
#: Text labels per kind (peak / trough) on the next forecast turns; the rest
#: carry hover only. A log x-axis gets 1 (20 yr is ~1/5 of the width); a
#: linear one 3 / 2 / 1 at +10 / +20 / +30 yr (~70 px labels, review r2 M-2).
_LINEAR_LABELS = {10: 3, 20: 2, 30: 1}


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


def _sig4(a) -> list[float]:
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


def _cycle_label(mdl, key: str) -> str:
    return getattr(mdl, "legend_name", None) or key


@functools.lru_cache(maxsize=1)
def _tm_frame_t(genesis_str: str):
    import timemachine as tm
    g = pd.Timestamp(genesis_str)
    return np.array([_t_of(d, g) for d in tm.frames()])


@functools.lru_cache(maxsize=160)
def _qr_asof_weekly(frame_idx: int, fp: str):
    """Weekly percentile series of every close (t >= 1) against the Time
    Machine's as-of QR fan of one frame — a fan fitted only on data up to that
    frame. Cached per (frame, data version) and shared by every element and
    censor setting. QR is sigma-invariant, so it matches the live series."""
    del fp
    import timemachine as tm
    mdl = tm.asof_qr(frame_idx)
    if mdl is None:
        return None
    M = _app_ctx.M
    t_all = np.asarray(M.price_years, float)
    keep = t_all >= 1.0
    tw = t_all[keep][::_pc.STEP]
    pw = np.asarray(M.price_prices, float)[keep][::_pc.STEP]
    out = _percentile_series(mdl, tw, pw, sigma_mode="constant")
    return None if out is None else np.asarray(out, float)


def _qr_y_at(genesis, fp, n_weekly):
    """``y_at(cut)`` for an honest QR hindcast on the weekly series."""
    frame_t = _tm_frame_t(str(genesis))

    def y_at(cut):
        i = int(np.searchsorted(frame_t, cut, side="right")) - 1
        if i < 0:
            return None
        y = _qr_asof_weekly(i, fp)
        return None if y is None or y.size != n_weekly else y
    return y_at


@functools.lru_cache(maxsize=16)
def _cycle_result(model_key: str, sigma_mode: str, censor: bool, basis: str,
                  fp: str):
    """Fit, bootstrap, bands and hindcast of one basis element to one model's
    FULL-record percentile.

    ``fp`` (the model-data fingerprint) is only a cache key: a rebuilt
    model_data.pkl changes it, so a worker never serves a stale fit. The
    x-window never enters — the slider cannot change the fit. Bands live on a
    fixed weekly grid from the first close to last + 30 yr, so zoom and horizon
    never re-draw them. Returns None when the model is unavailable, the
    element unknown, or the fan missing.
    """
    mdl = _cycle_model(model_key)
    if mdl is None or basis not in _pc.BASES:
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
    if not np.all(np.isfinite(pct)) or t.size < 2 * 365:
        return None
    # censor removes the window from FITTING only (central fit, bootstrap,
    # each hindcast fit); scoring and coverage use every day (review r2 I-3)
    fit_mask = np.ones(t.size, bool)
    if censor:
        a = _t_of(_pc.CENSOR_WINDOW[0], M.genesis)
        b = _t_of(_pc.CENSOR_WINDOW[1], M.genesis)
        fit_mask &= ~((t >= a) & (t <= b))
    tf, yf = t[fit_mask], pct[fit_mask]
    fit = _pc.fit_cycle(tf, yf, basis=basis)
    params = _pc.bootstrap_params(tf, yf, fit)

    st = _pc.STEP
    tw, yw, mw = t[::st], pct[::st], fit_mask[::st]
    honest = False
    hind = None
    if model_key == "qr":
        try:
            import timemachine as tm
            if tm.available() and tm._load()["models"].get("qr"):
                hind = _pc.hindcast(tw, yw, basis=basis, step=1, fit_mask=mw,
                                    y_at=_qr_y_at(M.genesis, fp, tw.size))
                honest = bool(hind["n"])
        except Exception:
            hind, honest = None, False
    if not honest:
        hind = _pc.hindcast(tw, yw, basis=basis, step=1, fit_mask=mw)

    grid = np.arange(float(t[0]), float(t[-1]) + _GRID_PAST_LAST + _WEEK, _WEEK)
    fit_lo, fit_hi = _pc.band(params, grid, space=fit.space)
    lik_lo, lik_hi = _pc.likely_band(params, tf, yf, fit, grid)

    def cover(lo, hi):
        a_ = np.interp(tw, grid, lo)
        b_ = np.interp(tw, grid, hi)
        return float(((yw >= a_) & (yw <= b_)).mean())

    qs = np.asarray(mdl.quantiles, float)
    clamped = float(((pct <= qs[0] * 100 + 1e-6)
                     | (pct >= qs[-1] * 100 - 1e-6)).mean())
    return dict(model_key=model_key, t=t, pct=pct, fit=fit, params=params,
                hind=hind, honest=honest, weak=_pc.is_weak(fit),
                edge=_pc.beats_flat(hind),
                grid=grid, fit_lo=fit_lo, fit_hi=fit_hi,
                lik_lo=lik_lo, lik_hi=lik_hi,
                cov_fit=cover(fit_lo, fit_hi), cov_lik=cover(lik_lo, lik_hi),
                clamped=clamped, censor=censor)


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


def _cycle_info(cyc, label, mdl):
    """Plain-text summary lines for the info div (and tests)."""
    fit, hind, params = cyc["fit"], cyc["hind"], cyc["params"]
    blabel = _pc.BASES[fit.basis][3]
    r2_txt = "R\u00b2 (2019\u201320 excluded from the fit)" if cyc["censor"] else "R\u00b2"
    lines = []
    if cyc["weak"]:
        why = ("its period sits at the edge of the searched range"
               if fit.at_limit else f"{r2_txt} {fit.r2:.2f}")
        lines.append(f"\u26a0 Weak cycle in {label}'s percentile ({why}) \u2014 "
                     f"no forecast drawn. Fans that already model the cycle leave "
                     f"little to fit; try another element, or QR.")
    if fit.at_limit:
        per_txt = "period at the search limit (not an estimate)"
    elif fit.space == "cal":
        p_lo, p_hi = np.percentile(1.0 / params[:, 2], [10, 90])
        per_txt = f"period {fit.period:.2f} yr (10\u201390%: {p_lo:.2f}\u2013{p_hi:.2f})"
    else:
        p_lo, p_hi = np.percentile(np.exp(1.0 / params[:, 2]), [10, 90])
        per_txt = (f"turn ratio \u03bb {fit.period:.2f}: each turn comes at "
                   f"\u03bb\u00d7 the time since 2009 (10\u201390%: "
                   f"{p_lo:.2f}\u2013{p_hi:.2f})")
    lines.append(f"{label} \u00b7 {blabel} element \u00b7 {per_txt} \u00b7 "
                 f"amplitude {fit.A:.0f} pp \u00b7 {r2_txt} {fit.r2:.2f}")
    if not hind["n"]:
        lines.append("Out-of-sample check: not enough history.")
    elif cyc["honest"]:
        wins = int(round(hind["win_rate"] * hind["n"]))
        ln = (f"Out-of-sample (as-of fans, no look-ahead): the wave beat a flat "
              f"line in {wins} of {hind['n']} overlapping 2-yr windows; mean miss "
              f"{hind['rmse_wave']:.1f} vs {hind['rmse_flat']:.1f} pp.")
        if not cyc["edge"]:
            ln += " \u26a0 No out-of-sample edge over a flat line."
        lines.append(ln)
    else:
        ln = (f"Not checked out-of-sample: only QR has as-of fans. Against today's "
              f"fan, which has seen the later data (an upper bound), the wave's "
              f"mean miss is {hind['rmse_wave']:.1f} vs {hind['rmse_flat']:.1f} pp "
              f"for a flat line")
        ln += " \u2014 it loses even with that advantage." if not cyc["edge"] else "."
        lines.append(ln)
    lines.append(f"Bands: \u2018fit uncertainty\u2019 = 10\u201390% of "
                 f"{params.shape[0]} refits on 1-yr-block resamples (where the "
                 f"wave is; narrowest at turns \u2014 turn timing is the bars); "
                 f"\u2018likely range\u2019 adds the percentile's own scatter. "
                 f"In-sample the actual percentile sat inside them "
                 f"{cyc['cov_fit']:.0%} / {cyc['cov_lik']:.0%} of weeks. Prices "
                 f"read off today's {label} fan and hold only if that fan holds.")
    if cyc["clamped"] > 0.05:
        lines.append(f"{cyc['clamped']:.0%} of days sit at the edge of {label}'s "
                     f"fan (percentile pinned), which biases the fit.")
    lines.append("Descriptive fit, not a forecast.")
    return lines


def _cycle_overlay(cyc, mdl, p, m, t_lo, t_edge, xlog, forecast):
    """(back traces, front traces, shapes, annotations, info lines, subtitle)."""
    fit, params = cyc["fit"], cyc["params"]
    key = cyc["model_key"]
    label = _cycle_label(mdl, key)
    color = _get_model_color(key, p)
    sm = p.get("sigma_mode", "constant")
    sp = fit.space
    weak = cyc["weak"]
    t_last = float(cyc["t"][-1])
    grid = cyc["grid"]

    def ax_x(t):
        return math.log10(t) if xlog else t

    def at(tt, arr):
        return np.interp(tt, grid, arr)

    def rows(tt):
        """Wave + hover rows [fit lo, fit hi, likely lo, likely hi, $fit,
        $likely lo, $likely hi] at tt (date is prepended at customdata[0])."""
        yc = _pc.wave(fit.c, fit.A, fit.f, fit.phi, tt, sp)
        flo, fhi = at(tt, cyc["fit_lo"]), at(tt, cyc["fit_hi"])
        llo, lhi = at(tt, cyc["lik_lo"]), at(tt, cyc["lik_hi"])
        qs, lf = _logfan(mdl, tt, sm)
        pr = _pc.fan_prices(lf, qs, np.clip(yc, 0.0, 100.0))
        prl = _pc.fan_prices(lf, qs, llo)
        prh = _pc.fan_prices(lf, qs, lhi)
        cd = [[round(float(a), 1), round(float(b), 1), round(float(c), 1),
               round(float(d), 1), e, f_, g]
              for a, b, c, d, e, f_, g in zip(flo, fhi, llo, lhi, _sig4(pr),
                                               _sig4(prl), _sig4(prh))]
        return yc, cd

    t_end = t_edge if forecast else min(t_edge, t_last)
    G = grid[(grid >= t_lo) & (grid <= t_end)]
    if G.size == 0 or G[-1] < t_end:
        G = np.append(G, t_end)

    def band_pair(lo, hi, alpha, name):
        return [
            go.Scatter(x=G, y=_round_trace_data(at(G, hi)), mode="lines",
                       line=dict(width=0, color=_TRANSPARENT), hoverinfo="skip",
                       showlegend=False, name=f"{name} (upper)"),
            go.Scatter(x=G, y=_round_trace_data(at(G, lo)), mode="lines",
                       line=dict(width=0, color=_TRANSPARENT), hoverinfo="skip",
                       fill="tonexty", fillcolor=_hex_alpha(color, alpha),
                       name=name),
        ]

    back = []
    if not weak:
        back += band_pair(cyc["lik_lo"], cyc["lik_hi"], PCTILE_LIKELY_BAND_ALPHA,
                          "likely range (10–90%)")
    back += band_pair(cyc["fit_lo"], cyc["fit_hi"], PCTILE_FIT_BAND_ALPHA,
                      "fit uncertainty (10–90% of refits)")

    hover = ("<b>%{fullData.name}</b><br>%{customdata[0]}<br>"
             "fit Q%{y:.0f} ≈ $%{customdata[5]:.3s}<br>"
             "wave Q%{customdata[1]:.0f}–Q%{customdata[2]:.0f} (fit uncertainty)<br>"
             "likely Q%{customdata[3]:.0f}–Q%{customdata[4]:.0f} "
             "≈ $%{customdata[6]:.3s}–$%{customdata[7]:.3s}<extra></extra>")
    per_txt = (f"{fit.period:.2f} yr" if sp == "cal" else f"×{fit.period:.2f}")
    front = []
    ins = G[G <= t_last]
    if ins.size == 0 or ins[-1] < min(t_last, t_end):
        ins = np.append(ins, min(t_last, t_end))
    yc, cd = rows(ins)
    front.append(go.Scatter(
        x=ins, y=_round_trace_data(yc), mode="lines", customdata=cd,
        name=f"{label} cycle fit ({per_txt})", hovertemplate=hover,
        line=dict(color=color, width=_FIT_LINE_WIDTH,
                  dash="dot" if weak else "solid")))
    if forecast:
        fc = np.insert(G[G > t_last], 0, t_last)
        yc, cd = rows(fc)
        fc_name = f"{label} cycle extrapolation" + (
            "" if cyc["honest"] else " (not checked out-of-sample)")
        front.append(go.Scatter(
            x=fc, y=_round_trace_data(yc), mode="lines", customdata=cd,
            name=fc_name, hovertemplate=hover,
            line=dict(color=color, width=_FIT_LINE_WIDTH, dash="dash")))

    anns, shapes = [], []
    if not weak:
        tturns = _pc.turns(fit, t_lo, t_end)
        h = p.get("cycle_horizon")
        n_lab = 1 if xlog else _LINEAR_LABELS.get(h, 1)
        for past, tag in ((True, "fitted turns"), (False, "forecast turns")):
            sel = [r for r in tturns if (r[0] <= t_last) == past]
            if not sel:
                continue
            xs = np.array([r[0] for r in sel])
            vs = np.array([r[1] for r in sel])
            kinds = [r[2] for r in sel]
            qs, lf = _logfan(mdl, xs, sm)
            prices = _pc.fan_prices(lf, qs, vs)
            sym = ["triangle-up" if k == "peak" else "triangle-down" for k in kinds]
            if past:
                cd = [[k, float(f"{pr:.4g}")] for k, pr in zip(kinds, prices)]
                tmpl = ("<b>fitted %{customdata[1]}</b> %{customdata[0]}<br>"
                        "Q%{y:.0f} · ≈ $%{customdata[2]:.3s}<extra></extra>")
                err = None
            else:
                # cycle index from the central fit's last in-sample turn of
                # each kind (where replicates agree); without one, match
                # nearest (j = 0 at the turn itself)
                past_turns = _pc.turns(fit, float(cyc["t"][0]), t_last)
                anchor = {k: max((r[0] for r in past_turns if r[2] == k),
                                 default=None) for k in ("peak", "trough")}
                seen = {"peak": 0, "trough": 0}
                idx, timing = [], []
                for x, k in zip(xs, kinds):
                    idx.append(seen[k])
                    seen[k] += 1
                    t_a, j = ((anchor[k], seen[k]) if anchor[k] is not None
                              else (float(x), 0))
                    timing.append(_pc.turn_timing(params, t_a, j, k, float(x),
                                                  space=sp))
                cd = [[k, float(f"{pr:.4g}"), _mon_yr(a, m.genesis),
                       _mon_yr(b, m.genesis)]
                      for k, pr, (a, b) in zip(kinds, prices, timing)]
                tmpl = ("<b>forecast %{customdata[1]}</b> %{customdata[0]}<br>"
                        "Q%{y:.0f} · ≈ $%{customdata[2]:.3s}<br>"
                        "turn date, 10–90% of refits: "
                        "%{customdata[3]} – %{customdata[4]}<extra></extra>")
                err = dict(type="data", symmetric=False,
                           array=[max(0.0, b - x) for x, (a, b) in zip(xs, timing)],
                           arrayminus=[max(0.0, x - a) for x, (a, b) in zip(xs, timing)],
                           color=color, thickness=1.5, width=5)
                for x, v, k, pr, j in zip(xs, vs, kinds, prices, idx):
                    if j >= n_lab:
                        continue
                    anns.append(dict(
                        x=ax_x(float(x)), y=float(v), xref="x", yref="y",
                        text=f"{_mon_yr(x, m.genesis)}<br>Q{v:.0f} · {_money(pr)}",
                        showarrow=False,
                        yshift=_LABEL_SHIFT if k == "peak" else -_LABEL_SHIFT,
                        font=dict(size=CHART_FONT_LEGEND, color=color),
                        bgcolor=_hex_alpha(PLOT_BG_COLOR, MC_LEGEND_BG_ALPHA),
                        bordercolor=color, borderwidth=1, borderpad=2))
            front.append(go.Scatter(
                x=xs, y=vs, mode="markers", name=f"{label} {tag}",
                showlegend=False, customdata=cd, hovertemplate=tmpl,
                error_x=err,
                marker=dict(symbol=sym, size=_TURN_SIZE, color=color,
                            line=dict(color=PLOT_BG_COLOR, width=1))))
    if not weak and cyc["edge"] is False:
        anns.append(dict(
            x=0.01, y=0.99, xref="paper", yref="paper", xanchor="left",
            yanchor="top", showarrow=False,
            text="⚠ no out-of-sample edge over a flat line (details below)",
            font=dict(size=CHART_FONT_LEGEND),
            bgcolor=_hex_alpha(FALLBACK_MODEL_GRAY, PCTILE_WEAK_BOX_ALPHA),
            borderpad=3))
    if weak:
        anns.append(dict(
            x=0.01, y=0.99, xref="paper", yref="paper", xanchor="left",
            yanchor="top", showarrow=False,
            text=f"⚠ weak cycle in {label}'s percentile — no forecast "
                 f"(details below the chart)",
            font=dict(size=CHART_FONT_LEGEND),
            bgcolor=_hex_alpha(FALLBACK_MODEL_GRAY, PCTILE_WEAK_BOX_ALPHA),
            borderpad=3))

    if forecast:
        shapes.append(dict(
            type="rect", xref="x", yref="paper", x0=t_last, x1=t_edge, y0=0, y1=1,
            layer="below", line=dict(width=0),
            fillcolor=_hex_alpha(PCTILE_FORECAST_SHADE, PCTILE_FORECAST_SHADE_ALPHA)))
    if cyc["censor"]:
        a = _t_of(_pc.CENSOR_WINDOW[0], m.genesis)
        b = _t_of(_pc.CENSOR_WINDOW[1], m.genesis)
        if b > t_lo and a < t_edge:
            shapes.append(dict(
                type="rect", xref="x", yref="paper", x0=max(a, t_lo),
                x1=min(b, t_edge), y0=0, y1=1, layer="below", line=dict(width=0),
                fillcolor=_hex_alpha(PCTILE_RICH_COLOR, PCTILE_CENSOR_ALPHA)))
            anns.append(dict(
                x=ax_x((max(a, t_lo) + min(b, t_edge)) / 2), y=0.02, xref="x",
                yref="paper", text="withheld<br>from fit", showarrow=False,
                font=dict(size=CHART_FONT_LEGEND)))

    sub = (f"{label} · {_pc.BASES[fit.basis][3]} element · inner band: "
           f"fit uncertainty · outer band: likely range · descriptive, "
           f"not a forecast")
    return back, front, shapes, anns, _cycle_info(cyc, label, mdl), sub


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
                                p.get("cycle_basis") or _pc.DEFAULT_BASIS,
                                str(getattr(_app_ctx, "_MODEL_FP", "")))
    horizon = p.get("cycle_horizon")
    horizon = horizon if horizon in _pc.HORIZONS else _pc.DEFAULT_HORIZON
    # Forecast only for a fit that is not weak, and only while the right
    # handle reaches today (2027 = 1 Jan 2027 does; 2026 = 1 Jan 2026 not).
    forecast = (cyc is not None and not cyc["weak"]
                and yr_to_t(p["xmax"], m.genesis) >= td)
    t_edge = td + horizon if forecast else t_hi

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

    cyc_shapes, cyc_anns, cyc_info, cyc_sub = [], [], [], ""
    if cyc is not None:
        actual = _percentile_series(cyc_mdl, t_data, px_data, sigma_mode=sigma_mode)
        back, front, cyc_shapes, cyc_anns, cyc_info, cyc_sub = _cycle_overlay(
            cyc, cyc_mdl, p, m, t_lo, t_edge, p.get("xscale") == "log", forecast)
        mid = []
        if actual is not None and len(t_data):
            mid.append(go.Scatter(
                x=t_data, y=_round_trace_data(actual), mode="lines",
                name=f"{_cycle_label(cyc_mdl, cyc['model_key'])} percentile (actual)",
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
                 f"<br><sup>{cyc_sub}</sup>")
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
    if cyc is not None:
        layout["meta"] = {"cycle_info": cyc_info}

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
