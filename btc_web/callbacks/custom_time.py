"""Custom Time Axis callbacks — section 5 error cases encoded here.

See docs/superpowers/specs/2026-04-13-custom-time-axis-design.md §5.
"""
from __future__ import annotations

import logging
import math
import time
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, html, no_update
from dash.exceptions import PreventUpdate

import _app_ctx
from colors import (
    DIM_TEXT, MODEL_TRACE_COLORS, TRACE_WIDTH, FALLBACK_MODEL_GRAY,
    UI_FONT_SM,
)
from engines import custom_fit as cf
from layout.common import _bands_to_qs
from _custom_time_presets import CAL_PRESET_BY_KEY, BLK_PRESET_BY_KEY

_LOG = logging.getLogger("custom_time")
_CAP_DATE = date(2016, 1, 1)
_T_PLOT_POINTS = 400


def _eval_fit_on_range(r, t_min: float, t_max: float):
    """Re-evaluate a FitResult's curves on an extended t range using stored
    slope/intercept params. Used to extrapolate fit lines out to whatever
    the bub-xrange upper bound requests, not just 1.1 × data.max().

    Returns (t_plot, y_plot) with the same shape as the FitResult
    (np.ndarray for PL/BM-floor/Exp, dict{q: np.ndarray} for QR).
    """
    if r.name == "Exp":
        t_plot = np.linspace(t_min, t_max, _T_PLOT_POINTS)
    else:
        # Log-log models need strictly positive t
        lo = max(t_min, 1e-6)
        hi = max(t_max, lo * 2)
        t_plot = np.logspace(np.log10(lo), np.log10(hi), _T_PLOT_POINTS)

    if isinstance(r.y_plot, dict):
        slopes = r.params.get("slopes", {})
        intercepts = r.params.get("intercepts", {})
        log_t = np.log10(t_plot)
        y_plot = {q: slopes[q] * log_t + intercepts[q] for q in r.y_plot}
        return t_plot, y_plot

    slope = r.params.get("slope")
    intercept = r.params.get("intercept")
    if slope is None or intercept is None:
        # fit_support failure or similar — return the stored arrays unchanged
        return r.t_plot, r.y_plot
    if r.name == "Exp":
        y_plot = slope * t_plot + intercept
    else:
        y_plot = slope * np.log10(t_plot) + intercept
    return t_plot, y_plot


# ──────────────────────────────────────────────────────────────────────────
# Clientside: toggle cta-body visibility from the Activate checklist
# ──────────────────────────────────────────────────────────────────────────
_app_ctx.app.clientside_callback(
    "function(v) { return (v && v.length) ? {} : {display:'none'}; }",
    Output("cta-body", "style"),
    Input("cta-active", "value"),
)

# Clientside: toggle cta-t0-cal-wrap ↔ cta-t0-blk-wrap on scale radio
_app_ctx.app.clientside_callback(
    """
    function(scale) {
        var show = {};
        var hide = {display: 'none'};
        var isCal = (scale === 'calendar');
        return [isCal ? show : hide, isCal ? hide : show];
    }
    """,
    Output("cta-t0-cal-wrap", "style"),
    Output("cta-t0-blk-wrap", "style"),
    Input("cta-scale", "value"),
)

# Clientside: reveal custom date picker when dropdown == "custom"
_app_ctx.app.clientside_callback(
    "function(v) { return (v === 'custom') ? {} : {display:'none'}; }",
    Output("cta-t0-cal-custom-wrap", "style"),
    Input("cta-t0-cal", "value"),
)
_app_ctx.app.clientside_callback(
    "function(v) { return (v === 'custom') ? {} : {display:'none'}; }",
    Output("cta-t0-blk-custom-wrap", "style"),
    Input("cta-t0-blk", "value"),
)

# Clientside: when Custom Time Axis is active in calendar mode AND the chosen
# t₀ is before 2010, drop the bub-xrange slider's min so the user can pan
# back to t₀ year AND push the left handle's value to the new min so the
# server callback (which reads bub-xrange.value as Input, not .min) actually
# re-fires. When deactivating with a left handle currently below 2010, snap
# it back to 2010 so the standard bubble view doesn't break. Runs clientside
# so no server round-trip on state transitions.
_app_ctx.app.clientside_callback(
    """
    function(active, scale, cal_preset, cal_custom, current_value) {
        var DEFAULT_MIN = 2010;
        var MAX_YR = 2080;
        var PRESET_YEARS = {
            whitepaper: 2008, genesis: 2009, optimal: 2009,
            nls: 2009, pizza: 2010, mtgox: 2010, parity: 2011
        };
        var NU = window.dash_clientside.no_update;

        var effective_min = DEFAULT_MIN;
        var is_active = active && active.indexOf && active.indexOf("yes") !== -1;
        if (is_active && scale === "calendar") {
            var t0_year = null;
            if (cal_preset === "custom" && cal_custom) {
                var yr = parseInt(String(cal_custom).slice(0, 4), 10);
                if (!isNaN(yr)) t0_year = yr;
            } else if (PRESET_YEARS[cal_preset] !== undefined) {
                t0_year = PRESET_YEARS[cal_preset];
            }
            if (t0_year !== null && t0_year < DEFAULT_MIN) {
                effective_min = t0_year;
            }
        }

        // Build marks every 10 years from floor(effective_min/10)*10 to MAX_YR
        var start = Math.floor(effective_min / 10) * 10;
        var marks = {};
        for (var y = start; y <= MAX_YR; y += 10) {
            if (y >= effective_min) {
                var suffix = String(y % 100);
                if (suffix.length < 2) suffix = "0" + suffix;
                marks[String(y)] = "'" + suffix;
            }
        }

        // Push value[0] to match the new effective min when activating, and
        // snap it back to DEFAULT_MIN when deactivating a previously-extended
        // slider. Leave value untouched otherwise so user's manual adjustments
        // stick.
        var new_value = NU;
        if (current_value && current_value.length === 2) {
            var lo = current_value[0];
            var hi = current_value[1];
            if (effective_min < DEFAULT_MIN) {
                // Activating with pre-2010 t₀ — pull left handle to new min
                // unless user already dragged it below the new effective_min
                // (in which case respect their choice).
                if (lo > effective_min) {
                    new_value = [effective_min, hi];
                }
            } else {
                // Deactivating OR activating with a post-2010 t₀ — if the
                // slider's left handle is currently below 2010, snap it up.
                if (lo < DEFAULT_MIN) {
                    new_value = [DEFAULT_MIN, Math.max(hi, DEFAULT_MIN)];
                }
            }
        }
        return [effective_min, marks, new_value];
    }
    """,
    Output("bub-xrange", "min"),
    Output("bub-xrange", "marks"),
    Output("bub-xrange", "value", allow_duplicate=True),
    Input("cta-active", "value"),
    Input("cta-scale", "value"),
    Input("cta-t0-cal", "value"),
    Input("cta-t0-cal-custom", "date"),
    State("bub-xrange", "value"),
    prevent_initial_call=True,
)


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────

def _resolve_t0(scale, cal_preset, cal_custom, blk_preset, blk_custom):
    """Return (t0_value, error_msg). t0_value is None when error_msg is set."""
    if scale == "calendar":
        if cal_preset == "custom":
            if not cal_custom:
                return None, "Enter a date to fit."
            ts = pd.Timestamp(cal_custom).date()
            if ts >= _CAP_DATE:
                return None, "Custom t\u2080 must be before 2016-01-01."
            return ts.isoformat(), None
        return CAL_PRESET_BY_KEY[cal_preset][0].isoformat(), None

    # block mode
    if cf._BLOCKS is None:
        return None, ("\u26a0 Block mode unavailable: "
                      "BitcoinBlocksDaily.csv missing.")
    if blk_preset == "custom":
        if blk_custom is None:
            return None, "Enter a block number to fit."
        if cf._BLOCK_CAP is not None and int(blk_custom) > cf._BLOCK_CAP:
            return None, (f"Custom block must be \u2264 {cf._BLOCK_CAP} "
                          f"(before 2016).")
        return int(blk_custom), None
    return BLK_PRESET_BY_KEY[blk_preset][0], None


def _build_figure(results: dict, scale: str, t0_label: str,
                   xscale: str = "log", yscale: str = "log",
                   xrange=None, yrange=None, auto_y=True,
                   show_legend: bool = True) -> go.Figure:
    """Build the Custom Time Axis figure. Uses the numeric `t` (years since
    t₀ or block offset) as the x-axis so log scaling works correctly.
    Honors the Tab 1 Axes & Range panel + Display "Show legend" checkbox.
    """
    fig = go.Figure()
    if cf._DATES is None or cf._PRICES is None:
        return fig

    # Hover templates match the standard bubble tab format (see
    # figures/common.py:_HOVER_FMT_USD). customdata[0] = date string in
    # calendar mode, or (blockheight, date_str) in block mode.
    if scale == "calendar":
        _hover_scatter = (
            "<b>%{fullData.name}</b><br>%{customdata[0]}<br>"
            "$%{y:,.0f}<extra></extra>")
        _hover_line = _hover_scatter  # fit lines also show date + USD
    else:
        _hover_scatter = (
            "<b>%{fullData.name}</b><br>block %{x:,.0f}<br>"
            "%{customdata[0]}<br>$%{y:,.0f}<extra></extra>")
        _hover_line = (
            "<b>%{fullData.name}</b><br>block %{x:,.0f}<br>"
            "$%{y:,.0f}<extra></extra>")

    # x values for the price scatter: years since t₀ OR raw block offset
    if scale == "calendar":
        t0_ts = pd.Timestamp(t0_label)
        x_scatter_all = (
            (cf._DATES - t0_ts).days.values.astype(float) / 365.25)
        date_strs_all = cf._DATES.strftime("%Y-%m-%d").values
    else:
        t0_ts = None
        x_scatter_all = (cf._BLOCKS - int(t0_label)).astype(float)
        date_strs_all = cf._DATES.strftime("%Y-%m-%d").values

    # On log-x, points with x <= 0 vanish; mask them so the legend count
    # matches what the user actually sees, and drop them from the scatter
    # so Plotly doesn't log-warn.
    if xscale == "log":
        vis_mask = x_scatter_all > 0
    else:
        vis_mask = np.ones_like(x_scatter_all, dtype=bool)
    x_scatter = x_scatter_all[vis_mask]
    p_scatter = cf._PRICES[vis_mask]
    d_scatter = date_strs_all[vis_mask]
    hidden_n = int((~vis_mask).sum())

    scatter_label = f"Price (n={len(p_scatter):,})"
    if hidden_n:
        scatter_label += f" [−{hidden_n:,} before t\u2080]"
    fig.add_trace(go.Scatter(
        x=x_scatter, y=p_scatter, mode="markers",
        marker=dict(size=3, color=DIM_TEXT, opacity=0.5),
        name=scatter_label,
        customdata=[[d] for d in d_scatter],
        hovertemplate=_hover_scatter,
    ))

    # Determine the t range the fit lines should cover. We want the lines
    # to extrapolate BACKWARD to cover whatever bub-xrange[0] points at
    # (e.g. user drags slider to 2005 when t₀ is earlier) AND forward to
    # bub-xrange[1] (e.g. 2080).
    t_min_data = float(x_scatter.min()) if len(x_scatter) else 1.0 / 365.25
    t_max_data = float(x_scatter_all.max()) if len(x_scatter_all) else 1.0
    t_max_ext = t_max_data * 1.1
    t_min_ext = t_min_data
    if scale == "calendar" and xrange is not None and len(xrange) == 2:
        t0_ts_ext = pd.Timestamp(t0_label)
        t0_dy = t0_ts_ext.year + (t0_ts_ext.dayofyear - 1) / 365.25
        xrange_lo = float(xrange[0]) - t0_dy
        xrange_hi = float(xrange[1]) - t0_dy
        if xrange_hi > t_max_ext:
            t_max_ext = xrange_hi
        # Backward extrapolation: extend t_min_ext to whichever is lower,
        # xrange_lo or t_min_data. Negative xrange_lo (user panned left of
        # t₀) is accepted here; _eval_fit_on_range clamps log-log models to
        # a tiny positive floor internally, while Exp (linear-t) can accept
        # any real t. The log-x axis clamp below still snaps the AXIS
        # lower bound to a positive value, so negative-t regions of the
        # fit line simply get clipped off-canvas in log-x mode.
        if xrange_lo < t_min_data:
            t_min_ext = xrange_lo
    # Block mode: bub-xrange is in years and doesn't apply.

    colors = {
        "PL":       MODEL_TRACE_COLORS.get("pl", FALLBACK_MODEL_GRAY),
        "QR":       MODEL_TRACE_COLORS.get("qr", FALLBACK_MODEL_GRAY),
        "BM floor": MODEL_TRACE_COLORS.get("bub", FALLBACK_MODEL_GRAY),
        "Exp":      MODEL_TRACE_COLORS.get("exp", FALLBACK_MODEL_GRAY),
    }

    # Pre-compute per-t_plot date strings for calendar mode so each fit
    # trace's hover tooltip shows the date corresponding to that x position.
    def _dates_for_t(t_plot_arr):
        if scale == "calendar":
            ts = t0_ts + pd.to_timedelta(t_plot_arr * 365.25, unit="D")
            return [[d.strftime("%Y-%m-%d")] for d in ts]
        return None  # block mode uses %{x} from the trace x values

    for r in results.values():
        if r is None:
            continue
        color = colors.get(r.name, FALLBACK_MODEL_GRAY)
        label_n = f"{r.n_samples:,}"
        # R² shown when finite; BM-floor is NaN by design (support line sits
        # below mean so vs-mean R² is misleading), QR is NaN because each
        # quantile is a separate fit with no single-quantity R².
        r2_str = ""
        if r.r2 is not None and np.isfinite(r.r2):
            r2_str = f", R\u00b2={r.r2:.3f}"
        # Exp slope is translation-invariant (log(p) = a + b·t → shifting t
        # by Δ only changes a). Annotate so users don't expect the slope to
        # move when they drag t₀.
        name_suffix = ""
        if r.name == "Exp":
            name_suffix = " \u00b7 slope is t\u2080-invariant"
        # Re-evaluate the fit on the extended t range so the curves extend
        # out to whatever bub-xrange requests (e.g. 2080).
        t_plot_ext, y_plot_ext = _eval_fit_on_range(r, t_min_ext, t_max_ext)
        trace_customdata = _dates_for_t(t_plot_ext)
        if isinstance(y_plot_ext, dict):
            # QR: per-quantile slopes live in r.params["slopes"][q]
            slopes = r.params.get("slopes", {}) if r.params else {}
            for q, y in y_plot_ext.items():
                slope_q = slopes.get(q)
                slope_str = (f", b={slope_q:.3f}" if slope_q is not None
                              and np.isfinite(slope_q) else "")
                fig.add_trace(go.Scatter(
                    x=t_plot_ext, y=10 ** y, mode="lines",
                    line=dict(color=color, width=TRACE_WIDTH),
                    name=f"{r.name} Q{int(q*100)}% (n={label_n}{slope_str})",
                    legendgroup=r.name,
                    customdata=trace_customdata,
                    hovertemplate=_hover_line,
                ))
        else:
            slope = r.params.get("slope") if r.params else None
            slope_str = (f", b={slope:.3f}" if slope is not None
                          and np.isfinite(slope) else "")
            fig.add_trace(go.Scatter(
                x=t_plot_ext, y=10 ** y_plot_ext, mode="lines",
                line=dict(color=color, width=TRACE_WIDTH),
                name=f"{r.name} (n={label_n}{slope_str}{r2_str}){name_suffix}",
                customdata=trace_customdata,
                hovertemplate=_hover_line,
            ))

    yaxis_cfg = dict(type=yscale, title="USD")
    if not auto_y and yrange is not None and len(yrange) == 2:
        # bub-yrange is in log10 price units — apply directly when log,
        # or convert to linear price range when linear.
        if yscale == "log":
            yaxis_cfg["range"] = list(yrange)  # log axes take log-space range
        else:
            yaxis_cfg["range"] = [10 ** yrange[0], 10 ** yrange[1]]

    xaxis_title = ("Year" if scale == "calendar"
                    else f"Blocks since t\u2080 = {t0_label}")
    xaxis_cfg = dict(title=xaxis_title)

    # Calendar-year tick labels: the underlying x values are numeric
    # years-since-t₀ so log-log works, but we relabel the ticks to show
    # calendar years (2010, 2020, 2030, ...) matching the standard bubble
    # chart's convention (figures/bubble.py:410-414).
    if scale == "calendar":
        t0_ts_tk = pd.Timestamp(t0_label)
        t0_dy_tk = t0_ts_tk.year + (t0_ts_tk.dayofyear - 1) / 365.25
        # Determine the calendar year range to tick over
        if xrange is not None and len(xrange) == 2:
            tick_yr_lo = int(xrange[0])
            tick_yr_hi = int(xrange[1])
        else:
            tick_yr_lo = int(t0_dy_tk) + 1
            tick_yr_hi = int(pd.Timestamp.today().year) + 5
        span = max(tick_yr_hi - tick_yr_lo, 1)
        step = 1 if span <= 10 else (2 if span <= 20 else
                                      (5 if span <= 60 else 10))
        first = ((tick_yr_lo + step - 1) // step) * step
        tick_years = list(range(first, tick_yr_hi + 1, step))
        tick_vals = [float(y) - t0_dy_tk for y in tick_years]
        tick_labels = [str(y) for y in tick_years]
        # Log-x can't display non-positive ticks; drop them
        if xscale == "log":
            positive = [(v, lbl) for v, lbl in zip(tick_vals, tick_labels)
                         if v > 0]
            tick_vals = [p[0] for p in positive]
            tick_labels = [p[1] for p in positive]
        if tick_vals:
            xaxis_cfg["tickvals"] = tick_vals
            xaxis_cfg["ticktext"] = tick_labels

    # Always honor bub-xrange (calendar-year integers) in calendar mode.
    # Linear mode: apply [x_lo, x_hi] directly, even if x_lo < 0.
    # Log mode: if x_lo ≤ 0 (e.g. t₀ is after the slider's lower year),
    # clamp to the smallest visible data point so the axis still covers
    # the user's intended range instead of decaying into an empty decade.
    xaxis_cfg["type"] = xscale if scale == "calendar" else "linear"
    if scale == "calendar" and xrange is not None and len(xrange) == 2:
        t0_ts = pd.Timestamp(t0_label)
        t0_decimal_year = t0_ts.year + (t0_ts.dayofyear - 1) / 365.25
        x_lo = float(xrange[0]) - t0_decimal_year
        x_hi = float(xrange[1]) - t0_decimal_year
        if x_hi > 0 and x_hi > x_lo:
            if xscale == "log":
                # Respect x_lo when positive (user may be panning BACKWARD
                # past the first data point — the fit line extrapolates
                # there via t_min_ext). Only floor when x_lo ≤ 0, since
                # log-x can't cross zero; fall back to the smallest visible
                # data point or a 1-day minimum.
                if x_lo > 0:
                    effective_lo = x_lo
                else:
                    effective_lo = (
                        float(x_scatter.min()) if len(x_scatter) > 0
                        else 1.0 / 365.25)
                if effective_lo < x_hi:
                    xaxis_cfg["range"] = [
                        math.log10(effective_lo), math.log10(x_hi)]
            else:
                xaxis_cfg["range"] = [x_lo, x_hi]

    fig.update_layout(
        yaxis=yaxis_cfg,
        xaxis=xaxis_cfg,
        title=f"Custom Time Axis \u2014 t\u2080 = {t0_label}",
        template="plotly_white",
        margin=dict(l=60, r=30, t=60, b=60),
        showlegend=show_legend,
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────
# Main server callback
# ──────────────────────────────────────────────────────────────────────────

@callback(
    Output("bubble-graph", "figure", allow_duplicate=True),
    Output("cta-status", "children"),
    Output("bub-redraw-tick", "data"),
    Input("cta-active", "value"),
    Input("cta-scale", "value"),
    Input("cta-t0-cal", "value"),
    Input("cta-t0-cal-custom", "date"),
    Input("cta-t0-blk", "value"),
    Input("cta-t0-blk-custom", "value"),
    Input("cta-weighting", "value"),
    Input("cta-models", "value"),
    # Tab 1 Axes & Range — re-fit when the user tweaks them via Input
    Input("bub-xscale", "value"),
    Input("bub-yscale", "value"),
    Input("bub-xrange", "value"),
    Input("bub-yrange", "value"),
    Input("bub-auto-y", "value"),
    # Tab 1 Display — read "show_legend" so the custom figure honors the
    # same checkbox as the standard bubble chart.
    Input("bub-toggles", "value"),
    # Tab 1 Projection Quantiles — QR fit uses the user's selection.
    Input("bub-qs", "value"),
    Input("bub-qs-mode", "value"),
    Input("bub-qs-adv", "value"),
    State("bub-redraw-tick", "data"),
    prevent_initial_call=True,
)
def custom_time_callback(active, scale, cal_preset, cal_custom,
                          blk_preset, blk_custom, weighting, models,
                          bub_xscale, bub_yscale, bub_xrange, bub_yrange,
                          bub_auto_y, bub_toggles,
                          bub_qs, bub_qs_mode, bub_qs_adv, tick):
    """Route Custom Time Axis state changes to the bubble figure.

    On activate: computes a fresh custom figure.
    On deactivate: bumps the redraw tick so update_bubble re-fires and writes
        the standard figure back. Status is updated in-place.
    On error: preserves the previous figure via no_update, writes an error
        message to cta-status.
    """
    try:
        # 1. Deactivate → bump tick, preserve figure, restore status
        if not active or "yes" not in active:
            return no_update, "Standard view restored.", (tick or 0) + 1

        # 2. Module readiness (block-mode only)
        if scale == "block" and cf._BLOCKS is None:
            return (no_update,
                    ("\u26a0 Block mode unavailable: "
                     "BitcoinBlocksDaily.csv missing."),
                    no_update)

        # 3. Input validity
        if not models:
            return no_update, "Select at least one model to fit.", no_update

        # Guard: picker is on "custom" but no value yet
        if scale == "calendar" and cal_preset == "custom" and not cal_custom:
            return no_update, "Enter a date to fit.", no_update
        if scale == "block" and blk_preset == "custom" and blk_custom is None:
            return no_update, "Enter a block number to fit.", no_update

        t0, err = _resolve_t0(scale, cal_preset, cal_custom,
                               blk_preset, blk_custom)
        if err:
            return no_update, err, no_update

        # 4. Build fit input
        t_start = time.perf_counter()
        fi = cf.build_fit_input(scale=scale, t0=t0, weighting=weighting)

        # 5. Run each selected model
        # Resolve the user's Projection Quantiles selection for QR:
        # default mode → bub-qs band names → float pairs via _bands_to_qs;
        # advanced mode → bub-qs-adv is already a list of floats.
        if bub_qs_mode and "advanced" in bub_qs_mode:
            qr_quantiles = tuple(sorted(bub_qs_adv or [])) or None
        else:
            bands = _bands_to_qs(bub_qs or [])
            qr_quantiles = tuple(bands) if bands else None

        results = {}
        if "pl" in models:
            results["pl"] = cf.fit_pl(fi)
        if "qr" in models:
            if qr_quantiles:
                results["qr"] = cf.fit_qr(fi, quantiles=qr_quantiles)
            else:
                # User deselected every quantile — skip QR rather than fit
                # the default 9-quantile set against a list they've cleared.
                results["qr"] = None
        if "bm_floor" in models:
            results["bm_floor"] = cf.fit_bm_floor(fi)
        if "exp" in models:
            results["exp"] = cf.fit_exp(fi)

        elapsed_ms = int((time.perf_counter() - t_start) * 1000)
        if elapsed_ms > 5000:
            _LOG.warning("custom_fit slow: %dms params=%s",
                          elapsed_ms,
                          {"scale": scale, "t0": t0, "weighting": weighting})

        # 6. Build figure + status (honor Tab 1 Axes & Range panel)
        auto_y = bool(bub_auto_y and "yes" in bub_auto_y)
        show_legend = bool(bub_toggles and "show_legend" in bub_toggles)
        fig = _build_figure(
            results, scale, str(t0),
            xscale=bub_xscale or "log",
            yscale=bub_yscale or "log",
            xrange=bub_xrange,
            yrange=bub_yrange,
            auto_y=auto_y,
            show_legend=show_legend,
        )
        sample_counts = [r.n_samples for r in results.values() if r is not None]
        total_n = max(sample_counts, default=0)
        skipped = sum(1 for r in results.values() if r is None)
        status_parts = [
            f"Fit on {total_n:,} samples from t\u2080={t0}.",
            f"{elapsed_ms}ms.",
        ]
        if skipped:
            status_parts.append(f"{skipped} model(s) skipped (see legend).")
        # Surface any FitResult.note diagnostics so the user sees why
        # weighting was degraded, why QR fell back to 3 quantiles, etc.
        notes = [f"{r.name}: {r.note}" for r in results.values()
                  if r is not None and r.note]
        if notes:
            status_parts.append("\u00b7 " + " \u00b7 ".join(notes))
        return fig, " ".join(status_parts), no_update

    except Exception as exc:
        _LOG.error("custom_fit crash: %s", exc, exc_info=True)
        return (no_update,
                f"\u26a0 Internal error: {type(exc).__name__}",
                no_update)


# ══════════════════════════════════════════════════════════════════════════
# $1M Projection Table Modal
# ══════════════════════════════════════════════════════════════════════════

# Live-fit helpers removed 2026-04-14: the projection table is now a
# static JSON produced by tools/build_projection_table.py (rebuilt monthly
# via daily_update.sh). The modal reads _projection_table.json instead of
# running 96 fits per click.


from pathlib import Path as _Path
_PROJECTION_JSON_PATH = (
    _Path(_app_ctx.__file__).resolve().parent / "_projection_table.json"
)


def _load_projection_blob():
    """Load the pre-computed projection table from _projection_table.json.
    Returns None if the file is missing or unreadable — the UI will fall
    back to a 'not available' message."""
    import json
    try:
        return json.loads(_PROJECTION_JSON_PATH.read_text())
    except Exception as exc:
        _LOG.warning("projection table JSON missing/unreadable: %s", exc)
        return None


def _fmt_b(b_val):
    if b_val is None:
        return "\u2014"
    if not np.isfinite(b_val):
        return "\u2014"
    return f"{b_val:.3f}"


def _build_projection_table():
    """Render the pre-computed projection table as Dash components. Reads
    from btc_web/_projection_table.json (rebuilt monthly by
    tools/build_projection_table.py via daily_update.sh). Falls back to
    a 'data unavailable' message if the JSON is missing."""
    blob = _load_projection_blob()
    if blob is None:
        return html.Div([
            html.P(
                "\u26a0 Projection table data not available "
                "(btc_web/_projection_table.json missing or unreadable). "
                "Run `python3 tools/build_projection_table.py --force` on "
                "dev and deploy to regenerate.",
                style={"color": DIM_TEXT, "fontSize": UI_FONT_SM},
            ),
        ])

    generated_at = blob.get("generated_at", "unknown")
    target_usd = blob.get("target_price_usd", 1_000_000)
    models = blob.get("models", [])

    tables = []
    for weighting in blob.get("weightings", []):
        wlbl = weighting.get("label", "?")
        header = [html.Th("t\u2080 preset")]
        for mname in models:
            header.append(html.Th(f"{mname} \u2014 b"))
            header.append(html.Th(f"{mname} \u2014 ${target_usd // 1_000_000}M"))

        body_rows = []
        for row in weighting.get("rows", []):
            lbl = row.get("preset_label", "?")
            cells = [html.Td(lbl)]
            row_cells = row.get("cells", {})
            for mname in models:
                cell = row_cells.get(mname, {"b": None, "date": "\u2014"})
                b_str = _fmt_b(cell.get("b"))
                date_str = cell.get("date", "\u2014")
                cells.append(html.Td(b_str, style={
                    "fontFamily": "monospace", "fontSize": "0.85em"}))
                cells.append(html.Td(date_str, style={
                    "fontFamily": "monospace", "fontSize": "0.85em"}))
            body_rows.append(html.Tr(cells))

        table = dbc.Table(
            [html.Thead(html.Tr(header)), html.Tbody(body_rows)],
            bordered=True, hover=True, responsive=True, size="sm",
            className="mb-4",
        )
        tables.append(html.Div([
            html.H6(f"Weighting: {wlbl}",
                     style={"marginTop": "12px", "marginBottom": "6px"}),
            table,
        ]))

    # Generated-at header (top of body)
    gen_display = generated_at.replace("T", " ").split("+")[0]
    header_div = html.Div([
        html.P([
            html.B("Table generated: "),
            html.Code(gen_display, style={"fontSize": UI_FONT_SM}),
            html.Span(" UTC", style={"fontSize": UI_FONT_SM, "color": DIM_TEXT}),
            html.Span(" \u00b7 refreshed monthly via daily_update.sh",
                       style={"fontSize": UI_FONT_SM, "color": DIM_TEXT,
                              "marginLeft": "8px"}),
        ], style={"marginBottom": "4px"}),
        html.P([
            "Each cell shows the fit's log-log ",
            html.B("exponent (b)"), " and the projected calendar ",
            html.B(f"month Bitcoin price crosses ${target_usd:,} USD"),
            " by solving log\u2081\u2080(price) against the stored slope/"
            "intercept. BM floor uses the 20-percentile support line; QR "
            "uses the median (Q50%) line; Exp is linear-t so it is ",
            html.Em("both"), " t\u2080- and weighting-invariant by design.",
        ], style={"fontSize": UI_FONT_SM, "color": DIM_TEXT}),
        html.P([
            html.Small([
                "Legend: ",
                html.Code("YYYY-MM"), " projected month \u00b7 ",
                html.Code(">YYYY"), " projection past year \u00b7 ",
                html.Code("<2010"), " fit already above target pre-2010 \u00b7 ",
                html.Code("\u2014"), " degenerate/failed fit.",
            ]),
        ], style={"color": DIM_TEXT}),
    ])
    return [header_div] + tables


@callback(
    Output("cta-proj-modal", "is_open"),
    Output("cta-proj-modal-body", "children"),
    Input("cta-open-proj-modal", "n_clicks"),
    Input("cta-proj-modal-close", "n_clicks"),
    State("cta-proj-modal", "is_open"),
    prevent_initial_call=True,
)
def toggle_projection_modal(open_clicks, close_clicks, is_open):
    """Open the modal on button click (computing the table lazily); close
    on the close button. Computation is ~5s for the full 6×4×4 grid, so
    wrap the body in dcc.Loading (already done in layout)."""
    from dash import ctx
    trigger = ctx.triggered_id
    if trigger == "cta-open-proj-modal":
        try:
            body = _build_projection_table()
        except Exception as exc:
            _LOG.error("projection table build failed: %s", exc, exc_info=True)
            body = html.Pre(f"Failed to build table: {type(exc).__name__}: {exc}")
        return True, body
    if trigger == "cta-proj-modal-close":
        return False, no_update
    return no_update, no_update
