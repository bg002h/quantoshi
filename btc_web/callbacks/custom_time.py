"""Custom Time Axis callbacks — section 5 error cases encoded here.

See docs/superpowers/specs/2026-04-13-custom-time-axis-design.md §5.
"""
from __future__ import annotations

import logging
import time
from datetime import date

import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, no_update
from dash.exceptions import PreventUpdate

import _app_ctx
from colors import (
    DIM_TEXT, MODEL_TRACE_COLORS, TRACE_WIDTH, FALLBACK_MODEL_GRAY,
)
from engines import custom_fit as cf
from _custom_time_presets import CAL_PRESET_BY_KEY, BLK_PRESET_BY_KEY

_LOG = logging.getLogger("custom_time")
_CAP_DATE = date(2016, 1, 1)


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


def _build_figure(results: dict, scale: str, t0_label: str) -> go.Figure:
    """Build the Custom Time Axis figure. Log-log axes for calendar mode;
    linear-blockheight x-axis for block mode."""
    fig = go.Figure()
    x_series = cf._DATES if scale == "calendar" else cf._BLOCKS
    if x_series is None:
        return fig
    fig.add_trace(go.Scatter(
        x=x_series, y=cf._PRICES, mode="markers",
        marker=dict(size=3, color=DIM_TEXT, opacity=0.5),
        name=f"Price (n={len(cf._PRICES):,})",
    ))

    colors = {
        "PL":       MODEL_TRACE_COLORS.get("pl", FALLBACK_MODEL_GRAY),
        "QR":       MODEL_TRACE_COLORS.get("qr", FALLBACK_MODEL_GRAY),
        "BM floor": MODEL_TRACE_COLORS.get("bub", FALLBACK_MODEL_GRAY),
        "Exp":      MODEL_TRACE_COLORS.get("exp", FALLBACK_MODEL_GRAY),
    }

    # Convert t_plot (in fit-space units — years or block offsets) back to
    # the x-axis space (calendar dates or absolute blockheights) so the trace
    # aligns with the price scatter.
    t0_ts = pd.Timestamp(t0_label) if scale == "calendar" else None
    t0_int = int(t0_label) if scale == "block" else None

    def _t_plot_to_x(t_plot):
        if scale == "calendar":
            return t0_ts + pd.to_timedelta(t_plot * 365.25, unit="D")
        return t0_int + t_plot

    for r in results.values():
        if r is None:
            continue
        color = colors.get(r.name, FALLBACK_MODEL_GRAY)
        label_n = f"{r.n_samples:,}"
        if isinstance(r.y_plot, dict):
            x_plot = _t_plot_to_x(r.t_plot)
            for q, y in r.y_plot.items():
                fig.add_trace(go.Scatter(
                    x=x_plot, y=10 ** y, mode="lines",
                    line=dict(color=color, width=TRACE_WIDTH),
                    name=f"{r.name} Q{int(q*100)}% (n={label_n})",
                    legendgroup=r.name,
                ))
        else:
            x_plot = _t_plot_to_x(r.t_plot)
            fig.add_trace(go.Scatter(
                x=x_plot, y=10 ** r.y_plot, mode="lines",
                line=dict(color=color, width=TRACE_WIDTH),
                name=f"{r.name} (n={label_n})",
            ))

    fig.update_layout(
        yaxis=dict(type="log", title="USD"),
        xaxis=dict(
            title=("Years since " + t0_label if scale == "calendar"
                    else f"Blockheight since block {t0_label}"),
        ),
        title=f"Custom Time Axis — t\u2080 = {t0_label}",
        template="plotly_white",
        margin=dict(l=60, r=30, t=60, b=60),
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
    State("bub-redraw-tick", "data"),
    prevent_initial_call=True,
)
def custom_time_callback(active, scale, cal_preset, cal_custom,
                          blk_preset, blk_custom, weighting, models, tick):
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
        results = {}
        if "pl" in models:
            results["pl"] = cf.fit_pl(fi)
        if "qr" in models:
            results["qr"] = cf.fit_qr(fi)
        if "bm_floor" in models:
            results["bm_floor"] = cf.fit_bm_floor(fi)
        if "exp" in models:
            results["exp"] = cf.fit_exp(fi)

        elapsed_ms = int((time.perf_counter() - t_start) * 1000)
        if elapsed_ms > 5000:
            _LOG.warning("custom_fit slow: %dms params=%s",
                          elapsed_ms,
                          {"scale": scale, "t0": t0, "weighting": weighting})

        # 6. Build figure + status
        fig = _build_figure(results, scale, str(t0))
        sample_counts = [r.n_samples for r in results.values() if r is not None]
        total_n = max(sample_counts, default=0)
        skipped = sum(1 for r in results.values() if r is None)
        status_parts = [
            f"Fit on {total_n:,} samples from t\u2080={t0}.",
            f"{elapsed_ms}ms.",
        ]
        if skipped:
            status_parts.append(f"{skipped} model(s) skipped (see legend).")
        return fig, " ".join(status_parts), no_update

    except Exception as exc:
        _LOG.error("custom_fit crash: %s", exc, exc_info=True)
        return (no_update,
                f"\u26a0 Internal error: {type(exc).__name__}",
                no_update)
