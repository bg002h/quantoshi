"""Time Machine (as-of date) callbacks — Task 10.

Wires the Task-9 controls on Tab 1 (Bubble) so they actually do something:

  1. Reveal the body panel when the toggle is on          (clientside)
  2. ``_asof_frame`` — pure gate feeding ``update_bubble`` (imported there)
  3. ▶ Play: toggle the interval + advance the slider      (clientside)
  4. As-of label: format the active frame's date           (clientside)
  5. Single-model constraint: restrict Display Models to
     {BM, EPPL master} while as-of mode is on              (server, rare)
  6. MC + CTA mutual exclusion: hide both control blocks    (clientside)

Pure-visibility toggles are clientside (ALARA — no server round-trips). The
only server callback (#5) fires on the rare toggle event and rebuilds the
Display-Models options from the SSOT so the "off" restore is always correct.
"""
import dash
from dash import Input, Output, State, callback

import _app_ctx


# ── 2. Pure gate helper (imported by callbacks/charts/__init__.py) ───────────
def _asof_frame(toggle, slider):
    """Translate the Time Machine toggle + slider into an as-of frame index.

    Returns the slider value (an int frame index into ``timemachine.frames()``)
    ONLY when the toggle is on AND a slider value is present; otherwise None so
    the bubble chart renders its ordinary live-data view.

    Uses an explicit ``is not None`` check, never a falsy test — frame index 0
    (the first frame) is legitimate and must not be swallowed.
    """
    if toggle and "on" in toggle and slider is not None:
        return slider
    return None


# ── 1. Reveal the body panel (clientside) ────────────────────────────────────
# Same idiom as the MC body toggles (callbacks/mc_controls.py). Nothing else
# writes bub-timemachine-body.style → no allow_duplicate; fires on load so the
# hidden-by-default state is confirmed.
_app_ctx.app.clientside_callback(
    "function(v) { return (v && v.length) ? {} : {display:'none'}; }",
    Output("bub-timemachine-body", "style"),
    Input("bub-timemachine-toggle", "value"),
)


# ── 3. Play button + interval tick (clientside) ──────────────────────────────
# The ▶ button enables/disables the dcc.Interval. Each tick advances the
# slider by one frame; on reaching the last frame it stops (auto-disables the
# interval) and resets the label. Pressing ▶ at the end restarts from frame 0.
# Both callbacks write bub-asof-interval.disabled + bub-asof-slider.value +
# bub-asof-play.children → every such Output uses allow_duplicate=True.
_PLAY_JS = """
function(n_clicks, disabled, cur, mx) {
    var NU = window.dash_clientside.no_update;
    if (!n_clicks) { return [NU, NU, NU]; }
    if (disabled) {
        // Start playing. If parked at the end, rewind to the first frame.
        var restart = (cur != null && mx != null && cur >= mx) ? 0 : NU;
        return [false, restart, "⏸ Pause"];
    }
    return [true, NU, "▶ Play"];   // pause
}
"""
_app_ctx.app.clientside_callback(
    _PLAY_JS,
    Output("bub-asof-interval", "disabled", allow_duplicate=True),
    Output("bub-asof-slider", "value", allow_duplicate=True),
    Output("bub-asof-play", "children", allow_duplicate=True),
    Input("bub-asof-play", "n_clicks"),
    State("bub-asof-interval", "disabled"),
    State("bub-asof-slider", "value"),
    State("bub-asof-slider", "max"),
    prevent_initial_call=True,
)

_TICK_JS = """
function(n_intervals, cur, mx) {
    var NU = window.dash_clientside.no_update;
    if (cur == null) { cur = 0; }
    if (mx == null || cur >= mx) { return [NU, true, "▶ Play"]; }
    var nv = cur + 1;
    if (nv >= mx) { return [mx, true, "▶ Play"]; }   // reached the end
    return [nv, NU, NU];
}
"""
_app_ctx.app.clientside_callback(
    _TICK_JS,
    Output("bub-asof-slider", "value", allow_duplicate=True),
    Output("bub-asof-interval", "disabled", allow_duplicate=True),
    Output("bub-asof-play", "children", allow_duplicate=True),
    Input("bub-asof-interval", "n_intervals"),
    State("bub-asof-slider", "value"),
    State("bub-asof-slider", "max"),
    prevent_initial_call=True,
)


# ── 4. As-of date label (clientside) ─────────────────────────────────────────
# Frames list is passed via the bub-asof-frames Store (populated at layout
# time). Fires on load so the label reflects the slider's initial position.
_LABEL_JS = """
function(idx, frames) {
    if (!frames || !frames.length) { return ""; }
    if (idx == null) { idx = frames.length - 1; }
    if (idx < 0 || idx >= frames.length) { return ""; }
    return "As of " + frames[idx];
}
"""
_app_ctx.app.clientside_callback(
    _LABEL_JS,
    Output("bub-asof-label", "children"),
    Input("bub-asof-slider", "value"),
    State("bub-asof-frames", "data"),
)


# ── 5. Single-model constraint (server; rare toggle event) ───────────────────
# When Time Machine is ON, only the Bubble Model (BM) and the EPPL master can
# render an as-of view (figures/bubble.py::_asof_resolve rebuilds ecfg_* keys
# and BM from the grid; other models have no as-of path). Restrict the Display
# Models checklist to {bub, eppl} and collapse the value to a single eligible
# entry (default "bub"). When OFF, restore the full option list from the SSOT.
#
# FOOTGUN: "bub-model-show".options is already output by
# callbacks/charts/__init__.py::_update_bub_swatches WITHOUT allow_duplicate,
# so both Outputs here MUST use allow_duplicate=True → prevent_initial_call
# is therefore required (never allow_duplicate + prevent_initial_call=False).
_TM_ELIGIBLE = ("bub", "eppl")


@callback(
    Output("bub-model-show", "options", allow_duplicate=True),
    Output("bub-model-show", "value", allow_duplicate=True),
    Input("bub-timemachine-toggle", "value"),
    State("palette-store", "data"),
    State("display-model-summaries", "data"),
    State("bub-model-show", "value"),
    prevent_initial_call=True,
)
def _tm_single_model(toggle, palette_key, summaries, cur_value):
    # Lazy import avoids a circular import at module-load time
    # (callbacks.charts imports _asof_frame from this module).
    from callbacks.charts import _swatches_for
    full_opts = _swatches_for("bub", palette_key, summaries)
    if toggle and "on" in toggle:
        opts = [o for o in full_opts if o["value"] in _TM_ELIGIBLE]
        keep = [v for v in (cur_value or []) if v in _TM_ELIGIBLE]
        value = [keep[0]] if keep else ["bub"]
        return opts, value
    # Off → restore full options; leave the current selection untouched.
    return full_opts, dash.no_update


# ── 6. MC + CTA mutual exclusion (clientside) ────────────────────────────────
# The MC and Custom Time Axis control blocks are wrapped in
# bub-mc-exclude-wrap / bub-cta-exclude-wrap (layout/bubble.py). Hide both
# whole wrappers while Time Machine is on so their inner toggles vanish too.
_app_ctx.app.clientside_callback(
    "function(v) { var s = (v && v.length) ? {display:'none'} : {}; return [s, s]; }",
    Output("bub-mc-exclude-wrap", "style"),
    Output("bub-cta-exclude-wrap", "style"),
    Input("bub-timemachine-toggle", "value"),
)


# ── 6b. Force conflicting controls OFF when Time Machine turns ON ─────────────
# Hiding the MC/CTA blocks (#6) is NOT enough — their live *values* still break
# the as-of chart:
#   * cta-active=["yes"] → custom_time_callback owns bubble-graph.figure and
#     update_bubble PreventUpdates on its guard (charts/__init__.py), so the
#     as-of slider becomes a silent no-op. Clearing cta-active drives
#     custom_time_callback's DEACTIVATE branch, which bumps bub-redraw-tick
#     (an Input to update_bubble) → update_bubble re-fires with cta_active=[]
#     and renders the as-of view.
#   * bub-mc-enable=["yes"] → the MC spaghetti overlay paints PRESENT-DAY
#     forward paths (gated on neither model_show nor asof_date) onto a chart
#     labelled "As of <past date>". Clearing it removes the overlay.
# When TM turns OFF we deliberately do NOT restore the user's CTA/MC choices
# (no_update — leave whatever they are), but we DO stop any running Play
# interval and reset the ▶ label (folded #3) so a re-open isn't stuck on
# "⏸ Pause". Every Output here is written elsewhere too (or shared with the
# Play/tick callbacks) → allow_duplicate=True + prevent_initial_call=True
# (never allow_duplicate with prevent_initial_call=False → gunicorn crash).
_TM_FORCE_OFF_JS = """
function(v) {
    var NU = window.dash_clientside.no_update;
    if (v && v.length) {
        // TM ON: force CTA + MC off; the Play interval is already idle.
        return [[], [], NU, NU];
    }
    // TM OFF: keep the user's CTA/MC selections; stop any running Play.
    return [NU, NU, true, "▶ Play"];
}
"""
_app_ctx.app.clientside_callback(
    _TM_FORCE_OFF_JS,
    Output("cta-active", "value", allow_duplicate=True),
    Output("bub-mc-enable", "value", allow_duplicate=True),
    Output("bub-asof-interval", "disabled", allow_duplicate=True),
    Output("bub-asof-play", "children", allow_duplicate=True),
    Input("bub-timemachine-toggle", "value"),
    prevent_initial_call=True,
)
