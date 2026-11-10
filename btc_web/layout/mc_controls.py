"""Monte Carlo simulation controls — shared across DCA, Retire, Heatmap, Supercharger tabs."""

import json
import logging

import pandas as pd

from dash import dcc, html
import dash_bootstrap_components as dbc

import _app_ctx
import raw_mc
from mc_cache import (CACHED_START_YRS, WD_AMOUNTS,
                      ENTRY_PCT_BINS, MC_YEARS_OPTIONS, INFL_OPTIONS,
                      is_master_cached)
from mc_overlay import bin_regime_labels
from snapshot_defaults import SNAPSHOT_DEFAULTS

from layout.common import (_section_card, _ctrl_card, _row, _lbl,
                            _STYLE_HIDDEN, _STYLE_HINT, _CB_MARGIN)
from colors import (BLACK, SILVER, NEAR_BLACK, MUTED_TEXT,
                    MODAL_DIVIDER_DARK, DIM_TEXT, WHITE, BADGE_GLOW_RED, _hex_alpha,
                    FONT_BRAND, UI_FONT_SM, UI_FONT_MD, UI_FONT_LG, UI_FONT_XL, UI_FONT_XXL,
                    CTX_MENU_SHADOW_ALPHA, BADGE_INSET_ALPHA, BADGE_GLOW_ALPHA)

_QUANT_FONT = {"fontFamily": FONT_BRAND,
               "color": BLACK, "letterSpacing": "1px"}


def _mc_model_src_options(include_raw=False):
    """Curated master-only set for the MC quantile-bands dropdown.

    Mirrors the heatmap pill set (`layout.heatmap._HM_PILL_MODELS_BASE`,
    minus 'mc') so configurable families (LPPL/HybPPL/EPPL) appear once
    as a master entry. The specific config variant is picked up from the
    global LPPL/HybPPL/EPPL config modals via `_resolve_mc_model_src`
    in the callbacks before `_mc_setup` runs.

    Models present in the precomputed MC cache (free tier) render bold;
    others require live compute (paid).

    include_raw: append the model-free Raw variants (Tab 1 only).
    """
    from layout.heatmap import _HM_PILL_MODELS_BASE
    keys = list(_HM_PILL_MODELS_BASE)
    if "ef" in _app_ctx.PRICE_MODELS:
        keys.append("ef")
    if "u1" in _app_ctx.PRICE_MODELS:
        keys.append("u1")
    out = []
    for k in keys:
        if k not in _app_ctx.PRICE_MODELS:
            continue
        name = _app_ctx.PRICE_MODELS[k].name
        if is_master_cached(k):
            label = html.Span(f" {name}", style={"fontWeight": "bold"})
        else:
            label = f" {name}"
        out.append({"label": label, "value": k})
    if include_raw:
        out += [{"label": f" {RAW_SRC_LABELS[k]}", "value": k}
                for k in raw_mc.RAW_SRCS]
    return out


# ── Tab-1 Raw (model-free) MC ───────────────────────────────────────────────
RAW_SRC_LABELS = {"raw_iid": "Raw \u00b7 independent",
                  "raw_cycle": "Raw \u00b7 halving cycle"}
RAW_HINT = "Raw starts today at the current price"
RAW_TREND_OPTIONS = [{"label": "Power law", "value": "power"},
                     {"label": "Exponential decay", "value": "exp"},
                     {"label": "None (honest replay)", "value": "none"}]
RAW_YEARS_OPTIONS = [{"label": f"{v} yr", "value": v} for v in (10, 20, 30, 40)]
RAW_BINS_OPTIONS = [{"label": str(v), "value": v} for v in range(3, 11)]

# bub-mc-panel.className from bub-mc-model-src.value (ONE Input). The class
# hides the model-only groups and shows the Raw-only ones (assets/style.css).
RAW_MODE_CLASS_JS = """
function(src) {
    var RAW = %s;
    return (RAW.indexOf(src) >= 0) ? "qs-mc-raw-mode" : "";
}
""" % json.dumps(list(raw_mc.RAW_SRCS))


def raw_regime_options(bins, window, trend):
    """Checklist options for the Raw trend-deviation bins. Plain "Bin 1..n"
    labels when no trend / state can be built (e.g. an exp fit failure, plan
    review M-9) or prices are not registered, so the checklist is never
    empty.  ANY failure here falls back too: this runs while the Tab-1
    layout is built, so a Raw-only fault must never take the page down
    (final review I-1)."""
    try:
        opts = raw_mc.regime_options(bins, window, trend)
    except RuntimeError:
        opts = []                          # prices not registered: expected
    except Exception:                      # noqa: BLE001 -- fail soft, see above
        logger.exception("Raw regime options failed; using plain bin labels")
        opts = []
    return opts or [{"label": f"Bin {i + 1}", "value": i} for i in range(int(bins))]


def _wrap(cls, on, *children):
    """``children`` inside one Div of class ``cls`` when ``on``, else as is
    (the other tabs' layout is unchanged)."""
    return [html.Div(className=cls, children=list(children))] if on else list(children)


_MO, _RO = "qs-mc-model-only", "qs-mc-raw-only"
logger = logging.getLogger(__name__)

# bub-mc-lines-wrap.style from bub-mc-paths.value (ONE Input): "Lines shown"
# and "Show extremes" act only on drawn paths, so they show only with them
# (final review M-6).  On a wrapper inside the qs-mc-raw-only Div, so the
# inline display never fights the Raw/model class rules.
RAW_LINES_VIS_JS = "function(v) { return (v && v.length) ? {} : {display: 'none'}; }"
_SMALL_NOTE = {"display": "block", "fontSize": UI_FONT_SM, "color": DIM_TEXT,
               "lineHeight": "1.3", "marginBottom": "6px"}


def _raw_lines_block(prefix):
    """Lines shown + Show extremes, directly under Show individual price paths;
    hidden until that box is ticked (RAW_LINES_VIS_JS).  No HTML min/max on
    the number input: the browser would send null for an out-of-range value
    (CLAUDE.md gotcha), so the server clamps instead and the label states the
    range (final review M-10)."""
    return html.Div(className=_RO, children=[html.Div(
        id=f"{prefix}-mc-lines-wrap", style={"display": "none"}, children=[
            _lbl(f"Lines shown ({raw_mc.RAW_LINES_MIN}\u2013{raw_mc.RAW_LINES_MAX})"),
            dbc.Input(id=f"{prefix}-mc-lines", type="number",
                      value=raw_mc.RAW_LINES_DEFAULT, step=1, size="sm",
                      debounce=True),
            dcc.Checklist(id=f"{prefix}-mc-extremes",
                          options=[{"label": " Show extremes", "value": "yes"}],
                          value=[], inputStyle=_CB_MARGIN,
                          style={"fontSize": UI_FONT_MD, "color": MUTED_TEXT,
                                 "margin": "4px 0 6px"}),
        ])])


def _raw_controls_block(prefix, yr_now):
    return html.Div(className=_RO, children=[
        _lbl(f"Simulations (1\u2013{raw_mc.RAW_SIMS_MAX}, free)"),
        # no HTML min/max: the server clamps (see _raw_lines_block)
        dbc.Input(id=f"{prefix}-mc-raw-sims", type="number", value=1000,
                  step=1, size="sm", debounce=True),
        _lbl("Years to simulate"),
        dcc.Dropdown(id=f"{prefix}-mc-raw-years", options=RAW_YEARS_OPTIONS,
                     value=40, clearable=False),
        _lbl("History window (resampled months)"),
        dcc.RangeSlider(id=f"{prefix}-mc-raw-window", min=2010, max=yr_now,
                        value=[2010, yr_now], step=1,
                        marks={y: str(y) for y in range(2010, yr_now + 1, 5)}),
        _lbl("Trend"),
        dcc.RadioItems(id=f"{prefix}-mc-raw-trend", options=RAW_TREND_OPTIONS,
                       value="power", inputStyle=_CB_MARGIN,
                       labelStyle={"display": "block", "fontSize": UI_FONT_MD}),
        _lbl("Trend-deviation bins"),
        dcc.Dropdown(id=f"{prefix}-mc-raw-bins", options=RAW_BINS_OPTIONS,
                     value=5, clearable=False),
        _lbl("Trend-deviation regime (price level vs trend)"),
        dcc.Checklist(
            id=f"{prefix}-mc-raw-regime",
            options=raw_regime_options(5, (2010, yr_now), "power"),
            value=list(range(5)), inputStyle=_CB_MARGIN,
            labelStyle={"display": "block", "fontSize": UI_FONT_MD,
                        "lineHeight": "1.6", "color": MODAL_DIVIDER_DARK},
            style={"marginBottom": "2px"}),
        html.Small("Unticked bins steer paths away from those levels; leave all "
                   "ticked for no filter.", style=_SMALL_NOTE),
    ])
_MC_CACHED_START_YRS = set(CACHED_START_YRS)
_MC_CACHED_ENTRY_QS = {int(v * 100) for v in ENTRY_PCT_BINS}   # {10,20,...,90}
_MC_CACHED_YEARS    = set(MC_YEARS_OPTIONS)                      # {10,20,30,40}
_MC_CACHED_WD       = set(WD_AMOUNTS)
_MC_CACHED_INFL     = set(INFL_OPTIONS)


def _bold_opts(values, fmt, cached_set):
    """Build dropdown options, bolding+enlarging values in the pre-computed cache."""
    return [
        {"label": html.Span(fmt(v), style={"fontWeight": "bold", "fontSize": UI_FONT_XXL})
                  if v in cached_set else fmt(v),
         "value": v}
        for v in values
    ]

# ── MC pricing (sats) ────────────────────────────────────────────────────────
# Cached: pre-computed paths on server, instant lookup
# Non-cached: live Markov chain simulation (~1-3s compute)
_MC_PRICE_LIVE   = {10: 500, 20: 1000, 30: 1500, 40: 2000}
_MC_START_YR_OPTIONS = _bold_opts(range(2026, 2051), str, _MC_CACHED_START_YRS)
# Entry Q dropdown: show 1%–90% in 10% steps, with 1%/10%/50% bolded
_MC_ENTRY_Q_VALS = sorted(set([1] + [int(v * 100) for v in ENTRY_PCT_BINS] + list(range(10, 100, 10))))
_MC_ENTRY_Q_OPTIONS  = _bold_opts(
    _MC_ENTRY_Q_VALS,
    lambda v: f"{v}%", _MC_CACHED_ENTRY_QS)
_MC_ENTRY_Q_OPTIONS_ADV = _bold_opts(
    [round(i / 10, 1) for i in range(1, 1000)],
    lambda v: f"{v}%", _MC_CACHED_ENTRY_QS)
_MC_YEARS_OPTIONS    = _bold_opts([10, 20, 30, 40], lambda v: f"{v} yr", _MC_CACHED_YEARS)
_MC_WD_OPTIONS       = _bold_opts(WD_AMOUNTS, lambda v: f"${v:,}/mo", _MC_CACHED_WD)
_MC_INFL_OPTIONS     = _bold_opts(INFL_OPTIONS, lambda v: f"{v}%", _MC_CACHED_INFL)

def _regime_options(n_bins=5):
    """Build checklist options for the bin regime filter."""
    labels = bin_regime_labels(n_bins)
    return [{"label": labels[i], "value": i} for i in range(n_bins)]

_MC_REGIME_OPTIONS_5 = _regime_options(5)   # pre-computed for default 5-bin case

# Tab 1's regime-filter note: shown exactly when the server applies the filter
# (callbacks/mc_helpers.py: blocked only when 0 < checked < bins; none checked
# is treated as all allowed). Inputs: (bub-mc-regime.value, bub-mc-bins.value).
MC_REGIME_NOTE_JS = """
function(checked, bins) {
    var n = parseInt(bins, 10);
    var k = Array.isArray(checked) ? checked.length : 0;
    return (n > 0 && k > 0 && k < n) ? {} : {display: 'none'};
}
"""


# Tab 1: a real click that APPLIES a regime filter ticks "Show individual price
# paths" (operator request 2026-11-10). Inputs: (bub-mc-regime.value,
# bub-mc-raw-regime.value) with State (bub-mc-bins.value, bub-mc-raw-bins.value,
# bub-mc-paths.value). Two Inputs are safe only because the Output is NOT
# allow_duplicate (allow_duplicate + several Inputs + prevent_initial_call
# silently never fires in Dash 4.0).
#   * only a fresh click stamp acts: assets/mc_regime_paths.js stamps the model
#     checklist, assets/mc_raw_regime_paths.js the Raw one. A share-link
#     restore or any other programmatic write never touches the paths box;
#   * only on the transition no-filter -> filter, so a user who unticks paths
#     while filtered is not overridden by their next regime click;
#   * never unticks. "Filter" is read as the server reads it: 0 < checked <
#     bins (callbacks/mc_helpers.py); for Raw only distinct in-range indices
#     count, because its value is never reset on a bins change
#     (raw_mc.effective_checked).
# The tracker remembers the previous checked COUNT, not a verdict: a bins change
# rewrites only the checklist's options, so this callback does not run for it,
# yet it changes what a given count means. The count is judged against the
# CURRENT bins, and the first run starts from the default value's length
# (review I-1, 2026-11-10).
# Each checklist has its own tracker and stamp (__qsRegimeK /
# __qsMcRegimeClickAt, __qsRawRegimeK / __qsMcRawRegimeClickAt). BOTH trackers
# are updated on every call, and the decision is made only for a checklist
# whose stamp is fresh -- never from triggered[0]: a share-link restore writes
# both checklists in one batch (plan review I-6).
# One writer of bub-mc-paths.value, NOT allow_duplicate: an allow_duplicate
# writer made Dash fire update_bubble twice per tick click (review I-2).
MC_REGIME_CLICK_MS = 3000
MC_REGIME_DEFAULT_K = len(SNAPSHOT_DEFAULTS["bub-mc-regime:value"])
MC_RAW_REGIME_DEFAULT_K = 5      # layout default value list(range(5)), bins 5
MC_REGIME_PATHS_JS = """
function(checked, rawChecked, bins, rawBins, paths) {
    var NU = window.dash_clientside.no_update;
    var now = Date.now();
    function count(c, n, raw) {
        if (!Array.isArray(c)) { return 0; }
        if (!raw) { return c.length; }
        var seen = {}, k = 0;
        for (var i = 0; i < c.length; i++) {
            var v = parseInt(c[i], 10);
            if (v >= 0 && v < n && !seen[v]) { seen[v] = true; k++; }
        }
        return k;
    }
    function applies(c, b, kKey, atKey, dflt, raw) {
        var n = parseInt(b, 10);
        var k = count(c, n, raw);
        var on = (n > 0 && k > 0 && k < n);
        var pk = (window[kKey] === undefined) ? dflt : window[kKey];
        var was = (n > 0 && pk > 0 && pk < n);
        window[kKey] = k;
        var at = window[atKey] || 0;
        window[atKey] = 0;
        return on && !was && !!at && now - at <= %(click_ms)s;
    }
    var m = applies(checked, bins, "__qsRegimeK", "__qsMcRegimeClickAt",
                    %(default_k)s, false);
    var r = applies(rawChecked, rawBins, "__qsRawRegimeK", "__qsMcRawRegimeClickAt",
                    %(raw_default_k)s, true);
    if (!m && !r) { return NU; }
    if (Array.isArray(paths) && paths.indexOf("yes") >= 0) { return NU; }
    return ["yes"];
}
""" % {"click_ms": MC_REGIME_CLICK_MS, "default_k": MC_REGIME_DEFAULT_K,
       "raw_default_k": MC_RAW_REGIME_DEFAULT_K}


def _bub_regime_note():
    from figures.bubble import MC_REGIME_FAN_PATHS
    return html.Div(id="bub-mc-regime-note", style={"display": "none"}, children=html.Small(
        f"Filter on: the fan and paths use the {MC_REGIME_FAN_PATHS} of 200 "
        "cached paths that spend the least time in the unchecked regimes "
        "\u2014 a ranked subset, not a new simulation.",
        style={"display": "block", "fontSize": UI_FONT_SM, "color": DIM_TEXT,
               "lineHeight": "1.3", "marginBottom": "8px"},
    ))

def _mc_controls(prefix, amount_label="Purchase amount ($)", amount_default=100,
                  show_inflation=False, show_amount=True,
                  show_stack=False, show_mc_entry_q=False,
                  default_entry_q=50, start_yr_label=None,
                  shared_controls=frozenset(), mc_enabled_default=False,
                  default_sims=200, sims_options=None,
                  show_paths_toggle=False, include_raw_src=False):
    """Monte Carlo simulation controls, reusable across tabs.

    show_paths_toggle: add the "{prefix}-mc-paths" checklist (Tab 1 only --
    the one tab that draws MC price paths). Default off = fan only.
    include_raw_src: Tab 1 only -- the Raw (model-free) entries in the model
    source dropdown, the Raw controls and status line, and the literal
    ``bub-mc-panel`` id on the outer Div that carries the mode class.
    """
    yr_now = pd.Timestamp.today().year
    _mc_enable_val = ["yes"] if mc_enabled_default else []
    if not _app_ctx._HAS_MARKOV:
        # Hidden placeholders so callback IDs exist even without markov module
        _ph = []  # placeholder components
        _ph.append(dcc.Checklist(id=f"{prefix}-mc-enable", value=[]))
        _ph.append(dcc.Checklist(id=f"{prefix}-mc-advanced", value=[]))
        if show_paths_toggle:
            _ph.append(dcc.Checklist(id=f"{prefix}-mc-paths", value=[]))
        if "amount" not in shared_controls:
            _ph.append(dbc.Input(id=f"{prefix}-mc-amount", value=amount_default))
        if "infl" not in shared_controls:
            _ph.append(dbc.Input(id=f"{prefix}-mc-infl", value=4))
        _ph.append(dbc.Input(id=f"{prefix}-mc-bins", value=5))
        _ph.append(dcc.Checklist(id=f"{prefix}-mc-regime", value=list(range(5))))
        if prefix == "bub":
            _ph.append(html.Div(id="bub-mc-regime-note"))
        _ph.append(dcc.Dropdown(id=f"{prefix}-mc-sims", value=default_sims))
        _ph.append(dcc.Dropdown(id=f"{prefix}-mc-years", value=40))
        if "freq" not in shared_controls:
            _ph.append(dcc.Dropdown(id=f"{prefix}-mc-freq", value="Monthly"))
            _ph.append(dbc.Input(id=f"{prefix}-mc-ppy", value="12/yr"))
        _ph.append(dcc.RangeSlider(id=f"{prefix}-mc-window", min=2010,
                                    max=yr_now, value=[2010, yr_now]))
        _ph.append(html.Div(id=f"{prefix}-mc-adv-body"))
        _ph.append(html.Div(id=f"{prefix}-mc-cost"))
        _ph.append(dcc.Store(id=f"{prefix}-mc-price-val", storage_type="memory", data=0))
        _ph.append(html.Div(id=f"{prefix}-mc-body"))
        _ph.append(html.Div(id=f"{prefix}-mc-status"))
        _ph.append(dbc.Button(id=f"{prefix}-mc-dl-btn", style=_STYLE_HIDDEN))
        _ph.append(dbc.Button(id={"type": "mc-run-btn", "tab": prefix}, style=_STYLE_HIDDEN))
        _ph.append(html.Div(id={"type": "mc-run-status", "tab": prefix}))
        _ph.append(dcc.Store(id=f"{prefix}-mc-rendered-key", storage_type="memory"))
        _ph.append(html.Div(id=f"{prefix}-mc-match"))
        _ph.append(dbc.Button(id=f"{prefix}-mc-restore-btn", style=_STYLE_HIDDEN))
        _ph.append(dcc.Upload(id=f"{prefix}-mc-upload"))
        _ph.append(html.Div(id=f"{prefix}-mc-upload-status"))
        _ph.append(dcc.Slider(id=f"{prefix}-mc-entry-yr", value=yr_now))
        _ph.append(dcc.Dropdown(id=f"{prefix}-mc-entry-q",
                    value=max(10, min(90, round(_app_ctx._HM_ENTRY_Q_DEFAULT / 10) * 10 or 50))))
        if "stack" not in shared_controls:
            _ph.append(dbc.Input(id=f"{prefix}-mc-stack", type="number", value=1.0))
        _ph.append(dcc.Dropdown(id=f"{prefix}-mc-start-yr", value=2031))
        _ph.append(dcc.Dropdown(id=f"{prefix}-mc-model-src", value="bub"))
        if include_raw_src:
            _ph += [dbc.Input(id=f"{prefix}-mc-raw-sims", value=1000),
                    dcc.Dropdown(id=f"{prefix}-mc-raw-years", value=40),
                    dcc.RangeSlider(id=f"{prefix}-mc-raw-window", min=2010,
                                    max=yr_now, value=[2010, yr_now]),
                    dcc.RadioItems(id=f"{prefix}-mc-raw-trend", value="power"),
                    dcc.Dropdown(id=f"{prefix}-mc-raw-bins", value=5),
                    dcc.Checklist(id=f"{prefix}-mc-raw-regime", value=list(range(5))),
                    html.Div(id=f"{prefix}-mc-lines-wrap"),
                    dbc.Input(id=f"{prefix}-mc-lines", value=raw_mc.RAW_LINES_DEFAULT),
                    dcc.Checklist(id=f"{prefix}-mc-extremes", value=[]),
                    html.Div(id=f"{prefix}-mc-raw-status"),
                    html.Div(id="bub-mc-panel")]
        return html.Div(style=_STYLE_HIDDEN, children=_ph)
    yr_now = pd.Timestamp.today().year
    raw = include_raw_src
    return html.Div(style={"position": "relative"},
                    **({"id": "bub-mc-panel", "className": ""} if raw else {}),
                    children=[
        html.Span([
            html.Span("\u2694", style={"fontSize": UI_FONT_XXL, "marginRight": "3px"}),
            html.Span("NEW", style={"position": "relative", "top": "-2px"}),
        ], className="mc-new-badge", style={
            "position": "absolute", "top": "4px", "right": "-2px",
            "fontWeight": "900", "color": SILVER,
            "fontFamily": "'Impact', 'Arial Black', sans-serif",
            "textTransform": "uppercase",
            "backgroundColor": NEAR_BLACK,
            "borderRadius": "5px", "transform": "rotate(18deg)",
            "zIndex": "1", "lineHeight": "1.2",
            "boxShadow": f"0 2px 6px {_hex_alpha(BLACK, CTX_MENU_SHADOW_ALPHA)}, inset 0 1px 0 {_hex_alpha(WHITE, BADGE_INSET_ALPHA)}",
            "textShadow": f"0 0 4px {_hex_alpha(BADGE_GLOW_RED, BADGE_GLOW_ALPHA)}",
        }),
        _section_card(
        "Monte Carlo Simulation",
        html.Div(id=f"{prefix}-mc-body", style=_STYLE_HIDDEN, children=[
            *([dcc.Checklist(
                id=f"{prefix}-mc-paths",
                options=[{"label": " Show individual price paths",
                          "value": "yes"}],
                value=[], inputStyle=_CB_MARGIN,
                style={"fontSize": UI_FONT_MD, "color": MUTED_TEXT,
                       "marginBottom": "6px"})]
              if show_paths_toggle else []),
            *([_raw_lines_block(prefix)] if raw else []),
            *_wrap(_MO, raw, dcc.Checklist(id=f"{prefix}-mc-advanced",
                          options=[{"label": " Advanced simulator options", "value": "yes"}],
                          value=[], inputStyle=_CB_MARGIN,
                          style={"fontSize": UI_FONT_MD, "color": MUTED_TEXT, "marginBottom": "6px"})),
            html.Div(dcc.Slider(id=f"{prefix}-mc-entry-yr", value=yr_now),
                     style=_STYLE_HIDDEN),
            *_wrap(_MO, raw,
                _lbl((start_yr_label or "MC start year") + " (bold = cached)"),
                dcc.Dropdown(id=f"{prefix}-mc-start-yr",
                             options=_MC_START_YR_OPTIONS,
                             value=2031 if show_stack else 2028, clearable=False),
                _lbl("Entry percentile (10% steps, cache-aligned)"),
                dcc.Dropdown(id=f"{prefix}-mc-entry-q",
                             options=_MC_ENTRY_Q_OPTIONS,
                             value=default_entry_q, clearable=False)),
            # Wrapped: an inline display on the classed element itself would
            # beat the class rules in assets/style.css.
            *([html.Div(className=_RO, children=html.Small(RAW_HINT, style=_SMALL_NOTE))]
              if raw else []),
            # Raw draws no model's bands: "Source" there (final review M-6)
            (html.Label([html.Span("Model source (quantile bands)", className=_MO),
                         html.Span("Source", className=_RO)],
                        className="form-label mb-0 small")
             if raw else _lbl("Model source (quantile bands)")),
            dcc.Dropdown(id=f"{prefix}-mc-model-src",
                         options=_mc_model_src_options(include_raw=raw),
                         value="bub", clearable=False),
            *([html.Div(id=f"{prefix}-mc-raw-status", className=_RO,
                        style={"fontSize": UI_FONT_SM, "color": DIM_TEXT,
                               "lineHeight": "1.35", "margin": "4px 0 6px"})]
              if raw else []),
            *([_lbl(amount_label + " (0–4,294,967,295)"),
               dbc.Input(id=f"{prefix}-mc-amount", type="number",
                         value=amount_default, min=0, max=_app_ctx.MAX_USD,
                         step=1, size="sm", debounce=True),
              ] if show_amount and "amount" not in shared_controls else (
              [dbc.Input(id=f"{prefix}-mc-amount", type="number",
                         value=amount_default, style=_STYLE_HIDDEN),
              ] if "amount" not in shared_controls else [])),
            *([ _lbl("Inflation rate (% / yr)"),
                dcc.Dropdown(id=f"{prefix}-mc-infl",
                             options=_MC_INFL_OPTIONS,
                             value=4, clearable=False),
            ] if show_inflation and "infl" not in shared_controls else (
            [   dcc.Dropdown(id=f"{prefix}-mc-infl", value=0,
                             style=_STYLE_HIDDEN),
            ] if "infl" not in shared_controls else [])),
            *([ _lbl("Starting BTC stack (0–1000)"),
                dbc.Input(id=f"{prefix}-mc-stack", type="number",
                          min=0, max=1000, step=0.01, value=1.0, size="sm",
                          debounce=True),
            ] if show_stack and "stack" not in shared_controls else (
            [   dbc.Input(id=f"{prefix}-mc-stack", type="number",
                          value=1.0, style=_STYLE_HIDDEN),
            ] if "stack" not in shared_controls else [])),
            *_wrap(_MO, raw,
                _lbl("Years to model"),
                dcc.Dropdown(id=f"{prefix}-mc-years",
                             options=_MC_YEARS_OPTIONS,
                             value=40, clearable=False)),
            *([_raw_controls_block(prefix, yr_now)] if raw else []),
            # Advanced controls (hidden until checkbox toggled)
            *_wrap(_MO, raw, html.Div(id=f"{prefix}-mc-adv-body", style=_STYLE_HIDDEN, children=[
                _lbl("Markov transition matrix dimension"),
                dcc.Dropdown(id=f"{prefix}-mc-bins",
                             options=_bold_opts(
                                 list(range(5, 11)),
                                 lambda v: f"{v}\u00d7{v}", {5}),
                             value=5, clearable=False),
                _lbl("Price regime filter"),
                dcc.Checklist(
                    id=f"{prefix}-mc-regime",
                    options=_MC_REGIME_OPTIONS_5,
                    value=list(range(5)),
                    inputStyle=_CB_MARGIN,
                    labelStyle={"display": "block", "fontSize": UI_FONT_MD,
                                "lineHeight": "1.6", "color": MODAL_DIVIDER_DARK},
                    style={"marginBottom": "2px"},
                ),
                _bub_regime_note() if prefix == "bub" else
                html.Small(
                    "Free-tier: cached paths sorted by alignment with your "
                    "selection (each path may visit blocked regimes briefly). "
                    "Filter only materially changes output when sims < 200 "
                    "(cache size); at 200, paths are reordered but the same "
                    "set is used — percentiles unchanged. Paid live MC: paths "
                    "cannot leave selected regimes.",
                    style={"display": "block", "fontSize": UI_FONT_SM,
                           "color": DIM_TEXT, "lineHeight": "1.3",
                           "marginBottom": "8px"},
                ),
                _lbl("Simulations (1–3200; ≤200 is free)"),
                # Number input + HTML5 datalist: type any integer 1-3200,
                # or pick from the preset list. ≤200 is free-tier (cache
                # holds 200 paths). >200 triggers paid flow per is_free_tier.
                html.Datalist(
                    id=f"{prefix}-mc-sims-presets",
                    children=[html.Option(value=str(v))
                              for v in (sims_options
                                        or [8, 16, 32, 64, 128, 200, 400, 800, 1600, 3200])],
                ),
                dbc.Input(id=f"{prefix}-mc-sims", type="number",
                          value=default_sims,
                          min=1, max=3200, step=1, size="sm",
                          debounce=True, list=f"{prefix}-mc-sims-presets"),
                *([ _lbl("Frequency"),
                    dcc.Dropdown(id=f"{prefix}-mc-freq",
                                 options=_bold_opts(
                                     ["Daily", "Weekly", "Monthly", "Quarterly", "Annually"],
                                     str, {"Monthly"}),
                                 value="Monthly", clearable=False),
                    _lbl("Periods per year"),
                    dbc.Input(id=f"{prefix}-mc-ppy", value="12/yr", size="sm",
                              disabled=True),
                ] if "freq" not in shared_controls else []),
                _lbl("Historical window"),
                dcc.RangeSlider(id=f"{prefix}-mc-window", min=2010,
                                max=yr_now, value=[2010, yr_now],
                                marks={y: str(y) for y in range(2010, yr_now + 1, 5)}),
            ])),
            *_wrap(_MO, raw, html.Div(id=f"{prefix}-mc-cost",
                     style={"fontSize": UI_FONT_MD, "color": DIM_TEXT, "marginTop": "6px",
                            "lineHeight": "1.4"})),
            dcc.Store(id=f"{prefix}-mc-price-val", storage_type="memory", data=0),
            # ── Run Simulation button (payment-gated when BTCPay active) ──
            *_wrap(_MO, raw, dbc.Button(
                [html.Span("\u26a1 ", style={"fontSize": UI_FONT_XL}), "Run MC Simulation"],
                id={"type": "mc-run-btn", "tab": prefix}, size="sm", color="warning",
                className="w-100 mt-2",
                style={"fontWeight": "600"},
            ),
            html.Div(id={"type": "mc-run-status", "tab": prefix},
                     style={"fontSize": UI_FONT_SM, "color": DIM_TEXT, "marginTop": "4px",
                            "textAlign": "center"})),
            dcc.Store(id=f"{prefix}-mc-rendered-key", storage_type="memory"),
            html.Div(id=f"{prefix}-mc-match",
                     style={"fontSize": UI_FONT_SM, "marginTop": "4px",
                            "textAlign": "center"}),
            *_wrap(_MO, raw, dbc.Button("\u21a9 Restore last settings",
                       id=f"{prefix}-mc-restore-btn",
                       color="warning", size="sm", outline=True,
                       className="w-100 mt-1",
                       style=_STYLE_HIDDEN),
            html.Hr(className="my-2"),
            _section_card("Saved Simulation",
                html.Div(id=f"{prefix}-mc-status",
                         style={"fontSize": UI_FONT_SM, "color": DIM_TEXT, "marginBottom": "4px"}),
                dbc.Row([
                    dbc.Col(
                        dbc.Button("\u2b07 Save", id=f"{prefix}-mc-dl-btn",
                                   size="sm", color="secondary", className="w-100"),
                        width=6),
                    dbc.Col(
                        dcc.Upload(
                            id=f"{prefix}-mc-upload",
                            children=dbc.Button("\u2b06 Load", size="sm",
                                                color="secondary", className="w-100"),
                            accept=".json", multiple=False,
                        ),
                        width=6),
                ], className="g-1"),
                html.Div(id=f"{prefix}-mc-upload-status", className="mt-1",
                         style={"fontSize": UI_FONT_SM}),
            )),
        ]),
        header_right=[
            dcc.Checklist(
                id=f"{prefix}-mc-enable",
                options=[{"label": " Activate", "value": "yes"}],
                value=_mc_enable_val, inputStyle=_CB_MARGIN,
                className="model-panel-activate",
            ),
            html.Span([
                html.Span("\u26a1", style={"fontSize": UI_FONT_LG}),
                " Paid",
            ], className="model-panel-paid-badge" + (f" {_MO}" if raw else "")),
        ],
    ),
    ])
