"""Model Scanner callbacks — bidirectional price/date/quantile lookup."""

import numpy as np
import pandas as pd
from dash import html, Input, Output, State, callback, no_update, ctx, ALL

import _app_ctx
from btc_core import today_t, fmt_price, yr_to_t
from colors import UCL_LINE_COLOR, UI_FONT_XS, UI_FONT_MD

# LPPL family keys — filtered by LPPL config panel. Union of visible
# bases {lppl, lp2, lp3, lp4} with weighted/no-13 variants.
_LPPL_FAMILY_KEYS = (
    {"lppl", "lp2", "lp3", "lp4"}
    | set(_app_ctx.LPPL_FAMILY_HIDDEN_FROM_BUBBLE)
)

# Canonical model ordering — mirrors the Display Models config panel:
# BM → PL → QR → Entropy PPL → Hybrid PPL → LPPL → PCA → Greedy → EF →
# deprioritized (S2F, Gompertz, BPL, Exponential-excluded) → U₁ → MC.
# LPPL family variants (lp2/lp3/lp4/…) all collapse to the LPPL slot.
_SCANNER_ORDER = (
    "bub", "pl", "qr", "eppl", "hybppl", "lppl",
    "pca", "grdy", "ef", "s2f", "s2f_inst", "gomp", "bpl", "spl", "u1", "mc",
)


def _scanner_sort_key(model_key: str) -> tuple[int, str]:
    """Sort key for Display Models-ordered scanner rows.

    LPPL family variants (lp2, lp3, lp4, lppl_w, lp4_n13, etc.) all map
    to the LPPL slot so the single active variant appears where "LPPL"
    lives in the Display Models panel.
    """
    if model_key in _LPPL_FAMILY_KEYS:
        return (_SCANNER_ORDER.index("lppl"), model_key)
    try:
        return (_SCANNER_ORDER.index(model_key), model_key)
    except ValueError:
        return (len(_SCANNER_ORDER), model_key)  # unknown keys sink to bottom


def _solve_date(model, q_frac, target_price, sigma_mode="constant"):
    """Root-find t where model.price_at(q, t) = target_price."""
    from scipy.optimize import brentq
    log_target = np.log10(max(target_price, 1e-10))
    def f(t):
        return np.log10(max(float(model.price_at(q_frac, t, sigma_mode=sigma_mode)), 1e-10)) - log_target
    try:
        t = brentq(f, 0.5, 72.0)
        genesis = _app_ctx.M.genesis
        date = genesis + pd.Timedelta(days=t * 365.25)
        return date.strftime("%Y-%m-%d")
    except (ValueError, RuntimeError):
        return "\u2014"


def _output_from_history(history):
    """The output field is whichever of p/d/q is NOT in the last-two-edited list."""
    all_fields = {"p", "d", "q"}
    remaining = all_fields - set(history[-2:] if len(history) >= 2 else history)
    if len(remaining) == 1:
        return remaining.pop()
    # Ambiguous — default to q
    return "q"


@callback(
    Output("scan-output-field", "data"),
    Output("scan-results", "children"),
    Output("scan-price-hint", "style"),
    Input("bubble-first-render", "data"),
    Input("scan-price",          "value"),
    Input("scan-date",           "value"),
    Input("scan-q",              "value"),
    State("scan-output-field",   "data"),
    State("btc-price-store",     "data"),
    State("user-model-store",    "data"),
    State("lppl-n-freqs",        "value"),
    State("lppl-weighted",       "value"),
    State("lppl-no-13",          "value"),
    State("bub-sigma-mode",      "value"),
    prevent_initial_call=True,
)
def update_scanner(_first_render, price_val, date_val, q_val,
                   edit_history, live_price, user_model_data,
                   lppl_n_freqs, lppl_weighted, lppl_no_13,
                   sigma_mode):
    """Compute the missing variable across all models."""
    genesis = _app_ctx.M.genesis

    # Merge registered models + user model (if defined)
    from btc_core import UserModel
    _models = dict(_app_ctx.PRICE_MODELS)
    if user_model_data:
        um = UserModel.from_store_dict(user_model_data)
        if um:
            _models["u1"] = um

    # Filter LPPL family variants to match the LPPL config panel selection.
    # Translate ["lppl"] + config → specific LPPL variant keys via the same
    # master-gate resolver the chart callbacks use.
    from callbacks.charts import _resolve_lppl_master
    _active_lppl = set(_resolve_lppl_master(
        ["lppl"], lppl_n_freqs, lppl_weighted, lppl_no_13))
    _models = {k: m for k, m in _models.items()
               if k not in _LPPL_FAMILY_KEYS or k in _active_lppl}
    _models.pop("exp", None)  # Exponential is excluded from the scanner
    # Exclude HybPPL/EPPL config variants + HybPPL family variants.
    # Keep only the base hybppl and eppl entries (default configurations).
    _HYBPPL_FAMILY_EXTRAS = {"hybppl_dd", "hyb2l", "hyb2c", "hyb2b", "hyb4d", "linppl"}
    _models = {k: m for k, m in _models.items()
               if not k.startswith("cfg_") and not k.startswith("ecfg_")
               and k not in _HYBPPL_FAMILY_EXTRAS}

    # Sort models by Display Models panel order (defined in _SCANNER_ORDER above)
    _models = dict(sorted(_models.items(), key=lambda kv: _scanner_sort_key(kv[0])))

    trigger = ctx.triggered_id if ctx.triggered_id else None

    # edit_history is a list of the last 2 edited fields, e.g. ["p", "d"]
    if not isinstance(edit_history, list):
        edit_history = ["p", "d"]  # default: price+date edited → q is output

    # Update history based on what was just edited
    trigger_map = {"scan-price": "p", "scan-date": "d", "scan-q": "q"}
    val_map = {"scan-price": price_val, "scan-date": date_val, "scan-q": q_val}
    if trigger in trigger_map:
        field = trigger_map[trigger]
        raw = val_map[trigger]
        field_empty = (raw is None or raw == "")
        # Remove this field from history
        edit_history = [f for f in edit_history if f != field]
        if not field_empty:
            # Field has a value → it's an input → add to history
            edit_history.append(field)
            edit_history = edit_history[-2:]
        # else: field was cleared → leave it out → becomes output

    output_field = _output_from_history(edit_history)

    # Resolve defaults
    use_live = (price_val is None or price_val == "")
    if use_live and output_field != "p":
        price = float(live_price) if live_price else None
    elif use_live:
        price = None
    else:
        price = float(price_val)

    hint_style = {"fontSize": UI_FONT_XS} if (use_live and output_field != "p") else {"fontSize": UI_FONT_XS, "display": "none"}

    if (date_val is None or date_val == "") and output_field != "d":
        date_val = pd.Timestamp.today().strftime("%Y-%m-%d")

    date_empty = (date_val is None or date_val == "")
    t = (pd.to_datetime(date_val) - genesis).days / 365.25 if not date_empty else None
    if t is not None and t <= 0:
        t = 0.5

    q_frac = float(q_val) / 100.0 if q_val is not None and q_val != "" else None

    rows = []
    sigma_mode = sigma_mode or "constant"

    if output_field == "q" and price is not None and t is not None:
        # Unfairly Cheap Line
        ucl_price = 10 ** (_app_ctx.UCL_INTERCEPT + _app_ctx.UCL_SLOPE * np.log10(t))
        ucl_ratio = price / ucl_price
        ucl_style = {"fontSize": UI_FONT_MD, "color": UCL_LINE_COLOR}
        rows.append(html.Tr([
            html.Td("Unfairly Cheap Line", style=ucl_style),
            html.Td(f"{ucl_ratio:.2f}x above", style={**ucl_style, "fontWeight": "bold"}),
        ]))

        for key, mdl in _models.items():
            if not mdl.quantized:
                continue  # skip S2F etc — quantiles don't apply
            pct = mdl.find_percentile(t, price, sigma_mode=sigma_mode)
            rows.append(html.Tr([
                html.Td(mdl.name, style={"fontSize": UI_FONT_MD}),
                html.Td(f"Q{pct*100:.1f}%", style={"fontSize": UI_FONT_MD,
                         "fontWeight": "bold"}),
            ], id={"type": "scan-row", "model": key},
               style={"cursor": "pointer"},
               **{"data-t": f"{t:.6f}", "data-price": f"{price:.2f}",
                  "data-model": key}))

    elif output_field == "p" and q_frac is not None and t is not None:
        for key, mdl in _models.items():
            if not mdl.quantized:
                continue
            try:
                p = float(mdl.price_at(q_frac, t, sigma_mode=sigma_mode))
                price_str = fmt_price(p)
            except Exception:
                p = None
                price_str = "\u2014"
            row_data = {"data-t": f"{t:.6f}", "data-model": key}
            if p is not None:
                row_data["data-price"] = f"{p:.2f}"
            rows.append(html.Tr([
                html.Td(mdl.name, style={"fontSize": UI_FONT_MD}),
                html.Td(price_str, style={"fontSize": UI_FONT_MD,
                         "fontWeight": "bold"}),
            ], id={"type": "scan-row", "model": key},
               style={"cursor": "pointer"}, **row_data))

    elif output_field == "d" and price is not None and q_frac is not None:
        for key, mdl in _models.items():
            if not mdl.quantized:
                continue
            date_str = _solve_date(mdl, q_frac, price, sigma_mode=sigma_mode)
            row_data = {"data-price": f"{price:.2f}", "data-model": key}
            if date_str != "\u2014":
                try:
                    solved_t = (pd.to_datetime(date_str) - genesis).days / 365.25
                    row_data["data-t"] = f"{solved_t:.6f}"
                except Exception:
                    pass
            rows.append(html.Tr([
                html.Td(mdl.name, style={"fontSize": UI_FONT_MD}),
                html.Td(date_str, style={"fontSize": UI_FONT_MD,
                         "fontWeight": "bold"}),
            ], id={"type": "scan-row", "model": key},
               style={"cursor": "pointer"}, **row_data))

    header_map = {"q": "Quantile", "p": "Price", "d": "Date"}
    table = html.Table([
        html.Thead(html.Tr([
            html.Th("Model", style={"fontSize": UI_FONT_MD, "paddingRight": "12px"}),
            html.Th(header_map.get(output_field, ""),
                     style={"fontSize": UI_FONT_MD}),
        ])),
        html.Tbody(rows),
    ], style={"width": "100%", "borderCollapse": "collapse",
              "marginTop": "6px"}) if rows else html.Small(
                  "Enter values above", className="text-muted")

    return (edit_history, table, hint_style)


# Clientside: toggle a model's scanner line on/off when its row is clicked.
# Moved from server to clientside (no state needed beyond the trigger id
# and the existing active list) so a click doesn't round-trip to gunicorn.
_app_ctx.app.clientside_callback(
    """
    function(n_clicks_list, active) {
        var NU = window.dash_clientside.no_update;
        var ctx = window.dash_clientside.callback_context;
        if (!ctx.triggered || !ctx.triggered.length) return NU;
        var trig = ctx.triggered[0];
        if (!trig.value) return NU;  // component creation / no real click
        var prop_id = trig.prop_id;
        // prop_id looks like '{"model":"lppl","type":"scan-row"}.n_clicks'
        var open_brace = prop_id.indexOf('{');
        var close_brace = prop_id.lastIndexOf('}');
        if (open_brace < 0 || close_brace < 0) return NU;
        var id_str = prop_id.slice(open_brace, close_brace + 1);
        var model_key;
        try { model_key = JSON.parse(id_str).model; } catch (e) { return NU; }
        active = (active || []).slice();
        var idx = active.indexOf(model_key);
        if (idx >= 0) active.splice(idx, 1);
        else active.push(model_key);
        return active;
    }
    """,
    Output("scan-active-rows", "data"),
    Input({"type": "scan-row", "model": ALL}, "n_clicks"),
    State("scan-active-rows", "data"),
    prevent_initial_call=True,
)


# ── Auto-extend bubble x-range when radar beacon falls outside ───────────────

_app_ctx.app.clientside_callback(
    """
    function(active, xrange) {
        var NU = window.dash_clientside.no_update;
        if (!active || active.length === 0) return NU;

        var genesis = new Date("2009-07-25T00:00:00");
        var MS_PER_YR = 365.25 * 86400000;
        var rows = document.querySelectorAll("#scan-results tr[data-model]");
        var needMin = xrange[0], needMax = xrange[1];
        var changed = false;

        rows.forEach(function(row) {
            var model = row.getAttribute("data-model");
            if (active.indexOf(model) === -1) return;
            var t = parseFloat(row.getAttribute("data-t"));
            if (!t) return;
            var d = new Date(genesis.getTime() + t * MS_PER_YR);
            var yr = d.getFullYear() + d.getMonth() / 12;
            if (yr < needMin) { needMin = Math.floor(yr) - 1; changed = true; }
            if (yr > needMax) { needMax = Math.ceil(yr) + 1; changed = true; }
        });

        if (!changed) return NU;
        needMin = Math.max(2010, needMin);
        needMax = Math.min(2080, needMax);
        return [needMin, needMax];
    }
    """,
    Output("bub-xrange", "value", allow_duplicate=True),
    Input("scan-active-rows", "data"),
    State("bub-xrange", "value"),
    prevent_initial_call=True,
)


# ── Scanner header label tracks σ mode (clientside, no round-trip) ─────────
_app_ctx.app.clientside_callback(
    """function(mode) {
        var label = (mode === "resqr") ? "Residual quantile" : "Constant \u03c3";
        return "Model Scanner \u00b7 " + label;
    }""",
    Output("bub-scanner-header", "children"),
    Input("bub-sigma-mode", "value"),
)

# ── σ mode change bumps bubble-first-render to re-trigger chart callback ───
# bub-sigma-mode is a State (not Input) on update_bubble to avoid changing
# the callback signature and causing stale /_dash-dependencies errors on
# mobile browsers. This clientside callback converts the radio change into
# a first-render bump, which IS an existing Input.
_app_ctx.app.clientside_callback(
    """function(mode, cur) { return (cur || 0) + 1; }""",
    Output("bubble-first-render", "data", allow_duplicate=True),
    Input("bub-sigma-mode", "value"),
    State("bubble-first-render", "data"),
    prevent_initial_call=True,
)
