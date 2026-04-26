"""Master → variant resolvers + Component Decomposition callbacks.

Pure-function helpers:
  _resolve_lppl_master, _resolve_hm_lppl_master
  _resolve_hybppl_master, _resolve_hm_hybppl_master, _build_hybppl_config_key
  _resolve_eppl_master, _resolve_hm_eppl_master, _build_eppl_config_key
  _decomp_warning_banner, _r2_of_log_pred, _component_label
  update_decomp_options, _prune_decomp_value, _resolve_decomp_model_key

Decomposition @callbacks (registered on import):
  _update_decomp_options_cb  → bub-decomp-components options + warning + body style
  _update_active_formula_cb  → bub-decomp-active-formula
  _update_decomp_formula_cb  → bub-decomp-formula
  _prune_decomp_value_cb     → bub-decomp-components value (prune on model change)

Extracted from ``callbacks/charts.py`` to keep the chart-update callbacks
file focused on the six main figure-builder callbacks.
"""

import dash
from dash import Input, Output, State, ctx, callback, html

import numpy as np

import _app_ctx
from colors import (
    LINK, FALLBACK_MODEL_GRAY,
    LOT_MARKER_OUTLINE,
    DECOMP_ERROR_RED, ERROR_BG, ERROR_BORDER,
    UI_FONT_SM, UI_FONT_MD,
)


def _resolve_hm_lppl_master(hm_model, lppl_n_freqs, lppl_weighted, lppl_no_13):
    """Translate 'lppl' master to specific flavor key for heatmap (single-select).

    Returns the flavor key the heatmap figure builder should use. For non-lppl
    models or when lppl flavor cannot be resolved, returns input unchanged.
    """
    if hm_model != "lppl":
        return hm_model
    _weighted = "weighted" in (lppl_weighted or [])
    _no_13 = "no13" in (lppl_no_13 or [])
    _n_list = (lppl_n_freqs or [3])
    _n = _n_list[0] if _n_list else 3
    if _n == 1:
        return "lppl_w" if _weighted else "lppl"
    if _n == 2:
        return "lp2_w" if _weighted else "lp2"
    if _n == 3 and not _no_13:
        return "lp3_w" if _weighted else "lp3"
    if _n == 4:
        if _no_13:
            return "lp4_w_n13" if _weighted else "lp4_n13"
        return "lp4_w" if _weighted else "lp4"
    return "lppl"  # fallback


def _resolve_lppl_master(model_show, lppl_n_freqs, lppl_weighted, lppl_no_13):
    """Translate the 'lppl' master in model_show into specific flavor key(s).

    Strips 'lppl' from the list and appends one flavor key per checked
    n_freqs entry, applying weighted and no_13 modifiers. When no master
    is present, returns the list unchanged. When master is present but
    no flavor is selected, the master is stripped with no replacement.
    """
    model_show = list(model_show or [])
    if "lppl" not in model_show:
        return model_show
    model_show = [v for v in model_show if v != "lppl"]
    _weighted = "weighted" in (lppl_weighted or [])
    _no_13 = "no13" in (lppl_no_13 or [])
    for n in (lppl_n_freqs or []):
        if n == 1:
            model_show.append("lppl_w" if _weighted else "lppl")
        elif n == 2:
            model_show.append("lp2_w" if _weighted else "lp2")
        elif n == 3 and not _no_13:
            model_show.append("lp3_w" if _weighted else "lp3")
        elif n == 4:
            if _no_13:
                model_show.append("lp4_w_n13" if _weighted else "lp4_n13")
            else:
                model_show.append("lp4_w" if _weighted else "lp4")
    return model_show


def _build_hybppl_config_key(nlog, ncal, log1d, log2d, cal1d, cal2d):
    """Build a cfg_* key from the modal's radio-item values."""
    def _spec(n, d1, d2):
        if n == 0:
            return "0"
        if n == 1:
            return f"1{d1 or 'd'}"
        dirs = sorted([d1 or 'd', d2 or 'd'])
        return f"2{dirs[0]}{dirs[1]}"
    return f"cfg_{_spec(nlog, log1d, log2d)}_{_spec(ncal, cal1d, cal2d)}"


def _resolve_hybppl_master(model_show,
                            cfg_a_nlog, cfg_a_ncal,
                            cfg_a_log1d, cfg_a_log2d,
                            cfg_a_cal1d, cfg_a_cal2d,
                            cfg_b_enabled,
                            cfg_b_nlog, cfg_b_ncal,
                            cfg_b_log1d, cfg_b_log2d,
                            cfg_b_cal1d, cfg_b_cal2d):
    """Translate the 'hybppl' master in model_show into concrete cfg_* key(s).

    Strips 'hybppl' (and named variants) from the list and appends 1-2
    resolved cfg_* keys based on the modal config. When no master is
    present, returns the list unchanged.
    """
    model_show = list(model_show or [])
    _HYBPPL_NAMES = {"hybppl", "hybppl_dd", "hyb2l", "hyb2c", "hyb2b", "hyb4d"}
    if "hybppl" not in model_show:
        return model_show
    model_show = [v for v in model_show if v not in _HYBPPL_NAMES]
    # Model A
    key_a = _build_hybppl_config_key(
        cfg_a_nlog or 1, cfg_a_ncal or 1,
        cfg_a_log1d, cfg_a_log2d, cfg_a_cal1d, cfg_a_cal2d)
    if key_a in _app_ctx.PRICE_MODELS:
        model_show.append(key_a)
    # Model B (if enabled)
    if cfg_b_enabled and "yes" in (cfg_b_enabled or []):
        key_b = _build_hybppl_config_key(
            cfg_b_nlog or 0, cfg_b_ncal or 0,
            cfg_b_log1d, cfg_b_log2d, cfg_b_cal1d, cfg_b_cal2d)
        if key_b in _app_ctx.PRICE_MODELS and key_b != key_a:
            model_show.append(key_b)
    return model_show


def _resolve_hm_hybppl_master(hm_model,
                               cfg_a_nlog, cfg_a_ncal,
                               cfg_a_log1d, cfg_a_log2d,
                               cfg_a_cal1d, cfg_a_cal2d):
    """Translate 'hybppl' master to specific cfg_* key for heatmap (single-select).

    Returns the resolved model key. For non-hybppl models, returns unchanged.
    """
    if hm_model != "hybppl":
        return hm_model
    key = _build_hybppl_config_key(
        cfg_a_nlog or 1, cfg_a_ncal or 1,
        cfg_a_log1d, cfg_a_log2d, cfg_a_cal1d, cfg_a_cal2d)
    return key if key in _app_ctx.PRICE_MODELS else hm_model


def _build_eppl_config_key(nlog, ncal, log1d, log2d, cal1d, cal2d):
    """Build an ecfg_* key from the modal's radio-item values."""
    def _spec(n, d1, d2):
        if n == 0:
            return "0"
        if n == 1:
            return f"1{d1 or 'd'}"
        dirs = sorted([d1 or 'd', d2 or 'd'])
        return f"2{dirs[0]}{dirs[1]}"
    return f"ecfg_{_spec(nlog, log1d, log2d)}_{_spec(ncal, cal1d, cal2d)}"


def _resolve_eppl_master(model_show,
                          cfg_a_nlog, cfg_a_ncal,
                          cfg_a_log1d, cfg_a_log2d,
                          cfg_a_cal1d, cfg_a_cal2d,
                          cfg_b_enabled,
                          cfg_b_nlog, cfg_b_ncal,
                          cfg_b_log1d, cfg_b_log2d,
                          cfg_b_cal1d, cfg_b_cal2d):
    """Translate the 'eppl' master in model_show into concrete ecfg_* key(s).

    Strips 'eppl' from the list and appends 1-2 resolved ecfg_* keys
    based on the modal config. When no master is present, returns the
    list unchanged.
    """
    model_show = list(model_show or [])
    if "eppl" not in model_show:
        return model_show
    model_show = [v for v in model_show if v != "eppl"]
    # Model A
    key_a = _build_eppl_config_key(
        cfg_a_nlog or 1, cfg_a_ncal or 1,
        cfg_a_log1d, cfg_a_log2d, cfg_a_cal1d, cfg_a_cal2d)
    if key_a in _app_ctx.PRICE_MODELS:
        model_show.append(key_a)
    # Model B (if enabled)
    if cfg_b_enabled and "yes" in (cfg_b_enabled or []):
        key_b = _build_eppl_config_key(
            cfg_b_nlog or 0, cfg_b_ncal or 0,
            cfg_b_log1d, cfg_b_log2d, cfg_b_cal1d, cfg_b_cal2d)
        if key_b in _app_ctx.PRICE_MODELS and key_b != key_a:
            model_show.append(key_b)
    return model_show


def _resolve_hm_eppl_master(hm_model,
                              cfg_a_nlog, cfg_a_ncal,
                              cfg_a_log1d, cfg_a_log2d,
                              cfg_a_cal1d, cfg_a_cal2d):
    """Translate 'eppl' master to specific ecfg_* key for heatmap (single-select).

    Returns the resolved model key. For non-eppl models, returns unchanged.
    """
    if hm_model != "eppl":
        return hm_model
    key = _build_eppl_config_key(
        cfg_a_nlog or 1, cfg_a_ncal or 1,
        cfg_a_log1d, cfg_a_log2d, cfg_a_cal1d, cfg_a_cal2d)
    return key if key in _app_ctx.PRICE_MODELS else hm_model


def _resolve_mc_model_src(src,
                          lppl_n_freqs, lppl_weighted, lppl_no_13,
                          hyb_a_nlog, hyb_a_ncal,
                          hyb_a_log1d, hyb_a_log2d,
                          hyb_a_cal1d, hyb_a_cal2d,
                          ep_a_nlog, ep_a_ncal,
                          ep_a_log1d, ep_a_log2d,
                          ep_a_cal1d, ep_a_cal2d):
    """Translate the MC quantile-bands master in `mc_model_src` to a concrete
    PRICE_MODELS key, using the global config-modal state.

    Composes the existing single-string heatmap resolvers — same pattern,
    so 'hybppl' / 'eppl' / 'lppl' masters resolve via the cfg/ecfg modals.
    Non-master values pass through unchanged.

    Master fallback: if the chain resolves to a variant not on disk, look
    up the master's preferred cached alias in MASTER_TO_CACHED_FALLBACK
    (e.g. 'lppl' → 'lppl', 'eppl' → 'ecfg_1d_1u'). Masters with no entry
    keep their resolved variant — silent cache miss falls through to live
    compute on paid tier.
    """
    # Pin the master before the chain mutates `src` — required for the
    # MASTER_TO_CACHED_FALLBACK lookup below to key on the original input.
    # See test_resolver_eppl_master_falls_back_when_variant_uncached.
    from mc_cache import _CACHED_MODEL_KEYS, MASTER_TO_CACHED_FALLBACK
    master = src

    resolved = _resolve_hm_lppl_master(master, lppl_n_freqs, lppl_weighted, lppl_no_13)
    resolved = _resolve_hm_hybppl_master(
        resolved,
        hyb_a_nlog, hyb_a_ncal, hyb_a_log1d, hyb_a_log2d,
        hyb_a_cal1d, hyb_a_cal2d)
    resolved = _resolve_hm_eppl_master(
        resolved,
        ep_a_nlog, ep_a_ncal, ep_a_log1d, ep_a_log2d,
        ep_a_cal1d, ep_a_cal2d)
    if resolved not in _CACHED_MODEL_KEYS:
        # `.get(master, resolved)` default-to-resolved means: masters without
        # a fallback entry (e.g. hybppl) keep their resolved variant — silent
        # cache miss falls through to live compute on paid tier.
        resolved = MASTER_TO_CACHED_FALLBACK.get(master, resolved)
    return resolved


def _decomp_warning_banner(n_checked):
    """Inline banner shown when LPPL decomposition needs exactly 1 n_freqs."""
    return html.Div(
        html.Small(
            f"Pick exactly one LPPL variant in the LPPL config panel "
            f"to decompose (currently {n_checked} checked).",
            style={"color": DECOMP_ERROR_RED},
        ),
        style={"padding": "6px 8px", "backgroundColor": ERROR_BG,
                "border": f"1px solid {ERROR_BORDER}", "borderRadius": "4px",
                "fontSize": UI_FONT_MD, "marginTop": "6px"},
    )


def _r2_of_log_pred(log_pred, log_actual):
    """Compute R² of a log10 prediction array vs actual log10 prices."""
    residuals = log_actual - log_pred
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((log_actual - np.mean(log_actual)) ** 2))
    if ss_tot <= 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _component_label(model, name):
    """Return the component name as the checkbox label (clean)."""
    return f" {name}"


def update_decomp_options(family, n_freqs, weighted, no_13, hybppl_cfg=None, eppl_cfg=None):
    """Populate Component Decomposition checklist options + warning + visibility.

    Returns (options, warning_children, body_style). NEVER modifies
    bub-decomp-components.value (see prune_decomp_value_on_model_change).
    """
    if not family:
        return [], [], {"display": "none"}
    if family == "lppl" and len(n_freqs or []) != 1:
        return [], _decomp_warning_banner(len(n_freqs or [])), {"display": "block"}
    key = _resolve_decomp_model_key(family, n_freqs, weighted, no_13,
                                     hybppl_cfg=hybppl_cfg, eppl_cfg=eppl_cfg)
    if key is None:
        return [], _decomp_warning_banner(len(n_freqs or [])), {"display": "block"}
    model = _app_ctx.PRICE_MODELS.get(key)
    if model is None:
        return [], [], {"display": "none"}
    opts = [{"label": _component_label(model, name), "value": name}
            for name in model.component_names]
    return opts, [], {"display": "block"}


@callback(
    Output("bub-decomp-components", "options"),
    Output("bub-decomp-warning",    "children"),
    Output("bub-decomp-body",       "style"),
    Input("bubble-first-render", "data"),
    Input("bub-decomp-model",  "value"),
    State("lppl-n-freqs",      "value"),
    State("lppl-weighted",     "value"),
    State("lppl-no-13",        "value"),
    State("hybppl-cfg-a-nlog", "value"),
    State("hybppl-cfg-a-ncal", "value"),
    State("hybppl-cfg-a-log1d", "value"),
    State("hybppl-cfg-a-log2d", "value"),
    State("hybppl-cfg-a-cal1d", "value"),
    State("hybppl-cfg-a-cal2d", "value"),
    State("hybppl-cfg-b-nlog", "value"),
    State("hybppl-cfg-b-ncal", "value"),
    State("hybppl-cfg-b-log1d", "value"),
    State("hybppl-cfg-b-log2d", "value"),
    State("hybppl-cfg-b-cal1d", "value"),
    State("hybppl-cfg-b-cal2d", "value"),
    State("eppl-cfg-a-nlog", "value"),
    State("eppl-cfg-a-ncal", "value"),
    State("eppl-cfg-a-log1d", "value"),
    State("eppl-cfg-a-log2d", "value"),
    State("eppl-cfg-a-cal1d", "value"),
    State("eppl-cfg-a-cal2d", "value"),
    State("eppl-cfg-b-nlog", "value"),
    State("eppl-cfg-b-ncal", "value"),
    State("eppl-cfg-b-log1d", "value"),
    State("eppl-cfg-b-log2d", "value"),
    State("eppl-cfg-b-cal1d", "value"),
    State("eppl-cfg-b-cal2d", "value"),
    prevent_initial_call=True,
)
def _update_decomp_options_cb(_first_render, family, n_freqs, weighted, no_13,
                               a_nlog, a_ncal, a_log1d, a_log2d, a_cal1d, a_cal2d,
                               b_nlog, b_ncal, b_log1d, b_log2d, b_cal1d, b_cal2d,
                               ea_nlog, ea_ncal, ea_log1d, ea_log2d, ea_cal1d, ea_cal2d,
                               eb_nlog, eb_ncal, eb_log1d, eb_log2d, eb_cal1d, eb_cal2d):
    hybppl_cfg = {
        "a_nlog": a_nlog, "a_ncal": a_ncal,
        "a_log1d": a_log1d, "a_log2d": a_log2d,
        "a_cal1d": a_cal1d, "a_cal2d": a_cal2d,
        "b_nlog": b_nlog, "b_ncal": b_ncal,
        "b_log1d": b_log1d, "b_log2d": b_log2d,
        "b_cal1d": b_cal1d, "b_cal2d": b_cal2d,
    }
    eppl_cfg = {
        "a_nlog": ea_nlog, "a_ncal": ea_ncal,
        "a_log1d": ea_log1d, "a_log2d": ea_log2d,
        "a_cal1d": ea_cal1d, "a_cal2d": ea_cal2d,
        "b_nlog": eb_nlog, "b_ncal": eb_ncal,
        "b_log1d": eb_log1d, "b_log2d": eb_log2d,
        "b_cal1d": eb_cal1d, "b_cal2d": eb_cal2d,
    }
    return update_decomp_options(family, n_freqs, weighted, no_13, hybppl_cfg=hybppl_cfg, eppl_cfg=eppl_cfg)


@callback(
    Output("bub-decomp-active-formula", "children"),
    Input("bubble-first-render", "data"),
    Input("bub-decomp-model",          "value"),
    Input("bub-decomp-components",     "value"),
    Input("bub-decomp-show-formulas",  "value"),
    State("lppl-n-freqs",      "value"),
    State("lppl-weighted",     "value"),
    State("lppl-no-13",        "value"),
    State("hybppl-cfg-a-nlog", "value"), State("hybppl-cfg-a-ncal", "value"),
    State("hybppl-cfg-a-log1d", "value"), State("hybppl-cfg-a-log2d", "value"),
    State("hybppl-cfg-a-cal1d", "value"), State("hybppl-cfg-a-cal2d", "value"),
    State("hybppl-cfg-b-nlog", "value"), State("hybppl-cfg-b-ncal", "value"),
    State("hybppl-cfg-b-log1d", "value"), State("hybppl-cfg-b-log2d", "value"),
    State("hybppl-cfg-b-cal1d", "value"), State("hybppl-cfg-b-cal2d", "value"),
    State("eppl-cfg-a-nlog", "value"), State("eppl-cfg-a-ncal", "value"),
    State("eppl-cfg-a-log1d", "value"), State("eppl-cfg-a-log2d", "value"),
    State("eppl-cfg-a-cal1d", "value"), State("eppl-cfg-a-cal2d", "value"),
    State("eppl-cfg-b-nlog", "value"), State("eppl-cfg-b-ncal", "value"),
    State("eppl-cfg-b-log1d", "value"), State("eppl-cfg-b-log2d", "value"),
    State("eppl-cfg-b-cal1d", "value"), State("eppl-cfg-b-cal2d", "value"),
    prevent_initial_call=True,
)
def _update_active_formula_cb(_first_render, family, selected, show_toggles, n_freqs, weighted, no_13,
                               a_nlog, a_ncal, a_log1d, a_log2d, a_cal1d, a_cal2d,
                               b_nlog, b_ncal, b_log1d, b_log2d, b_cal1d, b_cal2d,
                               ea_nlog, ea_ncal, ea_log1d, ea_log2d, ea_cal1d, ea_cal2d,
                               eb_nlog, eb_ncal, eb_log1d, eb_log2d, eb_cal1d, eb_cal2d):
    """Display the formula for the currently-checked subset -- gated on toggle."""
    from dash import html
    if "selected" not in (show_toggles or []):
        return []
    hybppl_cfg = {
        "a_nlog": a_nlog, "a_ncal": a_ncal,
        "a_log1d": a_log1d, "a_log2d": a_log2d,
        "a_cal1d": a_cal1d, "a_cal2d": a_cal2d,
        "b_nlog": b_nlog, "b_ncal": b_ncal,
        "b_log1d": b_log1d, "b_log2d": b_log2d,
        "b_cal1d": b_cal1d, "b_cal2d": b_cal2d,
    }
    eppl_cfg = {
        "a_nlog": ea_nlog, "a_ncal": ea_ncal,
        "a_log1d": ea_log1d, "a_log2d": ea_log2d,
        "a_cal1d": ea_cal1d, "a_cal2d": ea_cal2d,
        "b_nlog": eb_nlog, "b_ncal": eb_ncal,
        "b_log1d": eb_log1d, "b_log2d": eb_log2d,
        "b_cal1d": eb_cal1d, "b_cal2d": eb_cal2d,
    }
    key = _resolve_decomp_model_key(family, n_freqs, weighted, no_13,
                                     hybppl_cfg=hybppl_cfg, eppl_cfg=eppl_cfg)
    if key is None:
        return []
    model = _app_ctx.PRICE_MODELS.get(key)
    if model is None:
        return []
    details = getattr(model, "component_details", {})
    if not details:
        return []
    selected = list(selected or [])
    canonical = [n for n in model.component_names if n in selected]
    header = html.Div(html.Strong("Selected:"),
                      style={"marginBottom": "3px", "color": LOT_MARKER_OUTLINE})
    if not canonical:
        return [header, html.Small("(no components selected)",
                                    style={"color": FALLBACK_MODEL_GRAY})]
    log_parts = []
    product_parts = []
    for name in canonical:
        d = details.get(name)
        if not d:
            continue
        formula_str = d[0]
        log_parts.append(formula_str)
        product_parts.append(f"10^({formula_str})")
    if not log_parts:
        return [header]
    log_str = " + ".join(log_parts)
    product_str = " \u00b7 ".join(product_parts)
    return [
        header,
        html.Div([html.Strong("log\u2081\u2080(price) = "), log_str],
                 style={"marginBottom": "3px"}),
        html.Div([html.Strong("price = "), product_str]),
    ]


@callback(
    Output("bub-decomp-formula", "children"),
    Input("bubble-first-render", "data"),
    Input("bub-decomp-model",          "value"),
    Input("bub-decomp-show-formulas",  "value"),
    State("lppl-n-freqs",      "value"),
    State("lppl-weighted",     "value"),
    State("lppl-no-13",        "value"),
    State("hybppl-cfg-a-nlog", "value"), State("hybppl-cfg-a-ncal", "value"),
    State("hybppl-cfg-a-log1d", "value"), State("hybppl-cfg-a-log2d", "value"),
    State("hybppl-cfg-a-cal1d", "value"), State("hybppl-cfg-a-cal2d", "value"),
    State("hybppl-cfg-b-nlog", "value"), State("hybppl-cfg-b-ncal", "value"),
    State("hybppl-cfg-b-log1d", "value"), State("hybppl-cfg-b-log2d", "value"),
    State("hybppl-cfg-b-cal1d", "value"), State("hybppl-cfg-b-cal2d", "value"),
    prevent_initial_call=True,
)
def _update_decomp_formula_cb(_first_render, family, show_toggles, n_freqs, weighted, no_13,
                               a_nlog, a_ncal, a_log1d, a_log2d, a_cal1d, a_cal2d,
                               b_nlog, b_ncal, b_log1d, b_log2d, b_cal1d, b_cal2d):
    """Show the model's full formula — gated on toggle."""
    from dash import dcc, html
    if "full" not in (show_toggles or []):
        return []
    hybppl_cfg = {
        "a_nlog": a_nlog, "a_ncal": a_ncal,
        "a_log1d": a_log1d, "a_log2d": a_log2d,
        "a_cal1d": a_cal1d, "a_cal2d": a_cal2d,
        "b_nlog": b_nlog, "b_ncal": b_ncal,
        "b_log1d": b_log1d, "b_log2d": b_log2d,
        "b_cal1d": b_cal1d, "b_cal2d": b_cal2d,
    }
    key = _resolve_decomp_model_key(family, n_freqs, weighted, no_13,
                                     hybppl_cfg=hybppl_cfg)
    if key is None:
        return []
    model = _app_ctx.PRICE_MODELS.get(key)
    if model is None:
        return []
    latex = getattr(model, "formula_log10_latex", None)
    product = getattr(model, "formula_product_latex", None)
    if not latex:
        return []
    price_line = (rf"$$\text{{price}} = {product}$$" if product
                  else rf"$$\text{{price}} = 10^{{\,\log_{{10}}(\text{{price}})}}$$")
    return [
        html.Div(html.Strong("Full model:"),
                 style={"marginBottom": "3px", "color": LOT_MARKER_OUTLINE,
                        "fontSize": UI_FONT_MD}),
        dcc.Markdown(
            rf"""
$$\log_{{10}}(\text{{price}}) = {latex}$$

{price_line}
""",
            mathjax=True, className="small",
            style={"fontSize": UI_FONT_SM},
        ),
    ]


def _prune_decomp_value(family, options, current):
    """Keep only currently-valid values from the checklist."""
    if not family:
        return []
    valid = {o["value"] for o in (options or [])}
    return [v for v in (current or []) if v in valid]


@callback(
    Output("bub-decomp-components", "value", allow_duplicate=True),
    Input("bub-decomp-model",       "value"),
    State("bub-decomp-components", "options"),
    State("bub-decomp-components", "value"),
    prevent_initial_call=True,
)
def _prune_decomp_value_cb(family, opts, current):
    if ctx.triggered_id != "bub-decomp-model":
        raise dash.exceptions.PreventUpdate
    return _prune_decomp_value(family, opts, current)


def _resolve_decomp_model_key(family, lppl_n_freqs, lppl_weighted, lppl_no_13,
                               hybppl_cfg=None, eppl_cfg=None):
    """Translate (family, LPPL config) into a concrete model short_name.

    Returns None if family is empty OR if family is 'lppl' but exactly one
    n_freqs entry is not selected. Otherwise returns the model's short_name
    (e.g., 'bub', 'lp3_w', 'hybppl_dd').
    """
    if not family:
        return None
    # "hybppl" master (from Display Models consolidation) → resolve via
    # slot A of the HybPPL modal, matching the Display Models master → variant
    # routing behavior. Same for "eppl".
    if family == "hybppl" and hybppl_cfg:
        family = "hybppl_cfg_a"
    elif family == "eppl" and eppl_cfg:
        family = "eppl_cfg_a"
    # HybPPL config A/B -> resolve to cfg_* key
    if family in ("hybppl_cfg_a", "hybppl_cfg_b") and hybppl_cfg:
        slot = "a" if family.endswith("_a") else "b"
        nlog = hybppl_cfg.get(f"{slot}_nlog", 1)
        ncal = hybppl_cfg.get(f"{slot}_ncal", 1)
        log1d = hybppl_cfg.get(f"{slot}_log1d", "d")
        log2d = hybppl_cfg.get(f"{slot}_log2d", "d")
        cal1d = hybppl_cfg.get(f"{slot}_cal1d", "u")
        cal2d = hybppl_cfg.get(f"{slot}_cal2d", "u")
        key = _build_hybppl_config_key(nlog, ncal, log1d, log2d, cal1d, cal2d)
        return key if key in _app_ctx.PRICE_MODELS else None
    # EPPL config A/B -> resolve to ecfg_* key
    if family in ("eppl_cfg_a", "eppl_cfg_b") and eppl_cfg:
        slot = "a" if family.endswith("_a") else "b"
        nlog = eppl_cfg.get(f"{slot}_nlog", 1)
        ncal = eppl_cfg.get(f"{slot}_ncal", 1)
        log1d = eppl_cfg.get(f"{slot}_log1d", "d")
        log2d = eppl_cfg.get(f"{slot}_log2d", "d")
        cal1d = eppl_cfg.get(f"{slot}_cal1d", "u")
        cal2d = eppl_cfg.get(f"{slot}_cal2d", "u")
        key = _build_eppl_config_key(nlog, ncal, log1d, log2d, cal1d, cal2d)
        return key if key in _app_ctx.PRICE_MODELS else None
    if family != "lppl":
        return family
    if not lppl_n_freqs or len(lppl_n_freqs) != 1:
        return None
    n = lppl_n_freqs[0]
    weighted = "weighted" in (lppl_weighted or [])
    no13 = "no13" in (lppl_no_13 or [])
    if n == 1:
        return "lppl_w" if weighted else "lppl"
    if n == 2:
        return "lp2_w" if weighted else "lp2"
    if n == 3:
        return "lp3_w" if weighted else "lp3"
    if n == 4:
        if no13:
            return "lp4_w_n13" if weighted else "lp4_n13"
        return "lp4_w" if weighted else "lp4"
    return None
