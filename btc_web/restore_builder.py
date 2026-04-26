"""Restore figure builder — compute Plotly figures directly from snapshot
state dicts, server-side, without going through the Dash callback graph.

This module is the load-bearing component for fast share-link restore.
`restore_from_url` calls `_build_bubble_figure_from_state` and returns
the resulting Plotly figure as one of its Outputs (alongside the decoded
state and pending flag). The browser receives the figure in the same
HTTP response that decoded the URL hash, so the chart paints in
~150–300 ms instead of waiting 7+ seconds for the existing callback
cascade to eventually re-trigger update_bubble via CTA's tick bump.

See:
- memory/restore_callback_architecture.md (canonical analysis)
- docs/superpowers/plans/2026-04-25-restore-figure-in-decode-plan.md
- callbacks/charts/__init__.py update_bubble (mirrored param construction)
"""
from __future__ import annotations

import dash

from colors import (LOT_MARKER_COLOR, FALLBACK_MODEL_GRAY,
                    TRACE_WIDTH_COMPOSITE, TRACE_WIDTH_SUPPORT)
from tab_defaults import BUBBLE
from utils import _get_bubble_fig
# Resolvers are imported lazily inside the function — importing them at
# module load triggers callbacks/__init__.py which has clientside_callback
# registrations that need _app_ctx.app to be initialized.


# Mirrors layout.common._DEFAULT_BANDS / _bands_to_qs. Inlined to avoid
# pulling layout/__init__.py (which requires app context).
_DEFAULT_BANDS = (
    ("inner",  (0.15, 0.85)),
    ("outer",  (0.01, 0.99)),
    ("median", (0.50,)),
)
def _bands_to_qs(band_values):
    qs = []
    for name, vals in _DEFAULT_BANDS:
        if name in (band_values or []):
            qs.extend(vals)
    return sorted(set(qs))


_V_NO_DEFAULT = object()


def _v(state: dict, cid: str, prop: str = "value", default=_V_NO_DEFAULT):
    """Fetch a control's value from a snapshot state dict.

    When `default` is not explicitly passed, falls back to
    `SNAPSHOT_DEFAULTS[f"{cid}:{prop}"]` — making `snapshot_defaults.py`
    the mechanical authority for both encoding (sparse-diff baseline)
    AND decoding (builder fallback when state is missing the key).
    Pre-2026-04-25 each call site hardcoded its own default, which
    drifted from snapshot_defaults (51 divergences across 6 builders,
    e.g., `bub-legend-pos` hardcoded `"outside"` vs snapshot
    `"top-left"`). The AST-walker regression test
    `test_restore_builder_explicit_defaults_match_snapshot_defaults`
    blocks new explicit defaults from being added without a matching
    SNAPSHOT_DEFAULTS entry (or an INTENTIONAL_OVERRIDES allowlist
    entry in test_snapshot.py for deliberate exceptions).
    """
    if default is _V_NO_DEFAULT:
        from snapshot_defaults import SNAPSHOT_DEFAULTS
        default = SNAPSHOT_DEFAULTS.get(f"{cid}:{prop}")
    val = state.get(f"{cid}:{prop}", default)
    return default if val is None else val


def _build_bubble_figure_from_state(state: dict):
    """Build a bubble-tab Plotly figure from a snapshot state dict.

    Mirrors `update_bubble` (callbacks/charts/__init__.py) line-for-line
    in its param construction, but reads from the decoded snapshot dict
    instead of widget Inputs. Returns:
        - go.Figure on the standard path
        - None when CTA-active is set in the snapshot (caller should
          fall back to the existing callback path; CTA owns the figure
          and computing a standard figure here would cause a double-
          paint flicker when CTA's callback overwrites it ~300 ms later)

    The state dict is the dict produced by snapshot._decode_snapshot_*.
    Keys are formatted as "{component_id}:{property}". Special key
    "_lots" carries the snapshot's lot list (when shared with lots).
    """
    # CTA-active fallback. When the snapshot was generated with CTA on,
    # the user expects the custom-time-axis figure, NOT the standard
    # bubble figure. Building standard here and letting CTA overwrite
    # later produces a flicker. Fall back to existing callback path.
    if "yes" in (_v(state, "cta-active") or []):
        return None

    # ── Widget values (Input) ──
    sel_qs   = _v(state, "bub-qs")
    adv_qs   = _v(state, "bub-qs-adv")
    qs_mode  = _v(state, "bub-qs-mode")
    toggles  = _v(state, "bub-toggles")
    bubble_toggles = _v(state, "bub-bubble-toggles")
    xscale   = _v(state, "bub-xscale")
    yscale   = _v(state, "bub-yscale")
    xrange   = _v(state, "bub-xrange")
    yrange   = _v(state, "bub-yrange")
    n_future = _v(state, "bub-n-future")
    ptsize   = _v(state, "bub-ptsize")
    ptalpha  = _v(state, "bub-ptalpha")
    stack    = _v(state, "bub-stack")
    show_stack = _v(state, "bub-show-stack")
    legend_pos = _v(state, "bub-legend-pos")
    model_show = _v(state, "bub-model-show")
    sigma_mode = _v(state, "bub-sigma-mode")  # matches SNAPSHOT_DEFAULTS

    # ── Lots resolution (clientside cascade replicated) ──
    # Snapshot dict has _lots (raw list); existing callback reads
    # effective-lots which doesn't exist in the dict. We resolve here
    # exactly as the clientside cascade does.
    _lots = state.get("_lots") or []
    use_lots_val = _v(state, "bub-use-lots")
    use_lots = bool("yes" in (use_lots_val or []))

    # ── LPPL/HybPPL/EPPL config (State in update_bubble) ──
    lppl_n_freqs  = _v(state, "lppl-n-freqs")
    lppl_weighted = _v(state, "lppl-weighted")
    lppl_no_13    = _v(state, "lppl-no-13")

    hyb_a_nlog  = _v(state, "hybppl-cfg-a-nlog")
    hyb_a_ncal  = _v(state, "hybppl-cfg-a-ncal")
    hyb_a_log1d = _v(state, "hybppl-cfg-a-log1d")
    hyb_a_log2d = _v(state, "hybppl-cfg-a-log2d")
    hyb_a_cal1d = _v(state, "hybppl-cfg-a-cal1d")
    hyb_a_cal2d = _v(state, "hybppl-cfg-a-cal2d")

    hyb_b_enabled = _v(state, "hybppl-cfg-b-enabled")
    hyb_b_nlog  = _v(state, "hybppl-cfg-b-nlog")
    hyb_b_ncal  = _v(state, "hybppl-cfg-b-ncal")
    hyb_b_log1d = _v(state, "hybppl-cfg-b-log1d")
    hyb_b_log2d = _v(state, "hybppl-cfg-b-log2d")
    hyb_b_cal1d = _v(state, "hybppl-cfg-b-cal1d")
    hyb_b_cal2d = _v(state, "hybppl-cfg-b-cal2d")

    ep_a_nlog  = _v(state, "eppl-cfg-a-nlog")
    ep_a_ncal  = _v(state, "eppl-cfg-a-ncal")
    ep_a_log1d = _v(state, "eppl-cfg-a-log1d")
    ep_a_log2d = _v(state, "eppl-cfg-a-log2d")
    ep_a_cal1d = _v(state, "eppl-cfg-a-cal1d")
    ep_a_cal2d = _v(state, "eppl-cfg-a-cal2d")

    ep_b_enabled = _v(state, "eppl-cfg-b-enabled")
    ep_b_nlog  = _v(state, "eppl-cfg-b-nlog")
    ep_b_ncal  = _v(state, "eppl-cfg-b-ncal")
    ep_b_log1d = _v(state, "eppl-cfg-b-log1d")
    ep_b_log2d = _v(state, "eppl-cfg-b-log2d")
    ep_b_cal1d = _v(state, "eppl-cfg-b-cal1d")
    ep_b_cal2d = _v(state, "eppl-cfg-b-cal2d")

    # ── Component decomposition ──
    decomp_model      = _v(state, "bub-decomp-model")
    decomp_components = _v(state, "bub-decomp-components")
    decomp_mode       = _v(state, "bub-decomp-mode")

    # ── Other Stores ──
    palette_key = _v(state, "palette-store", "data")
    user_model_store = state.get("user-model-store:data")  # may be None

    # ── Scanner ──
    scan_q_val = _v(state, "scan-q")
    # scan-active-rows is not in _SNAPSHOT_CONTROLS (Store-only state).
    # Treat as no scanner lines on restore.
    scan_active = []

    # ── Master-key resolution (mirrors update_bubble) ──
    # Lazy import — see top-of-module comment.
    from callbacks.charts._resolvers import (
        _resolve_lppl_master, _resolve_hybppl_master, _resolve_eppl_master,
        _build_hybppl_config_key, _build_eppl_config_key,
    )
    from callbacks.coerce import _ci, _cf
    model_show = _resolve_lppl_master(
        model_show, lppl_n_freqs, lppl_weighted, lppl_no_13)
    model_show = _resolve_hybppl_master(
        model_show,
        hyb_a_nlog, hyb_a_ncal, hyb_a_log1d, hyb_a_log2d, hyb_a_cal1d, hyb_a_cal2d,
        hyb_b_enabled, hyb_b_nlog, hyb_b_ncal, hyb_b_log1d, hyb_b_log2d,
        hyb_b_cal1d, hyb_b_cal2d)
    model_show = _resolve_eppl_master(
        model_show,
        ep_a_nlog, ep_a_ncal, ep_a_log1d, ep_a_log2d, ep_a_cal1d, ep_a_cal2d,
        ep_b_enabled, ep_b_nlog, ep_b_ncal, ep_b_log1d, ep_b_log2d,
        ep_b_cal1d, ep_b_cal2d)

    # ── config_b_keys (mirrors update_bubble:213-225) ──
    _config_b_keys = set()
    if hyb_b_enabled and "yes" in (hyb_b_enabled or []):
        _hb = _build_hybppl_config_key(
            hyb_b_nlog or 0, hyb_b_ncal or 0,
            hyb_b_log1d, hyb_b_log2d, hyb_b_cal1d, hyb_b_cal2d)
        if _hb in model_show:
            _config_b_keys.add(_hb)
    if ep_b_enabled and "yes" in (ep_b_enabled or []):
        _eb = _build_eppl_config_key(
            ep_b_nlog or 0, ep_b_ncal or 0,
            ep_b_log1d, ep_b_log2d, ep_b_cal1d, ep_b_cal2d)
        if _eb in model_show:
            _config_b_keys.add(_eb)

    # ── Scanner lines (mirrors update_bubble:228-232) ──
    scanner_lines = []
    if scan_active and scan_q_val is not None:
        q_frac = float(scan_q_val) / 100.0
        for model_key in (scan_active or []):
            scanner_lines.append({"model": model_key, "q": q_frac})

    # ── Effective quantiles (mirrors update_bubble:234-238) ──
    if "advanced" in (qs_mode or []):
        _effective_qs = adv_qs or []
    else:
        _effective_qs = _bands_to_qs(sel_qs)

    # ── Defensive coercion of yrange/xrange to numerics ──
    yrange = yrange or [0, 7]
    xrange = xrange or [2012, 2030]

    toggles = toggles or []
    bubble_toggles = bubble_toggles or []

    # _get_bubble_fig returns (fig, mc_result); mc_result is None on the
    # non-MC fast path. Task 12 will add the MC gate here.
    fig, _mc_result = _get_bubble_fig(dict(
        selected_qs = _effective_qs,
        shade       = "shade"     in toggles,
        show_ols    = "show_ols"  in toggles,
        show_ucl    = "show_ucl"  in toggles,
        show_data   = "show_data"   in toggles,
        show_today  = "show_today"  in toggles,
        show_legend = "show_legend" in toggles,
        minor_grid  = "minor_grid" in toggles,
        show_comp   = "show_comp" in bubble_toggles,
        show_sup    = "show_sup"  in bubble_toggles,
        xscale      = xscale or BUBBLE["xscale"],
        yscale      = yscale or "log",
        xmin        = int(xrange[0]), xmax = int(xrange[1]),
        ymin        = 10 ** yrange[0], ymax = 10 ** yrange[1],
        n_future    = _ci(n_future, BUBBLE["n_future"]),
        pt_size     = _ci(ptsize, BUBBLE["pt_size"]),
        pt_alpha    = _cf(ptalpha, BUBBLE["pt_alpha"]),
        stack       = _cf(stack, BUBBLE["stack"]),
        show_stack  = bool(show_stack),
        use_lots    = use_lots,
        lots        = _lots,
        legend_pos  = legend_pos or "outside",
        comp_color  = LOT_MARKER_COLOR, comp_lw = TRACE_WIDTH_COMPOSITE,
        sup_color   = FALLBACK_MODEL_GRAY, sup_lw  = TRACE_WIDTH_SUPPORT,
        active_models = model_show or [],
        palette = palette_key or "default",
        scanner_lines = scanner_lines,
        user_model = user_model_store,
        qs_mode = qs_mode or [],
        decomp_model       = decomp_model or "",
        decomp_components  = list(decomp_components or []),
        decomp_mode        = decomp_mode or "individual",
        lppl_n_freqs       = list(lppl_n_freqs or []),
        lppl_weighted      = list(lppl_weighted or []),
        lppl_no_13         = list(lppl_no_13 or []),
        sigma_mode         = sigma_mode or "constant",
        config_b_keys      = sorted(_config_b_keys),
    ))
    if "chart_zoom" not in toggles:
        fig.update_layout(dragmode=False)
        fig.update_xaxes(fixedrange=True)
        fig.update_yaxes(fixedrange=True)
    return fig


def _build_dca_figure_from_state(state: dict):
    """Build a DCA-tab Plotly figure from a snapshot state dict.

    Mirrors `update_dca` (callbacks/charts/__init__.py:958) param construction
    line-for-line, but reads from the decoded snapshot dict instead of widget
    Inputs.

    Returns:
        - go.Figure on the standard fast path
        - None when MC is enabled (snapshot has dca-mc-enable=["yes"]) — caller
          falls back to the existing chart-callback path (Phase 2 doesn't
          replicate the MC pipeline; modal closes via 7 s timer for those).
        - None when Saylor-live mode is active (sc-enable + sc-entry-mode=
          "live") — builder has no live BTC price source. Caller falls back.

    Critical: the builder must NOT attempt to read btc-price-store from
    state. There is no such key in snapshots. sc_live_price is passed as 0
    (unused on non-live SC paths and on non-SC paths).
    """
    # ── Gates: return None for MC and Saylor-live, fall back to cascade ──
    mc_enable = _v(state, "dca-mc-enable") or []
    if "yes" in mc_enable:
        return None
    sc_enable = _v(state, "dca-sc-enable") or []
    sc_entry_mode = _v(state, "dca-sc-entry-mode")
    if "yes" in sc_enable and sc_entry_mode == "live":
        return None

    # ── DCA widget values (Inputs) ──
    stack    = _v(state, "dca-stack")
    use_lots = _v(state, "dca-use-lots")
    amount   = _v(state, "dca-amount")
    freq     = _v(state, "dca-freq")
    dca_infl = _v(state, "dca-infl")
    yr_range = _v(state, "dca-yr-range")
    disp     = _v(state, "dca-disp")
    toggles  = _v(state, "dca-toggles")
    legend_pos = _v(state, "dca-legend-pos")
    sel_qs   = _v(state, "dca-qs")
    adv_qs   = _v(state, "dca-qs-adv")
    qs_mode  = _v(state, "dca-qs-mode")  # State at line 953
    model_show = _v(state, "dca-model-show")

    # ── Saylor-mode (Stack-celerator) widget values ──
    sc_loan        = _v(state, "dca-sc-loan")
    sc_rate        = _v(state, "dca-sc-rate")
    sc_term        = _v(state, "dca-sc-term")
    sc_type        = _v(state, "dca-sc-type")
    sc_repeats     = _v(state, "dca-sc-repeats")
    sc_custom_price = _v(state, "dca-sc-custom-price")
    sc_tax         = _v(state, "dca-sc-tax")
    sc_rollover    = _v(state, "dca-sc-rollover")

    # ── Lots resolution (clientside cascade replicated, mirrors bubble) ──
    _lots = state.get("_lots") or []
    use_lots_bool = bool("yes" in (use_lots or []))

    # ── Shared model-config States (LPPL/HybPPL/EPPL) ──
    lppl_n_freqs  = _v(state, "lppl-n-freqs")
    lppl_weighted = _v(state, "lppl-weighted")
    lppl_no_13    = _v(state, "lppl-no-13")

    hyb_a_nlog  = _v(state, "hybppl-cfg-a-nlog")
    hyb_a_ncal  = _v(state, "hybppl-cfg-a-ncal")
    hyb_a_log1d = _v(state, "hybppl-cfg-a-log1d")
    hyb_a_log2d = _v(state, "hybppl-cfg-a-log2d")
    hyb_a_cal1d = _v(state, "hybppl-cfg-a-cal1d")
    hyb_a_cal2d = _v(state, "hybppl-cfg-a-cal2d")

    hyb_b_enabled = _v(state, "hybppl-cfg-b-enabled")
    hyb_b_nlog  = _v(state, "hybppl-cfg-b-nlog")
    hyb_b_ncal  = _v(state, "hybppl-cfg-b-ncal")
    hyb_b_log1d = _v(state, "hybppl-cfg-b-log1d")
    hyb_b_log2d = _v(state, "hybppl-cfg-b-log2d")
    hyb_b_cal1d = _v(state, "hybppl-cfg-b-cal1d")
    hyb_b_cal2d = _v(state, "hybppl-cfg-b-cal2d")

    ep_a_nlog  = _v(state, "eppl-cfg-a-nlog")
    ep_a_ncal  = _v(state, "eppl-cfg-a-ncal")
    ep_a_log1d = _v(state, "eppl-cfg-a-log1d")
    ep_a_log2d = _v(state, "eppl-cfg-a-log2d")
    ep_a_cal1d = _v(state, "eppl-cfg-a-cal1d")
    ep_a_cal2d = _v(state, "eppl-cfg-a-cal2d")

    ep_b_enabled = _v(state, "eppl-cfg-b-enabled")
    ep_b_nlog  = _v(state, "eppl-cfg-b-nlog")
    ep_b_ncal  = _v(state, "eppl-cfg-b-ncal")
    ep_b_log1d = _v(state, "eppl-cfg-b-log1d")
    ep_b_log2d = _v(state, "eppl-cfg-b-log2d")
    ep_b_cal1d = _v(state, "eppl-cfg-b-cal1d")
    ep_b_cal2d = _v(state, "eppl-cfg-b-cal2d")

    palette_key = _v(state, "palette-store", "data")
    user_model_store = state.get("user-model-store:data")

    # ── Master-key resolution (mirrors update_dca) ──
    from callbacks.charts._resolvers import (
        _resolve_lppl_master, _resolve_hybppl_master, _resolve_eppl_master,
    )
    from callbacks.coerce import _ci, _cf
    model_show = list(model_show or [])
    model_show = _resolve_lppl_master(
        model_show, lppl_n_freqs, lppl_weighted, lppl_no_13)
    model_show = _resolve_hybppl_master(
        model_show,
        hyb_a_nlog, hyb_a_ncal, hyb_a_log1d, hyb_a_log2d, hyb_a_cal1d, hyb_a_cal2d,
        hyb_b_enabled, hyb_b_nlog, hyb_b_ncal, hyb_b_log1d, hyb_b_log2d,
        hyb_b_cal1d, hyb_b_cal2d)
    model_show = _resolve_eppl_master(
        model_show,
        ep_a_nlog, ep_a_ncal, ep_a_log1d, ep_a_log2d, ep_a_cal1d, ep_a_cal2d,
        ep_b_enabled, ep_b_nlog, ep_b_ncal, ep_b_log1d, ep_b_log2d,
        ep_b_cal1d, ep_b_cal2d)

    # ── Effective quantiles (mirrors update_dca:983-984) ──
    yr_range = yr_range or [2024, 2034]
    toggles = toggles or []
    _advanced = "advanced" in (qs_mode or [])
    _effective_qs = (adv_qs or []) if _advanced else (
        _bands_to_qs(sel_qs) if sel_qs and isinstance(sel_qs[0], str) else (sel_qs or []))

    # ── Build figure via _get_dca_fig. mc_enabled=False is sufficient;
    # _get_mc_or_cached strips all mc_* keys before quantizing the cache key
    # (utils.py:166-168). No _mc_setup, no mc_p stub needed. ──
    from utils import _get_dca_fig
    from tab_defaults import DCA
    try:
        params = dict(
            start_stack    = _cf(stack, DCA["start_stack"]),
            use_lots       = use_lots_bool,
            amount         = _ci(amount, DCA["amount"]),
            freq           = freq or "Monthly",
            inflation      = _cf(dca_infl, DCA["inflation"]),
            start_yr       = int(yr_range[0]),
            end_yr         = int(yr_range[1]),
            disp_mode      = disp or "btc",
            log_y          = "log_y"      in toggles,
            annotate       = "annotate"   in toggles,
            discrete       = "discrete"   in toggles,
            shade          = "shade"      in toggles,
            show_today     = "show_today" in toggles,
            show_legend    = "show_legend" in toggles,
            legend_pos     = legend_pos or "outside",
            minor_grid     = "minor_grid" in toggles,
            selected_qs    = _effective_qs,
            lots           = _lots,
            sc_enabled     = bool("yes" in (sc_enable or [])),
            sc_loan_amount = _cf(sc_loan, 0),
            sc_rate        = _cf(sc_rate, DCA["sc_rate"]),
            sc_loan_type   = sc_type or "interest_only",
            sc_term_months = _cf(sc_term, DCA["sc_term_months"]),
            sc_repeats     = _ci(sc_repeats, 0),
            sc_live_price   = 0,  # builder has no live source — gated above for sc-live
            sc_entry_mode   = sc_entry_mode or "live",
            sc_custom_price = _cf(sc_custom_price, DCA["sc_custom_price"]),
            sc_tax_rate     = _cf(sc_tax, 33, lo=0, hi=100) / 100.0,
            sc_rollover     = bool(sc_rollover),
            show_qr        = "bub" in model_show,
            show_mc        = False,
            active_models  = [k for k in model_show if k != "mc"],
            palette        = palette_key or "default",
            user_model     = user_model_store,
            mc_enabled     = False,  # sufficient on its own — see utils.py:166-168
        )
        result = _get_dca_fig(params)
        fig = result[0] if isinstance(result, tuple) else result
        return fig
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(
            "_build_dca_figure_from_state failed: %s; caller will fall back", e)
        return None


def _build_retire_params(
    *,
    stack, use_lots, wd, freq, yr_range, infl, disp, toggles,
    legend_pos, sel_qs, adv_qs, qs_mode, model_show,
    lots, palette_key, user_model_store, mc_overrides,
):
    """Construct the retire `_get_retire_fig` params dict.

    Single source of truth for retire param construction, used by both:
    - `update_retire` callback (cascade path) — passes resolved widget
      Inputs and `mc_overrides=dict(show_mc=..., **mc_p)` from `_mc_setup`.
    - `_build_retire_figure_from_state` (fast restore path) — passes
      `_v(state, ...)` resolved values and `mc_overrides=dict(show_mc=False,
      mc_enabled=False)`. `_get_mc_or_cached` strips remaining mc_* keys
      when mc_enabled is False, so the fast path needs only those two.

    Pre-2026-04-25 these were two separate dicts in two files that drifted
    (e.g., callback's `disp or "btc"` vs builder's `disp or "usd"`). The
    extraction is Driver 1 of the architect-recommended complexity-
    reduction work — see memory/restore_callback_architecture.md.

    Args:
        stack, use_lots, wd, freq, yr_range, infl, disp, toggles,
        legend_pos, sel_qs, adv_qs, qs_mode, model_show: raw widget values.
            `model_show` MUST be pre-resolved by the caller (LPPL/HybPPL/
            EPPL master-key resolvers run upstream).
        lots: resolved lots list (callback: from `effective-lots` Store;
            builder: from `state["_lots"]`).
        palette_key: palette name (e.g., "default", "cb_brian").
        user_model_store: user-model dict or None.
        mc_overrides: dict merged into params last. Callback passes the
            full mc_p plus show_mc; builder passes the two-key fast-path
            override.

    Returns:
        Params dict ready for `_get_retire_fig(...)`.
    """
    from tab_defaults import RETIRE
    from callbacks.coerce import _ci, _cf
    import _app_ctx
    yr_range = yr_range or [RETIRE["start_yr"], RETIRE["end_yr"]]
    toggles  = toggles or []
    _advanced = "advanced" in (qs_mode or [])
    _effective_qs = (adv_qs or []) if _advanced else (
        _bands_to_qs(sel_qs) if sel_qs and isinstance(sel_qs[0], str)
        else (sel_qs or []))
    model_show = model_show or []
    params = dict(
        start_stack   = _cf(stack, RETIRE["start_stack"]),
        use_lots      = "yes" in (use_lots or []),
        wd_amount     = _ci(wd, RETIRE["wd_amount"], lo=0, hi=_app_ctx.MAX_USD),
        freq          = freq or "Monthly",
        start_yr      = int(yr_range[0]),
        end_yr        = int(yr_range[1]),
        inflation     = _cf(infl, RETIRE["inflation"]),
        disp_mode     = disp or RETIRE["disp_mode"],
        log_y         = "log_y"      in toggles,
        annotate      = "annotate"   in toggles,
        discrete      = "discrete"   in toggles,
        shade         = "shade"      in toggles,
        show_legend   = "show_legend" in toggles,
        legend_pos    = legend_pos or "outside",
        minor_grid    = "minor_grid" in toggles,
        selected_qs   = _effective_qs,
        lots          = lots or [],
        show_qr       = "bub" in model_show,
        active_models = [k for k in model_show if k != "mc"],
        palette       = palette_key or "default",
        user_model    = user_model_store,
    )
    params.update(mc_overrides)
    return params


def _build_retire_figure_from_state(state: dict):
    """Build a retire-tab Plotly figure from a snapshot state dict.

    Returns:
        - go.Figure on the standard fast path
        - None when MC is enabled (snapshot has ret-mc-enable=["yes"]) —
          caller falls back to the existing chart-callback path.
    """
    # ── Gate: return None for MC, fall back to cascade ──
    if "yes" in (_v(state, "ret-mc-enable") or []):
        return None

    # ── Master-key resolution ──
    from callbacks.charts._resolvers import (
        _resolve_lppl_master, _resolve_hybppl_master, _resolve_eppl_master,
    )
    model_show = list(_v(state, "ret-model-show") or [])
    model_show = _resolve_lppl_master(
        model_show,
        _v(state, "lppl-n-freqs"),
        _v(state, "lppl-weighted"),
        _v(state, "lppl-no-13"))
    model_show = _resolve_hybppl_master(
        model_show,
        _v(state, "hybppl-cfg-a-nlog"),  _v(state, "hybppl-cfg-a-ncal"),
        _v(state, "hybppl-cfg-a-log1d"), _v(state, "hybppl-cfg-a-log2d"),
        _v(state, "hybppl-cfg-a-cal1d"), _v(state, "hybppl-cfg-a-cal2d"),
        _v(state, "hybppl-cfg-b-enabled"),
        _v(state, "hybppl-cfg-b-nlog"),  _v(state, "hybppl-cfg-b-ncal"),
        _v(state, "hybppl-cfg-b-log1d"), _v(state, "hybppl-cfg-b-log2d"),
        _v(state, "hybppl-cfg-b-cal1d"), _v(state, "hybppl-cfg-b-cal2d"))
    model_show = _resolve_eppl_master(
        model_show,
        _v(state, "eppl-cfg-a-nlog"),  _v(state, "eppl-cfg-a-ncal"),
        _v(state, "eppl-cfg-a-log1d"), _v(state, "eppl-cfg-a-log2d"),
        _v(state, "eppl-cfg-a-cal1d"), _v(state, "eppl-cfg-a-cal2d"),
        _v(state, "eppl-cfg-b-enabled"),
        _v(state, "eppl-cfg-b-nlog"),  _v(state, "eppl-cfg-b-ncal"),
        _v(state, "eppl-cfg-b-log1d"), _v(state, "eppl-cfg-b-log2d"),
        _v(state, "eppl-cfg-b-cal1d"), _v(state, "eppl-cfg-b-cal2d"))

    # ── Build figure via the shared param-extractor ──
    from utils import _get_retire_fig
    try:
        params = _build_retire_params(
            stack       = _v(state, "ret-stack"),
            use_lots    = _v(state, "ret-use-lots"),
            wd          = _v(state, "ret-wd"),
            freq        = _v(state, "ret-freq"),
            yr_range    = _v(state, "ret-yr-range"),
            infl        = _v(state, "ret-infl"),
            disp        = _v(state, "ret-disp"),
            toggles     = _v(state, "ret-toggles"),
            legend_pos  = _v(state, "ret-legend-pos"),
            sel_qs      = _v(state, "ret-qs"),
            adv_qs      = _v(state, "ret-qs-adv"),
            qs_mode     = _v(state, "ret-qs-mode"),
            model_show  = model_show,
            lots        = state.get("_lots") or [],
            palette_key = _v(state, "palette-store", "data"),
            user_model_store = state.get("user-model-store:data"),
            mc_overrides = dict(show_mc=False, mc_enabled=False),
        )
        result = _get_retire_fig(params)
        return result[0] if isinstance(result, tuple) else result
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(
            "_build_retire_figure_from_state failed: %s; caller will fall back", e)
        return None


def _build_supercharge_figure_from_state(state: dict):
    """Build a supercharge-tab Plotly figure from a snapshot state dict.

    Mirrors `update_supercharge` (callbacks/charts/__init__.py:1364) param
    construction. Supercharge has 5 delays (sc-d0..sc-d4), sc-mode (a/b),
    sc-chart-layout (bands/shading toggle), sc-display-q, sc-target-yr.

    Returns:
        - go.Figure on the standard fast path
        - None when MC is enabled (sc-mc-enable=["yes"]) — fall back to cascade.
    """
    mc_enable = _v(state, "sc-mc-enable") or []
    if "yes" in mc_enable:
        return None

    # ── Supercharge widget values ──
    stack    = _v(state, "sc-stack")
    use_lots = _v(state, "sc-use-lots")
    start_yr = _v(state, "sc-start-yr")
    d0       = _v(state, "sc-d0")
    d1       = _v(state, "sc-d1")
    d2       = _v(state, "sc-d2")
    d3       = _v(state, "sc-d3")
    d4       = _v(state, "sc-d4")
    freq     = _v(state, "sc-freq")
    infl     = _v(state, "sc-infl")
    sel_qs   = _v(state, "sc-qs")
    adv_qs   = _v(state, "sc-qs-adv")
    qs_mode  = _v(state, "sc-qs-mode")
    mode     = _v(state, "sc-mode")
    wd       = _v(state, "sc-wd")
    end_yr   = _v(state, "sc-end-yr")
    target_yr = _v(state, "sc-target-yr")
    disp     = _v(state, "sc-disp")
    toggles  = _v(state, "sc-toggles")
    legend_pos = _v(state, "sc-legend-pos")
    chart_layout = _v(state, "sc-chart-layout")
    display_q = _v(state, "sc-display-q")
    model_show = _v(state, "sc-model-show")

    # ── Lots resolution ──
    _lots = state.get("_lots") or []
    use_lots_bool = bool("yes" in (use_lots or []))

    # ── Shared model-config States ──
    lppl_n_freqs  = _v(state, "lppl-n-freqs")
    lppl_weighted = _v(state, "lppl-weighted")
    lppl_no_13    = _v(state, "lppl-no-13")

    hyb_a_nlog  = _v(state, "hybppl-cfg-a-nlog")
    hyb_a_ncal  = _v(state, "hybppl-cfg-a-ncal")
    hyb_a_log1d = _v(state, "hybppl-cfg-a-log1d")
    hyb_a_log2d = _v(state, "hybppl-cfg-a-log2d")
    hyb_a_cal1d = _v(state, "hybppl-cfg-a-cal1d")
    hyb_a_cal2d = _v(state, "hybppl-cfg-a-cal2d")

    hyb_b_enabled = _v(state, "hybppl-cfg-b-enabled")
    hyb_b_nlog  = _v(state, "hybppl-cfg-b-nlog")
    hyb_b_ncal  = _v(state, "hybppl-cfg-b-ncal")
    hyb_b_log1d = _v(state, "hybppl-cfg-b-log1d")
    hyb_b_log2d = _v(state, "hybppl-cfg-b-log2d")
    hyb_b_cal1d = _v(state, "hybppl-cfg-b-cal1d")
    hyb_b_cal2d = _v(state, "hybppl-cfg-b-cal2d")

    ep_a_nlog  = _v(state, "eppl-cfg-a-nlog")
    ep_a_ncal  = _v(state, "eppl-cfg-a-ncal")
    ep_a_log1d = _v(state, "eppl-cfg-a-log1d")
    ep_a_log2d = _v(state, "eppl-cfg-a-log2d")
    ep_a_cal1d = _v(state, "eppl-cfg-a-cal1d")
    ep_a_cal2d = _v(state, "eppl-cfg-a-cal2d")

    ep_b_enabled = _v(state, "eppl-cfg-b-enabled")
    ep_b_nlog  = _v(state, "eppl-cfg-b-nlog")
    ep_b_ncal  = _v(state, "eppl-cfg-b-ncal")
    ep_b_log1d = _v(state, "eppl-cfg-b-log1d")
    ep_b_log2d = _v(state, "eppl-cfg-b-log2d")
    ep_b_cal1d = _v(state, "eppl-cfg-b-cal1d")
    ep_b_cal2d = _v(state, "eppl-cfg-b-cal2d")

    palette_key = _v(state, "palette-store", "data")
    user_model_store = state.get("user-model-store:data")

    # ── Master-key resolution ──
    from callbacks.charts._resolvers import (
        _resolve_lppl_master, _resolve_hybppl_master, _resolve_eppl_master,
    )
    from callbacks.coerce import _ci, _cf
    model_show = list(model_show or [])
    model_show = _resolve_lppl_master(
        model_show, lppl_n_freqs, lppl_weighted, lppl_no_13)
    model_show = _resolve_hybppl_master(
        model_show,
        hyb_a_nlog, hyb_a_ncal, hyb_a_log1d, hyb_a_log2d, hyb_a_cal1d, hyb_a_cal2d,
        hyb_b_enabled, hyb_b_nlog, hyb_b_ncal, hyb_b_log1d, hyb_b_log2d,
        hyb_b_cal1d, hyb_b_cal2d)
    model_show = _resolve_eppl_master(
        model_show,
        ep_a_nlog, ep_a_ncal, ep_a_log1d, ep_a_log2d, ep_a_cal1d, ep_a_cal2d,
        ep_b_enabled, ep_b_nlog, ep_b_ncal, ep_b_log1d, ep_b_log2d,
        ep_b_cal1d, ep_b_cal2d)

    # ── Effective quantiles + chart_layout normalization (mirrors update_supercharge) ──
    delays = [float(x) for x in [d0, d1, d2, d3, d4] if x is not None]
    toggles = toggles or []
    _advanced = "advanced" in (qs_mode or [])
    _effective_qs = (adv_qs or []) if _advanced else (
        _bands_to_qs(sel_qs) if sel_qs and isinstance(sel_qs[0], str) else (sel_qs or []))
    _cl = (2 if "shade" in (chart_layout or []) else 0) \
          if isinstance(chart_layout, list) \
          else (int(chart_layout) if chart_layout is not None else 2)

    # ── Build figure ──
    from utils import _get_supercharge_fig
    from tab_defaults import SUPERCHARGE
    try:
        params = dict(
            mode         = mode or "a",
            start_stack  = _cf(stack, SUPERCHARGE["start_stack"]),
            start_yr     = _ci(start_yr, SUPERCHARGE["start_yr"]),
            delays       = delays if delays else [0, 1, 2, 4, 8],
            freq         = freq or "Monthly",
            inflation    = _cf(infl, SUPERCHARGE["inflation"]),
            selected_qs  = _effective_qs,
            chart_layout = _cl,
            display_q    = _cf(display_q, SUPERCHARGE["display_q"]),
            wd_amount    = _ci(wd, SUPERCHARGE["wd_amount"]),
            end_yr       = _ci(end_yr, SUPERCHARGE["end_yr"]),
            disp_mode    = disp or "usd",
            log_y        = "log_y"      in toggles,
            annotate     = "annotate"   in toggles,
            discrete     = "discrete"   in toggles,
            show_legend  = "show_legend" in toggles,
            legend_pos   = legend_pos or "outside",
            minor_grid   = "minor_grid" in toggles,
            target_yr    = _ci(target_yr, SUPERCHARGE["target_yr"]),
            lots         = _lots,
            use_lots     = use_lots_bool,
            show_qr      = "bub" in model_show,
            show_mc      = False,
            active_models = [k for k in model_show if k != "mc"],
            palette      = palette_key or "default",
            is_mobile    = False,  # builder doesn't have viewport-width
            user_model   = user_model_store,
            mc_enabled   = False,
        )
        result = _get_supercharge_fig(params)
        fig = result[0] if isinstance(result, tuple) else result
        return fig
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(
            "_build_supercharge_figure_from_state failed: %s; caller will fall back", e)
        return None


def _build_leverage_figure_from_state(state: dict):
    """Build a leverage-tab Plotly figure from a snapshot state dict.

    Mirrors `update_leverage` (callbacks/leverage_cb.py:112) figure-build
    path. Leverage has no MC, no SC — just a calculator. 3 Outputs total
    (figure, readout, table), but this builder only computes the figure;
    readout + table come from the cascade after apply_tab_leverage writes
    the lev-* widget values.

    Returns:
        - go.Figure on the standard path
        - None on builder errors (caller falls back to existing path)
    """
    import datetime as _dt
    from figures.leverage import build_leverage_figure, floor_price, _parse_date

    date_val = _v(state, "lev-date", "date")
    price_val = _v(state, "lev-price")
    model = _v(state, "lev-model")
    q_val = _v(state, "lev-floor-q-store", "data")
    rb_val = _v(state, "lev-rb")
    rl_val = _v(state, "lev-rl")
    H_val = _v(state, "lev-horizon")
    c_val = _v(state, "lev-cagr")
    toggles = _v(state, "lev-toggles")

    try:
        price = float(price_val) if price_val is not None else 65000.0
        rb    = float(rb_val)    if rb_val    is not None else 13.0
        rl    = float(rl_val)    if rl_val    is not None else 4.5
        H_yr  = float(H_val)     if H_val     is not None else 4.0
        c_pct = float(c_val)     if c_val     is not None else 20.0
        model = str(model or "bub")
        q     = float(q_val) if q_val is not None else 0.01

        # Guards (spec §5.5)
        H_yr  = max(H_yr, 0.01)
        price = max(price, 1.0)

        buy_date = _parse_date(date_val) if date_val else _dt.date.today()

        # Validate floor_price availability before building figure
        # (matches update_leverage's defensive try/except at line 144).
        sell_date = buy_date + _dt.timedelta(days=int(round(H_yr * 365.25)))
        try:
            floor_price(model, q, sell_date)
        except (KeyError, AttributeError, ValueError):
            return None

        p = {
            "lev_price": price, "lev_date": buy_date,
            "lev_model": model, "lev_floor_q": q,
            "lev_rb": rb, "lev_rl": rl,
            "lev_horizon": H_yr, "lev_cagr": c_pct,
            "lev_toggles": tuple(toggles or ()),
            "palette": "default",
        }
        return build_leverage_figure(p)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(
            "_build_leverage_figure_from_state failed: %s; caller will fall back", e)
        return None


def _build_heatmap_figure_from_state(state: dict):
    """Build a heatmap-tab Plotly figure from a snapshot state dict.

    Mirrors `update_heatmap` (callbacks/charts/__init__.py:733). Heatmap is
    structurally different from DCA/retire/SC: active model comes from
    `hm-active-model` Store (set by pill bar), NOT from `hm-model-show`
    checklist.

    Returns:
        - go.Figure when MC is disabled AND hm_model != "mc"
        - None otherwise (caller falls back to existing path)
    """
    mc_enable = _v(state, "hm-mc-enable") or []
    if "yes" in mc_enable:
        return None

    # Heatmap-active-model is a Store (data prop), not a checklist.
    hm_model = _v(state, "hm-active-model", "data")
    if hm_model == "mc":
        # Active model is MC — fall back to cascade (which routes to MC heatmap).
        return None

    # Resolve legacy / master model keys
    from callbacks.routing import _HM_PILL_MODELS, _HM_LEGACY_MODEL_FALLBACK
    if hm_model not in _HM_PILL_MODELS and hm_model not in ("lppl", "hybppl", "eppl"):
        hm_model = _HM_LEGACY_MODEL_FALLBACK.get(hm_model, hm_model)

    # ── Heatmap controls ──
    entry_yr = _v(state, "hm-entry-yr")
    entry_q  = _v(state, "hm-entry-q")
    exit_range = _v(state, "hm-exit-range")
    exit_qs  = _v(state, "hm-exit-qs")
    mode     = _v(state, "hm-mode")
    b1       = _v(state, "hm-b1")
    b2       = _v(state, "hm-b2")
    c_lo     = _v(state, "hm-c-lo")
    c_mid1   = _v(state, "hm-c-mid1")
    c_mid2   = _v(state, "hm-c-mid2")
    c_hi     = _v(state, "hm-c-hi")
    grad     = _v(state, "hm-grad")
    vfmt     = _v(state, "hm-vfmt")
    cell_fs  = _v(state, "hm-cell-fs")
    line_w   = _v(state, "hm-line-width")
    line_c   = _v(state, "hm-line-color")
    toggles  = _v(state, "hm-toggles")
    stack    = _v(state, "hm-stack")
    use_lots = _v(state, "hm-use-lots")
    model_show = _v(state, "hm-model-show")

    # ── Shared model-config States (for LPPL/HybPPL/EPPL master resolution) ──
    lppl_n_freqs  = _v(state, "lppl-n-freqs")
    lppl_weighted = _v(state, "lppl-weighted")
    lppl_no_13    = _v(state, "lppl-no-13")

    hyb_a_nlog  = _v(state, "hybppl-cfg-a-nlog")
    hyb_a_ncal  = _v(state, "hybppl-cfg-a-ncal")
    hyb_a_log1d = _v(state, "hybppl-cfg-a-log1d")
    hyb_a_log2d = _v(state, "hybppl-cfg-a-log2d")
    hyb_a_cal1d = _v(state, "hybppl-cfg-a-cal1d")
    hyb_a_cal2d = _v(state, "hybppl-cfg-a-cal2d")

    ep_a_nlog  = _v(state, "eppl-cfg-a-nlog")
    ep_a_ncal  = _v(state, "eppl-cfg-a-ncal")
    ep_a_log1d = _v(state, "eppl-cfg-a-log1d")
    ep_a_log2d = _v(state, "eppl-cfg-a-log2d")
    ep_a_cal1d = _v(state, "eppl-cfg-a-cal1d")
    ep_a_cal2d = _v(state, "eppl-cfg-a-cal2d")

    palette_key = _v(state, "palette-store", "data")

    # ── Lots ──
    _lots = state.get("_lots") or []
    use_lots_bool = bool("yes" in (use_lots or []))

    # ── Master-key resolution (heatmap has its own resolvers) ──
    from callbacks.charts._resolvers import (
        _resolve_hm_lppl_master, _resolve_hm_hybppl_master, _resolve_hm_eppl_master,
    )
    from callbacks.coerce import _ci, _cf
    hm_model = _resolve_hm_lppl_master(
        hm_model, lppl_n_freqs, lppl_weighted, lppl_no_13)
    hm_model = _resolve_hm_hybppl_master(
        hm_model,
        hyb_a_nlog, hyb_a_ncal, hyb_a_log1d, hyb_a_log2d,
        hyb_a_cal1d, hyb_a_cal2d)
    hm_model = _resolve_hm_eppl_master(
        hm_model,
        ep_a_nlog, ep_a_ncal, ep_a_log1d, ep_a_log2d,
        ep_a_cal1d, ep_a_cal2d)

    # ── Build figure ──
    import pandas as _pd
    yr_now = _pd.Timestamp.today().year
    if exit_range is None:
        exit_range = [(entry_yr or yr_now), (entry_yr or yr_now) + 10]

    from utils import _get_heatmap_fig
    from tab_defaults import HEATMAP
    import _app_ctx as _ctx
    try:
        shared_params = dict(
            entry_yr     = _ci(entry_yr, yr_now),
            entry_q      = _cf(entry_q, 50),
            live_price   = None,  # no live ticker source in builder
            exit_yr_lo   = int(exit_range[0]),
            exit_yr_hi   = int(exit_range[1]),
            exit_qs      = exit_qs or [],
            color_mode   = _ci(mode, HEATMAP["color_mode"]),
            b1           = _cf(b1, _ctx.M.CAGR_SEG_B1),
            b2           = _cf(b2, _ctx.M.CAGR_SEG_B2),
            c_lo         = c_lo   or _ctx.M.CAGR_SEG_C_LO,
            c_mid1       = c_mid1 or _ctx.M.CAGR_SEG_C_MID1,
            c_mid2       = c_mid2 or _ctx.M.CAGR_SEG_C_MID2,
            c_hi         = c_hi   or _ctx.M.CAGR_SEG_C_HI,
            n_disc       = _ci(grad, HEATMAP["n_disc"]),
            vfmt         = vfmt or HEATMAP["vfmt"],
            cell_font_size = _ci(cell_fs, HEATMAP["cell_font_size"]),
            line_width   = _cf(line_w, HEATMAP["line_width"]),
            line_color   = line_c or HEATMAP["line_color"],
            show_colorbar = "colorbar" in (toggles or []),
            stack        = _cf(stack, HEATMAP["stack"]),
            use_lots     = use_lots_bool,
            lots         = _lots,
            active_models = [k for k in (model_show or []) if k not in _ctx.MODEL_SENTINELS],
            palette      = palette_key or "default",
            hm_model     = hm_model,
        )
        fig = _get_heatmap_fig(shared_params)
        if "chart_zoom" not in (toggles or []):
            fig.update_layout(dragmode=False)
        return fig
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(
            "_build_heatmap_figure_from_state failed: %s; caller will fall back", e)
        return None
