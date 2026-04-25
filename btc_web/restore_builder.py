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


def _v(state: dict, cid: str, prop: str = "value", default=None):
    """Fetch a control's value from a snapshot state dict, returning
    `default` when the key is absent (older link format) or when the
    stored value is None."""
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
    if "yes" in (_v(state, "cta-active", default=[]) or []):
        return None

    # ── Widget values (Input) ──
    sel_qs   = _v(state, "bub-qs",          default=["median"])
    adv_qs   = _v(state, "bub-qs-adv",      default=[])
    qs_mode  = _v(state, "bub-qs-mode",     default=[])
    toggles  = _v(state, "bub-toggles",     default=["shade", "show_data", "show_today"])
    bubble_toggles = _v(state, "bub-bubble-toggles", default=["show_comp", "show_sup"])
    xscale   = _v(state, "bub-xscale",      default="log")
    yscale   = _v(state, "bub-yscale",      default="log")
    xrange   = _v(state, "bub-xrange",      default=[2010, 2033])
    yrange   = _v(state, "bub-yrange",      default=[-1.5, 6.05])
    n_future = _v(state, "bub-n-future",    default=BUBBLE["n_future"])
    ptsize   = _v(state, "bub-ptsize",      default=BUBBLE["pt_size"])
    ptalpha  = _v(state, "bub-ptalpha",     default=BUBBLE["pt_alpha"])
    stack    = _v(state, "bub-stack",       default=0)
    show_stack = _v(state, "bub-show-stack", default=[])
    legend_pos = _v(state, "bub-legend-pos", default="outside")
    model_show = _v(state, "bub-model-show", default=[])
    sigma_mode = _v(state, "bub-sigma-mode", default="resqr")  # matches SNAPSHOT_DEFAULTS

    # ── Lots resolution (clientside cascade replicated) ──
    # Snapshot dict has _lots (raw list); existing callback reads
    # effective-lots which doesn't exist in the dict. We resolve here
    # exactly as the clientside cascade does.
    _lots = state.get("_lots") or []
    use_lots_val = _v(state, "bub-use-lots", default=[])
    use_lots = bool("yes" in (use_lots_val or []))

    # ── LPPL/HybPPL/EPPL config (State in update_bubble) ──
    lppl_n_freqs  = _v(state, "lppl-n-freqs",  default=[])
    lppl_weighted = _v(state, "lppl-weighted", default=[])
    lppl_no_13    = _v(state, "lppl-no-13",    default=[])

    hyb_a_nlog  = _v(state, "hybppl-cfg-a-nlog",  default=1)
    hyb_a_ncal  = _v(state, "hybppl-cfg-a-ncal",  default=1)
    hyb_a_log1d = _v(state, "hybppl-cfg-a-log1d", default="d")
    hyb_a_log2d = _v(state, "hybppl-cfg-a-log2d", default="d")
    hyb_a_cal1d = _v(state, "hybppl-cfg-a-cal1d", default="u")
    hyb_a_cal2d = _v(state, "hybppl-cfg-a-cal2d", default="u")

    hyb_b_enabled = _v(state, "hybppl-cfg-b-enabled", default=[])
    hyb_b_nlog  = _v(state, "hybppl-cfg-b-nlog",  default=0)
    hyb_b_ncal  = _v(state, "hybppl-cfg-b-ncal",  default=0)
    hyb_b_log1d = _v(state, "hybppl-cfg-b-log1d", default="d")
    hyb_b_log2d = _v(state, "hybppl-cfg-b-log2d", default="d")
    hyb_b_cal1d = _v(state, "hybppl-cfg-b-cal1d", default="u")
    hyb_b_cal2d = _v(state, "hybppl-cfg-b-cal2d", default="u")

    ep_a_nlog  = _v(state, "eppl-cfg-a-nlog",  default=1)
    ep_a_ncal  = _v(state, "eppl-cfg-a-ncal",  default=1)
    ep_a_log1d = _v(state, "eppl-cfg-a-log1d", default="d")
    ep_a_log2d = _v(state, "eppl-cfg-a-log2d", default="d")
    ep_a_cal1d = _v(state, "eppl-cfg-a-cal1d", default="u")
    ep_a_cal2d = _v(state, "eppl-cfg-a-cal2d", default="u")

    ep_b_enabled = _v(state, "eppl-cfg-b-enabled", default=[])
    ep_b_nlog  = _v(state, "eppl-cfg-b-nlog",  default=0)
    ep_b_ncal  = _v(state, "eppl-cfg-b-ncal",  default=0)
    ep_b_log1d = _v(state, "eppl-cfg-b-log1d", default="d")
    ep_b_log2d = _v(state, "eppl-cfg-b-log2d", default="d")
    ep_b_cal1d = _v(state, "eppl-cfg-b-cal1d", default="u")
    ep_b_cal2d = _v(state, "eppl-cfg-b-cal2d", default="u")

    # ── Component decomposition ──
    decomp_model      = _v(state, "bub-decomp-model",      default="")
    decomp_components = _v(state, "bub-decomp-components", default=[])
    decomp_mode       = _v(state, "bub-decomp-mode",       default="individual")

    # ── Other Stores ──
    palette_key = _v(state, "palette-store", "data", default="default")
    user_model_store = state.get("user-model-store:data")  # may be None

    # ── Scanner ──
    scan_q_val = _v(state, "scan-q", default=None)
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

    fig = _get_bubble_fig(dict(
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
    mc_enable = _v(state, "dca-mc-enable", default=[]) or []
    if "yes" in mc_enable:
        return None
    sc_enable = _v(state, "dca-sc-enable", default=[]) or []
    sc_entry_mode = _v(state, "dca-sc-entry-mode", default="live")
    if "yes" in sc_enable and sc_entry_mode == "live":
        return None

    # ── DCA widget values (Inputs) ──
    stack    = _v(state, "dca-stack",    default=0.0)
    use_lots = _v(state, "dca-use-lots", default=[])
    amount   = _v(state, "dca-amount",   default=100)
    freq     = _v(state, "dca-freq",     default="Monthly")
    dca_infl = _v(state, "dca-infl",     default=0.0)
    yr_range = _v(state, "dca-yr-range", default=[2024, 2034])
    disp     = _v(state, "dca-disp",     default="btc")
    toggles  = _v(state, "dca-toggles",  default=[])
    legend_pos = _v(state, "dca-legend-pos", default="outside")
    sel_qs   = _v(state, "dca-qs",       default=[0.5])
    adv_qs   = _v(state, "dca-qs-adv",   default=[])
    qs_mode  = _v(state, "dca-qs-mode",  default=[])  # State at line 953
    model_show = _v(state, "dca-model-show", default=[])

    # ── Saylor-mode (Stack-celerator) widget values ──
    sc_loan        = _v(state, "dca-sc-loan",        default=0.0)
    sc_rate        = _v(state, "dca-sc-rate",        default=8.0)
    sc_term        = _v(state, "dca-sc-term",        default=120)
    sc_type        = _v(state, "dca-sc-type",        default="interest_only")
    sc_repeats     = _v(state, "dca-sc-repeats",     default=0)
    sc_custom_price = _v(state, "dca-sc-custom-price", default=70000.0)
    sc_tax         = _v(state, "dca-sc-tax",         default=33.0)
    sc_rollover    = _v(state, "dca-sc-rollover",    default=False)

    # ── Lots resolution (clientside cascade replicated, mirrors bubble) ──
    _lots = state.get("_lots") or []
    use_lots_bool = bool("yes" in (use_lots or []))

    # ── Shared model-config States (LPPL/HybPPL/EPPL) ──
    lppl_n_freqs  = _v(state, "lppl-n-freqs",  default=[])
    lppl_weighted = _v(state, "lppl-weighted", default=[])
    lppl_no_13    = _v(state, "lppl-no-13",    default=[])

    hyb_a_nlog  = _v(state, "hybppl-cfg-a-nlog",  default=1)
    hyb_a_ncal  = _v(state, "hybppl-cfg-a-ncal",  default=1)
    hyb_a_log1d = _v(state, "hybppl-cfg-a-log1d", default="d")
    hyb_a_log2d = _v(state, "hybppl-cfg-a-log2d", default="d")
    hyb_a_cal1d = _v(state, "hybppl-cfg-a-cal1d", default="u")
    hyb_a_cal2d = _v(state, "hybppl-cfg-a-cal2d", default="u")

    hyb_b_enabled = _v(state, "hybppl-cfg-b-enabled", default=[])
    hyb_b_nlog  = _v(state, "hybppl-cfg-b-nlog",  default=0)
    hyb_b_ncal  = _v(state, "hybppl-cfg-b-ncal",  default=0)
    hyb_b_log1d = _v(state, "hybppl-cfg-b-log1d", default="d")
    hyb_b_log2d = _v(state, "hybppl-cfg-b-log2d", default="d")
    hyb_b_cal1d = _v(state, "hybppl-cfg-b-cal1d", default="u")
    hyb_b_cal2d = _v(state, "hybppl-cfg-b-cal2d", default="u")

    ep_a_nlog  = _v(state, "eppl-cfg-a-nlog",  default=1)
    ep_a_ncal  = _v(state, "eppl-cfg-a-ncal",  default=1)
    ep_a_log1d = _v(state, "eppl-cfg-a-log1d", default="d")
    ep_a_log2d = _v(state, "eppl-cfg-a-log2d", default="d")
    ep_a_cal1d = _v(state, "eppl-cfg-a-cal1d", default="u")
    ep_a_cal2d = _v(state, "eppl-cfg-a-cal2d", default="u")

    ep_b_enabled = _v(state, "eppl-cfg-b-enabled", default=[])
    ep_b_nlog  = _v(state, "eppl-cfg-b-nlog",  default=0)
    ep_b_ncal  = _v(state, "eppl-cfg-b-ncal",  default=0)
    ep_b_log1d = _v(state, "eppl-cfg-b-log1d", default="d")
    ep_b_log2d = _v(state, "eppl-cfg-b-log2d", default="d")
    ep_b_cal1d = _v(state, "eppl-cfg-b-cal1d", default="u")
    ep_b_cal2d = _v(state, "eppl-cfg-b-cal2d", default="u")

    palette_key = _v(state, "palette-store", "data", default="default")
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


def _build_retire_figure_from_state(state: dict):
    """Build a retire-tab Plotly figure from a snapshot state dict.

    Mirrors `update_retire` (callbacks/charts/__init__.py:1164) param
    construction. Retire has NO Saylor mode (Stack-celerator) — only MC.

    Returns:
        - go.Figure on the standard fast path
        - None when MC is enabled (snapshot has ret-mc-enable=["yes"]) —
          caller falls back to the existing chart-callback path.
    """
    # ── Gate: return None for MC, fall back to cascade ──
    mc_enable = _v(state, "ret-mc-enable", default=[]) or []
    if "yes" in mc_enable:
        return None

    # ── Retire widget values (Inputs) ──
    stack    = _v(state, "ret-stack",    default=0.0)
    use_lots = _v(state, "ret-use-lots", default=[])
    wd       = _v(state, "ret-wd",       default=5000)
    freq     = _v(state, "ret-freq",     default="Monthly")
    yr_range = _v(state, "ret-yr-range", default=[2031, 2075])
    infl     = _v(state, "ret-infl",     default=4.0)
    disp     = _v(state, "ret-disp",     default="usd")
    toggles  = _v(state, "ret-toggles",  default=[])
    legend_pos = _v(state, "ret-legend-pos", default="outside")
    sel_qs   = _v(state, "ret-qs",       default=["inner"])  # match snapshot_defaults.py:292
    adv_qs   = _v(state, "ret-qs-adv",   default=[])
    qs_mode  = _v(state, "ret-qs-mode",  default=[])
    model_show = _v(state, "ret-model-show", default=[])

    # ── Lots resolution (clientside cascade replicated, mirrors bubble) ──
    _lots = state.get("_lots") or []
    use_lots_bool = bool("yes" in (use_lots or []))

    # ── Shared model-config States (LPPL/HybPPL/EPPL) ──
    lppl_n_freqs  = _v(state, "lppl-n-freqs",  default=[])
    lppl_weighted = _v(state, "lppl-weighted", default=[])
    lppl_no_13    = _v(state, "lppl-no-13",    default=[])

    hyb_a_nlog  = _v(state, "hybppl-cfg-a-nlog",  default=1)
    hyb_a_ncal  = _v(state, "hybppl-cfg-a-ncal",  default=1)
    hyb_a_log1d = _v(state, "hybppl-cfg-a-log1d", default="d")
    hyb_a_log2d = _v(state, "hybppl-cfg-a-log2d", default="d")
    hyb_a_cal1d = _v(state, "hybppl-cfg-a-cal1d", default="u")
    hyb_a_cal2d = _v(state, "hybppl-cfg-a-cal2d", default="u")

    hyb_b_enabled = _v(state, "hybppl-cfg-b-enabled", default=[])
    hyb_b_nlog  = _v(state, "hybppl-cfg-b-nlog",  default=0)
    hyb_b_ncal  = _v(state, "hybppl-cfg-b-ncal",  default=0)
    hyb_b_log1d = _v(state, "hybppl-cfg-b-log1d", default="d")
    hyb_b_log2d = _v(state, "hybppl-cfg-b-log2d", default="d")
    hyb_b_cal1d = _v(state, "hybppl-cfg-b-cal1d", default="u")
    hyb_b_cal2d = _v(state, "hybppl-cfg-b-cal2d", default="u")

    ep_a_nlog  = _v(state, "eppl-cfg-a-nlog",  default=1)
    ep_a_ncal  = _v(state, "eppl-cfg-a-ncal",  default=1)
    ep_a_log1d = _v(state, "eppl-cfg-a-log1d", default="d")
    ep_a_log2d = _v(state, "eppl-cfg-a-log2d", default="d")
    ep_a_cal1d = _v(state, "eppl-cfg-a-cal1d", default="u")
    ep_a_cal2d = _v(state, "eppl-cfg-a-cal2d", default="u")

    ep_b_enabled = _v(state, "eppl-cfg-b-enabled", default=[])
    ep_b_nlog  = _v(state, "eppl-cfg-b-nlog",  default=0)
    ep_b_ncal  = _v(state, "eppl-cfg-b-ncal",  default=0)
    ep_b_log1d = _v(state, "eppl-cfg-b-log1d", default="d")
    ep_b_log2d = _v(state, "eppl-cfg-b-log2d", default="d")
    ep_b_cal1d = _v(state, "eppl-cfg-b-cal1d", default="u")
    ep_b_cal2d = _v(state, "eppl-cfg-b-cal2d", default="u")

    palette_key = _v(state, "palette-store", "data", default="default")
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

    # ── Effective quantiles ──
    yr_range = yr_range or [2031, 2075]
    toggles = toggles or []
    _advanced = "advanced" in (qs_mode or [])
    _effective_qs = (adv_qs or []) if _advanced else (
        _bands_to_qs(sel_qs) if sel_qs and isinstance(sel_qs[0], str) else (sel_qs or []))

    # ── Build figure via _get_retire_fig with mc_enabled=False (sufficient;
    # _get_mc_or_cached strips all mc_* keys before quantizing the cache key). ──
    from utils import _get_retire_fig
    from tab_defaults import RETIRE
    try:
        params = dict(
            start_stack    = _cf(stack, RETIRE["start_stack"]),
            use_lots       = use_lots_bool,
            wd_amount      = _ci(wd, RETIRE["wd_amount"]),
            freq           = freq or "Monthly",
            start_yr       = int(yr_range[0]),
            end_yr         = int(yr_range[1]),
            inflation      = _cf(infl, RETIRE["inflation"]),
            disp_mode      = disp or "usd",
            log_y          = "log_y"      in toggles,
            annotate       = "annotate"   in toggles,
            discrete       = "discrete"   in toggles,
            shade          = "shade"      in toggles,
            show_legend    = "show_legend" in toggles,
            legend_pos     = legend_pos or "outside",
            minor_grid     = "minor_grid" in toggles,
            selected_qs    = _effective_qs,
            lots           = _lots,
            show_qr        = "bub" in model_show,
            show_mc        = False,
            active_models  = [k for k in model_show if k != "mc"],
            palette        = palette_key or "default",
            user_model     = user_model_store,
            mc_enabled     = False,
        )
        result = _get_retire_fig(params)
        fig = result[0] if isinstance(result, tuple) else result
        return fig
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(
            "_build_retire_figure_from_state failed: %s; caller will fall back", e)
        return None
