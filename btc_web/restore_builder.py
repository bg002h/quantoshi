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
