"""Snapshot / URL state encoding and decoding for share links."""

import json
import gzip
import base64
import logging

import _app_ctx

log = logging.getLogger(__name__)

_SNAPSHOT_CONTROLS = [
    # ── Bubble tab (indices 0–12) ──
    ("bub-qs",            "value"),   # selected quantile lines
    ("bub-xscale",        "value"),   # x-axis scale (Log/Linear)
    ("bub-yscale",        "value"),   # y-axis scale (Log/Linear)
    ("bub-xrange",        "value"),   # x-axis year range [start, end]
    ("bub-yrange",        "value"),   # y-axis price range [lo, hi]
    ("bub-toggles",       "value"),   # shade/data/today/legend toggles
    ("bub-bubble-toggles","value"),   # bubble composite overlay toggles
    ("bub-n-future",      "value"),   # number of projected future bubbles
    ("bub-ptsize",        "value"),   # scatter point size (1–20)
    ("bub-ptalpha",       "value"),   # scatter point opacity (0–1)
    ("bub-stack",         "value"),   # starting BTC stack
    ("bub-show-stack",    "value"),   # show stack value in legend
    ("bub-use-lots",      "value"),   # use Stack Tracker lots for starting BTC
    # ── Heatmap tab (indices 13–28) ──
    ("hm-entry-yr",       "value"),   # heatmap entry year
    ("hm-entry-q",        "value"),   # entry percentile (0.1–99.9%)
    ("hm-exit-range",     "value"),   # exit year range [start, end]
    ("hm-exit-qs",        "value"),   # exit quantile lines
    ("hm-mode",           "value"),   # colorscale mode (Segmented/DataScaled/Diverging)
    ("hm-b1",             "value"),   # segmented colorscale breakpoint 1 (CAGR %)
    ("hm-b2",             "value"),   # segmented colorscale breakpoint 2 (CAGR %)
    ("hm-c-lo",           "value"),   # color below breakpoint 1
    ("hm-c-mid1",         "value"),   # color at breakpoint 1
    ("hm-c-mid2",         "value"),   # color at breakpoint 2
    ("hm-c-hi",           "value"),   # color above breakpoint 2
    ("hm-grad",           "value"),   # gradient steps (cosmetic)
    ("hm-vfmt",           "value"),   # cell text format (cagr/price/both/stack/...)
    ("hm-cell-fs",        "value"),   # cell text font size
    ("hm-toggles",        "value"),   # heatmap display toggles
    ("hm-stack",          "value"),   # starting BTC stack
    ("hm-use-lots",       "value"),   # use Stack Tracker lots
    # ── DCA tab + Stack-celerator (indices 29–52) ──
    ("dca-stack",         "value"),   # starting BTC stack
    ("dca-use-lots",      "value"),   # use Stack Tracker lots
    ("dca-amount",        "value"),   # DCA amount per period ($)
    ("dca-freq",          "value"),   # DCA frequency (Daily/Weekly/Monthly/...)
    ("dca-freq-unlock",   "value"),   # custom frequency unlock
    ("dca-infl",          "value"),   # inflation rate (%)
    ("dca-yr-range",      "value"),   # simulation year range [start, end]
    ("dca-disp",          "value"),   # display mode (BTC/USD)
    ("dca-toggles",       "value"),   # log_y/dual_y/annotate/legend toggles
    ("dca-qs",            "value"),   # selected quantile lines
    ("dca-sc-enable",     "value"),   # Stack-celerator enable
    ("dca-sc-loan",       "value"),   # SC loan principal ($)
    ("dca-sc-rate",       "value"),   # SC annual interest rate (%)
    ("dca-sc-term",       "value"),   # SC loan term (months)
    ("dca-sc-type",       "value"),   # SC loan type (amortizing/interest_only)
    ("dca-sc-repeats",    "value"),   # SC loan repeat cycles (0=one-shot)
    ("dca-sc-entry-mode", "value"),   # SC entry price mode (live/model/custom)
    ("dca-sc-custom-price","value"),  # SC custom entry price ($)
    ("dca-sc-tax",        "value"),   # SC capital gains tax rate (%)
    ("dca-sc-rollover",   "value"),   # SC rollover (interest-only: defer repayment)
    # ── Retire tab (indices 53–63) ──
    ("ret-stack",         "value"),   # starting BTC stack
    ("ret-use-lots",      "value"),   # use Stack Tracker lots
    ("ret-wd",            "value"),   # withdrawal amount per period ($)
    ("ret-freq",          "value"),   # withdrawal frequency
    ("ret-freq-unlock",   "value"),   # custom frequency unlock
    ("ret-yr-range",      "value"),   # simulation year range [start, end]
    ("ret-infl",          "value"),   # inflation rate (%)
    ("ret-disp",          "value"),   # display mode (BTC/USD)
    ("ret-toggles",       "value"),   # log_y/dual_y/annotate/legend toggles
    ("ret-legend-pos",    "value"),   # legend position
    ("ret-qs",            "value"),   # selected quantile lines
    # ── Supercharger tab (indices 64–83) ──
    ("sc-stack",          "value"),   # starting BTC stack
    ("sc-use-lots",       "value"),   # use Stack Tracker lots
    ("sc-start-yr",       "value"),   # withdrawal start year
    ("sc-d0",             "value"),   # delay offset 0 (years before withdrawal)
    ("sc-d1",             "value"),   # delay offset 1
    ("sc-d2",             "value"),   # delay offset 2
    ("sc-d3",             "value"),   # delay offset 3
    ("sc-d4",             "value"),   # delay offset 4
    ("sc-freq",           "value"),   # withdrawal frequency
    ("sc-freq-unlock",    "value"),   # custom frequency unlock
    ("sc-infl",           "value"),   # inflation rate (%)
    ("sc-qs",             "value"),   # selected quantile lines
    ("sc-mode",           "value"),   # Mode A (fixed spending) / Mode B (fixed depletion)
    ("sc-wd",             "value"),   # Mode A: withdrawal amount per period ($)
    ("sc-end-yr",         "value"),   # Mode A: simulation end year
    ("sc-target-yr",      "value"),   # Mode B: target depletion year
    ("sc-disp",           "value"),   # display mode (BTC/USD)
    ("sc-toggles",        "value"),   # log_y/annotate/legend toggles
    ("sc-chart-layout",   "value"),   # chart layout (line/bands)
    ("sc-display-q",      "value"),   # single quantile display (line layout only)
    # ── Cross-tab settings (indices 84–92) ──
    ("bub-auto-y",        "value"),   # auto-fit Y axis to selected quantiles
    ("bub-legend-pos",    "value"),   # bubble legend position
    ("dca-legend-pos",    "value"),   # DCA legend position
    ("sc-legend-pos",     "value"),   # SC legend position
    ("main-tabs",         "active_tab"),  # active tab selection
    # ── Model display toggles (indices 89–92) ──
    ("dca-model-show",    "value"),   # QR/MC display toggle (DCA)
    ("ret-model-show",    "value"),   # QR/MC display toggle (Retire)
    ("sc-model-show",     "value"),   # QR/MC display toggle (SC)
    ("hm-model-show",     "value"),   # QR/MC display toggle (Heatmap)
    # ── MC model source (indices 93–96) ──
    ("dca-mc-model-src",  "value"),   # MC model source (DCA)
    ("ret-mc-model-src",  "value"),   # MC model source (Retire)
    ("sc-mc-model-src",   "value"),   # MC model source (SC)
    ("hm-mc-model-src",   "value"),   # MC model source (Heatmap)
    # ── Bubble overlay models (index 97) ──
    ("bub-model-show",    "value"),   # PL/S2F overlay toggle (Bubble)
]

_SNAP_PREFIX    = "q3:"   # current format (v3: shared settings consolidation)
_SNAP_PREFIX_V2 = "q2:"   # prior format (positional array, different control list)
_SNAP_PREFIX_V1 = "q1:"   # legacy format (dict-based)

# Why bitmask encoding: storing 17 quantile checkboxes as a list in JSON costs
# ~150 chars; a single bitmask integer costs ~5 chars. Across 20 checklist fields,
# this saves ~660 characters in share URLs — significant for link-sharing UX.
#
# All checklist component IDs → ordered list of their possible values.
# Encoded as bitmask integers in new links (bit i set ↔ opts[i] selected).
# Old q2 links store lists; the decoder handles both formats transparently
# via isinstance(val, int).
_CHECKLIST_OPTIONS = {
    # quantile checklists (float values)
    "bub-qs":             list(_app_ctx._ALL_QS),
    "hm-exit-qs":         list(_app_ctx._ALL_QS),
    "dca-qs":             list(_app_ctx._ALL_QS),
    "ret-qs":             list(_app_ctx._ALL_QS),
    "sc-qs":              list(_app_ctx._ALL_QS),
    # toggle/boolean checklists (string values)
    "bub-toggles":        ["shade", "show_ols", "show_data", "show_today", "show_legend", "minor_grid", "chart_zoom"],
    "bub-bubble-toggles": ["show_comp", "show_sup"],
    "bub-show-stack":     ["yes"],
    "bub-use-lots":       ["yes"],
    "hm-toggles":         ["colorbar", "chart_zoom"],
    "hm-use-lots":        ["yes"],
    "dca-use-lots":       ["yes"],
    "dca-freq-unlock":    ["yes"],
    "dca-toggles":        ["log_y", "annotate", "show_legend", "minor_grid", "chart_zoom"],
    "dca-sc-enable":      ["yes"],
    "dca-sc-rollover":    ["yes"],
    "ret-use-lots":       ["yes"],
    "ret-freq-unlock":    ["yes"],
    "ret-toggles":        ["log_y", "annotate", "show_legend", "minor_grid", "chart_zoom"],
    "sc-use-lots":        ["yes"],
    "sc-freq-unlock":     ["yes"],
    "sc-toggles":         ["annotate", "log_y", "show_legend", "minor_grid", "chart_zoom"],
    "sc-chart-layout":    ["shade"],
    "bub-auto-y":         ["yes"],
    "dca-model-show":     ["qr", "mc", "pl", "s2f"],
    "ret-model-show":     ["qr", "mc", "pl", "s2f"],
    "sc-model-show":      ["qr", "mc", "pl", "s2f"],
    "hm-model-show":      ["qr", "mc", "pl", "s2f"],
    "bub-model-show":     ["pl", "s2f"],
}


# ── Validation: every checklist ID in _CHECKLIST_OPTIONS must appear in _SNAPSHOT_CONTROLS
_snap_cids = {cid for cid, _ in _SNAPSHOT_CONTROLS}
_checklist_missing = set(_CHECKLIST_OPTIONS) - _snap_cids
assert not _checklist_missing, f"Checklist IDs not in _SNAPSHOT_CONTROLS: {_checklist_missing}"
del _snap_cids, _checklist_missing


def _list_to_mask(val, opts):
    """Encode a checklist value list as a bitmask integer."""
    if not val:
        return 0
    sel = set(val)
    return sum(1 << i for i, o in enumerate(opts) if o in sel)


def _mask_to_list(mask, opts):
    """Decode a bitmask integer back to a checklist value list."""
    return [opts[i] for i in range(len(opts)) if mask & (1 << i)]


def _encode_snapshot(state_dict, tab_filter=None):
    """v2: positional array — no key names, ~50% smaller than v1.

    All checklist fields (quantiles and toggles) are stored as bitmask
    integers for compactness.  Old links that stored lists are still decoded
    transparently.

    If tab_filter is a set of component IDs, only those controls (plus
    main-tabs) are encoded; all others become None and fall back to defaults
    on restore.
    """
    values = []
    for cid, prop in _SNAPSHOT_CONTROLS:
        val = state_dict.get(f"{cid}:{prop}")
        if tab_filter is not None and cid != "main-tabs" and cid not in tab_filter:
            val = None
        if val is not None and cid in _CHECKLIST_OPTIONS:
            val = _list_to_mask(val, _CHECKLIST_OPTIONS[cid])
        values.append(val)
    lots   = state_dict.get("_lots")
    payload = [values, lots]
    j = json.dumps(payload, separators=(',', ':'))
    return base64.urlsafe_b64encode(gzip.compress(j.encode())).decode()


def _decode_snapshot(encoded):
    """Decode v2 (positional array) snapshot.

    Checklist fields may be either a bitmask int (new links) or a list
    (old links) — both are handled transparently.
    """
    try:
        payload = json.loads(gzip.decompress(base64.urlsafe_b64decode(encoded)))
        values, lots = payload
        # Forward/backward compat: pad or truncate to match current control count
        n_expected = len(_SNAPSHOT_CONTROLS)
        if len(values) < n_expected:
            log.info("Snapshot has %d controls, expected %d — padding with defaults",
                     len(values), n_expected)
            values.extend([None] * (n_expected - len(values)))
        elif len(values) > n_expected:
            log.info("Snapshot has %d controls, expected %d — truncating",
                     len(values), n_expected)
            values = values[:n_expected]
        state = {}
        for (cid, prop), val in zip(_SNAPSHOT_CONTROLS, values):
            if val is None:
                continue
            if cid in _CHECKLIST_OPTIONS and isinstance(val, int):
                val = _mask_to_list(val, _CHECKLIST_OPTIONS[cid])
            state[f"{cid}:{prop}"] = val
        if lots:
            state["_lots"] = lots
        return state
    except Exception:
        return None


def _decode_snapshot_v1(encoded):
    """Decode legacy v1 (dict-based) snapshot."""
    try:
        return json.loads(gzip.decompress(base64.urlsafe_b64decode(encoded)))
    except Exception:
        return None
