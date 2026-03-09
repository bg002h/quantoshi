"""Snapshot / URL state encoding and decoding for share links."""

import json
import gzip
import base64

import _app_ctx

_SNAPSHOT_CONTROLS = [
    ("bub-qs",            "value"),
    ("bub-xscale",        "value"),
    ("bub-yscale",        "value"),
    ("bub-xrange",        "value"),
    ("bub-yrange",        "value"),
    ("bub-toggles",       "value"),
    ("bub-bubble-toggles","value"),
    ("bub-n-future",      "value"),
    ("bub-ptsize",        "value"),
    ("bub-ptalpha",       "value"),
    ("bub-stack",         "value"),
    ("bub-show-stack",    "value"),
    ("bub-use-lots",      "value"),
    ("hm-entry-yr",       "value"),
    ("hm-entry-q",        "value"),
    ("hm-exit-range",     "value"),
    ("hm-exit-qs",        "value"),
    ("hm-mode",           "value"),
    ("hm-b1",             "value"),
    ("hm-b2",             "value"),
    ("hm-c-lo",           "value"),
    ("hm-c-mid1",         "value"),
    ("hm-c-mid2",         "value"),
    ("hm-c-hi",           "value"),
    ("hm-grad",           "value"),
    ("hm-vfmt",           "value"),
    ("hm-cell-fs",        "value"),
    ("hm-toggles",        "value"),
    ("hm-stack",          "value"),
    ("hm-use-lots",       "value"),
    ("dca-stack",         "value"),
    ("dca-use-lots",      "value"),
    ("dca-amount",        "value"),
    ("dca-freq",          "value"),
    ("dca-yr-range",      "value"),
    ("dca-disp",          "value"),
    ("dca-toggles",       "value"),
    ("dca-qs",            "value"),
    ("dca-sc-enable",     "value"),
    ("dca-sc-loan",       "value"),
    ("dca-sc-rate",       "value"),
    ("dca-sc-term",       "value"),
    ("dca-sc-type",       "value"),
    ("dca-sc-repeats",    "value"),
    ("dca-sc-entry-mode", "value"),
    ("dca-sc-custom-price","value"),
    ("dca-sc-tax",        "value"),
    ("dca-sc-rollover",   "value"),
    ("ret-stack",         "value"),
    ("ret-use-lots",      "value"),
    ("ret-wd",            "value"),
    ("ret-freq",          "value"),
    ("ret-yr-range",      "value"),
    ("ret-infl",          "value"),
    ("ret-disp",          "value"),
    ("ret-toggles",       "value"),
    ("ret-legend-pos",    "value"),
    ("ret-qs",            "value"),
    ("sc-stack",          "value"),
    ("sc-use-lots",       "value"),
    ("sc-start-yr",       "value"),
    ("sc-d0",             "value"),
    ("sc-d1",             "value"),
    ("sc-d2",             "value"),
    ("sc-d3",             "value"),
    ("sc-d4",             "value"),
    ("sc-freq",           "value"),
    ("sc-infl",           "value"),
    ("sc-qs",             "value"),
    ("sc-mode",           "value"),
    ("sc-wd",             "value"),
    ("sc-end-yr",         "value"),
    ("sc-target-yr",      "value"),
    ("sc-disp",           "value"),
    ("sc-toggles",        "value"),
    ("sc-chart-layout",   "value"),
    ("sc-display-q",      "value"),
    ("bub-auto-y",        "value"),
    ("main-tabs",         "active_tab"),
]

_SNAP_PREFIX    = "q2:"   # current format
_SNAP_PREFIX_V1 = "q1:"   # legacy format (dict-based), kept for backward compat

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
    "bub-toggles":        ["shade", "show_ols", "show_data", "show_today", "show_legend", "minor_grid"],
    "bub-bubble-toggles": ["show_comp", "show_sup"],
    "bub-show-stack":     ["yes"],
    "bub-use-lots":       ["yes"],
    "hm-toggles":         ["colorbar"],
    "hm-use-lots":        ["yes"],
    "dca-use-lots":       ["yes"],
    "dca-toggles":        ["log_y", "dual_y", "show_legend", "minor_grid"],
    "dca-sc-enable":      ["yes"],
    "dca-sc-rollover":    ["yes"],
    "ret-use-lots":       ["yes"],
    "ret-toggles":        ["log_y", "dual_y", "annotate", "show_legend", "minor_grid", "show_today"],
    "sc-use-lots":        ["yes"],
    "sc-toggles":         ["annotate", "log_y", "show_legend", "minor_grid", "show_today"],
    "sc-chart-layout":    ["shade"],
    "bub-auto-y":         ["yes"],
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
