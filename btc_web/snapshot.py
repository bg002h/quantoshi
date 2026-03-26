"""Snapshot / URL state encoding and decoding for share links."""

import json
import gzip
import base64
import logging

import _app_ctx

log = logging.getLogger(__name__)

_SNAPSHOT_CONTROLS = [
    # ── Bubble tab (indices 0–15) ──
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
    ("scan-price",        "value"),   # scanner price input
    ("scan-date",         "value"),   # scanner date input
    ("scan-q",            "value"),   # scanner quantile input
    # ── Heatmap tab (indices 16–31) ──
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
    # ── Palette (index 98) ──
    ("palette-store",     "data"),    # colorblind palette key
    # ── Heatmap model selector (index 99) ──
    ("hm-active-model",   "data"),    # active heatmap model pill (bub/pl/s2f/mc)
    # ── MC controls (4 tabs x 9 controls) ────────────────────────────────
    # DCA MC
    ("dca-mc-enable",    "value"),   # 100
    ("dca-mc-start-yr",  "value"),   # 101
    ("dca-mc-entry-q",   "value"),   # 102
    ("dca-mc-years",     "value"),   # 103
    ("dca-mc-bins",      "value"),   # 104
    ("dca-mc-regime",    "value"),   # 105
    ("dca-mc-sims",      "value"),   # 106
    ("dca-mc-window",    "value"),   # 107
    ("dca-mc-advanced",  "value"),   # 108
    # Retire MC
    ("ret-mc-enable",    "value"),   # 109
    ("ret-mc-start-yr",  "value"),   # 110
    ("ret-mc-entry-q",   "value"),   # 111
    ("ret-mc-years",     "value"),   # 112
    ("ret-mc-bins",      "value"),   # 113
    ("ret-mc-regime",    "value"),   # 114
    ("ret-mc-sims",      "value"),   # 115
    ("ret-mc-window",    "value"),   # 116
    ("ret-mc-advanced",  "value"),   # 117
    # Heatmap MC
    ("hm-mc-enable",     "value"),   # 118
    ("hm-mc-start-yr",   "value"),   # 119
    ("hm-mc-entry-q",    "value"),   # 120
    ("hm-mc-years",      "value"),   # 121
    ("hm-mc-bins",       "value"),   # 122
    ("hm-mc-regime",     "value"),   # 123
    ("hm-mc-sims",       "value"),   # 124
    ("hm-mc-window",     "value"),   # 125
    ("hm-mc-advanced",   "value"),   # 126
    # Supercharger MC
    ("sc-mc-enable",     "value"),   # 127
    ("sc-mc-start-yr",   "value"),   # 128
    ("sc-mc-entry-q",    "value"),   # 129
    ("sc-mc-years",      "value"),   # 130
    ("sc-mc-bins",       "value"),   # 131
    ("sc-mc-regime",     "value"),   # 132
    ("sc-mc-sims",       "value"),   # 133
    ("sc-mc-window",     "value"),   # 134
    ("sc-mc-advanced",   "value"),   # 135
    # ── Heatmap palette ──────────────────────────────────────────────────
    ("hm-palette",       "value"),   # 136
    # ── Citadel Planner tab ──────────────────────────────────────────────
    ("cp-stack",            "value"),   # 137
    ("cp-use-lots",         "value"),   # 138
    ("cp-cash-init",        "value"),   # 139
    ("cp-cash-rate",        "value"),   # 140
    ("cp-res-short-init",   "value"),   # 141
    ("cp-res-short-rate",   "value"),   # 142
    ("cp-res-short-vol",    "value"),   # 143
    ("cp-res-med-init",     "value"),   # 144
    ("cp-res-med-rate",     "value"),   # 145
    ("cp-res-med-vol",      "value"),   # 146
    ("cp-res-long-init",    "value"),   # 147
    ("cp-res-long-rate",    "value"),   # 148
    ("cp-res-long-vol",     "value"),   # 149
    ("cp-inv-eq-init",      "value"),   # 150
    ("cp-inv-eq-rate",      "value"),   # 151
    ("cp-inv-eq-vol",       "value"),   # 152
    ("cp-inv-bd-init",      "value"),   # 153
    ("cp-inv-bd-rate",      "value"),   # 154
    ("cp-inv-bd-vol",       "value"),   # 155
    ("cp-spend",            "value"),   # 156
    ("cp-infl",             "value"),   # 157
    ("cp-spend-growth",     "value"),   # 158
    ("cp-high-q-thresh",    "value"),   # 159
    ("cp-high-q-mode",      "value"),   # 160
    ("cp-high-q-rate",      "value"),   # 161
    ("cp-high-q-dur",       "value"),   # 162
    ("cp-high-q-split-cash","value"),   # 163
    ("cp-high-q-split-rs",  "value"),   # 164
    ("cp-high-q-split-rm",  "value"),   # 165
    ("cp-high-q-split-rl",  "value"),   # 166
    ("cp-high-q-split-eq",  "value"),   # 167
    ("cp-high-q-split-bd",  "value"),   # 168
    ("cp-low-q-thresh",     "value"),   # 169
    ("cp-low-q-mode",       "value"),   # 170
    ("cp-low-q-rate",       "value"),   # 171
    ("cp-low-q-dur",        "value"),   # 172
    ("cp-low-q-split-cash", "value"),   # 173
    ("cp-low-q-split-rs",   "value"),   # 174
    ("cp-low-q-split-rm",   "value"),   # 175
    ("cp-low-q-split-rl",   "value"),   # 176
    ("cp-low-q-split-eq",   "value"),   # 177
    ("cp-low-q-split-bd",   "value"),   # 178
    ("cp-lump-cooldown",    "value"),   # 179
    ("cp-cash-floor",       "value"),   # 180
    ("cp-res-short-floor",  "value"),   # 181
    ("cp-res-med-floor",    "value"),   # 182
    ("cp-res-long-floor",   "value"),   # 183
    ("cp-scf-enable",       "value"),   # 184
    ("cp-scf-amount",       "value"),   # 185
    ("cp-scf-type",         "value"),   # 186
    ("cp-scf-rate",         "value"),   # 187
    ("cp-scf-term",         "value"),   # 188
    ("cp-scf-trigger",      "value"),   # 189
    ("cp-yr-range",         "value"),   # 190
    ("cp-freq",             "value"),   # 191
    ("cp-qs",               "value"),   # 192
    ("cp-model-src",        "value"),   # 193
    ("cp-disp",             "value"),   # 194
    ("cp-toggles",          "value"),   # 195
    ("cp-legend-pos",       "value"),   # 196
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
_QS_LIST = list(_app_ctx._ALL_QS)

_CHECKLIST_OPTIONS = {
    # quantile checklists (float values) — all share the same list object
    "bub-qs":             _QS_LIST,
    "hm-exit-qs":         _QS_LIST,
    "dca-qs":             _QS_LIST,
    "ret-qs":             _QS_LIST,
    "sc-qs":              _QS_LIST,
    # toggle/boolean checklists (string values)
    "bub-toggles":        ["shade", "show_ols", "show_data", "show_today", "show_legend", "minor_grid", "chart_zoom"],
    "bub-bubble-toggles": ["show_comp", "show_sup"],
    "bub-show-stack":     ["yes"],
    "bub-use-lots":       ["yes"],
    "hm-toggles":         ["colorbar", "chart_zoom"],
    "hm-use-lots":        ["yes"],
    "dca-use-lots":       ["yes"],
    "dca-freq-unlock":    ["yes"],
    "dca-toggles":        ["log_y", "annotate", "show_legend", "minor_grid", "chart_zoom", "discrete"],
    "dca-sc-enable":      ["yes"],
    "dca-sc-rollover":    ["yes"],
    "ret-use-lots":       ["yes"],
    "ret-freq-unlock":    ["yes"],
    "ret-toggles":        ["log_y", "annotate", "show_legend", "minor_grid", "chart_zoom", "discrete"],
    "sc-use-lots":        ["yes"],
    "sc-freq-unlock":     ["yes"],
    "sc-toggles":         ["annotate", "log_y", "show_legend", "minor_grid", "chart_zoom", "discrete"],
    "sc-chart-layout":    ["shade"],
    "bub-auto-y":         ["yes"],
    "dca-model-show":     ["qr", "mc", "pl", "lppl", "exp", "s2f", "ef"],
    "ret-model-show":     ["qr", "mc", "pl", "lppl", "exp", "s2f", "ef"],
    "sc-model-show":      ["qr", "mc", "pl", "lppl", "exp", "s2f", "ef"],
    "hm-model-show":      ["qr", "mc", "pl", "lppl", "exp", "s2f", "ef"],
    "bub-model-show":     ["pl", "lppl", "exp", "s2f", "ef", "bub", "qr"],
    # MC enable/advanced checklists (1 bit each)
    "dca-mc-enable":    ["yes"],
    "dca-mc-advanced":  ["yes"],
    "ret-mc-enable":    ["yes"],
    "ret-mc-advanced":  ["yes"],
    "hm-mc-enable":     ["yes"],
    "hm-mc-advanced":   ["yes"],
    "sc-mc-enable":     ["yes"],
    "sc-mc-advanced":   ["yes"],
    # MC regime checklists (5 bits each — int values 0-4)
    "dca-mc-regime":    [0, 1, 2, 3, 4],
    "ret-mc-regime":    [0, 1, 2, 3, 4],
    "hm-mc-regime":     [0, 1, 2, 3, 4],
    "sc-mc-regime":     [0, 1, 2, 3, 4],
    # Citadel Planner checklists (cp-qs is a dropdown, not a checklist)
    "cp-toggles":       ["log_y", "annotate", "show_legend", "minor_grid", "chart_zoom"],
    "cp-use-lots":      ["yes"],
    "cp-scf-enable":    ["yes"],
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
    # ── Hybrid MC encoding: null-out MC controls for disabled tabs ────────
    _mc_prefixes = {"dca": "dca-mc-", "ret": "ret-mc-", "hm": "hm-mc-", "sc": "sc-mc-"}
    for _pfx_tab, _pfx_mc in _mc_prefixes.items():
        enable_idx = next(i for i, (cid, _) in enumerate(_SNAPSHOT_CONTROLS)
                          if cid == f"{_pfx_mc}enable")
        mc_on = values[enable_idx] not in (None, [], 0)
        if not mc_on:
            for i, (cid, _) in enumerate(_SNAPSHOT_CONTROLS):
                if cid.startswith(_pfx_mc) and cid != f"{_pfx_mc}model-src":
                    values[i] = None
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
