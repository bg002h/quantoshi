# MC Share Link Overhaul — Design Spec

**Date:** 2026-03-25
**Scope:** Snapshot/share system — add MC simulation controls to share links

## Goal

Share links currently encode 100 controls but omit all MC simulation settings except `{prefix}-mc-model-src`. A user who configures a MC simulation and shares the link loses all MC state. This overhaul adds 36 MC controls + 1 heatmap palette control to `_SNAPSHOT_CONTROLS`, with hybrid encoding that skips MC controls when MC is disabled.

## Current State

- `_SNAPSHOT_CONTROLS`: 100 entries (positional array, gzip+base64, URL hash `#q3:...`)
- MC controls in snapshot: only 4 (`dca-mc-model-src`, `ret-mc-model-src`, `sc-mc-model-src`, `hm-mc-model-src`)
- MC controls NOT in snapshot: enable, start_yr, entry_q, years, bins, regime, sims, window, advanced (x4 tabs = 36)
- `hm-palette` dropdown: missing from snapshot (colors ARE encoded but palette name is lost)
- Backward compat: decoder pads short arrays with `None`, truncates long arrays — no version bump needed

## Approach

Append 37 new entries to `_SNAPSHOT_CONTROLS` (indices 100–136). Use hybrid encoding: when MC is disabled on a tab, that tab's MC controls encode as `null`. Bitmask encoding for `mc-regime` checklists and `mc-enable`/`mc-advanced` checklists.

## Changes

### 1. New `_SNAPSHOT_CONTROLS` entries — `btc_web/snapshot.py`

Append these 37 entries at the end of the `_SNAPSHOT_CONTROLS` list (after index 99):

```python
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
# Heatmap palette
("hm-palette",       "value"),   # 136
```

**Total: 100 + 37 = 137 entries.**

**Intentionally excluded:** `{prefix}-mc-entry-yr` is a hidden slider used for internal ticker sync, not user-set state. Its value is derived from `mc-entry-q` and the current price — it should NOT be snapshotted.

### 2. New `_CHECKLIST_OPTIONS` entries — `btc_web/snapshot.py`

Add bitmask encoding for MC checklists:

```python
# MC enable/advanced checklists (1 bit each)
"dca-mc-enable":    ["yes"],
"dca-mc-advanced":  ["yes"],
"ret-mc-enable":    ["yes"],
"ret-mc-advanced":  ["yes"],
"hm-mc-enable":     ["yes"],
"hm-mc-advanced":   ["yes"],
"sc-mc-enable":     ["yes"],
"sc-mc-advanced":   ["yes"],
# MC regime checklists (5 bits each — regime bins 0-4, int values)
"dca-mc-regime":    [0, 1, 2, 3, 4],
"ret-mc-regime":    [0, 1, 2, 3, 4],
"hm-mc-regime":     [0, 1, 2, 3, 4],
"sc-mc-regime":     [0, 1, 2, 3, 4],
```

**Total: 12 new entries in `_CHECKLIST_OPTIONS`.**

**CRITICAL: Atomic deployment.** The `_CHECKLIST_OPTIONS` validation assertion at module load checks that every key in `_CHECKLIST_OPTIONS` exists in `_SNAPSHOT_CONTROLS`. Adding `_CHECKLIST_OPTIONS` entries without the matching `_SNAPSHOT_CONTROLS` entries raises `AssertionError` and crashes gunicorn. Both changes must be in the same commit.

### 3. Update `_TAB_CONTROLS` — `btc_web/callbacks/nav.py`

Add MC component IDs to each tab's control set:

```python
_TAB_CONTROLS["dca"].update({
    "dca-mc-enable", "dca-mc-start-yr", "dca-mc-entry-q", "dca-mc-years",
    "dca-mc-bins", "dca-mc-regime", "dca-mc-sims", "dca-mc-window", "dca-mc-advanced",
})
_TAB_CONTROLS["retire"].update({
    "ret-mc-enable", "ret-mc-start-yr", "ret-mc-entry-q", "ret-mc-years",
    "ret-mc-bins", "ret-mc-regime", "ret-mc-sims", "ret-mc-window", "ret-mc-advanced",
})
_TAB_CONTROLS["heatmap"].update({
    "hm-mc-enable", "hm-mc-start-yr", "hm-mc-entry-q", "hm-mc-years",
    "hm-mc-bins", "hm-mc-regime", "hm-mc-sims", "hm-mc-window", "hm-mc-advanced",
    "hm-palette",
})
_TAB_CONTROLS["supercharge"].update({
    "sc-mc-enable", "sc-mc-start-yr", "sc-mc-entry-q", "sc-mc-years",
    "sc-mc-bins", "sc-mc-regime", "sc-mc-sims", "sc-mc-window", "sc-mc-advanced",
})
```

Note: `hm-palette` added to heatmap tab controls.

### 4. Hybrid encoding — `btc_web/snapshot.py`

In `_encode_snapshot()`, add MC-disable optimization after building the `values` list:

```python
# Null-out MC controls for tabs where MC is not enabled
_mc_prefixes = {"dca": "dca-mc-", "ret": "ret-mc-", "hm": "hm-mc-", "sc": "sc-mc-"}
for _pfx_tab, _pfx_mc in _mc_prefixes.items():
    enable_idx = next(i for i, (cid, _) in enumerate(_SNAPSHOT_CONTROLS)
                      if cid == f"{_pfx_mc}enable")
    mc_on = values[enable_idx] not in (None, [], 0)
    if not mc_on:
        for i, (cid, _) in enumerate(_SNAPSHOT_CONTROLS):
            if cid.startswith(_pfx_mc) and cid != f"{_pfx_mc}model-src":
                values[i] = None
```

This runs after values are collected but before gzip+base64 encoding. When MC is off, all MC controls for that tab become `null` — they compress to nearly zero bytes.

Note: `{prefix}-mc-model-src` is NOT nulled even when MC is off, because it's used by the "Display models" checklist independently of MC activation.

### 5. Decoding — no changes

The existing decoder handles this automatically:
- `null` values → controls stay at defaults (MC off, all MC controls at initial values)
- Old links with 100 entries → padded with 37 `None` values → MC defaults
- New links on old code → truncated to 100 entries → MC controls lost, base controls work

### 6. `restore_from_url` callback — `btc_web/callbacks/snapshot_cb.py`

The existing `restore_from_url` callback outputs to all component IDs in `_SNAPSHOT_CONTROLS`. Adding 37 new entries means 37 new `Output()` declarations in the callback decorator. The callback body already iterates `_SNAPSHOT_CONTROLS` and returns values positionally — no logic change needed, just the Output list grows.

### 7. MC body visibility on restore — cascading callbacks

When `{prefix}-mc-enable` is restored to `["yes"]`, the existing callback `_toggle_mc_body` (in `callbacks/mc_controls.py`) fires reactively and sets `{prefix}-mc-body` style to `{}` (visible). Similarly, `{prefix}-mc-advanced` = `["yes"]` triggers `_toggle_mc_advanced` which reveals the advanced body and switches `mc-entry-q` options to fine-grained mode.

**Cascade dependency:** This relies on `restore_from_url` having `prevent_initial_call=False` — it fires on initial page load, writes MC control values, which then trigger the downstream visibility callbacks. If `restore_from_url` ever changes to `prevent_initial_call=True`, MC body restore will silently break.

**`mc-entry-q` race condition:** When `mc-advanced=["yes"]` was active at link creation, the user may have set a fine-grained `mc-entry-q` value (e.g. `7.5%`). On restore, `restore_from_url` sets both `mc-advanced=["yes"]` and `mc-entry-q=7.5` simultaneously. The `_toggle_mc_advanced` callback fires reactively, switching the dropdown to fine-grained options. During the brief window before this fires, the dropdown has standard options (10% steps) but a non-matching value. Dash handles this gracefully — the dropdown shows the numeric value and once options update, it renders correctly. This only affects advanced-mode links with non-10%-aligned entry quantiles and is cosmetic, not functional.

### 8. Free tier behavior

Share links that encode free-tier MC settings (entry_q=10%, start_yr in {2028, 2031, 2035}, years=40, sims=200 — the pre-computed cache parameters from `mc_cache.py`) will work immediately on the recipient's browser. The pre-computed cache serves these without payment. Non-cached settings restore the controls but require the user to click Run / pay for the simulation.

### 9. `hm-palette` fix

Add `("hm-palette", "value")` to `_SNAPSHOT_CONTROLS` (index 136) and `"hm-palette"` to `_TAB_CONTROLS["heatmap"]`.

**Restore mechanics:** `restore_from_url` is a single callback that outputs ALL values at once — including both `hm-palette` (index 136) and `hm-c-lo`/`hm-c-mid1`/`hm-c-mid2`/`hm-c-hi` (indices 23–26). These are set simultaneously. The `apply_hm_palette` callback has `prevent_initial_call=True`, so it does NOT fire during initial restore. The individual color values from the snapshot are the authoritative final state; the palette dropdown provides cosmetic context showing which preset was selected.

## Backward Compatibility

| Scenario | Behavior |
|----------|----------|
| Old link (100 entries) on new code (137) | Pad 37 `None` → MC disabled, hm-palette=default |
| New link (137 entries) on old code (100) | Truncate → MC controls lost, base controls work |
| New link, MC disabled | MC controls encode as `null` → compress to ~0 bytes |
| New link, MC enabled (free tier) | MC controls restored, simulation renders from cache |
| New link, MC enabled (paid tier) | MC controls restored, user must run/pay to render |

## URL Size Impact

- **MC disabled:** ~15-20 extra URL chars (37 nulls compress extremely well)
- **MC enabled, 1 tab:** ~60-80 extra URL chars (9 non-null values + 27 nulls)
- **MC enabled, all tabs:** ~200-250 extra URL chars (36 non-null values)
- **Current average URL length:** ~800-1200 chars (single tab), ~2000-3000 chars (all tabs)

## Not In Scope

- MC results/cache sharing (simulation data is too large for URL encoding)
- MC payment state (server-side, not shareable)
- MC upload/download state (ephemeral)
- `{prefix}-mc-entry-yr` — hidden internal sync slider, not user-set state
- `{prefix}-mc-amount/infl/stack` for DCA/Ret/SC — shared with tab controls, already encoded
- `hm-mc-amount/infl/stack` — derived from heatmap entry settings
- `hm-mc-freq` — locked to Monthly for cache alignment; uncommon to change and free-tier cache is Monthly-only. Could be added in a future iteration if needed.
