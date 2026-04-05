# Display Models Standardization — Phase 2 Design Spec

**Date:** 2026-04-04
**Scope:** Heatmap pill-bar refactor — collapse individual LPPL-family pills into a single "LPPL" master pill + add LPPL Models sub-config panel on tab 2.
**Depends on:** Phase 1 (shared `_lppl_config_panel` helper, `hm-lppl-activate` control already reserved in snapshot).
**Out of scope:** Hierarchical deep-link routing `/2.4.N` for LPPL flavor config (deferred to Phase 3).

## Goal

Bring the Heatmap tab's model selector into line with the standardized LPPL-master pattern. The pill bar should show one "LPPL" master pill instead of 10 individual LPPL-family pills, with flavor selection delegated to the LPPL Models sub-config panel (same panel used on Phase 1 tabs).

Also remove Exp and S2F pills (display-only models, Bubble tab only per policy).

## Constraints

**Per user directive: tab 2 deep-link backward compat is NOT required for Phase 2.** Old `/2.N` links that resolved to specific LPPL flavors (`/2.5` → lp3, etc.) are allowed to land on different models (or the closest non-LPPL-flavor pill). A new hierarchical format `/2.4.N` for encoding LPPL flavor is deferred to Phase 3.

**Snapshot `q3:…` link compat still required.** `hm-model-show` is a hidden compat-only placeholder checklist (see `layout/heatmap.py:120`); `_CHECKLIST_OPTIONS` entries stay stable for snapshot decoding. The pill-bar state is snapshot'd via `hm-active-model` (a single string), which is also stable.

## Architecture

### Layer 1 — Pill bar refactor (`layout/heatmap.py`)

Current `_hm_pill_bar()` iterates `_app_ctx.PRICE_MODELS` emitting a pill per model (skipping `bub` and `mc`, adding MC separately). Result: ~15 pills including all LPPL flavors + Exp + S2F.

New `_hm_pill_bar()`:
- Pill 1: **BM** (Bubble Model) — unchanged.
- Pill 2: **PL** (Power Law) — unchanged.
- Pill 3: **LPPL** master (new; color `#FF6D00`).
- Pill 4: **LinPPL** — unchanged.
- Pill 5: **HybPPL** — unchanged.
- Pill 6: **EF** (BM Empirical Floor) — unchanged.
- Pill 7: **U₁** (User Model) — added if `u1` is in `PRICE_MODELS`.
- Pill 8: **MC** (conditional, if `_HAS_MARKOV`) — unchanged.

**Removed from pill bar:** lp2, lp3, lp4, lppl_w, lp2_w, lp3_w, lp4_w, lp4_n13, lp4_w_n13, exp, s2f.

Pill IDs follow the existing `hm-pill-{key}` pattern. Pill positions are renumbered; see Layer 2 for the `_HM_PILL_IDS` / `_HM_PILL_MODELS` update.

### Layer 2 — Deep-link routing (`callbacks/routing.py`)

`_HM_PILL_MODELS` and `_HM_PILL_IDS` are rebuilt to match the new pill order. Resulting list (assuming all optional pills present):

```python
_HM_PILL_MODELS = ["bub", "pl", "lppl", "linppl", "hybppl", "ef", "u1", "mc"]
_HM_PILL_IDS = [f"hm-pill-{k}" for k in _HM_PILL_MODELS]
```

`_hm_pill_click` and `_hm_pill_sync` adapt to the new list length automatically (they iterate `_HM_PILL_IDS`).

Deep-link `/2.N` behavior changes:
- `/2.1` → BM (unchanged)
- `/2.2` → PL (unchanged)
- `/2.3` → LPPL master (was S2F — changed)
- `/2.4` → LinPPL (was LPPL — changed)
- `/2.5` → HybPPL
- `/2.6` → EF
- `/2.7` → U₁
- `/2.8` → MC

Old `/2.N` URLs resolve to different models. This is **accepted per user directive**.

### Layer 3 — LPPL sub-config panel on Heatmap

Uses the **same `_lppl_config_panel("hm")` helper** from Phase 1. Panel appears below the pill bar, above the heatmap chart controls.

`hm-lppl-activate` checkbox is the activate control (already reserved in snapshot via Phase 1).

### Layer 4 — Clientside sync callbacks (heatmap-specific)

Two new clientside callbacks specific to the pill+panel interaction:

**1. LPPL pill active ↔ hm-lppl-activate:**

When `hm-active-model` changes to `"lppl"` → `hm-lppl-activate` becomes `["yes"]` and `hm-lppl-body` expands.
When `hm-active-model` changes away from `"lppl"` → `hm-lppl-activate` becomes `[]` and body collapses.

This is a one-way callback from `hm-active-model` (data) to `hm-lppl-activate` (value) + `hm-lppl-body` (style).

**2. hm-lppl-activate → hm-active-model (reverse sync):**

When user clicks the activate checkbox (not via pill), sync back to `hm-active-model`:
- Check → set `hm-active-model="lppl"` (and trigger `_hm_pill_sync` via store update).
- Uncheck → if `hm-active-model` is currently `"lppl"`, revert to `"bub"` (default).

**Race condition guard:** `_hm_pill_click` writes `hm-active-model`, and the above callbacks read it. The bi-directional sync must `no_update` when state already matches, else we risk infinite loops. Pattern:

```js
function(active_model, cur_activate) {
    var should_activate = (active_model === "lppl");
    var is_activated = (cur_activate || []).length > 0;
    if (should_activate === is_activated) return window.dash_clientside.no_update;
    return should_activate ? ["yes"] : [];
}
```

### Layer 5 — Chart callback translation (`update_heatmap`)

Current `update_heatmap` passes `hm_model` (string) directly to the figure builder. After Phase 2:

```python
# New State inputs
State("lppl-n-freqs", "value"),
State("lppl-weighted", "value"),
State("lppl-no-13", "value"),

# In callback body, before figure builder:
if hm_model == "lppl":
    _weighted = "weighted" in (lppl_weighted or [])
    _no_13 = "no13" in (lppl_no_13 or [])
    # Pick first selected n_freq (pill bar is single-select; only first applies)
    n = (lppl_n_freqs or [3])[0]
    if n == 1: hm_model = "lppl_w" if _weighted else "lppl"
    elif n == 2: hm_model = "lp2_w" if _weighted else "lp2"
    elif n == 3 and not _no_13: hm_model = "lp3_w" if _weighted else "lp3"
    elif n == 4:
        if _no_13: hm_model = "lp4_w_n13" if _weighted else "lp4_n13"
        else: hm_model = "lp4_w" if _weighted else "lp4"
    # else fallback: keep as "lppl" (LPPL₁ default)
```

Translation happens **only** in the non-MC heatmap figure path (QR bands via `_get_heatmap_fig`). The MC heatmap path (`_get_mc_heatmap_fig`) uses `hm-mc-model-src` (separate dropdown) which stays at model-keys like `"lppl"` — MC cache is keyed on the master, not flavors. No translation for the MC path.

### Layer 6 — MC cache-key isolation

`mc_cache._CACHED_MODEL_KEYS = frozenset(["bub", "qr", "pl", "lppl", "exp", "ef"])` stays unchanged. The MC cache knows about the LPPL master only (not individual flavors). When user picks LPPL master on the pill bar and also activates MC, the MC path uses `hm-mc-model-src` value for its model lookup, which remains `"lppl"` for LPPL.

If in future users want MC to respond to flavor selection, that's a Phase 3+ discussion requiring cache rebuild — out of scope.

### Layer 7 — Palette callback update

`update_model_swatches` in `callbacks/charts.py` currently only rebuilds `bub-model-show` / `dca-model-show` / `ret-model-show` / `sc-model-show` options. Phase 2 adds:

- Rebuild heatmap pill-bar styling on palette change (swatches in pill labels).
- Mechanism: either a new small callback `Output("hm-pill-bar-container", "children")` that rebuilds the pill bar, or pass palette-derived swatch colors via CSS vars.

Simpler path: a dedicated `update_heatmap_pill_swatches` callback that re-renders the pill bar HTML. Added in Phase 2.

### Layer 8 — Snapshot (`snapshot.py`)

No snapshot-format changes. `hm-lppl-activate` was already appended to `_SNAPSHOT_CONTROLS` and `_CHECKLIST_OPTIONS` in Phase 1. `hm-active-model` is already snapshotted as a string; new values work automatically.

**`_CHECKLIST_OPTIONS["hm-model-show"]` untouched** for snapshot compat.

## Data flow example — user activates LPPL₃ on Heatmap

1. User clicks the `LPPL` pill (`hm-pill-lppl`).
2. `_hm_pill_click` callback sets `hm-active-model = "lppl"`.
3. `_hm_pill_sync` re-styles the pills (LPPL solid, others outline).
4. Clientside callback #1 detects `hm-active-model="lppl"`, sets `hm-lppl-activate=["yes"]`, expands `hm-lppl-body`.
5. User sees LPPL panel expanded with current n_freqs=[3] (default) + weighted=[].
6. User leaves defaults (LPPL₃, unweighted).
7. `update_heatmap` fires with `hm_model="lppl"`, `lppl_n_freqs=[3]`, `lppl_weighted=[]`, `lppl_no_13=[]`.
8. Translation logic sets `hm_model="lp3"`.
9. Heatmap renders using LPPL₃ quantile bands.

## Error handling

- If `lppl-n-freqs` is empty, translation falls back to `"lppl"` (LPPL₁ default). Figure renders.
- If user activates LPPL pill and the resulting flavor key (e.g., `"lp4_n13"`) isn't in `PRICE_MODELS`, figure builder's existing `_resolve_model` handles gracefully.
- Race condition on simultaneous pill+activate-checkbox writes → `no_update` guards prevent loops.
- Old `/2.5` URL → `_hm_deep_link_route` sets `hm-active-model` to whichever key is at index 5 in the new `_HM_PILL_MODELS` list. User accepts this may land on a different model.

## Testing

### Unit tests (pytest)

- `test_heatmap_pill_bar_new` — verify pill list has expected count and ordering.
- `test_hm_active_model_lppl_translates` — `update_heatmap` with `hm_model="lppl"` + config produces correct flavor key call.
- `test_hm_active_model_mc_path_untranslated` — MC path with `hm-mc-model-src="lppl"` doesn't get translated.
- `test_hm_lppl_activate_sync_bidirectional` — clientside callback registration assertions.
- `test_hm_deep_link_new_index_map` — `/2.3` resolves to LPPL master in new numbering.

### Manual / Playwright verification

- Visit `/2`, count pills, verify no individual LPPL-flavor pills, no Exp, no S2F.
- Click LPPL pill → verify panel expands, LPPL heatmap renders.
- Change n_freqs in panel → verify heatmap redraws with new flavor.
- Click away from LPPL → verify panel collapses.
- Change palette → pill swatches update.
- Old `/2.5` link → land on whatever new index 5 is (doc in release notes).

## File list

**Modified:**
- `btc_web/layout/heatmap.py` — rewrite `_hm_pill_bar()`, add `_lppl_config_panel("hm")` integration.
- `btc_web/callbacks/routing.py` — update `_HM_PILL_MODELS`, `_HM_PILL_IDS`.
- `btc_web/callbacks/charts.py` — update `update_heatmap` translation, update `_hm_pill_sync` to handle new pill set, new clientside sync callbacks, add heatmap pill swatch update callback.
- `btc_web/test_web.py` — add heatmap-specific tests.

**No new files.**

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Old `/2.N` links land on wrong model | User explicitly accepted this. Documented in release notes. |
| Clientside callback race condition (pill ↔ activate checkbox) | `no_update` guards tested in unit tests. |
| MC path accidentally receives flavor key | Translation isolated to non-MC path; covered by unit test. |
| Palette-change doesn't re-style pill bar | New `update_heatmap_pill_swatches` callback. |
| User expects flavor choice to affect MC heatmap | Documented limitation; MC cache scope. Potential Phase 3 work. |

## Success criteria

- Heatmap pill bar shows 6-8 pills (depending on MC / U₁ availability), no individual LPPL flavors.
- Clicking LPPL pill expands the LPPL Models panel.
- Changing n_freqs/weighted/no_13 in the panel updates the heatmap.
- Palette changes restyle the pill bar.
- MC heatmap unaffected by the refactor.
- All tests pass.
