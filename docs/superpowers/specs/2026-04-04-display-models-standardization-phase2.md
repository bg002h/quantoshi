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

Current `_hm_pill_bar()` iterates `_app_ctx.PRICE_MODELS` emitting a pill per model (skipping `bub` and `mc`, adding MC separately). Result: ~17 pills including QR, all LPPL flavors, LinPPL, HybPPL, Exp, S2F, EF.

New `_hm_pill_bar()`:
- Pill 1: **BM** (Bubble Model) — unchanged.
- Pill 2: **PL** (Power Law) — unchanged.
- Pill 3: **LPPL** master (new; color `#FF6D00`).
- Pill 4: **LinPPL** — unchanged.
- Pill 5: **HybPPL** — unchanged.
- Pill 6: **EF** (BM Empirical Floor) — unchanged.
- Pill 7: **U₁** (User Model) — added if `u1` is in `PRICE_MODELS`.
- Pill 8: **MC** (conditional, if `_HAS_MARKOV`) — unchanged.

**Removed from pill bar:** qr, lp2, lp3, lp4, lppl_w, lp2_w, lp3_w, lp4_w, lp4_n13, lp4_w_n13, exp, s2f. (QR removal is a design decision — see next layer.)

Pill IDs follow the existing `hm-pill-{key}` pattern. Pill positions are renumbered; see Layer 2 for the `_HM_PILL_IDS` / `_HM_PILL_MODELS` update.

### Layer 2 — Deep-link routing (`callbacks/routing.py`)

`_HM_PILL_MODELS` and `_HM_PILL_IDS` are rebuilt to match the new pill order. Resulting list (assuming all optional pills present):

```python
_HM_PILL_MODELS = ["bub", "pl", "lppl", "linppl", "hybppl", "ef", "u1", "mc"]
_HM_PILL_IDS = [f"hm-pill-{k}" for k in _HM_PILL_MODELS]
```

`_hm_pill_click` and `_hm_pill_sync` adapt to the new list length automatically (they iterate `_HM_PILL_IDS`).

Deep-link `/2.N` behavior changes.

**Current order** (verified against `_app_ctx.PRICE_MODELS` insertion order in `app.py:173–195`):

| /2.N | Current | New |
|---|---|---|
| /2.1 | bub | bub |
| /2.2 | qr | pl |
| /2.3 | pl | **lppl** (master) |
| /2.4 | lppl | linppl |
| /2.5 | lp2 | hybppl |
| /2.6 | lp3 | ef |
| /2.7 | lp4 | u1 |
| /2.8 | lppl_w | mc |
| /2.9 | lp2_w | — |
| /2.10 | lp3_w | — |
| /2.11 | lp4_w | — |
| /2.12 | lp4_n13 | — |
| /2.13 | lp4_w_n13 | — |
| /2.14 | linppl | — |
| /2.15 | hybppl | — |
| /2.16 | exp | — |
| /2.17 | s2f | — |
| /2.18 | ef | — |
| /2.19 | mc | — |

Notably `qr` currently occupies index 2; it's **dropped from the new pill bar** because QR is a bands-only model (not a separate heatmap). Users who want QR heatmap should continue using BM. Alternative: keep `qr` in the pill bar. **Decision point for spec review.**

Phase 2 explicitly renumbers these routes. Link-breakage accepted per user directive.

### Layer 3 — LPPL sub-config panel on Heatmap

Uses the **same `_lppl_config_panel("hm")` helper** from Phase 1 (which emits only an activate checkbox + summary + Configure-LPPL button; the actual n_freqs/weighted/no_13 controls live in the global `lppl-config-modal` created in Phase 1). Panel appears below the pill bar, above the heatmap chart controls.

`hm-lppl-activate` checkbox is the activate control (already reserved in snapshot via Phase 1). Clicking the "⚙️ Configure LPPL" button opens the same global modal users access from any other tab.

### Layer 4 — Clientside sync callbacks (heatmap-specific)

Two new clientside callbacks specific to the pill+panel interaction:

**1. LPPL pill active ↔ hm-lppl-activate:**

When `hm-active-model` changes to `"lppl"` → `hm-lppl-activate` becomes `["yes"]`.
When `hm-active-model` changes away from `"lppl"` → `hm-lppl-activate` becomes `[]`.

This is a one-way callback from `hm-active-model` (data) to `hm-lppl-activate` (value). The LPPL sub-panel on the heatmap tab has no expandable body (only activate checkbox + summary + Configure button, per the Phase 1 design).

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

`update_model_swatches` in `callbacks/charts.py` currently only rebuilds `bub-model-show` / `dca-model-show` / `ret-model-show` / `sc-model-show` options. Phase 2 needs the heatmap pill-bar swatches to update on palette change.

**Constraint:** The pills have stable Dash IDs (`hm-pill-{key}`) that are Outputs/Inputs of `_hm_pill_click` and `_hm_pill_sync` callbacks. **Rebuilding the pill bar container (`Output("hm-pill-bar", "children")`) would invalidate those callback bindings** — in-flight pill clicks would fire into orphaned components and raise component-not-found errors.

**Chosen mechanism:** per-pill `Output("hm-pill-{key}", "children")` callback that emits new `html.Span` children with updated swatch colors. Pill button shell stays mounted; only the inner span children change. One callback with N Outputs (one per pill ID), same as `update_model_swatches`.

Pseudocode:

```python
@callback(
    [Output(f"hm-pill-{k}", "children") for k in _HM_PILL_MODELS],
    Input("palette-store", "data"),
    prevent_initial_call=True,
)
def update_heatmap_pill_swatches(palette_key):
    pal = _app_ctx.PALETTES.get(palette_key, _app_ctx.PALETTES["default"])
    mc = pal.get("model_colors", _app_ctx.MODEL_TRACE_COLORS)
    return [_pill_label(k, mc) for k in _HM_PILL_MODELS]
```

Where `_pill_label(key, mc)` returns the `html.Span([swatch_span, label])` structure.

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
- `btc_web/layout/heatmap.py` — rewrite `_hm_pill_bar()`, add `_lppl_config_panel("hm")` integration below pill bar.
- `btc_web/callbacks/routing.py` — update `_HM_PILL_MODELS`, `_HM_PILL_IDS` (list shrinks), update the comment.
- `btc_web/callbacks/charts.py` — update `update_heatmap` (3 new State inputs, translation logic for `hm_model == "lppl"`), update `_hm_pill_sync` to handle new pill set, new clientside sync callbacks, add per-pill `update_heatmap_pill_swatches` callback.
- `btc_web/test_web.py` — **update** existing `TestUpdateHeatmapCallback` (add 3 new kwargs to `update_heatmap` calls); add heatmap-pill-specific new tests.

**No new files.**

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Old `/2.N` links land on wrong model | User explicitly accepted this. Documented in release notes. |
| Clientside callback race condition (pill ↔ activate checkbox) | `no_update` guards tested in unit tests. |
| MC path accidentally receives flavor key | Translation isolated to non-MC path; covered by unit test. |
| Palette-change doesn't re-style pill bar | New per-pill `update_heatmap_pill_swatches` callback (pill shells stay mounted). |
| User expects flavor choice to affect MC heatmap | Documented limitation; MC cache scope. Potential Phase 3 work. |
| `lppl-n-freqs` is global + checklist (multi-select) but heatmap is single-select | Translation picks the first checked entry (`n = (lppl_n_freqs or [3])[0]`). If a user had 1+3 both checked from the Bubble tab, the heatmap silently uses LP1. **Mitigation:** document this in the help text of the LPPL Models panel, OR add a Heatmap-only visual hint "(heatmap uses: LPPL₃)" that mirrors the chosen flavor. Nice-to-have, not blocking. |
| QR removal from pill bar | User-confirm before Phase 2 lands. If user wants QR kept, revert that one pill. |

## Success criteria

- Heatmap pill bar shows 6-8 pills (depending on MC / U₁ availability), no individual LPPL flavors.
- Clicking LPPL pill expands the LPPL Models panel.
- Changing n_freqs/weighted/no_13 in the panel updates the heatmap.
- Palette changes restyle the pill bar.
- MC heatmap unaffected by the refactor.
- All tests pass.
