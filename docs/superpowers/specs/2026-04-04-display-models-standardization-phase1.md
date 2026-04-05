# Display Models Standardization — Phase 1 Design Spec

**Date:** 2026-04-04
**Scope:** DCA / Retire / SuperCharger tabs + Bubble-tab BM collapse + Citadel S2F cleanup. Brings those tabs in line with the LPPL-master pattern already shipping on the Bubble tab.
**Out of scope:** Heatmap standardization (see Phase 2 spec).

## Goal

Unify the Display Models UX across the four multi-select chart tabs (Bubble, DCA, Retire, SuperCharger). Each tab should present the same Display Models checklist (with a single `LPPL` master entry + palette-colored swatches) and the same collapsible LPPL Models sub-config panel. The Bubble Model sub-panel gains a symmetric collapse gate.

Citadel's `cp-model-src` dropdown loses its S2F option (S2F is a demonstration-only model meant for the Bubble tab).

## Constraints

**Snapshot backward-compatibility is a hard requirement.** Existing share links must decode to the same logical state after this change. This imposes:

1. `_CHECKLIST_OPTIONS` arrays for `*-model-show` checklists are **append-only**. Existing entries (`exp`, `s2f`, `ef`, `bub`, `qr`, …) keep their bit positions even when hidden from the UI. Removing entries would shift bit indices and silently corrupt old snapshots.
2. `_SNAPSHOT_CONTROLS` is positional — new entries append to the end. No reordering.
3. `_TAB_CONTROLS` sets gain new entries; no removals.

## User stories

- **As a user on the DCA tab**, I want to see the same Display Models checklist layout I see on the Bubble tab, with one "LPPL" master entry — not a confusing mix of `LPPL`, `LPPL₂`, `LPPL₃`, `LPPL₄`, `LPPL (weighted)`, … eleven checkboxes.
- **As a user configuring the LPPL overlay**, I want the "Activate LPPL overlay" + n_freqs / weighted / no_13 controls to be available on every chart tab where LPPL can render, not just the Bubble tab.
- **As a user**, I expect toggling "Bubble Model" in Display Models on the Bubble tab to collapse/expand the Bubble Model sub-panel — symmetric with how "LPPL" drives the LPPL Models panel.
- **As a user sharing a snapshot link**, my old share URLs must keep decoding to the correct state after this change ships.

## Architecture

### Layer 1 — Shared layout helpers (`layout/common.py`)

**`_model_show_checklist(prefix, standardized=False, include_u1=False)`** — existing helper, extended.

When `standardized=True`:
- Emits a single `LPPL` master entry (swatch = LPPL₁ color `#FF6D00`) between Bubble Model and the rest of the checklist.
- Skips individual LPPL family variants (`lppl`, `lp2`, `lp3`, `lp4`, `lppl_w`, `lp2_w`, `lp3_w`, `lp4_w`, `lp4_n13`, `lp4_w_n13`).
- Skips `exp` and `s2f` from the rendered options.
- Preserves the existing option ordering otherwise (dedprioritized models last, U₁ last).

Identical to the existing `_build_bub_model_options` in `layout/bubble.py`, generalized for reuse.

Note: existing `_build_model_opts` helper lives in `callbacks/charts.py:1000` (used for palette-update rebuilds); existing `_build_bub_model_options` lives in `layout/bubble.py:16` (used for initial Bubble-tab render). Both emit the same "standardized" option set today for `bubble_mode=True`. Phase 1 will consolidate by making `_model_show_checklist(prefix, standardized=True)` in `common.py` delegate to (or share code with) `_build_model_opts`. One source of truth.

**`_lppl_config_panel(prefix)`** — new helper.

Emits a `_section_card("LPPL Models", ...)` containing:
- `{prefix}-lppl-activate` checkbox ("Activate LPPL overlay").
- `{prefix}-lppl-body` div (collapsible) with n_freqs / weighted / no_13 controls.

The config controls inside the body use **un-prefixed global IDs** (`lppl-n-freqs`, `lppl-weighted`, `lppl-no-13`) so state is shared across all tabs — editing on any tab affects every tab's LPPL overlay.

### Layer 2 — Layout updates per tab

**DCA / Retire / SuperCharger** layouts:
- Replace existing `_model_show_checklist(prefix)` call with `_model_show_checklist(prefix, standardized=True)`.
- Add `_lppl_config_panel(prefix)` card directly below the Display Models section.

**Bubble tab:**
- Existing `_build_bub_model_options` already emits the master. Keep the layout but:
  - Wrap the `_section_card("Bubble Model", ...)` body contents in a new `html.Div(id="bub-bm-body")` so it can be collapsed.
  - The existing `bub-bubble-panel` div is a separate wrapper (used for show/hide on view-mode switches); `bub-bm-body` is the collapse-by-checkbox target.

**Heatmap / Citadel:** not touched in Phase 1.

### Layer 3 — Clientside sync callbacks

Per tab in `{dca, ret, sc}` — mirror the existing `bub-lppl-*` pattern in `callbacks/charts.py`:

1. **Body collapse:** `{prefix}-lppl-activate` value → `{prefix}-lppl-body` style.
2. **Activate → model-show:** adds/removes "lppl" from `{prefix}-model-show` list. **MUST include `no_update` guard** when the target value already matches (prevents infinite loop with callback #3). `allow_duplicate=True`.
3. **model-show → activate:** mirrors "lppl" membership back to the checkbox. Idempotent write — Dash will stop the loop when the value doesn't change, but guard anyway for safety.

Plus one Bubble-specific new callback:
4. **BM collapse:** "bub" in `bub-model-show` → `bub-bm-body` style. No inverse direction (unchecking Bubble Model collapses the panel; panel doesn't toggle the checkbox).

All clientside; no server round-trips. Loop-prevention pattern:

```js
// callback #2 — activate → model-show
function(act, cur_models) {
    var want = (act && act.length) > 0;
    var models = (cur_models || []).slice();
    var has = models.indexOf('lppl') !== -1;
    if (want && !has) { models.push('lppl'); return models; }
    if (!want && has) { return models.filter(function(v){return v !== 'lppl';}); }
    return window.dash_clientside.no_update;  // no-op if already aligned
}
```

### Layer 4 — Chart callback translation

`update_dca`, `update_retire`, `update_supercharge` in `callbacks/charts.py` currently pass `model_show` through to their figure builders. Updated behavior:

```python
model_show = list(model_show or [])
if "lppl" in model_show:
    model_show = [v for v in model_show if v != "lppl"]
    _weighted = "weighted" in (lppl_weighted or [])
    _no_13 = "no13" in (lppl_no_13 or [])
    for n in (lppl_n_freqs or []):
        if n == 1: model_show.append("lppl_w" if _weighted else "lppl")
        elif n == 2: model_show.append("lp2_w" if _weighted else "lp2")
        elif n == 3 and not _no_13: model_show.append("lp3_w" if _weighted else "lp3")
        elif n == 4:
            if _no_13: model_show.append("lp4_w_n13" if _weighted else "lp4_n13")
            else: model_show.append("lp4_w" if _weighted else "lp4")
```

This is the exact same translation `update_bubble` already performs (lifted into a helper `_resolve_lppl_master` in `callbacks/charts.py` to avoid duplication).

Each callback gains three new `Input` declarations: `lppl-n-freqs`, `lppl-weighted`, `lppl-no-13`.

### Layer 5 — `update_model_swatches` palette callback

Currently emits `bubble_mode=True` options only for `bub-model-show`. Update: emit `bubble_mode=True` (renamed `standardized=True` for consistency) for DCA / Retire / SC model-show outputs as well, since their layouts now use the same option set.

Heatmap pill-bar rebuild on palette change is **not** part of Phase 1.

### Layer 6 — Cache key alignment (no changes required)

Verified: `lppl-n-freqs` / `lppl-weighted` / `lppl-no-13` are used for **translation at the callback layer** (producing a specific flavor key that goes into `active_models`). They do **not** enter the figure-params dict. The cache key is computed from the post-translation params dict (`active_models` contains the final flavor key, e.g. `"lp3_w"`), so different LPPL config selections produce different cache keys via the translated list — no new fields needed in `tab_defaults.py` or the `_quantize_params` exempt set.

(The same pattern already works correctly on the Bubble tab in its shipped form.)

### Layer 7 — `_MODEL_LABELS` audit

`figures/common.py:378` currently defines labels for `{bub, qr, pl, lppl, exp, s2f, ef, u1}`. Missing flavor keys fall through to `_MODEL_LABELS.get(m, m)` → the raw short_name (cosmetic degradation, no crash).

Phase 1 adds labels for each LPPL flavor:

```python
"lp2":        "LPPL₂",
"lp3":        "LPPL₃",
"lp4":        "LPPL₄",
"lppl_w":     "LPPL (w)",
"lp2_w":      "LPPL₂ (w)",
"lp3_w":      "LPPL₃ (w)",
"lp4_w":      "LPPL₄ (w)",
"lp4_n13":    "LPPL₄ (no ω≈13)",
"lp4_w_n13":  "LPPL₄ (w, no ω≈13)",
"linppl":     "LinPPL",   # verify — may already exist
"hybppl":     "HybPPL",   # verify — may already exist
```

### Layer 8 — Citadel S2F removal

Verified: grep finds zero `"s2f"` references in `btc_web/engines/citadel.py` or `callbacks/citadel_cb.py`. Citadel's `price_model` field (citadel_cb.py:~383) passes through to `_resolve_model` which handles arbitrary keys gracefully.

Only changes needed:
- `layout/citadel.py:245` — delete `{"label": "S2F", "value": "s2f"}` from `cp-model-src` options.
- `tab_defaults.py` — if `CITADEL["price_model"]` is `"s2f"`, switch to `"bub"`.

No engine or callback changes needed. Snapshot decoding of an old citadel snapshot with `cp-model-src="s2f"` will still route through `_resolve_model("s2f")` — the model exists in `PRICE_MODELS`, just isn't user-selectable. Safe.

### Layer 9 — Snapshot (`snapshot.py`)

**Already in place (from Bubble-tab work, don't duplicate):**
- `("lppl-n-freqs", "value")`, `("lppl-weighted", "value")`, `("lppl-no-13", "value")` already in `_SNAPSHOT_CONTROLS` as global un-prefixed entries.
- `"lppl-n-freqs": [1,2,3,4]`, `"lppl-weighted": ["weighted"]`, `"lppl-no-13": ["no13"]` already in `_CHECKLIST_OPTIONS`.
- `("bub-lppl-activate", "value")` already added in commit 80ac01d.

**Phase 1 additions** — append 4 new activate checkboxes (one reserved now for Phase 2):
```python
("dca-lppl-activate", "value"),
("ret-lppl-activate", "value"),
("sc-lppl-activate", "value"),
("hm-lppl-activate", "value"),  # reserved, wired in Phase 2
```

Add to `_CHECKLIST_OPTIONS`:
```python
"dca-lppl-activate": ["yes"],
"ret-lppl-activate": ["yes"],
"sc-lppl-activate": ["yes"],
"hm-lppl-activate": ["yes"],
```

Update `_TAB_CONTROLS` in `callbacks/routing.py`: add the 3 existing global IDs (`lppl-n-freqs`, `lppl-weighted`, `lppl-no-13`) plus the new `{prefix}-lppl-activate` to the dca / retire / supercharge / heatmap sets.

**`*-model-show` arrays in `_CHECKLIST_OPTIONS` are not modified.** Exp/S2F/individual-LPPL bits stay in the bitmask encoding even though they're no longer rendered in the UI. Old snapshot links continue to decode correctly.

## Data flow example — user activates LPPL₃ weighted on DCA tab

1. User clicks "Activate LPPL overlay" checkbox (`dca-lppl-activate`) in the LPPL Models card.
2. Clientside callback #1 expands `dca-lppl-body` (removes `display:none` style).
3. Clientside callback #2 adds `"lppl"` to `dca-model-show` value.
4. Clientside callback #3 (inverse mirror) is a no-op — activate is already ["yes"].
5. User checks `lppl-weighted` (global state) and leaves n_freqs=[3], no_13=[].
6. `update_dca` fires with `model_show=["bub","lppl"]`, `lppl_weighted=["weighted"]`, `lppl_n_freqs=[3]`, `lppl_no_13=[]`.
7. Translation helper strips `"lppl"`, appends `"lp3_w"`.
8. Figure builder renders DCA accumulation using LPPL₃ (weighted) quantile bands.

## Error handling

- If `lppl-n-freqs` is empty (`[]`), the translation loop runs 0 iterations → no LPPL flavor appended → the "lppl" master is effectively stripped with no replacement. Chart renders without LPPL trace even though master is checked. UX: user can open the panel, see no flavor selected, pick one.
- If `_resolve_model(flavor_key)` fails in the figure builder (e.g., `"lp4_w_n13"` not in `PRICE_MODELS`), the figure falls through existing gracefully-ignore path (`figures/common.py` logs and skips).
- If an old snapshot link injects a specific-flavor key directly into `dca-model-show` (e.g., `"lp3"`), the chart still renders that flavor correctly — the translation only acts on `"lppl"` master. This is the snapshot-compat path.

## Testing

### Unit tests (pytest)

- `test_web.py` gets ~10 new tests:
  - `test_model_show_checklist_standardized` — verify helper emits LPPL master + skips variants.
  - `test_lppl_config_panel` — verify panel structure and control IDs.
  - `test_resolve_lppl_master` — translation helper with 1/2/3/4 freqs × weighted/unweighted × no_13/with_13 matrix.
  - `test_update_dca_with_lppl_master` — callback integration: pass master, assert flavor key appears.
  - `test_update_retire_with_lppl_master` — same.
  - `test_update_supercharge_with_lppl_master` — same.
  - `test_snapshot_backward_compat_old_link` — decode an old `q3:…` link with exp/s2f bits set, verify no crash.
  - `test_bm_collapse_clientside_registered` — verify clientside callback for `bub-bm-body` exists.
  - `test_citadel_s2f_removed` — verify `cp-model-src` options no longer include s2f.

### Manual / Playwright verification

- Visit Bubble/DCA/Retire/SC tabs, verify Display Models shows LPPL master entry.
- Click "Activate LPPL overlay" on each tab → verify LPPL body expands and "lppl" ends up in Display Models.
- Uncheck LPPL on each tab → verify body collapses.
- Toggle Bubble Model on Bubble tab → BM panel collapses/expands.
- Change palette → swatches update on all tabs without reintroducing individual LPPL variants.
- Visit Citadel → confirm S2F option gone from dropdown.
- Decode an old snapshot link with `lppl` checked + LPPL₃ selected → verify renders correctly.

## File list

**Modified:**
- `btc_web/layout/common.py` — extend `_model_show_checklist`, add `_lppl_config_panel`, consolidate with `_build_model_opts`.
- `btc_web/layout/bubble.py` — use `_lppl_config_panel` helper, wrap BM panel in `bub-bm-body` div. Refactor `_build_bub_model_options` to delegate to shared helper.
- `btc_web/layout/sim_tabs.py` — DCA/Retire layout updates (uses `_model_show_checklist` at sim_tabs.py:38).
- `btc_web/layout/supercharge.py` — SC layout updates (uses `_model_show_checklist` at supercharge.py:91).
- `btc_web/layout/citadel.py` — remove S2F option from `cp-model-src` (line 245).
- `btc_web/callbacks/charts.py` — add `_resolve_lppl_master` helper, update 3 callbacks (new Inputs + translation), update `update_model_swatches` to pass `standardized=True` for dca/ret/sc, add clientside callbacks (3 per tab × 3 tabs + 1 BM collapse).
- `btc_web/callbacks/routing.py` — extend `_TAB_CONTROLS` sets.
- `btc_web/snapshot.py` — append new `*-lppl-activate` entries to `_SNAPSHOT_CONTROLS` + `_CHECKLIST_OPTIONS`.
- `btc_web/tab_defaults.py` — switch `CITADEL["price_model"]` from `"s2f"` to `"bub"` if currently `"s2f"`.
- `btc_web/figures/common.py` — fill `_MODEL_LABELS` gaps for LPPL flavor keys.
- `btc_web/test_web.py` — **update** existing `TestUpdateDcaCallback`, `TestUpdateRetireCallback`, `TestUpdateSuperchargeCallback` (add 3 new kwargs to each `update_*` call — signatures grew by 3 Inputs); **add** new unit tests for helpers + translation.

**Not modified (contrary to earlier draft):**
- `btc_web/engines/citadel.py` — zero s2f references.
- `btc_web/callbacks/citadel_cb.py` — zero s2f references.
- `btc_web/utils.py` — `_quantize_params` exempt list fine as-is (lppl config doesn't enter params dict).

**No new files created.**

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Snapshot backward-compat break | `_CHECKLIST_OPTIONS` arrays treated as append-only; deprecated entries stay in place. Explicit test `test_snapshot_backward_compat_old_link`. |
| Prewarm cache-key drift after adding lppl config inputs | `tab_defaults.py` updated to include new fields; `_quantize_params` exempt list updated. Verified via existing cache-hit test. |
| Citadel S2F removal breaks engine code | Grep + graceful fallback; switch citadel default model if needed. |
| LPPL flavor legend labels missing on DCA/Retire/SC | Pre-audit `_MODEL_LABELS` and fill gaps as part of this spec. |
| State leakage — global LPPL config across tabs | Documented behavior; matches Q2 user decision. Not a bug, a feature. |

## Success criteria

- Bubble / DCA / Retire / SC tabs all render the same Display Models checklist UX (same entries, same swatches, same layout).
- Each of the 4 tabs has a working LPPL Models sub-panel that gates an LPPL overlay on its chart.
- Bubble Model sub-panel on the Bubble tab collapses/expands with its Display Models checkbox.
- Citadel's price model dropdown no longer shows S2F.
- All 830+ existing tests pass; ~10 new tests added and passing.
- Decoding any pre-existing snapshot link produces the same rendered state as before the change.
