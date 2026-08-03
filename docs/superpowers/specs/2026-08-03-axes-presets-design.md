# One-Tap Axes Presets (Tab 1)

**Date**: 2026-08-03
**Branch**: `time-basis-toggle-phase2b`
**Status**: Design — pending implementation plan

---

## 1. Goal

Add a row of one-tap preset buttons to the **Axes & Range** panel on Tab 1
(Price & Model Overlays) so common axis configurations are reachable without
dragging two range sliders.

Ship **two** presets, designed so that reaching 4–8 is a one-line registry
addition:

| Preset | Effect |
|---|---|
| **Default** | Return the axes to what the page loaded with — the share-link URL's values if the page came from one, system defaults otherwise. |
| **Current year** | X range = `[current_year, current_year + 1]` (today: `[2026, 2027]`). |

## 2. Scope

**In scope**

- Tab 1 only (`bub-*` controls).
- The five Axes & Range controls: `bub-xscale`, `bub-yscale`, `bub-xrange`,
  `bub-auto-y`, `bub-yrange`.
- A registry + wiring shape that scales to 4–8 presets without redesign.

**Out of scope (explicit non-goals)**

- Presets on tabs 2–10.
- User-defined / saved presets.
- Any change to `snapshot.py`, `snapshot_defaults.py`, or the defaults
  fingerprint registry.
- Collapsing the two-redraw behaviour of an X-range change (§8).
- Fixing the pre-existing lazy-tab "nonexistent object" orphan class.

## 3. Decisions

Each row records a decision made during brainstorming and why, so the
implementation does not silently relitigate it.

| # | Decision | Rationale |
|---|---|---|
| D1 | **Per-preset field ownership.** A preset writes only the fields it declares; every other field returns `no_update`. | "Default" means all five; "Current year" means X only. Forcing every preset to specify all five would make each new preset decide things it does not care about. |
| D2 | **Auto-Y off ⇒ "Current year" leaves Y alone.** | Unchecking auto-Y is an explicit claim on manual Y control. Overriding it (or silently re-checking the box) would read as a bug. |
| D3 | **One clientside callback *per* preset.** | `callbacks/plot_appearance.py:22-28` documents a Dash 4.0 bug: a clientside callback combining `allow_duplicate=True` + **multiple Inputs** + `prevent_initial_call` silently fails to fire. A shared multi-button callback would land exactly in that shape. |
| D4 | **Uniform five Outputs for every preset.** | Makes D1 structural rather than conventional, and makes registration a loop. |
| D5 | **CTA limitation accepted, not mitigated.** | See §7.1. Disabling the row under CTA would also disable "Current year", which works correctly there. |
| D6 | **"Default" is view-aware, but only in its fallback.** | See §6.2. |

## 4. Placement

Appended as the last children of `_section_card("Axes & Range", …)` in
`btc_web/layout/bubble.py`, after the `bub-yrange-wrap` Div.

`_section_card(title, *children, …)` takes varargs and spreads them into the
CardBody (`btc_web/layout/common.py:242, 270-273`), so this is a plain append.

No conditional layout logic is required. `bub-yrange-wrap` is shown/hidden by
an existing clientside callback bound to the auto-Y checkbox
(`btc_web/callbacks/charts/__init__.py:845-853`), so the preset row sits:

- directly under the **Auto** checkbox when auto-Y is checked (Y slider hidden), and
- under the **Y range** slider when auto-Y is unchecked.

Markup:

```python
_lbl("Presets"),
html.Div(
    className="d-flex flex-wrap gap-1",
    children=[
        dbc.Button(p["label"], id=f"bub-axes-preset-{p['key']}",
                   size="sm", color="secondary", outline=True,
                   className="flex-fill")
        for p in AXES_PRESETS
    ],
),
```

Style follows the existing `{prefix}-plot-appearance-reset` button
(`btc_web/layout/common.py:335-341`). A wrapping flex row is used instead of
that button's `width: 100%` because full-width stacking does not scale past
about three presets.

## 5. Registry

Single source of truth in `btc_web/layout/bubble.py`, imported by the
callbacks module. This mirrors the documented pattern for
`_HM_PILL_MODELS_BASE` / `_HM_PILL_LABELS` (layout owns the list, callbacks
import it) and the existing `callbacks → layout` import direction
(`btc_web/callbacks/charts/__init__.py:62`).

```python
AXES_PRESETS = (
    {"key": "default",  "label": "Default",      "js": _JS_DEFAULT},
    {"key": "cur_year", "label": "Current year", "js": _JS_CUR_YEAR},
)
```

Adding a preset = one tuple entry + its JS body. Button id is
`f"bub-axes-preset-{key}"`.

## 6. Wiring

New module `btc_web/callbacks/axes_presets.py` loops over `AXES_PRESETS` and
registers one clientside callback each:

```
Input  : bub-axes-preset-{key}.n_clicks          ← exactly ONE Input (D3)
State  : snapshot-state-store.data               ← "default" preset only
State  : bub-view-mode.data                      ← "default" preset only
Output : bub-xscale.value   (allow_duplicate=True)
Output : bub-yscale.value   (allow_duplicate=True)
Output : bub-xrange.value   (allow_duplicate=True)
Output : bub-yrange.value   (allow_duplicate=True)
Output : bub-auto-y.value   (allow_duplicate=True)
prevent_initial_call=True
```

System defaults are read from `SNAPSHOT_DEFAULTS` and `json.dumps`-interpolated
into each JS body at import time. No extra `dcc.Store` and no extra State: the
values are static per deploy, and reading them from the SSOT at import keeps
them from drifting.

**`btc_web/callbacks/__init__.py` must gain `import callbacks.axes_presets  # noqa: F401`.**
Callback registration in this app is import-driven (`callbacks/__init__.py:4-39`).
Omitting this line yields buttons that render and silently do nothing — the
same failure signature as the D3 bug, which is why §9 tests for it statically.

### 6.1 "Current year"

```js
function(n) {
    var NU = window.dash_clientside.no_update;
    if (!n) return [NU, NU, NU, NU, NU];
    var y = new Date().getFullYear();
    // clamp to the bub-xrange slider bounds (2010..2080)
    y = Math.max(2010, Math.min(y, 2079));
    return [NU, NU, [y, y + 1], NU, NU];
}
```

The year is computed in JS, not baked at layout time, so a tab left open
across New Year's still does the right thing.

### 6.2 "Default"

For each of the five fields, in order of precedence:

1. **URL value** — `snapshot-state-store["<id>:value"]`, if the key is present
   and not `null`.
2. **System default** — the baked `SNAPSHOT_DEFAULTS` value, except for
   `bub-xrange`, which is view-aware: `[2025, 2050]` when
   `bub-view-mode == "cagr"`, otherwise `[2010, 2033]`.

View-awareness applies **only at step 2**. A share link that carried
`bub-xrange = [2015, 2040]` restores that value in every view; the CAGR
default is used only when there is no URL value to restore. This keeps the
URL requirement primary.

The CAGR figure is `[2025, 2050]` because `toggle_bub_view` swaps to it when
entering CAGR view (`btc_web/callbacks/charts/__init__.py:547`); in that view
the X slider means *exit years*, so the price-view `[2010, 2033]` would
restore a range of exit years in the past.

**Null-check requirement.** The presence test must be
`v !== undefined && v !== null` — never `v || fallback`. `bub-auto-y`'s
legitimate "off" value is `[]`, which is falsy; a truthiness test would
silently re-enable auto-Y.

**Why the null fallback exists.** Current `q4` links do *not* store nulls:
`_decode_snapshot_v4` backfills every control in `_SNAPSHOT_CONTROLS` with
registry defaults for keys absent from the sparse diff
(`btc_web/snapshot.py:849-861`). The fallback is still required for legacy
`q2`/`q3` links, where nulls are skipped at decode
(`btc_web/snapshot.py:650-651`) and keys are genuinely absent, and for any
registry default that is itself `None`. Do not remove it as dead code.

### 6.3 Why `snapshot-state-store` is the right source

It is written only by `restore_from_url` on a successful hash decode
(`btc_web/callbacks/snapshot_cb.py:44-57`), returns `no_update` for an empty
hash (`:66-67`), initialises to `None`
(`btc_web/layout/__init__.py:362`), is never cleared, and is overwritten only
by decoding a new share hash. It is the same store `apply_tab_bubble` reads
when it writes the controls at load (`btc_web/callbacks/snapshot_cb.py:273, 285`),
so "Default" is consistent with page-load state by construction.

It is a top-level Store, not lazy-tab content, so it is mounted regardless of
which tab the user landed on.

## 7. Known limitations (accepted, documented)

### 7.1 Custom Time Axis

With `cta-active` set, `update_bubble` raises `PreventUpdate` because the CTA
callback owns `bubble-graph.figure`
(`btc_web/callbacks/charts/__init__.py:252-253`), and the CTA fit reads
`bub-xrange` but not the scales or Y range. Therefore, under CTA:

- **Current year** works normally.
- **Default** moves the X window; its scale/auto-Y/Y-range writes land in the
  controls but do not reach the visible chart until CTA is switched off.

Accepted per D5.

### 7.2 Auto-Y wins when auto-Y is on

After "Default" writes `bub-xrange`, the existing `auto_bubble_yrange`
clientside callback fires and may overwrite the restored `bub-yrange`. Its
skip-if-unchanged guard (`btc_web/callbacks/charts/__init__.py:823-828`) only
no-ops when the restored value equals the freshly computed envelope after
0.1-rounding, and that envelope depends on `AUTO_Y_GRID` (rebuilt with the
daily model) and on `bub-model-show` / `bub-stack` / `bub-show-stack`, none of
which "Default" restores.

The rule to document and test is therefore **"auto-Y recompute wins whenever
auto-Y is on"**, not "the restored Y range is preserved verbatim". This fails
safe: with auto-Y on, the computed envelope *is* the correct Y range, and the
Y slider is hidden anyway.

### 7.3 Preset tapped during post-restore settling

`update_bubble` returns all-`no_update` while `snapshot-pending` is truthy
(`btc_web/callbacks/charts/__init__.py:210-213`). A preset tapped inside that
window (released by `apply_tab_bubble`, or by the 4 s safety timer at
`btc_web/callbacks/snapshot_cb.py:697-719`) updates the controls but does not
refire the chart until the next interaction. Narrow; not mitigated.

The other post-restore gate — the `active-chart-committed` short-circuit at
`btc_web/callbacks/charts/__init__.py:239-243` — does **not** swallow preset
clicks: the document-level mousedown/keydown capture listener
(`btc_web/callbacks/snapshot_cb.py:1171-1197`) clears it before `n_clicks`
dispatches.

### 7.4 Restored Y range can exceed the slider max

`update_yrange_slider_limits` sets `bub-yrange.max` to 9 or 20 depending on
`bub-model-show` (`btc_web/callbacks/charts/__init__.py:860-870`). "Default"
restores `bub-yrange` without restoring `bub-model-show`, so a link made with
S2F/Exp active can restore a Y range above the current max. The chart honours
it; the slider UI looks inconsistent until dragged. Cosmetic.

## 8. Redraw cost

- **Default** writes five controls in one clientside return → one update wave.
- **Current year** with auto-Y on causes two chart renders: the X-range change,
  then `auto_bubble_yrange`'s Y-range write. This is identical to dragging the
  X slider today — no preset-specific regression.

Note that any axes change already drives **three** server callbacks, not one:
`update_bubble`, `update_bub_cagr` (`:625-643`), and `update_bub_resid`
(`:682-704`) all listen to these controls. That is pre-existing.

Collapsing "Current year" to a single render would require duplicating the
auto-Y envelope math into the preset path. Explicit non-goal (§2).

## 9. Testing

**pytest**

1. Registry integrity: keys unique, non-empty, and label non-empty.
2. Every preset key renders a `bub-axes-preset-{key}` button inside the
   rendered Tab 1 layout.
3. **A registered callback exists for every preset button id.** This catches
   the missing `callbacks/__init__.py` import statically, so the silent no-op
   cannot reach a browser.
4. Each preset callback has exactly one Input (guards D3 against a future
   refactor that merges them).
5. Drift guard: the baked defaults equal `SNAPSHOT_DEFAULTS` for all five axis
   keys.

**Playwright E2E** (extend the `test_plot_appearance_e2e.py` pattern) — required,
not optional, because the D3 failure mode is a silent no-op that no static test
detects. Every button must actually be clicked.

6. Tap "Current year" → `bub-xrange == [yr, yr+1]` and the figure's x-range
   updates.
7. Change all five axis controls, tap "Default" → all five return to system
   defaults (asserting Y per §7.2: with auto-Y on, assert the computed
   envelope, not a literal).
8. **Load a share link with a custom X range, change it, tap "Default" → the
   link's value is restored, not the system default.** This is the acceptance
   test for the URL requirement.
9. In CAGR view with no share link, tap "Default" → `bub-xrange == [2025, 2050]`.

## 10. Files touched

| File | Change |
|---|---|
| `btc_web/layout/bubble.py` | `AXES_PRESETS` registry + JS bodies + preset row markup |
| `btc_web/callbacks/axes_presets.py` | **new** — registration loop |
| `btc_web/callbacks/__init__.py` | one import line |
| `btc_web/test_axes_presets.py` | **new** — tests 1–5 |
| `btc_web/test_axes_presets_e2e.py` | **new** — tests 6–9 |

Not touched: `snapshot.py`, `snapshot_defaults.py`,
`snapshot_defaults_registry.json`, `tab_defaults.py`.

## 11. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| The D3 Dash bug also affects one-Input-plus-State clientside callbacks, silently no-opping "Default". | High if real | **Verify first.** The exact shape is already in production three times — `hm-palette` presets (`snapshot_cb.py:609-626`), the reset loop (`plot_appearance.py:39-46`), and with a State, `dca-build-count` (`snapshot_cb.py:969-979`) — so the risk is assessed low. Confirm with a live click before building on it. Fallback: bake initial axes into JS via a bridge callback instead of a State. |
| Buttons render but nothing fires (missing import). | Medium | pytest #3. |
| `v \|\| fallback` truthiness bug silently re-enables auto-Y. | Medium | §6.2 requirement + E2E #7. |
| +2 lazy-tab "nonexistent object" console entries on `/2`–`/10` loads (the buttons do not exist there). | Low | Accepted. Pre-existing class — `plot_appearance.py` already does this for four tab prefixes. Grows with preset count; at ~8 presets, revisit the pure-JS `set_props` variant. |

## 12. Open questions

None.
