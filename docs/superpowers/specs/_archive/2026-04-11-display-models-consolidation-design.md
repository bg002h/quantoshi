# Display Models Consolidation — Design Spec (v2)

**Date:** 2026-04-11
**Revision:** v2 — rewritten after reviewer feedback. v1 had three blockers (heatmap pill-bar composition wrong, missing snapshot.py deletions, unsafe task sequencing), one cosmetic bug (palette-rebuild staleness of inline summaries), and one mis-framed task (Task 0 as "investigation" when it's really a migration).

**Scope:** Eliminate the three "mini config card" panels (`_lppl_config_panel`, `_hybppl_config_panel`, `_eppl_config_panel`) from the app. Consolidate model-family configuration behind a single pattern: **a gear icon rendered next to the visible model selector** (checklist entry or pill-bar status row), opening a global config modal. Apply to tabs 1 (Bubble), 2 (Heatmap), 3 (DCA), 4 (Retire), 5 (Supercharger).

---

## Motivation

Today three separate builders disagree on what the Display Models checklist should contain and how it's rendered, AND every model-display tab renders redundant mini config cards below its model selector. The mini cards are pure duplication:

- Activate checkbox → mirrors the checklist (or `hm-active-model` on heatmap) via ~24 clientside sync callbacks.
- ⚙ button → opens the same global modal as the bubble tab's in-checklist gear.
- "Current: X" summary → preserved by inlining into the selector label.

**Rule we're establishing:** gears live next to the visible model selector. Mini config cards do not exist anywhere in the app. `_lppl_config_panel` / `_hybppl_config_panel` / `_eppl_config_panel` get deleted from `layout/common.py`.

---

## Task 0 — Safety migration (not "investigation")

Task 0 is code-change-free but establishes facts that the rest of the plan depends on.

### Findings from the reviewer's audit

- `hm-*-activate` checkboxes are **load-bearing**, bidirectionally synced with `hm-active-model` via 6 clientside callbacks (charts.py 176–209, 301–334, 545–583).
- `hm-active-model` is **already** in `_SNAPSHOT_CONTROLS` (`snapshot.py:136`), so heatmap snapshots persist pill state directly, independent of the activate checkboxes.
- Therefore: deleting `hm-*-activate` after deleting the sync callbacks is safe **if** snapshot restoration still works via `hm-active-model` alone. This is already how it works today — the activate checkboxes are vestigial state mirrors of `hm-active-model`.

### Acceptance gate for Task 0

1. **Manual snapshot-restore test on production data.** Before Task 1 coding starts: generate a current-format `q3:` link on heatmap with LPPL active, HybPPL active, and EPPL active (try each). Decode on a throwaway local layout with `hm-*-activate` + their sync callbacks stubbed out but `hm-active-model` preserved. Confirm the correct pill becomes active and the correct figure renders.
2. **Grep the whole codebase** for references to the 15 `{prefix}-{family}-activate` ids. Expected references:

| Location | Reference count | Action |
|---|---|---|
| `btc_web/snapshot.py` | 30 entries (15 in `_SNAPSHOT_CONTROLS`, 15 in `_CHECKLIST_OPTIONS`) | Delete in Tasks 2–5 |
| `btc_web/callbacks/routing.py` | 15 entries in `_TAB_CONTROLS` (lines 128, 129, 137, 152, 153, 161, 176, 177, 185, 197, 198, 206, 219, 220, 228) | Delete in Task 5 |
| `btc_web/callbacks/charts.py` | ~30 clientside callback references (activate↔selector mirrors + configure-btn handlers) | Delete in Task 5 |
| `btc_web/layout/common.py` | 3 creator functions (`_lppl_config_panel`, `_hybppl_config_panel`, `_eppl_config_panel`) | Delete in Task 5 |
| `btc_web/layout/bubble.py` | 3 call sites (lines 207–209) | Delete in Task 2 (single commit) |
| `btc_web/layout/sim_tabs.py` | 3 call sites (lines 53–55) | Delete in Task 3 (single commit) |
| `btc_web/layout/supercharge.py` | 3 call sites (lines 107–109) | Delete in Task 3 (single commit) |
| `btc_web/layout/heatmap.py` | 3 call sites (lines 126–128) | Delete in Task 4 (single commit) |
| `btc_web/test_models.py` | Assertions at lines ~1668–1670, ~2120–2122, ~2360–2362 | Update in Task 6 |
| `btc_web/test_web.py`, `test_callbacks.py`, `test_figures.py` | TBD — grep during Task 0 | Update in Task 6 |

3. **Document snapshot positional stability + hidden-placeholder requirement.** `snapshot.py::_decode_snapshot` is **positional** (`zip(_SNAPSHOT_CONTROLS, values)` at line 515), not key-based. Each tuple in `_SNAPSHOT_CONTROLS` occupies a fixed bit-index in the encoded `q3:` payload — removing tuples would shift every downstream position, silently corrupting all pre-refactor share links. **Therefore the 15 `_SNAPSHOT_CONTROLS` tuples and 15 `_CHECKLIST_OPTIONS` keys STAY** as defunct placeholders.

   **However:** `callbacks/snapshot_cb.py:58` statically registers `*[Output(cid, prop, allow_duplicate=True) for cid, prop in _SNAPSHOT_CONTROLS]` at callback import time. Dash raises `ComponentIDNotFound` at registration if any `cid` is missing from the layout. So keeping the tuples requires the component ids to still exist somewhere in the DOM — even if hidden and dead.

   **Solution:** `_serve_layout` renders a hidden placeholder block **unconditionally at the root of the returned layout tree** — NOT inside any flag-gated branch (not inside `if _HAS_MARKOV`, not inside a tab filter, not inside URL-based routing branches). Hot-reload under `DEV=1` re-runs `_serve_layout` on every request; if any branch ever returns a layout missing the placeholder ids, the next callback fire raises `ComponentIDNotFound`. Unconditional emission is the only safe pattern.
   ```python
   html.Div(
       id="_defunct-snapshot-placeholders",
       style={"display": "none"},
       children=[
           dcc.Checklist(id=f"{prefix}-{family}-activate",
                         options=[{"label": "", "value": "yes"}],
                         value=[])
           for prefix in ("bub", "dca", "ret", "sc", "hm")
           for family in ("lppl", "hybppl", "eppl")
       ],
   )
   ```
   15 dead `dcc.Checklist` components with no visible label, no value updates, no callbacks. Pure scaffolding to satisfy the snapshot callback's static Output list. Add an explanatory comment: "these are vestigial — deleted from the UI during display-models consolidation but retained for q3: share link positional stability."

   Old `q3:` links with `["yes"]` at those positions decode → `restore_from_url` writes to the hidden placeholder → no visible effect. New links write `[]` to them (default). Positional stability preserved without logic changes. Add regression test: "a pre-refactor `q3:` link with all 15 `*-activate` keys set to `['yes']` decodes without error on the new layout, does not raise, and restores `hm-active-model` via its own snapshot entry."

Task 0 deliverable: a one-page report confirming (1), (2), (3). No code changes.

---

## Architecture

### File structure

| File | Action | Purpose |
|---|---|---|
| `btc_web/layout/display_models.py` | **Create** | New home for the shared checklist component. Exports `display_models_panel(prefix, ...)` + `build_display_models_options(mc, ..., family_state=None)` |
| `btc_web/layout/common.py` | Modify | **Delete** `_lppl_config_panel`, `_hybppl_config_panel`, `_eppl_config_panel`, `_model_show_checklist`. **Move** `_GEAR_STYLE` + a new `_MUTED_STYLE` dict from bubble.py to common.py so display_models.py + heatmap.py can import. Keep `_global_lppl_modal`, `_global_hybppl_modal`, `_global_eppl_modal` |
| `btc_web/layout/bubble.py` | Modify | Delete `_build_bub_model_options`; call `display_models_panel("bub", include_bm_master=True)`. Drop `_lppl_config_panel("bub")` / `_hybppl_config_panel("bub")` / `_eppl_config_panel("bub")`. **Keep** `bub-bm-body` collapse + `bub-bm-activate` checkbox (these belong to the Bubble Model *primary* settings card, not a mini config card — see "BM carve-out" below) |
| `btc_web/layout/sim_tabs.py` | Modify | Replace `_model_show_checklist(prefix, …)` with `display_models_panel(prefix, include_mc=…)`. Drop the three `_*_config_panel(prefix)` calls |
| `btc_web/layout/supercharge.py` | Modify | Same pattern as sim_tabs.py for `sc` prefix |
| `btc_web/layout/heatmap.py` | Modify | Add a status row **below** the pill bar containing the gear + inline summary for the currently-active configurable family. Drop all three `_*_config_panel("hm")` calls. Pill bar itself unchanged — same flat ids, same `dbc.ButtonGroup` |
| `btc_web/callbacks/charts.py` | Modify | Delete `_build_model_opts`. Rewrite `update_model_swatches` to call `build_display_models_options` with modal state as State inputs (palette-rebuild staleness fix — see below). Simplify modal-open Input lists to pure gear IDs (one per tab: bub/dca/ret/sc/hm). Delete activate↔selector mirror callbacks for bub/dca/ret/sc/hm. Rewire summary update callbacks to Output to new `*-summary-inline` span ids |
| `btc_web/snapshot.py` | **Unchanged (content)** / comment-only | The 15 `_SNAPSHOT_CONTROLS` tuples (line 129; 284–287; 295; 309–312; 314; 328–331) and 15 `_CHECKLIST_OPTIONS` keys (391–409) for `{prefix}-{family}-activate` **stay** for positional stability (decoder at line 515 is positional, not key-based — see Task 0 finding 3). Add an inline comment at each block explaining "defunct after display-models consolidation; kept for q3: link positional stability." No logic changes |
| `btc_web/app.py` or layout root | Modify | Add a hidden `html.Div(id="_defunct-snapshot-placeholders", style={"display":"none"})` containing 15 dead `dcc.Checklist` components for the retained `{prefix}-{family}-activate` ids. Pure scaffolding to keep `callbacks/snapshot_cb.py:58` callback registration happy. See Task 0 finding 3 for rationale |
| `btc_web/callbacks/routing.py` | Modify | Delete 15 `_TAB_CONTROLS` entries for `{prefix}-{family}-activate` (lines 128/129/137, 152/153/161, 176/177/185, 197/198/206, 219/220/228) |
| `btc_web/layout/citadel.py` | **Unchanged** | Citadel is out of scope |
| `btc_web/test_palette_roundtrip.py` | Modify | Extend per "Test plan" below |
| `btc_web/test_models.py`, `test_web.py`, `test_callbacks.py`, `test_figures.py` | Modify | Update/delete assertions that reference deleted ids (audited in Task 0) |

### BM carve-out (explicit)

The Bubble Model has **two distinct activate checkboxes in the current layout**, and only one of them is being deleted:

- `bub-bm-activate` (`layout/bubble.py:200`) — lives in the header of the **Bubble Model primary settings card** (containing Composite/Support toggles + `N future bubbles` slider in `bub-bm-body`). **Survives.** It's the show/hide toggle for the collapse body of the main BM card, not a mini config card satellite. The `bub-bm-gear` in the Display Models master entry scrolls to `bub-bm-body` as a convenience.
- `bub-bm-activate` ↔ `bub-model-show` mirror callbacks at `charts.py:82-113` — **survive** for the same reason.
- No other tab has `*-bm-activate` — sim/heatmap tabs never had a BM mini card or BM master entry.

**Updated deletion rule:** "Delete all `{prefix}-{family}-activate` mirror callbacks for `family ∈ {lppl, hybppl, eppl}`." `bm` is excluded.

### Shared component signature

```python
# btc_web/layout/display_models.py
def display_models_panel(
    prefix: str,
    *,
    include_bm_master: bool = False,
    include_mc: bool = False,
    include_u1: bool = True,
):
    """Return the Display Models section_card for one checklist-style tab.

    Parameters
    ----------
    prefix : {"bub", "dca", "ret", "sc"}
        Used for per-tab component IDs: {prefix}-model-show checklist,
        {prefix}-{family}-gear button ids (family ∈ {lppl, hybppl, eppl}),
        and {prefix}-{family}-summary-inline spans inside the checklist labels.
    include_bm_master : bool
        Bubble only. When True, emits "Bubble Model" as a master entry
        with a gear icon (id bub-bm-gear) that scrolls to bub-bm-body.
        Sim tabs pass False and get a plain "Bubble Model" checklist entry
        with no gear.
    include_mc : bool
        Retire tab passes True to include the MC option in the list.
    include_u1 : bool
        Usually True. Exposes the User Model (U₁) entry.
    """
```

```python
def build_display_models_options(
    mc: dict,
    *,
    include_bm_master: bool = False,
    include_mc: bool = False,
    include_u1: bool = True,
    lppl_state: tuple | None = None,
    hybppl_state: tuple | None = None,
    eppl_state: tuple | None = None,
) -> list[dict]:
    """Pure function. Builds the checklist options list for one tab.

    Used by:
      1. display_models_panel (initial layout — state args default to None,
         in which case the inline summary defaults to the static label like
         "LPPL₃" and will be overwritten by the summary callback on its
         first fire)
      2. callbacks/charts.py::update_model_swatches (palette rebuild —
         state args ARE passed so the rebuilt option labels carry the
         current modal state, not the static default)
    """
```

### Palette-rebuild staleness fix — via `dcc.Store("display-model-summaries")`

The reviewer flagged: when `update_model_swatches` returns a fresh `options` list on palette change, Dash replaces every option label. Spans with id `{prefix}-lppl-summary-inline` get re-created with static `children` from the builder. The summary-update callbacks don't re-fire (their inputs are `lppl-n-freqs` / `lppl-weighted` / `lppl-no-13`, unchanged by palette). Result: inline summary reverts to default until the user toggles a modal control.

**Fix: a single source of truth for family summaries.** Add `dcc.Store(id="display-model-summaries", storage_type="memory", data={"lppl": "LPPL₃", "hybppl": "1d+1u", "eppl": "1d+1u"})` to the layout root. A single computation callback populates it:

```python
@callback(
    Output("display-model-summaries", "data"),
    # LPPL (3 inputs)
    Input("lppl-n-freqs", "value"),
    Input("lppl-weighted", "value"),
    Input("lppl-no-13", "value"),
    # HybPPL (13 inputs — slot-A 6 knobs + slot-B 6 knobs + slot-B enable)
    Input("hybppl-cfg-a-nlog", "value"),
    Input("hybppl-cfg-a-ncal", "value"),
    Input("hybppl-cfg-a-log1d", "value"),
    Input("hybppl-cfg-a-log2d", "value"),
    Input("hybppl-cfg-a-cal1d", "value"),
    Input("hybppl-cfg-a-cal2d", "value"),
    Input("hybppl-cfg-b-enabled", "value"),
    Input("hybppl-cfg-b-nlog", "value"),
    Input("hybppl-cfg-b-ncal", "value"),
    Input("hybppl-cfg-b-log1d", "value"),
    Input("hybppl-cfg-b-log2d", "value"),
    Input("hybppl-cfg-b-cal1d", "value"),
    Input("hybppl-cfg-b-cal2d", "value"),
    # EPPL (13 inputs — same structure as HybPPL)
    Input("eppl-cfg-a-nlog", "value"),
    Input("eppl-cfg-a-ncal", "value"),
    Input("eppl-cfg-a-log1d", "value"),
    Input("eppl-cfg-a-log2d", "value"),
    Input("eppl-cfg-a-cal1d", "value"),
    Input("eppl-cfg-a-cal2d", "value"),
    Input("eppl-cfg-b-enabled", "value"),
    Input("eppl-cfg-b-nlog", "value"),
    Input("eppl-cfg-b-ncal", "value"),
    Input("eppl-cfg-b-log1d", "value"),
    Input("eppl-cfg-b-log2d", "value"),
    Input("eppl-cfg-b-cal1d", "value"),
    Input("eppl-cfg-b-cal2d", "value"),
    # Total: 3 + 13 + 13 = 29 inputs.
)
def compute_family_summaries(*args):
    return {
        "lppl":   _build_lppl_summary(*args[0:3]),
        "hybppl": _build_hybppl_summary(*args[3:16]),
        "eppl":   _build_eppl_summary(*args[16:29]),
    }
```

Exact input count pinned at **29** (3 LPPL + 13 HybPPL + 13 EPPL). If this count changes during implementation because the HybPPL/EPPL panels gain/lose a knob, update the Store-writer's slice boundaries.

**Three consumers read the Store:**

1. **Live update of inline spans (checklist tabs)** — one clientside callback per family:
   ```python
   # Writes `children` of the 4 tab-specific summary spans when the store updates
   Output("bub-lppl-summary-inline", "children"),
   Output("dca-lppl-summary-inline", "children"),
   Output("ret-lppl-summary-inline", "children"),
   Output("sc-lppl-summary-inline", "children"),
   Input("display-model-summaries", "data"),
   # return [data.lppl, data.lppl, data.lppl, data.lppl]
   ```

2. **Palette rebuild (`update_model_swatches`)** — takes the Store as a **single** `State` instead of ~24 modal state args:
   ```python
   @callback(
       Output("bub-model-show", "options"),
       Output("dca-model-show", "options", allow_duplicate=True),
       Output("ret-model-show", "options", allow_duplicate=True),
       Output("sc-model-show", "options", allow_duplicate=True),
       Input("palette-store", "data"),
       State("display-model-summaries", "data"),
       prevent_initial_call=True,
   )
   def update_model_swatches(palette_key, summaries):
       ...
       return (
           build_display_models_options(mc, include_bm_master=True, summaries=summaries),
           build_display_models_options(mc, summaries=summaries),
           build_display_models_options(mc, summaries=summaries, include_mc=True),
           build_display_models_options(mc, summaries=summaries),
       )
   ```
   The rebuilt options carry the correct inline summary text baked into the label tree.

3. **Heatmap status row** — a clientside callback reads the Store + `hm-active-model` and sets `hm-active-family-summary-inline.children`:
   ```python
   function(summaries, active) {
       if (!summaries || !active) return "";
       return summaries[active] || "";
   }
   Output("hm-active-family-summary-inline", "children"),
   Input("display-model-summaries", "data"),
   Input("hm-active-model", "data"),
   ```

**Why the Store approach beats ~24 State args:**
- `update_model_swatches` has ONE extra State instead of 24.
- `build_display_models_options(summaries=...)` takes one dict kwarg instead of three tuple kwargs.
- Heatmap status row gets its summary source for free — no dedicated callback needed.
- The summary computation lives in one place (the Store-writer callback), not duplicated between the palette-rebuild path and the live-update path.

**What replaces the old `update_*_summary` callbacks** (which currently write to `{prefix}-lppl-summary` on the mini cards): one Store-writer callback (29 Inputs, 1 Output) + three clientside Store-reader callbacks (one per family, 4 Outputs each). Simpler overall than the current fan-out pattern.

**Alternative considered and rejected:** chain the summary callback to also re-fire on `{prefix}-model-show.options` change. Rejected because Dash evaluates Outputs → Inputs in graph order and chaining options → summary → options creates a re-entrant loop that Dash rejects at registration.

### Inline summary spans (checklist tabs)

The checklist label for each master entry has the shape:

```python
html.Span([
    html.Span(" ", style=_swatch_style(color)),   # color swatch
    "Entropy PPL",                                 # family label
    html.Span(" (", style=_MUTED_STYLE),
    html.Span(id=f"{prefix}-eppl-summary-inline",  # live summary target
              children="1d+1u",                     # static default (overwritten)
              style=_MUTED_STYLE),
    html.Span(")", style=_MUTED_STYLE),
    html.Span("⚙",
              id=f"{prefix}-eppl-gear", n_clicks=0,
              style=_GEAR_STYLE,
              title="Configure Entropy PPL"),
])
```

Verified precedent: current bubble tab already uses nested spans with ids (`bub-lppl-gear` etc.) inside Checklist option labels as callback Inputs. This spec extends the pattern to callback Outputs (`children` targeting). Dash's dependency resolution is id-based, not path-based — it finds a component by id anywhere in the layout. Smoke-test during Task 2 before wiring the rest.

### Build helper signature (updated)

```python
def build_display_models_options(
    mc: dict,
    *,
    include_bm_master: bool = False,
    include_mc: bool = False,
    include_u1: bool = True,
    summaries: dict | None = None,   # {"lppl": "...", "hybppl": "...", "eppl": "..."}
) -> list[dict]:
    """Pure function. Builds the checklist options list for one tab.

    `summaries` is optional; when None, the builder emits static defaults
    (e.g. "LPPL₃") that get overwritten by the Store-reader clientside
    callback after initial layout render. When provided (palette rebuild
    path), the inline summary spans carry the live values.
    """
```

### Heatmap — status row below pill bar

**Approach chosen: Option B — separate status row.** The pill bar itself is unchanged (same flat `hm-pill-{key}` ids, same `dbc.ButtonGroup`). A new status row renders immediately below the bar:

```python
# layout/heatmap.py inside _hm_pill_bar() or as a sibling
html.Div(id="hm-active-family-row", children=[
    html.Span("Active: ", style={"fontSize": "11px", "color": DIM_TEXT}),
    html.Span(id="hm-active-family-label",
              style={"fontWeight": "600", "fontSize": "12px"}),
    html.Span(" · (", style=_MUTED_STYLE),
    html.Span(id="hm-active-family-summary-inline",
              children="", style=_MUTED_STYLE),
    html.Span(") ", style=_MUTED_STYLE),
    html.Span("⚙", id="hm-active-family-gear", n_clicks=0,
              style=_GEAR_STYLE,
              title="Configure active model"),
], style={"display": "none"})   # hidden unless active family is configurable
```

**Visibility gating + label (clientside — single callback):** reads `hm-active-model` and sets row style + label children. The summary text is populated by a separate clientside callback reading `display-model-summaries` Store (see §Palette-rebuild staleness fix).

```python
# Visibility + label
function(active) {
    var CONFIGURABLE = {"lppl": "LPPL", "hybppl": "HybPPL", "eppl": "Entropy PPL"};
    if (!active || !(active in CONFIGURABLE)) {
        return [{display: "none"}, ""];
    }
    return [{display: "flex"}, CONFIGURABLE[active]];
}
Output("hm-active-family-row", "style"),
Output("hm-active-family-label", "children"),
Input("hm-active-model", "data"),
```

**One gear, three modals — routing via `hm-active-model` State (not a data attribute):**
```python
_app_ctx.app.clientside_callback(
    """
    function(n, active) {
        if (!n || !active) return [false, false, false];
        return [active === "lppl", active === "hybppl", active === "eppl"];
    }
    """,
    Output("lppl-config-modal",   "is_open", allow_duplicate=True),
    Output("hybppl-config-modal", "is_open", allow_duplicate=True),
    Output("eppl-config-modal",   "is_open", allow_duplicate=True),
    Input("hm-active-family-gear", "n_clicks"),
    State("hm-active-model", "data"),
    prevent_initial_call=True,   # mandatory — pairs with allow_duplicate
)
```

Simpler than the earlier data-attribute approach: the dispatcher reads the active-model Store directly at click time. No intermediate attribute hop.

**Why `prevent_initial_call=True` is critical here:** `hm-active-model` has a default value at page load (some pill is the active one at startup). Without `prevent_initial_call=True`, the dispatcher would fire on initial load with a populated `hm-active-model` and open a config modal on first visit — terrible UX. With it, the dispatcher only fires on actual `n_clicks` increments.

**Benefits of Option B over "gear inside ButtonGroup":**
- Doesn't touch the `dbc.ButtonGroup` — rounded corners, focus rings, ARIA roles all unchanged.
- Gear only visible when meaningful (active family is configurable). On BM/PL/S2F/MC the row is hidden entirely.
- Single gear across the three configurable families — simpler ID space (no `hm-lppl-gear` / `hm-hybppl-gear` / `hm-eppl-gear`, just `hm-active-family-gear`).
- Naturally displays the "Current: X" summary for the active family only, which is what users care about — the other families' summaries are irrelevant until the user switches.

**Callback consequences:**
- The three modal-open callbacks at `charts.py:117-141` (and hybppl/eppl equivalents) get the `hm-*-configure-btn` Inputs REMOVED entirely. Heatmap's modal access is via the new single clientside dispatcher, not via adding a 5th gear Input to each modal-open callback.
- The three `hm-{family}-summary` spans in the old mini cards are replaced by a single `hm-active-family-summary-inline`, driven by a new clientside callback that reads `hm-active-model` and the cached per-family summary strings.
- The three `hm-{family}-activate` checkboxes + their 6 bidirectional sync callbacks are deleted (Task 0 confirmed safe — `hm-active-model` already drives snapshot restore).

### Checklist tab modal-open callbacks

After refactor, each family's modal-open callback Input list becomes:

```python
_app_ctx.app.clientside_callback(
    """
    function(bub_n, dca_n, ret_n, sc_n, close_n, cur_open) {
        var ctx = window.dash_clientside.callback_context;
        if (!ctx.triggered || !ctx.triggered.length) return window.dash_clientside.no_update;
        var src = ctx.triggered[0].prop_id;
        if (src.indexOf('modal-close-btn') !== -1) return false;
        if (src.indexOf('-gear') !== -1) return true;
        return window.dash_clientside.no_update;
    }
    """,
    Output("lppl-config-modal", "is_open", allow_duplicate=True),   # CHANGED — allow_duplicate
    Input("bub-lppl-gear", "n_clicks"),
    Input("dca-lppl-gear", "n_clicks"),
    Input("ret-lppl-gear", "n_clicks"),
    Input("sc-lppl-gear", "n_clicks"),
    Input("lppl-modal-close-btn", "n_clicks"),
    State("lppl-config-modal", "is_open"),
    prevent_initial_call=True,   # REQUIRED alongside allow_duplicate
)
# heatmap is NOT an Input here — dispatched via hm-active-family-gear clientside callback below
```

**`allow_duplicate=True` is mandatory** on all three modal `is_open` Outputs for both this callback AND the heatmap dispatcher (see next subsection). Reason: the heatmap dispatcher also writes `lppl-config-modal.is_open`, so Dash requires every callback writing to that Output to declare `allow_duplicate=True`. Without it, Dash raises `DuplicateCallback` at registration and gunicorn fails to start.

**`prevent_initial_call=True` is mandatory** per CLAUDE.md's rule ("`allow_duplicate=True` is incompatible with `prevent_initial_call=False` — crashes gunicorn"). Applies to both callbacks.

### Ordering rule (single source of truth)

```python
# Inside build_display_models_options
_LPPL_FAM   = {"lppl", "lp2", "lp3", "lp4"} | set(_app_ctx.LPPL_FAMILY_HIDDEN_FROM_BUBBLE)
_HYBPPL_FAM = set(_app_ctx.HYBPPL_FAMILY_HIDDEN)
_PROMOTED   = ("pca", "grdy")
_DEPRIORITIZED = {"exp", "s2f", "gomp", "bpl"}

_EXCLUDED_SHORT_NAMES = (
    set(_app_ctx.MODEL_SENTINELS)
    | {"bub", "eppl"}       # master entries (emitted above)
    | _LPPL_FAM             # LPPL family → master
    | _HYBPPL_FAM           # HybPPL family → master
)

options = []
# 1. Bubble Model (with or without gear, depending on include_bm_master)
# 2. Entropy PPL (master, gear)
# 3. LPPL (master, gear)
# 4. Hybrid PPL (master, gear)
# 5. PROMOTED ordered: pca, grdy
# 6. Primary (not PROMOTED and not DEPRIORITIZED)
# 7. DEPRIORITIZED: exp, s2f, gomp, bpl
# 8. U₁ (if include_u1)
# 9. MC (if include_mc)
```

Belt-and-suspenders: explicit `startswith("cfg_")` / `startswith("ecfg_")` filter (the bug we just fixed in Approach D).

---

## What gets deleted

**Layout:**
- `layout/common.py::_model_show_checklist`
- `layout/common.py::_lppl_config_panel`
- `layout/common.py::_hybppl_config_panel`
- `layout/common.py::_eppl_config_panel`

**Callbacks:**
- `callbacks/charts.py::_build_model_opts` (replaced by shared builder)
- `callbacks/charts.py`: all `for _lp in ("dca", "ret", "sc")` activate↔checklist mirror callbacks (6 callbacks)
- `callbacks/charts.py`: bubble's separate `bub-lppl-activate` / `bub-hybppl-activate` / `bub-eppl-activate` mirror callbacks (6 callbacks — NOT including `bub-bm-activate`)
- `callbacks/charts.py`: heatmap's 6 bidirectional `hm-{family}-activate ↔ hm-active-model` sync callbacks
- `callbacks/charts.py`: all `{prefix}-{family}-configure-btn` click handlers (scroll-to-modal-open) — 15 callbacks

**Snapshot compatibility — NOT deleted, plus one layout addition:**
- The 15 `_SNAPSHOT_CONTROLS` tuples and 15 `_CHECKLIST_OPTIONS` keys **stay** in `snapshot.py`. Deleting them would shift bit indices in the positional `q3:` encoding and silently corrupt every pre-refactor share link (decoder at line 515 is `zip(_SNAPSHOT_CONTROLS, values)`).
- **Add** `html.Div(id="_defunct-snapshot-placeholders", style={"display":"none"})` containing 15 hidden `dcc.Checklist` components with ids `{prefix}-{family}-activate` for `prefix ∈ {bub, dca, ret, sc, hm}` × `family ∈ {lppl, hybppl, eppl}`. Required because `callbacks/snapshot_cb.py:58` statically registers `Output(cid, prop)` for every `_SNAPSHOT_CONTROLS` entry at import time — Dash raises `ComponentIDNotFound` if the component is missing from the layout.
- Old `q3:` links with `["yes"]` at these positions restore to the hidden placeholder → no visible effect, no error.

**Routing:**
- `callbacks/routing.py::_TAB_CONTROLS`: remove 15 `*-activate` entries at lines 128, 129, 137, 152, 153, 161, 176, 177, 185, 197, 198, 206, 219, 220, 228

**CSS (if unused post-refactor):**
- `.model-panel-activate` class
- `.model-panel-configure-btn` class

## What gets added

- `btc_web/layout/display_models.py` (~150 lines)
- `btc_web/layout/common.py`: `_GEAR_STYLE` moved from bubble.py, new `_MUTED_STYLE`
- Checklist tabs: `{prefix}-bm-gear` (bub only), `{prefix}-lppl-gear`, `{prefix}-hybppl-gear`, `{prefix}-eppl-gear` — 13 ids total
- Heatmap: `hm-active-family-row`, `hm-active-family-label`, `hm-active-family-summary-inline`, `hm-active-family-gear` — 4 ids
- Checklist tabs: 4 × 3 = 12 new `{prefix}-{family}-summary-inline` span ids
- `dcc.Store(id="display-model-summaries", storage_type="memory")` at layout root
- `html.Div(id="_defunct-snapshot-placeholders", style={"display":"none"})` at layout root containing 15 hidden `dcc.Checklist` for retained snapshot ids
- **Store-writer callback** `compute_family_summaries` (29 Inputs → 1 Output) — single source of truth for family summary strings
- **Three Store-reader clientside callbacks** (one per family) that write `{prefix}-{family}-summary-inline` children on all 4 checklist tabs
- **Clientside callback:** heatmap status-row visibility + label — reads `hm-active-model`
- **Clientside callback:** heatmap status-row summary text — reads `display-model-summaries` + `hm-active-model`
- **Clientside callback:** heatmap `hm-active-family-gear` → modal dispatcher — reads `hm-active-model` State
- **(Nothing new for `bub-bm-gear`.)** The scroll handler already exists at `charts.py:55-68` and stays unchanged — it does a single `scrollIntoView()` on `bub-bm-body`. Visibility of `bub-bm-body` is gated by a separate clientside callback (`charts.py:71-80`) that reads `bub-model-show` membership and toggles `display:none`. No "expand if collapsed" logic needed — `bub-bm-body` is an `html.Div`, not a `dbc.Collapse`.

---

## Rollout (Path A — atomic refactor commit)

v1 proposed per-tab commits. v1 was wrong: each per-tab swap deletes ids that existing callbacks still reference, so every intermediate commit would crash gunicorn at Dash callback registration. **Tasks 2–5 land as a single atomic commit.**

### Task sequence

| Task | Type | Commit strategy |
|---|---|---|
| **Task 0** | Investigation + migration planning | No commit. One-page report delivered as chat message. |
| **Task 1** | Scaffolding | Standalone commit: create `display_models.py`, move `_GEAR_STYLE` and add `_MUTED_STYLE` to `common.py`. No wiring. App still works identically. |
| **Tasks 2–5** | Refactor | **Single atomic commit** combining: bubble swap (Task 2), sim-tabs swap (Task 3), heatmap status row (Task 4), `common.py` / `snapshot.py` / `routing.py` / `charts.py` cleanup (Task 5). Large diff, but individually verifiable via the `test_palette_roundtrip.py` suite + a local smoke test before commit. |
| **Task 6** | Test audit | Standalone commit: update `test_models.py`, `test_web.py`, `test_callbacks.py`, `test_figures.py` assertions that reference deleted ids. Extend `test_palette_roundtrip.py` per "Test plan" below. |
| **Task 7** | Syntax check + local smoke test | No commit. Dev-server manual run at desktop + mobile widths. Pass/fail report. |

### Why atomic for Tasks 2–5

Each of Tasks 2/3/4/5 individually breaks callback registration:
- Task 2 alone: modal-open callbacks still have `Input("bub-lppl-configure-btn", ...)` but bubble no longer renders that id.
- Task 3 alone: same for dca/ret/sc.
- Task 4 alone: same for hm, plus the heatmap status row references ids that the modal-open callbacks don't know about yet.
- Task 5 alone (cleanup): deletes the Input references but leaves the *renderers* still emitting the old ids.

Only a simultaneous change across all five files (`bubble.py`, `sim_tabs.py`, `supercharge.py`, `heatmap.py`, `common.py` + `charts.py` + `snapshot.py` + `routing.py`) leaves the app runnable. Subagent-driven-development can still use the task boundaries as review checkpoints — each task is a *mental* milestone, not a git commit.

**Commit message for the atomic commit** should list what each sub-task accomplished, so git-blame remains useful.

---

## Risks

1. **Task 0 false-negative.** If the snapshot-restore manual test passes but a less-common code path still reads `hm-*-activate`, deletion silently breaks that path. Mitigation: grep (`btc_web/` + `archive/` + top-level `tools/`) for all 15 ids and enumerate every reference during Task 0. Expected reference set is documented above.
2. **Nested-id callback Output support.** We're targeting `{prefix}-{family}-summary-inline` span ids inside Checklist option labels. This is a callback Output pattern, not an Input pattern. The current codebase has Inputs on nested ids (gear buttons), which proves Dash finds them by id. Outputs *should* work the same way but **smoke-test this explicitly in Task 2** before wiring Tasks 3–5. If it fails, fall back: store the summary strings in a `dcc.Store` and have each tab's checklist wrap its label tree in a layout callback that reads the store (more complexity, but works).
3. **Palette-rebuild re-entry.** `update_model_swatches` now has ~24 State inputs. Dash allows this but callback arg tuples get unwieldy. Use a named helper to unpack into `lppl_state`, `hybppl_state`, `eppl_state` structs.
4. **Stale callback graph in browsers on deploy.** `Cache-Control: no-cache` on `/_dash-dependencies` already handles this per CLAUDE.md.
5. **Old `q3:` snapshot backward-compat.** Decoder must silently ignore the 15 deleted keys in old links. Verify `snapshot.py::_decode_snapshot` uses `.get()`. Add regression test: a synthetic old `q3:` link with all 15 `*-activate` keys set to `["yes"]` must decode without error on the new layout and must restore `hm-active-model` via its own snapshot entry.
6. **Visual regression.** Mini cards disappearing shortens each tab's control column. Alignment of cards below (Chart Settings, Plot Appearance) may shift. Manual smoke test each tab at desktop + mobile widths. Heatmap status row must wrap cleanly on narrow viewports.
7. **Heatmap: gear click while transitioning pills.** User clicks LPPL pill then immediately clicks the gear while the pill-sync callback is still running — could open the wrong modal. Mitigation: `hm-active-family-gear` reads `hm-active-model` at click time, not at render time. Current pattern already handles this because `hm-active-model` is a Store — always current.
8. **Task 2 smoke-test must precede Task 3–5 work.** If nested-id Outputs don't work (Risk 2), the whole design collapses and we revert to mini-card visibility gating. Make this an explicit gate in the plan.
9. **Palette + modal rapid succession race (hypothetical, low-probability).** User changes palette in tick T; in the same tick toggles `lppl-weighted`. `update_model_swatches` rebuilds options with `summaries=state_A` (pre-toggle); `compute_family_summaries` writes `state_B` (post-toggle) to the store. The Store-reader clientside callback then writes `state_B` `children` to spans created by step 1 with `state_A` content. Whether step 3 lands on the new DOM nodes depends on Dash's internal reconciliation order. In practice this race window is ~milliseconds and requires near-simultaneous user actions. **Manual smoke test:** change palette → within 200ms toggle `lppl-weighted` → verify inline summary ends on the new value. If it fails, migrate inline summary spans out of the option label entirely and render them as siblings adjacent to the checklist — that eliminates the race but is larger DOM surgery. Deferred until observed.

---

## Test plan

`btc_web/test_palette_roundtrip.py` grows:

- **Extend** `test_palette_rebuild_matches_initial_bub` → `test_palette_rebuild_matches_initial` parametrized over `prefix in ("bub", "dca", "ret", "sc")`.
- **Add** `test_no_mini_card_ids_anywhere(prefix, family)`: parametrized 5 × 3 = 15 cases. For each, assert that `{prefix}-{family}-activate`, `{prefix}-{family}-configure-btn`, `{prefix}-{family}-summary` do NOT appear in the rendered initial layout. (Heatmap included — its old mini cards are also deleted.)
- **Add** `test_inline_summary_spans_exist(prefix, family)`: parametrized 4 × 3 = 12 cases for checklist tabs. Asserts `{prefix}-{family}-summary-inline` exists.
- **Add** `test_gear_ids_per_tab(prefix)`: 4 checklist tabs. Bubble includes `bub-bm-gear`; others don't.
- **Add** `test_heatmap_status_row_exists`: asserts `hm-active-family-row`, `hm-active-family-label`, `hm-active-family-summary-inline`, `hm-active-family-gear` exist in the rendered heatmap layout.
- **Add** `test_palette_summary_not_stale`: extract modal state, switch palette, verify inline summary children match the modal state (not the builder default). Directly tests Risk #2's mitigation.
- **Add** `test_defunct_placeholders_unconditional`: assert `_defunct-snapshot-placeholders` div is in the rendered layout regardless of `_HAS_MARKOV`, `_HAS_CELERY`, or URL path. Parametrize over several `_serve_layout` invocations with different flag combinations (requires patching `_app_ctx`). Directly tests Risk #1's mitigation (hot-reload ComponentIDNotFound).
- **Add** `test_modal_open_callbacks_use_gear_inputs`: static assertion that each of the three modal-open callbacks' Input list contains only `*-gear` + `*-modal-close-btn` — no `*-configure-btn`.
- **Add** `test_old_snapshot_link_decodes_cleanly`: construct a synthetic `q3:` link with all 15 `*-activate` keys set to `["yes"]`, decode via `_decode_snapshot`, assert no exception, assert `hm-active-model` restores correctly.
- **Keep** existing `test_palette_rebuild_sim_no_leaks` — still valid.

Existing tests in `test_models.py`, `test_web.py`, `test_callbacks.py`, `test_figures.py` that touch `*-configure-btn`, `*-activate`, or `*-summary` (non-inline) must be updated or deleted. **Audit completed in Task 0.**

Manual smoke tests (post-commit, pre-deploy):
- Each of the 5 tabs: switch palette 4× (default → cb-brian → cb-rg → cb-full → default). Display Models section identical each time. Inline summaries show current modal state (not static default).
- Heatmap: click each of the 5 pills. Status row appears for LPPL/HybPPL/EPPL, hides for BM/PL/S2F/MC/others. Click the gear → correct modal opens. Click a non-gear element adjacent to gear → no accidental modal open.
- Each tab: click every gear icon → correct modal opens. Modal close → returns to correct state.
- Mobile viewport (< 768px): Display Models section wraps cleanly, heatmap status row doesn't overflow.

---

## Open questions

- **None remaining.** v1's "heatmap consistency drift" question is resolved (heatmap gets the same pattern via status row). v1's "hm-*-activate investigation" question is resolved (migration, not investigation — Task 0 handles it). v1's "BM master on sim tabs" stays "never".

## Deferred follow-ups

- **MC as a master with gear.** Currently MC is a flat checklist option (no gear). For consistency with BM/LPPL/HybPPL/EPPL, it could become a master with a gear that scrolls to `{prefix}-mc-body` or opens an MC config modal. Out of scope for this refactor; file as a separate feature.
- **Heatmap pill swatch palette response.** `_hm_pill_sync` currently restyles pill outlines based on `hm-active-model` but not based on palette color. If the palette changes, the pill inline-block swatch stays the old color until the layout is rebuilt. Out of scope — file separately.
- **Unify `_PROMOTED` / `_DEPRIORITIZED` naming across the codebase.** v1 used both; the refactor settles on `_PROMOTED = ("pca", "grdy")` + `_DEPRIORITIZED = {"exp", "s2f", "gomp", "bpl"}` as the single source of truth inside `display_models.py`. Other modules may still reference old names; spot-check during Task 1.
