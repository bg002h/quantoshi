# Display Models Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate the "Display Models" UI pattern across tabs 1/2/3/4/5 of the Quantoshi Dash web app. Delete the 3 redundant mini config card panels (`_lppl_config_panel`, `_hybppl_config_panel`, `_eppl_config_panel`). Replace with a single shared checklist component for the 4 checklist-style tabs and a status row for heatmap. Gears live next to the visible selector; mini cards don't exist anywhere.

**Architecture:** New shared file `layout/display_models.py` exports `display_models_panel(prefix, ...)` + pure builder `build_display_models_options(mc, ..., summaries=None)`. A `dcc.Store("display-model-summaries")` becomes the single source of truth for family summary strings, written by one callback (29 Inputs), read by three clientside Store-readers + `update_model_swatches` + heatmap status row. Heatmap gets a status row below its pill bar that dispatches to modals via a single clientside callback. Snapshot compatibility preserved via 15 hidden placeholder `dcc.Checklist` components at the unconditional root of `_serve_layout`.

**Tech Stack:** Dash 4.0, dbc, Plotly, Python 3.12, pytest

**Spec:** `docs/superpowers/specs/2026-04-11-display-models-consolidation-design.md` (v2.3, approved)

---

## Pre-flight

Before beginning any task, the executing agent must:
1. Read the full spec at `docs/superpowers/specs/2026-04-11-display-models-consolidation-design.md`. The spec is authoritative for architecture decisions; this plan is the execution script.
2. Verify prerequisite symbols exist (don't trust the plan — grep for them):
   ```bash
   cd /scratch/code/bitcoinprojections
   grep -n "LPPL_FAMILY_HIDDEN_FROM_BUBBLE\|HYBPPL_FAMILY_HIDDEN" btc_web/_app_ctx.py
   grep -n "^\(BLACK\|FALLBACK_MODEL_GRAY\|LOT_MARKER_OUTLINE\|MODEL_TRACE_COLORS\|CITADEL_OVERLAY_COLORS\|LINK\)\b" btc_web/colors.py
   ```
   Expected: both greps return results. If any symbol is missing, STOP and report — the plan's imports won't work.
3. Run the broad baseline test suite to establish a green starting point:
   ```bash
   btc_venv/bin/python3 -m pytest btc_web/test_palette_roundtrip.py btc_web/test_models.py btc_web/test_web.py btc_web/test_callbacks.py btc_web/test_figures.py -q --tb=line 2>&1 | tail -20
   ```
   Expected: all green (exact count depends on environment). Record the pass count. Any failure here is PRE-EXISTING and must be distinguishable from failures the refactor introduces.
4. Confirm the dev server starts cleanly:
   ```bash
   lsof -ti :8050 | xargs -r kill -9 ; DEV=1 nohup bash run_web.sh > /tmp/quantoshi_dev.log 2>&1 &
   sleep 4 && tail -20 /tmp/quantoshi_dev.log
   ```
   Expected: `Dash is running on http://0.0.0.0:8050/` with no tracebacks.

---

## Task 0 — Investigation + migration planning (no code changes)

**Files touched:** None. Deliverable is a chat-visible report.

**Purpose:** Lock down the facts the rest of the plan depends on. The spec says the `hm-*-activate` checkboxes are vestigial and the decoder at `snapshot.py:515` is positional. Task 0 verifies these assumptions empirically before any deletion happens.

- [ ] **Step 1: Grep the entire codebase for every reference to the 15 defunct ids**

Run these greps and record the output:
```bash
cd /scratch/code/bitcoinprojections
```

```bash
grep -rn "lppl-activate\|hybppl-activate\|eppl-activate" btc_web/ archive/ tools/ --include="*.py" --include="*.js" --include="*.css" --include="*.html" 2>&1 | tee /tmp/activate_refs.txt
```

Expected locations (verify each is present):
- `btc_web/snapshot.py` lines 129, 284-287, 295, 309-312, 314, 328-331 (in `_SNAPSHOT_CONTROLS`)
- `btc_web/snapshot.py` lines 391-409 (in `_CHECKLIST_OPTIONS`)
- `btc_web/callbacks/routing.py` lines 128, 129, 137, 152, 153, 161, 176, 177, 185, 197, 198, 206, 219, 220, 228 (in `_TAB_CONTROLS`)
- `btc_web/layout/common.py` (inside `_lppl_config_panel`, `_hybppl_config_panel`, `_eppl_config_panel`)
- `btc_web/callbacks/charts.py` (activate↔selector mirror callbacks, configure-btn click handlers, and hm-active-model sync)
- `btc_web/test_models.py` (assertions — approximate lines 1668-1670, 2120-2122, 2360-2362)
- `btc_web/test_palette_roundtrip.py` line 96 (gear id assertion — keep this test, rewrite target ids)

Flag any references outside these expected locations.

- [ ] **Step 2: Verify `_decode_snapshot` is positional**

```bash
sed -n '495,525p' btc_web/snapshot.py
```

Confirm line 515 reads `for (cid, prop), val in zip(_SNAPSHOT_CONTROLS, values):`. Positional decoding is the reason we cannot delete tuples.

- [ ] **Step 3: Verify `callbacks/snapshot_cb.py:58` statically registers Outputs**

```bash
sed -n '55,85p' btc_web/callbacks/snapshot_cb.py
```

Confirm line 58 reads `*[Output(cid, prop, allow_duplicate=True) for cid, prop in _SNAPSHOT_CONTROLS],`. This is why the 15 tuples need hidden placeholder components in the layout.

- [ ] **Step 4: Verify `hm-active-model` is in `_SNAPSHOT_CONTROLS`**

```bash
grep -n "hm-active-model" btc_web/snapshot.py
```

Expected: line 136 `("hm-active-model",   "data"),`. This confirms heatmap snapshots persist pill state directly — the `hm-*-activate` checkboxes are vestigial mirrors, safe to delete.

- [ ] **Step 5: Identify the exact activate↔selector mirror callbacks to delete**

```bash
grep -n "activate\|configure-btn" btc_web/callbacks/charts.py
```

Build a line-ranged list of callbacks to delete in Task 5. Save to `/tmp/charts_deletions.txt`. Expected count: ~24 mirror callbacks + ~15 configure-btn handlers + ~6 heatmap sync callbacks = ~45 callbacks.

- [ ] **Step 6: Deliver Task 0 report**

Paste a structured summary into the chat:
```
Task 0 Report — Display Models Consolidation
=============================================
Codebase grep:
  - _SNAPSHOT_CONTROLS refs: <N> confirmed (expected 15)
  - _CHECKLIST_OPTIONS refs: <N> confirmed (expected 15)
  - _TAB_CONTROLS refs: <N> confirmed (expected 15)
  - charts.py callbacks to delete: <N> identified
  - test_models.py assertions to update: <N> at lines <list>
  - Unexpected references: <list or NONE>

Decoder verification:
  - snapshot.py:515 is zip(positional): CONFIRMED
  - snapshot_cb.py:58 is static: CONFIRMED
  - hm-active-model in _SNAPSHOT_CONTROLS at line 136: CONFIRMED

Deletion safety:
  - hm-*-activate safe to delete (vestigial of hm-active-model): CONFIRMED
  - bub-bm-activate preservation required (lives in BM primary card, not mini card): CONFIRMED

Go/no-go: GO for Task 1
```

No commit in Task 0 — report only.

---

## Task 1 — Scaffolding (standalone commit, app still runnable)

**Files:**
- Create: `btc_web/layout/display_models.py`
- Modify: `btc_web/layout/common.py` — add `_GEAR_STYLE` + `_MUTED_STYLE` constants (copied from bubble.py; do NOT yet remove from bubble.py — that happens in Task 2)
- Modify: `btc_web/app.py` OR `btc_web/layout/__init__.py` (wherever `_serve_layout` lives) — add the hidden `_defunct-snapshot-placeholders` div at the unconditional root
- Modify: `btc_web/app.py` or layout root — add `dcc.Store(id="display-model-summaries", storage_type="memory", data={"lppl": "", "hybppl": "", "eppl": ""})` at the unconditional root

**Intent:** Add dead code that future tasks will consume. App behavior is unchanged.

- [ ] **Step 1: Locate `_serve_layout`**

```bash
grep -rn "_serve_layout\|def serve_layout" btc_web/ --include="*.py" | head -10
```

Record the file + line. The placeholder div and Store must go into the unconditional top of whatever this function returns.

- [ ] **Step 2: Create `btc_web/layout/display_models.py` scaffolding**

Write the file with this content (the full ~200 line implementation comes in Task 2; for Task 1 we only need a minimal scaffold so imports work):

```python
"""Shared Display Models panel + option builder.

Used by tabs 1/3/4/5 (Bubble, DCA, Retire, Supercharger) via
`display_models_panel(prefix, **flags)` for initial layout, and by
`callbacks/charts.py::update_model_swatches` via
`build_display_models_options(mc, ..., summaries=dict)` for palette rebuild.

Heatmap (tab 2) does NOT use this module — it has a pill bar, not a
checklist. Heatmap's status row lives in `layout/heatmap.py`.

See spec: docs/superpowers/specs/2026-04-11-display-models-consolidation-design.md
"""
from __future__ import annotations

from dash import dcc, html

import _app_ctx
from colors import (
    BLACK, FALLBACK_MODEL_GRAY, LOT_MARKER_OUTLINE,
    MODEL_TRACE_COLORS, CITADEL_OVERLAY_COLORS, LINK,
)
from layout.common import (
    _GEAR_STYLE, _MUTED_STYLE,
    _model_info_link, _section_card, _legend_pos_dropdown,
    _INFO_ICON,
)


def build_display_models_options(
    mc: dict,
    *,
    include_bm_master: bool = False,
    include_mc: bool = False,
    include_u1: bool = True,
    summaries: dict | None = None,
) -> list[dict]:
    """Pure function. Builds checklist options for a Display Models panel.

    `summaries` keys: "lppl", "hybppl", "eppl". When None, inline summary
    spans emit static defaults that get overwritten by the Store-reader
    clientside callback on page load.
    """
    # FULL implementation lands in Task 2 Step 3. Stub for Task 1:
    return []


def display_models_panel(
    prefix: str,
    *,
    include_bm_master: bool = False,
    include_mc: bool = False,
    include_u1: bool = True,
):
    """Return the Display Models section_card for one checklist-style tab."""
    # FULL implementation lands in Task 2 Step 4. Stub for Task 1:
    return _section_card("Display Models",
        dcc.Checklist(id=f"{prefix}-model-show", options=[], value=[]),
        *_legend_pos_dropdown(prefix, "bottom-right"),
    )
```

- [ ] **Step 3: Add `_GEAR_STYLE` and `_MUTED_STYLE` to `layout/common.py`**

Append near the top of `layout/common.py` (after existing style constants):

```python
# ── Styles for Display Models in-checklist-label controls ───────────
# Moved from layout/bubble.py during display-models consolidation so
# that display_models.py and heatmap.py can share them.
_GEAR_STYLE = {
    "cursor": "pointer", "fontSize": "11px", "marginLeft": "4px",
    "opacity": "0.6", "textDecoration": "none",
}
_MUTED_STYLE = {
    "color": "#9a9a9a", "fontSize": "11px", "fontStyle": "italic",
}
```

Do NOT remove the local `_GEAR_STYLE` in `layout/bubble.py` yet — bubble.py still uses its own copy until Task 2.

- [ ] **Step 4: Add placeholder div + summary Store to `_serve_layout`**

Locate `_serve_layout` (from Step 1). At the unconditional top of the returned layout tree (before any flag-gated branches), add:

```python
# ── Display Models consolidation placeholder block (always emitted) ──
# These 15 hidden dcc.Checklist components satisfy callback registration
# for _SNAPSHOT_CONTROLS tuples that are defunct after the display-models
# refactor but MUST remain in _SNAPSHOT_CONTROLS to preserve positional
# bit-index stability for old q3: share links.
# See spec: 2026-04-11-display-models-consolidation-design.md, Task 0 finding 3.
html.Div(
    id="_defunct-snapshot-placeholders",
    style={"display": "none"},
    children=[
        dcc.Checklist(
            id=f"{prefix}-{family}-activate",
            options=[{"label": "", "value": "yes"}],
            value=[],
        )
        for prefix in ("bub", "dca", "ret", "sc", "hm")
        for family in ("lppl", "hybppl", "eppl")
    ],
),

# ── Display Models family summary Store ────────────────────────────
# Single source of truth for LPPL/HybPPL/EPPL summary strings.
# Written by compute_family_summaries callback; read by
# update_model_swatches + inline Store-reader clientside callbacks +
# heatmap status row.
dcc.Store(
    id="display-model-summaries",
    storage_type="memory",
    data={"lppl": "LPPL\u2083", "hybppl": "1d+1u", "eppl": "1d+1u"},
),
```

Both must be unconditional — not inside any `if _HAS_MARKOV`, not inside tab routing, not inside flag branches.

- [ ] **Step 5: Syntax check**

```bash
cd btc_web && PYTHONPATH=".:../archive/btc_app:../" ../btc_venv/bin/python3 -c "import app; print('OK')"
```

Expected: `OK` (and the initial MC-cache load message). No ImportError, no ComponentIDNotFound.

- [ ] **Step 6: Run the existing test suite**

```bash
cd /scratch/code/bitcoinprojections
btc_venv/bin/python3 -m pytest btc_web/test_palette_roundtrip.py -v
```

Expected: 18 passed (unchanged baseline).

- [ ] **Step 7: Start dev server and manually verify no visible regression**

```bash
lsof -ti :8050 | xargs -r kill -9 ; DEV=1 nohup bash run_web.sh > /tmp/quantoshi_dev.log 2>&1 &
sleep 4 && tail -30 /tmp/quantoshi_dev.log
```

Expected: server starts clean. Open the browser to `http://localhost:8050/1` — bubble tab renders identically to before (mini cards still present, gears still present, layout unchanged). The placeholder div is invisible; the Store is invisible.

- [ ] **Step 8: Commit Task 1**

```bash
cd /scratch/code/bitcoinprojections
git add btc_web/layout/display_models.py btc_web/layout/common.py btc_web/app.py btc_web/layout/__init__.py
git commit -m "$(cat <<'EOF'
feat(display-models): scaffold shared builder + placeholder block

Prep commit for display-models consolidation. Adds:
- btc_web/layout/display_models.py (scaffold — builder stubs return empty)
- _GEAR_STYLE and _MUTED_STYLE moved to layout/common.py (still duplicated in bubble.py until Task 2)
- Unconditional html.Div(_defunct-snapshot-placeholders) + dcc.Store(display-model-summaries) at _serve_layout root

No behavioral change; app renders identically. Tasks 2-5 will atomically swap all tabs to use the new builder.

See docs/superpowers/specs/2026-04-11-display-models-consolidation-design.md

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

Only stage the files you actually changed — adjust the `git add` list to match. Never use `git add -A` or `git add .`.

---

## Task 2 — Smoke-test the nested-id Output pattern (atomic commit boundary starts here)

**⚠ CRITICAL GATE:** Task 2 verifies that Dash 4.0 supports callback Outputs targeting span `id`s nested inside `Checklist.options[i]["label"]`. If this fails, the entire design collapses and we must rewrite to render summary spans as siblings adjacent to the checklist. **Do not proceed to Tasks 3-5 until Task 2 Step 6 passes.**

This task also begins the **atomic refactor commit**. Task 2 through Task 5 accumulate changes in the working tree (unstaged). The single commit lands at Task 5 Step 14.

**⚠ Git discipline between Tasks 2 and 5:**
- Do NOT run `git add`, `git commit`, `git stash`, or `git reset` between Task 2 Step 1 and Task 5 Step 14.
- Expect `git status` to show many unstaged modifications — this is normal and intentional.
- If anything in Tasks 2–5 fails: `git reset --hard HEAD` reverts everything cleanly back to the Task 1 commit. (Task 1's scaffolding survives because it was already committed.)
- NEVER use `git add -A` or `git add .` — use explicit file paths only (CLAUDE.md memory rule).

**Files:**
- Modify: `btc_web/layout/display_models.py` — flesh out `build_display_models_options` and `display_models_panel`
- Modify: `btc_web/layout/bubble.py` — delete `_build_bub_model_options`, delete local `_GEAR_STYLE`, swap call to `display_models_panel("bub", include_bm_master=True)`, drop `_lppl_config_panel("bub")`, `_hybppl_config_panel("bub")`, `_eppl_config_panel("bub")` (lines 207-209)
- Modify: `btc_web/callbacks/charts.py` — add `compute_family_summaries` callback, add 3 clientside Store-reader callbacks, add smoke-test temporary callback

- [ ] **Step 1: Write `build_display_models_options` full implementation**

Replace the stub in `btc_web/layout/display_models.py` with the full builder. The signature takes `prefix: str` as a positional arg (required for constructing per-tab nested ids like `{prefix}-lppl-summary-inline`). Start from `layout/bubble.py::_build_bub_model_options` as the logic baseline, then apply these changes:

- Add `prefix: str` positional parameter.
- Accept `summaries: dict | None` kwarg (populated on palette rebuild, `None` for initial layout).
- Filter `_HYBPPL_FAM = set(_app_ctx.HYBPPL_FAMILY_HIDDEN)` + `startswith("cfg_")` / `startswith("ecfg_")`.
- Emit 4 master entries: Bubble Model (conditional on `include_bm_master`), Entropy PPL, LPPL, Hybrid PPL — each with a gear span and inline summary span inside the label.
- Add U₁ at end if `include_u1`, MC if `include_mc`.

Full code:

```python
def build_display_models_options(
    mc: dict,
    prefix: str,
    *,
    include_bm_master: bool = False,
    include_mc: bool = False,
    include_u1: bool = True,
    summaries: dict | None = None,
) -> list[dict]:
    summaries = summaries or {}
    _LPPL_FAM   = {"lppl", "lp2", "lp3", "lp4"} | set(_app_ctx.LPPL_FAMILY_HIDDEN_FROM_BUBBLE)
    _HYBPPL_FAM = set(_app_ctx.HYBPPL_FAMILY_HIDDEN)
    _PROMOTED   = ("pca", "grdy")
    _DEPRIORITIZED = {"exp", "s2f", "gomp", "bpl"}

    def _swatch_span(color):
        return html.Span(" ", style={
            "display": "inline-block", "width": "12px", "height": "12px",
            "borderRadius": "2px", "verticalAlign": "middle",
            "marginRight": "4px", "backgroundColor": color,
        })

    def _gear_span(gear_id, title):
        return html.Span(
            "\u2699\uFE0F", id=gear_id, n_clicks=0,
            style=_GEAR_STYLE, title=title,
        )

    def _inline_summary(span_id, default):
        return [
            html.Span(" (", style=_MUTED_STYLE),
            html.Span(id=span_id, children=default, style=_MUTED_STYLE),
            html.Span(")", style=_MUTED_STYLE),
        ]

    def _info_link(model_key, label):
        href, exists = _model_info_link(model_key)
        if not exists:
            return None
        return html.A(
            _INFO_ICON, href=href,
            style={"cursor": "pointer", "fontSize": "11px",
                   "marginLeft": "4px", "opacity": "0.6",
                   "textDecoration": "none", "color": LINK},
            title=f"View {label} details on Model Info tab",
        )

    def _master_label(color, name, gear_id, summary_id, summary_default, gear_title):
        return html.Span([
            _swatch_span(color),
            name,
            *_inline_summary(summary_id, summary_default),
            _gear_span(gear_id, gear_title),
        ])

    def _plain_label(color, name, model_key=None):
        children = [_swatch_span(color), name]
        if model_key:
            link = _info_link(model_key, name)
            if link:
                children.append(link)
        return html.Span(children)

    opts = []

    # 1. Bubble Model — master with gear on bubble, plain elsewhere
    if include_bm_master:
        opts.append({
            "label": html.Span([
                _swatch_span(mc.get("bub", BLACK)),
                "Bubble Model",
                _gear_span(f"{prefix}-bm-gear", "Open Bubble Model settings"),
            ]),
            "value": "bub",
        })
    else:
        opts.append({
            "label": _plain_label(mc.get("bub", BLACK), "Bubble Model"),
            "value": "bub",
        })

    # 2. Entropy PPL master
    opts.append({
        "label": _master_label(
            mc.get("eppl", MODEL_TRACE_COLORS["eppl"]),
            "\U0001FAE0 Entropy PPL",
            gear_id=f"{prefix}-eppl-gear",
            summary_id=f"{prefix}-eppl-summary-inline",
            summary_default=summaries.get("eppl", "1d+1u"),
            gear_title="Configure Entropy PPL",
        ),
        "value": "eppl",
    })

    # 3. LPPL master
    opts.append({
        "label": _master_label(
            mc.get("lppl", MODEL_TRACE_COLORS["lppl"]),
            "LPPL",
            gear_id=f"{prefix}-lppl-gear",
            summary_id=f"{prefix}-lppl-summary-inline",
            summary_default=summaries.get("lppl", "LPPL\u2083"),
            gear_title="Configure LPPL",
        ),
        "value": "lppl",
    })

    # 4. Hybrid PPL master
    opts.append({
        "label": _master_label(
            mc.get("hybppl", CITADEL_OVERLAY_COLORS["reserves_total"]),
            "Hybrid PPL",
            gear_id=f"{prefix}-hybppl-gear",
            summary_id=f"{prefix}-hybppl-summary-inline",
            summary_default=summaries.get("hybppl", "1d+1u"),
            gear_title="Configure Hybrid PPL",
        ),
        "value": "hybppl",
    })

    # 5-7. Non-master model entries
    _HIDDEN = (
        set(_app_ctx.MODEL_SENTINELS)
        | {"bub", "eppl"}
        | _LPPL_FAM
        | _HYBPPL_FAM
    )
    all_models = [
        m for m in _app_ctx.PRICE_MODELS.values()
        if m.short_name not in _HIDDEN
        and not m.short_name.startswith("cfg_")
        and not m.short_name.startswith("ecfg_")
    ]
    promoted = [m for m in all_models if m.short_name in _PROMOTED]
    promoted.sort(key=lambda m: _PROMOTED.index(m.short_name))
    primary  = [m for m in all_models
                if m.short_name not in _PROMOTED
                and m.short_name not in _DEPRIORITIZED]
    deprior  = [m for m in all_models if m.short_name in _DEPRIORITIZED]

    for mdl in promoted + primary + deprior:
        opts.append({
            "label": _plain_label(
                mc.get(mdl.short_name, FALLBACK_MODEL_GRAY),
                mdl.name,
                model_key=mdl.short_name,
            ),
            "value": mdl.short_name,
        })

    # 8. U₁
    if include_u1:
        opts.append({
            "label": _plain_label(
                mc.get("u1", LOT_MARKER_OUTLINE),
                "U\u2081 (User)",
                model_key="u1",
            ),
            "value": "u1",
        })

    # 9. MC
    if include_mc:
        opts.append({
            "label": _plain_label(
                mc.get("mc", FALLBACK_MODEL_GRAY),
                "MC Simulation",
            ),
            "value": "mc",
        })

    return opts
```

- [ ] **Step 2: Write `display_models_panel` full implementation**

Replace the Task 1 stub:

```python
def display_models_panel(
    prefix: str,
    *,
    include_bm_master: bool = False,
    include_mc: bool = False,
    include_u1: bool = True,
    legend_pos_default: str = "bottom-right",
):
    """Return the Display Models section_card for one checklist-style tab."""
    mc = _app_ctx.MODEL_TRACE_COLORS
    options = build_display_models_options(
        mc, prefix,
        include_bm_master=include_bm_master,
        include_mc=include_mc,
        include_u1=include_u1,
    )
    return _section_card(
        "Display Models",
        dcc.Checklist(
            id=f"{prefix}-model-show",
            options=options,
            value=[],  # actual default comes from tab_defaults via existing callbacks
            labelStyle={"display": "block"},
            inputStyle={"marginRight": "4px"},
        ),
        *_legend_pos_dropdown(prefix, legend_pos_default),
    )
```

- [ ] **Step 3: Swap `layout/bubble.py` to use `display_models_panel`**

Edit `btc_web/layout/bubble.py`:

1. Delete `_build_bub_model_options` function (lines ~25-117).
2. Delete local `_GEAR_STYLE` definition (~line 30).
3. At the site that currently calls `dcc.Checklist(id="bub-model-show", options=_build_bub_model_options(...))`, replace with:
   ```python
   display_models_panel("bub", include_bm_master=True, legend_pos_default=BUBBLE["legend_pos"]),
   ```
4. Delete the three lines at ~207-209:
   ```python
   _lppl_config_panel("bub"),
   _hybppl_config_panel("bub"),
   _eppl_config_panel("bub"),
   ```
5. Add `from layout.display_models import display_models_panel` to the imports at the top.
6. Remove unused imports (`_lppl_config_panel`, `_hybppl_config_panel`, `_eppl_config_panel` if no longer used).

Leave `bub-bm-activate` + `bub-bm-body` alone — they stay as-is inside the primary BM settings card (see spec BM carve-out).

- [ ] **Step 4: Wire the `compute_family_summaries` callback**

Add to `btc_web/callbacks/charts.py` (near the top of the file, before other modal-related callbacks). `callback`, `Input`, `Output`, `State` are already imported — do not re-import. Full 29-Input callback per the spec:

```python
@callback(
    Output("display-model-summaries", "data"),
    # LPPL (3)
    Input("lppl-n-freqs", "value"),
    Input("lppl-weighted", "value"),
    Input("lppl-no-13",   "value"),
    # HybPPL (13)
    Input("hybppl-cfg-a-nlog",    "value"),
    Input("hybppl-cfg-a-ncal",    "value"),
    Input("hybppl-cfg-a-log1d",   "value"),
    Input("hybppl-cfg-a-log2d",   "value"),
    Input("hybppl-cfg-a-cal1d",   "value"),
    Input("hybppl-cfg-a-cal2d",   "value"),
    Input("hybppl-cfg-b-enabled", "value"),
    Input("hybppl-cfg-b-nlog",    "value"),
    Input("hybppl-cfg-b-ncal",    "value"),
    Input("hybppl-cfg-b-log1d",   "value"),
    Input("hybppl-cfg-b-log2d",   "value"),
    Input("hybppl-cfg-b-cal1d",   "value"),
    Input("hybppl-cfg-b-cal2d",   "value"),
    # EPPL (13)
    Input("eppl-cfg-a-nlog",    "value"),
    Input("eppl-cfg-a-ncal",    "value"),
    Input("eppl-cfg-a-log1d",   "value"),
    Input("eppl-cfg-a-log2d",   "value"),
    Input("eppl-cfg-a-cal1d",   "value"),
    Input("eppl-cfg-a-cal2d",   "value"),
    Input("eppl-cfg-b-enabled", "value"),
    Input("eppl-cfg-b-nlog",    "value"),
    Input("eppl-cfg-b-ncal",    "value"),
    Input("eppl-cfg-b-log1d",   "value"),
    Input("eppl-cfg-b-log2d",   "value"),
    Input("eppl-cfg-b-cal1d",   "value"),
    Input("eppl-cfg-b-cal2d",   "value"),
)
def compute_family_summaries(*args):
    lppl_args   = args[0:3]
    hybppl_args = args[3:16]
    eppl_args   = args[16:29]
    return {
        "lppl":   _format_lppl_summary(*lppl_args),
        "hybppl": _format_hybppl_summary(*hybppl_args),
        "eppl":   _format_eppl_summary(*eppl_args),
    }
```

Implement `_format_lppl_summary`, `_format_hybppl_summary`, `_format_eppl_summary` as module-level helpers in `charts.py`. **The existing summary callbacks are CLIENTSIDE JS** (charts.py around line 216-225 for LPPL, and similar blocks for HybPPL/EPPL). The JS is the **source of truth** — the Python port MUST produce byte-identical output for every input, or the inline summary text will silently differ from today's mini card summaries.

**Existing JS for LPPL** (charts.py:216-225 — verify by reading the file before porting):
```javascript
function(n_freqs, weighted, no_13) {
    var ns = (n_freqs || []).slice().sort();
    if (ns.length === 0) return "(no flavor)";
    var names = {1:'LPPL\u2081', 2:'LPPL\u2082', 3:'LPPL\u2083', 4:'LPPL\u2084'};
    var parts = ns.map(function(n){ return names[n] || ("LPPL"+n); });
    var txt = parts.join('+');
    if ((weighted || []).indexOf('weighted') !== -1) txt += ' (w)';
    if ((no_13 || []).indexOf('no13') !== -1) txt += ' (no \u03c9\u224813)';
    return txt;
}
```

**Python port — verbatim equivalent:**
```python
def _format_lppl_summary(n_freqs, weighted, no_13):
    ns = sorted(n_freqs or [])
    if not ns:
        return "(no flavor)"
    names = {1: "LPPL\u2081", 2: "LPPL\u2082", 3: "LPPL\u2083", 4: "LPPL\u2084"}
    txt = "+".join(names.get(n, f"LPPL{n}") for n in ns)
    if weighted and "weighted" in weighted:
        txt += " (w)"
    if no_13 and "no13" in no_13:
        txt += " (no \u03c9\u224813)"
    return txt
```

Apply the same read-and-port process for HybPPL and EPPL:

1. **Locate the existing JS** for each family by grepping `charts.py` for `hybppl-summary` and `eppl-summary`.
2. **Read the full JS body** character-by-character.
3. **Port to Python** preserving exact output: same join characters, same suffix strings (note the non-ASCII `(no ω≈13)` — `\u03c9\u224813`), same conditional logic.
4. **Verify** by diffing the outputs: for at least 3 representative inputs (default, weighted, no_13), run both the existing JS-rendered string (from a browser click test) and the new Python output side-by-side.

If the JS and Python diverge, the Python is wrong — fix the Python, not the JS.

- [ ] **Step 5: Add the three Store-reader clientside callbacks**

Add to `btc_web/callbacks/charts.py` after `compute_family_summaries`:

```python
# Store-reader: LPPL summary fan-out to all checklist tabs
_app_ctx.app.clientside_callback(
    """
    function(data) {
        if (!data) return ['', '', '', ''];
        var s = data.lppl || '';
        return [s, s, s, s];
    }
    """,
    Output("bub-lppl-summary-inline", "children"),
    Output("dca-lppl-summary-inline", "children"),
    Output("ret-lppl-summary-inline", "children"),
    Output("sc-lppl-summary-inline",  "children"),
    Input("display-model-summaries", "data"),
)

# Store-reader: HybPPL summary fan-out
_app_ctx.app.clientside_callback(
    """
    function(data) {
        if (!data) return ['', '', '', ''];
        var s = data.hybppl || '';
        return [s, s, s, s];
    }
    """,
    Output("bub-hybppl-summary-inline", "children"),
    Output("dca-hybppl-summary-inline", "children"),
    Output("ret-hybppl-summary-inline", "children"),
    Output("sc-hybppl-summary-inline",  "children"),
    Input("display-model-summaries", "data"),
)

# Store-reader: EPPL summary fan-out
_app_ctx.app.clientside_callback(
    """
    function(data) {
        if (!data) return ['', '', '', ''];
        var s = data.eppl || '';
        return [s, s, s, s];
    }
    """,
    Output("bub-eppl-summary-inline", "children"),
    Output("dca-eppl-summary-inline", "children"),
    Output("ret-eppl-summary-inline", "children"),
    Output("sc-eppl-summary-inline",  "children"),
    Input("display-model-summaries", "data"),
)
```

These callbacks reference ids (`dca-*-summary-inline`, `ret-*-summary-inline`, `sc-*-summary-inline`) that don't exist yet (sim tabs are still using the old `_model_show_checklist`). Dash WILL raise `ComponentIDNotFound` at registration until Task 3 mounts those tabs. **This is expected** — we're inside the atomic commit boundary. Do NOT test-run the app between Task 2 and Task 3.

- [ ] **Step 6: Smoke-test the nested-id Output pattern immediately**

**This is the critical gate.** Before going further, prove that a callback Output targeting a span `id` nested inside `Checklist.options[i]["label"]` actually updates the children. Use an **auto-fire trigger**, NOT a click, because `<label>` elements intercept clicks to toggle the checkbox — a failed click handler would look identical to a failed nested-id Output and hide the real bug.

  1. **Comment out** the three Store-reader clientside callbacks added in Task 2 Step 5 (they reference dca/ret/sc/sc/hm-*-summary-inline ids that don't exist yet and will fail registration). Use `"""` triple-quoted Python string comments or `# ` prefix on every line.

  2. **Add a dcc.Interval** to `_serve_layout` alongside the summary Store:
     ```python
     dcc.Interval(id="_nested-id-smoke-test", interval=500, max_intervals=1),
     ```
     (Temporary — deleted at end of Step 6.)

  3. **Add a temporary auto-fire clientside callback** to `charts.py`:
     ```python
     # SMOKE TEST — delete before Task 3 starts
     _app_ctx.app.clientside_callback(
         """function(n) {
             if (!n) return window.dash_clientside.no_update;
             return "SMOKE_OK";
         }""",
         Output("bub-lppl-summary-inline", "children", allow_duplicate=True),
         Input("_nested-id-smoke-test", "n_intervals"),
         prevent_initial_call=True,
     )
     ```
     This fires exactly once ~500ms after page load. No click required.

  4. **Start the dev server:**
     ```bash
     lsof -ti :8050 | xargs -r kill -9 ; DEV=1 nohup bash run_web.sh > /tmp/quantoshi_dev.log 2>&1 &
     sleep 4 && tail -30 /tmp/quantoshi_dev.log
     ```
     Check the log shows "Dash is running on http://0.0.0.0:8050/" and no traceback. If there's a traceback mentioning a missing id, fix before proceeding.

  5. **Open `http://localhost:8050/1` in a browser.** Wait 1 second for the Interval to fire.

  6. **Verify via DevTools OR page source:**
     - Open DevTools (F12), go to the Elements panel.
     - Ctrl+F within Elements → search for `bub-lppl-summary-inline`.
     - The matching `<span id="bub-lppl-summary-inline">` element should contain the text `SMOKE_OK` (not the static default `LPPL₃`).
     - Alternative verification without DevTools: use `curl` to grab the Dash layout endpoint:
       ```bash
       curl -s http://localhost:8050/_dash-layout | python3 -c "import sys,json; d=json.load(sys.stdin); print('smoke check: depends on Dash payload format — fall back to DevTools')"
       ```
       This is brittle; DevTools is the reliable path.

  7. **Interpret the result:**
     - **Span contains `SMOKE_OK`:** nested-id Outputs work. Proceed.
     - **Span contains `LPPL₃` (unchanged):** nested-id Output pattern is BROKEN. STOP the refactor. Fallback: render summary spans as siblings adjacent to the Checklist (outside the option label). Escalate to user for re-plan approval. Do NOT continue to Task 3.
     - **Span doesn't exist at all:** either the builder didn't run, or the id is wrong. Inspect the `/_dash-layout` JSON for the bubble tab section and confirm the span's presence. If missing, there's a bug in Task 2 Step 1 (builder) — fix before retrying the smoke test.
     - **Browser console shows a callback error:** read the error. A `ComponentIDNotFound` indicates the Interval or the temp callback has a typo; fix and retry.

  8. **Clean up:** delete the `dcc.Interval(id="_nested-id-smoke-test", ...)` from `_serve_layout`, delete the temp smoke-test clientside callback from `charts.py`. **Leave the Task 2 Step 5 Store-readers STILL COMMENTED OUT** at end of Task 2 — Task 3 Step 3 owns the uncomment. Rationale: if they're uncommented here, Task 2 Step 7's syntax check fails because the sim-tab `dca-lppl-summary-inline` ids don't exist until Task 3 mounts the new layouts. Keeping them commented until Task 3 Step 3 makes Task 2 Step 7's expected success criteria unambiguous (clean import).

  After cleanup, verify via `git diff _serve_layout`-related file: the Interval should be gone.

- [ ] **Step 7: Syntax check (bubble-only state)**

State at this point: bubble tab's mini cards are deleted; sim tabs and heatmap still have theirs. `compute_family_summaries`'s 29 Inputs all live inside `_global_lppl_modal` / `_global_hybppl_modal` / `_global_eppl_modal`, which are rendered unconditionally by `_serve_layout` — they exist regardless of mini cards. Store-readers were commented out at the end of Step 6, so the `{dca,ret,sc}-lppl-summary-inline` ids they target don't need to exist yet.

**Expected result: clean import.**

```bash
cd btc_web && PYTHONPATH=".:../archive/btc_app:../" ../btc_venv/bin/python3 -c "import app" 2>&1 | head -20
```

Acceptable: successful import (silent or just the MC-cache load banner).
NOT acceptable: any error. Specifically:
- `ComponentIDNotFound` pointing at `compute_family_summaries` inputs → one of the 29 inputs was typo'd. Grep charts.py for the id and fix.
- `ComponentIDNotFound` pointing at `*-summary-inline` → a Store-reader wasn't commented out. Re-comment and retry.
- `ImportError` in `display_models.py` → typo in the builder. Read the traceback and fix.

- [ ] **Step 8: Do NOT commit yet**

Task 2 ends without a commit. Leave changes unstaged in the working tree. Continue directly to Task 3.

---

## Task 3 — Swap sim tabs (DCA, Retire, Supercharger)

**Files:**
- Modify: `btc_web/layout/sim_tabs.py`
- Modify: `btc_web/layout/supercharge.py`

- [ ] **Step 1: Swap `layout/sim_tabs.py`**

In `btc_web/layout/sim_tabs.py`, replace the `*_model_show_checklist(prefix, standardized=True, include_mc=include_mc)` call inside `_accum_withdraw_controls` (line 37 area) with:

```python
display_models_panel(prefix, include_mc=include_mc, legend_pos_default=legend_pos_default),
```

Delete lines 53-55:
```python
children.append(_lppl_config_panel(prefix))
children.append(_hybppl_config_panel(prefix))
children.append(_eppl_config_panel(prefix))
```

Update imports:
```python
from layout.display_models import display_models_panel
```

Remove unused `_model_show_checklist`, `_lppl_config_panel`, `_hybppl_config_panel`, `_eppl_config_panel` from the `layout.common` import.

- [ ] **Step 2: Swap `layout/supercharge.py`**

In `btc_web/layout/supercharge.py`, same pattern. Replace the call to `_model_show_checklist("sc", standardized=True)` with:

```python
display_models_panel("sc", legend_pos_default=SUPERCHARGE["legend_pos"]),
```

Delete lines 107-109 (the three mini card calls).

Update imports.

- [ ] **Step 3: Re-enable the Store-reader clientside callbacks in charts.py**

Uncomment the three Store-readers from Task 2 Step 5 if they were commented out.

- [ ] **Step 4: Syntax check**

```bash
cd btc_web && PYTHONPATH=".:../archive/btc_app:../" ../btc_venv/bin/python3 -c "import app" 2>&1 | head -20
```

At this point the checklist-tab summary-inline ids exist for bub/dca/ret/sc (just uncommented the Store-readers in Step 3). Heatmap still has its mini cards — don't touch yet. `hm-*-summary-inline` doesn't exist; that's Task 4.

The Store-reader clientside callbacks target bub/dca/ret/sc summary-inline ids only — those now all exist. `compute_family_summaries` targets `lppl-*`, `hybppl-cfg-*`, `eppl-cfg-*` inputs, which live in the `_global_*_modal` functions (rendered once in `_serve_layout`, independent of mini cards). Import should succeed.

Expected: successful import. If not, read the error and fix before continuing.

- [ ] **Step 5: Do NOT commit yet**

Continue to Task 4.

---

## Task 4 — Heatmap status row

**Files:**
- Modify: `btc_web/layout/heatmap.py`

- [ ] **Step 1: Add the status row to `_hm_pill_bar` (or as a sibling in the parent layout)**

Locate `_hm_pill_bar()` in `btc_web/layout/heatmap.py`. Below the `dbc.ButtonGroup` return, wrap it in a `html.Div` that also contains the new status row:

```python
def _hm_pill_bar():
    buttons = [...]  # existing pill buttons — UNCHANGED
    pill_bar = dbc.ButtonGroup(buttons, size="sm")

    status_row = html.Div(
        id="hm-active-family-row",
        style={"display": "none",           # hidden until gated clientside
               "alignItems": "center",
               "gap": "4px",
               "marginTop": "6px",
               "fontSize": "11px"},
        children=[
            html.Span("Active: ", style={"color": FALLBACK_MODEL_GRAY}),
            html.Span(id="hm-active-family-label",
                      style={"fontWeight": "600"}),
            html.Span(" · (", style=_MUTED_STYLE),
            html.Span(id="hm-active-family-summary-inline",
                      children="", style=_MUTED_STYLE),
            html.Span(") ", style=_MUTED_STYLE),
            html.Span("\u2699\uFE0F",
                      id="hm-active-family-gear", n_clicks=0,
                      style=_GEAR_STYLE,
                      title="Configure active model"),
        ],
    )

    return html.Div([pill_bar, status_row])
```

Import `_MUTED_STYLE` from `layout.common`. Also verify `FALLBACK_MODEL_GRAY` is imported from `colors` at the top of `heatmap.py` — if not, add it. (The status row references it for the "Active:" label color.)

**Note:** the new `hm-active-family-row`, `hm-active-family-label`, `hm-active-family-summary-inline`, `hm-active-family-gear` ids are **intentionally NOT added** to `_SNAPSHOT_CONTROLS` or `_TAB_CONTROLS`. They are derived/presentational state driven entirely by clientside callbacks from `hm-active-model` (which IS in snapshot). A future audit might flag them as "missing from tab controls" — the answer is that they're not user-settable inputs.

- [ ] **Step 2: Delete the three mini card calls**

Remove lines 126-128:
```python
_lppl_config_panel("hm"),
_hybppl_config_panel("hm"),
_eppl_config_panel("hm"),
```

Remove unused imports.

- [ ] **Step 3: Add the heatmap clientside callbacks**

Add to `btc_web/callbacks/charts.py`:

```python
# Heatmap status row — visibility + label (driven by hm-active-model)
_app_ctx.app.clientside_callback(
    """
    function(active) {
        var CONFIGURABLE = {"lppl": "LPPL", "hybppl": "HybPPL", "eppl": "\u{1FAE0} Entropy PPL"};
        if (!active || !(active in CONFIGURABLE)) {
            return [{display: "none"}, ""];
        }
        return [{display: "inline-flex", alignItems: "center",
                 gap: "4px", marginTop: "6px", fontSize: "11px"}, CONFIGURABLE[active]];
    }
    """,
    Output("hm-active-family-row", "style"),
    Output("hm-active-family-label", "children"),
    Input("hm-active-model", "data"),
)

# Heatmap status row — summary text (driven by display-model-summaries + hm-active-model)
_app_ctx.app.clientside_callback(
    """
    function(data, active) {
        if (!data || !active) return "";
        return data[active] || "";
    }
    """,
    Output("hm-active-family-summary-inline", "children"),
    Input("display-model-summaries", "data"),
    Input("hm-active-model", "data"),
)

# Heatmap gear dispatcher — routes clicks to the correct modal
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
    prevent_initial_call=True,
)
```

- [ ] **Step 4: Do NOT commit yet**

Continue to Task 5 — cleanup and `update_model_swatches` rewrite.

---

## Task 5 — Callback cleanup + `update_model_swatches` rewrite + atomic commit

**Files:**
- Modify: `btc_web/callbacks/charts.py` (heavy deletions + rewrite `update_model_swatches`)
- Modify: `btc_web/layout/common.py` (delete `_model_show_checklist`, `_lppl_config_panel`, `_hybppl_config_panel`, `_eppl_config_panel`)
- Modify: `btc_web/callbacks/routing.py` (delete 15 `*-activate` entries from `_TAB_CONTROLS`)
- Modify: `btc_web/snapshot.py` (add comments only, no deletions — the 15 tuples + 15 keys STAY)
- Modify: `btc_web/layout/bubble.py` (final cleanup — remove any stale imports)

- [ ] **Step 1: Delete ALL activate-mirror generator loops in `charts.py`**

There are **three** separate `for _lp in ("dca", "ret", "sc"):` loops — one for each family (LPPL, HybPPL, EPPL). Each loop generates 2 clientside callbacks per tab × 3 tabs = 6 callbacks per loop. Total to delete: **3 loops × 6 callbacks = 18 clientside callbacks**.

Grep first to confirm the count:
```bash
grep -n "for _lp in" btc_web/callbacks/charts.py
```

Expected: 3 matches. Delete all three loops and the code they contain (each loop body has two `_app_ctx.app.clientside_callback(...)` invocations — the activate→checklist mirror and the checklist→activate mirror).

- [ ] **Step 2: Delete the separate `bub-lppl-activate`, `bub-hybppl-activate`, `bub-eppl-activate` mirror callbacks**

Leave `bub-bm-activate` ↔ `bub-model-show` mirror alone (spec BM carve-out). Only delete the lppl/hybppl/eppl ones.

- [ ] **Step 3: Delete the 6 `hm-*-activate ↔ hm-active-model` bidirectional sync callbacks**

Use Task 0 Step 5 line list.

- [ ] **Step 4: Delete the 15 `{prefix}-{family}-configure-btn` click handlers**

All of them are scroll-to-modal-open handlers that are now replaced by the per-tab gear handlers (already in display_models.py via the modal-open callback rewrite in Step 6).

- [ ] **Step 5: Delete the old `update_*_summary` callbacks**

Find every callback that currently writes to `{prefix}-lppl-summary`, `{prefix}-hybppl-summary`, `{prefix}-eppl-summary` (the non-inline versions on the mini cards) and delete them. The new `compute_family_summaries` + Store-reader chain replaces them.

- [ ] **Step 6: Rewrite the three modal-open callbacks**

Find the existing LPPL modal-open callback (charts.py ~line 117-141). Replace with the atomic-commit version per spec:

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
    Output("lppl-config-modal", "is_open", allow_duplicate=True),
    Input("bub-lppl-gear", "n_clicks"),
    Input("dca-lppl-gear", "n_clicks"),
    Input("ret-lppl-gear", "n_clicks"),
    Input("sc-lppl-gear",  "n_clicks"),
    Input("lppl-modal-close-btn", "n_clicks"),
    State("lppl-config-modal", "is_open"),
    prevent_initial_call=True,
)
```

Same pattern for `hybppl-config-modal` and `eppl-config-modal`. `allow_duplicate=True` + `prevent_initial_call=True` on all three — REQUIRED because the heatmap dispatcher also writes to these Outputs.

- [ ] **Step 7: Delete `_build_model_opts` and rewrite `update_model_swatches`**

Delete the existing `_build_model_opts` function (charts.py ~line 2485-2539). Replace `update_model_swatches` (charts.py ~line 2542) with:

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
    from layout.display_models import build_display_models_options
    pal = _app_ctx.PALETTES.get(palette_key or "default",
                                 _app_ctx.PALETTES["default"])
    mc = pal.get("model_colors", _app_ctx.MODEL_TRACE_COLORS)
    return (
        build_display_models_options(mc, "bub", include_bm_master=True, summaries=summaries),
        build_display_models_options(mc, "dca", summaries=summaries),
        build_display_models_options(mc, "ret", include_mc=True, summaries=summaries),
        build_display_models_options(mc, "sc",  summaries=summaries),
    )
```

- [ ] **Step 8: Delete `_lppl_config_panel`, `_hybppl_config_panel`, `_eppl_config_panel`, `_model_show_checklist` from `layout/common.py`**

These functions now have zero callers. Keep `_global_lppl_modal`, `_global_hybppl_modal`, `_global_eppl_modal` — they're still rendered by `_serve_layout`.

- [ ] **Step 9: Remove the 15 `*-activate` entries from `routing.py::_TAB_CONTROLS`**

Edit `btc_web/callbacks/routing.py`. There are 5 tab sets, each containing 3 `*-activate` entries. Delete all 15 — by string-match anchor, NOT by line number (line numbers drift as you edit).

Advisory-only line locations (to orient the subagent — do NOT use these as Edit anchors):
- Lines ~128, ~129, ~137 in the `"bubble"` set
- Lines ~152, ~153, ~161 in the `"heatmap"` set
- Lines ~176, ~177, ~185 in the `"dca"` set
- Lines ~197, ~198, ~206 in the `"retire"` set
- Lines ~219, ~220, ~228 in the `"supercharge"` set

**Correct approach:** grep for `-lppl-activate|-hybppl-activate|-eppl-activate` in `routing.py` to get the current set of strings. For each, use Edit with `old_string` = the exact string literal like `"dca-lppl-activate",` (including comma and quotes) and `new_string` = empty (or merge with the previous line's whitespace). Repeat for all 15. Verify post-edit with another grep — should return 0 matches.

- [ ] **Step 10: Add comments to `snapshot.py` (no deletions)**

**Use the Edit tool with exact-string `old_string` anchors. Do NOT rely on line numbers — they drift as edits are applied.**

Add this comment block (verbatim) in three places:
```
# ── Defunct after display-models consolidation (2026-04-11) ──
# The *-lppl-activate / *-hybppl-activate / *-eppl-activate tuples below
# are retained for q3: link positional bit-index stability. The decoder
# at line ~515 is positional; deleting these would silently corrupt all
# pre-refactor share links. The components are rendered as hidden
# placeholders by _serve_layout.
```

Location 1 — anchor: the literal tuple `("bub-lppl-activate", "value")`. Insert the comment block on the line above it. Use Edit with `old_string = '    ("bub-lppl-activate", "value"),'` and `new_string` = the comment + newline + the same tuple.

Location 2 — anchor: the literal tuple `("dca-lppl-activate", "value"),` (Phase 1 additions block). Same Edit pattern.

Location 3 — anchor: the literal key `"bub-lppl-activate":  ["yes"],` inside `_CHECKLIST_OPTIONS`. Same Edit pattern, but use a briefer comment since this is a dict literal:
```
    # Defunct after display-models consolidation (2026-04-11) — retained
    # for q3: link positional stability. See _SNAPSHOT_CONTROLS comment above.
```

No other changes to `snapshot.py`.

- [ ] **Step 11: Final syntax check**

```bash
cd btc_web && PYTHONPATH=".:../archive/btc_app:../" ../btc_venv/bin/python3 -c "import app; print('OK')"
```

Expected: `OK`. If any error: read carefully. Most likely causes:
- Missing import of new symbol (add it)
- Typo in a callback Output id (grep for the id)
- `allow_duplicate` missing on a modal Output (add it)
- `prevent_initial_call=False` combined with `allow_duplicate=True` (change to `True`)

- [ ] **Step 12: Run existing test suite**

```bash
cd /scratch/code/bitcoinprojections
btc_venv/bin/python3 -m pytest btc_web/test_palette_roundtrip.py -v
```

Expected: most tests pass, some may fail because they assert old option ordering. Task 6 extends/fixes these tests. Acceptable: failures that are CLEARLY about old assertion strings (e.g. "expected `bub-bm-gear` at position X"). Not acceptable: ImportErrors or "component id X not found" errors — those indicate a missing wiring.

- [ ] **Step 13: Dev-server smoke test**

```bash
lsof -ti :8050 | xargs -r kill -9 ; DEV=1 nohup bash run_web.sh > /tmp/quantoshi_dev.log 2>&1 &
sleep 5 && tail -40 /tmp/quantoshi_dev.log
```

Expected: dev server starts, Dash is running on http://0.0.0.0:8050/. No traceback.

Navigate through `http://localhost:8050/1` → `/2` → `/3` → `/4` → `/5`. On each:
- Display Models / pill-bar renders.
- Mini cards are GONE.
- Gears are present next to the master entries (or next to heatmap pills via status row).
- Clicking each gear opens the correct modal.
- Inline summary spans show current modal state ("LPPL₃", "1d+1u", etc).
- Palette switch: change palette dropdown to each of cb-brian, cb-rg, cb-full, default. Display Models list stays visually identical (swatches recolor, nothing else shifts).

- [ ] **Step 14: Commit Tasks 2-5 atomically**

```bash
git status          # sanity check — review the file list
git diff --stat     # sanity check the diff size
```

Review: all changes should be in `btc_web/layout/{bubble,sim_tabs,supercharge,heatmap,display_models,common}.py`, `btc_web/callbacks/{charts,routing}.py`, `btc_web/snapshot.py`, plus the Task 1 scaffolding that was already staged.

Task 1 was already committed. Tasks 2-5 have been accumulating unstaged changes — `git status` will show many modifications across `layout/`, `callbacks/`, and `snapshot.py`. Stage explicitly and commit:

```bash
git add btc_web/layout/display_models.py btc_web/layout/bubble.py btc_web/layout/sim_tabs.py btc_web/layout/supercharge.py btc_web/layout/heatmap.py btc_web/layout/common.py btc_web/callbacks/charts.py btc_web/callbacks/routing.py btc_web/snapshot.py
git commit -m "$(cat <<'EOF'
refactor(display-models): consolidate mini cards into in-checklist gears

Eliminate the three `_{family}_config_panel` mini cards from tabs
1/2/3/4/5. Replace with the Tab 1 in-checklist-gear pattern on
Bubble/DCA/Retire/SC (via new shared `display_models_panel` builder)
and with a status row below the pill bar on Heatmap.

New architecture:
- `layout/display_models.py` exports `display_models_panel(prefix, ...)`
  and the pure `build_display_models_options(mc, prefix, summaries=...)`
  used by both initial layout and palette rebuild.
- `dcc.Store("display-model-summaries")` is the single source of truth
  for family summary strings. `compute_family_summaries` callback (29
  Inputs) writes it; three clientside Store-readers fan it out to the
  12 checklist-tab inline summary spans; heatmap status row reads it.
- Heatmap gear dispatches to modals via a single clientside callback
  reading `hm-active-model` as State.
- All modal-open callbacks use `allow_duplicate=True` +
  `prevent_initial_call=True` because the heatmap dispatcher also
  writes to the three modal `is_open` Outputs.
- Snapshot positional stability: the 15 `*-activate` tuples in
  `_SNAPSHOT_CONTROLS` + 15 keys in `_CHECKLIST_OPTIONS` stay as
  defunct placeholders. 15 hidden `dcc.Checklist` components are
  rendered unconditionally at the root of `_serve_layout` to satisfy
  `apply_snapshot` static callback registration.

Deleted:
- `_lppl_config_panel`, `_hybppl_config_panel`, `_eppl_config_panel`
- `_model_show_checklist`
- `_build_model_opts`
- `_build_bub_model_options`
- ~45 clientside callbacks (activate↔selector mirrors, configure-btn
  scroll handlers, hm-activate↔hm-active-model syncs,
  `update_*_summary` legacy writers)
- 15 `*-activate` entries from `_TAB_CONTROLS`

Preserved (BM carve-out):
- `bub-bm-activate` + `bub-bm-body` + their mirror callbacks
- `bub-bm-gear` scroll handler (already existed; unchanged)

See docs/superpowers/specs/2026-04-11-display-models-consolidation-design.md

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6 — Test audit and new test coverage

**Files:**
- Modify: `btc_web/test_palette_roundtrip.py`
- Modify: `btc_web/test_models.py`
- Modify: `btc_web/test_web.py`, `btc_web/test_callbacks.py`, `btc_web/test_figures.py` (audit only; likely minimal changes)

- [ ] **Step 1: Grep for tests that reference deleted ids**

```bash
cd /scratch/code/bitcoinprojections
grep -rn "lppl-activate\|hybppl-activate\|eppl-activate\|lppl-configure-btn\|hybppl-configure-btn\|eppl-configure-btn\|-lppl-summary[^-]\|-hybppl-summary[^-]\|-eppl-summary[^-]" btc_web/test_*.py 2>&1 | tee /tmp/test_refs.txt
```

The `[^-]` suffix excludes the new `-summary-inline` ids.

**⚠ Preservation rule:** do NOT delete `test_palette_rebuild_sim_no_leaks` in `test_palette_roundtrip.py`. It asserts that dca/ret/sc checklists don't leak HybPPL family variants — still valid after the refactor and required per spec. Only modify it if its assertion references a deleted id.

- [ ] **Step 2: For each hit, decide UPDATE or DELETE**

For each test reference:
- If the test asserts "mini card is present" → delete the assertion (mini cards are gone).
- If the test asserts "activate checkbox mirrors checklist" → delete the test entirely (mirror callbacks are gone).
- If the test asserts "configure-btn click opens modal" → update to use `{prefix}-{family}-gear` instead.
- If the test touches `test_models.py` lines ~1668-1670, ~2120-2122, ~2360-2362 — rewrite those assertions to use the new gear ids.

Make the changes.

- [ ] **Step 3: Extend `test_palette_roundtrip.py`**

**Note:** test the pure builder function directly, not the `@callback`-decorated `update_model_swatches` (which may require `.__wrapped__` or mocking Dash's callback context). The builder IS the logic that matters; the callback is just Dash plumbing.

Add these new tests to `btc_web/test_palette_roundtrip.py`:

```python
@pytest.mark.parametrize("prefix,palette_key", [
    (p, pk) for p in ("bub", "dca", "ret", "sc")
             for pk in ("default", "cb-brian", "cb-rg", "cb-full")
])
def test_palette_rebuild_matches_initial(prefix, palette_key):
    """Value-list of rebuilt options on palette change matches the initial builder."""
    from layout.display_models import build_display_models_options
    flags = {
        "bub": {"include_bm_master": True},
        "dca": {},
        "ret": {"include_mc": True},
        "sc":  {},
    }[prefix]
    pal = _app_ctx.PALETTES.get(palette_key, _app_ctx.PALETTES["default"])
    mc  = pal.get("model_colors", _app_ctx.MODEL_TRACE_COLORS)
    # Two independent calls with the same mc+flags must yield identical value lists
    a = [o["value"] for o in build_display_models_options(mc, prefix, **flags)]
    b = [o["value"] for o in build_display_models_options(mc, prefix, **flags)]
    assert a == b
    # AND the value list must match the default-palette baseline (palette only
    # changes colors, not ordering)
    mc_default = _app_ctx.PALETTES["default"].get("model_colors", _app_ctx.MODEL_TRACE_COLORS)
    baseline = [o["value"] for o in build_display_models_options(mc_default, prefix, **flags)]
    assert a == baseline


@pytest.mark.parametrize("prefix,family",
    [(p, f) for p in ("bub", "dca", "ret", "sc", "hm")
            for f in ("lppl", "hybppl", "eppl")])
def test_no_mini_card_ids_anywhere(prefix, family):
    """Mini card activate ids must ONLY appear inside the defunct placeholder subtree.

    The defunct placeholder div is always emitted, so we can't just check
    `activate_id not in _live_component_ids(layout)` — we must walk the tree
    with the placeholder subtree EXCLUDED and assert the id doesn't appear.
    """
    import app  # noqa: F401 — registers Dash callbacks
    from layout import _serve_layout
    layout = _serve_layout()
    ids_outside_placeholders = _live_component_ids_excluding(
        layout, "_defunct-snapshot-placeholders"
    )
    activate_id = f"{prefix}-{family}-activate"
    assert activate_id not in ids_outside_placeholders, (
        f"{activate_id} leaked outside _defunct-snapshot-placeholders — "
        f"a real checklist is still emitting it"
    )
    # configure-btn ids are always forbidden (no placeholder carve-out)
    assert f"{prefix}-{family}-configure-btn" not in _live_component_ids(layout)


def test_palette_summary_not_stale():
    """Palette-rebuild path must bake current modal state into re-rendered labels.

    Regression guard for the whole staleness-fix architecture (Risk #2 in spec).
    When update_model_swatches rebuilds checklist options, the inline summary
    spans in the new label trees must carry the summaries dict, not the
    builder's static defaults.
    """
    from layout.display_models import build_display_models_options

    mc = _app_ctx.PALETTES["default"].get("model_colors", _app_ctx.MODEL_TRACE_COLORS)
    custom_summaries = {
        "lppl":   "CUSTOM_LPPL",
        "hybppl": "CUSTOM_HYB",
        "eppl":   "CUSTOM_EPPL",
    }
    opts = build_display_models_options(
        mc, "bub", include_bm_master=True, summaries=custom_summaries,
    )

    # Walk each option's label tree collecting (id, children) pairs
    collected = {}
    def _visit(node):
        nid = getattr(node, "id", None)
        if isinstance(nid, str) and nid.endswith("-summary-inline"):
            collected[nid] = getattr(node, "children", None)
    for opt in opts:
        _walk_component(opt["label"], _visit)

    assert collected.get("bub-lppl-summary-inline")   == "CUSTOM_LPPL"
    assert collected.get("bub-hybppl-summary-inline") == "CUSTOM_HYB"
    assert collected.get("bub-eppl-summary-inline")   == "CUSTOM_EPPL"


@pytest.mark.parametrize("prefix,family",
    [(p, f) for p in ("bub", "dca", "ret", "sc")
            for f in ("lppl", "hybppl", "eppl")])
def test_inline_summary_spans_exist(prefix, family):
    """All 12 {prefix}-{family}-summary-inline spans must exist in the layout."""
    import app  # noqa: F401 — registers Dash callbacks
    from layout import _serve_layout
    layout = _serve_layout()
    assert f"{prefix}-{family}-summary-inline" in _live_component_ids(layout)


def test_heatmap_status_row_exists():
    """Heatmap has the 4 new status-row ids."""
    import app  # noqa: F401 — registers Dash callbacks
    from layout import _serve_layout
    layout = _serve_layout()
    ids = _live_component_ids(layout)
    for cid in ("hm-active-family-row", "hm-active-family-label",
                "hm-active-family-summary-inline", "hm-active-family-gear"):
        assert cid in ids, f"missing {cid} in heatmap layout"


def test_defunct_placeholders_unconditional():
    """Placeholder div is emitted and contains all 15 activate ids."""
    import app  # noqa: F401 — registers Dash callbacks
    from layout import _serve_layout
    layout = _serve_layout()
    ids = _live_component_ids(layout)
    for prefix in ("bub", "dca", "ret", "sc", "hm"):
        for family in ("lppl", "hybppl", "eppl"):
            assert f"{prefix}-{family}-activate" in ids, \
                f"placeholder {prefix}-{family}-activate missing"


def test_modal_open_callbacks_use_gear_inputs():
    """Static source check: modal-open callbacks no longer reference -configure-btn ids."""
    import pathlib
    src = pathlib.Path("btc_web/callbacks/charts.py").read_text()
    for prefix in ("bub", "dca", "ret", "sc", "hm"):
        for family in ("lppl", "hybppl", "eppl"):
            assert f"{prefix}-{family}-configure-btn" not in src, \
                f"stale {prefix}-{family}-configure-btn reference in charts.py"


def test_old_snapshot_link_decodes_cleanly():
    """A pre-refactor q3: link with *-activate keys decodes without error."""
    import json, gzip, base64
    from snapshot import _decode_snapshot, _SNAPSHOT_CONTROLS
    # Build a minimal state array with all *-activate positions set to ["yes"]
    values = []
    for cid, prop in _SNAPSHOT_CONTROLS:
        if "-activate" in cid:
            values.append(["yes"])
        else:
            values.append(None)
    payload = [values, []]
    encoded = base64.urlsafe_b64encode(
        gzip.compress(json.dumps(payload, separators=(',', ':')).encode())
    ).decode()
    result = _decode_snapshot(encoded)
    assert result is not None
    # The activate keys decoded but restore_from_url writes land on hidden
    # placeholders (no visible effect, no error).
```

Add these helpers at the top of the test file:
```python
def _walk_component(node, visit):
    """Walk a Dash layout tree, calling `visit(node)` for every component.

    Recurses into:
      - `.children` (scalar, list, or tuple of children)
      - `.options[i]["label"]` for components that have `options` (Checklist
        etc.), because Display Models inline summary spans and gear buttons
        live NESTED inside Checklist option labels — not in `.children`.
      - Arbitrary dict/list structures encountered along the way.
    """
    if node is None:
        return
    if isinstance(node, (list, tuple)):
        for sub in node:
            _walk_component(sub, visit)
        return
    if isinstance(node, dict):
        for v in node.values():
            _walk_component(v, visit)
        return
    if hasattr(node, "id") or hasattr(node, "children"):
        visit(node)
        # children
        c = getattr(node, "children", None)
        if c is not None:
            _walk_component(c, visit)
        # options (Checklist, RadioItems, Dropdown, etc.) — labels are
        # where our nested summary spans and gear buttons live.
        opts = getattr(node, "options", None)
        if opts is not None and isinstance(opts, (list, tuple)):
            for opt in opts:
                if isinstance(opt, dict):
                    label = opt.get("label")
                    if label is not None:
                        _walk_component(label, visit)


def _live_component_ids(layout):
    """Collect every component id in the layout tree, including ids
    nested inside Checklist option labels (where summary-inline spans
    and gear buttons live)."""
    ids = set()
    def _visit(node):
        nid = getattr(node, "id", None)
        if isinstance(nid, str):
            ids.add(nid)
    _walk_component(layout, _visit)
    return ids


def _live_component_ids_excluding(layout, exclude_id):
    """Collect ids EXCEPT those inside the subtree rooted at `exclude_id`.

    Used by `test_no_mini_card_ids_anywhere` to assert that activate ids
    appear ONLY inside `_defunct-snapshot-placeholders`, not anywhere else
    in the layout.
    """
    ids = set()
    def _walk(node, inside_excluded):
        if node is None:
            return
        if isinstance(node, (list, tuple)):
            for sub in node:
                _walk(sub, inside_excluded)
            return
        if isinstance(node, dict):
            for v in node.values():
                _walk(v, inside_excluded)
            return
        if not (hasattr(node, "id") or hasattr(node, "children")):
            return
        nid = getattr(node, "id", None)
        # Entering the excluded subtree?
        now_excluded = inside_excluded or (nid == exclude_id)
        if isinstance(nid, str) and not now_excluded:
            ids.add(nid)
        c = getattr(node, "children", None)
        if c is not None:
            _walk(c, now_excluded)
        opts = getattr(node, "options", None)
        if opts is not None and isinstance(opts, (list, tuple)):
            for opt in opts:
                if isinstance(opt, dict):
                    label = opt.get("label")
                    if label is not None:
                        _walk(label, now_excluded)
    _walk(layout, False)
    return ids
```

**Walker correctness note:** the naive walker that only recurses `.children` will MISS the inline summary spans and gear buttons (they live inside `Checklist.options[i]["label"]`, not under `.children`). Every test that relies on `_live_component_ids` depends on the `.options[i]["label"]` recursion. Do not simplify it away.

- [ ] **Step 4: Run the extended test suite**

```bash
cd /scratch/code/bitcoinprojections
btc_venv/bin/python3 -m pytest btc_web/test_palette_roundtrip.py -v
```

Expected: all new tests pass. Total count grows from 18 to ~50 (depends on exact parametrize count).

- [ ] **Step 5: Run the affected existing test files**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_models.py btc_web/test_web.py btc_web/test_callbacks.py btc_web/test_figures.py -v 2>&1 | tail -80
```

Expected: pass. If failures, read and fix.

- [ ] **Step 6: Commit Task 6**

Only stage test files that were ACTUALLY modified by Step 2 (the audit may have found zero references in some files — those files stay out of the commit). Check `git status` first:

```bash
git status btc_web/test_*.py
```

Then `git add` only the files shown as modified. At minimum, `test_palette_roundtrip.py` will be modified (new tests added). The others may or may not be:

```bash
# Example — adjust to match actual modified files
git add btc_web/test_palette_roundtrip.py
git commit -m "$(cat <<'EOF'
test(display-models): extend palette_roundtrip coverage for consolidation

Add tests covering the new Display Models architecture:
- All 4 checklist tabs parametrized for palette-rebuild value-list match
- No mini card ids (activate/configure-btn/plain summary) remain in layout
- All 12 {prefix}-{family}-summary-inline spans exist
- Heatmap has the 4 new status-row ids
- Defunct placeholder div survives _serve_layout
- Modal-open callbacks use only -gear Inputs (no -configure-btn)
- Pre-refactor q3: links with *-activate keys decode cleanly (positional stability)

Update test_models.py assertions that referenced deleted mini card ids.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7 — Full suite + local smoke test (no commit)

- [ ] **Step 1: Run the full test suite**

```bash
cd /scratch/code/bitcoinprojections
btc_venv/bin/python3 -m pytest btc_web/ -q --tb=short 2>&1 | tail -30
```

Expected: all green (~900+ tests). Any failure halts deploy.

- [ ] **Step 2: Dev-server smoke test (exhaustive)**

```bash
lsof -ti :8050 | xargs -r kill -9 ; DEV=1 nohup bash run_web.sh > /tmp/quantoshi_dev.log 2>&1 &
sleep 5 && tail -40 /tmp/quantoshi_dev.log
```

Navigate to each of `/1`, `/2`, `/3`, `/4`, `/5` and manually verify:

**Each tab:**
- Display Models section renders.
- Mini cards (LPPL / HybPPL / EPPL) are GONE.
- BM card on bubble is still present (bub-bm-activate in header).
- Gears are visible next to master entries (Bubble Model [bub only], LPPL, Hybrid PPL, Entropy PPL).
- Clicking each gear opens the correct modal.
- Inline summary text matches modal state (e.g. "LPPL (LPPL₃)").
- Change modal settings → summary updates live.
- Close modal → summary persists.

**Palette round-trip:**
- Default → cb-brian → cb-rg → cb-full → default.
- On each transition, Display Models list stays visually identical (only swatch colors change).
- Inline summaries persist through palette changes (this is the staleness fix).

**Heatmap specifically:**
- Pill bar renders, one pill active.
- Click LPPL pill → status row appears below pill bar with "Active: LPPL (LPPL₃) ⚙".
- Click gear → LPPL modal opens.
- Click HybPPL pill → status row updates.
- Click BM pill → status row hides.
- Click a pill while a modal is open → modal stays open, pill changes.

**Race test (Risk #9):**
- Change palette → immediately toggle `lppl-weighted` (via modal). Inline summary should end on the new value. If it doesn't, document and flag.

**Mobile viewport:**
- Resize browser to <768px width.
- Display Models section stacks cleanly.
- Heatmap status row wraps without overflow.

- [ ] **Step 3: Report smoke-test results**

Paste a pass/fail report into chat:
```
Task 7 Smoke Test Report
========================
Full test suite: PASSED (N tests)

Per-tab manual verification:
  Tab 1 (Bubble):    PASS
  Tab 2 (Heatmap):   PASS
  Tab 3 (DCA):       PASS
  Tab 4 (Retire):    PASS
  Tab 5 (Supercharger): PASS

Palette round-trip:  PASS
Heatmap pill + gear: PASS
Race test:           PASS (or FLAG)
Mobile viewport:     PASS

Ready for user approval to deploy: YES/NO
```

**No commit in Task 7.** Wait for user approval before any `git push` or deploy.

---

## Deployment (separate step, user-initiated only)

Do NOT deploy automatically. Per CLAUDE.md user preferences: "Never auto-deploy. Stop at committing locally. Only push/deploy when explicitly asked."

When the user says "deploy":
```bash
git push origin master && ssh root@89.167.70.45 "cd /opt/quantoshi && git pull && redis-cli FLUSHDB && systemctl restart quantoshi"
```

---

## Rollback

If any task 2-5 fails in a way that can't be cleanly fixed, abort the atomic commit:
```bash
git reset --hard HEAD   # reverts staged changes from Tasks 2-5
```

Task 1's scaffolding commit stays — it's inert and doesn't break anything. The system returns to "pre-refactor + placeholder div + empty Store" state.

Document the failure mode and re-plan before retrying.
