# Display Models Standardization — Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collapse the Heatmap tab's 15+ pill buttons into a standardized set matching the other tabs. Adds a single "LPPL" master pill that delegates flavor selection to the Phase 1 modal. Removes QR, Exp, S2F, and all individual LPPL-family pills.

**Architecture:** Rewrite `_hm_pill_bar()` to emit a fixed set of pills. Update `_HM_PILL_MODELS` / `_HM_PILL_IDS` in routing to match. Reuse Phase 1 helpers (`_lppl_config_panel("hm")`, `_resolve_lppl_master`, `_global_lppl_modal`). Extend Phase 1 modal-open callback to include the heatmap's Configure button. Chart callback `update_heatmap` gains 3 new State inputs (global LPPL config) and translates `"lppl"` → flavor key before building the non-MC heatmap figure.

**Tech Stack:** Plotly Dash 4.0.0, Dash Bootstrap Components 2.0.4, clientside callbacks, pytest.

**Spec:** `docs/superpowers/specs/2026-04-04-display-models-standardization-phase2.md`
**Depends on:** Phase 1 complete.

---

## Pre-flight

```bash
# Verify Phase 1 is shipped
git log --oneline -30 | grep "single master .LPPL. gate\|Phase 1"
ls docs/superpowers/specs/2026-04-04-display-models-standardization-phase2.md

# Verify the global modal exists
grep -c "lppl-config-modal" btc_web/layout/common.py
# Expected: 2 or 3 (modal id + close button wiring)

btc_venv/bin/python3 -m pytest btc_web/test_web.py -q
# Expected: all passing from Phase 1
```

---

## Task 1: Rewrite `_hm_pill_bar` to emit the new standardized set

**Files:** `btc_web/layout/heatmap.py`

- [ ] **Step 1: Locate current `_hm_pill_bar`**

Run: `grep -n "def _hm_pill_bar\|hm-pill-" btc_web/layout/heatmap.py | head -15`

- [ ] **Step 2: Replace the function body**

In `btc_web/layout/heatmap.py`, replace the existing `_hm_pill_bar()` with:

```python
def _hm_pill_bar():
    """Model-selector pill bar — standardized set (BM, PL, LPPL master,
    LinPPL, HybPPL, EF, U1, MC). LPPL family variants collapse into the
    single LPPL master pill; the actual flavor is chosen via the global
    LPPL config modal."""
    mc = _app_ctx.PALETTES["default"]["model_colors"]

    def _pill_label(key, display_name):
        return html.Span([
            html.Span(" ", style={
                "display": "inline-block", "width": "8px", "height": "8px",
                "borderRadius": "2px", "verticalAlign": "middle",
                "marginRight": "4px",
                "backgroundColor": mc.get(key, "#888"),
            }),
            display_name,
        ])

    buttons = [
        dbc.Button(_pill_label("bub", "BM"),
                   id="hm-pill-bub", color="primary", size="sm"),
        dbc.Button(_pill_label("pl", "PL"),
                   id="hm-pill-pl", outline=True, color="primary", size="sm"),
        dbc.Button(_pill_label("lppl", "LPPL"),
                   id="hm-pill-lppl", outline=True, color="primary", size="sm"),
        dbc.Button(_pill_label("linppl", "LinPPL"),
                   id="hm-pill-linppl", outline=True, color="primary", size="sm"),
        dbc.Button(_pill_label("hybppl", "HybPPL"),
                   id="hm-pill-hybppl", outline=True, color="primary", size="sm"),
    ]
    if "ef" in _app_ctx.PRICE_MODELS:
        buttons.append(
            dbc.Button(_pill_label("ef", "EF"),
                       id="hm-pill-ef", outline=True, color="primary", size="sm"),
        )
    if "u1" in _app_ctx.PRICE_MODELS:
        buttons.append(
            dbc.Button(_pill_label("u1", "U\u2081"),
                       id="hm-pill-u1", outline=True, color="primary", size="sm"),
        )
    if _app_ctx._HAS_MARKOV:
        buttons.append(
            dbc.Button("MC", id="hm-pill-mc", outline=True,
                       color="warning", size="sm"),
        )

    return html.Div([
        dbc.ButtonGroup(buttons, size="sm"),
    ], className="mb-1 text-center")
```

- [ ] **Step 3: Verify syntax**

Run: `btc_venv/bin/python3 -m py_compile btc_web/layout/heatmap.py && echo OK`

- [ ] **Step 4: Commit (will still break pill callbacks — fix in next task)**

```bash
git add btc_web/layout/heatmap.py
git commit -m "feat(heatmap): shrink pill bar to standardized set

Single LPPL master pill collapses the 10 flavor variants. Removes
QR, Exp, S2F pills (display-only models belong on Bubble tab).

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Update `_HM_PILL_MODELS` / `_HM_PILL_IDS` to match new pill set

**Files:** `btc_web/callbacks/routing.py`

- [ ] **Step 1: Locate the pill-model list**

Run: `grep -n "_HM_PILL_MODELS\|_HM_PILL_IDS" btc_web/callbacks/routing.py | head -10`

- [ ] **Step 2: Replace the list with the new hardcoded ordering**

In `btc_web/callbacks/routing.py`, replace:
```python
_HM_PILL_MODELS = ["bub"] + [k for k in _app_ctx.PRICE_MODELS if k not in ("bub", "mc")]
if _app_ctx._HAS_MARKOV:
    _HM_PILL_MODELS.append("mc")
# Log the mapping for CLAUDE.md reference
# /2.1=bub, /2.2=qr, /2.3=pl, /2.4=lppl, /2.5=exp, /2.6=ef (if loaded), /2.7=s2f, /2.N+1=mc
```

with:
```python
# Standardized pill set (Phase 2). Tab 2 deep-link routes /2.N are renumbered;
# old URLs will land on different models — accepted per design decision.
# /2.1=bub, /2.2=pl, /2.3=lppl (master), /2.4=linppl, /2.5=hybppl,
# /2.6=ef (if loaded), /2.7=u1 (if loaded), /2.N+1=mc (if HAS_MARKOV)
_HM_PILL_MODELS = ["bub", "pl", "lppl", "linppl", "hybppl"]
if "ef" in _app_ctx.PRICE_MODELS:
    _HM_PILL_MODELS.append("ef")
if "u1" in _app_ctx.PRICE_MODELS:
    _HM_PILL_MODELS.append("u1")
if _app_ctx._HAS_MARKOV:
    _HM_PILL_MODELS.append("mc")

# Map removed pill IDs (Phase 1 share links may have these in hm-active-model)
# → surviving pill. Used as a graceful fallback when old snapshot decodes.
_HM_LEGACY_MODEL_FALLBACK = {
    "qr": "pl",        # QR was bands-only; PL is the closest match
    "lp2": "lppl", "lp3": "lppl", "lp4": "lppl",
    "lppl_w": "lppl", "lp2_w": "lppl", "lp3_w": "lppl", "lp4_w": "lppl",
    "lp4_n13": "lppl", "lp4_w_n13": "lppl",
    "exp": "bub",      # display-only demo
    "s2f": "bub",      # display-only demo
}
```

Leave `_HM_PILL_IDS = [f"hm-pill-{k}" for k in _HM_PILL_MODELS]` unchanged — it regenerates automatically.

- [ ] **Step 2b: Add legacy-value fallback in `_hm_pill_sync`**

Still in `btc_web/callbacks/routing.py`, find the `_hm_pill_sync` callback. It currently reads `hm-active-model` (a string) and sets the `outline` property on each pill button. When the stored value is a legacy key like `"qr"` or `"lp3_w"`, no pill matches → no pill lit up. Insert a normalization step at the top of the callback:

```python
def _hm_pill_sync(active_model):
    # Normalize legacy snapshot values to a surviving pill key
    if active_model not in _HM_PILL_MODELS:
        active_model = _HM_LEGACY_MODEL_FALLBACK.get(active_model, "bub")
    # ... rest of existing callback ...
```

This ensures old share links still render with a valid pill highlighted.

- [ ] **Step 3: Verify `_hm_pill_click` and `_hm_pill_sync` still bind correctly**

Run: `grep -B2 -A10 "_hm_pill_click\|_hm_pill_sync" btc_web/callbacks/routing.py | head -60`
Expected: both callbacks iterate `_HM_PILL_IDS` dynamically, so they adapt automatically.

- [ ] **Step 4: Restart + verify**

```bash
lsof -ti :8050 | xargs -r kill -9 2>/dev/null; sleep 1
DEV=1 nohup bash run_web.sh > /tmp/quantoshi_dev.log 2>&1 &
sleep 8
grep -i "traceback\|error" /tmp/quantoshi_dev.log || echo clean
```

Open http://localhost:8050/2 (Heatmap). Verify pill bar has only: BM, PL, LPPL, LinPPL, HybPPL, EF, U₁, MC. Click each pill to verify the heatmap redraws.

- [ ] **Step 5: Commit**

```bash
git add btc_web/callbacks/routing.py
git commit -m "chore(routing): update _HM_PILL_MODELS for standardized pill set

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Add `_lppl_config_panel("hm")` to heatmap layout

**Files:** `btc_web/layout/heatmap.py`

- [ ] **Step 1: Locate where pill bar is inserted into the tab layout**

Run: `grep -n "_hm_pill_bar\|_heatmap_tab" btc_web/layout/heatmap.py`

- [ ] **Step 2: Import `_lppl_config_panel`**

At top of `btc_web/layout/heatmap.py`, update import:
```python
from layout.common import (..., _lppl_config_panel)
```

- [ ] **Step 3: Insert the panel below the pill bar**

In `_heatmap_tab()`, find the line where `_hm_pill_bar()` is added to the column, and add `_lppl_config_panel("hm")` right after it:

```python
# Model selector pills
_hm_pill_bar(),

dcc.Store(id="hm-active-model", storage_type="memory", data="bub"),

# LPPL sub-config panel (compact — links to global modal)
_lppl_config_panel("hm"),
```

- [ ] **Step 4: Restart + verify**

```bash
lsof -ti :8050 | xargs -r kill -9 2>/dev/null; sleep 1
DEV=1 nohup bash run_web.sh > /tmp/quantoshi_dev.log 2>&1 &
sleep 8
curl -sS http://localhost:8050/_dash-layout | grep -c "hm-lppl-activate"
```
Expected: 1.

- [ ] **Step 5: Commit**

```bash
git add btc_web/layout/heatmap.py
git commit -m "feat(heatmap): add LPPL sub-config panel

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Extend modal-open callback to include heatmap's Configure button

**Files:** `btc_web/callbacks/charts.py`

- [ ] **Step 1: Locate the existing modal-open callback (from Phase 1)**

Run: `grep -B2 -A20 "lppl-config-modal.*is_open" btc_web/callbacks/charts.py | head -40`

- [ ] **Step 2: Add the hm-lppl-configure-btn Input**

In the clientside_callback for the modal, add `Input("hm-lppl-configure-btn", "n_clicks")` to the Input list, update the JS function signature to accept the new arg, and add `hm_n` to the callback parameter list. Current (Phase 1):

```python
_app_ctx.app.clientside_callback(
    """
    function(bub_n, dca_n, ret_n, sc_n, close_n, cur_open) {
        var ctx = window.dash_clientside.callback_context;
        if (!ctx.triggered || !ctx.triggered.length) {
            return window.dash_clientside.no_update;
        }
        var src = ctx.triggered[0].prop_id;
        if (src.indexOf('lppl-modal-close-btn') !== -1) return false;
        if (src.indexOf('lppl-configure-btn') !== -1) return true;
        return window.dash_clientside.no_update;
    }
    """,
    Output("lppl-config-modal", "is_open"),
    Input("bub-lppl-configure-btn", "n_clicks"),
    Input("dca-lppl-configure-btn", "n_clicks"),
    Input("ret-lppl-configure-btn", "n_clicks"),
    Input("sc-lppl-configure-btn", "n_clicks"),
    Input("lppl-modal-close-btn", "n_clicks"),
    State("lppl-config-modal", "is_open"),
    prevent_initial_call=True,
)
```

Update to:

```python
_app_ctx.app.clientside_callback(
    """
    function(bub_n, dca_n, ret_n, sc_n, hm_n, close_n, cur_open) {
        var ctx = window.dash_clientside.callback_context;
        if (!ctx.triggered || !ctx.triggered.length) {
            return window.dash_clientside.no_update;
        }
        var src = ctx.triggered[0].prop_id;
        if (src.indexOf('lppl-modal-close-btn') !== -1) return false;
        if (src.indexOf('lppl-configure-btn') !== -1) return true;
        return window.dash_clientside.no_update;
    }
    """,
    Output("lppl-config-modal", "is_open"),
    Input("bub-lppl-configure-btn", "n_clicks"),
    Input("dca-lppl-configure-btn", "n_clicks"),
    Input("ret-lppl-configure-btn", "n_clicks"),
    Input("sc-lppl-configure-btn", "n_clicks"),
    Input("hm-lppl-configure-btn", "n_clicks"),
    Input("lppl-modal-close-btn", "n_clicks"),
    State("lppl-config-modal", "is_open"),
    prevent_initial_call=True,
)
```

- [ ] **Step 2b: Extend the summary-text loop**

The Phase 1 summary-updater loop currently iterates `("bub", "dca", "ret", "sc")`. Add `"hm"` to that tuple:

Find:
```python
for _sum_prefix in ("bub", "dca", "ret", "sc"):
```
Replace with:
```python
for _sum_prefix in ("bub", "dca", "ret", "sc", "hm"):
```

- [ ] **Step 3: Restart + verify**

```bash
lsof -ti :8050 | xargs -r kill -9 2>/dev/null; sleep 1
DEV=1 nohup bash run_web.sh > /tmp/quantoshi_dev.log 2>&1 &
sleep 8
grep -i "traceback" /tmp/quantoshi_dev.log || echo clean
```

Open http://localhost:8050/2 (Heatmap), click the "⚙️ Configure LPPL" button. Modal should open. Close it.

- [ ] **Step 4: Commit**

```bash
git add btc_web/callbacks/charts.py
git commit -m "feat(modal): heatmap Configure button opens global LPPL modal

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Sync callback — LPPL pill active ↔ hm-lppl-activate

**Files:** `btc_web/callbacks/charts.py`

- [ ] **Step 1: Add bi-directional sync callbacks**

Append to `btc_web/callbacks/charts.py`:

```python
# hm-active-model == "lppl"  ->  hm-lppl-activate
_app_ctx.app.clientside_callback(
    """
    function(active_model, cur_activate) {
        var should_activate = (active_model === 'lppl');
        var is_activated = (cur_activate || []).length > 0;
        if (should_activate === is_activated) {
            return window.dash_clientside.no_update;
        }
        return should_activate ? ['yes'] : [];
    }
    """,
    Output("hm-lppl-activate", "value", allow_duplicate=True),
    Input("hm-active-model", "data"),
    State("hm-lppl-activate", "value"),
    prevent_initial_call='initial_duplicate',
)

# hm-lppl-activate  ->  hm-active-model  (user clicks Activate LPPL)
_app_ctx.app.clientside_callback(
    """
    function(activate, cur_model) {
        var want_lppl = (activate || []).length > 0;
        var is_lppl = (cur_model === 'lppl');
        if (want_lppl === is_lppl) return window.dash_clientside.no_update;
        if (want_lppl) return 'lppl';
        // Turn off: revert to BM
        return 'bub';
    }
    """,
    Output("hm-active-model", "data", allow_duplicate=True),
    Input("hm-lppl-activate", "value"),
    State("hm-active-model", "data"),
    prevent_initial_call='initial_duplicate',
)
```

- [ ] **Step 2: Restart + Playwright verify**

```bash
lsof -ti :8050 | xargs -r kill -9 2>/dev/null; sleep 1
DEV=1 nohup bash run_web.sh > /tmp/quantoshi_dev.log 2>&1 &
sleep 8
grep -i "traceback" /tmp/quantoshi_dev.log || echo clean
```

In browser on /2: click LPPL pill → verify hm-lppl-activate becomes checked + summary text updates. Click BM pill → verify hm-lppl-activate unchecks.

- [ ] **Step 3: Commit**

```bash
git add btc_web/callbacks/charts.py
git commit -m "feat(heatmap): LPPL pill <-> hm-lppl-activate bi-directional sync

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: `update_heatmap` — translate LPPL master to flavor

**Files:** `btc_web/callbacks/charts.py`, `btc_web/test_web.py`

- [ ] **Step 1: Update `TestUpdateHeatmapCallback` signatures**

In `btc_web/test_web.py`, find `TestUpdateHeatmapCallback`. Each `update_heatmap(...)` call has a large arg list. Add these kwargs to the end of each call:

```python
lppl_n_freqs=[3], lppl_weighted=[], lppl_no_13=[],
```

- [ ] **Step 2: Locate update_heatmap signature in charts.py**

Run: `grep -n "def update_heatmap" btc_web/callbacks/charts.py`

- [ ] **Step 3: Add State inputs**

In the `@callback` decorator for `update_heatmap`, after the existing State inputs (but inside the decorator), add:

```python
State("lppl-n-freqs", "value"),
State("lppl-weighted", "value"),
State("lppl-no-13", "value"),
```

Update the function signature to accept them (add as last 3 positional parameters before any existing `user_model_store=None` etc.).

- [ ] **Step 4: Translate `hm_model` before calling figure builder**

In the `update_heatmap` body, find where `hm_model` is passed to `_get_heatmap_fig` or similar. Insert translation BEFORE the figure builder call:

```python
    # Translate LPPL master pill to specific flavor via global config.
    # Only for the non-MC path — MC cache uses its own model-src dropdown.
    if hm_model == "lppl":
        _weighted = "weighted" in (lppl_weighted or [])
        _no_13 = "no13" in (lppl_no_13 or [])
        _n_list = (lppl_n_freqs or [3])
        _n = _n_list[0] if _n_list else 3  # first checked entry
        if _n == 1:
            hm_model = "lppl_w" if _weighted else "lppl"
        elif _n == 2:
            hm_model = "lp2_w" if _weighted else "lp2"
        elif _n == 3 and not _no_13:
            hm_model = "lp3_w" if _weighted else "lp3"
        elif _n == 4:
            if _no_13:
                hm_model = "lp4_w_n13" if _weighted else "lp4_n13"
            else:
                hm_model = "lp4_w" if _weighted else "lp4"
        # else: keep hm_model="lppl" (LP1 fallback)
```

- [ ] **Step 5: Run tests**

Run: `btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestUpdateHeatmapCallback -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add btc_web/callbacks/charts.py btc_web/test_web.py
git commit -m "feat(heatmap): translate LPPL master to specific flavor

Non-MC path only. MC cache uses its own hm-mc-model-src dropdown
and stays flavor-agnostic.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Per-pill swatch update on palette change

**Files:** `btc_web/callbacks/charts.py`

- [ ] **Step 1: Add palette-driven pill-swatch callback**

Append to `btc_web/callbacks/charts.py`:

```python
# Heatmap pill swatches — update children (swatch + label) on palette change.
# Per-pill Output to avoid rebuilding the pill bar container, which would
# invalidate _hm_pill_click / _hm_pill_sync bindings.
def _hm_pill_label_html(key, mc):
    from dash import html
    _label_by_key = {
        "bub": "BM", "pl": "PL", "lppl": "LPPL",
        "linppl": "LinPPL", "hybppl": "HybPPL",
        "ef": "EF", "u1": "U\u2081", "mc": "MC",
    }
    return html.Span([
        html.Span(" ", style={
            "display": "inline-block", "width": "8px", "height": "8px",
            "borderRadius": "2px", "verticalAlign": "middle",
            "marginRight": "4px",
            "backgroundColor": mc.get(key, "#888"),
        }),
        _label_by_key.get(key, key),
    ])


from callbacks.routing import _HM_PILL_MODELS  # noqa: E402

@callback(
    [Output(f"hm-pill-{k}", "children") for k in _HM_PILL_MODELS],
    Input("palette-store", "data"),
    prevent_initial_call=True,
)
def update_heatmap_pill_swatches(palette_key):
    pal = _app_ctx.PALETTES.get(palette_key or "default", _app_ctx.PALETTES["default"])
    mc = pal.get("model_colors", _app_ctx.MODEL_TRACE_COLORS)
    return [_hm_pill_label_html(k, mc) for k in _HM_PILL_MODELS]
```

- [ ] **Step 2: Restart + verify in browser**

On /2, change palette via navbar dropdown, verify pill swatches update without breaking pill click behavior.

```bash
lsof -ti :8050 | xargs -r kill -9 2>/dev/null; sleep 1
DEV=1 nohup bash run_web.sh > /tmp/quantoshi_dev.log 2>&1 &
sleep 8
grep -i "traceback" /tmp/quantoshi_dev.log || echo clean
```

- [ ] **Step 3: Commit**

```bash
git add btc_web/callbacks/charts.py
git commit -m "feat(heatmap): update pill swatches on palette change

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Unit tests for heatmap LPPL master translation

**Files:** `btc_web/test_web.py`

- [ ] **Step 1: Extract translation logic into a testable helper**

Since the translation in Task 6 is inline in `update_heatmap`, extract it into a standalone helper in `btc_web/callbacks/charts.py` for direct unit testing:

```python
def _resolve_hm_lppl_master(hm_model, lppl_n_freqs, lppl_weighted, lppl_no_13):
    """Translate 'lppl' master to specific flavor key for heatmap (single-select).

    Returns the flavor key the heatmap figure builder should use. For non-lppl
    models or when lppl flavor cannot be resolved, returns input unchanged.
    """
    if hm_model != "lppl":
        return hm_model
    _weighted = "weighted" in (lppl_weighted or [])
    _no_13 = "no13" in (lppl_no_13 or [])
    _n_list = (lppl_n_freqs or [3])
    _n = _n_list[0] if _n_list else 3
    if _n == 1:
        return "lppl_w" if _weighted else "lppl"
    if _n == 2:
        return "lp2_w" if _weighted else "lp2"
    if _n == 3 and not _no_13:
        return "lp3_w" if _weighted else "lp3"
    if _n == 4:
        if _no_13:
            return "lp4_w_n13" if _weighted else "lp4_n13"
        return "lp4_w" if _weighted else "lp4"
    return "lppl"  # fallback
```

Then Task 6's inline translation in `update_heatmap` collapses to:
```python
hm_model = _resolve_hm_lppl_master(
    hm_model, lppl_n_freqs, lppl_weighted, lppl_no_13)
```

- [ ] **Step 2: Add real tests against the helper**

Append to `btc_web/test_web.py`:

```python
class TestResolveHmLpplMaster:
    """Unit test for heatmap LPPL master translation."""

    def test_non_lppl_passes_through(self):
        from callbacks.charts import _resolve_hm_lppl_master
        assert _resolve_hm_lppl_master("bub", [3], [], []) == "bub"
        assert _resolve_hm_lppl_master("pl", [3], [], []) == "pl"
        assert _resolve_hm_lppl_master("linppl", [3], [], []) == "linppl"

    def test_lppl_default_n3_unweighted(self):
        from callbacks.charts import _resolve_hm_lppl_master
        assert _resolve_hm_lppl_master("lppl", [3], [], []) == "lp3"

    def test_lppl_n3_weighted(self):
        from callbacks.charts import _resolve_hm_lppl_master
        assert _resolve_hm_lppl_master("lppl", [3], ["weighted"], []) == "lp3_w"

    def test_lppl_n4_no_13(self):
        from callbacks.charts import _resolve_hm_lppl_master
        assert _resolve_hm_lppl_master("lppl", [4], [], ["no13"]) == "lp4_n13"

    def test_lppl_n4_weighted_no_13(self):
        from callbacks.charts import _resolve_hm_lppl_master
        assert _resolve_hm_lppl_master("lppl", [4], ["weighted"], ["no13"]) == "lp4_w_n13"

    def test_lppl_picks_first_when_multi_selected(self):
        from callbacks.charts import _resolve_hm_lppl_master
        # Heatmap is single-select: takes first entry, ignores rest
        assert _resolve_hm_lppl_master("lppl", [2, 4], [], []) == "lp2"

    def test_lppl_empty_n_freqs_defaults_to_3(self):
        from callbacks.charts import _resolve_hm_lppl_master
        assert _resolve_hm_lppl_master("lppl", [], [], []) == "lp3"

    def test_lppl_n3_with_no_13_falls_through_to_lppl(self):
        from callbacks.charts import _resolve_hm_lppl_master
        # n=3 and no_13 both set → LP3 disabled → fallback to "lppl"
        assert _resolve_hm_lppl_master("lppl", [3], [], ["no13"]) == "lppl"
```

- [ ] **Step 3: Run tests**

Run: `btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestResolveHmLpplMaster -v`
Expected: 8 tests pass.

- [ ] **Step 4: Commit**

```bash
git add btc_web/callbacks/charts.py btc_web/test_web.py
git commit -m "test(heatmap): unit tests for LPPL master flavor translation

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Playwright E2E verification

**Files:** manual (no code changes)

- [ ] **Step 1: Verify pill bar**

Open http://localhost:8050/2. Count pills: BM, PL, LPPL, LinPPL, HybPPL, EF, U₁, MC (or whichever subset is available). Verify NO pills for individual LPPL flavors, Exp, S2F, QR.

- [ ] **Step 2: LPPL pill flow**

Click LPPL pill. Verify:
- LPPL pill becomes solid; others outline.
- hm-lppl-activate checkbox checks.
- Summary text shows "LPPL₃" (default).
- Heatmap redraws (should look like the old /2.6 = lp3 heatmap).

Click "⚙️ Configure LPPL". Modal opens. Toggle n_freqs to [2]. Close modal. Verify summary updates to "LPPL₂" and heatmap redraws using LPPL₂ bands.

- [ ] **Step 3: Palette swatch update**

Change palette. Verify pill swatches update but pills remain clickable.

- [ ] **Step 4: Old share link compat**

Paste a Phase 1 snapshot URL with heatmap state and verify heatmap renders (shared un-prefixed LPPL config state flows through, `hm-active-model` falls back to "bub" if not in link).

- [ ] **Step 5: Kill dev server**

```bash
lsof -ti :8050 | xargs -r kill -9 2>/dev/null
```

- [ ] **Step 6: No commit (verification only)**

---

## Post-implementation checks

- [ ] All tests pass: `btc_venv/bin/python3 -m pytest btc_web/test_web.py -q`
- [ ] Syntax check: `btc_venv/bin/python3 -m py_compile btc_web/layout/heatmap.py btc_web/callbacks/charts.py btc_web/callbacks/routing.py`
- [ ] Pill bar has expected 5-8 pills (depending on EF / U₁ / MC availability)
- [ ] LPPL pill opens + syncs with activate + summary + modal
- [ ] Palette change updates pill swatches
- [ ] MC path unaffected — MC + LPPL model-src dropdown still works as before
