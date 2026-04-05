# Display Models Standardization — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generalize the Bubble tab's LPPL-master pattern to DCA/Retire/SuperCharger; add a global LPPL config modal; wrap Bubble-tab BM panel in a collapse gate; delete S2F from Citadel; rename Bubble tab label.

**Architecture:** Shared layout helpers in `layout/common.py` emit (a) the standardized Display Models checklist with a single "LPPL" master entry, (b) a compact per-tab LPPL sub-panel (activate + summary + configure-button), and (c) a root-level LPPL config modal (holds n_freqs/weighted/no_13 with un-prefixed global IDs). Chart callbacks on DCA/Retire/SC gain 3 new Inputs reading the global modal state and reuse a shared `_resolve_lppl_master` helper to translate the master into a flavor key. Snapshot controls are append-only (existing un-prefixed IDs unchanged, new per-tab activate IDs appended).

**Tech Stack:** Plotly Dash 4.0.0, Dash Bootstrap Components 2.0.4, clientside callbacks, pytest.

**Spec:** `docs/superpowers/specs/2026-04-04-display-models-standardization-phase1.md`

---

## Pre-flight

```bash
ls docs/superpowers/specs/2026-04-04-display-models-standardization-phase1.md
git log --oneline -20 | grep "single master .LPPL. gate"
btc_venv/bin/python3 -m pytest btc_web/test_web.py -q
# Expected: 830+ passed, 5 skipped
```

---

## Task 1: Fill `_MODEL_LABELS` for LPPL flavors

**Files:** `btc_web/figures/common.py:378`

- [ ] **Step 1: Read the existing `_MODEL_LABELS` dict**

Run: `grep -A2 "_MODEL_LABELS = " btc_web/figures/common.py`

- [ ] **Step 2: Replace the dict at line 378**

In `btc_web/figures/common.py`, replace:
```python
    _MODEL_LABELS = {"bub": "BM", "qr": "QR", "pl": "PL", "lppl": "LPPL",
                     "exp": "Exp", "s2f": "S2F", "ef": "EF", "u1": "U\u2081"}
```
with:
```python
    _MODEL_LABELS = {
        "bub": "BM", "qr": "QR", "pl": "PL", "lppl": "LPPL",
        "lp2": "LPPL\u2082", "lp3": "LPPL\u2083", "lp4": "LPPL\u2084",
        "lppl_w": "LPPL (w)", "lp2_w": "LPPL\u2082 (w)",
        "lp3_w": "LPPL\u2083 (w)", "lp4_w": "LPPL\u2084 (w)",
        "lp4_n13": "LPPL\u2084 (no \u03c9\u224813)",
        "lp4_w_n13": "LPPL\u2084 (w, no \u03c9\u224813)",
        "linppl": "LinPPL", "hybppl": "HybPPL",
        "exp": "Exp", "s2f": "S2F", "ef": "EF", "u1": "U\u2081",
    }
```

- [ ] **Step 3: Verify syntax**

Run: `btc_venv/bin/python3 -m py_compile btc_web/figures/common.py && echo OK`

- [ ] **Step 4: Run existing tests**

Run: `btc_venv/bin/python3 -m pytest btc_web/test_web.py -q`
Expected: 830+ passed, 5 skipped.

- [ ] **Step 5: Commit**

```bash
git add btc_web/figures/common.py
git commit -m "feat(labels): add _MODEL_LABELS for LPPL flavor keys

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Add `_resolve_lppl_master` translation helper + tests

**Files:** `btc_web/callbacks/charts.py`, `btc_web/test_web.py`

- [ ] **Step 1: Write failing tests**

Append to `btc_web/test_web.py` (near other helper tests):

```python
class TestResolveLpplMaster:
    """Unit test for the LPPL master -> flavor translation helper."""

    def test_no_master_passes_through(self):
        from callbacks.charts import _resolve_lppl_master
        result = _resolve_lppl_master(["bub", "pl"], [3], [], [])
        assert result == ["bub", "pl"]

    def test_master_1_unweighted(self):
        from callbacks.charts import _resolve_lppl_master
        result = _resolve_lppl_master(["bub", "lppl"], [1], [], [])
        assert "lppl" in result and "bub" in result
        assert "lp2" not in result

    def test_master_3_weighted(self):
        from callbacks.charts import _resolve_lppl_master
        result = _resolve_lppl_master(["lppl"], [3], ["weighted"], [])
        assert result == ["lp3_w"]

    def test_master_3_disabled_by_no_13(self):
        from callbacks.charts import _resolve_lppl_master
        # no_13 disables LP3
        result = _resolve_lppl_master(["lppl"], [3], [], ["no13"])
        assert result == []

    def test_master_4_no_13(self):
        from callbacks.charts import _resolve_lppl_master
        result = _resolve_lppl_master(["lppl"], [4], [], ["no13"])
        assert result == ["lp4_n13"]

    def test_master_4_weighted_no_13(self):
        from callbacks.charts import _resolve_lppl_master
        result = _resolve_lppl_master(["lppl"], [4], ["weighted"], ["no13"])
        assert result == ["lp4_w_n13"]

    def test_master_all_freqs_unweighted(self):
        from callbacks.charts import _resolve_lppl_master
        result = _resolve_lppl_master(["lppl"], [1, 2, 3, 4], [], [])
        assert set(result) == {"lppl", "lp2", "lp3", "lp4"}

    def test_empty_n_freqs_strips_master(self):
        from callbacks.charts import _resolve_lppl_master
        # Master checked but no flavor selected -> master stripped with no replacement
        result = _resolve_lppl_master(["bub", "lppl"], [], [], [])
        assert result == ["bub"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestResolveLpplMaster -v`
Expected: FAIL with `ImportError: cannot import name '_resolve_lppl_master'`.

- [ ] **Step 3: Add the helper to charts.py**

Append to `btc_web/callbacks/charts.py` (near other helpers, before `update_bubble`):

```python
def _resolve_lppl_master(model_show, lppl_n_freqs, lppl_weighted, lppl_no_13):
    """Translate the 'lppl' master in model_show into specific flavor key(s).

    Strips 'lppl' from the list and appends one flavor key per checked
    n_freqs entry, applying weighted and no_13 modifiers. When no master
    is present, returns the list unchanged. When master is present but
    no flavor is selected, the master is stripped with no replacement.
    """
    model_show = list(model_show or [])
    if "lppl" not in model_show:
        return model_show
    model_show = [v for v in model_show if v != "lppl"]
    _weighted = "weighted" in (lppl_weighted or [])
    _no_13 = "no13" in (lppl_no_13 or [])
    for n in (lppl_n_freqs or []):
        if n == 1:
            model_show.append("lppl_w" if _weighted else "lppl")
        elif n == 2:
            model_show.append("lp2_w" if _weighted else "lp2")
        elif n == 3 and not _no_13:
            model_show.append("lp3_w" if _weighted else "lp3")
        elif n == 4:
            if _no_13:
                model_show.append("lp4_w_n13" if _weighted else "lp4_n13")
            else:
                model_show.append("lp4_w" if _weighted else "lp4")
    return model_show
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestResolveLpplMaster -v`
Expected: 8 tests pass.

- [ ] **Step 5: Commit**

```bash
git add btc_web/callbacks/charts.py btc_web/test_web.py
git commit -m "feat(charts): add _resolve_lppl_master translation helper

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Refactor `update_bubble` to use `_resolve_lppl_master`

**Files:** `btc_web/callbacks/charts.py` (inside `update_bubble`)

- [ ] **Step 1: Locate the existing inline translation**

Run: `grep -n '"lppl" in model_show' btc_web/callbacks/charts.py`
Expected: one match inside update_bubble.

- [ ] **Step 2: Replace inline translation with helper call**

In `btc_web/callbacks/charts.py`, inside `update_bubble`, replace the block starting with the comment about "lppl" master gate and ending at the last `model_show.append(...)` of the translation loop:

Find:
```python
    # The "lppl" entry in bub-model-show is a MASTER gate — only when it's
    # present do we consult the LPPL Models config panel to decide which
    # flavor(s) to render. Strip the master value before passing to the
    # chart so it doesn't get rendered as raw LPPL1 by mistake.
    model_show = list(model_show or [])
    if "lppl" in model_show:
        model_show = [v for v in model_show if v != "lppl"]
        _weighted = "weighted" in (lppl_weighted or [])
        _no_13 = "no13" in (lppl_no_13 or [])
        for n in (lppl_n_freqs or []):
            if n == 1:
                model_show.append("lppl_w" if _weighted else "lppl")
            elif n == 2:
                model_show.append("lp2_w" if _weighted else "lp2")
            elif n == 3 and not _no_13:  # LP3 disabled when excluding ω=13
                model_show.append("lp3_w" if _weighted else "lp3")
            elif n == 4:
                if _no_13:
                    model_show.append("lp4_w_n13" if _weighted else "lp4_n13")
                else:
                    model_show.append("lp4_w" if _weighted else "lp4")
```

Replace with:
```python
    # The "lppl" entry in bub-model-show is a MASTER gate — translate
    # to specific flavor key(s) via global LPPL config.
    model_show = _resolve_lppl_master(
        model_show, lppl_n_freqs, lppl_weighted, lppl_no_13)
```

- [ ] **Step 3: Run Bubble callback tests**

Run: `btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestUpdateBubbleCallback -v`
Expected: 3 tests pass.

- [ ] **Step 4: Commit**

```bash
git add btc_web/callbacks/charts.py
git commit -m "refactor(bubble): use _resolve_lppl_master helper

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Add `standardized` kwarg to `_model_show_checklist`

**Files:** `btc_web/layout/common.py`, `btc_web/test_web.py`

- [ ] **Step 1: Write failing tests**

Append to `btc_web/test_web.py`:

```python
class TestModelShowChecklistStandardized:
    """Unit tests for _model_show_checklist standardized=True mode."""

    def test_has_lppl_master(self):
        from layout.common import _model_show_checklist
        elems = _model_show_checklist("dca", standardized=True)
        rendered = str(elems)
        # LPPL master entry present with color swatch
        assert "\"value\": \"lppl\"" in rendered.replace("'", '"')

    def test_omits_lppl_variants(self):
        from layout.common import _model_show_checklist
        elems = _model_show_checklist("dca", standardized=True)
        rendered = str(elems)
        assert '"lp2"' not in rendered.replace("'", '"')
        assert '"lp3"' not in rendered.replace("'", '"')
        assert '"lp4"' not in rendered.replace("'", '"')
        assert '"lppl_w"' not in rendered.replace("'", '"')

    def test_omits_exp_and_s2f(self):
        from layout.common import _model_show_checklist
        elems = _model_show_checklist("dca", standardized=True)
        rendered = str(elems)
        assert '"exp"' not in rendered.replace("'", '"')
        assert '"s2f"' not in rendered.replace("'", '"')

    def test_non_standardized_unchanged(self):
        from layout.common import _model_show_checklist
        elems = _model_show_checklist("dca", standardized=False)
        rendered = str(elems)
        # Default behavior: all LPPL variants + exp + s2f present
        assert '"lppl"' in rendered.replace("'", '"')
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestModelShowChecklistStandardized -v`
Expected: FAIL (current helper has no `standardized` kwarg → TypeError).

- [ ] **Step 3: Extend `_model_show_checklist` in `btc_web/layout/common.py:445`**

Replace the existing function:
```python
def _model_show_checklist(prefix):
    """Display models checklist with palette-aware color swatches."""
    mc = _app_ctx.PALETTES["default"]["model_colors"]
    _DEPRIORITIZED = {"exp", "s2f"}

    opts = [ ... ]
    # ... builds opts ...
    return [ _lbl("Display models"), dcc.Checklist(...) ]
```

with:
```python
def _model_show_checklist(prefix, standardized=False):
    """Display models checklist with palette-aware color swatches.

    standardized=True: emits single "LPPL" master (skip individual LPPL
    family variants), omits Exp + S2F. For tabs 1/3/4/5 (Phase 1) and
    tab 2 (Phase 2) that share the standardized UX.
    """
    mc = _app_ctx.PALETTES["default"]["model_colors"]
    _DEPRIORITIZED = {"exp", "s2f"}
    _LPPL_FAM = {"lppl", "lp2", "lp3", "lp4"} | set(
        _app_ctx.LPPL_FAMILY_HIDDEN_FROM_BUBBLE)

    def _swatch(color, label):
        return html.Span([
            html.Span(" ", style={
                "display": "inline-block", "width": "12px", "height": "12px",
                "borderRadius": "2px", "verticalAlign": "middle",
                "marginRight": "4px", "backgroundColor": color,
            }),
            label,
        ])

    opts = [{"label": _swatch(mc.get("bub", "#000"), "Bubble Model"),
             "value": "bub"}]

    if standardized:
        # Inject master LPPL entry right after Bubble Model.
        opts.append({
            "label": _swatch(mc.get("lppl", "#FF6D00"), "LPPL"),
            "value": "lppl",
        })

    all_models = [mdl for mdl in _app_ctx.PRICE_MODELS.values()
                  if mdl.short_name not in _app_ctx.MODEL_SENTINELS
                  and mdl.short_name != "bub"]
    if standardized:
        all_models = [m for m in all_models
                      if m.short_name not in _LPPL_FAM
                      and m.short_name not in _DEPRIORITIZED]
        ordered = all_models
    else:
        ordered = [m for m in all_models if m.short_name not in _DEPRIORITIZED] + \
                  [m for m in all_models if m.short_name in _DEPRIORITIZED]
    for mdl in ordered:
        opts.append({
            "label": _swatch(mc.get(mdl.short_name, "#888"), mdl.name),
            "value": mdl.short_name,
        })
    return [
        _lbl("Display models"),
        dcc.Checklist(id=f"{prefix}-model-show",
                      options=opts,
                      value=["bub"],
                      inline=True,
                      inputStyle=_CB_MARGIN,
                      labelStyle={"marginRight": "12px", "fontSize": "11px"},
                      style={"marginBottom": "8px"}),
    ]
```

- [ ] **Step 4: Run tests**

Run: `btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestModelShowChecklistStandardized -v`
Expected: 4 pass.

Also run full suite to ensure no regressions:
Run: `btc_venv/bin/python3 -m pytest btc_web/test_web.py -q`
Expected: 830+ pass.

- [ ] **Step 5: Commit**

```bash
git add btc_web/layout/common.py btc_web/test_web.py
git commit -m "feat(common): _model_show_checklist standardized mode

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Add `_lppl_config_panel` and `_global_lppl_modal` helpers

**Files:** `btc_web/layout/common.py`, `btc_web/test_web.py`

- [ ] **Step 1: Write failing tests**

Append to `btc_web/test_web.py`:

```python
class TestLpplConfigPanel:
    """Unit test for _lppl_config_panel compact helper."""

    def test_has_activate_and_summary_and_button(self):
        from layout.common import _lppl_config_panel
        card = _lppl_config_panel("dca")
        rendered = str(card)
        assert "dca-lppl-activate" in rendered
        assert "dca-lppl-summary" in rendered
        assert "dca-lppl-configure-btn" in rendered

    def test_no_inline_config_controls(self):
        """The un-prefixed config IDs live in the global modal, not here."""
        from layout.common import _lppl_config_panel
        card = _lppl_config_panel("ret")
        rendered = str(card)
        # These IDs must NOT appear inside a per-tab panel
        assert '"lppl-n-freqs"' not in rendered.replace("'", '"')
        assert '"lppl-weighted"' not in rendered.replace("'", '"')
        assert '"lppl-no-13"' not in rendered.replace("'", '"')


class TestGlobalLpplModal:
    """Unit test for _global_lppl_modal root-level modal."""

    def test_has_all_config_controls(self):
        from layout.common import _global_lppl_modal
        modal = _global_lppl_modal()
        rendered = str(modal)
        assert "lppl-config-modal" in rendered
        assert "lppl-n-freqs" in rendered
        assert "lppl-weighted" in rendered
        assert "lppl-no-13" in rendered
        assert "lppl-modal-close-btn" in rendered
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestLpplConfigPanel btc_web/test_web.py::TestGlobalLpplModal -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Add both helpers**

Append to `btc_web/layout/common.py` (after `_model_show_checklist`):

```python
def _lppl_config_panel(prefix):
    """Compact LPPL sub-panel: activate + summary + modal launcher.

    The actual n_freqs/weighted/no_13 controls live in the global
    modal (_global_lppl_modal) so they have unique IDs. Each tab's
    version here links to that one modal.
    """
    return _section_card("LPPL Models",
        dcc.Checklist(id=f"{prefix}-lppl-activate",
                      options=[{"label": " Activate LPPL overlay",
                                "value": "yes"}],
                      value=[], inputStyle=_CB_MARGIN),
        html.Div([
            html.Small("Current: ", style={"color": "#888", "fontSize": "11px"}),
            html.Span(id=f"{prefix}-lppl-summary", children="LPPL\u2083",
                      style={"color": "#FF6D00", "fontSize": "11px",
                             "fontWeight": "600"}),
        ], style={"marginTop": "4px", "marginBottom": "4px"}),
        dbc.Button("\u2699\ufe0f Configure LPPL",
                   id=f"{prefix}-lppl-configure-btn",
                   size="sm", color="secondary", outline=True,
                   style={"fontSize": "11px", "padding": "2px 8px"}),
    )


def _global_lppl_modal():
    """Root-level modal holding the n_freqs/weighted/no_13 controls.

    Rendered once in _serve_layout; opened by any tab's
    {prefix}-lppl-configure-btn click.
    """
    return dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("LPPL Model Configuration")),
        dbc.ModalBody([
            _lbl("Oscillation frequencies (N)"),
            dcc.Checklist(id="lppl-n-freqs",
                          options=[
                              {"label": " LPPL\u2081 (1 freq)", "value": 1},
                              {"label": " LPPL\u2082 (2 freqs)", "value": 2},
                              {"label": " LPPL\u2083 (3 freqs) \u2014 recommended",
                               "value": 3},
                              {"label": " LPPL\u2084 (4 freqs) \u2014 \u26A0 likely overfit",
                               "value": 4},
                          ],
                          value=[3],
                          labelStyle={"display": "block"},
                          inputStyle=_CB_MARGIN),
            html.Hr(style={"margin": "6px 0", "borderColor": "#444"}),
            dcc.Checklist(id="lppl-weighted",
                          options=[{"label": " Log-time weighted fits",
                                    "value": "weighted"}],
                          value=[], inputStyle=_CB_MARGIN,
                          className="small"),
            html.Small("Emphasizes early-history structure over recent era",
                       style=_STYLE_HINT),
            dcc.Checklist(id="lppl-no-13",
                          options=[{"label": " Exclude \u03c9\u224813 intermod (disables LPPL\u2083)",
                                    "value": "no13"}],
                          value=[], inputStyle=_CB_MARGIN,
                          className="small"),
            html.Small("LP\u2084's \u03c9\u224813 may be an intermodulation artifact",
                       style=_STYLE_HINT),
        ]),
        dbc.ModalFooter(
            dbc.Button("Close", id="lppl-modal-close-btn",
                       size="sm", color="primary"),
        ),
    ], id="lppl-config-modal", is_open=False, centered=True, size="md")
```

- [ ] **Step 4: Run tests**

Run: `btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestLpplConfigPanel btc_web/test_web.py::TestGlobalLpplModal -v`
Expected: 4 pass.

- [ ] **Step 5: Commit**

```bash
git add btc_web/layout/common.py btc_web/test_web.py
git commit -m "feat(common): _lppl_config_panel + _global_lppl_modal helpers

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Render `_global_lppl_modal` at app root

**Files:** `btc_web/layout/__init__.py`

- [ ] **Step 1: Locate the `dbc.Tabs` container in `_serve_layout`**

Run: `grep -n "dbc.Tabs\|_serve_layout\|^def " btc_web/layout/__init__.py | head -20`

- [ ] **Step 2: Import the modal helper and render it**

In `btc_web/layout/__init__.py`, find the import block for `layout.common` (around top of file) and ensure `_global_lppl_modal` is imported. Add it to the existing import if the list already pulls from `layout.common`, otherwise add a new import line.

Then in `_serve_layout` (or wherever `dbc.Tabs(...)` is wrapped into the root layout), add `_global_lppl_modal()` as a sibling of `dbc.Tabs`:

```python
return html.Div([
    # ... existing root-level components (stores, navbar, etc.) ...
    dbc.Tabs([ ... ]),
    _global_lppl_modal(),   # NEW — single instance, all tabs link here
    # ... any other existing trailing components ...
])
```

(If the layout structure differs, place `_global_lppl_modal()` inside the same html.Div as the Tabs, after them.)

- [ ] **Step 3: Verify app boots**

Restart the dev server and check the app loads without Dash "Duplicate component id" errors:
```bash
lsof -ti :8050 | xargs -r kill -9 2>/dev/null; sleep 1
DEV=1 nohup bash run_web.sh > /tmp/quantoshi_dev.log 2>&1 &
sleep 8
tail -5 /tmp/quantoshi_dev.log
```
Expected: `Dash is running on http://0.0.0.0:8050/` with no errors.

- [ ] **Step 4: Verify modal is present in layout**

```bash
curl -sS http://localhost:8050/_dash-layout | grep -c "lppl-config-modal"
```
Expected: `1` (modal rendered exactly once).

- [ ] **Step 5: Commit**

```bash
git add btc_web/layout/__init__.py
git commit -m "feat(layout): render _global_lppl_modal at app root

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Remove old LPPL panel from Bubble layout; insert compact version

**Files:** `btc_web/layout/bubble.py`

- [ ] **Step 1: Locate the existing LPPL Models section_card**

Run: `grep -n 'LPPL Models\|lppl-n-freqs\|bub-lppl-activate\|bub-lppl-body' btc_web/layout/bubble.py`

- [ ] **Step 2: Delete the old panel**

In `btc_web/layout/bubble.py`, delete the entire `_section_card("LPPL Models", ...)` block that currently contains `bub-lppl-activate` + `bub-lppl-body` + the inline `lppl-n-freqs` / `lppl-weighted` / `lppl-no-13` controls.

- [ ] **Step 3: Insert compact version**

Replace the deleted block with:
```python
        _lppl_config_panel("bub"),
```

Ensure `_lppl_config_panel` is imported at the top of `layout/bubble.py`:
```python
from layout.common import (_tab_hints, _section_card, _row, _lbl,
                            _STYLE_HIDDEN, _STYLE_HINT, _q_panel, _q_panel_with_mode,
                            _q_options, _ctrl_card, _legend_pos_dropdown,
                            _chart_tab_layout, _CB_MARGIN, _palette_selector,
                            _lppl_config_panel)
```
(Add `_lppl_config_panel` to the existing import list.)

- [ ] **Step 4: Verify syntax and restart**

```bash
btc_venv/bin/python3 -m py_compile btc_web/layout/bubble.py && echo OK
lsof -ti :8050 | xargs -r kill -9 2>/dev/null; sleep 1
DEV=1 nohup bash run_web.sh > /tmp/quantoshi_dev.log 2>&1 &
sleep 8
grep -c "lppl-n-freqs" /tmp/quantoshi_dev.log || echo "no duplicate errors"
curl -sS http://localhost:8050/_dash-layout | grep -oc "bub-lppl-activate"
```
Expected: No errors. `bub-lppl-activate` appears once.

- [ ] **Step 5: Commit**

```bash
git add btc_web/layout/bubble.py
git commit -m "refactor(bubble): swap inline LPPL panel for compact + global modal

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Wrap Bubble-tab BM panel in `bub-bm-body` div

**Files:** `btc_web/layout/bubble.py`

- [ ] **Step 1: Locate the BM `_section_card`**

Run: `grep -n 'Bubble Model\|bub-bubble-toggles\|bub-n-future-wrap\|bub-bubble-panel' btc_web/layout/bubble.py`

- [ ] **Step 2: Wrap section content**

Inside the `_section_card("Bubble Model", ...)` call, wrap the body children (the `_lbl("Bubble")`, `dcc.Checklist(id="bub-bubble-toggles", ...)`, and the `html.Div(id="bub-n-future-wrap", ...)` block) in a single `html.Div(id="bub-bm-body", children=[...])`:

```python
_section_card("Bubble Model",
    html.Div(id="bub-bm-body", children=[
        _lbl("Bubble"),
        dcc.Checklist(id="bub-bubble-toggles",
                      options=[{"label":" Composite","value":"show_comp"},
                               {"label":" Support","value":"show_sup"}],
                      value=["show_comp","show_sup"],
                      labelStyle={"display":"block"},
                      inputStyle=_CB_MARGIN),
        html.Div(id="bub-n-future-wrap", children=[
            _lbl("N future bubbles"),
            dcc.Slider(id="bub-n-future", min=0, max=_app_ctx.M.n_future_max,
                       value=BUBBLE["n_future"], step=1, marks=None,
                       tooltip={"always_visible":True}),
        ]),
    ]),
),
```

- [ ] **Step 3: Verify syntax + restart**

```bash
btc_venv/bin/python3 -m py_compile btc_web/layout/bubble.py && echo OK
lsof -ti :8050 | xargs -r kill -9 2>/dev/null; sleep 1
DEV=1 nohup bash run_web.sh > /tmp/quantoshi_dev.log 2>&1 &
sleep 8
curl -sS http://localhost:8050/_dash-layout | grep -oc "bub-bm-body"
```
Expected: `1`.

- [ ] **Step 4: Commit**

```bash
git add btc_web/layout/bubble.py
git commit -m "feat(bubble): wrap BM panel body in bub-bm-body div

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: BM collapse clientside callback

**Files:** `btc_web/callbacks/charts.py` (near other clientside callbacks)

- [ ] **Step 1: Add the callback**

Append to `btc_web/callbacks/charts.py` near the existing `bub-lppl-*` clientside callbacks:

```python
# "bub" in bub-model-show → bub-bm-body collapse
_app_ctx.app.clientside_callback(
    """
    function(models) {
        var has = (models || []).indexOf('bub') !== -1;
        return has ? {} : {display: 'none'};
    }
    """,
    Output("bub-bm-body", "style"),
    Input("bub-model-show", "value"),
)
```

- [ ] **Step 2: Restart and verify**

```bash
lsof -ti :8050 | xargs -r kill -9 2>/dev/null; sleep 1
DEV=1 nohup bash run_web.sh > /tmp/quantoshi_dev.log 2>&1 &
sleep 8
curl -sS http://localhost:8050/_dash-dependencies | grep -oc "bub-bm-body"
```
Expected: `1`.

- [ ] **Step 3: Commit**

```bash
git add btc_web/callbacks/charts.py
git commit -m "feat(bubble): BM panel collapses when Bubble Model unchecked

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: Modal open/close clientside callbacks

**Files:** `btc_web/callbacks/charts.py`

- [ ] **Step 1: Add open and close callbacks**

Append to `btc_web/callbacks/charts.py`:

```python
# Any per-tab Configure-LPPL button click → open modal
_app_ctx.app.clientside_callback(
    """
    function(bub_n, dca_n, ret_n, sc_n, hm_n, close_n, cur_open) {
        var ctx = window.dash_clientside.callback_context;
        if (!ctx.triggered || !ctx.triggered.length) {
            return window.dash_clientside.no_update;
        }
        var src = ctx.triggered[0].prop_id;
        if (src.indexOf('lppl-modal-close-btn') !== -1) {
            return false;
        }
        if (src.indexOf('lppl-configure-btn') !== -1) {
            return true;
        }
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

**Note:** This callback references `hm-lppl-configure-btn` which only exists after Phase 2. For Phase 1, hm-lppl-configure-btn is still emitted (because `_lppl_config_panel("hm")` will be called by Phase 2). For Phase 1, just make sure `hm-lppl-configure-btn` doesn't get wired yet — remove the `Input("hm-lppl-configure-btn", ...)` line and remove `hm_n` from the function args until Phase 2.

Revised Phase 1 version:
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

- [ ] **Step 2: Restart and verify**

```bash
lsof -ti :8050 | xargs -r kill -9 2>/dev/null; sleep 1
DEV=1 nohup bash run_web.sh > /tmp/quantoshi_dev.log 2>&1 &
sleep 8
grep -i "error\|traceback" /tmp/quantoshi_dev.log || echo clean
curl -sS http://localhost:8050/_dash-dependencies | grep -oc "lppl-config-modal.is_open"
```
Expected: clean; `1`.

- [ ] **Step 3: Commit**

```bash
git add btc_web/callbacks/charts.py
git commit -m "feat(modal): open/close LPPL config modal from any tab

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 11: Summary text updater clientside callback

**Files:** `btc_web/callbacks/charts.py`

- [ ] **Step 1: Add one clientside callback per tab**

Append to `btc_web/callbacks/charts.py`:

```python
# LPPL config → compact summary text, per tab
for _sum_prefix in ("bub", "dca", "ret", "sc"):
    _app_ctx.app.clientside_callback(
        """
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
        """,
        Output(f"{_sum_prefix}-lppl-summary", "children"),
        Input("lppl-n-freqs", "value"),
        Input("lppl-weighted", "value"),
        Input("lppl-no-13", "value"),
    )
```

- [ ] **Step 2: Restart and verify in browser**

Manually confirm via the app UI that the summary text on the Bubble tab updates when you open the modal and toggle n_freqs between [3] and [3,4].

- [ ] **Step 3: Commit**

```bash
git add btc_web/callbacks/charts.py
git commit -m "feat(modal): per-tab LPPL summary text follows modal config

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 12: Add `_lppl_config_panel` to DCA layout + wire sync callbacks

**Files:** `btc_web/layout/sim_tabs.py`, `btc_web/callbacks/charts.py`

- [ ] **Step 1: Add panel to DCA/Retire layout builder**

In `btc_web/layout/sim_tabs.py`, modify the function that builds DCA/Retire layouts (around line 36-46 where `_model_show_checklist(prefix)` is called). Add `_lppl_config_panel` import and insert its card after Chart Settings:

At top of file, update the import:
```python
from layout.common import (_tab_hints, _section_card, _row, _lbl,
                            ... existing imports ...,
                            _model_show_checklist, _lppl_config_panel)
```

In the main layout builder, change:
```python
children.append(
    _section_card("Chart Settings",
        *_model_show_checklist(prefix),
        _lbl("Year range"),
        ...
    ),
)
```
to:
```python
children.append(
    _section_card("Chart Settings",
        *_model_show_checklist(prefix, standardized=True),
        _lbl("Year range"),
        ...
    ),
)
children.append(_lppl_config_panel(prefix))
```

- [ ] **Step 2: Add clientside sync callbacks for dca and ret**

Append to `btc_web/callbacks/charts.py`:

```python
# Activate ↔ "lppl" in {prefix}-model-show for DCA, Retire, SC.
for _lp in ("dca", "ret", "sc"):
    _app_ctx.app.clientside_callback(
        """
        function(act, cur_models) {
            var want = (act && act.length) > 0;
            var models = (cur_models || []).slice();
            var has = models.indexOf('lppl') !== -1;
            if (want && !has) { models.push('lppl'); return models; }
            if (!want && has) {
                return models.filter(function(v) { return v !== 'lppl'; });
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output(f"{_lp}-model-show", "value", allow_duplicate=True),
        Input(f"{_lp}-lppl-activate", "value"),
        State(f"{_lp}-model-show", "value"),
        prevent_initial_call='initial_duplicate',
    )
    _app_ctx.app.clientside_callback(
        """
        function(models) {
            var has = (models || []).indexOf('lppl') !== -1;
            return has ? ['yes'] : [];
        }
        """,
        Output(f"{_lp}-lppl-activate", "value", allow_duplicate=True),
        Input(f"{_lp}-model-show", "value"),
        prevent_initial_call='initial_duplicate',
    )
```

- [ ] **Step 3: Restart + verify**

```bash
lsof -ti :8050 | xargs -r kill -9 2>/dev/null; sleep 1
DEV=1 nohup bash run_web.sh > /tmp/quantoshi_dev.log 2>&1 &
sleep 8
grep -i "traceback\|duplicate" /tmp/quantoshi_dev.log || echo clean
curl -sS http://localhost:8050/_dash-layout > /tmp/layout.json
grep -c "dca-lppl-activate\|ret-lppl-activate" /tmp/layout.json
```
Expected: clean; 2 matches (one for dca, one for ret).

- [ ] **Step 4: Commit**

```bash
git add btc_web/layout/sim_tabs.py btc_web/callbacks/charts.py
git commit -m "feat(dca,ret): add LPPL panel + activate sync callbacks

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 13: Add `_lppl_config_panel` to SuperCharger layout

**Files:** `btc_web/layout/supercharge.py`

- [ ] **Step 1: Update supercharge layout**

In `btc_web/layout/supercharge.py` around line 91, change:
```python
*_model_show_checklist("sc"),
```
to:
```python
*_model_show_checklist("sc", standardized=True),
```

And add `_lppl_config_panel` to imports:
```python
from layout.common import (..., _model_show_checklist, _lppl_config_panel)
```

Then add the panel below the Chart Settings section_card (as a new sibling section):
```python
_section_card("Chart Settings",
    ...existing sc chart settings...
),
_lppl_config_panel("sc"),
```

- [ ] **Step 2: Restart + verify**

```bash
lsof -ti :8050 | xargs -r kill -9 2>/dev/null; sleep 1
DEV=1 nohup bash run_web.sh > /tmp/quantoshi_dev.log 2>&1 &
sleep 8
curl -sS http://localhost:8050/_dash-layout | grep -c "sc-lppl-activate"
```
Expected: 1.

- [ ] **Step 3: Commit**

```bash
git add btc_web/layout/supercharge.py
git commit -m "feat(sc): add LPPL panel to SuperCharger layout

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 14: Wire `_resolve_lppl_master` into update_dca

**Files:** `btc_web/callbacks/charts.py`, `btc_web/test_web.py`

- [ ] **Step 1: Update existing test signatures first (they break otherwise)**

In `btc_web/test_web.py`, find `TestUpdateDcaCallback` (around line 2700). Each `update_dca(...)` call currently passes `sel_qs=[...], adv_qs=[], lots_data=[],`. Add 3 new kwargs between `adv_qs` and `lots_data`:

```python
sel_qs=[0.5], adv_qs=[],
lppl_n_freqs=[3], lppl_weighted=[], lppl_no_13=[],
lots_data=[],
```

Do this for all 3 `update_dca(...)` calls in the file (one per test method in the class).

- [ ] **Step 2: Update `update_dca` callback signature**

In `btc_web/callbacks/charts.py`, find the `@callback` decorator for `update_dca` and the `Input("dca-qs-adv", "value"),` line. Add 3 new Inputs right after it:

```python
Input("dca-qs-adv",   "value"),
Input("lppl-n-freqs", "value"),
Input("lppl-weighted","value"),
Input("lppl-no-13",   "value"),
Input("effective-lots","data"),
```

And update the function signature to accept them after `adv_qs` and before `lots_data`:

```python
def update_dca(_first_render, stack, use_lots, amount, freq, dca_infl, yr_range, disp, toggles, legend_pos, sel_qs, adv_qs,
               lppl_n_freqs, lppl_weighted, lppl_no_13,
               lots_data,
               sc_enable, ...):
```

- [ ] **Step 3: Call translation in callback body**

In the `update_dca` body, BEFORE the fig = `_get_dca_fig(...)` call, add:

```python
    model_show = _resolve_lppl_master(
        model_show, lppl_n_freqs, lppl_weighted, lppl_no_13)
```

(Place this after `model_show = model_show if model_show is not None else []` if present, or after the initial toggles/yr_range setup.)

- [ ] **Step 4: Run tests**

Run: `btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestUpdateDcaCallback -v`
Expected: 3 tests pass.

Run: `btc_venv/bin/python3 -m pytest btc_web/test_web.py -q`
Expected: 833+ pass (3 tests added in Task 2 earlier, some in other tasks).

- [ ] **Step 5: Commit**

```bash
git add btc_web/callbacks/charts.py btc_web/test_web.py
git commit -m "feat(dca): translate 'lppl' master via _resolve_lppl_master

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 15: Wire translation into update_retire

**Files:** `btc_web/callbacks/charts.py`, `btc_web/test_web.py`

- [ ] **Step 1: Update TestUpdateRetireCallback signatures**

Add `lppl_n_freqs=[3], lppl_weighted=[], lppl_no_13=[],` between `adv_qs=[],` and `lots_data=[],` in each `update_retire(...)` call.

- [ ] **Step 2: Update `update_retire` callback signature**

Mirror Task 14's changes for the retire callback: add 3 `Input` lines after `Input("ret-qs-adv", "value"),`, add the 3 function parameters, and call `_resolve_lppl_master` before building the figure.

- [ ] **Step 3: Run tests**

Run: `btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestUpdateRetireCallback -v`
Expected: pass.

- [ ] **Step 4: Commit**

```bash
git add btc_web/callbacks/charts.py btc_web/test_web.py
git commit -m "feat(retire): translate 'lppl' master via _resolve_lppl_master

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 16: Wire translation into update_supercharge

**Files:** `btc_web/callbacks/charts.py`, `btc_web/test_web.py`

- [ ] **Step 1: Update TestUpdateSuperchargeCallback signatures**

Add `lppl_n_freqs=[3], lppl_weighted=[], lppl_no_13=[],` in both `update_supercharge(...)` calls (after `adv_qs=[]`).

- [ ] **Step 2: Update `update_supercharge` callback signature**

Mirror Task 14 + 15 for supercharge: add `Input("lppl-n-freqs", "value"), Input("lppl-weighted", "value"), Input("lppl-no-13", "value"),` after `Input("sc-qs-adv", "value"),`. Update function params + call `_resolve_lppl_master` before figure build.

- [ ] **Step 3: Run tests**

Run: `btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestUpdateSuperchargeCallback -v`
Expected: 2 tests pass.

- [ ] **Step 4: Commit**

```bash
git add btc_web/callbacks/charts.py btc_web/test_web.py
git commit -m "feat(sc): translate 'lppl' master via _resolve_lppl_master

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 17: Update `update_model_swatches` to standardized mode for DCA/Retire/SC

**Files:** `btc_web/callbacks/charts.py`

- [ ] **Step 1: Locate `update_model_swatches`**

Run: `grep -n "def update_model_swatches\|_build_model_opts(mc" btc_web/callbacks/charts.py`

- [ ] **Step 2: Change `bubble_mode` → `standardized` naming for consistency, and apply to all 4 tabs**

Update `_build_model_opts` signature to accept `standardized=True` kwarg (current impl has `bubble_mode=True` — rename for semantic accuracy, or pass both kwargs if you want to preserve backward compat). Then in `update_model_swatches`, return standardized-mode options for all four tabs:

```python
def update_model_swatches(palette_key):
    pal = _app_ctx.PALETTES.get(palette_key or "default", _app_ctx.PALETTES["default"])
    mc = pal.get("model_colors", _app_ctx.MODEL_TRACE_COLORS)
    bub_opts   = _build_model_opts(mc, include_u1=True,  bubble_mode=True)
    other_opts = _build_model_opts(mc, include_u1=False, bubble_mode=True)
    return bub_opts, other_opts, other_opts, other_opts
```

(Rename `bubble_mode` → `standardized` throughout if you prefer. The existing Bubble tab uses `bubble_mode=True` already from commit 80ac01d, so leave the kwarg name alone for this task.)

- [ ] **Step 3: Restart + verify palette change doesn't reintroduce LPPL variants**

```bash
lsof -ti :8050 | xargs -r kill -9 2>/dev/null; sleep 1
DEV=1 nohup bash run_web.sh > /tmp/quantoshi_dev.log 2>&1 &
sleep 8
```

Open http://localhost:8050/3 (DCA tab) in browser, change palette via navbar dropdown. Verify Display Models still has single "LPPL" master (not individual variants).

- [ ] **Step 4: Commit**

```bash
git add btc_web/callbacks/charts.py
git commit -m "fix(palette): emit standardized model options for DCA/Retire/SC

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 18: Delete S2F from Citadel dropdown

**Files:** `btc_web/layout/citadel.py`

- [ ] **Step 1: Locate the dropdown options**

Run: `grep -n "S2F\|cp-model-src" btc_web/layout/citadel.py`
Expected: one match around line 245.

- [ ] **Step 2: Delete the S2F option**

In `btc_web/layout/citadel.py`, remove the line:
```python
                         {"label": "S2F", "value": "s2f"}],
```
and make the preceding line end with `]` instead of `,`:

Before:
```python
                options=[{"label": "Bubble Model", "value": "bub"},
                         {"label": "Power Law", "value": "pl"},
                         {"label": "S2F", "value": "s2f"}],
```
After:
```python
                options=[{"label": "Bubble Model", "value": "bub"},
                         {"label": "Power Law", "value": "pl"}],
```

- [ ] **Step 3: Restart + verify**

```bash
lsof -ti :8050 | xargs -r kill -9 2>/dev/null; sleep 1
DEV=1 nohup bash run_web.sh > /tmp/quantoshi_dev.log 2>&1 &
sleep 8
curl -sS http://localhost:8050/_dash-layout | grep -o '"S2F"' | wc -l
```
Expected: `0` (no S2F option in Citadel dropdown).

- [ ] **Step 4: Commit**

```bash
git add btc_web/layout/citadel.py
git commit -m "chore(citadel): remove S2F from cp-model-src dropdown

S2F is a demonstration model (meant for Bubble tab only). Citadel's
MC engine doesn't support it meaningfully.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 19: Rename Bubble tab label

**Files:** `btc_web/layout/__init__.py`

- [ ] **Step 1: Change the Tab label**

In `btc_web/layout/__init__.py`, locate the Bubble `dbc.Tab`:
```bash
grep -n "Bubble + QR Overlay" btc_web/layout/__init__.py
```

Replace `"📈 Bubble + QR Overlay"` with `"📈 Price & Model Overlays"`.

- [ ] **Step 2: Restart + verify**

```bash
lsof -ti :8050 | xargs -r kill -9 2>/dev/null; sleep 1
DEV=1 nohup bash run_web.sh > /tmp/quantoshi_dev.log 2>&1 &
sleep 8
curl -sS http://localhost:8050/_dash-layout | grep -c "Price & Model Overlays"
```
Expected: 1.

- [ ] **Step 3: Commit**

```bash
git add btc_web/layout/__init__.py
git commit -m "chore(bubble): rename tab 1 'Bubble + QR Overlay' -> 'Price & Model Overlays'

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 20: Snapshot additions

**Files:** `btc_web/snapshot.py`, `btc_web/callbacks/routing.py`

- [ ] **Step 1: Add new activate IDs to `_SNAPSHOT_CONTROLS`**

In `btc_web/snapshot.py`, find the existing `("bub-lppl-activate", "value"),` entry. Append the new ones right after:

```python
    ("bub-lppl-activate", "value"),
    ("dca-lppl-activate", "value"),
    ("ret-lppl-activate", "value"),
    ("sc-lppl-activate",  "value"),
    ("hm-lppl-activate",  "value"),  # reserved for Phase 2
```

- [ ] **Step 2: Add to `_CHECKLIST_OPTIONS`**

Append to the `_CHECKLIST_OPTIONS` dict:
```python
    "dca-lppl-activate": ["yes"],
    "ret-lppl-activate": ["yes"],
    "sc-lppl-activate":  ["yes"],
    "hm-lppl-activate":  ["yes"],
```

- [ ] **Step 3: Extend `_TAB_CONTROLS` in `btc_web/callbacks/routing.py`**

Add `"dca-lppl-activate"` to the `"dca"` set; `"ret-lppl-activate"` to `"retire"`; `"sc-lppl-activate"` to `"supercharge"`; `"hm-lppl-activate"` to `"heatmap"`. Also add `"lppl-n-freqs"`, `"lppl-weighted"`, `"lppl-no-13"` to each of dca/retire/supercharge/heatmap sets (they're already in bubble from commit 80ac01d).

- [ ] **Step 4: Restart + verify snapshot validates**

```bash
lsof -ti :8050 | xargs -r kill -9 2>/dev/null; sleep 1
DEV=1 nohup bash run_web.sh > /tmp/quantoshi_dev.log 2>&1 &
sleep 8
grep -i "traceback\|validation" /tmp/quantoshi_dev.log || echo clean
```

Also run snapshot-related tests:
```bash
btc_venv/bin/python3 -m pytest btc_web/test_web.py -q -k snapshot
```
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add btc_web/snapshot.py btc_web/callbacks/routing.py
git commit -m "feat(snapshot): register new per-tab LPPL activate IDs

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 21: End-to-end verification via Playwright

**Files:** manual Playwright testing (no code changes)

- [ ] **Step 1: Navigate to each standardized tab and verify layout**

Visit http://localhost:8050/1 (Bubble), /3 (DCA), /4 (Retire), /5 (SC). Confirm each tab shows:
- Display Models checklist with single "LPPL" master entry
- LPPL Models compact sub-panel (activate checkbox + summary + Configure button)

- [ ] **Step 2: Test modal flow**

Click "⚙️ Configure LPPL" on DCA tab. Modal opens with n_freqs=[3] defaults. Click "Close". Modal closes. Reopen from Retire tab → same modal, same state.

- [ ] **Step 3: Test activate ↔ model-show sync**

On DCA tab: click "Activate LPPL overlay" → verify "LPPL" in Display Models checks on. Click LPPL in Display Models → verify activate checkbox flips. Repeat on Retire, SC.

- [ ] **Step 4: Test BM collapse on Bubble tab**

On Bubble tab: uncheck "Bubble Model" in Display Models → verify BM panel body collapses. Check it again → body expands.

- [ ] **Step 5: Test snapshot backward-compat**

Generate a share link from Bubble tab with LPPL active and a specific flavor. Copy URL, open in fresh browser window, verify chart renders identically.

- [ ] **Step 6: Run full test suite**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_web.py -q
```
Expected: 845+ pass (adding ~15 new tests across tasks).

- [ ] **Step 7: Kill dev server**

```bash
lsof -ti :8050 | xargs -r kill -9 2>/dev/null
```

- [ ] **Step 8: No commit needed (verification only)**

---

## Post-implementation checks

- [ ] All tests passing: `btc_venv/bin/python3 -m pytest btc_web/test_web.py -q`
- [ ] Syntax check: `btc_venv/bin/python3 -m py_compile btc_web/callbacks/charts.py btc_web/layout/common.py btc_web/layout/bubble.py btc_web/layout/__init__.py btc_web/layout/sim_tabs.py btc_web/layout/supercharge.py btc_web/layout/citadel.py btc_web/snapshot.py btc_web/callbacks/routing.py btc_web/figures/common.py`
- [ ] App boots in DEV mode
- [ ] Old snapshot link decodes correctly (Bubble tab LPPL active state preserved)
- [ ] Citadel dropdown no longer shows S2F
- [ ] Tab 1 label reads "Price & Model Overlays"
- [ ] Global LPPL modal opens from each of 4 tabs (bub/dca/ret/sc)
