# Simplified Quantile Panel — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a default/advanced toggle to the Projection Quantiles panel on all 4 chart tabs (Bubble, DCA, Retire, Supercharger). Default mode shows 5 simplified quantile options (Q1/15/50/85/99%), limits selection to 3 bands, and shows Q50% at 50% opacity when nothing is selected. Advanced mode retains current behavior. Each tab has its own independent instance.

**Architecture:** 3 tasks: (1) Layout — shared `_q_panel_with_mode()` builder for all 4 tabs, (2) Callbacks — per-tab mode switch + 3-band limit via factory, (3) Figure builders — interpolated Q3%/Q97% support + 50% opacity Q50% fallback in default mode.

**Tabs affected:**
- Tab 1 (Bubble): `bub-qs` — uses `_q_panel()` in `layout/bubble.py`
- Tab 3 (DCA): `dca-qs` — uses `sim_tabs.py`
- Tab 4 (Retire): `ret-qs` — uses `sim_tabs.py`
- Tab 5 (Supercharger): `sc-qs` — uses `layout/supercharge.py`
- Tab 2 (Heatmap): `hm-exit-qs` — NOT affected (exit quantiles, different purpose)

**Q1%/Q99%:** These quantiles exist in the fitted model — no interpolation needed.

**Tech Stack:** Python 3.14, Dash 4.0.0, DBC 2.0.4

**Test command:** `PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py -x -q --tb=short`

**Full suite:** `PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py --tb=short 2>&1 | tail -5`

---

## File Structure

### Modified Files
| File | Change |
|------|--------|
| `btc_web/layout/common.py` | Add `_DEFAULT_QS`, `_q_options_default()`, `_q_panel_with_mode()` |
| `btc_web/layout/bubble.py` | Use `_q_panel_with_mode("bub-qs", ...)` |
| `btc_web/layout/sim_tabs.py` | Use `_q_panel_with_mode(f"{prefix}-qs", ...)` for DCA + Retire |
| `btc_web/layout/supercharge.py` | Use `_q_panel_with_mode("sc-qs", ...)` |
| `btc_web/callbacks/charts.py` | Factory callbacks for mode switch + 3-band limit; pass `qs_mode` to params |
| `btc_web/figures/bubble.py` | Use `interp_price()` for non-fitted quantiles; 50% opacity Q50% fallback |
| `btc_web/figures/common.py` | Shared helper `build_overlay_traces()` — add interpolation support |
| `btc_web/tab_defaults.py` | Add `"qs_mode": []` to BUBBLE, DCA, RETIRE, SUPERCHARGE defaults |
| `btc_web/snapshot.py` | Register `{prefix}-qs-mode` and `{prefix}-qs-adv` for all 4 tabs |
| `btc_web/callbacks/routing.py` | Register new IDs in `_TAB_CONTROLS` |
| `btc_web/test_web.py` | All new tests |

---

### Task 1: Layout — shared `_q_panel_with_mode()` for all 4 tabs

**Files:**
- Modify: `btc_web/layout/common.py`
- Modify: `btc_web/layout/bubble.py`
- Modify: `btc_web/layout/sim_tabs.py`
- Modify: `btc_web/layout/supercharge.py`
- Test: `btc_web/test_web.py`

Creates `_q_panel_with_mode()` with default/advanced toggle, replaces quantile panel code in all 4 tabs.

- [ ] **Step 1: Write failing tests**

```python
class TestSimplifiedQuantilePanel:
    def test_default_quantile_options(self):
        from layout.common import _q_options_default, _DEFAULT_QS
        opts = _q_options_default()
        values = [o["value"] for o in opts]
        assert values == _DEFAULT_QS
        assert 0.03 in values
        assert 0.50 in values
        assert 0.97 in values

    def test_default_qs_values(self):
        from layout.common import _DEFAULT_QS
        assert _DEFAULT_QS == [0.01, 0.15, 0.50, 0.85, 0.99]

    def test_quantile_mode_toggle_in_bubble(self):
        from layout.bubble import _bubble_controls
        import json
        layout_str = json.dumps(_bubble_controls().to_plotly_json())
        assert "bub-qs-mode" in layout_str
        assert "bub-qs-default-wrap" in layout_str
        assert "bub-qs-advanced-wrap" in layout_str
        assert "bub-qs-adv" in layout_str
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestSimplifiedQuantilePanel -x -q --tb=short`

- [ ] **Step 3: Add `_DEFAULT_QS`, `_q_options_default()`, and `_q_panel_with_mode()` to `btc_web/layout/common.py`**

Add after `_q_options()`:

```python
_DEFAULT_QS = [0.01, 0.15, 0.50, 0.85, 0.99]

def _q_options_default() -> list[dict]:
    """Simplified quantile options for default mode (5 options).

    All values exist in the fitted model — no interpolation needed.
    """
    opts = []
    for q in _DEFAULT_QS:
        pct = q * 100
        lbl_text = f"Q{pct:.4g}%" if pct >= 1 else f"Q{pct:.3g}%"
        col = _app_ctx.M.qr_colors.get(q, "#888888")
        lbl = html.Span([
            html.Span("\u25CF ", style={"color": col, "fontSize": "10px"}),
            lbl_text,
        ])
        opts.append({"label": lbl, "value": q})
    return opts


def _q_panel_with_mode(checklist_id: str, default_value: list,
                       hint: str | None = None):
    """Quantile panel with default/advanced toggle.

    Default mode: 5 options (Q1/15/50/85/99%), max 3 bands.
    Advanced mode: all 27 quantiles, no limit.
    Each tab gets its own independent instance via unique checklist_id prefix.
    """
    children = []
    if hint:
        children.append(html.Small(hint, style=_STYLE_HINT))

    mode_id = f"{checklist_id}-mode"
    children.append(
        dcc.Checklist(id=mode_id,
                      options=[{"label": " Advanced", "value": "advanced"}],
                      value=[], inputStyle=_CB_MARGIN,
                      className="small mb-1"),
    )
    # Filter default_value to only include values in _DEFAULT_QS
    default_filtered = [v for v in default_value if v in _DEFAULT_QS] or [0.5]
    # Default mode checklist (visible initially)
    children.append(
        html.Div(id=f"{checklist_id}-default-wrap", children=[
            dcc.Checklist(id=checklist_id, options=_q_options_default(),
                          value=default_filtered,
                          className="q-panel-grid",
                          inputStyle=_CB_MARGIN),
            html.Small("Up to 3 bands", className="text-muted",
                       style={"fontSize": "10px"}),
        ]),
    )
    # Advanced mode checklist (hidden initially)
    children.append(
        html.Div(id=f"{checklist_id}-advanced-wrap",
                 style=_STYLE_HIDDEN, children=[
            dcc.Checklist(id=f"{checklist_id}-adv", options=_q_options(),
                          value=default_value, className="q-panel-grid",
                          inputStyle=_CB_MARGIN),
        ]),
    )

    return _section_card("Projection Quantiles", *children)
```

- [ ] **Step 4: Update `btc_web/layout/bubble.py`**

Replace:
```python
        _q_panel("bub-qs", [0.5],
                 hint="If none selected, Q50% is shown for active models."),
```
with:
```python
        _q_panel_with_mode("bub-qs", [0.5],
                           hint="If none selected, Q50% is shown at 50% opacity."),
```

Add `_q_panel_with_mode` to the import from `layout.common`.

- [ ] **Step 5: Update `btc_web/layout/sim_tabs.py`**

Replace the inline quantile panel (around line 29-34):
```python
        _section_card("Projection Quantiles",
            html.Small("Select quantiles to follow.", style=_STYLE_HINT),
            html.Small(q_hint, style=_STYLE_HINT),
            dcc.Checklist(id=f"{prefix}-qs", options=_q_options(),
                          value=q_defaults, className="q-panel-grid",
                          inputStyle=_CB_MARGIN),
        ),
```
with:
```python
        _q_panel_with_mode(f"{prefix}-qs", q_defaults, hint=q_hint),
```

Add `_q_panel_with_mode` to the import from `layout.common`.

- [ ] **Step 6: Update `btc_web/layout/supercharge.py`**

Replace the inline quantile panel (around line 27-36):
```python
        _section_card("Projection Quantiles",
            html.Small(_Q_HINT_BASE, style=_STYLE_HINT),
            html.Small("Lower prices mean earlier depletion.", style=_STYLE_HINT),
            dcc.Checklist(id="sc-qs",
                          options=_q_options(),
                          value=[q for q in [0.001, 0.10] if q in (_app_ctx.DEFAULT_MODEL.fits or {})],
                          className="q-panel-grid",
                          inputStyle=_CB_MARGIN),
        ),
```
with:
```python
        _q_panel_with_mode("sc-qs",
                           [q for q in [0.001, 0.10] if q in (_app_ctx.DEFAULT_MODEL.fits or {})],
                           hint="Lower prices mean earlier depletion."),
```

Add `_q_panel_with_mode` to the import from `layout.common`.

- [ ] **Step 7: Run test to verify it passes**

- [ ] **Step 8: Run full test suite**

- [ ] **Step 9: Commit**

```bash
git add btc_web/layout/common.py btc_web/layout/bubble.py btc_web/layout/sim_tabs.py btc_web/layout/supercharge.py btc_web/test_web.py
git commit -m "feat: add default/advanced quantile toggle to all 4 chart tabs"
```

---

### Task 2: Callbacks — per-tab mode switch + 3-band limit + snapshot

**Files:**
- Modify: `btc_web/callbacks/charts.py`
- Modify: `btc_web/snapshot.py`
- Modify: `btc_web/callbacks/routing.py`
- Modify: `btc_web/tab_defaults.py`
- Test: `btc_web/test_web.py`

Factory pattern registers callbacks for all 4 prefixes. Also adds `qs_mode` to tab defaults for cache key alignment.

- [ ] **Step 1: Write failing tests**

```python
class TestQuantileModeSwitch:
    def test_mode_controls_in_snapshot(self):
        from snapshot import _SNAPSHOT_CONTROLS
        ids = {c[0] for c in _SNAPSHOT_CONTROLS}
        for prefix in ["bub", "dca", "ret", "sc"]:
            assert f"{prefix}-qs-mode" in ids, f"{prefix}-qs-mode missing"
            assert f"{prefix}-qs-adv" in ids, f"{prefix}-qs-adv missing"

    def test_mode_controls_in_tab_controls(self):
        from callbacks.routing import _TAB_CONTROLS
        assert "bub-qs-mode" in _TAB_CONTROLS["bubble"]
        assert "dca-qs-mode" in _TAB_CONTROLS["dca"]
        assert "ret-qs-mode" in _TAB_CONTROLS["retire"]
        assert "sc-qs-mode" in _TAB_CONTROLS["supercharge"]

    def test_qs_mode_in_tab_defaults(self):
        from tab_defaults import BUBBLE, DCA, RETIRE, SUPERCHARGE
        assert "qs_mode" in BUBBLE
        assert "qs_mode" in DCA
        assert "qs_mode" in RETIRE
        assert "qs_mode" in SUPERCHARGE
```

- [ ] **Step 2: Run test to verify it fails**

- [ ] **Step 3: Add mode switch + band limit callbacks to `btc_web/callbacks/charts.py`**

Add at the bottom (after existing callbacks):

```python
# ── Quantile mode toggle (default ↔ advanced) — all tabs ─────────────────────

def _register_qs_mode_callbacks(prefix):
    """Register mode toggle + band limit callbacks for one tab's quantile panel."""

    @callback(
        Output(f"{prefix}-qs-default-wrap", "style"),
        Output(f"{prefix}-qs-advanced-wrap", "style"),
        Output(f"{prefix}-qs", "value", allow_duplicate=True),
        Output(f"{prefix}-qs-adv", "value", allow_duplicate=True),
        Input(f"{prefix}-qs-mode", "value"),
        State(f"{prefix}-qs", "value"),
        State(f"{prefix}-qs-adv", "value"),
        prevent_initial_call=True,
    )
    def toggle_mode(mode, default_vals, adv_vals):
        is_advanced = "advanced" in (mode or [])
        if is_advanced:
            return ({"display": "none"}, {}, no_update, default_vals or [])
        else:
            from layout.common import _DEFAULT_QS
            filtered = [q for q in (adv_vals or []) if q in _DEFAULT_QS]
            return ({}, {"display": "none"}, filtered[:3], no_update)

    @callback(
        Output(f"{prefix}-qs", "value", allow_duplicate=True),
        Input(f"{prefix}-qs", "value"),
        State(f"{prefix}-qs-mode", "value"),
        prevent_initial_call=True,
    )
    def enforce_limit(selected, mode):
        if "advanced" in (mode or []):
            return no_update
        if selected and len(selected) > 3:
            return selected[-3:]
        return no_update


for _prefix in ("bub", "dca", "ret", "sc"):
    _register_qs_mode_callbacks(_prefix)
```

- [ ] **Step 4: Register in `btc_web/snapshot.py`**

Add to the appropriate tab sections:
```python
    # Bubble section
    ("bub-qs-mode",           "value"),
    ("bub-qs-adv",            "value"),
    # DCA section
    ("dca-qs-mode",           "value"),
    ("dca-qs-adv",            "value"),
    # Retire section
    ("ret-qs-mode",           "value"),
    ("ret-qs-adv",            "value"),
    # Supercharge section
    ("sc-qs-mode",            "value"),
    ("sc-qs-adv",             "value"),
```

- [ ] **Step 5: Register in `_CHECKLIST_OPTIONS` in `btc_web/snapshot.py`**

Add bitmask encoding entries for the new checklists:
```python
    "bub-qs-mode": ["advanced"],
    "dca-qs-mode": ["advanced"],
    "ret-qs-mode": ["advanced"],
    "sc-qs-mode":  ["advanced"],
    "bub-qs-adv":  _QS_LIST,
    "dca-qs-adv":  _QS_LIST,
    "ret-qs-adv":  _QS_LIST,
    "sc-qs-adv":   _QS_LIST,
```

Where `_QS_LIST` is the existing quantile value list used for `bub-qs` etc.

- [ ] **Step 6: Register in `btc_web/callbacks/routing.py`**

Add to each tab's `_TAB_CONTROLS` set:
```python
    "bub-qs-mode", "bub-qs-adv"        → bubble
    "dca-qs-mode", "dca-qs-adv"        → dca
    "ret-qs-mode", "ret-qs-adv"        → retire
    "sc-qs-mode",  "sc-qs-adv"         → supercharge
```

- [ ] **Step 6: Add `qs_mode` to `btc_web/tab_defaults.py`**

Add `"qs_mode": []` to the frozen dicts for BUBBLE, DCA, RETIRE, and SUPERCHARGE.

- [ ] **Step 7: Pass `qs_mode` through chart callbacks**

In the bubble callback in `btc_web/callbacks/charts.py`:
1. Add `State("bub-qs-mode", "value")` and `State("bub-qs-adv", "value")`
2. Add `qs_mode` and `adv_qs` parameters
3. In the params dict: `qs_mode=qs_mode or []`
4. For selected_qs: `selected_qs = list(adv_qs) if "advanced" in (qs_mode or []) else list(sel_qs)` where `sel_qs` is the existing `bub-qs` value

Repeat for DCA, Retire, and Supercharger callbacks (or note the exact file/line for each).

- [ ] **Step 8: Run tests — verify PASS**

- [ ] **Step 9: Run full test suite**

- [ ] **Step 10: Commit**

```bash
git add btc_web/callbacks/charts.py btc_web/snapshot.py btc_web/callbacks/routing.py btc_web/tab_defaults.py btc_web/test_web.py
git commit -m "feat: mode switch + 3-band limit callbacks for all 4 quantile tabs"
```

---

### Task 3: Figure builders — 50% opacity Q50% fallback in default mode

**Files:**
- Modify: `btc_web/figures/bubble.py`
- Test: `btc_web/test_web.py`

Show Q50% at 50% opacity in default mode when no quantiles selected.

- [ ] **Step 1: Write failing tests**

```python
class TestDefaultModeOpacity:
    def test_fallback_q50_has_opacity_in_default_mode(self):
        """Q50% fallback in default mode should have 50% opacity."""
        from figures.bubble import build_bubble_figure
        import _app_ctx
        M = _app_ctx.M
        p = dict(selected_qs=[], shade=False, xscale="log", yscale="log",
                 xmin=2012, xmax=2030, ymin=0, ymax=7, n_future=3,
                 show_comp=False, show_ols=False, show_data=False,
                 show_today=False, pt_size=2, pt_alpha=0.3,
                 stack=0, show_stack=False, lots=[], use_lots=False,
                 show_legend=False, active_models=["bub"],
                 qs_mode=[])
        fig = build_bubble_figure(M, p)
        q50_traces = [t for t in fig.data if hasattr(t, 'name') and t.name and 'Q50%' in str(t.name)]
        assert len(q50_traces) > 0
        assert q50_traces[0].opacity == 0.5

    def test_fallback_q50_full_opacity_in_advanced_mode(self):
        """Q50% fallback in advanced mode should have full opacity."""
        from figures.bubble import build_bubble_figure
        import _app_ctx
        M = _app_ctx.M
        p = dict(selected_qs=[], shade=False, xscale="log", yscale="log",
                 xmin=2012, xmax=2030, ymin=0, ymax=7, n_future=3,
                 show_comp=False, show_ols=False, show_data=False,
                 show_today=False, pt_size=2, pt_alpha=0.3,
                 stack=0, show_stack=False, lots=[], use_lots=False,
                 show_legend=False, active_models=["bub"],
                 qs_mode=["advanced"])
        fig = build_bubble_figure(M, p)
        q50_traces = [t for t in fig.data if hasattr(t, 'name') and t.name and 'Q50%' in str(t.name)]
        assert len(q50_traces) > 0
        assert q50_traces[0].opacity is None
```

- [ ] **Step 2: Run test to verify it fails**

- [ ] **Step 3: Modify `btc_web/figures/bubble.py`**

Change the Q50% fallback to include opacity:
```python
    _fallback_q50 = not sel_qs and p.get("active_models")
    _default_mode = "advanced" not in (p.get("qs_mode") or [])
    if _fallback_q50:
        sel_qs = [0.5]
        _thermal = _build_thermal_colors(sel_qs, palette)
```

In the trace creation (around line 135):
```python
            traces.append(go.Scatter(
                x=list(t_arr), y=list(prices),
                mode="lines", name=lbl,
                line=dict(color=col, width=_QR_LINE_WIDTH),
                opacity=0.5 if (_fallback_q50 and _default_mode) else None,
            ))
```

- [ ] **Step 4: Run test to verify it passes**

- [ ] **Step 5: Run full test suite**

- [ ] **Step 6: Commit**

```bash
git add btc_web/figures/bubble.py btc_web/test_web.py
git commit -m "feat(bubble): interpolated Q3%/Q97% support + 50% opacity Q50% fallback"
```

---

## Verification Checklist

After all 3 tasks:

```bash
# Full test suite
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/ --tb=short 2>&1 | tail -10

# Syntax check
cd btc_web && PYTHONPATH=".:../archive/btc_app" ../btc_venv/bin/python3 -c \
  "import layout, figures, callbacks; print('OK')"
```

## Notes

- All 4 chart tabs (Bubble, DCA, Retire, Supercharger) get the toggle. Heatmap is excluded.
- Each tab has independent state: `{prefix}-qs`, `{prefix}-qs-adv`, `{prefix}-qs-mode`.
- `{prefix}-qs` remains the primary ID read by each tab's chart callback — in advanced mode, the mode switch callback copies `{prefix}-qs-adv` values into `{prefix}-qs`.
- The 3-band limit uses `selected[-3:]` to keep the most recently checked items.
- Q1% and Q99% exist in the fitted model — no interpolation needed.
- `qs_mode` added to all 4 tab defaults in `tab_defaults.py` for cache key alignment.
- Both `{prefix}-qs-adv` and `{prefix}-qs-mode` registered in `_SNAPSHOT_CONTROLS` for share link fidelity.
- The 50% opacity Q50% fallback applies to all tabs in default mode (consistent behavior). In advanced mode, the existing full-opacity fallback is preserved.
