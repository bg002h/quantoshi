# Tab 5 Annotation Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make depletion/terminal annotations toggle with legend, extend to overlay models, stagger readably, optimize for mobile.

**Architecture:** Keep depletion arrows as `layout.annotations` (real arrow styling). Tag each with `name` attribute for legendgroup. New JS handler on `plotly_legendclick` toggles annotations matching the clicked trace's legendgroup. Overlay models get depletion detection + arrows + endpoint annotations. Mobile detection via `dcc.Store("viewport-width")` skips on-chart endpoint annotations.

**Tech Stack:** Python (Plotly/Dash), JavaScript (Plotly events), CSS

**Spec:** `docs/superpowers/specs/2026-03-24-sc-annotations-design.md`

---

### Task 1: Expand stagger heights

**Files:**
- Modify: `btc_web/_app_ctx.py:13`
- Modify: `btc_web/figures/common.py:203`

- [ ] **Step 1: Update `ANNOT_STAGGER_Y` from 3 to 5 heights**

In `btc_web/_app_ctx.py` line 13, change:
```python
ANNOT_STAGGER_Y = [-20, -33, -46, -59, -72]
```

- [ ] **Step 2: Update modulo in `_stagger_depletion_annots`**

In `btc_web/figures/common.py` line 203, the `i % 3` will automatically work with 5 elements since `len(_ANNOT_STAGGER_Y)` determines the cycle. But change the hardcoded `3` to use the list length:
```python
a["ay"] = _ANNOT_STAGGER_Y[i % len(_ANNOT_STAGGER_Y)]
```

- [ ] **Step 3: Syntax check and commit**

Run: `btc_venv/bin/python3 -m py_compile btc_web/_app_ctx.py && btc_venv/bin/python3 -m py_compile btc_web/figures/common.py && echo "OK"`

```bash
git add btc_web/_app_ctx.py btc_web/figures/common.py
git commit -m "feat: expand depletion stagger from 3 to 5 heights"
```

---

### Task 2: Add viewport-width Store for mobile detection

**Files:**
- Modify: `btc_web/layout/common.py`
- Modify: `btc_web/callbacks/charts.py`

- [ ] **Step 1: Add `dcc.Store("viewport-width")` to the app layout**

In `btc_web/layout/common.py`, find the main layout assembly function and add:
```python
dcc.Store(id="viewport-width", data=1200, storage_type="memory"),
```

Add a clientside callback (in `btc_web/callbacks/nav.py` or `charts.py`) that updates it:
```python
_app_ctx.app.clientside_callback(
    """
    function() {
        var w = window.innerWidth;
        window.addEventListener('resize', function() {
            // Dash clientside callbacks can't re-fire on resize,
            // but initial width is sufficient for server-side decisions
        });
        return w;
    }
    """,
    Output("viewport-width", "data"),
    Input("url", "pathname"),  # fires on page load
)
```

- [ ] **Step 2: Add `State("viewport-width", "data")` to `update_supercharge` callback**

In `btc_web/callbacks/charts.py`, add after line 596 (the last State):
```python
State("viewport-width", "data"),
```

Add `viewport_width` parameter to the function signature (line 607, after `palette_key`).

Pass `is_mobile = (viewport_width or 1200) < 768` into the figure params dict:
```python
is_mobile = (viewport_width or 1200) < 768,
```

- [ ] **Step 3: Syntax check and commit**

```bash
git add btc_web/layout/common.py btc_web/callbacks/charts.py
git commit -m "feat: add viewport-width Store for mobile detection"
```

---

### Task 3: Tag depletion annotations with `name` for legend toggle

**Files:**
- Modify: `btc_web/figures/supercharge.py:121-131`

- [ ] **Step 1: Update `_depl_annot` to accept and set `name` parameter**

Change `_depl_annot` (line 121) to accept a `legendgroup` parameter and set it as the annotation's `name`:

```python
def _depl_annot(depl_t, t_start_d, d, arrow_col, text_col, legendgroup, model_prefix="", stagger=0):
    depl_yr = int((syr + d) + (depl_t - t_start_d) *
                  (eyr - (syr + d)) / max(t_end - t_start_d, 1e-6))
    prefix = f"{model_prefix} " if model_prefix else ""
    return dict(
        x=depl_t - dt, xref="x",
        y=0, yref="paper",
        ax=28, ay=_AY_LEVELS[stagger % len(_AY_LEVELS)],  # also fix line 127 of current code: % 3 -> % len()
        text=f"{prefix}\u2248{depl_yr}",
        showarrow=True, arrowhead=2, arrowsize=1,
        arrowcolor=arrow_col,
        font=dict(size=_FONT_ANNOT, color=text_col),
        name=legendgroup,
    )
```

- [ ] **Step 2: Update all `_depl_annot` call sites to pass new params**

Compute `_tcol_annot` once before the layout branches (around line 133):
```python
_tcol_annot = _app_ctx.MODEL_TRACE_COLORS.get(model.short_name, "#000000")
```

**Line 170-172 (layout 0):** Arrow uses `annot_colors[di]`:
```python
deplete_annots.append(_depl_annot(depl_t, t_start_d, d,
    arrow_col=annot_colors[di % len(annot_colors)],
    text_col=_tcol_annot, legendgroup=grp_model,
    model_prefix=model.legend_name, stagger=len(deplete_annots)))
```

**Line 193-194 (layout 1):** Arrow uses thermal `col` (quantile color):
```python
deplete_annots.append(_depl_annot(depl_t, t_start_d, d,
    arrow_col=col,
    text_col=_tcol_annot, legendgroup=grp_model,
    model_prefix=model.legend_name, stagger=len(deplete_annots)))
```

**Line 239-241 (layout 2):** Arrow uses `annot_colors[di]`:
```python
deplete_annots.append(_depl_annot(depl_t, t_start_d, d,
    arrow_col=annot_colors[di % len(annot_colors)],
    text_col=_tcol_annot, legendgroup=grp_model,
    model_prefix=model.legend_name, stagger=len(deplete_annots)))
```

- [ ] **Step 3: Syntax check and commit**

```bash
git add btc_web/figures/supercharge.py
git commit -m "feat: tag depletion annotations with name for legend toggle

Arrow color = delay color, text color = model trace color,
name = legendgroup for JS toggle."
```

---

### Task 4: Expand overlay model results and add depletion arrows

**Files:**
- Modify: `btc_web/figures/supercharge.py:257-320`

- [ ] **Step 1: Expand `ov_results` tuple to include depletion data**

At line 271, change from:
```python
ov_results[(d, q)] = (ts_d, y_vals)
```
To:
```python
depl_mask = vals == 0.0
depl_t_ov = float(ts_d[np.argmax(depl_mask)]) if depl_mask.any() else None
ov_results[(d, q)] = (ts_d, y_vals, depl_t_ov, t_start_d, vals, prices)
```

- [ ] **Step 2: Update all `ov_results` unpacking sites**

Line 277: `[ov_results[(d, q)][1] ...]` — unchanged (index 1 is still y_vals)
Line 280: `ov_results[(d, _sc_overlay_qs[0])][0]` — unchanged (index 0 is still ts_d)
Lines 299-301: change `ts_d_q, y_vals_q = ov_results[(d, q)]` to `ts_d_q, y_vals_q, *_ = ov_results[(d, q)]`
Line 311: change `for (d, q), (ts_d, y_vals) in ov_results.items():` to `for (d, q), (ts_d, y_vals, depl_t_ov, t_start_d_ov, *_) in ov_results.items():`

- [ ] **Step 3: Add depletion arrows for overlay models**

After the overlay trace rendering (inside the delay loop for both layout 2 and the else branch), detect depletion and append:

```python
# After quantile traces for this delay in layout 2 (around line 308):
for q in _sc_overlay_qs:
    if (d, q) not in ov_results:
        continue
    _, _, depl_t_ov, t_start_d_ov, *_ = ov_results[(d, q)]
    if depl_t_ov is not None:
        _ov_tcol = _app_ctx.MODEL_TRACE_COLORS.get(mdl.short_name, "#CCCCCC")
        deplete_annots.append(_depl_annot(depl_t_ov, t_start_d_ov, d,
            arrow_col=delay_colors[di % len(delay_colors)],
            text_col=_ov_tcol,
            legendgroup=_ov_grp,
            model_prefix=mdl.legend_name,
            stagger=len(deplete_annots)))
        break  # one depletion arrow per (model, delay) — use first depleting quantile
```

Same pattern for the `else` branch (individual lines, around line 320).

- [ ] **Step 4: Syntax check and commit**

```bash
git add btc_web/figures/supercharge.py
git commit -m "feat: overlay models get depletion detection + arrows

ov_results expanded to include depletion data. Overlay depletion
arrows added to shared stagger pool with model-colored text."
```

---

### Task 5: Add overlay terminal annotations + terminal values in legend labels

**Files:**
- Modify: `btc_web/figures/supercharge.py:348-401`

- [ ] **Step 1: Move `_pending_annots` initialization before the overlay loop**

Currently `_pending_annots = []` is at line 348, after the overlay loop ends (~line 320). Move it to before the overlay loop (~line 243) so overlay code can append to it:

```python
# Before the overlay model loop:
_pending_annots = []
```

Then build overlay endpoint annotations **inside the overlay model loop** (Task 4 area), right after the overlay trace rendering for each model. For each overlay model, find the best surviving endpoint and append:

```python
# Inside overlay loop, after traces for this model:
if p.get("annotate") and not p.get("is_mobile"):
    _ov_tcol = _app_ctx.MODEL_TRACE_COLORS.get(mdl.short_name, "#CCCCCC")
    for di, d in enumerate(delays):
        # Find best surviving quantile at this delay
        surviving = [(q, ov_results[(d, q)]) for q in _sc_overlay_qs
                     if (d, q) in ov_results and ov_results[(d, q)][2] is None]  # depl_t is None = survived
        if not surviving:
            continue
        best_q, best_r = max(surviving, key=lambda x: float(x[1][1][-1]))
        y_final = float(best_r[1][-1])
        if y_final <= 0:
            continue
        lbl = fmt_price(y_final) if disp_mode == "usd" else f"{y_final:.4f}"
        _pending_annots.append(dict(
            x_arr=best_r[0], y_arr=best_r[1],
            label=f"{mdl.legend_name} {lbl}",
            short_label=f"{mdl.legend_name} {lbl}",
            color=_ov_tcol, y_last=y_final))
```

The primary model's endpoint annotations (line 354+) still append to the same `_pending_annots` list after the overlay loop. `_resolve_edge_annotations` sorts by `y_last` so insertion order doesn't matter.

- [ ] **Step 2: Add terminal values to all trace `name=` fields**

Update trace `name=` strings to include terminal value. For layout 2 primary model (line 233), the name currently is `f"{model.legend_name} {q_range}"`. Change to include a representative terminal value:

```python
# Compute a representative final value (e.g., median quantile at last delay)
_med_q = sel_qs[len(sel_qs) // 2] if sel_qs else 0.5
_rep_key = (delays[0], _med_q)
if _rep_key in results:
    _rep_final = fmt_price(float(results[_rep_key][1][-1])) if disp_mode == "usd" else f"{float(results[_rep_key][1][-1]):.4f}"
    _legend_name = f"{model.legend_name} {q_range} \u2192 {_rep_final}"
else:
    _legend_name = f"{model.legend_name} {q_range}"
```

Apply similar pattern for overlay models.

- [ ] **Step 3: Syntax check and commit**

```bash
git add btc_web/figures/supercharge.py
git commit -m "feat: overlay terminal annotations + terminal values in legend labels"
```

---

### Task 6: Create `sc_legend.js` — plotly_legendclick handler

**Files:**
- Create: `btc_web/assets/sc_legend.js`

- [ ] **Step 1: Write the JS handler**

```javascript
/**
 * sc_legend.js — Toggle depletion annotations when legend entries are clicked.
 *
 * Depletion arrows are layout.annotations with `name` set to the legendgroup
 * of their owning model. When a legend entry is clicked, this handler:
 * 1. Looks up the clicked trace's legendgroup via gd.data[curveNumber]
 * 2. Toggles ALL traces in that legendgroup (visible <-> "legendonly")
 * 3. Toggles ALL annotations whose name matches the legendgroup
 * 4. Returns false to prevent Plotly's default toggle (we handle everything)
 */
(function() {
    "use strict";

    function _bind(graphId) {
        var wrapper = document.getElementById(graphId);
        if (!wrapper) return;
        // Dash wraps the Plotly div — bind to the actual .js-plotly-plot child
        var gd = wrapper.querySelector(".js-plotly-plot") || wrapper;
        if (gd._scLegendBound) return;
        gd._scLegendBound = true;

        gd.on("plotly_legendclick", function(eventData) {
            var clickedTrace = gd.data[eventData.curveNumber];
            if (!clickedTrace || !clickedTrace.legendgroup) return;

            var lg = clickedTrace.legendgroup;

            // Determine new visibility: currently visible -> hide, hidden -> show
            var wasVisible = clickedTrace.visible !== "legendonly" && clickedTrace.visible !== false;
            var newVis = wasVisible ? "legendonly" : true;
            var newAnnotVis = !wasVisible;

            // Build per-trace visibility update (only traces in this legendgroup)
            var visArray = gd.data.map(function(t) {
                if (t.legendgroup === lg) return newVis;
                return t.visible === undefined ? true : t.visible;
            });

            // Build updated annotations (toggle matching name)
            var newAnnots = (gd.layout.annotations || []).map(function(a) {
                if (a.name === lg) {
                    return Object.assign({}, a, {visible: newAnnotVis});
                }
                return a;
            });

            // Atomic update: traces + annotations in one call
            Plotly.update(gd, {visible: [visArray]}, {annotations: newAnnots});

            // Return false — we handled everything, skip Plotly's default toggle
            return false;
        });
    }

    // Observe DOM for the supercharge graph (re-bind after Dash re-renders)
    var _observer = new MutationObserver(function() {
        _bind("supercharge-graph");
    });
    _observer.observe(document.body, {childList: true, subtree: true});

    document.addEventListener("DOMContentLoaded", function() {
        _bind("supercharge-graph");
    });
})();
```

**Key design decisions:**
- Returns `false` to take full control (no race condition with Plotly's internal toggle)
- Uses `Plotly.update()` for atomic trace + annotation update in one call
- Binds to `.querySelector(".js-plotly-plot")` child of the Dash wrapper (matches `scanner.js` pattern)
- `visible: [visArray]` — the outer array wraps all trace indices (Plotly.update format)

- [ ] **Step 2: Commit**

```bash
git add btc_web/assets/sc_legend.js
git commit -m "feat: sc_legend.js toggles depletion annotations on legend click"
```

---

### Task 7: Mobile — skip endpoint annotations when narrow viewport

**Files:**
- Modify: `btc_web/figures/supercharge.py`

- [ ] **Step 1: Read `is_mobile` from params and skip edge annotations**

In the endpoint annotation section (around line 348), wrap the `_pending_annots` building in a mobile check:

```python
_sc_log = bool(p.get("log_y"))
_pending_annots = []
if p.get("annotate") and not p.get("is_mobile"):
    # ... existing _pending_annots building code ...
```

When `is_mobile=True`, `_pending_annots` stays empty, `_resolve_edge_annotations` returns no traces, and users rely on legend labels for terminal values.

- [ ] **Step 2: Syntax check and commit**

```bash
git add btc_web/figures/supercharge.py
git commit -m "feat: skip on-chart endpoint annotations on mobile portrait

Legend labels always include terminal values; mobile users see
those instead of on-chart text traces."
```

---

### Task 8: Integration test and deploy

- [ ] **Step 1: Syntax check all modified files**

```bash
for f in btc_web/figures/supercharge.py btc_web/figures/common.py btc_web/_app_ctx.py btc_web/layout/common.py btc_web/callbacks/charts.py; do
    btc_venv/bin/python3 -m py_compile $f || echo "FAIL: $f"
done && echo "ALL OK"
```

- [ ] **Step 2: Run test suite**

```bash
PYTHONPATH=".:btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py -n 8 --timeout=120
```

- [ ] **Step 3: Manual verification**

Start dev server: `lsof -ti :8050 | xargs kill -9 2>/dev/null; DEV=1 nohup bash run_web.sh > /tmp/quantoshi_dev.log 2>&1 &`

Test on tab 5:
1. Enable BM + PL with 2 quantiles (Q1%, Q10%) and 2 delays (0, 2yr)
2. Verify: both models show depletion arrows with model-colored text
3. Click BM legend entry — BM traces AND BM depletion arrows disappear
4. Click PL legend entry — PL traces AND PL depletion arrows disappear
5. Verify arrows are staggered (not overlapping) at different heights
6. Verify terminal value appears in legend label text
7. Resize browser to < 768px — on-chart endpoint annotations disappear, legend labels still show values

- [ ] **Step 4: Deploy**

```bash
git push origin master
ssh root@89.167.70.45 "cd /opt/quantoshi && git pull && systemctl restart quantoshi"
```

---

## Summary

| Task | What | Risk |
|------|------|------|
| 1 | Expand stagger heights 3→5 | None — constant change |
| 2 | Viewport-width Store for mobile | Low — new Store + clientside callback |
| 3 | Tag depletion annotations with `name` | Low — add param to existing helper |
| 4 | Overlay depletion detection + arrows | Medium — ov_results tuple refactor |
| 5 | Overlay terminal annotations + legend values | Medium — scoping ov_results |
| 6 | sc_legend.js legendclick handler | Medium — JS event handling |
| 7 | Mobile skip endpoint annotations | Low — conditional on is_mobile |
| 8 | Integration test + deploy | None — verification |
