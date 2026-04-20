# Mobile Performance — Tab Islands Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut mobile control-interaction lag on tabs 1–6 by tightening Dash's island model — no cross-tab `Input` crosstalk, no full figure rebuilds for display-only changes.

**Architecture:** Five independently-deployable batches shipped as one commit each to `master`. Every batch ends with `FLUSHDB + systemctl restart quantoshi` on prod and a user mobile-verification gate; revert the single commit if regression. Uses existing patterns: commit-store (slider debounce + in-flight guard), clientside `Patch()` for display-only changes, per-tab first-render triggers for cross-tab isolation.

**Tech Stack:** Dash 4.0.0, `dash_clientside` JS callbacks, Plotly.js `Patch()`, Python 3.14 (dev) / 3.12 (prod), pytest, Redis (L2 cache).

**Spec:** `docs/superpowers/specs/2026-04-20-mobile-perf-tab-islands-design.md`

**Implementation order:** Batch 3 → Batch 1 → Batch 2 → Batch 4 → Batch 5.
Batch 3 first so spurious bridge-triggered rebuilds are gone before we validate slider-debounce perf on mobile.

---

## File Structure

| File | Role in this plan | Batches |
|---|---|---|
| `btc_web/callbacks/charts/_clientside.py` | All new clientside callbacks land here (slider commits, MC commits, Citadel palette patch). Bridge callback (lines 487-520) deleted in Batch 3. | 1, 2, 3, 5 |
| `btc_web/callbacks/charts/__init__.py` | Chart callbacks swap `Input("slider", "value")` → `Input("slider-commit", "data")`; `Input("{tab}-mc-*")` → single `Input("{tab}-mc-commit", "data")` + State. Heatmap loses color/text Inputs (become State). | 1, 2, 4 |
| `btc_web/callbacks/plot_appearance.py` | Palette-change callbacks bump `{active-tab}-first-render`. | 3 |
| `btc_web/callbacks/lots.py` | Lots-write callbacks bump `{active-tab}-first-render`. | 3 |
| `btc_web/callbacks/snapshot_cb.py` | Snapshot-restore bumps `{active-tab}-first-render`. | 3 |
| `btc_web/layout/` (bubble, heatmap, sim_tabs, citadel) | Add `{id}-commit` memory Stores + `{tab}-mc-commit` Stores + `{tab}-render-done` counter Stores. | 1, 2 |
| `btc_web/tab_defaults.py` | Keep in lockstep when chart-callback param dicts change (Batch 2 may add `mc_commit` placeholder). | 2 |
| `btc_web/figures/citadel.py` | Emit trace→model-key map alongside `cp-mc-results` for Batch 5 palette Patch. | 5 |
| `btc_web/test_cache_key_alignment.py` | Must pass after every batch that touches chart-callback params. | 2, (others) |

No new modules. No restructuring. All additions plug into existing islands.

---

## Pre-flight (shared across all batches)

- [ ] **Step P1: Run the existing suite on master, record baseline**

```bash
cd /scratch/code/bitcoinprojections && btc_venv/bin/python3 -m pytest btc_web/ -q --ignore-glob='*_e2e.py' 2>&1 | tail -5
```

Expected: all tests pass (1 pre-existing failure is acceptable — record which one in the commit message if it appears).

- [ ] **Step P2: Confirm Batch 4 heatmap cache routing is already correct**

Read `btc_web/utils.py:186-195`. Confirm `_get_heatmap_fig` calls `_quantize_params(p)` on line 194. If confirmed, the Batch 4 pre-batch audit is satisfied — note in the Batch 4 task that no cache-routing fix is needed.

---

## Task 1 (Batch 3) — Eliminate palette/lots bridge

**Rationale:** The bridge clientside callback at `btc_web/callbacks/charts/_clientside.py:487-520` watches `palette-store` and `effective-lots` as Inputs and bumps the active tab's `-first-render` counter. It fires on initial Store hydration (redundant rebuild) and on snapshot restore (race with the snapshot relay). Replacing it with per-source bumps eliminates both hazards.

**Files:**
- Modify: `btc_web/callbacks/charts/_clientside.py:487-520` (delete bridge)
- Modify: `btc_web/callbacks/plot_appearance.py` (palette dropdown click → bump active tab's first-render)
- Modify: `btc_web/callbacks/lots.py` (lots-write callbacks → bump active tab's first-render)
- Modify: `btc_web/callbacks/snapshot_cb.py` (snapshot restore → bump active tab's first-render)

- [ ] **Step 1.1: Delete the bridge callback**

Remove lines 487-520 of `btc_web/callbacks/charts/_clientside.py` (the block starting `# ── Palette / lots bridge` and ending at the bottom of the file). Keep the trailing newline.

- [ ] **Step 1.2: Add a shared active-tab-bump clientside helper**

Append to `btc_web/callbacks/charts/_clientside.py`:

```python
# ── Shared helper: bump the active chart tab's first-render counter ────
# Invoked by palette-change, lots-write, and snapshot-restore callbacks
# as a chained clientside output. The input trigger is a tick counter
# each source increments; we translate active_tab -> 5-output tuple.
_app_ctx.app.clientside_callback(
    """
    function(tick, active, bfr, hfr, dfr, rfr, sfr) {
        var NU = window.dash_clientside.no_update;
        if (!tick) return [NU, NU, NU, NU, NU];
        var map = {bubble:0, heatmap:1, dca:2, retire:3, supercharge:4};
        // Citadel patches its figure directly (Batch 5) -- exclude it here.
        var idx = map[active];
        if (idx === undefined) return [NU, NU, NU, NU, NU];
        var frs = [bfr, hfr, dfr, rfr, sfr];
        var out = [NU, NU, NU, NU, NU];
        out[idx] = (frs[idx] || 0) + 1;
        return out;
    }
    """,
    Output("bubble-first-render",      "data", allow_duplicate=True),
    Output("heatmap-first-render",     "data", allow_duplicate=True),
    Output("dca-first-render",         "data", allow_duplicate=True),
    Output("retire-first-render",      "data", allow_duplicate=True),
    Output("supercharge-first-render", "data", allow_duplicate=True),
    Input("active-tab-bump-tick", "data"),
    State("main-tabs",                 "active_tab"),
    State("bubble-first-render",       "data"),
    State("heatmap-first-render",      "data"),
    State("dca-first-render",          "data"),
    State("retire-first-render",       "data"),
    State("supercharge-first-render",  "data"),
    prevent_initial_call=True,
)
```

- [ ] **Step 1.3: Register the tick store in layout**

Modify `btc_web/layout/__init__.py` to add the tick store alongside the existing stores (search for `first-render` store registrations to find the right spot):

```python
dcc.Store(id="active-tab-bump-tick", storage_type="memory", data=0),
```

- [ ] **Step 1.4: Palette dropdown → bump tick**

In `btc_web/callbacks/plot_appearance.py`, find the callback that writes `palette-store.data`. Add `Output("active-tab-bump-tick", "data", allow_duplicate=True)` and return `(tick_now or 0) + 1` for that output. Corresponding `State("active-tab-bump-tick", "data")`. If the callback is server-side, add a tiny clientside tick-bump triggered by `palette-store.data`:

```python
_app_ctx.app.clientside_callback(
    "function(p, cur) { if (p === undefined || p === null) return window.dash_clientside.no_update; return (cur || 0) + 1; }",
    Output("active-tab-bump-tick", "data", allow_duplicate=True),
    Input("palette-store", "data"),
    State("active-tab-bump-tick", "data"),
    prevent_initial_call=True,
)
```

Place this in `btc_web/callbacks/charts/_clientside.py` right after the shared helper from Step 1.2.

- [ ] **Step 1.5: Lots + snapshot → bump tick**

Add matching clientside tick-bumps next to the palette one:

```python
_app_ctx.app.clientside_callback(
    """
    function(eff, snap, local, cur) {
        var ctx = window.dash_clientside.callback_context;
        if (!ctx.triggered || !ctx.triggered.length) return window.dash_clientside.no_update;
        return (cur || 0) + 1;
    }
    """,
    Output("active-tab-bump-tick", "data", allow_duplicate=True),
    Input("effective-lots", "data"),
    Input("snapshot-lots",  "data"),
    Input("lots-store",     "data"),
    State("active-tab-bump-tick", "data"),
    prevent_initial_call=True,
)
```

All three Stores trigger the tick — forecloses the 1-frame race where `effective-lots` lags a direct `lots-store` or `snapshot-lots` write. All three are cheap memory/local Stores; multiple triggers in quick succession just increment the tick more times (the downstream bump callback is idempotent on value equality from Dash's perspective, but each increment fires it — harmless).

- [ ] **Step 1.6: Run tests**

```bash
cd /scratch/code/bitcoinprojections && btc_venv/bin/python3 -m pytest btc_web/test_palette_roundtrip.py btc_web/test_snapshot.py btc_web/test_cache_key_alignment.py -v
```

Expected: all pass. If `test_palette_roundtrip.py` fails, the bridge deletion broke something — revert Step 1.1 and diagnose before continuing.

- [ ] **Step 1.7: Smoke-test on dev server**

```bash
lsof -ti :8050 | xargs -r kill -9
DEV=1 nohup bash /scratch/code/bitcoinprojections/run_web.sh > /tmp/quantoshi_dev.log 2>&1 &
sleep 5 && curl -s http://localhost:8050/ > /dev/null && tail -20 /tmp/quantoshi_dev.log
```

Manually verify in a browser tab (http://localhost:8050/1):
- Switch palettes on each of tabs 1–5 → chart re-renders with new colors.
- Import lots on tab 7, switch to tab 1 → bubble shows lot markers.
- No "nonexistent object" errors in `/tmp/quantoshi_dev.log`.

- [ ] **Step 1.8: Commit**

```bash
cd /scratch/code/bitcoinprojections && \
  git add btc_web/callbacks/charts/_clientside.py btc_web/layout/__init__.py && \
  git commit -m "perf(islands): replace palette/lots bridge with source-bumps

Eliminates the initial-hydration redundant rebuild and the snapshot-
restore race. Per-source tick + active-tab router replaces the Input-
driven bridge."
```

- [ ] **Step 1.9: Deploy and validate**

```bash
git push origin mobile-perf-tab-islands && \
  ssh root@89.167.70.45 "cd /opt/quantoshi && git pull origin mobile-perf-tab-islands && redis-cli FLUSHDB && systemctl restart quantoshi"
```

Mobile validation gate: user confirms "no regression" on prod (tabs load, palette switch works, snapshot links restore lots). If regression, `git revert HEAD && git push && ssh … redeploy`.

---

## Task 2 (Batch 1) — Slider debounce with 100ms in-flight guard

**Rationale:** Dash 4.0.0 re-introduces the every-pixel-fires problem on sliders (drag_value isn't available). Adding a client-side debounce with an in-flight guard cuts mobile round-trips to roughly 1 per drag instead of dozens, without queue buildup.

Target sliders (12): `bub-xrange`, `bub-yrange`, `hm-entry-yr`, `hm-exit-range`, `hm-b1`, `hm-b2`, `dca-yr`, `ret-yr`, `sc-yr`, `cp-yr-range`, `bub-ptsize`, `bub-ptalpha`.

Not a slider: `hm-entry-q` is a `dbc.Input(type=number)` — handle separately with `debounce=True` on the component itself.

**Files:**
- Modify: `btc_web/layout/` (bubble, heatmap, sim_tabs, citadel, supercharge) — add `{id}-commit` Stores + `{tab}-render-done` counter Stores
- Modify: `btc_web/callbacks/charts/_clientside.py` — 12 debounce callbacks + 5 render-done counters
- Modify: `btc_web/callbacks/charts/__init__.py` — swap `Input("{id}", "value")` → `Input("{id}-commit", "data")` on chart callbacks
- Modify: component for `hm-entry-q` — add `debounce=True`

- [ ] **Step 2.1: Add a per-tab render-done counter**

In `btc_web/layout/__init__.py`, alongside the existing `{tab}-first-render` Stores, add:

```python
dcc.Store(id="bubble-render-done",      storage_type="memory", data=0),
dcc.Store(id="heatmap-render-done",     storage_type="memory", data=0),
dcc.Store(id="dca-render-done",         storage_type="memory", data=0),
dcc.Store(id="retire-render-done",      storage_type="memory", data=0),
dcc.Store(id="supercharge-render-done", storage_type="memory", data=0),
dcc.Store(id="citadel-render-done",     storage_type="memory", data=0),
```

- [ ] **Step 2.2: Add the render-done incrementer clientside callback (one per tab)**

In `btc_web/callbacks/charts/_clientside.py`, append:

```python
# ── Render-done counters: Plotly fires relayoutData/afterplot on each render.
# We bump the counter once per figure update so slider debouncers can
# gate the next commit on previous render completion. ───────────────────
for _tab, _graph in [
    ("bubble", "bubble-graph"),
    ("heatmap", "heatmap-graph"),
    ("dca", "dca-graph"),
    ("retire", "retire-graph"),
    ("supercharge", "supercharge-graph"),
    ("citadel", "citadel-graph"),
]:
    _app_ctx.app.clientside_callback(
        """
        function(fig, cur) {
            if (!fig) return window.dash_clientside.no_update;
            return (cur || 0) + 1;
        }
        """,
        Output(f"{_tab}-render-done", "data"),
        Input(f"{_graph}", "figure"),
        State(f"{_tab}-render-done", "data"),
        prevent_initial_call=True,
    )
```

- [ ] **Step 2.3: Add `{id}-commit` Stores for all 12 sliders**

In the same layout file(s) where the slider lives, add a memory Store with the same id suffixed with `-commit`. Example for bubble (`btc_web/layout/bubble.py` — find the slider definitions, add Stores nearby, or batch them all in `layout/__init__.py`):

```python
dcc.Store(id="bub-xrange-commit",   storage_type="memory", data=None),
dcc.Store(id="bub-yrange-commit",   storage_type="memory", data=None),
dcc.Store(id="bub-ptsize-commit",   storage_type="memory", data=None),
dcc.Store(id="bub-ptalpha-commit",  storage_type="memory", data=None),
dcc.Store(id="hm-entry-yr-commit",  storage_type="memory", data=None),
dcc.Store(id="hm-exit-range-commit",storage_type="memory", data=None),
dcc.Store(id="hm-b1-commit",        storage_type="memory", data=None),
dcc.Store(id="hm-b2-commit",        storage_type="memory", data=None),
dcc.Store(id="dca-yr-commit",       storage_type="memory", data=None),
dcc.Store(id="ret-yr-commit",       storage_type="memory", data=None),
dcc.Store(id="sc-yr-commit",        storage_type="memory", data=None),
dcc.Store(id="cp-yr-range-commit",  storage_type="memory", data=None),
```

Initial data should be `None` so first render isn't blocked (chart callbacks read `None` as "unset, use slider value fallback").

- [ ] **Step 2.4: Add the 12 debounce clientside callbacks**

Append to `btc_web/callbacks/charts/_clientside.py`. One clientside callback per slider, generated in a loop. Each has `Input({slider}, "value")` + `Input({tab}-render-done, "data")` and writes `Output({slider}-commit, "data")` after a 100ms debounce gated by the render-done counter.

```python
_SLIDER_IDS = [
    ("bub-xrange", "bubble"), ("bub-yrange", "bubble"),
    ("bub-ptsize", "bubble"), ("bub-ptalpha", "bubble"),
    ("hm-entry-yr", "heatmap"), ("hm-exit-range", "heatmap"),
    ("hm-b1", "heatmap"), ("hm-b2", "heatmap"),
    ("dca-yr", "dca"), ("ret-yr", "retire"),
    ("sc-yr", "supercharge"), ("cp-yr-range", "citadel"),
]
for _sid, _tab in _SLIDER_IDS:
    _app_ctx.app.clientside_callback(
        """
        function(v, renderDone, cur) {
            var NU = window.dash_clientside.no_update;
            if (v === undefined || v === null) return NU;
            var W = window;
            W.__qs_slider = W.__qs_slider || {};
            var key = "__SID__";
            var st = W.__qs_slider[key] || {timer: null, lastEmit: 0, pending: null, resolve: null};
            st.pending = v;
            if (st.timer) clearTimeout(st.timer);
            // Promise that resolves to the commit value when debounce fires.
            var promise = new Promise(function(res) { st.resolve = res; });
            st.timer = setTimeout(function tick() {
                if (st.lastEmit && renderDone < st.lastEmit) {
                    st.timer = setTimeout(tick, 100);
                    return;
                }
                st.lastEmit = (cur || 0) + 1;
                W.__qs_slider[key] = st;
                st.resolve(st.pending);
            }, 100);
            W.__qs_slider[key] = st;
            return promise;
        }
        """.replace("__SID__", _sid),
        Output(f"{_sid}-commit", "data"),  # no allow_duplicate — each slider has its own unique Output
        Input(_sid, "value"),
        Input(f"{_tab}-render-done", "data"),
        State(f"{_sid}-commit", "data"),
        prevent_initial_call=True,
    )
```

**Convention:** Each `-commit` store is written by exactly one callback, so no `allow_duplicate=True` is needed. Dash supports returning a Promise from a clientside callback (resolved value becomes the Output), which is how the async 100ms debounce writes through the declared Output rather than `set_props`. This avoids the `allow_duplicate` + `prevent_initial_call` footgun pattern flagged in CLAUDE.md.

- [ ] **Step 2.5: Swap chart-callback Inputs from slider.value → slider-commit.data**

In `btc_web/callbacks/charts/__init__.py`, for each of the 12 sliders in Step 2.3: locate the `@callback` decorator above the matching `update_*` function (`update_bubble`, `update_heatmap`, `update_dca`, `update_retire`, `update_supercharge`, `update_citadel`) and change:

```python
Input("bub-xrange", "value"),   →   Input("bub-xrange-commit", "data"),
```

…for all 12 slider inputs.

**Guard for initial value:** inside the callback, if the commit value is `None`, fall through to the slider's `State("{id}", "value")` as a fallback. Add a matching `State("{id}", "value")` for each slider right after its commit-Input. Example for `update_bubble`:

```python
Input("bub-xrange-commit", "data"),
State("bub-xrange",        "value"),  # fallback on initial render
```

And at the top of the function body:
```python
xrange = xrange_commit if xrange_commit is not None else xrange_fallback
```

**Arg-order verification:** Dash passes callback args in declaration order (all Inputs first, then all States). When adding the 12 fallback States, put each fallback State immediately after the other States (not interleaved with the commit Inputs). Before committing, grep each `update_*` signature against its decorator and verify name/position match. Mismatched order silently swaps values.

- [ ] **Step 2.6: Set `debounce=True` on `hm-entry-q`**

Find `hm-entry-q` in `btc_web/layout/heatmap.py` (or wherever the `dbc.Input(id="hm-entry-q", ...)` is declared). Add `debounce=True` to that component's kwargs. No other changes for this control.

- [ ] **Step 2.7: Run tests**

```bash
cd /scratch/code/bitcoinprojections && btc_venv/bin/python3 -m pytest btc_web/ -q --ignore-glob='*_e2e.py'
```

Expected: all pass. Snapshot tests read `slider.value`, which is unchanged — only the chart callbacks moved to `-commit`.

- [ ] **Step 2.8: Smoke-test on dev server**

```bash
lsof -ti :8050 | xargs -r kill -9
DEV=1 nohup bash /scratch/code/bitcoinprojections/run_web.sh > /tmp/quantoshi_dev.log 2>&1 &
sleep 5
```

Drag each slider; confirm:
- Chart updates within ~200ms of release
- No queue-buildup lag if dragging continuously for 5 seconds
- `/tmp/quantoshi_dev.log` shows roughly one callback per drag, not dozens

- [ ] **Step 2.9: Commit**

```bash
cd /scratch/code/bitcoinprojections && \
  git add btc_web/callbacks/charts/_clientside.py btc_web/callbacks/charts/__init__.py \
          btc_web/layout/ && \
  git commit -m "perf(islands): 100ms slider-commit debounce with in-flight guard

12 sliders across tabs 1-6 now write through {id}-commit stores. Chart
callbacks read commit-stores; render-done counter gates concurrent
emissions. hm-entry-q uses built-in debounce=True."
```

- [ ] **Step 2.10: Deploy and mobile-validate**

```bash
git push origin mobile-perf-tab-islands && \
  ssh root@89.167.70.45 "cd /opt/quantoshi && git pull origin mobile-perf-tab-islands && redis-cli FLUSHDB && systemctl restart quantoshi"
```

User tests on mobile. Approve or revert.

---

## Task 3 (Batch 2) — Per-tab MC commit Store

**Rationale:** Each of the 5 MC-enabled tabs currently lists ~15 `{tab}-mc-*` controls as individual Inputs to its chart callback. Consolidating into a single `{tab}-mc-commit` Store removes 75 Input slots in exchange for 5 commit stores and matches the island model.

MC-enabled tabs: `dca`, `retire`, `supercharge`, `heatmap`, `citadel`.

**Files:**
- Modify: `btc_web/layout/__init__.py` — add 5 `{tab}-mc-commit` Stores
- Modify: `btc_web/callbacks/charts/_clientside.py` — 5 aggregator clientside callbacks
- Modify: `btc_web/callbacks/charts/__init__.py` — demote 75 `{tab}-mc-*` Inputs to State, add 5 commit-Inputs
- Modify: `btc_web/tab_defaults.py` + `cache._prewarm_caches()` — keep cache keys aligned

- [ ] **Step 3.1: Add the 5 MC commit Stores**

In `btc_web/layout/__init__.py` alongside existing Stores:

```python
dcc.Store(id="dca-mc-commit",        storage_type="memory", data=None),
dcc.Store(id="retire-mc-commit",     storage_type="memory", data=None),
dcc.Store(id="supercharge-mc-commit",storage_type="memory", data=None),
dcc.Store(id="heatmap-mc-commit",    storage_type="memory", data=None),
dcc.Store(id="citadel-mc-commit",    storage_type="memory", data=None),
```

- [ ] **Step 3.2: Enumerate per-tab MC control IDs**

Read `btc_web/callbacks/charts/__init__.py` and list every `Input("{tab}-mc-*", ...)` for each of the 5 tabs. Example for DCA (verify by reading the actual `update_dca` decorator):

```
dca: dca-mc-enable, dca-mc-amount, dca-mc-infl, dca-mc-bins, dca-mc-regime,
     dca-mc-sims, dca-mc-years, dca-mc-freq, dca-mc-window, dca-mc-start-yr,
     dca-mc-entry-q, dca-mc-loaded, dca-mc-model-src, dca-mc-run-btn,
     dca-mc-rendered-key
```

Write the full list per tab into a comment at the top of the aggregator block in `_clientside.py` so a future reader can audit.

- [ ] **Step 3.3: Add one aggregator clientside callback per tab**

In `btc_web/callbacks/charts/_clientside.py`, for each MC tab:

```python
# Example: DCA. Repeat for retire, supercharge, heatmap, citadel.
_app_ctx.app.clientside_callback(
    """
    function(enable, amount, infl, bins, regime, sims, years, freq, window,
             start_yr, entry_q, loaded, model_src, run_btn, rendered_key,
             renderDone, cur) {
        var NU = window.dash_clientside.no_update;
        var W = window;
        W.__qs_mc = W.__qs_mc || {};
        var key = "dca-mc";
        var st = W.__qs_mc[key] || {timer: null, lastEmit: 0};
        st.pending = {
            enable: enable, amount: amount, infl: infl, bins: bins,
            regime: regime, sims: sims, years: years, freq: freq,
            window: window, start_yr: start_yr, entry_q: entry_q,
            loaded: loaded, model_src: model_src, run_btn: run_btn,
            rendered_key: rendered_key,
        };
        if (st.timer) clearTimeout(st.timer);
        var promise = new Promise(function(res) { st.resolve = res; });
        st.timer = setTimeout(function tick() {
            if (st.lastEmit && renderDone < st.lastEmit) {
                st.timer = setTimeout(tick, 100);
                return;
            }
            st.lastEmit = (cur || 0) + 1;
            W.__qs_mc[key] = st;
            st.resolve(st.pending);
        }, 100);
        W.__qs_mc[key] = st;
        return promise;
    }
    """,
    Output("dca-mc-commit", "data"),  # unique Output, no allow_duplicate
    Input("dca-mc-enable",       "value"),
    Input("dca-mc-amount",       "value"),
    Input("dca-mc-infl",         "value"),
    Input("dca-mc-bins",         "value"),
    Input("dca-mc-regime",       "value"),
    Input("dca-mc-sims",         "value"),
    Input("dca-mc-years",        "value"),
    Input("dca-mc-freq",         "value"),
    Input("dca-mc-window",       "value"),
    Input("dca-mc-start-yr",     "value"),
    Input("dca-mc-entry-q",      "value"),
    Input("dca-mc-loaded",       "data"),
    Input("dca-mc-model-src",    "value"),
    Input("dca-mc-run-btn",      "n_clicks"),
    Input("dca-mc-rendered-key", "data"),
    Input("dca-render-done",     "data"),
    State("dca-mc-commit",       "data"),
    prevent_initial_call=True,
)
```

Copy/paste and rename for `retire`, `supercharge`, `heatmap`, `citadel`. Each tab's control list is the one captured in Step 3.2. **Do not paraphrase** — the key names in `pending` must match the field names the chart callbacks look for.

- [ ] **Step 3.4: Demote MC Inputs to State on each chart callback**

In `btc_web/callbacks/charts/__init__.py`:

For `update_dca`, `update_retire`, `update_supercharge`, `update_heatmap`: replace every `Input("{tab}-mc-*", ...)` in the `@callback` decorator with `State(...)` of the same id/prop, and add ONE new Input at the top:

```python
Input("{tab}-mc-commit", "data"),
```

For `update_citadel` in `btc_web/callbacks/citadel_cb.py`: same treatment — add `Input("citadel-mc-commit", "data")`, demote existing cp-mc-* Inputs to State.

The function signature gets a new first-or-last arg `mc_commit` (a dict or None). Inside the function, if the callback already reads individual values, keep using the State-sourced values; the commit-Input is just a trigger. No body changes needed beyond adding the arg name.

- [ ] **Step 3.5: Enumerate MC params actually in each `params` dict**

Before touching defaults, read each of `update_dca`, `update_retire`, `update_supercharge`, `update_heatmap`, `update_citadel` and list every `mc_*` key actually written into the dict that's passed to `_get_*_fig` (or to `_get_mc_or_cached`). Expected: typically `mc_enabled`, `mc_amount`, `mc_infl`, `mc_bins`, `mc_sims`, `mc_years`, `mc_freq`, `mc_window`, `mc_start_yr`, `mc_entry_q`, `mc_regime`, `mc_model_src` — but verify per callback since they diverge.

- [ ] **Step 3.6: Update `tab_defaults.py` to match**

For each MC key enumerated in Step 3.5, confirm it appears in the matching tab's `MappingProxyType` default dict in `btc_web/tab_defaults.py`. If any key is missing, add it with the frozen default value (match what the callback produces when MC is disabled). The point is cache-key alignment: the prewarm L1 key must equal the runtime L1 key on first tab visit.

- [ ] **Step 3.7: Update `_prewarm_caches()` in `btc_web/cache.py`**

Read `_prewarm_caches()` and confirm it builds each tab's params from `_defaults()` (or equivalent). If it hardcodes a subset of MC keys, update it to produce the same params dict the runtime callback produces when MC is off (i.e., iterate the defaults). Commit this change in the same commit as the callback edits.

- [ ] **Step 3.8: Run cache-alignment tests**

```bash
cd /scratch/code/bitcoinprojections && btc_venv/bin/python3 -m pytest btc_web/test_cache_key_alignment.py -v
```

Expected: all pass. If any alignment test fails, the mismatch it reports names the missing key — add it to `tab_defaults.py` and rerun.

- [ ] **Step 3.9: Run full suite**

```bash
cd /scratch/code/bitcoinprojections && btc_venv/bin/python3 -m pytest btc_web/ -q --ignore-glob='*_e2e.py'
```

Expected: all pass.

- [ ] **Step 3.10: Invoke cache-key-aligner subagent**

Per CLAUDE.md and the cache-key-aligner agent description, run that agent against this commit before deploying — MC refactors are exactly the pattern it catches.

```
[Agent invocation with prompt: "Verify cache-key alignment for the Batch 2
  MC commit-store refactor. tab_defaults.py, _prewarm_caches(), and the
  chart-callback params dicts in callbacks/charts/__init__.py for update_dca,
  update_retire, update_supercharge, update_heatmap, plus update_citadel in
  callbacks/citadel_cb.py. Report any mismatched keys."]
```

Apply any fixes the agent finds, then rerun the suite.

- [ ] **Step 3.11: Smoke test on dev**

```bash
lsof -ti :8050 | xargs -r kill -9
DEV=1 nohup bash /scratch/code/bitcoinprojections/run_web.sh > /tmp/quantoshi_dev.log 2>&1 &
sleep 5
```

DEV mode has MC disabled (Markov .so not built locally), so MC control changes should no-op. Confirm no new "nonexistent object" errors. Visit `/3`, `/4`, `/5`, `/2`, `/6` — each tab's chart renders.

- [ ] **Step 3.12: Commit + deploy + mobile-validate**

```bash
cd /scratch/code/bitcoinprojections && \
  git add btc_web/callbacks/charts/_clientside.py \
          btc_web/callbacks/charts/__init__.py \
          btc_web/callbacks/citadel_cb.py \
          btc_web/layout/__init__.py \
          btc_web/tab_defaults.py \
          btc_web/cache.py && \
  git commit -m "perf(islands): per-tab MC commit-store (Batch 2)

5 MC-enabled tabs (DCA, retire, supercharge, heatmap, citadel) each get
a {tab}-mc-commit memory Store aggregating ~15 MC controls. Chart
callbacks demote MC Inputs to State, wake on single commit-Input.
Removes 75 Input slots; 5 commit-Inputs in their place."

git push origin mobile-perf-tab-islands && \
  ssh root@89.167.70.45 "cd /opt/quantoshi && git pull origin mobile-perf-tab-islands && redis-cli FLUSHDB && systemctl restart quantoshi"
```

Mobile validation: MC runs on all 5 tabs; revert if regression.

---

## Task 4 (Batch 4) — Heatmap color/text Patch

**Rationale:** `update_heatmap` currently rebuilds the full figure for every Input, including text-only (`hm-vfmt`, `hm-cell-fs`) changes. A Patch-only clientside update avoids the rebuild.

**Scope note:** The spec's full ambition (colorscale Patch for `hm-b1/b2/palette/c-*/grad/mode`) requires porting `_dense_colorscale` to JS and is deferred. This batch ships the text-only Patch, which is the minimal safe slice.

Pre-batch audit (Step P2) confirmed `_get_heatmap_fig` already routes through `_quantize_params` — no cache fix needed.

**Files:**
- Modify: `btc_web/callbacks/charts/__init__.py` — demote `hm-vfmt`, `hm-cell-fs` Inputs to State
- Modify: `btc_web/callbacks/charts/_clientside.py` — 1 new text-Patch callback

- [ ] **Step 4.1: Demote text-only Inputs to State on `update_heatmap`**

In `btc_web/callbacks/charts/__init__.py`, inside the `@callback` for `update_heatmap` (starts at line 670): change **only the two text-only controls** from `Input` to `State`:

```
hm-vfmt, hm-cell-fs
```

Keep color controls (`hm-b1`, `hm-b2`, `hm-palette`, `hm-c-*`, `hm-grad`, `hm-mode`) as `Input` in this batch — the correct Patch path requires porting `_dense_colorscale` (256-pt rgb scale with diverging zmin/zmax symmetry) to JS, which is out of scope. Batch 1's slider commit-store already gates `hm-b1`/`hm-b2` drag frequency, which is the biggest color perceptible win.

Keep matrix-dependent Inputs (entry_yr, entry_q, exit_range, stack, use_lots, active_model, all MC-commit) as `Input`.

- [ ] **Step 4.2: Add text-only Patch clientside callback**

```python
# ── Heatmap text-only Patch: cell font size + value format ────────────
_app_ctx.app.clientside_callback(
    """
    function(vfmt, cellFs, curFig) {
        var NU = window.dash_clientside.no_update;
        if (!curFig || !curFig.data || !curFig.data[0]) return NU;
        var p = window.dash_clientside.Patch();
        if (cellFs !== undefined && cellFs !== null) {
            p["data"][0]["textfont"]["size"] = cellFs;
        }
        // vfmt drives texttemplate. Map enum -> template string.
        var fmtMap = {
            pct:  "%{z:.1%}",
            pct0: "%{z:.0%}",
            pct2: "%{z:.2%}",
        };
        if (vfmt && fmtMap[vfmt]) {
            p["data"][0]["texttemplate"] = fmtMap[vfmt];
        }
        return p;
    }
    """,
    Output("heatmap-graph", "figure", allow_duplicate=True),
    Input("hm-vfmt",    "value"),
    Input("hm-cell-fs", "value"),
    State("heatmap-graph", "figure"),
    prevent_initial_call=True,
)
```

Verify `fmtMap` keys by reading `btc_web/figures/heatmap.py` — whatever the server writes to `texttemplate` for each `vfmt` value, mirror here.

- [ ] **Step 4.3: Run tests + smoke test**

```bash
cd /scratch/code/bitcoinprojections && btc_venv/bin/python3 -m pytest btc_web/test_cache_key_alignment.py btc_web/test_figures.py -v
```

Dev smoke: drag `hm-cell-fs`, change `hm-vfmt` → heatmap text updates without a server round-trip (no new log line in `/tmp/quantoshi_dev.log`).

- [ ] **Step 4.4: Commit + deploy + mobile-validate**

```bash
cd /scratch/code/bitcoinprojections && \
  git add btc_web/callbacks/charts/__init__.py btc_web/callbacks/charts/_clientside.py && \
  git commit -m "perf(islands): heatmap text-only Patch (Batch 4)

hm-vfmt and hm-cell-fs demote to State on update_heatmap; a clientside
Patch applies texttemplate + textfont.size with no server round-trip.

Color Patch deferred pending a JS port of _dense_colorscale.
hm-palette / hm-c-* / hm-grad / hm-mode remain full-rebuild Inputs
and still fire server round-trips per click. hm-b1 / hm-b2 drag
frequency is already gated by Batch 1 slider-commit."

git push origin mobile-perf-tab-islands && \
  ssh root@89.167.70.45 "cd /opt/quantoshi && git pull origin mobile-perf-tab-islands && redis-cli FLUSHDB && systemctl restart quantoshi"
```

Mobile: change heatmap cell font size / value format / palette → responsive. Revert if regression.

---

## Task 5 (Batch 5) — Citadel palette Patch

**Rationale:** Citadel already keeps display-only controls as State. The remaining mobile win: palette switching post-simulation currently requires re-clicking "▶ Run Simulation". A Patch-only clientside callback updates trace colors in place.

**Files:**
- Modify: `btc_web/figures/citadel.py` — emit trace→model-key map alongside `cp-mc-results`
- Modify: `btc_web/callbacks/charts/_clientside.py` — single palette Patch clientside callback

- [ ] **Step 5.1: Emit a trace→model-key map from `build_citadel_figure`**

In `btc_web/figures/citadel.py`, inside `build_citadel_figure`, after each trace is appended, write the trace's model key into `fig.data[i].customdata` OR into a side Store. Simpler: set `meta={"qs_model_key": model_short_name}` on each trace:

```python
fig.add_trace(go.Scatter(
    x=..., y=...,
    line=dict(color=color, width=width),
    fillcolor=fillcolor,
    meta={"qs_model_key": model_key},   # NEW
    ...
))
```

The palette-Patch callback reads `fig.data[i].meta.qs_model_key` to know which color to assign.

- [ ] **Step 5.2: Add the palette Patch clientside callback**

Append to `btc_web/callbacks/charts/_clientside.py`:

```python
# ── Citadel palette Patch: swap trace colors post-sim without rebuild ──
_app_ctx.app.clientside_callback(
    """
    function(palette, curFig) {
        var NU = window.dash_clientside.no_update;
        if (!palette || !curFig || !curFig.data) return NU;
        var QS = window.QS_PALETTES && window.QS_PALETTES[palette];
        if (!QS || !QS.model_colors) return NU;
        // Hex -> rgb helper so we can re-apply any existing fillcolor alpha.
        function hexToRgb(h) {
            var m = (h || "").match(/^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i);
            if (!m) return null;
            return [parseInt(m[1],16), parseInt(m[2],16), parseInt(m[3],16)];
        }
        // Extract alpha from an existing "rgba(r,g,b,a)" string, else 1.
        function extractAlpha(s) {
            var m = (s || "").match(/rgba?\([^)]*,\s*([0-9.]+)\s*\)/);
            return m ? parseFloat(m[1]) : 1.0;
        }
        var p = window.dash_clientside.Patch();
        for (var i = 0; i < curFig.data.length; i++) {
            var tr = curFig.data[i];
            var key = tr.meta && tr.meta.qs_model_key;
            if (!key) continue;
            var c = QS.model_colors[key];
            if (!c) continue;
            p["data"][i]["line"]["color"] = c;
            if (tr.fillcolor) {
                var rgb = hexToRgb(c);
                if (rgb) {
                    var a = extractAlpha(tr.fillcolor);
                    p["data"][i]["fillcolor"] = "rgba(" + rgb[0] + "," + rgb[1] + "," + rgb[2] + "," + a + ")";
                }
            }
        }
        return p;
    }
    """,
    Output("citadel-graph", "figure", allow_duplicate=True),
    Input("palette-store", "data"),
    State("citadel-graph", "figure"),
    prevent_initial_call=True,
)
```

`window.QS_PALETTES` is already populated by `_colors_generated.js`; verify by opening the dev server and checking `window.QS_PALETTES.default.model_colors` in the browser console.

- [ ] **Step 5.3: Re-check Batch 3's active-tab skip**

Because this batch lands, revisit the Batch 3 helper from Step 1.2 and confirm `citadel` is already excluded from the active-tab bump map (it is — only `bubble/heatmap/dca/retire/supercharge` are in `map`). No code change needed here, but verify before committing.

- [ ] **Step 5.4: Smoke test**

```bash
lsof -ti :8050 | xargs -r kill -9
DEV=1 nohup bash /scratch/code/bitcoinprojections/run_web.sh > /tmp/quantoshi_dev.log 2>&1 &
sleep 5
```

Go to `/6`, click Run Simulation, wait for figure; then switch palettes → trace colors change without a full rebuild (no new server log line for `update_citadel`).

- [ ] **Step 5.5: Run tests**

```bash
cd /scratch/code/bitcoinprojections && btc_venv/bin/python3 -m pytest btc_web/test_citadel.py btc_web/test_palette_roundtrip.py -v
```

Expected: all pass. If a test checks `customdata` / `meta`, it may need an update — adjust the test to assert `meta["qs_model_key"]` equals the expected short name.

- [ ] **Step 5.6: Commit + deploy + mobile-validate**

```bash
cd /scratch/code/bitcoinprojections && \
  git add btc_web/figures/citadel.py btc_web/callbacks/charts/_clientside.py && \
  git commit -m "perf(islands): Citadel palette Patch (Batch 5)

build_citadel_figure tags each trace with meta.qs_model_key. A single
clientside callback patches trace line.color/fillcolor on palette-store
change. Post-sim palette switches no longer require re-running the sim."

git push origin mobile-perf-tab-islands && \
  ssh root@89.167.70.45 "cd /opt/quantoshi && git pull origin mobile-perf-tab-islands && redis-cli FLUSHDB && systemctl restart quantoshi"
```

Mobile: run a sim on Citadel, switch palettes → colors update instantly.

---

## Rollback policy (applies to every batch)

If a batch regresses on prod:

```bash
cd /scratch/code/bitcoinprojections && \
  git revert --no-edit HEAD && \
  git push origin mobile-perf-tab-islands && \
  ssh root@89.167.70.45 "cd /opt/quantoshi && git pull origin mobile-perf-tab-islands && redis-cli FLUSHDB && systemctl restart quantoshi"
```

Then diagnose on dev with fresh context before re-attempting the batch.

---

## Self-review checklist (for the implementer)

Before starting a batch:
- [ ] Read the spec section for that batch.
- [ ] Confirm the previous batch landed cleanly on prod (no open regression).

After finishing a batch:
- [ ] Full test suite passes (`pytest btc_web/ -q --ignore-glob='*_e2e.py'`).
- [ ] Dev server boots without new "nonexistent object" errors in `/tmp/quantoshi_dev.log`.
- [ ] Commit message names the batch and the user-visible change.
- [ ] Prod health check (`scripts/quantoshi-health`) clean after restart.
- [ ] User explicitly approves on mobile before starting the next batch.
