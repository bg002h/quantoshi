# Single-Redraw-per-Snapshot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut chart-callback fires during share-link restore from 2–3 to 1 by gating figure writes on a `snapshot-pending` Store, saving ~800 ms on the critical path.

**Architecture:** Add `dcc.Store(id="snapshot-pending", data=False)`. `restore_from_url` arms it `True`; each `apply_tab_{tab}` releases it `False` in the same output batch as tab controls; a 3-second clientside safety timer releases as fallback for non-chart-tab restores. All 9 figure-writing chart callbacks early-return `no_update` when the gate is True, before any other logic (so restores always settle deterministically).

**Tech Stack:** Dash 4.0.0, dbc 2.0.4, Python 3.14 dev / 3.12 prod, pytest.

**Spec:** `docs/superpowers/specs/2026-04-24-single-redraw-per-snapshot-design.md`
**Parent refactor:** commit `7e929cd` (drop-all-tabs snapshot, shipped earlier today)

**Deploy command (user-delegated autonomous):**
```bash
git push origin master && ssh root@89.167.70.45 "cd /opt/quantoshi && git pull && redis-cli FLUSHDB && systemctl restart quantoshi"
```

---

## Task 1: Add failing tests (RED)

**Files:**
- Modify: `btc_web/test_snapshot.py`

- [ ] **Step 1.1: Append the new test class to `test_snapshot.py`**

```python


class TestSnapshotPendingGate:
    """Tests for the snapshot-pending gate that reduces chart redraws.
    Per spec docs/superpowers/specs/2026-04-24-single-redraw-per-snapshot-design.md."""

    def test_snapshot_pending_in_layout(self):
        """The snapshot-pending Store must exist in the rendered layout."""
        import layout
        import json
        rendered = layout._serve_layout() if hasattr(layout, "_serve_layout") else None
        serialised = json.dumps(rendered, default=str) if rendered else ""
        assert "snapshot-pending" in serialised, (
            "snapshot-pending Store missing from layout")

    def test_restore_from_url_uses_initial_duplicate(self):
        """restore_from_url must use prevent_initial_call='initial_duplicate'
        because it now has an allow_duplicate=True Output."""
        import _app_ctx
        app = _app_ctx.app
        for cb_key, entry in app.callback_map.items():
            if "loaded-hash-store.data" in cb_key and "snapshot-state-store.data" in cb_key:
                assert entry.get("prevent_initial_call") == "initial_duplicate", (
                    f"restore_from_url must use 'initial_duplicate', got "
                    f"{entry.get('prevent_initial_call')}")
                return
        raise AssertionError("restore_from_url callback not found in callback_map")

    def test_apply_tab_outputs_include_snapshot_pending(self):
        """Each of the 7 apply_tab_{tab} callbacks must output snapshot-pending.data."""
        import _app_ctx
        app = _app_ctx.app
        hits = 0
        for cb_key in app.callback_map:
            outputs = cb_key.split("...")
            clean = [o.split("@")[0] for o in outputs]
            # apply_tab_{tab} signature: writes many per-tab cids + snapshot-pending
            # Match by presence of tab-specific cids AND snapshot-pending.
            if ("snapshot-pending.data" in clean and
                any(c.split(".")[0].startswith(("bub-", "hm-", "dca-",
                                                 "ret-", "sc-", "cp-", "lev-"))
                    for c in clean)):
                hits += 1
        assert hits == 7, (
            f"Expected 7 apply_tab_* callbacks with snapshot-pending output; got {hits}")

    def test_apply_tab_releases_gate_on_populated_state(self):
        """apply_tab_bubble with populated state returns False (release) as last output."""
        from callbacks.snapshot_cb import apply_tab_bubble
        state = {"bub-xscale:value": "Lin", "bub-qs:value": [0.5]}
        result = apply_tab_bubble(1, state)
        assert result[-1] is False, (
            f"Last output must be False to release gate; got {result[-1]!r}")

    def test_apply_tab_does_not_clear_gate_when_state_none(self):
        """apply_tab_bubble with state=None returns no_update for gate output
        (NOT False) — so non-restore first-render bumps don't accidentally
        clear the gate."""
        from callbacks.snapshot_cb import apply_tab_bubble
        from dash import no_update
        result = apply_tab_bubble(None, None)
        assert result[-1] is no_update, (
            f"Gate output must be no_update when state is None; got {result[-1]!r}")
        assert all(x is no_update for x in result), (
            "All outputs must be no_update when state is None")

    def test_snapshot_pending_writers_have_allow_duplicate(self):
        """Every callback that outputs snapshot-pending.data must use
        allow_duplicate=True (the @ marker in the callback_map key)."""
        import _app_ctx
        app = _app_ctx.app
        for cb_key in app.callback_map:
            parts = cb_key.split("...")
            for part in parts:
                base = part.split("@")[0]
                if base == "snapshot-pending.data":
                    assert "@" in part, (
                        f"Callback {cb_key} outputs snapshot-pending without "
                        f"allow_duplicate (part: {part!r})")

    def test_apply_globals_does_not_output_snapshot_pending(self):
        """Guard: apply_globals must NOT output snapshot-pending. If a future
        editor moves the release into apply_globals, it clears the gate before
        apply_tab_{active} runs — breaking the single-redraw invariant."""
        import _app_ctx
        app = _app_ctx.app
        for cb_key in app.callback_map:
            parts = cb_key.split("...")
            clean = [p.split("@")[0] for p in parts]
            # Identify apply_globals by output = main-tabs.active_tab AND palette-store.data
            if ("main-tabs.active_tab" in clean and "palette-store.data" in clean):
                assert "snapshot-pending.data" not in clean, (
                    "apply_globals must NOT output snapshot-pending "
                    "(would break single-redraw invariant)")

    def test_safety_timer_at_least_3000ms(self):
        """Clientside safety timer must wait at least 3000 ms before
        unconditionally clearing the gate. Shorter values regress to
        pre-gate behavior on cold-cache citadel/supercharge."""
        import _app_ctx
        # Walk clientside callbacks — look for the one with snapshot-pending
        # as both Input and Output.
        found = False
        for src in _app_ctx.app._callback_list:
            cb_str = str(src)
            if "snapshot-pending" in cb_str and "setTimeout" in cb_str:
                import re
                m = re.search(r"setTimeout\s*\([^,]+,\s*(\d+)", cb_str)
                assert m, f"Could not parse setTimeout duration: {cb_str!r}"
                duration = int(m.group(1))
                assert duration >= 3000, (
                    f"Safety timer must be >= 3000ms; got {duration}")
                found = True
                break
        assert found, "Safety-timer clientside callback not registered"
```

- [ ] **Step 1.2: Append chart-callback-gate tests to `test_callbacks.py`**

Insert near the bottom of `btc_web/test_callbacks.py`, before the final class if any:

```python


@pytest.mark.skipif(_q3 is None, reason="app.py import failed")
class TestSnapshotPendingChartGate:
    """The 9 figure-writing chart callbacks must early-return no_update
    when snapshot-pending is True. Spec 2026-04-24-single-redraw-per-snapshot."""

    _CHART_OUTPUTS_EXPECTED = {
        "bubble-graph.figure":    "update_bubble",
        "heatmap-graph.figure":   "update_heatmap",
        "dca-graph.figure":       "update_dca",
        "retire-graph.figure":    "update_retire",
        "supercharge-graph.figure": "update_supercharge",
        "bub-cagr-graph.figure":  "update_bub_cagr",
        "bub-resid-graph.figure": "update_bub_resid",
        "citadel-graph.figure":   "update_citadel",
        "lev-graph.figure":       "update_leverage",
    }

    def test_nine_chart_callbacks_have_snapshot_pending_state(self):
        """Each figure-writing chart callback must declare
        State('snapshot-pending','data')."""
        import _app_ctx
        app = _app_ctx.app
        missing = []
        for cb_key in app.callback_map:
            parts = cb_key.split("...")
            outputs = [p.split("@")[0] for p in parts]
            # Look for the output we care about as first Output
            if outputs[0] in self._CHART_OUTPUTS_EXPECTED:
                # Check that the callback's State list includes snapshot-pending
                entry = app.callback_map[cb_key]
                state_spec = entry.get("state") or entry.get("raw_inputs") or []
                state_ids = [getattr(s, "component_id", None) if not isinstance(s, dict)
                             else s.get("id") for s in state_spec]
                # Also check 'inputs' — Dash stores State items there too
                inputs_spec = entry.get("inputs") or []
                for i in inputs_spec:
                    cid = getattr(i, "component_id", None) if not isinstance(i, dict) \
                        else i.get("id")
                    state_ids.append(cid)
                if "snapshot-pending" not in state_ids:
                    missing.append(outputs[0])
        assert not missing, (
            f"Chart callbacks missing State('snapshot-pending','data'): {missing}")
```

- [ ] **Step 1.3: Run the new tests — all should fail (RED)**

```bash
cd /scratch/code/bitcoinprojections
btc_venv/bin/python3 -m pytest btc_web/test_snapshot.py::TestSnapshotPendingGate btc_web/test_callbacks.py::TestSnapshotPendingChartGate -v --ignore-glob='*_e2e.py' 2>&1 | tail -30
```

Expected: all 9 tests FAIL (snapshot-pending Store missing, callbacks missing outputs, etc.).

- [ ] **Step 1.4: Commit RED**

```bash
git add btc_web/test_snapshot.py btc_web/test_callbacks.py
git commit -m "test(snapshot): RED tests for snapshot-pending gate"
```

---

## Task 2: Add Store, arm gate, implement safety timer (GREEN pt.1)

**Files:**
- Modify: `btc_web/layout/__init__.py`
- Modify: `btc_web/callbacks/snapshot_cb.py`

- [ ] **Step 2.1: Add `snapshot-pending` Store to layout**

Find the block of snapshot-related Stores in `btc_web/layout/__init__.py`. The drop-all-tabs refactor already removed the 7 relay stores; `snapshot-state-store` and `loaded-hash-store` remain. Add one line after `loaded-hash-store`:

```bash
grep -n 'snapshot-state-store\|loaded-hash-store' /scratch/code/bitcoinprojections/btc_web/layout/__init__.py | head -4
```

Use the Edit tool to insert after the `loaded-hash-store` Store:

```python
    dcc.Store(id="snapshot-pending", storage_type="memory", data=False),
```

- [ ] **Step 2.2: Update `restore_from_url` — add snapshot-pending output + switch to initial_duplicate**

In `btc_web/callbacks/snapshot_cb.py`, locate the `@callback` for `restore_from_url` (line ~38):

Current decorator:
```python
@callback(
    Output("snapshot-state-store", "data"),
    Output("loaded-hash-store",    "data"),
    Input("url", "hash"),
    prevent_initial_call=False,
)
def restore_from_url(hash_str):
```

Replace with:

```python
@callback(
    Output("snapshot-state-store", "data"),
    Output("loaded-hash-store",    "data"),
    Output("snapshot-pending",     "data", allow_duplicate=True),
    Input("url", "hash"),
    prevent_initial_call='initial_duplicate',
)
def restore_from_url(hash_str):
```

And update the function body's return statements. Current body:

```python
    if not hash_str:
        return no_update, no_update
    h = hash_str.lstrip("#")
    state, prefix, encoded = _decode_snapshot_by_prefix(h)
    if not state:
        logger.warning("Snapshot decode failed for hash: %s…", hash_str[:20])
        return no_update, no_update
    # ... coerce logic ...
    logger.info("Snapshot restored: %d controls, lots=%s", ...)
    return state, hash_str
```

Change all `return no_update, no_update` to `return no_update, no_update, no_update`. Change the final successful-decode return to `return state, hash_str, True`. Read the current function body first:

```bash
sed -n '38,65p' /scratch/code/bitcoinprojections/btc_web/callbacks/snapshot_cb.py
```

Apply edits precisely.

- [ ] **Step 2.3: Update `_make_apply_tab_callback` to write snapshot-pending**

Locate `_make_apply_tab_callback` in `snapshot_cb.py` (~line 130–160). Current:

```python
def _make_apply_tab_callback(tab_id, first_render_id, controls):
    @callback(
        *[Output(cid, prop, allow_duplicate=True) for cid, prop in controls],
        Input(first_render_id, "data"),
        State("snapshot-state-store", "data"),
        prevent_initial_call=True,
    )
    def _apply(_trigger, state, _ctrls=controls):
        if not state:
            return [no_update] * len(_ctrls)
        return [state.get(f"{cid}:{prop}", no_update) for cid, prop in _ctrls]

    _apply.__name__ = f"apply_tab_{tab_id}"
    _apply.__qualname__ = _apply.__name__
    globals()[_apply.__name__] = _apply
    return _apply
```

Replace with:

```python
def _make_apply_tab_callback(tab_id, first_render_id, controls):
    """Factory: register one apply_tab_{tab} callback.

    Fires on {tab}-first-render change. Reads snapshot-state-store as State.
    Writes that tab's controls, and releases snapshot-pending=False in the
    SAME output batch. When state is None, all outputs are no_update
    (including the gate — see spec invariant)."""
    @callback(
        *[Output(cid, prop, allow_duplicate=True) for cid, prop in controls],
        Output("snapshot-pending", "data", allow_duplicate=True),
        Input(first_render_id, "data"),
        State("snapshot-state-store", "data"),
        prevent_initial_call=True,
    )
    def _apply(_trigger, state, _ctrls=controls):
        if not state:
            # Including gate: no_update. Never False here — non-restore
            # first-render bumps must not clear the gate.
            return [no_update] * (len(_ctrls) + 1)
        values = [state.get(f"{cid}:{prop}", no_update) for cid, prop in _ctrls]
        values.append(False)  # release gate
        return values

    _apply.__name__ = f"apply_tab_{tab_id}"
    _apply.__qualname__ = _apply.__name__
    globals()[_apply.__name__] = _apply
    return _apply
```

- [ ] **Step 2.4: Add safety-timer clientside callback at end of `snapshot_cb.py`**

Append to the end of `btc_web/callbacks/snapshot_cb.py`:

```python


# ── Safety-timer: clear snapshot-pending after 3s unconditionally ──────────
# Protects paths where no apply_tab_{tab} fires to release the gate — e.g.
# share links landing on non-chart tabs (stack, model-info, faq), or
# unexpected broken paths. 3000 ms chosen to exceed cold-cache compute time
# on citadel/supercharge (see spec 2026-04-24-single-redraw-per-snapshot).
_app_ctx.app.clientside_callback(
    """
    function(pending) {
        if (window._snapshotPendingTimer) {
            clearTimeout(window._snapshotPendingTimer);
            window._snapshotPendingTimer = null;
        }
        if (pending === true) {
            window._snapshotPendingTimer = setTimeout(function () {
                if (window.dash_clientside && window.dash_clientside.set_props) {
                    window.dash_clientside.set_props(
                        'snapshot-pending', { data: false });
                }
                window._snapshotPendingTimer = null;
            }, 3000);
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("snapshot-pending", "data", allow_duplicate=True),
    Input("snapshot-pending", "data"),
    prevent_initial_call=True,
)
```

- [ ] **Step 2.5: Commit GREEN pt.1**

```bash
cd /scratch/code/bitcoinprojections
git add btc_web/layout/__init__.py btc_web/callbacks/snapshot_cb.py
git commit -m "feat(snapshot): add snapshot-pending gate + 3s safety timer (pt.1)"
```

---

## Task 3: Gate chart callbacks (GREEN pt.2)

**Files:**
- Modify: `btc_web/callbacks/charts/__init__.py`
- Modify: `btc_web/callbacks/citadel_cb.py`
- Modify: `btc_web/callbacks/leverage_cb.py`

- [ ] **Step 3.1: Gate `update_bubble`**

In `btc_web/callbacks/charts/__init__.py`, find the `@callback` for `update_bubble` (line ~75). The Input/State list ends with:

```python
    State("scan-active-rows",  "data"),
    State("scan-q",            "value"),
    State("bub-sigma-mode",    "value"),
    prevent_initial_call=True,
)
def update_bubble(_first_render, sel_qs, adv_qs, toggles, bubble_toggles, ...):
```

Add a `State("snapshot-pending","data")` after `State("bub-sigma-mode", "value")` and add the parameter at the end of the function signature:

```python
    State("scan-active-rows",  "data"),
    State("scan-q",            "value"),
    State("bub-sigma-mode",    "value"),
    State("snapshot-pending",  "data"),
    prevent_initial_call=True,
)
def update_bubble(_first_render, sel_qs, adv_qs, toggles, bubble_toggles,
                  xscale, yscale, xrange, yrange,
                  n_future, ptsize, ptalpha, stack, show_stack, use_lots, legend_pos, model_show,
                  lppl_n_freqs, lppl_weighted, lppl_no_13,
                  hyb_a_nlog, hyb_a_ncal, hyb_a_log1d, hyb_a_log2d,
                  hyb_a_cal1d, hyb_a_cal2d,
                  hyb_b_enabled, hyb_b_nlog, hyb_b_ncal, hyb_b_log1d, hyb_b_log2d,
                  hyb_b_cal1d, hyb_b_cal2d,
                  ep_a_nlog, ep_a_ncal, ep_a_log1d, ep_a_log2d,
                  ep_a_cal1d, ep_a_cal2d,
                  ep_b_enabled, ep_b_nlog, ep_b_ncal, ep_b_log1d, ep_b_log2d,
                  ep_b_cal1d, ep_b_cal2d,
                  decomp_model, decomp_components, decomp_mode,
                  effective_lots, palette_key, user_model, redraw_tick,
                  hybppl_commit, eppl_commit, bm_commit,
                  cta_active, qs_mode, scan_rows, scan_q, sigma_mode,
                  snapshot_pending):
    # Gate: skip render while a share-link restore is in progress.
    # Must be the very first statement, BEFORE any existing PreventUpdate
    # or hydration guards, so restore always settles deterministically.
    # See spec 2026-04-24-single-redraw-per-snapshot-design.md.
    if snapshot_pending:
        return no_update
```

(The existing function body — including its current `PreventUpdate` blocks starting around line 170 — follows unchanged immediately after the new gate block. `no_update` must be imported; verify near the top of the file.)

Verify `no_update` is imported in `charts/__init__.py`:

```bash
grep -n "from dash import" /scratch/code/bitcoinprojections/btc_web/callbacks/charts/__init__.py | head -3
```

If `no_update` is not imported, add it.

- [ ] **Step 3.2: Gate `update_heatmap` (same file)**

Find the `@callback` for `update_heatmap` (line ~688). Add `State("snapshot-pending","data")` to the end of the State/Input list, add `snapshot_pending` as final param, add the same gate:

```python
    # ... existing State/Input declarations end ...
    State("snapshot-pending", "data"),
    prevent_initial_call=True,
)
def update_heatmap(_first_render, hm_model, entry_yr, entry_q, exit_range, exit_qs, mode,
                   # ... rest of existing params ...
                   snapshot_pending):
    if snapshot_pending:
        return no_update
    # ... existing body unchanged ...
```

- [ ] **Step 3.3: Gate `update_dca` (same file)**

Same pattern as 3.2 for `update_dca` (line ~908).

- [ ] **Step 3.4: Gate `update_retire` (same file)**

Same pattern for `update_retire` (line ~1079).

- [ ] **Step 3.5: Gate `update_supercharge` (same file)**

Same pattern for `update_supercharge` (line ~1253).

- [ ] **Step 3.6: Gate `update_bub_cagr` (same file)**

Find `@callback` at line ~382. After `State("bub-qs-mode", "value")` add `State("snapshot-pending","data")`:

```python
    State("bub-qs-mode", "value"),
    State("snapshot-pending", "data"),
    prevent_initial_call=True,
)
def update_bub_cagr(view_mode, _first_render, sel_qs, adv_qs, xrange,
                    toggles, xscale, yscale, model_show, legend_pos,
                    fwd_yrs, palette_key, qs_mode, snapshot_pending):
    if snapshot_pending:
        return no_update
    from utils import _get_cagr_fig
    # ... existing body unchanged ...
```

- [ ] **Step 3.7: Gate `update_bub_resid` (same file)**

Find `@callback` at line ~432. After `State("user-model-store", "data")`:

```python
    State("user-model-store", "data"),
    State("snapshot-pending", "data"),
    prevent_initial_call=True,
)
def update_bub_resid(view_mode, xrange, toggles, xscale, model_show,
                    bubble_toggles, n_future, bm_commit_trigger,
                    legend_pos, palette_key, decomp_model, decomp_components,
                    lppl_n_freqs, lppl_weighted, lppl_no_13,
                    user_model, snapshot_pending):
    if snapshot_pending:
        return no_update
    # ... existing body unchanged ...
```

Confirm actual signature before editing with `sed -n '432,460p' charts/__init__.py`.

- [ ] **Step 3.8: Gate `update_citadel`**

Open `btc_web/callbacks/citadel_cb.py`. The `@callback` for `update_citadel` ends around line 233. Locate the last State before `prevent_initial_call=True` (currently `State("cp-mc-rendered-key", "data")`), add `State("snapshot-pending","data")` after it, add `snapshot_pending` as final param in the function signature, and add the gate as the very first statement:

```python
    State("cp-mc-rendered-key",  "data"),
    State("snapshot-pending",    "data"),
    prevent_initial_call=True,
    background=True,
    # ... rest of decorator unchanged ...
)
def update_citadel(
    _first_render, run_clicks, _pay_trigger, _mc_loaded,
    # ... existing params unchanged ...
    snapshot_pending,
):
    # Gate. See spec 2026-04-24-single-redraw-per-snapshot-design.md.
    if snapshot_pending:
        # update_citadel has multiple Outputs; return tuple of no_update.
        from dash import no_update
        n_outputs = 5  # cp-graph.figure + cp-mc-results + cp-mc-status + cp-mc-rendered-key + mc-save-modal
        # The exact count must match the current decorator Output list.
        return (no_update,) * n_outputs
    # ... existing body unchanged ...
```

VERIFY the actual Output count by counting `Output(` lines in the decorator before editing. Adjust `n_outputs`.

```bash
sed -n '100,200p' /scratch/code/bitcoinprojections/btc_web/callbacks/citadel_cb.py | grep -c "^    Output"
```

Use the returned integer as `n_outputs`.

- [ ] **Step 3.9: Gate `update_leverage`**

Open `btc_web/callbacks/leverage_cb.py` (line ~95). Decorator has 3 Outputs + many Inputs. Insert a `State("snapshot-pending","data")` at the end of the decorator's Input/State list (before `prevent_initial_call=False`), add `snapshot_pending` to function signature, add gate:

```python
    Input("lev-toggles", "value"),
    State("snapshot-pending", "data"),
    prevent_initial_call=False,
)
def update_leverage(first_render, date_val, price_val, model, q,
                    rb_val, rl_val, H_val, c_val, toggles,
                    snapshot_pending):
    if snapshot_pending:
        return no_update, no_update, no_update
    if not first_render:
        raise PreventUpdate
    # ... existing body unchanged ...
```

Verify `no_update` is imported in leverage_cb.py.

- [ ] **Step 3.10: Run full non-E2E test suite**

```bash
cd /scratch/code/bitcoinprojections
btc_venv/bin/python3 -m pytest btc_web/ --ignore-glob='*_e2e.py' 2>&1 | tail -20
```

Expected: all 9 gate tests now pass; pre-existing static_pages hex-literal failure unchanged.

- [ ] **Step 3.11: Commit GREEN pt.2**

```bash
git add btc_web/callbacks/charts/__init__.py btc_web/callbacks/citadel_cb.py btc_web/callbacks/leverage_cb.py
git commit -m "feat(charts): gate 9 chart callbacks on snapshot-pending"
```

---

## Task 4: dash-callback-reviewer gate on the diff

- [ ] **Step 4.1: Dispatch dash-callback-reviewer**

Use the Agent tool with `subagent_type: "dash-callback-reviewer"`. Prompt:

```
Review the diff from 4f5632f..HEAD in /scratch/code/bitcoinprojections. This implements the spec at
docs/superpowers/specs/2026-04-24-single-redraw-per-snapshot-design.md (which you've already reviewed).

Verify:
- 9 chart callbacks all received State("snapshot-pending","data") + gate early-return
- restore_from_url correctly switched to prevent_initial_call='initial_duplicate'
- _make_apply_tab_callback None-state branch returns no_update for gate (never False)
- Safety-timer clientside callback has a declared Output with allow_duplicate=True
- No chart callback was accidentally double-gated or missed
- No cache-key alignment regression (snapshot_pending must not enter params dict anywhere)
- Total test count should match: 9 new gate tests all pass

Flag BLOCKING issues. Under 400 words.
```

- [ ] **Step 4.2: Fix any BLOCKING findings**

Iterate until reviewer returns "ship it" with zero BLOCKING findings. Commit fixes as separate commits with descriptive messages.

---

## Task 5: Deploy + verify on prod

- [ ] **Step 5.1: Dev smoke**

```bash
cd /scratch/code/bitcoinprojections
lsof -ti :8050 2>/dev/null | xargs -r kill -9
DEV=1 nohup bash run_web.sh > /tmp/quantoshi_dev.log 2>&1 &
sleep 12
curl -s -o /dev/null -w "/1: %{http_code}\n" http://localhost:8050/1
curl -s -o /dev/null -w "/2: %{http_code}\n" http://localhost:8050/2
curl -s -o /dev/null -w "/6: %{http_code}\n" http://localhost:8050/6
tail -10 /tmp/quantoshi_dev.log
```

Expected: 200s, no Python tracebacks.

- [ ] **Step 5.2: Push + deploy prod**

```bash
cd /scratch/code/bitcoinprojections
git push origin master
ssh root@89.167.70.45 'cd /opt/quantoshi && git pull && redis-cli FLUSHDB && systemctl restart quantoshi'
```

- [ ] **Step 5.3: Prod smoke**

```bash
sleep 6
curl -s -o /dev/null -w "prod /: %{http_code}\n" https://quantoshi.xyz/
curl -s -o /dev/null -w "prod /1: %{http_code}\n" https://quantoshi.xyz/1
echo "--- snapshot-pending in layout (expect 1+):"
curl -s https://quantoshi.xyz/_dash-layout | grep -c "snapshot-pending"
ssh root@89.167.70.45 'journalctl -u quantoshi --since "60 seconds ago" --no-pager | grep -iE "error|traceback|critical|nonexistent" | head -20'
```

Expected: HTTP 200, `snapshot-pending` in layout, zero error output from log scan.

- [ ] **Step 5.4: Re-profile a real share link on prod**

Generate a fresh share link (similar to the one used in the pre-work profile):

```bash
cd /scratch/code/bitcoinprojections/btc_web
PYTHONPATH=".:.." ../btc_venv/bin/python3 - <<'PY' 2>&1 | tail -3
import os; os.environ['DEV']='1'
import app
from snapshot import _encode_snapshot
from callbacks.routing import _TAB_CONTROLS

state = {
    "main-tabs:active_tab": "bubble",
    "bub-xscale:value": "Lin",
    "bub-yscale:value": "Lin",
    "bub-qs:value": [0.01, 0.05, 0.5, 0.95, 0.99],
    "bub-toggles:value": ["shade", "show_data", "annotate"],
    "bub-xrange:value": [2014, 2055],
    "bub-model-show:value": ["bub", "pl"],
    "bub-legend-pos:value": "top-left",
    "palette-store:data": "cb_brian",
}
encoded = _encode_snapshot(state, tab_filter=_TAB_CONTROLS["bubble"])
print("q3:" + encoded)
PY
```

- [ ] **Step 5.5: Record post-deploy timing (Playwright)**

Navigate to `https://quantoshi.xyz/1#<hash-from-5.4>` with Playwright, wait 8s, pull `performance.getEntriesByType('resource')` filtered to `_dash-update-component`, count payloads ≥ 40 KB (the Plotly figure bundles). Expected: **ONE** 40-KB+ chart payload during the restore window (down from 3). If more than one, the gate didn't release in time or the ordering assumption broke — rollback via `git revert` and open a follow-up debugging spec.

- [ ] **Step 5.6: Tail prod logs 2 minutes post-deploy**

```bash
ssh root@89.167.70.45 'journalctl -u quantoshi --since "3 minutes ago" --no-pager | tail -80'
```

Expected: normal startup output; no repeated errors.

- [ ] **Step 5.7: Update memory**

Edit `~/.claude/projects/-scratch-code-bitcoinprojections/memory/parked_single_redraw_brainstorm.md`: replace its body with "RESOLVED — single-redraw gate shipped 2026-04-24 at commit `<final-commit-sha>`. See spec 2026-04-24-single-redraw-per-snapshot-design.md and plan 2026-04-24-single-redraw-per-snapshot-plan.md for the shipped design."

- [ ] **Step 5.8: Mark task 22 completed**

---

## Self-Review Checklist

**Spec coverage:**
- `dcc.Store(id="snapshot-pending")` added → Task 2.1 ✓
- `restore_from_url` arms gate + `initial_duplicate` → Task 2.2 ✓
- `_make_apply_tab_callback` releases gate (None→no_update, state→False) → Task 2.3 ✓
- Safety timer 3000ms clientside → Task 2.4 ✓
- Gate check in 9 chart callbacks as first statement → Tasks 3.1–3.9 ✓
- All 8 spec-required tests → Task 1.1–1.2 ✓
- Negative test on apply_globals → Task 1.1 ✓ (`test_apply_globals_does_not_output_snapshot_pending`)
- Cache-key isolation: gate read + return BEFORE params build → body of each Task 3.x ✓

**Placeholder scan:** no TBDs, "handle appropriately", or vague steps. Each code step shows exact content.

**Type consistency:** `snapshot_pending` (Python param name, snake_case) consistently matches `snapshot-pending` (Dash id, kebab-case) across all tasks. Output tuple counts match each callback's actual Output declaration.

**Missing:** None identified.

---

## Execution choice

User delegated autonomous execution. Use **superpowers:executing-plans** inline with dash-callback-reviewer hard gate at Task 4.
