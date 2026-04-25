# Restore Figure In Decode — Implementation Plan (v2, post-4-failures)

> Self-imposed discipline rules: (1) no prod deploy until I've personally watched `[trace] bubble-fig BUILT` in dev journal in response to a real `?trace=1` URL load; (2) reviewer hedge-words mean STOP; (3) synthetic Python tests don't substitute for Dash dispatch verification; (4) three-tier verification before deploy; (5) no "ship N commits hoping it works" — ship and verify ONE commit at a time.

**Goal:** Deliver the active tab's restored figure in the SAME server response that decodes the share-link hash. The user sees the chart in ~150–300 ms instead of 7+ seconds.

**Scope:** Bubble (Tab 1) ONLY for this iteration. Other tabs (heatmap, dca, retire, supercharge, citadel) fall back to existing behavior. Bubble is the most common share target; citadel's MC simulations make it expensive to compute synchronously in `restore_from_url`.

**Memory:** `memory/restore_callback_architecture.md` — canonical reference.

**Working tree:** `/scratch/code/bitcoinprojections` on `master`, last commit `cb8a417` (revert of Option D / commit chain). Trace instrumentation deployed.

---

## Architecture

### Today's flow (concrete, verified from production traces 2026-04-25)

```
URL hash arrives
   ↓
restore_from_url server callback
   ↓ writes: snapshot-state-store, loaded-hash-store, snapshot-pending=True
   ↓
apply_globals + apply_tab_bubble (parallel) + clientside cascades
   ↓ apply_tab_bubble writes 36 bub-* widgets + snapshot-pending=False + bubble-snap-applied
   ↓ snapshot-lots → effective-lots (clientside cascade)
   ↓
update_bubble fires (multiple Inputs changed, batched into ONE dispatch)
   _trg = effective-lots (Dash picks this; cascade fired before apply_tab_bubble)
   snapshot_pending = False (apply_tab_bubble already released)
   use_lots = null/empty (default)
   PreventUpdate guard at line 184 raises ← FIRST FIRE LOST
   ↓
Time passes... worker pool busy with prefetch storm + leftover Citadel MC
   ↓
custom_time_callback (CTA) eventually fires from bub-* changes
   snapshot_pending = False; cta_active = [] (default)
   Returns (no_update, "Standard view restored.", tick+1)  ← bumps bub-redraw-tick
   ↓
update_bubble fires AGAIN (bub-redraw-tick is an Input; line 138)
   _trg = "bub-redraw-tick"
   No PreventUpdate guard matches
   Falls through, builds figure  ← SECOND FIRE BUILDS THE CHART
   ↓
Total elapsed: 7-21 seconds (worker contention dependent)
```

### New flow

```
URL hash arrives
   ↓
restore_from_url server callback (ENHANCED)
   ↓ Decodes hash → state dict
   ↓ Reads main-tabs.active_tab from state
   ↓ IF active_tab == "bubble":
   ↓     Calls _build_bubble_figure_from_state(state) → fig
   ↓     Returns: state, hash, pending=True, bubble-graph.figure=fig, active-chart-committed=hash
   ↓ ELSE: same as today (state, hash, pending=True, no_update for fig outputs)
   ↓
Browser receives one HTTP response
   ↓ Plotly applies bubble-graph.figure → renders → plotly_afterplot fires
   ↓ Modal close listener fires (existing) → modal closes
   ↓ Splash listener on active-chart-committed fires → prefetch-ready=1
   ↓
User sees the restored chart in ~150-300 ms total
   ↓
Background:
   apply_globals + apply_tab_bubble fire (write widget values)
   CTA eventually bumps bub-redraw-tick
   update_bubble fires from tick bump
   New early-return guard: if active_chart_committed == loaded_hash: return no_update
   ← redundant rebuild skipped
```

### What this preserves

- All existing callback paths still fire as today (apply_globals, apply_tab_bubble, CTA, etc.)
- The chart eventually re-fires from CTA tick bump — but the new early-return guard makes it a no-op
- Pre-existing PreventUpdate guards in update_bubble unchanged
- Non-bubble tab share-links unchanged (existing behavior, ~7s)
- Non-active-tab prefetch warming unchanged (still fires after active chart commits)

### What this changes

- One new helper function: `_build_bubble_figure_from_state(state_dict) → fig`
- One new Output on `restore_from_url`: `bubble-graph.figure` (allow_duplicate=True)
- One new Output on `restore_from_url`: `active-chart-committed` (Store, allow_duplicate=True)
- One new State on `update_bubble`: `loaded-hash-store` + `active-chart-committed`
- One new early-return in `update_bubble` (after gate check)
- One existing Output annotation: `update_bubble`'s `bubble-graph.figure` gains `allow_duplicate=True`
- One Input change in `splash.py`: `loaded-hash-store` → `active-chart-committed`
- One new Store in layout: `active-chart-committed`

---

## File map

| File | Status | Change |
|---|---|---|
| `btc_web/restore_builder.py` | NEW | `_build_bubble_figure_from_state(state)` helper |
| `btc_web/layout/__init__.py` | MOD | Add `dcc.Store(id="active-chart-committed", data=None)` |
| `btc_web/callbacks/snapshot_cb.py` | MOD | `restore_from_url` gains 2 Outputs, calls `_build_bubble_figure_from_state` for bubble shares |
| `btc_web/callbacks/charts/__init__.py` | MOD | `update_bubble`: `Output("bubble-graph", "figure")` → `allow_duplicate=True`; State `loaded-hash-store` + `active-chart-committed`; early-return guard |
| `btc_web/callbacks/splash.py` | MOD | Prefetch-ready listener: Input `loaded-hash-store` → `active-chart-committed` |
| `btc_web/test_callbacks.py` | MOD | (Optional) tests for the helper |
| `btc_web/test_restore_builder.py` | NEW | Pure-Python tests for `_build_bubble_figure_from_state` |

**NOT touched:** routing.py, mc_helpers, mc_upload, scanner.py, custom_time.py, citadel_cb.py, mc-related Stores, prefetch Intervals.

---

## Tasks

Each task verified in isolation. ONE commit per task. Reviewer dispatched after each. Dev verification before any deploy.

### Task 1: Write `_build_bubble_figure_from_state` helper (PURE PYTHON)

**Files:** `btc_web/restore_builder.py` (new)

This is the riskiest task because it duplicates `update_bubble`'s param-construction logic. The helper takes a snapshot state dict (keyed by `"{cid}:{prop}"`) and returns a Plotly figure built via `_get_bubble_fig`.

**Strategy:** mirror `update_bubble`'s exact param construction (charts/__init__.py:239-285), reading from the state dict instead of widget Inputs. No parallel callback dispatch — pure function call. The helper:

1. Extracts each widget value from `state.get("bub-X:value")` with the same defaults `update_bubble` uses.
2. Resolves master keys (LPPL/HybPPL/EPPL) via the same helper functions update_bubble uses.
3. Builds the params dict identically to update_bubble's `_get_bubble_fig` call.
4. Calls `_get_bubble_fig(params)` and returns the figure.

**Verification:**
- Unit test: feed a minimal default state dict, assert the helper returns a `go.Figure`.
- Unit test: feed a state dict with non-default values (e.g. xscale=lin), assert returned figure reflects them.
- Unit test: helper produces identical output to a reconstructed `update_bubble` invocation with the same effective inputs.

**Step 1.1: Write the helper.**
**Step 1.2: Write `btc_web/test_restore_builder.py` with 3 unit tests.**
**Step 1.3: Run tests.** Expected: all pass.
**Step 1.4: Reviewer dispatch.** Specifically request: "verify `_build_bubble_figure_from_state` produces identical params to `update_bubble`'s figure-building call. Identify any State value the helper reads incorrectly."
**Step 1.5: Address reviewer blockers if any.**
**Step 1.6: Commit.**

### Task 2: Wire `_build_bubble_figure_from_state` into `restore_from_url`

**Files:** `btc_web/callbacks/snapshot_cb.py`, `btc_web/layout/__init__.py`

**Step 2.1:** Add `dcc.Store(id="active-chart-committed", storage_type="memory", data=None)` to layout next to `prefetch-ready`.

**Step 2.2:** Modify `restore_from_url`:
- Add `Output("bubble-graph", "figure", allow_duplicate=True)`
- Add `Output("active-chart-committed", "data", allow_duplicate=True)`
- After decoding state, check `active_tab = state.get("main-tabs:active_tab")`. If "bubble":
  - Call `_build_bubble_figure_from_state(state)` → fig
  - Return: state, hash, pending=True, fig, hash_str (as committed value)
- Else: return state, hash, pending=True, dash.no_update, dash.no_update (preserves existing behavior for non-bubble tabs)

**Step 2.3:** Add `allow_duplicate=True` to `update_bubble`'s `Output("bubble-graph", "figure")` (line 76). This is required because `restore_from_url` is now also a writer.

**Step 2.4:** Add State `loaded-hash-store` and `active-chart-committed` to `update_bubble`. Add corresponding parameters to function signature.

**Step 2.5:** Add early-return guard to `update_bubble`:
```python
# After the snapshot_pending gate:
if active_chart_committed and active_chart_committed == loaded_hash:
    return dash.no_update  # restore_from_url already built it
```

**Step 2.6:** Verify tuple correctness in `restore_from_url`:
- 3 paths: empty hash, decode failure, success
- Each must return correct N-tuple matching the new Output count (5)
- Decode-failure path: no_update for all 5

**Step 2.7:** Run tests: `pytest test_callbacks.py test_snapshot.py test_restore_builder.py -q`

**Step 2.8:** Reviewer dispatch. Specifically: "Verify allow_duplicate=True is correctly added to update_bubble's bubble-graph.figure Output. Confirm restore_from_url's tuple length matches Output count in every return path. Verify the early-return guard's condition correctly handles None loaded_hash and steady-state (no restore) cases."

**Step 2.9:** Dev verification (FIRST DEV TEST):
- Start `DEV=1 bash run_web.sh`
- Use Playwright (Firefox) to navigate to a share link with `?trace=1`
- Watch journal for `[trace] bubble-fig BUILT` line
- Watch journal for `[trace-cb]` lines from `restore_from_url`
- Confirm chart appears restored in browser
- IF NOT: investigate, fix, re-verify. Do not proceed.

**Step 2.10:** Commit only after dev verification passes.

### Task 3: Rewire splash.py prefetch-ready listener

**Files:** `btc_web/callbacks/splash.py`

**Step 3.1:** Change the splash callback at line 405:
```python
Input("active-chart-committed", "data"),  # was: Input("loaded-hash-store", "data")
```

**Step 3.2:** Update accompanying comment on line 208 (mentions "loaded-hash-store callback") to reference `active-chart-committed`.

**Step 3.3:** Reviewer dispatch.

**Step 3.4:** Dev verification with Playwright + ?trace=1. Watch for:
- `[trace-cb] *-lazy.children` writes for non-bubble tabs come AFTER `bubble-fig BUILT`
- prefetch-ready stays at 0 until active-chart-committed fires
- Chart visibly updates in browser

**Step 3.5:** Commit only after dev verification passes.

### Task 4: Final integration test

**Step 4.1:** Run full suite: `pytest btc_web/ --ignore-glob='*_e2e.py' -q`. Expected: 1 pre-existing failure, zero new.

**Step 4.2:** Restart dev server. Multiple consecutive share-link restores (test for cross-session worker contention).

**Step 4.3:** Reviewer dispatch on the cumulative diff.

**Step 4.4:** If reviewer green: deploy to prod.

### Task 5: Prod deploy + verification

**Step 5.1:** `git push origin master && ssh root@... 'cd /opt/quantoshi && git pull && systemctl restart quantoshi'`

**Step 5.2:** Smoke test all paths.

**Step 5.3:** Use Playwright to load a share link on prod with `?trace=1`. Watch journal for BUILT and chart paint timing.

**Step 5.4:** If anything fails: revert immediately, document failure mode, restart from analysis.

---

## Risk register

| Risk | Detection | Mitigation |
|---|---|---|
| `_build_bubble_figure_from_state` produces different params than `update_bubble` | Visible figure mismatch in Playwright test | Refactor update_bubble to call the SAME helper. (Future task.) For now: careful manual mirror + unit tests. |
| Adding `allow_duplicate=True` to update_bubble's Output breaks Dash callback graph | Dev server fails to start; "duplicate output" error | Test in dev; revert if startup fails |
| State-dict missing keys cause helper to crash | Helper raises exception during restore | Helper uses `state.get(key, default)` for every lookup; tests cover defaults |
| Race: restore_from_url returns figure but Plotly applies it AFTER apply_tab_bubble's widget writes trigger update_bubble re-fire | Visible flicker / wrong figure briefly | Early-return guard in update_bubble blocks the re-fire when active_chart_committed matches loaded_hash |
| iPhone Safari paint timing: figure response → React apply → Plotly render → afterplot may still race against prefetch storm | Modal closes via fallback even though figure was sent | Verify in prod with Playwright; if happens, push prefetch storm later (gate on plotly_afterplot too) |
| `loaded-hash-store` → `active-chart-committed` Input swap: legacy listeners on loaded-hash-store still need it | manage_snapshot uses loaded-hash-store; check if anything else does | Grep for all loaded-hash-store Inputs; document each |
| Non-bubble tab shares (citadel /6, etc.) still take 7s | Acceptable for this iteration | Documented as out-of-scope; future work |

---

## Verification protocol

For each task that modifies callbacks:

1. **Unit tests pass.**
2. **Dev server starts cleanly:** `DEV=1 bash run_web.sh`, `lsof :8050` shows process, curl `/1` returns 200.
3. **Playwright smoke:**
   - Generate a share link via Playwright in a clean browser context (Tab 1, change a few controls)
   - Open the link with `?trace=1` in another browser context
   - Wait for plotly_afterplot OR fallback timeout
   - Capture browser console output
   - Capture journal `[trace]`/`[trace-cb]` output
4. **Pass criteria:**
   - Server journal shows `[trace] bubble-fig BUILT` line OR `restore_from_url` returned the figure (visible in Playwright as the chart drawn)
   - Modal closes within 2 seconds (NOT via 7s fallback)
   - Browser console has no errors
   - Restored controls match the snapshot (verify via Playwright assertions on widget values)

If any criterion fails: investigate, fix, re-verify. Don't proceed.

---

## Dispatch handoff

User has authorized autonomous execution. I am the implementer AND the verifier. After each task: dispatch a reviewer agent. Address blockers. Then verify in dev. Only then commit. No prod deploy until full integration test passes in dev. If a deploy fails: revert immediately.

The discipline rules at top of this doc are not aspirational; they are a contract.
