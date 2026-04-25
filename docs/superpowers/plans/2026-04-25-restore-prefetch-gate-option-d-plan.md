# Restore Prefetch-Gate Refactor (Option D) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax for tracking. Three prior failed attempts at this problem are documented; each task below has explicit verification gates.

**Goal:** Eliminate the 7-second iPhone share-link restore lag by gating non-active-tab prefetch on the active chart's actual figure commit (server-authoritative signal), not on snapshot-decode time. Plus fix a race condition between `apply_tab_bubble` and `update_bubble` that caused a prior attempt (Phase A) to silently fail.

**Architecture:** Option D from `memory/restore_callback_architecture.md`. Surgical: ~20 lines across 4 files. No new architecture; one new Store, one Input change in splash.py, six chart callbacks gain one Output, one PreventUpdate guard becomes restore-aware.

**Memory:** `memory/restore_callback_architecture.md` — full root-cause analysis from architect deep-dive on 2026-04-25.

**Hard prerequisite:** Working tree on `master`, last commit `d87deb0` (revert of Phase A). Dev box accessible. iPhone or desktop browser available for testing.

**Hard gate:** dev verification with `?trace=1` BEFORE prod deploy. The bug is invisible without trace logs.

---

## File Map

| File | Status | Change |
|---|---|---|
| `btc_web/layout/__init__.py` | MOD | Add `dcc.Store(id="active-chart-committed", ...)` next to `prefetch-ready` |
| `btc_web/callbacks/splash.py` | MOD | Change Input on the prefetch-release callback from `loaded-hash-store` to `active-chart-committed` |
| `btc_web/callbacks/charts/__init__.py` | MOD | 5 chart callbacks: `update_bubble`, `update_heatmap`, `update_dca`, `update_retire`, `update_supercharge`. Each: add Output + State + return tuple element. ALSO: fix `effective-lots` race-condition guard in `update_bubble`. |
| `btc_web/callbacks/citadel_cb.py` | MOD | `update_citadel`: same Output + State + return additions. 5 return paths to update. |
| `btc_web/test_callbacks.py` | MOD | Update tuple-length assertions for the 6 chart callbacks |

**NOT touched:** `snapshot_cb.py`, `routing.py`, `_clientside.py`, `mc_helpers.py`, `mc_upload.py`, `scanner.py`, `custom_time.py`. The fix is contained to the prefetch gate + chart callback signatures.

**Out of scope (deferred):**
- Citadel MC simulation triggered by `cp-mc-loaded` cascade (1108 ms wasted on bubble-tab shares). Workers are pooled so this doesn't directly block the active tab; CPU waste only.
- CTA Patch update path. Already gated correctly.
- `update_scanner` lacks a snapshot-pending gate; ~30ms × 3 fires during restore. Adding it is a 3-line follow-up; not in this plan.
- Stateless URL restore via `set_props` bulk write (Option C). Future PoC.

---

## Task 1: Add `active-chart-committed` Store

**Files:** `btc_web/layout/__init__.py`

- [ ] **Step 1.1: Locate the `prefetch-ready` Store**

```bash
grep -n "prefetch-ready" btc_web/layout/__init__.py
```

Expected: line ~291.

- [ ] **Step 1.2: Add the new Store directly below**

```python
dcc.Store(id="prefetch-ready", storage_type="memory", data=0),
# Set by chart callbacks (update_bubble / update_heatmap / etc.) to the
# loaded-hash-store value when the callback returns a real figure (not
# no_update / not PreventUpdate). Single canonical "active chart has
# committed its restored figure" signal. The prefetch-ready release
# (splash.py) watches this so non-active-tab work doesn't fire until
# the active chart is genuinely interactive.
dcc.Store(id="active-chart-committed", storage_type="memory", data=None),
```

- [ ] **Step 1.3: Syntax-check**

```bash
btc_venv/bin/python3 -c "
import sys; sys.path.insert(0, 'btc_web')
import app
print([s for s in app.app.layout()._traverse() if hasattr(s, 'id') and 'committed' in (s.id or '')])
" 2>&1 | tail -3
```

Or simpler:
```bash
btc_venv/bin/python3 -c "
import sys; sys.path.insert(0, 'btc_web')
import app
print('OK')
"
```

Expected: `OK`. No traceback.

- [ ] **Step 1.4: Commit**

```bash
git add btc_web/layout/__init__.py
git commit -m "feat(layout): add active-chart-committed Store

Used by the prefetch-ready release path (Option D) — chart callbacks
write the loaded_hash on real-figure return; splash listens here.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Wire `update_bubble` to write `active-chart-committed` + fix `effective-lots` race

**Files:** `btc_web/callbacks/charts/__init__.py`

This task contains the most subtle change of the plan: the `effective-lots` race fix.

- [ ] **Step 2.1: Add the new Output to `update_bubble`**

Locate the `@callback` decorator at line ~75:
```python
@callback(
    Output("bubble-graph", "figure"),
    Input("bubble-first-render", "data"),
    ...
```

Add the new Output and State:
```python
@callback(
    Output("bubble-graph", "figure"),
    Output("active-chart-committed", "data", allow_duplicate=True),
    Input("bubble-first-render", "data"),
    ...
    State("snapshot-pending",  "data"),
    State("loaded-hash-store", "data"),  # NEW: read at dispatch time
    prevent_initial_call=True,
)
```

The new `State("loaded-hash-store", "data")` must be added at the END of the State block (just before `prevent_initial_call=True`), then added to the function signature similarly.

- [ ] **Step 2.2: Update function signature**

In `def update_bubble(...)`, add `loaded_hash=None` as a new parameter. Place it right before `snapshot_pending=False`:
```python
def update_bubble(_first_render, sel_qs, adv_qs, ...,
                  ...,
                  qs_mode=None, scan_active=None, scan_q_val=None,
                  sigma_mode=None,
                  loaded_hash=None,
                  snapshot_pending=False):
```

- [ ] **Step 2.3: Update gated path return**

```python
if snapshot_pending:
    print(f"[trace] bubble-fig SKIPPED (gate) "
          f"{(_time.perf_counter() - _t0) * 1000:.1f}ms", flush=True)
    return dash.no_update, dash.no_update  # was: return dash.no_update
```

- [ ] **Step 2.4: Pre-verify apply_tab_bubble release semantics**

Reviewer flagged that the rest of the fix depends on `apply_tab_bubble` releasing `snapshot-pending=False` AND the resulting widget Input changes triggering a second `update_bubble` fire. Verify before continuing:

```bash
grep -n "snapshot-pending\|snap-applied" btc_web/callbacks/snapshot_cb.py | head -10
```

Confirm three things in the output:
1. `Output("snapshot-pending", "data", allow_duplicate=True)` appears in `_make_apply_tab_callback`'s decorator (the apply_tab_* factory).
2. The `_apply` function appends `False` to its return values (releases `snapshot-pending`).
3. The factory also writes `Output(applied_id, "data", allow_duplicate=True)` (i.e. `bubble-snap-applied`) — this is the new Input we'll add to `update_bubble`.

If any of these is absent, **STOP** — Option D's premise is broken and the fix needs redesign.

- [ ] **Step 2.5: Fix the `effective-lots` PreventUpdate race (CRITICAL — two-part fix)**

The reviewer caught that `not snapshot_pending` alone is insufficient: by the time `update_bubble` fires post-restore, `apply_tab_bubble` has already released `snapshot-pending=False`, so the `not snapshot_pending` clause is True AND if the user's snapshot has lots disabled (default), the guard fires and raises PreventUpdate before the figure can be built. Same Phase A failure mode.

The correct fix has two parts:

**(a) Add `Input("bubble-snap-applied", "data")` to `update_bubble`.**

This guarantees a SECOND fire of `update_bubble` after `apply_tab_bubble` commits all bub-* widgets, with `ctx.triggered_id == "bubble-snap-applied"` — which does NOT match the `effective-lots` guard, so the guard cannot fire on this canonical post-apply fire. The figure is built. `active-chart-committed` is written.

In `update_bubble`'s `@callback` decorator, find the existing Input list and append:
```python
Input("bubble-snap-applied", "data"),
```
Place it after the existing State/Input block but before `prevent_initial_call=True`. Best location: just before the existing `State("snapshot-pending", "data")` line.

The `update_bubble` function signature must accept this new positional Input. Add a corresponding parameter, e.g.:
```python
def update_bubble(_first_render, sel_qs, adv_qs, ...,
                  ...,
                  _bub_snap_applied=None,  # NEW
                  loaded_hash=None,
                  snapshot_pending=False):
```
Place `_bub_snap_applied=None` before `loaded_hash=None` to match the decorator's Input/State order. **Verify by counting decorator entries vs function parameters** — Dash 4 raises `TypeError` on mismatch.

**(b) Make the existing `effective-lots` guard restore-aware (defense-in-depth).**

Locate the guard at line ~184:
```python
if _trg == "effective-lots" and not (use_lots and "yes" in (use_lots or [])):
    raise PreventUpdate
```

Change to:
```python
# Steady-state guard: when user toggles lots off and effective-lots
# cascades, suppress the redundant redraw. NOT applicable during
# restore: snapshot_pending may already be False (apply_tab_bubble
# released it) but the post-apply fire we want to keep is triggered
# from bubble-snap-applied with _trg="bubble-snap-applied", so this
# branch only matches when an effective-lots change is genuinely the
# trigger — which post-restore means the user manually changed lots.
if _trg == "effective-lots" and not snapshot_pending and not (use_lots and "yes" in (use_lots or [])):
    raise PreventUpdate
```

Belt-and-suspenders: even if `bubble-snap-applied` Input fires correctly (part a), the `not snapshot_pending` clause prevents this guard from raising on any first-fire that happens to have `_trg="effective-lots"` while pending is still True.

- [ ] **Step 2.6: Update real-figure return**

Locate the final `return fig` at line ~285:
```python
print(f"[trace] bubble-fig BUILT "
      f"{(_time.perf_counter() - _t0) * 1000:.1f}ms", flush=True)
return fig
```

Change to (using `is not None` per reviewer Q2 — safer than truthy-check):
```python
print(f"[trace] bubble-fig BUILT "
      f"{(_time.perf_counter() - _t0) * 1000:.1f}ms", flush=True)
# active-chart-committed = loaded_hash when this is a restore context
# (loaded-hash-store is non-None). Steady-state interactions write
# no_update so the prefetch gate stays in its prior state.
return fig, (loaded_hash if loaded_hash is not None else dash.no_update)
```

- [ ] **Step 2.7: Verify all return paths in update_bubble**

```bash
btc_venv/bin/python3 -c "
import re
with open('btc_web/callbacks/charts/__init__.py') as f:
    src = f.read()
m = re.search(r'def update_bubble\(.*?\n(.*?)(?=\n@callback|\n# ── )', src, re.DOTALL)
body = m.group(1)
print('returns/raises:')
for r in re.findall(r'^\s+(return [^\n]+|raise [^\n]+)', body, re.MULTILINE):
    print(f'  {r[:120]}')
"
```

Expected: 4 return statements + 3 raise statements:
```
  return dash.no_update, dash.no_update      # gated path
  raise PreventUpdate                         # user-model-store hydration
  raise PreventUpdate                         # effective-lots steady-state guard
  raise PreventUpdate                         # cta-active routing
  return fig, (loaded_hash if loaded_hash else dash.no_update)  # real fig
```

If you see `return dash.no_update` (single-arg) or `return fig` (single-arg), the change is incomplete — fix before continuing.

- [ ] **Step 2.8: Run unit tests**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_callbacks.py::TestUpdateBubbleCallback -q 2>&1 | tail -10
```

Expected: tests fail with `assert isinstance(fig, go.Figure)` because fig is now a tuple. **Do not panic** — Task 6 fixes the tests. Just verify the failure is in `isinstance(fig, go.Figure)`, not in unrelated lines.

- [ ] **Step 2.9: Commit**

```bash
git add btc_web/callbacks/charts/__init__.py
git commit -m "feat(charts): update_bubble writes active-chart-committed + fix effective-lots race

Three coupled changes:

1. Add Output('active-chart-committed', 'data') + State('loaded-hash-store').
   On real-figure return: write loaded_hash if loaded_hash is not None
   else no_update. On gated/PreventUpdate paths: don't write (Dash handles).

2. Add Input('bubble-snap-applied', 'data') as a guaranteed re-trigger
   after apply_tab_bubble commits all bub-* widgets. The fire from this
   Input has ctx.triggered_id='bubble-snap-applied', which CANNOT match
   the effective-lots PreventUpdate guard, so the canonical post-apply
   figure build is guaranteed to run.

3. Make the effective-lots PreventUpdate guard restore-aware: add
   'not snapshot_pending' to the condition. Defense-in-depth — even if
   a transient _trg='effective-lots' fire happens with snapshot_pending
   still True, the guard skips. Steady-state behavior (user toggles
   lots off) is unchanged.

Phase A (commit ecaa07f, reverted) failed because update_bubble's
post-restore fire had ctx.triggered_id='effective-lots' and the guard
raised PreventUpdate before the figure could be built. Adding the
bubble-snap-applied Input + restore-aware guard removes both failure
modes.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Wire 4 more chart callbacks (heatmap, dca, retire, supercharge)

**Files:** `btc_web/callbacks/charts/__init__.py`

These are mechanically identical to bubble, but each has 8 existing Outputs (becomes 9), and a single gated path + single real-fig path. No PreventUpdate guards to modify.

- [ ] **Step 3.1: `update_heatmap` — add Output + State + tuple element**

Decorator (around line 641):
```python
@callback(
    Output("heatmap-graph",  "figure"),
    ...
    Output("mc-save-tab", "data", allow_duplicate=True),
    Output("active-chart-committed", "data", allow_duplicate=True),  # NEW
    Input("heatmap-first-render", "data"),
    ...
    State("eppl-cfg-a-cal2d", "value"),
    State("snapshot-pending", "data"),
    State("loaded-hash-store", "data"),  # NEW
    prevent_initial_call=True,
)
```

Function signature: add `loaded_hash=None` parameter before `snapshot_pending=False`.

Gated path (around line 722):
```python
if snapshot_pending:
    return (dash.no_update,) * 9  # was * 8
```

Real-fig return (around line 840):
```python
return (fig, store_val, status, mc_panel_style, indicator_style,
        rendered_key,
        show_modal, "hm" if show_modal else dash.no_update,
        loaded_hash if loaded_hash else dash.no_update)  # NEW: 9th element
```

- [ ] **Step 3.2: `update_dca` — same pattern**

Decorator (around line 848): add Output + 2 States.
Signature: add `loaded_hash=None`.
Gated path (around line 951): `(dash.no_update,) * 9`.
Real-fig return (around line 1030): append `loaded_hash if loaded_hash else dash.no_update` as 9th element.

- [ ] **Step 3.3: `update_retire` — same pattern**

Decorator (around line 1035): add Output + 2 States.
Signature: add `loaded_hash=None`.
Gated path (around line 1124): `(dash.no_update,) * 9`.
Real-fig return (around line 1195): append loaded_hash element.

- [ ] **Step 3.4: `update_supercharge` — same pattern**

Decorator (around line 1204): add Output + 2 States.
Signature: add `loaded_hash=None`.
Gated path (around line 1308): `(dash.no_update,) * 9`.
Real-fig return (around line 1388): append loaded_hash element.

- [ ] **Step 3.5: Verify all four**

```bash
btc_venv/bin/python3 -c "
import re
with open('btc_web/callbacks/charts/__init__.py') as f:
    src = f.read()
for fn in ['update_heatmap', 'update_dca', 'update_retire', 'update_supercharge']:
    m = re.search(rf'def {fn}\(.*?\n(.*?)(?=\n@callback|\n# ── |\Z)', src, re.DOTALL)
    if not m:
        print(f'{fn}: NOT FOUND')
        continue
    body = m.group(1)
    rets = re.findall(r'^\s+(return [^\n]+)', body, re.MULTILINE)
    raises = re.findall(r'^\s+(raise [^\n]+)', body, re.MULTILINE)
    print(f'{fn}: {len(rets)} returns, {len(raises)} raises')
    for r in rets:
        print(f'  {r[:140]}')
"
```

For each: expect 2 return statements (gated + real). Verify gated says `* 9` and real ends with `loaded_hash if loaded_hash else dash.no_update`.

- [ ] **Step 3.6: Commit**

```bash
git add btc_web/callbacks/charts/__init__.py
git commit -m "feat(charts): heatmap/dca/retire/supercharge write active-chart-committed

Same pattern as update_bubble. 8 outputs -> 9 outputs each. Real-figure
return appends loaded_hash; gated path appends no_update.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Wire `update_citadel`

**Files:** `btc_web/callbacks/citadel_cb.py`

Citadel has 5 return paths (gated + 2 cached-default + pending-MC + real). All five must be updated.

- [ ] **Step 4.1: Add Output + State**

Decorator (around line 113):
```python
@callback(
    Output("citadel-graph", "figure"),
    ...
    Output("cp-tax-annual-data", "data", allow_duplicate=True),
    Output("active-chart-committed", "data", allow_duplicate=True),  # NEW
    Input("citadel-first-render", "data"),
    ...
    State("mc-auth", "data"),
    State("snapshot-pending",    "data"),
    State("loaded-hash-store",   "data"),  # NEW
)
```

Signature: add `loaded_hash=None` before `snapshot_pending=False`.

- [ ] **Step 4.2: Update all 5 return paths**

Find them with:
```bash
grep -n "return (\|return (dash.no_update,)" btc_web/callbacks/citadel_cb.py
```

Expected: 5 return statements at approximately lines 293, 314, 332, 500, 511.

- Line 293 (gated): `return (dash.no_update,) * 9` → `* 10`
- Line 314 (cached default with fig): append `loaded_hash if loaded_hash else dash.no_update`
- Line 332 (live-fallback default with fig): append `loaded_hash if loaded_hash else dash.no_update`
- Line 500 (Celery-pending with fig): append `loaded_hash if loaded_hash else dash.no_update`
- Line 511 (real-fig): append `loaded_hash if loaded_hash else dash.no_update`

- [ ] **Step 4.3: Verify**

```bash
btc_venv/bin/python3 -c "
import re
with open('btc_web/callbacks/citadel_cb.py') as f:
    src = f.read()
m = re.search(r'def update_citadel\(.*?\n(.*?)(?=\n@callback|\n@_app_ctx)', src, re.DOTALL)
body = m.group(1)
rets = re.findall(r'^\s+(return [^\n]+)', body, re.MULTILINE)
print(f'{len(rets)} returns')
for r in rets:
    print(f'  {r[:140]}')
"
```

Expected: 5 returns. Gated has `* 10`. The other 4 end with the loaded_hash pattern.

- [ ] **Step 4.4: Commit**

```bash
git add btc_web/callbacks/citadel_cb.py
git commit -m "feat(citadel): update_citadel writes active-chart-committed (5 paths)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Rewire prefetch gate in splash.py

**Files:** `btc_web/callbacks/splash.py`

This is the keystone change — moves the prefetch release point.

- [ ] **Step 5.1: Locate the existing release callback**

```bash
grep -n "loaded-hash-store" btc_web/callbacks/splash.py
```

Expected: a callback around line 405 with `Input("loaded-hash-store", "data")` writing `Output("prefetch-ready", "data", allow_duplicate=True)`.

- [ ] **Step 5.2: Change the Input**

Current:
```python
_app_ctx.app.clientside_callback(
    """
    function(loaded_hash) {
        if (!loaded_hash) return window.dash_clientside.no_update;
        return 1;
    }
    """,
    Output("prefetch-ready", "data", allow_duplicate=True),
    Input("loaded-hash-store", "data"),
    prevent_initial_call=True,
)
```

Change to:
```python
# Release prefetch gate when the active tab's chart callback has
# committed its restored figure. active-chart-committed is written
# server-side by update_bubble / update_heatmap / update_dca /
# update_retire / update_supercharge / update_citadel on their real-
# figure return path. By gating prefetch on this signal (instead of
# loaded-hash-store, which fires at decode time), non-active-tab work
# is held back until the active chart is genuinely interactive — the
# point of the "active tab first, opportunistic prefetch after"
# principle. See memory/restore_callback_architecture.md.
_app_ctx.app.clientside_callback(
    """
    function(committed) {
        if (!committed) return window.dash_clientside.no_update;
        return 1;
    }
    """,
    Output("prefetch-ready", "data", allow_duplicate=True),
    Input("active-chart-committed", "data"),
    prevent_initial_call=True,
)
```

- [ ] **Step 5.3: Also update the explanatory comment in splash-init**

Around line 208, find:
```javascript
/* Share-hash loads: leave prefetch-ready unset so non-active tabs
   don't start lazy-loading until the snapshot finishes applying to
   the active tab. A separate callback on loaded-hash-store flips
   prefetch-ready once restore is complete. */
```

Change "loaded-hash-store" → "active-chart-committed". Don't change behavior — the early-return for share-hash is correct.

- [ ] **Step 5.4: Commit**

```bash
git add btc_web/callbacks/splash.py
git commit -m "feat(splash): release prefetch gate on active-chart-committed, not loaded-hash-store

The old Input fired at decode time (T+100ms), well before the active
chart actually rendered. This unleashed the non-active-tab prefetch
storm during the active chart's compute window, blocking iPhone
Safari's event loop and preventing plotly_afterplot from firing
before the 7s hard fallback.

active-chart-committed is written by chart callbacks only on their
real-figure return path — server-authoritative signal that the chart
ran to completion. Now non-active-tab work is held back until the
active chart is genuinely interactive.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Update tests for new tuple lengths

**Files:** `btc_web/test_callbacks.py`

- [ ] **Step 6.1: Run tests, observe failures**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_callbacks.py -q 2>&1 | tail -15
```

Expected failures:
- `TestUpdateBubbleCallback::test_*` — `isinstance(fig, go.Figure)` fails because `fig` is now a 2-tuple.
- `TestUpdateHeatmapCallback::test_*` — `assert len(result) == 8` fails (now 9).
- Same for DCA, Retire, Supercharge.

- [ ] **Step 6.2: Pre-verify match counts (catches regex-replace footguns)**

```bash
grep -c "assert isinstance(fig, go.Figure)" btc_web/test_callbacks.py
grep -c "assert len(result) == 8" btc_web/test_callbacks.py
```

Expected: 3 for bubble (the `update_bubble` test has 3 separate isinstance checks across 3 sub-tests), 4 for tuple-length (heatmap, dca×2 maybe, retire, sc — confirm by listing the matches).

```bash
grep -n "assert len(result) == 8" btc_web/test_callbacks.py
```

If counts differ from expected, **STOP** — the regex replace in Step 6.3 will incorrectly match unintended lines. Inspect manually first.

- [ ] **Step 6.3: Update assertions**

```bash
btc_venv/bin/python3 <<'EOF'
with open('btc_web/test_callbacks.py') as f:
    src = f.read()

# Ensure dash is imported (the new isinstance assertions reference dash.no_update).
if 'import dash\n' not in src and 'import dash ' not in src:
    # Add after the first import line so future merges don't conflict at top.
    src = src.replace('import pytest', 'import pytest\nimport dash', 1)

# Bubble: fig is now (fig, committed_hash). Update isinstance checks.
src = src.replace(
    'assert isinstance(fig, go.Figure)',
    'assert isinstance(fig[0], go.Figure)\n        # second element is active-chart-committed (None / hash str / no_update)\n        assert fig[1] is None or isinstance(fig[1], str) or fig[1] is dash.no_update',
)

# Heatmap/DCA/Retire/SC: 8-tuple → 9-tuple
src = src.replace('assert len(result) == 8', 'assert len(result) == 9')

with open('btc_web/test_callbacks.py', 'w') as f:
    f.write(src)
print('OK')
EOF
```

- [ ] **Step 6.4: Run tests**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_callbacks.py -q 2>&1 | tail -5
```

Expected: all pass.

- [ ] **Step 6.5: Run full suite**

```bash
btc_venv/bin/python3 -m pytest btc_web/ --ignore-glob='*_e2e.py' -q 2>&1 | tail -5
```

Expected: 1 pre-existing failure (`test_no_hex_literals_outside_colors_module`); zero new failures.

- [ ] **Step 6.6: Commit**

```bash
git add btc_web/test_callbacks.py
git commit -m "test(callbacks): update tuple-length assertions for active-chart-committed Output

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Dev verification with `?trace=1`

**Hard gate before prod deploy.** Three prior attempts failed only-in-prod; do not skip dev verification.

- [ ] **Step 7.1: Start dev server**

```bash
/usr/bin/lsof -ti :8050 2>/dev/null | /usr/bin/xargs -r /usr/bin/kill -9
sleep 1
DEV=1 nohup bash run_web.sh > /tmp/quantoshi_dev.log 2>&1 &
sleep 14
for path in / /1 /6; do
  printf "%s " "$path"
  /usr/bin/curl -s -o /dev/null -w "%{http_code}\n" "http://localhost:8050$path"
done
```

Expected: all 200.

- [ ] **Step 7.2: Generate a fresh share link**

In the user's browser (or via Playwright):
1. Visit `http://localhost:8050/1`
2. Change a few controls (toggle x-scale to linear, change future bubbles to 5)
3. Click 📸 Share → Generate link
4. Copy the URL

- [ ] **Step 7.3: Restore the link with `?trace=1`**

In a fresh tab (clean slate):
- Open the URL with `?trace=1` injected before the `#` — e.g. `http://localhost:8050/1?trace=1#q4:...`
- Wait for restore to complete

- [ ] **Step 7.4: Inspect dev log**

```bash
grep -E "\[trace\]|\[trace-cb\]" /tmp/quantoshi_dev.log | tail -50
```

**Pass criteria** (all must hold):

1. **`[trace] bubble-fig BUILT XXXms` line appears** — proves update_bubble ran to completion. Phase A failed because this didn't fire.
2. **`bubble-fig BUILT` precedes any `lazy-children` populating writes** for non-active tabs (heatmap/dca/retire/sc/citadel). The prefetch storm should be after, not during.
3. **No `bubble-fig SKIPPED (gate)` line as the LAST update_bubble fire** — that would mean the gate stuck and the figure never built.
4. **Modal closes via `plotly_afterplot`, not the 7s fallback.** Check the trace-client `modal-closed` line: `delta_ms` should be small (<500ms after `plotly_afterplot`), NOT ~7000ms.

If any criterion fails, **DO NOT DEPLOY**. Investigate. The architect's analysis (`memory/restore_callback_architecture.md`) is the reference.

- [ ] **Step 7.5: Visual smoke**

In the browser, verify:
- The chart shows the RESTORED state (not the default x-scale=log; should be linear if you changed it).
- All 7 tabs paint correctly when clicked through after restore (background warming working).
- Modal opens and closes cleanly; no flicker.

- [ ] **Step 7.6: Stop dev server**

```bash
/usr/bin/lsof -ti :8050 2>/dev/null | /usr/bin/xargs -r /usr/bin/kill -9
```

---

## Task 8: dash-callback-reviewer hard gate before push

- [ ] **Step 8.1: Dispatch reviewer**

Use the `dash-callback-reviewer` agent. Prompt template:

```
Review Option D restore-prefetch-gate refactor in /scratch/code/bitcoinprojections, commits <first-commit>..HEAD (6 commits across layout, splash, charts, citadel_cb, tests).

Reference: memory/restore_callback_architecture.md — root-cause analysis from prior architect deep-dive. Three prior fix attempts failed; this is attempt #4.

Files modified: <list>

Verify (BLOCKING only, under 400 words):

1. Tuple-length correctness for every chart callback. Reference Task 2 / 3 / 4 in plan. Each chart callback has N+1 Outputs (was N). Every return site (gated, real, cached-default for citadel) returns a tuple of exactly the new length. PreventUpdate paths are unchanged (Dash skips writes).

2. effective-lots PreventUpdate guard at update_bubble: now reads `if _trg == "effective-lots" and not snapshot_pending and not (use_lots ...)`. Confirm this prevents the race condition that broke Phase A: during restore, snapshot_pending=True when effective-lots cascade fires, so guard doesn't raise; update_bubble falls through to figure builder; even with stale bub-use-lots=null, the figure is built (lots overlay is empty); apply_tab_bubble's writes then trigger update_bubble again with all values committed. The second fire's figure is canonical.

3. active-chart-committed write semantics: written ONLY when callback returns a real figure AND loaded_hash is non-None. Steady-state interactions (no restore in flight) have loaded_hash=None, so callbacks write no_update for active-chart-committed; prefetch gate is not affected. Confirm.

4. splash.py prefetch release: changed Input from loaded-hash-store to active-chart-committed. The early return `if (!committed) return no_update` correctly handles None default (initial state) and intermediate writes. Confirm no false fires.

5. CLAUDE.md footguns:
   - No new prevent_initial_call=False + allow_duplicate=True combos.
   - No falsy-zero float(x or default) patterns.
   - All new States read at dispatch time; no order-sensitive reads.

6. Test coverage: tuple-length assertions updated to N+1.

7. Dev verification was performed (Task 7) and passed all 4 trace criteria. If not, BLOCK deploy.

Flag BLOCKING issues only. Three prior failed attempts; user is justifiably cautious.
```

- [ ] **Step 8.2: Fix any BLOCKING findings; re-dispatch.**

Proceed only when zero BLOCKING.

---

## Task 9: Push + deploy + prod smoke

- [ ] **Step 9.1: Push**

```bash
git push origin master
```

- [ ] **Step 9.2: Deploy**

```bash
/usr/bin/ssh root@89.167.70.45 'cd /opt/quantoshi && git pull && systemctl restart quantoshi'
```

- [ ] **Step 9.3: Prod smoke**

```bash
sleep 8
for path in / /1 /6 /9; do
  printf "%s " "$path"
  /usr/bin/curl -s -o /dev/null -w "%{http_code}\n" "https://quantoshi.xyz$path"
done
/usr/bin/ssh root@89.167.70.45 'journalctl -u quantoshi --since "60 seconds ago" --no-pager | grep -iE "error|traceback|critical" | head -5'
```

Expected: all 200; zero error/traceback lines.

- [ ] **Step 9.4: Prod end-to-end test with `?trace=1`**

User opens a fresh share link with `?trace=1` from iPhone or desktop. Server-side journal:

```bash
/usr/bin/ssh root@89.167.70.45 'journalctl -u quantoshi --since "5 minutes ago" --no-pager | grep -E "\[trace"' | tail -50
```

**Final acceptance criteria** (all must hold):

1. **Modal closes via `plotly_afterplot`** (not 7s fallback). `[trace-client]` shows `modal-closed` with `delta_ms` < 500.
2. **Total restore time < 4 seconds** on iPhone (was 7+ seconds). Server compute alone is ~108 ms; the savings come from eliminating the prefetch storm during the active-tab paint window.
3. **No non-active-tab `*-graph.figure` writes appear in `[trace-cb]` lines BEFORE `bubble-fig BUILT`**. The order of work is now: active-chart compute → active-chart paint → modal close → prefetch storm.
4. **Background prefetch still runs** — the storm appears AFTER `bubble-fig BUILT`. Background warming is preserved.
5. **No regressions on tab-switch performance.** Click through Heatmap, DCA, Retire, Supercharger, Citadel after restore; each should paint without long delay (cached or fresh compute).

If criterion 1 or 3 fails: **roll back immediately** with `git revert` and re-engage architect.

If criterion 2 fails by a small margin (e.g., 4-5 seconds): not a roll-back trigger but document the residual delay; further optimization may target Citadel MC trigger (see "Out of scope" in File Map).

---

## Self-Review

**Spec coverage (vs `memory/restore_callback_architecture.md`):**

| Spec section | Implemented in |
|---|---|
| Option D Change 1 (move prefetch gate) | Tasks 1, 5 |
| Option D Change 2 (effective-lots race fix) | Task 2 (Step 2.4) |
| Tuple-length correctness | Tasks 2, 3, 4, 6 |
| Dev verification before deploy | Task 7 (HARD GATE) |
| dash-callback-reviewer | Task 8 (HARD GATE) |
| Risk register from architect's Section 5 | Task 7.4 acceptance criteria |

**Placeholder scan:**

- No "TBD" / "implement later" / "fill in details".
- Every task shows exact code, command, or expected output.
- Verification queries shown for each task.
- Reviewer prompt is fully written.

**Type consistency:**

- `active-chart-committed` Store: `data=None` default; written as `loaded_hash` (string) on commit OR `dash.no_update` on non-restore fires. Splash listener handles None / falsy correctly.
- `loaded_hash` parameter to chart callbacks: type hint not added (matching existing convention); reads as `Optional[str]`.
- All 6 chart callbacks have identical Output signature for `active-chart-committed`.

**Three prior failed attempts: do they apply here?**

| Failure mode | Phase 1 | Phase 2 | Phase A | Option D | 
|---|---|---|---|---|
| Conflated triggers in `_first_render` | Yes (broken) | — | — | No (untouched) |
| `plotly_afterplot` false positives | — | Yes (broken) | — | No (uses server commit) |
| `effective-lots` race | — | — | Yes (broken) | **Fixed in Step 2.4** |
| Prefetch released too early | — | — | — | Fixed in Task 5 |

**Migration risk:**

- Reverting Option D is `git revert <last 6 commits>` cleanly. No data migration. No share-link encoding changes. No store schema changes that persist.

---

## Execution choice

User has authorized blue-sky and noted "use as many agents as necessary." After Task 8 review, this is autonomous through prod deploy with the explicit `[trace-client]` acceptance criteria as the gate.

Three prior failed attempts noted in plan preamble; the architect's `memory/restore_callback_architecture.md` is the canonical reference for why this attempt is structurally different. The single most important added safeguard vs Phase A is **Step 2.4** (effective-lots race fix).
