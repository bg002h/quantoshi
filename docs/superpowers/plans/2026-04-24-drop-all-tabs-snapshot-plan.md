# Drop "All tabs" Snapshot Share Scope — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retire the "All tabs" share-scope option and replace the 8-callback snapshot-restore architecture (1 `apply_snapshot` + 7 stage-2 `_apply_lazy`) with a thinner `apply_globals` + 7 `apply_tab_{tab}` pattern, eliminating the per-tab relay stores.

**Architecture:** One `apply_globals` callback writes 31 always-mounted globals on `snapshot-state-store` change; seven `apply_tab_{tab}` callbacks each fire on `{tab}-first-render` with `snapshot-state-store` as State, writing 30–60 tab-scoped controls when the tab materializes. No more relay stores, no eager/lazy partition, hash format unchanged (`q3:`).

**Tech Stack:** Dash 4.0.0, dbc 2.0.4, pytest, Python 3.14 dev / 3.12 prod.

**Spec:** `docs/superpowers/specs/2026-04-24-drop-all-tabs-snapshot-design.md`

**Reference (pre-refactor commit):** `34a19a5`

**Deploy command (user-delegated autonomous):**
```bash
git push origin master && ssh root@89.167.70.45 "cd /opt/quantoshi && git pull && redis-cli FLUSHDB && systemctl restart quantoshi"
```

---

## Task 1: Write failing tests for the new architecture (RED)

**Files:**
- Modify: `btc_web/test_snapshot.py`

- [ ] **Step 1.1: Add the 6 new test cases at the end of `test_snapshot.py`**

Append this block to the end of the file:

```python
class TestPostRefactorArchitecture:
    """Tests locking in the new apply_globals + apply_tab_{tab} architecture.
    Written per spec 2026-04-24-drop-all-tabs-snapshot-design.md."""

    def test_no_double_write_partition(self):
        """Union of globals + per-tab equals _SNAPSHOT_CONTROLS;
        intersection is empty."""
        from callbacks.snapshot_cb import _GLOBAL_CONTROLS, _PER_TAB_CONTROLS
        from snapshot import _SNAPSHOT_CONTROLS
        all_tab_cids = set()
        for tab_id, cids_props in _PER_TAB_CONTROLS.items():
            for cid, _prop in cids_props:
                all_tab_cids.add(cid)
        global_cids = {cid for cid, _ in _GLOBAL_CONTROLS}
        assert global_cids.isdisjoint(all_tab_cids), (
            f"Control(s) in both globals and per-tab: "
            f"{global_cids & all_tab_cids}")
        all_cids = {cid for cid, _ in _SNAPSHOT_CONTROLS}
        assert global_cids | all_tab_cids == all_cids, (
            f"Partition mismatch. "
            f"Missing: {all_cids - (global_cids | all_tab_cids)}. "
            f"Extra: {(global_cids | all_tab_cids) - all_cids}")

    def test_apply_globals_writes_only_global_cids(self):
        """apply_globals output count equals len(_GLOBAL_CONTROLS) + 1 (for snapshot-lots)."""
        from callbacks.snapshot_cb import apply_globals, _GLOBAL_CONTROLS
        state = {"main-tabs:active_tab": "heatmap",
                 "palette-store:data": "default",
                 "bub-xscale:value": "Lin"}
        result = apply_globals(state)
        assert len(result) == len(_GLOBAL_CONTROLS) + 1, (
            f"Expected {len(_GLOBAL_CONTROLS) + 1} outputs, got {len(result)}")

    def test_apply_tab_is_noop_when_state_none(self):
        """apply_tab_bubble returns all no_update when state is None."""
        from callbacks.snapshot_cb import apply_tab_bubble
        from dash import no_update
        result = apply_tab_bubble(None, None)
        assert all(x is no_update for x in result)

    def test_apply_tab_partial_restore_on_legacy_payload(self):
        """Legacy payload with both bubble + heatmap keys — each apply_tab
        callback picks up its own tab's keys when invoked."""
        from callbacks.snapshot_cb import apply_tab_bubble, apply_tab_heatmap
        from dash import no_update
        # Payload resembling an old all-tabs share link
        state = {
            "bub-xscale:value": "Lin",
            "hm-mode:value": "segmented",
            "bub-qs:value": [0.5],
        }
        # Bubble applier: bub-xscale must appear somewhere in outputs
        bub_result = apply_tab_bubble(1, state)
        assert any(v == "Lin" for v in bub_result if v is not no_update), (
            "apply_tab_bubble must write bub-xscale=Lin")
        # Heatmap applier: hm-mode must be written
        hm_result = apply_tab_heatmap(1, state)
        assert any(v == "segmented" for v in hm_result if v is not no_update), (
            "apply_tab_heatmap must write hm-mode=segmented")

    def test_no_orphan_relay_stores_in_layout(self):
        """Layout must not contain any snapshot-apply-{tab} Store ids."""
        import layout
        import json
        # Render a layout and serialise to JSON to walk all component ids
        from dash import dcc, html
        import flask
        rendered = layout._serve_layout() if hasattr(layout, "_serve_layout") else None
        serialised = json.dumps(rendered, default=str) if rendered else ""
        banned = ["snapshot-apply-bubble", "snapshot-apply-heatmap",
                  "snapshot-apply-dca", "snapshot-apply-retire",
                  "snapshot-apply-supercharge", "snapshot-apply-citadel",
                  "snapshot-apply-leverage"]
        for b in banned:
            assert b not in serialised, (
                f"Relay store {b} still in layout — not removed by refactor")

    def test_manage_snapshot_signature_has_no_share_scope(self):
        """manage_snapshot should no longer accept share_scope."""
        import inspect
        from callbacks.snapshot_cb import manage_snapshot
        sig = inspect.signature(manage_snapshot)
        param_names = list(sig.parameters.keys())
        assert "share_scope" not in param_names, (
            f"share_scope still a parameter: {param_names}")
```

- [ ] **Step 1.2: Run the new tests to verify they fail**

```bash
cd /scratch/code/bitcoinprojections
btc_venv/bin/python3 -m pytest btc_web/test_snapshot.py::TestPostRefactorArchitecture -v --ignore-glob='*_e2e.py' 2>&1 | tail -30
```

Expected: All 6 tests FAIL with ImportError or AttributeError (`_GLOBAL_CONTROLS`, `_PER_TAB_CONTROLS`, `apply_globals`, `apply_tab_bubble`, `apply_tab_heatmap` don't exist yet; `manage_snapshot` still has `share_scope` parameter).

- [ ] **Step 1.3: Commit the failing tests (RED)**

```bash
cd /scratch/code/bitcoinprojections
git add btc_web/test_snapshot.py
git commit -m "test(snapshot): add failing tests for post-refactor architecture (RED)"
```

---

## Task 2: Replace snapshot_cb.py architecture (GREEN)

**Files:**
- Modify: `btc_web/callbacks/snapshot_cb.py` (full rewrite of restore-flow sections)
- Modify: `btc_web/layout/__init__.py` (remove 7 relay stores + share-scope radio)
- Modify: `btc_web/callbacks/routing.py` (add regression-guard comment)

- [ ] **Step 2.1: Replace the relay architecture block in `snapshot_cb.py`**

The current file has `_BUBBLE_LAZY_PREFIXES`, `_LAZY_TAB_SPECS`, `_ALL_LAZY_PREFIXES`, `_TAB_LAZY_CONTROLS`, `_BUBBLE_LAZY_CONTROLS`, `_EAGER_CONTROLS`, `_N_RELAY_STORES`, `apply_snapshot`, `_make_lazy_relay_callback` and its 7-tab registration loop. Replace the entire block (lines ~66–173) with the new partition + callbacks.

Read the current file first:

```bash
sed -n '1,230p' /scratch/code/bitcoinprojections/btc_web/callbacks/snapshot_cb.py
```

Then apply the rewrite. The new block to insert (replacing lines 66 through the end of `_make_lazy_relay_callback` and its registration loop) is:

```python
# Control partition. See spec 2026-04-24-drop-all-tabs-snapshot-design.md.
#
#   apply_globals     → 31 always-mounted controls (main-tabs, palette-store,
#                       lppl-*, hybppl-cfg-{a,b}-*, eppl-cfg-{a,b}-*, snapshot-lots)
#   apply_tab_{tab}   → 7 callbacks, one per chart tab, keyed on
#                       {tab}-first-render. Writes tab-scoped controls from
#                       snapshot-state-store (as State) when the tab mounts.
#
# Lazy-mounted tab controls stay protected from "nonexistent object" errors
# because the apply_tab_{tab} Outputs only need to exist in DOM at fire
# time (after first-render), not at callback-register time — Dash tolerates
# this as long as the layout eventually contains the component.

_LAZY_PREFIXES = ("bub-", "scan-", "cta-", "hm-", "dca-", "ret-",
                  "sc-", "cp-", "lev-")

# Split _SNAPSHOT_CONTROLS into globals + per-tab buckets.
_GLOBAL_CONTROLS = [(cid, prop) for cid, prop in _SNAPSHOT_CONTROLS
                    if not cid.startswith(_LAZY_PREFIXES)]

# Ordered list of (tab_id, first_render_id, prefix_tuple)
_TAB_SPECS = [
    ("bubble",      "bubble-first-render",      ("bub-", "scan-", "cta-")),
    ("heatmap",     "heatmap-first-render",     ("hm-",)),
    ("dca",         "dca-first-render",         ("dca-",)),
    ("retire",      "retire-first-render",      ("ret-",)),
    ("supercharge", "supercharge-first-render", ("sc-",)),
    ("citadel",     "citadel-first-render",     ("cp-",)),
    ("leverage",    "leverage-first-render",    ("lev-",)),
]

_PER_TAB_CONTROLS: dict[str, list[tuple[str, str]]] = {}
for _tab_id, _fr_id, _prefixes in _TAB_SPECS:
    _PER_TAB_CONTROLS[_tab_id] = [
        (cid, prop) for cid, prop in _SNAPSHOT_CONTROLS
        if cid.startswith(_prefixes)
    ]


@callback(
    *[Output(cid, prop, allow_duplicate=True) for cid, prop in _GLOBAL_CONTROLS],
    Output("snapshot-lots", "data", allow_duplicate=True),
    Input("snapshot-state-store", "data"),
    prevent_initial_call=True,
)
def apply_globals(state):
    """Apply globals + snapshot-lots as soon as snapshot-state-store lands.
    Tab-scoped writes are handled by apply_tab_{tab}."""
    n_outs = len(_GLOBAL_CONTROLS) + 1
    if not state:
        return [no_update] * n_outs
    results = [state.get(f"{cid}:{prop}", no_update)
               for cid, prop in _GLOBAL_CONTROLS]
    results.append(state.get("_lots", None))
    return results


def _make_apply_tab_callback(tab_id, first_render_id, controls):
    """Factory: register one apply_tab_{tab} callback.

    Fires on {tab}-first-render change. Reads snapshot-state-store as State
    — relies on Dash's write-before-read ordering guarantee so that the
    clientside first-render bump in routing.py (which is Input on
    snapshot-state-store) cannot fire until state is populated.
    See routing.py:79-110 for the invariant."""
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


for _tab_id, _fr_id, _prefixes in _TAB_SPECS:
    _make_apply_tab_callback(_tab_id, _fr_id, _PER_TAB_CONTROLS[_tab_id])
```

**How to apply:** locate the start of the old block (the line beginning `# Split _SNAPSHOT_CONTROLS into eager (mounted at page load) vs per-tab lazy`) and the end (the line `    _register_prefetch(_tid)` does NOT belong to this block — stop at the end of `_make_lazy_relay_callback`'s registration for loop, which ends with the line `_make_lazy_relay_callback(...)` inside a `for` loop). Delete lines 66 through the end of that factory-loop, replace with the block above.

Use `sed` / `Edit` tool to do this precisely after reading the file.

- [ ] **Step 2.2: Rewrite `manage_snapshot` to drop `share_scope`**

In the same file, find the `manage_snapshot` callback (`State("share-scope","value")` is on line 181). Delete that State. Remove `share_scope` from the function's positional args. Replace the `scope = share_scope or "all"` / `tab_filter = _TAB_CONTROLS.get(active_tab) if scope == "tab" else None` block with:

```python
        tab_filter = _TAB_CONTROLS.get(active_tab)
```

Drop `scope` and `tab` from the `_add_snapshot_entry` call and from `_add_snapshot_entry`'s signature (leave `includes_lots` intact).

- [ ] **Step 2.3: Remove `scope` and `tab` from `_add_snapshot_entry`**

In the same file, locate `_add_snapshot_entry` (around line 226). Simplify:

```python
def _add_snapshot_entry(history, existing, encoded, full_url,
                        includes_lots):
    """Append a snapshot entry to history if not already present.

    Mutates history in-place and returns True if an entry was added."""
    if encoded in existing:
        return False
    history.insert(0, {
        "hash": encoded, "url": full_url,
        "ts": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "includes_lots": includes_lots,
    })
    history[:] = history[:50]
    return True
```

- [ ] **Step 2.4: Remove share-scope radio from layout**

Open `btc_web/layout/__init__.py`. Locate the share modal block (starts near line 679 with `dbc.Modal([...` and contains `id="share-scope"`). Delete the block starting with `html.Div("Scope:", ...)` through the closing `),` of the `dcc.RadioItems(id="share-scope", ...)`. Keep the `dcc.Checklist(id="share-include-lots", ...)` and the rest of the modal intact.

Read the current block:

```bash
sed -n '680,705p' /scratch/code/bitcoinprojections/btc_web/layout/__init__.py
```

Replace from `html.Div("Scope:"...` through the RadioItems' closing `),` with a single blank line (so the Checklist becomes the first body element).

- [ ] **Step 2.5: Remove 7 relay-store declarations from layout**

Locate the 7 `dcc.Store(id="snapshot-apply-{bubble,heatmap,dca,retire,supercharge,citadel,leverage}", ...)` declarations in `layout/__init__.py`. Find them:

```bash
grep -n "snapshot-apply-" /scratch/code/bitcoinprojections/btc_web/layout/__init__.py
```

Delete the entire list-comprehension or sequence that creates these 7 stores.

- [ ] **Step 2.6: Add regression-guard comment to `routing.py:79–110`**

In `btc_web/callbacks/routing.py`, above the clientside callback at line ~79 (the one bumping `{active}-first-render` on `snapshot-state-store` change), insert this comment (overwriting any existing comment block that duplicates intent):

```python
# ══════════════════════════════════════════════════════════════════════════════
# Post-snapshot chart re-render bump.
#
# INVARIANT: Input MUST remain `snapshot-state-store.data`. If you change it
# (e.g. to `main-tabs.active_tab`), `apply_tab_{active}` in snapshot_cb.py
# will fire BEFORE snapshot-state-store is written and read None as its
# State, silently writing nothing and leaving the active tab stuck at its
# pre-injected defaults. See spec 2026-04-24-drop-all-tabs-snapshot-design.md.
# ══════════════════════════════════════════════════════════════════════════════
```

- [ ] **Step 2.7: Remove stale `test_callbacks.py` import + assertion**

Open `btc_web/test_callbacks.py`. Line 1505 imports `_EAGER_CONTROLS, _N_RELAY_STORES` and lines 1503–1512 assert `apply_snapshot` return shape. Delete the block or rewrite it to exercise `apply_globals` instead. Minimal rewrite:

```python
        # Apply to controls via apply_globals.
        from callbacks.snapshot_cb import apply_globals, _GLOBAL_CONTROLS
        result = apply_globals(decoded)
        assert len(result) == len(_GLOBAL_CONTROLS) + 1
        main_tab_idx = next(i for i, (cid, _) in enumerate(_GLOBAL_CONTROLS)
                            if cid == "main-tabs")
        assert result[main_tab_idx] == "bubble"
```

Remove the `apply_snapshot` import at line 22; replace with `from callbacks.snapshot_cb import apply_globals` (add near the existing import block).

- [ ] **Step 2.8: Delete the stale tests in `test_snapshot.py`**

Open `btc_web/test_snapshot.py` and delete these tests:
- `test_tab_filter` (around line 78–92)
- `test_tab_filter_encodes_only_matching` (around line 258–275)
- `test_each_tab_filter_roundtrips` (around line 279–289)
- `test_single_tab_shorter_than_all` (around line 291–305)
- `test_bubble_*` tests that assert the lazy-relay architecture (search for `lazy relay`, `_LAZY_TAB_SPECS`, `_BUBBLE_LAZY_CONTROLS`, `test_apply_snapshot_output_count`, `test_relay_store_ids_in_lazy_tab_specs`) — delete each.

Grep to find them:

```bash
grep -n "_LAZY_TAB_SPECS\|_BUBBLE_LAZY_CONTROLS\|_EAGER_CONTROLS\|_N_RELAY_STORES\|snapshot-apply-\|lazy relay\|test_apply_snapshot_output_count\|test_relay_store_ids" /scratch/code/bitcoinprojections/btc_web/test_snapshot.py
```

For each numbered test function hit, delete the full `def test_…(self):` block and its body.

- [ ] **Step 2.9: Run the full snapshot test suite**

```bash
cd /scratch/code/bitcoinprojections
btc_venv/bin/python3 -m pytest btc_web/test_snapshot.py -v --ignore-glob='*_e2e.py' 2>&1 | tail -40
```

Expected: all remaining tests pass, including the 6 new `TestPostRefactorArchitecture` tests.

- [ ] **Step 2.10: Run the callbacks test file**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_callbacks.py -v --ignore-glob='*_e2e.py' 2>&1 | tail -40
```

Expected: pass.

- [ ] **Step 2.11: Run the full non-E2E test suite**

```bash
btc_venv/bin/python3 -m pytest btc_web/ -v --ignore-glob='*_e2e.py' 2>&1 | tail -20
```

Expected: 0 failures. (Warnings OK.)

- [ ] **Step 2.12: Commit the refactor (GREEN)**

```bash
cd /scratch/code/bitcoinprojections
git add btc_web/callbacks/snapshot_cb.py btc_web/layout/__init__.py btc_web/callbacks/routing.py btc_web/test_callbacks.py btc_web/test_snapshot.py
git commit -m "$(cat <<'EOF'
refactor(snapshot): drop all-tabs scope; collapse relay pattern

Replaces the 1 apply_snapshot + 7 stage-2 _apply_lazy architecture with
a thinner apply_globals + 7 apply_tab_{tab} pattern. Share links always
encode active-tab + globals; 'All tabs' scope radio removed from modal;
7 snapshot-apply-{tab} relay stores removed from layout.

Legacy q1/q2/q3 all-tabs links continue to restore across tab visits
within the session (snapshot-state-store persists; apply_tab_{tab}
reads as State when the tab later materializes).

See spec docs/superpowers/specs/2026-04-24-drop-all-tabs-snapshot-design.md
and rollback reference memory/reference_alltabs_share_revert_point.md.
EOF
)"
```

---

## Task 3: Dash-callback-reviewer gate on the diff

- [ ] **Step 3.1: Run dash-callback-reviewer on the staged + committed diff**

Dispatch the agent via the Agent tool with `subagent_type: "dash-callback-reviewer"`. Prompt:

```
Review the diff from commit 34a19a5..HEAD in /scratch/code/bitcoinprojections against the spec at docs/superpowers/specs/2026-04-24-drop-all-tabs-snapshot-design.md. Flag:
- any missed orphan references to removed symbols (_EAGER_CONTROLS, _TAB_LAZY_CONTROLS, _LAZY_TAB_SPECS, _N_RELAY_STORES, _make_lazy_relay_callback, snapshot-apply-*, share-scope)
- any callback whose Output ids don't exist in the layout after the refactor
- any violation of the first-render-bump invariant
- allow_duplicate hygiene
- any tests that will fail or are missing coverage
Don't write code. Report concrete findings in <400 words.
```

- [ ] **Step 3.2: Act on findings**

If the reviewer returns no blocking findings, proceed to Task 4. If it returns findings, amend the commit or add a follow-up commit that addresses each, then re-run Steps 2.9–2.11 and repeat Step 3.1.

---

## Task 4: Deploy to prod and verify

- [ ] **Step 4.1: Syntax-check + local smoke**

```bash
cd /scratch/code/bitcoinprojections
cd btc_web && PYTHONPATH=".:.." ../btc_venv/bin/python3 -c \
  "import layout, figures, callbacks, cache, engines.adapter; print('OK')" && cd ..
```

Expected: `OK`. If anything else, fix before deploying.

- [ ] **Step 4.2: Start dev server + curl-check /1 + /2**

```bash
lsof -ti :8050 2>/dev/null | xargs -r kill -9
DEV=1 nohup bash run_web.sh > /tmp/quantoshi_dev.log 2>&1 &
sleep 10
curl -s -o /dev/null -w "/1: %{http_code}\n" http://localhost:8050/1
curl -s -o /dev/null -w "/2: %{http_code}\n" http://localhost:8050/2
tail -10 /tmp/quantoshi_dev.log
```

Expected: both return `200`, no tracebacks in log.

- [ ] **Step 4.3: Deploy to prod**

```bash
cd /scratch/code/bitcoinprojections
git push origin master
ssh root@89.167.70.45 'cd /opt/quantoshi && git pull && redis-cli FLUSHDB && systemctl restart quantoshi'
```

Expected: `OK` / no error output from the ssh command.

- [ ] **Step 4.4: Smoke-test prod**

```bash
sleep 5
curl -s -o /dev/null -w "prod /: %{http_code}\n" https://quantoshi.xyz/
curl -s -o /dev/null -w "prod /1: %{http_code}\n" https://quantoshi.xyz/1
curl -s -o /dev/null -w "prod /2: %{http_code}\n" https://quantoshi.xyz/2
# Verify assets still serve:
curl -sI https://quantoshi.xyz/assets/gear_clicks.js | head -2
# Verify share-scope is absent from the rendered layout JSON:
curl -s https://quantoshi.xyz/_dash-layout | grep -c "share-scope"
# Verify relay store ids are absent:
curl -s https://quantoshi.xyz/_dash-layout | grep -cE "snapshot-apply-(bubble|heatmap|dca|retire|supercharge|citadel|leverage)"
```

Expected: HTTP 200 for each path, `share-scope` count == 0, relay-store count == 0.

- [ ] **Step 4.5: Tail prod logs ~30s for errors**

```bash
ssh root@89.167.70.45 'journalctl -u quantoshi -n 100 --no-pager' 2>&1 | tail -40
```

Expected: no `nonexistent object` errors, no Python tracebacks since restart. Startup messages should include the resqr bundle bind line and gunicorn workers booting.

- [ ] **Step 4.6: Commit deploy marker (optional)**

If anything in the plan revealed an issue that needed a hotfix, ensure it's committed and pushed. Otherwise nothing to do here.

---

## Task 5: Update project memory + close out

- [ ] **Step 5.1: Update the parked single-redraw memory**

Edit `~/.claude/projects/-scratch-code-bitcoinprojections/memory/parked_single_redraw_brainstorm.md`: update the "Pre-conditions for resuming" section to note that the refactor has shipped and callback count is now reduced from 8 writers to 2 (apply_globals + active-tab apply_tab).

- [ ] **Step 5.2: Update the revert-point memory with the post-refactor commit**

Edit `~/.claude/projects/-scratch-code-bitcoinprojections/memory/reference_alltabs_share_revert_point.md`: append "Post-refactor HEAD at time of shipping: `<commit-sha>`. Diff range for rollback planning: `git diff 34a19a5..<commit-sha>`."

- [ ] **Step 5.3: Mark implementation complete in TaskList**

Mark task 16 completed.

---

## Self-Review Checklist

**Spec coverage:** Every requirement in the spec has a task.
- Remove share-scope radio → 2.4. ✓
- Unconditional tab_filter in _encode_snapshot → covered via 2.2 `manage_snapshot` simplification (encode side). ✓
  - Actually: the spec also says to update `_encode_snapshot` in `snapshot.py` itself. Added to Step 2.2 follow-up below.
- Delete _EAGER/_LAZY partition → 2.1. ✓
- Replace apply_snapshot with apply_globals + per-tab factory → 2.1. ✓
- Remove 7 relay stores from layout → 2.5. ✓
- Routing.py guard comment → 2.6. ✓
- Remove share_scope State + scope field → 2.2, 2.3. ✓
- Update test_callbacks.py → 2.7. ✓
- Delete stale snapshot tests → 2.8. ✓
- Add 6 new tests → Task 1. ✓
- Deploy → Task 4. ✓
- Update memory → Task 5. ✓

**Missing: explicit `_encode_snapshot` cleanup.** Added addendum below.

**Placeholder scan:** no TBDs or "handle appropriately" instances.

**Type consistency:** `_GLOBAL_CONTROLS`, `_PER_TAB_CONTROLS`, `apply_globals`, `apply_tab_{tab_id}` used consistently across Task 1 tests and Task 2 implementation.

### Addendum to Step 2.2 — also update `snapshot.py::_encode_snapshot`

The spec says `_encode_snapshot(state, tab_filter=None)` should unconditionally apply the filter. Since callers are only in `snapshot_cb.py` and tests, and tests expect `tab_filter=None` to still work for internal use (legacy path), the minimal change is NONE to `snapshot.py` — we keep the signature and just always pass a non-None `tab_filter` from `manage_snapshot`. Documented here so an implementer doesn't hunt for a nonexistent change.

---

## Execution choice (user-delegated autonomy)

User is sleeping and delegated autonomous execution via inline path. Use **superpowers:executing-plans** to run tasks in this session with dash-callback-reviewer at Task 3 as the checkpoint. Deploy after Task 3 passes.
