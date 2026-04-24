# Restore-Progress Modal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a blocking modal that appears 150 ms after a share-link page load, shows a spinner + "Restoring your shared view…", and auto-dismisses when `snapshot-pending` flips `False` (subject to a 500 ms min-display + 5 s hard fallback). Bump the existing safety timer from 3 s → 4 s so it can't prematurely dismiss the modal on cold-cache paths.

**Architecture:** One `dbc.Modal` at root layout; two clientside callbacks in `callbacks/snapshot_cb.py` (one opens on `url.hash` matching `q[1-3]:`, one closes on `snapshot-pending` flipping False). Zero new server callbacks. Zero changes to `apply_globals` / `apply_tab_*` / `restore_from_url`.

**Tech Stack:** Dash 4.0.0, dbc 2.0.4, pytest, Playwright/Firefox for the E2E.

**Spec:** `docs/superpowers/specs/2026-04-24-restore-progress-modal-design.md`

**Deploy command (user-delegated autonomous):**
```bash
git push origin master && ssh root@89.167.70.45 "cd /opt/quantoshi && git pull && redis-cli FLUSHDB && systemctl restart quantoshi"
```

---

## Task 1: Failing grep tests (RED)

**Files:**
- Modify: `btc_web/test_snapshot.py` (append to `TestSnapshotPendingGate` class)

- [ ] **Step 1.1: Append new tests to `test_snapshot.py`**

Open `btc_web/test_snapshot.py` and find the class `TestSnapshotPendingGate`. Append these methods as the final tests of that class:

```python
    def test_restore_progress_modal_in_layout(self):
        """Modal id must exist in rendered layout."""
        import layout
        import json
        rendered = layout._serve_layout() if hasattr(layout, "_serve_layout") else None
        serialised = json.dumps(rendered, default=str) if rendered else ""
        assert "restore-progress-modal" in serialised, (
            "restore-progress-modal missing from layout")

    def test_restore_progress_modal_backdrop_static(self):
        """Modal must be blocking: backdrop='static', keyboard=False."""
        import os, pathlib
        here = pathlib.Path(os.path.dirname(__file__))
        src = (here / "layout" / "__init__.py").read_text()
        idx = src.find('id="restore-progress-modal"')
        assert idx > 0, "restore-progress-modal not declared in layout"
        block = src[max(0, idx-400):idx+200]
        assert 'backdrop="static"' in block, (
            "restore-progress-modal must use backdrop='static'")
        assert "keyboard=False" in block, (
            "restore-progress-modal must use keyboard=False")

    def test_open_callback_initial_duplicate(self):
        """Open-on-hash clientside must use prevent_initial_call='initial_duplicate'."""
        import os, pathlib
        here = pathlib.Path(os.path.dirname(__file__))
        src = (here / "callbacks" / "snapshot_cb.py").read_text()
        idx = src.find('"restore-progress-modal", "is_open"')
        assert idx > 0, "restore-progress-modal Output not found in snapshot_cb.py"
        # Walk back to find the enclosing clientside_callback decorator
        block = src[max(0, idx-3000):idx]
        # First callback block contains Input("url","hash") and
        # prevent_initial_call='initial_duplicate'
        opener = src[max(0, idx-3000):idx+500]
        assert 'prevent_initial_call=\'initial_duplicate\'' in opener or \
               'prevent_initial_call="initial_duplicate"' in opener, (
            "open callback must use prevent_initial_call='initial_duplicate'")

    def test_open_callback_debounce_150ms(self):
        """Open callback must debounce 150 ms."""
        import os, pathlib
        here = pathlib.Path(os.path.dirname(__file__))
        src = (here / "callbacks" / "snapshot_cb.py").read_text()
        idx = src.find('__restoreOpenTimer')
        assert idx > 0, "__restoreOpenTimer not found"
        block = src[idx:idx + 500]
        assert '150' in block, "150ms debounce not found in open callback"

    def test_fallback_timer_5000ms(self):
        """Open callback must arm a 5000 ms hard fallback."""
        import os, pathlib
        here = pathlib.Path(os.path.dirname(__file__))
        src = (here / "callbacks" / "snapshot_cb.py").read_text()
        idx = src.find('__restoreFallback')
        assert idx > 0, "__restoreFallback not found"
        block = src[idx:idx + 1500]
        assert '5000' in block, "5000ms hard fallback not found"

    def test_close_callback_min_display_500ms(self):
        """Close callback must enforce 500 ms min-display."""
        import os, pathlib
        here = pathlib.Path(os.path.dirname(__file__))
        src = (here / "callbacks" / "snapshot_cb.py").read_text()
        idx = src.find('__restoreOpenTime')
        assert idx > 0, "__restoreOpenTime not found"
        block = src[idx:idx + 2000]
        assert 'Math.max' in block and '500' in block, (
            "500ms min-display (Math.max) not found in close callback")
```

- [ ] **Step 1.2: Update existing safety-timer test**

Find `test_safety_timer_at_least_3000ms` in `btc_web/test_snapshot.py` and replace the assertion:

```python
    def test_safety_timer_at_least_3000ms(self):
        """Clientside safety timer must wait at least 4000 ms before
        unconditionally clearing the gate. Bumped from 3000 to 4000
        (2026-04-24) to prevent premature modal dismiss on cold cache."""
        import os, pathlib, re
        here = pathlib.Path(os.path.dirname(__file__))
        src = (here / "callbacks" / "snapshot_cb.py").read_text()
        idx = src.find("Safety-timer")
        assert idx > 0, "Safety-timer block comment not found in snapshot_cb.py"
        block = src[idx:idx + 2000]
        m = re.search(r"}\s*,\s*(\d+)\s*\)\s*;", block)
        assert m, f"setTimeout delay literal not found in safety-timer block"
        duration = int(m.group(1))
        assert duration >= 4000, (
            f"Safety timer must be >= 4000ms; got {duration}")
```

- [ ] **Step 1.3: Run the new tests — they must FAIL**

```bash
cd /scratch/code/bitcoinprojections
btc_venv/bin/python3 -m pytest btc_web/test_snapshot.py::TestSnapshotPendingGate -v --ignore-glob='*_e2e.py' 2>&1 | tail -20
```

Expected: at least 6 failures (modal-in-layout, backdrop-static, open-callback-initial-duplicate, debounce-150ms, fallback-5000ms, min-display-500ms). The existing `test_safety_timer_at_least_3000ms` should still pass (3000 >= 3000 was the old assertion; now with `>= 4000`, it should also fail because the code still has 3000).

Wait — the existing test's assertion is now `>= 4000`. Current code has `3000`. So this test will also fail. Expected: 7 failures.

- [ ] **Step 1.4: Commit RED**

```bash
git add btc_web/test_snapshot.py
git commit -m "test(modal): RED tests for restore-progress modal + 4s safety timer"
```

---

## Task 2: Add modal to layout (GREEN pt.1)

**Files:**
- Modify: `btc_web/layout/__init__.py`

- [ ] **Step 2.1: Locate the existing splash modal**

```bash
grep -n 'id="splash-modal"' /scratch/code/bitcoinprojections/btc_web/layout/__init__.py
```

Expected: one match around line 527.

- [ ] **Step 2.2: Add the new modal immediately before the splash modal**

Read the 5 lines preceding `id="splash-modal"` to orient:

```bash
sed -n '520,530p' /scratch/code/bitcoinprojections/btc_web/layout/__init__.py
```

Insert the new modal right before the `dbc.Modal([` that contains `id="splash-modal"`. Use the Edit tool to apply:

Replace the two-line header of the splash modal block:

```python
    ], id="splash-modal", is_open=False, centered=True, backdrop="static",
       className="splash-modal"),
```

with:

```python
    ], id="splash-modal", is_open=False, centered=True, backdrop="static",
       className="splash-modal"),
    # Blocking modal shown during share-link snapshot restore. See spec
    # docs/superpowers/specs/2026-04-24-restore-progress-modal-design.md.
    dbc.Modal([
        dbc.ModalBody([
            html.Div([
                html.Img(src="/assets/quantoshi_logo_nav.png",
                         style={"width": "80px", "height": "auto",
                                "marginBottom": "16px", "opacity": "0.9"}),
                html.Div([
                    dbc.Spinner(size="sm", color="primary",
                                spinner_style={"marginRight": "10px"}),
                    html.Span("Restoring your shared view…"),
                ], style={"display": "flex", "alignItems": "center",
                          "justifyContent": "center",
                          "fontSize": UI_FONT_BASE, "color": MUTED_TEXT,
                          "marginTop": "4px"}),
            ], style={"textAlign": "center", "padding": "24px"}),
        ]),
    ], id="restore-progress-modal", is_open=False, centered=True, size="sm",
       backdrop="static", keyboard=False),
```

- [ ] **Step 2.3: Verify the needed imports are already present**

```bash
grep -n "UI_FONT_BASE\|MUTED_TEXT" /scratch/code/bitcoinprojections/btc_web/layout/__init__.py | head -3
```

Expected: imports show up in the existing import block near the top of the file. If `MUTED_TEXT` is not imported, add it to the colors import list (line ~50).

- [ ] **Step 2.4: Syntax-check**

```bash
cd /scratch/code/bitcoinprojections/btc_web
PYTHONPATH=".:.." ../btc_venv/bin/python3 -c "import layout; print('OK')"
cd ..
```

Expected: `OK`. If module-level errors occur, fix before proceeding.

- [ ] **Step 2.5: Run the two layout-level tests that should now pass**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_snapshot.py::TestSnapshotPendingGate::test_restore_progress_modal_in_layout btc_web/test_snapshot.py::TestSnapshotPendingGate::test_restore_progress_modal_backdrop_static -v 2>&1 | tail -10
```

Expected: both PASS.

---

## Task 3: Bump safety timer + add two clientside callbacks (GREEN pt.2)

**Files:**
- Modify: `btc_web/callbacks/snapshot_cb.py`

- [ ] **Step 3.1: Bump existing safety-timer setTimeout from 3000 to 4000**

Locate the existing safety-timer clientside callback at the end of `snapshot_cb.py`. Grep:

```bash
grep -n "Safety-timer" /scratch/code/bitcoinprojections/btc_web/callbacks/snapshot_cb.py
```

Expected: one match around line 570. Read the surrounding block:

```bash
sed -n '560,620p' /scratch/code/bitcoinprojections/btc_web/callbacks/snapshot_cb.py
```

Find the line:

```python
            }, 3000);
```

Replace with:

```python
            }, 4000);
```

Update the accompanying comment above the callback from "3000 ms chosen to exceed cold-cache compute time" to "4000 ms (bumped from 3000 2026-04-24) chosen to exceed cold-cache compute time on citadel/supercharge AND keep the restore-progress modal from prematurely dismissing (see spec 2026-04-24-restore-progress-modal-design.md)."

- [ ] **Step 3.2: Append the two new clientside callbacks at the end of the file**

Run:

```bash
cat >> /scratch/code/bitcoinprojections/btc_web/callbacks/snapshot_cb.py <<'PY'


# ── Restore-progress modal open on share-hash (clientside) ────────────────
# 150 ms debounce so fast restores show nothing. 5 s hard fallback closes
# the modal unconditionally if the gate never releases. See spec
# docs/superpowers/specs/2026-04-24-restore-progress-modal-design.md.
_app_ctx.app.clientside_callback(
    """
    function(hash) {
        var h = (hash || window.location.hash || '').replace(/^#/, '');
        var isShare = h.indexOf('q1:') === 0 ||
                      h.indexOf('q2:') === 0 ||
                      h.indexOf('q3:') === 0;
        if (!isShare) return window.dash_clientside.no_update;

        // Debounce open by 150 ms — fast restores never show the modal.
        if (window.__restoreOpenTimer) clearTimeout(window.__restoreOpenTimer);
        window.__restoreOpenTimer = setTimeout(function () {
            window.dash_clientside.set_props(
                'restore-progress-modal', { is_open: true });
            window.__restoreOpenTime = performance.now();
        }, 150);

        // Hard fallback: 5 s unconditional close.
        if (window.__restoreFallback) clearTimeout(window.__restoreFallback);
        window.__restoreFallback = setTimeout(function () {
            window.dash_clientside.set_props(
                'restore-progress-modal', { is_open: false });
            window.__restoreOpenTime = null;
        }, 5000);

        return window.dash_clientside.no_update;
    }
    """,
    Output("restore-progress-modal", "is_open", allow_duplicate=True),
    Input("url", "hash"),
    prevent_initial_call='initial_duplicate',
)


# ── Restore-progress modal close on snapshot-pending release ──────────────
# Enforces 500 ms min-display so a fast restore that slipped past the 150 ms
# open debounce doesn't flash the modal for <50 ms. Uses requestAnimationFrame
# after the min-display delay so Plotly's trace update paints before fade.
_app_ctx.app.clientside_callback(
    """
    function(pending) {
        if (pending === true) return window.dash_clientside.no_update;
        // If the debounced open never fired, cancel everything and do nothing.
        if (!window.__restoreOpenTime) {
            if (window.__restoreOpenTimer) {
                clearTimeout(window.__restoreOpenTimer);
                window.__restoreOpenTimer = null;
            }
            if (window.__restoreFallback) {
                clearTimeout(window.__restoreFallback);
                window.__restoreFallback = null;
            }
            return window.dash_clientside.no_update;
        }
        // Min-display 500 ms. Extra delay = max(0, 500 - elapsed).
        var elapsed = performance.now() - window.__restoreOpenTime;
        var delay = Math.max(0, 500 - elapsed);
        setTimeout(function () {
            requestAnimationFrame(function () {
                window.dash_clientside.set_props(
                    'restore-progress-modal', { is_open: false });
                window.__restoreOpenTime = null;
                if (window.__restoreFallback) {
                    clearTimeout(window.__restoreFallback);
                    window.__restoreFallback = null;
                }
            });
        }, delay);
        return window.dash_clientside.no_update;
    }
    """,
    Output("restore-progress-modal", "is_open", allow_duplicate=True),
    Input("snapshot-pending", "data"),
    prevent_initial_call=True,
)
PY
echo ok
```

- [ ] **Step 3.3: Syntax-check**

```bash
cd /scratch/code/bitcoinprojections/btc_web
PYTHONPATH=".:.." ../btc_venv/bin/python3 -c "import app; print('OK')" 2>&1 | tail -3
cd ..
```

Expected: `OK`.

- [ ] **Step 3.4: Run all RED tests — they should now all pass**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_snapshot.py::TestSnapshotPendingGate -v --ignore-glob='*_e2e.py' 2>&1 | tail -20
```

Expected: all previously-RED tests now PASS. The `test_safety_timer_at_least_3000ms` (now asserting `>= 4000`) also passes.

- [ ] **Step 3.5: Run full non-E2E suite**

```bash
btc_venv/bin/python3 -m pytest btc_web/ --ignore-glob='*_e2e.py' 2>&1 | tail -5
```

Expected: 1 pre-existing failure (`test_no_hex_literals_outside_colors_module` in static_pages.py — unrelated). No new failures.

- [ ] **Step 3.6: Commit Task 2 + Task 3 together (GREEN)**

```bash
git add btc_web/layout/__init__.py btc_web/callbacks/snapshot_cb.py
git commit -m "$(cat <<'EOF'
feat(restore): blocking progress modal on share-link load

Adds a dbc.Modal(id=restore-progress-modal) at root layout; two
clientside callbacks in snapshot_cb.py:

- Open on url.hash matching q[1-3]: prefix. 150 ms debounce so fast
  restores never show the modal. 5 s hard fallback closes the modal
  unconditionally.
- Close on snapshot-pending flipping False. 500 ms min-display +
  requestAnimationFrame so Plotly paints the restored traces before
  the modal fades.

Also bumps the existing snapshot-pending safety timer 3000→4000 ms
(prevents premature modal dismiss on cold-cache paths where the
apply_tab_* chain can exceed 3 s).

See spec docs/superpowers/specs/2026-04-24-restore-progress-modal-design.md.
EOF
)"
```

---

## Task 4: dash-callback-reviewer gate on the diff (HARD GATE)

- [ ] **Step 4.1: Dispatch the reviewer**

Use the Agent tool with `subagent_type="dash-callback-reviewer"`. Prompt template:

```
Review the diff from 0e6f341..HEAD in /scratch/code/bitcoinprojections/btc_web/
against spec docs/superpowers/specs/2026-04-24-restore-progress-modal-design.md.

Verify:
1. Modal is declared exactly once in layout/__init__.py with backdrop='static', keyboard=False.
2. Two new clientside callbacks in snapshot_cb.py, with correct prevent_initial_call:
   - Open callback: Input('url','hash'), Output restore-progress-modal.is_open, PIC='initial_duplicate'.
   - Close callback: Input('snapshot-pending','data'), same Output, PIC=True.
3. No new Python server callbacks, no new Outputs on restore_from_url / apply_globals / apply_tab_*.
4. Existing safety timer bumped 3000→4000 ms. Comment updated accordingly.
5. gunicorn footgun: allow_duplicate=True + prevent_initial_call=False is fatal. Confirm both new callbacks avoid this.
6. Grep snapshot-pending in btc_web/ to confirm no other consumer embeds a hardcoded 3000 ms assumption that might now drift.

Flag BLOCKING issues only. Under 400 words.
```

- [ ] **Step 4.2: Fix any BLOCKING findings**

Iterate: commit fixes, re-dispatch reviewer. Only proceed to deploy after reviewer returns zero BLOCKING findings.

---

## Task 5: Deploy to prod

- [ ] **Step 5.1: Dev smoke**

```bash
cd /scratch/code/bitcoinprojections
lsof -ti :8050 2>/dev/null | xargs -r kill -9
DEV=1 nohup bash run_web.sh > /tmp/quantoshi_dev.log 2>&1 &
sleep 10
curl -s -o /dev/null -w "/1: %{http_code}\n" http://localhost:8050/1
tail -5 /tmp/quantoshi_dev.log
```

Expected: HTTP 200, no Python tracebacks.

- [ ] **Step 5.2: Push + deploy**

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
curl -s https://quantoshi.xyz/_dash-layout | grep -c "restore-progress-modal"
ssh root@89.167.70.45 'journalctl -u quantoshi --since "60 seconds ago" --no-pager | grep -iE "error|traceback|critical|nonexistent" | head -5'
```

Expected: HTTP 200 + `restore-progress-modal` count ≥ 1 + zero errors in logs.

- [ ] **Step 5.4: Manual behavior test on prod**

Open https://quantoshi.xyz/1#q3:H4sIAHyy62kC_-1U22rDMAz9Fz0rQXZz8b4lhJGLW8w0O83iff-UuoUW-tBSKAyCOQdLtoRAOmqaCgk1FQaBnbfdDHIJB8BGk67Wl6rFpsxLNC2qHa6nRMp3qHRuDCoJ95EZYf2fUZHpApJno3cTISxhytjul-seQB_7nu3jbVG10WX9oSVhIZC0o913kZetsxv9XwLZai2C86P7dWPsONnT7AYRB8kuUzLpkBAT6Dz_t9774ns1HoaOrR9POzhMi_teKzwrmMPw9UkX0wcvFdcIs_05zs-psk38B3kwbIT6BQAA in a browser. Expected timeline:
- 0–150 ms: no modal visible.
- 150 ms → ~4 s: modal visible with spinner + "Restoring your shared view…" text.
- After `snapshot-pending` flips False (typically 200–500 ms for warm cache, ≤ 4 s cold): modal fades out; restored controls (bub-qs = outer+median, etc.) + chart visible.

Open https://quantoshi.xyz/1 (no hash) — no modal, no behavior change.

---

## Task 6: Playwright E2E test

**Files:**
- Create: `btc_web/test_restore_modal_e2e.py`

- [ ] **Step 6.1: Copy the existing Playwright harness pattern**

Model after `btc_web/test_tax_e2e.py` or `btc_web/test_plot_appearance_e2e.py`. Read the top of one of them for the harness:

```bash
sed -n '1,40p' /scratch/code/bitcoinprojections/btc_web/test_plot_appearance_e2e.py
```

Note the `@pytest.fixture` for `page`, URL/port constants, any skip guards.

- [ ] **Step 6.2: Write the test file**

```python
"""E2E: restore-progress modal persists through cold-cache restore window.

Guards against premature dismiss regressions. See spec
docs/superpowers/specs/2026-04-24-restore-progress-modal-design.md."""
import pytest
import time


# Known-good share link (encodes bub-qs=['outer','median'] + 30 other
# bubble-tab controls). From user's own generated link 2026-04-24.
_SHARE_LINK = (
    "http://localhost:8050/1#q3:"
    "H4sIAHyy62kC_-1U22rDMAz9Fz0rQXZz8b4lhJGLW8w0O83iff-UuoUW-tBSKAy"
    "COQdLtoRAOmqaCgk1FQaBnbfdDHIJB8BGk67Wl6rFpsxLNC2qHa6nRMp3qHRuDC"
    "oJ95EZYf2fUZHpApJno3cTISxhytjul-seQB_7nu3jbVG10WX9oSVhIZC0o913"
    "kZetsxv9XwLZai2C86P7dWPsONnT7AYRB8kuUzLpkBAT6Dz_t9774ns1HoaOrR"
    "9POzhMi_teKzwrmMPw9UkX0wcvFdcIs_05zs-psk38B3kwbIT6BQAA"
)


@pytest.mark.e2e
def test_restore_modal_persists_until_restore_completes(page):
    """Load a share link and assert the modal is visible during the
    restore window and hidden before the 5 s hard fallback."""
    page.goto(_SHARE_LINK)

    # Poll is_open state via computed style (`display:block` = shown).
    def is_modal_visible():
        try:
            disp = page.evaluate(
                "() => { var m = document.getElementById('restore-progress-modal');"
                "  if (!m) return 'no-modal';"
                "  var parent = m.closest('.modal');"
                "  if (!parent) return 'no-parent';"
                "  return parent.classList.contains('show') ? 'visible' : 'hidden';"
                "}"
            )
            return disp == "visible"
        except Exception:
            return False

    # Sample at 3500 ms — must still be visible.
    time.sleep(3.5)
    assert is_modal_visible(), (
        "restore-progress modal must still be visible at 3500 ms — "
        "safety timer regression may have caused premature dismiss")

    # Sample at 5500 ms — must be hidden.
    time.sleep(2.0)  # now at 5.5 s
    assert not is_modal_visible(), (
        "restore-progress modal must be hidden by 5500 ms — hard "
        "fallback (5 s) regression may have left modal stuck open")
```

- [ ] **Step 6.3: Run the E2E test against dev**

```bash
cd /scratch/code/bitcoinprojections
lsof -ti :8050 2>/dev/null | xargs -r kill -9
DEV=1 nohup bash run_web.sh > /tmp/quantoshi_dev.log 2>&1 &
sleep 10
btc_venv/bin/python3 -m pytest btc_web/test_restore_modal_e2e.py -v 2>&1 | tail -10
```

Expected: test PASSES. Both assertions hold.

- [ ] **Step 6.4: Commit the E2E test**

```bash
git add btc_web/test_restore_modal_e2e.py
git commit -m "test(modal): E2E — modal persists at 3.5s, hidden by 5.5s"
```

- [ ] **Step 6.5: Push**

```bash
git push origin master
```

(No redeploy needed — this is a test-only change.)

---

## Self-Review

**Spec coverage:**
- Modal at root layout (spec §Architecture) → Task 2 ✓
- Two clientside callbacks (spec §Clientside callback 1 & 2) → Task 3 ✓
- Safety timer 3 s → 4 s (spec §Safety-timer adjustment) → Step 3.1 ✓
- 150 ms open debounce (spec §Fast-restore behavior) → Step 3.2 body ✓
- 500 ms min-display (spec §Minimum display time) → Step 3.2 body ✓
- 5 s hard fallback (spec §Hard fallback) → Step 3.2 body ✓
- All 7 grep tests (spec §Tests) → Task 1 ✓
- Playwright E2E (spec §Integration test) → Task 6 ✓
- `dash-callback-reviewer` gate (spec §Rollout) → Task 4 ✓
- Prod verification (spec §Rollout) → Task 5 ✓
- Implementation-time grep check (spec §Implementation-time checks) → folded into Task 4 reviewer prompt ✓

**Placeholder scan:** no TBDs, vague "handle appropriately", or skipped code. Every step shows exact content.

**Type consistency:** modal id `restore-progress-modal` identical across layout declaration, both clientside Output references, all 7 tests, and the E2E. Window globals (`__restoreOpenTimer`, `__restoreFallback`, `__restoreOpenTime`) spelled consistently. Timer literals (150, 500, 4000, 5000) match spec.

---

## Execution choice

User delegated autonomous execution through prod deploy. Use **superpowers:executing-plans** inline. Task 4 is the hard gate — don't push before reviewer returns zero BLOCKING findings.
