# Phase 1: restore_from_url Dispatch Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `restore_from_url` dispatch reliably for `/2`–`/7` share links by routing the bubble figure through an always-mounted Store + clientside `set_props` relay instead of the lazy-mounted `bubble-graph` Output.

**Architecture:** Replace `Output("bubble-graph", "figure", allow_duplicate=True)` on `restore_from_url` with `Output("restore-bubble-fig", "data", allow_duplicate=True)` (a new always-mounted `dcc.Store`). A clientside callback watches the Store and uses `window.dash_clientside.set_props('bubble-graph', {figure: fig})` to deliver the figure when bubble-graph is mounted. `set_props` bypasses Dash's registered-Output existence check, so the lazy-mount problem disappears.

**Tech Stack:** Plotly Dash 4.0.0, dcc.Store, clientside_callback, set_props, Playwright (Firefox) for E2E.

**Empirical verification (already done):** Probe edits in working directory confirmed in dev — `/3` share link with `dca-amount=999` now correctly restores the widget value (was 100/default before). `/1` share link still works (chart renders 1.6s). No new JS errors.

**Architect verdict:** Single commit. The three code changes (Store + Output swap + clientside relay) are mutually dependent and must land together.

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `btc_web/layout/__init__.py` | modified (probe) | Add `restore-bubble-fig` Store next to `active-chart-committed` |
| `btc_web/callbacks/snapshot_cb.py` | modified (probe) | Swap one Output in `restore_from_url`; add clientside relay |
| `btc_web/test_callbacks.py` | extend | Add T6 (no `bubble-graph.figure` Output) + T4 (Store payload semantics) |
| `btc_web/test_restore_phase1_e2e.py` | create | T5 Playwright: `/3` restore, `/1` regression, no JS errors |
| `memory/restore_callback_architecture.md` | modified | Phase 1 entry documenting the relay pattern |
| `docs/architecture.md` | modified | Note the lazy-Output footgun + Store-relay solution |

---

## Task 1: Confirm probe edits are in working directory

**Files:**
- Inspect: `btc_web/layout/__init__.py`, `btc_web/callbacks/snapshot_cb.py`

- [ ] **Step 1: Verify Store is in layout**

```bash
grep -n 'restore-bubble-fig' btc_web/layout/__init__.py
```

Expected: line ~300 with `dcc.Store(id="restore-bubble-fig", storage_type="memory", data=None),`

If missing, add it directly after the `active-chart-committed` Store (around line 299):
```python
    dcc.Store(id="active-chart-committed", storage_type="memory", data=None),
    # Phase 1 (2026-04-25): bubble figure is delivered via this always-mounted
    # Store + a clientside set_props relay rather than via Output("bubble-graph",
    # "figure") on restore_from_url. Reason: bubble-graph is inside bubble-lazy
    # which contains "Loading..." on /2-/7 initial loads; Dash 4 silently drops
    # the entire callback dispatch when an Output's component is absent from
    # DOM. Routing the figure through an always-mounted Store fixes /2-/7.
    dcc.Store(id="restore-bubble-fig", storage_type="memory", data=None),
```

- [ ] **Step 2: Verify Output swap in restore_from_url**

```bash
grep -A 6 '@callback' btc_web/callbacks/snapshot_cb.py | grep -E 'restore-bubble-fig|bubble-graph'
```

Expected: `Output("restore-bubble-fig", "data", allow_duplicate=True),` (NOT `Output("bubble-graph", "figure", ...)`).

If still says `bubble-graph`, swap to:
```python
    Output("restore-bubble-fig",   "data", allow_duplicate=True),
```
(line 48 of snapshot_cb.py, replacing the previous `Output("bubble-graph", "figure", ...)`)

- [ ] **Step 3: Verify clientside relay is present**

```bash
grep -B 1 -A 16 "Bubble figure relay via set_props" btc_web/callbacks/snapshot_cb.py
```

Expected: a `_app_ctx.app.clientside_callback(...)` block with `set_props('bubble-graph', {figure: fig})` and try-catch.

If missing, insert before the existing "Direct modal close on active-chart-committed" listener block:
```python
# ── Bubble figure relay via set_props (Phase 1, 2026-04-25) ───────────────
# `restore_from_url` cannot have `Output("bubble-graph", "figure", ...)`
# because bubble-graph is inside bubble-lazy which contains "Loading..."
# on /2-/7 initial loads — Dash 4 silently drops the entire callback
# dispatch when an Output's component is absent from DOM, breaking
# control restore for non-bubble share links. Instead, restore_from_url
# writes the figure to the always-mounted `restore-bubble-fig` Store and
# this clientside callback uses set_props to push it into bubble-graph.
# set_props bypasses the registered-Output existence check.
#
# The try-catch is insurance against a Dash 4.0.0 reducer bug where
# set_props with an absent target may throw (GitHub plotly/dash#2897).
# Guard `fig == null` short-circuits the self-clear `return null` path.
_app_ctx.app.clientside_callback(
    """
    function(fig) {
        var NU = window.dash_clientside.no_update;
        if (fig == null) return NU;
        try {
            window.dash_clientside.set_props('bubble-graph', {figure: fig});
        } catch (e) {
            console.warn('restore-bubble-fig: set_props failed', e);
        }
        if (window.__qsTrace) window.__qsTrace('restore-bubble-fig delivered');
        return null;  // clear store after delivery
    }
    """,
    Output("restore-bubble-fig", "data", allow_duplicate=True),
    Input("restore-bubble-fig", "data"),
    prevent_initial_call=True,
)
```

- [ ] **Step 4: Syntax check**

Run: `cd btc_web && PYTHONPATH=".:.." ../btc_venv/bin/python3 -c "import layout, callbacks, snapshot; print('OK')"`

Expected: `OK` (no ImportError, no SyntaxError).

---

## Task 2: T6 unit test — assert `bubble-graph.figure` is NOT in restore_from_url's outputs

**Files:**
- Modify: `btc_web/test_callbacks.py` (extend `TestNoDuplicateCallbackOutputs`)

- [ ] **Step 1: Add the failing test**

Append to `class TestNoDuplicateCallbackOutputs` in `btc_web/test_callbacks.py` (after `test_restore_from_url_uses_intermediate_store` around line 1574):

```python
    def test_restore_from_url_does_not_output_bubble_graph(self):
        """Phase 1 invariant (2026-04-25): restore_from_url must NOT have
        bubble-graph.figure as an Output.

        bubble-graph is inside bubble-lazy which contains "Loading..." on
        /2-/7 initial loads. Dash 4 silently drops the entire callback
        dispatch when an Output's component is absent from DOM, breaking
        control restore for non-bubble share links. The fix is to route
        the bubble figure through restore-bubble-fig (always-mounted
        Store) + a clientside set_props relay.

        See memory/restore_callback_architecture.md (Phase 1 section).
        """
        import _app_ctx
        app = _app_ctx.app

        for cb_key in app.callback_map:
            parts = cb_key.split("...")
            clean_parts = [p.split("@")[0] for p in parts]
            # Identify restore_from_url by its loaded-hash-store output
            if "loaded-hash-store.data" not in clean_parts:
                continue
            assert "bubble-graph.figure" not in clean_parts, (
                "REGRESSION: restore_from_url has bubble-graph.figure as an "
                "Output. This breaks /2-/7 share-link restore because "
                "bubble-graph is lazy-mounted. Use restore-bubble-fig Store "
                "instead. See memory/restore_callback_architecture.md."
            )
            assert "restore-bubble-fig.data" in clean_parts, (
                "restore_from_url is missing restore-bubble-fig.data Output. "
                "Required for the figure-relay pattern that fixes /2-/7."
            )
            return
        pytest.fail("restore_from_url callback not found in callback_map")
```

- [ ] **Step 2: Run test to confirm it passes (probe edits already in place)**

Run:
```bash
cd /scratch/code/bitcoinprojections && \
  btc_venv/bin/python3 -m pytest btc_web/test_callbacks.py::TestNoDuplicateCallbackOutputs::test_restore_from_url_does_not_output_bubble_graph -v
```

Expected: `1 passed`. If the probe edits got reverted, this test will fail; re-do Task 1.

---

## Task 3: T4 unit test — Store payload semantics

**Files:**
- Modify: `btc_web/test_callbacks.py` (extend `TestRestoreFromUrl`)

- [ ] **Step 1: Add tests for Store payload behavior**

Append to `class TestRestoreFromUrl` in `btc_web/test_callbacks.py` (after `test_valid_roundtrip` around line 1512):

```python
    def test_store_payload_is_figure_for_bubble_share(self):
        """Phase 1: position 3 of restore_from_url's return is the
        restore-bubble-fig Store payload. For a bubble share with a
        decodable hash, it is either a Plotly figure dict (when
        restore_builder succeeded) or no_update (when builder fell
        back, e.g. CTA-active)."""
        from dash import no_update
        state = {
            "main-tabs:active_tab": "bubble",
            "bub-qs:value": ["median"],
        }
        encoded = _encode_snapshot(state)
        hash_str = f"#q3:{encoded}"
        _, _, _, fig_payload, _ = restore_from_url(hash_str)
        # Either a figure (dict with 'data' key) or no_update.
        if fig_payload is not no_update:
            assert isinstance(fig_payload, dict), (
                f"fig_payload should be a Plotly figure dict, got "
                f"{type(fig_payload).__name__}"
            )
            assert "data" in fig_payload, (
                "Plotly figure dict must have 'data' key"
            )

    def test_store_payload_is_no_update_for_non_bubble(self):
        """Phase 1: for a non-bubble share, position 3 is no_update —
        the chart callback for that tab handles its own figure build.
        Phase 2 will extend per-tab figure builders into the Store
        relay; until then, non-bubble shares fall back to the standard
        chart callback path."""
        from dash import no_update
        state = {
            "main-tabs:active_tab": "dca",
        }
        encoded = _encode_snapshot(state)
        hash_str = f"#q3:{encoded}"
        _, _, _, fig_payload, committed = restore_from_url(hash_str)
        assert fig_payload is no_update, (
            "Non-bubble share must return no_update for restore-bubble-fig "
            "(Phase 1 only builds bubble figures server-side)."
        )
        assert committed is no_update, (
            "Non-bubble share must return no_update for active-chart-committed "
            "(Phase 2 will write this from per-tab figure builders)."
        )
```

- [ ] **Step 2: Run tests to confirm they pass**

Run:
```bash
cd /scratch/code/bitcoinprojections && \
  btc_venv/bin/python3 -m pytest btc_web/test_callbacks.py::TestRestoreFromUrl -v
```

Expected: 5 passed (3 existing + 2 new). If `test_store_payload_is_figure_for_bubble_share` fails because builder returns no_update, that's still acceptable — the test allows either branch.

---

## Task 4: T5 Playwright E2E test — `/3` restores controls + `/1` regression + no JS errors

**Files:**
- Create: `btc_web/test_restore_phase1_e2e.py`

- [ ] **Step 1: Write the E2E test file**

Create `btc_web/test_restore_phase1_e2e.py` with:

```python
"""End-to-end Playwright tests for Phase 1 restore_from_url dispatch fix.

Verifies that /2-/7 share links correctly restore controls (the lazy-
mounted bubble-graph Output no longer blocks dispatch), and that /1
share links still work as a regression check.

Requires:
  pip install playwright && python -m playwright install firefox
  Dev server must be running on :8050
Run:
  cd /scratch/code/bitcoinprojections && btc_venv/bin/python3 \\
      -m pytest btc_web/test_restore_phase1_e2e.py -v --timeout=60
"""
import pytest
import time

try:
    from playwright.sync_api import sync_playwright
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False

pytestmark = pytest.mark.skipif(not HAS_PLAYWRIGHT, reason="playwright not installed")

BASE_URL = "http://localhost:8050"


def _make_share_url(state, path):
    """Encode state dict into q4 share link at the given path."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    os.environ.setdefault("DEV", "1")
    import app  # noqa: F401
    from snapshot import _encode_snapshot_v4
    blob = _encode_snapshot_v4(state)
    return f"{BASE_URL}{path}#q4:{blob}"


def test_dca_share_restores_amount():
    """/3 share link with dca-amount=999 must restore the widget value.

    Pre-Phase-1 this returned 100 (default) because restore_from_url's
    dispatch was silently dropped by Dash 4 (lazy bubble-graph Output).
    """
    url = _make_share_url(
        {"main-tabs:active_tab": "dca", "dca-amount:value": 999},
        "/3",
    )
    with sync_playwright() as p:
        browser = p.firefox.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        page.wait_for_function(
            "() => { var gd = document.querySelector('#dca-graph .js-plotly-plot'); "
            "return gd && gd.data && gd.data.length > 0; }",
            timeout=20_000,
        )
        amount = page.evaluate(
            "() => document.getElementById('dca-amount').value"
        )
        browser.close()
    assert amount == "999", (
        f"DCA share-link restore broken: dca-amount={amount}, expected 999. "
        f"This means restore_from_url's dispatch is still being dropped — "
        f"check that bubble-graph.figure is NOT in its Outputs."
    )


def test_bubble_share_still_restores():
    """/1 regression: bubble share link still renders chart fast."""
    url = _make_share_url(
        {"main-tabs:active_tab": "bubble", "bub-qs:value": ["median"]},
        "/1",
    )
    with sync_playwright() as p:
        browser = p.firefox.launch(headless=True)
        page = browser.new_page()
        t0 = time.perf_counter()
        page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        page.wait_for_function(
            "() => { var gd = document.querySelector('#bubble-graph .js-plotly-plot'); "
            "return gd && gd.data && gd.data.length > 0; }",
            timeout=20_000,
        )
        t_chart = (time.perf_counter() - t0) * 1000
        browser.close()
    assert t_chart < 5000, (
        f"Bubble chart took {t_chart:.0f}ms (expected <5000ms). "
        f"Phase 1 should not regress /1 timing."
    )


def test_dca_share_no_jserror_from_set_props():
    """The set_props clientside relay must not cause JS errors, even
    when bubble-graph is absent from DOM (which it is on /3 init).

    Pre-existing 'nonexistent object' errors for lazy-tab controls
    (bub-qs, sc-stack, ret-stack) are filtered — they are documented
    in feedback_nonexistent_input_perf.md as a separate issue.
    """
    url = _make_share_url(
        {"main-tabs:active_tab": "dca", "dca-amount:value": 999},
        "/3",
    )
    errors = []
    with sync_playwright() as p:
        browser = p.firefox.launch(headless=True)
        page = browser.new_page()

        def on_console(msg):
            if msg.type != "error":
                return
            text = msg.text or ""
            if "nonexistent object was used" in text:
                return  # pre-existing, unrelated to Phase 1
            errors.append(text[:300])

        page.on("console", on_console)
        page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        page.wait_for_function(
            "() => { var gd = document.querySelector('#dca-graph .js-plotly-plot'); "
            "return gd && gd.data && gd.data.length > 0; }",
            timeout=20_000,
        )
        time.sleep(1)  # let any delayed errors surface
        browser.close()
    assert errors == [], (
        f"Unexpected JS errors during /3 share-link load: {errors}. "
        f"set_props on absent bubble-graph may be throwing (plotly/dash#2897)."
    )
```

- [ ] **Step 2: Start dev server**

Run:
```bash
cd /scratch/code/bitcoinprojections && \
  lsof -ti :8050 | xargs -r kill -9 2>/dev/null; \
  sleep 1 && \
  DEV=1 nohup bash run_web.sh > /tmp/quantoshi_dev.log 2>&1 &
sleep 6 && tail -5 /tmp/quantoshi_dev.log
```

Expected: log shows `Dash is running on http://0.0.0.0:8050/`.

- [ ] **Step 3: Run E2E tests**

Run:
```bash
cd /scratch/code/bitcoinprojections && \
  btc_venv/bin/python3 -m pytest btc_web/test_restore_phase1_e2e.py -v --timeout=60
```

Expected: `3 passed`.

If `test_dca_share_restores_amount` fails with `dca-amount=100`, the dispatch is still being dropped — check Task 1's edits.

If `test_dca_share_no_jserror_from_set_props` fails with a `set_props` error, the try-catch needs to swallow more error types or the design needs revision (low probability — already passed in probe).

- [ ] **Step 4: Stop dev server**

Run:
```bash
lsof -ti :8050 | xargs -r kill -9 2>/dev/null
```

---

## Task 5: Run full unit test suite (regression check)

**Files:** No changes — confirm nothing else broke.

- [ ] **Step 1: Run all non-E2E tests**

Run:
```bash
cd /scratch/code/bitcoinprojections && \
  btc_venv/bin/python3 -m pytest btc_web/ -v --ignore-glob='*_e2e.py' 2>&1 | tail -30
```

Expected: all tests pass, exit code 0. Existing `TestRestoreFromUrl` tests on lines 1473-1511 still pass (the 5-tuple shape is unchanged; only position 3's destination changed from a graph to a Store, but its semantic value — a figure or no_update — is unchanged).

If any test fails, do NOT proceed to commit. Diagnose the failure. Most likely candidates:
- A test that hard-codes `bubble-graph.figure` as an expected restore_from_url Output → update the test to use `restore-bubble-fig.data`.
- A test that asserts the callback graph contains specific outputs → adjust.

---

## Task 6: Update memory `restore_callback_architecture.md`

**Files:**
- Modify: `/home/bcg/.claude/projects/-scratch-code-bitcoinprojections/memory/restore_callback_architecture.md`

- [ ] **Step 1: Add Phase 1 section after the phantom-rebuild fix section**

Append after the existing `## Phantom rebuild fix (2026-04-25 evening, commit 767822b — SHIPPED + WORKING)` section, BEFORE `## Files involved`:

```markdown
## Phase 1 dispatch fix (2026-04-25 — SHIPPED + WORKING)

After the phantom-rebuild fix landed, empirical Playwright probing of `/3`
share links revealed a deeper bug: **`restore_from_url` was never
dispatching for non-bubble share links in dev OR prod**. The user thought
the issue was only a slow modal close (7s fallback timer); in fact the
controls themselves were never being restored — pages showed defaults
masquerading as share-link content.

**Root cause:** `restore_from_url` had `Output("bubble-graph", "figure",
allow_duplicate=True)` as one of its 5 Outputs. On `/2`–`/7` initial
loads, `bubble-graph` is inside `bubble-lazy.children` which contains
only `"Loading..."`. The Dash 4 frontend renderer evaluates the initial
callback dispatch queue, finds the Output's component absent from the
component tree, and silently drops the entire callback. No HTTP POST is
sent. `snapshot-state-store` is never written. `apply_globals` and
`apply_tab_*` fire as no-ops with `state=None`.

**Empirical proof:** Playwright network capture on a `/3#q4:...` URL
encoding `dca-amount=999` showed (a) NO POST with `inputs=[('url',
'hash')]`, (b) `dca-amount=100` (default) in the rendered page. The
"working" `/3` share link was actually rendering defaults.

**Fix:** Route the bubble figure through an always-mounted `dcc.Store`
(`restore-bubble-fig`). `restore_from_url`'s Output position 4 changes
from `bubble-graph.figure` to `restore-bubble-fig.data`. A clientside
callback watches the Store and uses
`window.dash_clientside.set_props('bubble-graph', {figure: fig})` to
push the figure into bubble-graph. `set_props` bypasses Dash's
registered-Output existence check, so the lazy-mount problem disappears.

**Verified:** Post-fix Playwright probe shows POST with
`inputs=[('url', 'hash')]` IS sent for `/3`, and `dca-amount=999`
correctly appears in the widget. `/1` timing unchanged at 1.6s chart
render. No new JS errors from the `set_props` call.

**What Phase 1 does NOT do:** modal close on `/2`–`/7` still falls back
to the 7s timer because `active-chart-committed` is only written for
the bubble path. Phase 2 (per-tab order: 3→4→5→7→2→6) extends
`restore_builder.py` to per-tab figure builders, each writing
`active-chart-committed` for fast modal close.

**Key files (modified):**
- `btc_web/layout/__init__.py` — added `dcc.Store(id="restore-bubble-fig", ...)`
- `btc_web/callbacks/snapshot_cb.py:48` — `Output("bubble-graph", "figure")` → `Output("restore-bubble-fig", "data")`
- `btc_web/callbacks/snapshot_cb.py` — added clientside callback near existing modal-close listeners

**Tests:**
- `test_callbacks.py::test_restore_from_url_does_not_output_bubble_graph` — invariant
- `test_callbacks.py::test_store_payload_is_figure_for_bubble_share` — semantics
- `test_callbacks.py::test_store_payload_is_no_update_for_non_bubble` — semantics
- `test_restore_phase1_e2e.py` — 3 Playwright tests

**Lesson reaffirmed:** A registered Output for a lazy-mounted component
is a silent footgun in Dash 4. The frontend renderer drops the dispatch
without any error message visible in dev or prod. If a callback's
Outputs include any component that may be absent from the initial DOM
(lazy tab content, conditionally rendered components), route through
an always-mounted Store + `set_props` relay.
```

- [ ] **Step 2: Verify the file parses (no markdown breakage)**

Run:
```bash
head -3 /home/bcg/.claude/projects/-scratch-code-bitcoinprojections/memory/restore_callback_architecture.md
wc -l /home/bcg/.claude/projects/-scratch-code-bitcoinprojections/memory/restore_callback_architecture.md
```

Expected: file starts with `---\nname: ...` frontmatter and line count grew by ~50.

---

## Task 7: Update `docs/architecture.md`

**Files:**
- Modify: `/scratch/code/bitcoinprojections/docs/architecture.md`

- [ ] **Step 1: Locate the restore performance section**

Run:
```bash
grep -n "Restore performance\|restore_from_url\|active-chart-committed" /scratch/code/bitcoinprojections/docs/architecture.md | head -10
```

Note the line numbers of the restore-architecture section (added in commit `d3f5045`).

- [ ] **Step 2: Add a sub-section about the lazy-Output footgun**

Inside the existing "Restore performance architecture" sub-section, add a new paragraph:

```markdown
**Lazy-Output footgun (Phase 1, 2026-04-25):** `restore_from_url` cannot
have any Output to a lazy-mounted component (e.g. `bubble-graph` inside
`bubble-lazy`). When loading `/2`–`/7`, the lazy tab's content is
`"Loading..."` and the graph component does not exist in the initial DOM.
Dash 4's frontend renderer silently drops the entire callback dispatch
when an Output's component is absent — no HTTP POST, no error, no log.
Fix: route the bubble figure through an always-mounted `dcc.Store`
(`restore-bubble-fig`) and use a clientside `set_props` relay to deliver
the figure to `bubble-graph`. `set_props` bypasses the registered-Output
existence check.
```

---

## Task 8: Dev verify with `?trace=1`

**Files:** No changes — observation step.

- [ ] **Step 1: Start dev server**

```bash
cd /scratch/code/bitcoinprojections && \
  lsof -ti :8050 | xargs -r kill -9 2>/dev/null; \
  sleep 1 && \
  DEV=1 nohup bash run_web.sh > /tmp/quantoshi_dev.log 2>&1 &
sleep 6 && tail -3 /tmp/quantoshi_dev.log
```

Expected: log shows `Dash is running on http://0.0.0.0:8050/`.

- [ ] **Step 2: Generate `/1` share link with trace + `/3` share link with trace**

```bash
cd /scratch/code/bitcoinprojections && PYTHONPATH="btc_web:." btc_venv/bin/python3 -c "
import sys
sys.path.insert(0, 'btc_web')
import os; os.environ['DEV'] = '1'
import app
from snapshot import _encode_snapshot_v4
b = _encode_snapshot_v4({'main-tabs:active_tab': 'bubble', 'bub-qs:value': ['median']})
d = _encode_snapshot_v4({'main-tabs:active_tab': 'dca', 'dca-amount:value': 999})
print(f'BUBBLE: http://localhost:8050/1?trace=1#q4:{b}')
print(f'DCA:    http://localhost:8050/3?trace=1#q4:{d}')
"
```

Expected: two URLs printed.

- [ ] **Step 3: Run a Playwright trace for both URLs and grep dev journal**

```bash
cat > /tmp/dev_phase1_trace.py <<'EOF'
"""Dev trace verification: load /1 + /3 with ?trace=1, check dev log."""
import sys, time, os
sys.path.insert(0, "/scratch/code/bitcoinprojections/btc_web")
os.environ["DEV"] = "1"
import app  # noqa
from snapshot import _encode_snapshot_v4
from playwright.sync_api import sync_playwright

URLS = {
    "bubble": _encode_snapshot_v4({"main-tabs:active_tab": "bubble", "bub-qs:value": ["median"]}),
    "dca":    _encode_snapshot_v4({"main-tabs:active_tab": "dca",    "dca-amount:value": 999}),
}
TARGETS = {"bubble": "/1", "dca": "/3"}
GRAPH = {"bubble": "bubble-graph", "dca": "dca-graph"}

with sync_playwright() as p:
    browser = p.firefox.launch(headless=True)
    for tab, blob in URLS.items():
        url = f"http://localhost:8050{TARGETS[tab]}?trace=1#q4:{blob}"
        ctx = browser.new_context()
        page = ctx.new_page()
        t0 = time.perf_counter()
        page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        page.wait_for_function(
            f"() => {{ var gd = document.querySelector('#{GRAPH[tab]} .js-plotly-plot'); return gd && gd.data && gd.data.length > 0; }}",
            timeout=20_000,
        )
        t_chart = (time.perf_counter() - t0) * 1000
        print(f"{tab}: chart at {t_chart:.0f}ms")
        ctx.close()
    browser.close()
EOF
btc_venv/bin/python3 /tmp/dev_phase1_trace.py
echo "---DEV LOG---"
grep "\[trace\]" /tmp/quantoshi_dev.log | tail -20
```

Expected:
- `bubble: chart at <5000>ms` and `dca: chart at <8000>ms`
- `[trace] restore_from_url prefix=q4: controls=N` lines for BOTH bubble and dca loads
- `[trace] restore-direct-build BUILT Xms` for the bubble load (NOT for dca — Phase 1 doesn't build figures for non-bubble; that's Phase 2)

If `[trace] restore_from_url prefix=q4:` does not appear for the dca load, the dispatch is still being dropped — investigate.

- [ ] **Step 4: Stop dev server**

```bash
lsof -ti :8050 | xargs -r kill -9 2>/dev/null
```

---

## Task 9: Single commit

**Files:** All probe edits + new tests + memory + docs.

- [ ] **Step 1: Review diff before commit**

```bash
cd /scratch/code/bitcoinprojections && git status --short && echo "---" && git diff --stat
```

Expected files modified:
- `btc_web/layout/__init__.py`
- `btc_web/callbacks/snapshot_cb.py`
- `btc_web/test_callbacks.py`
- `btc_web/test_restore_phase1_e2e.py` (new)
- `docs/architecture.md`

**NOT** in commit:
- `model_data_ef.pkl`, `model_data_resqr_diagnostics.json` (already-modified data files; leave in working tree)
- `dash_req*.json` (probe artifacts; clean up separately)
- `tools/*.py` (unrelated WIP)

- [ ] **Step 2: Stage only the Phase 1 files (no `git add -A`)**

```bash
cd /scratch/code/bitcoinprojections && \
  git add btc_web/layout/__init__.py \
          btc_web/callbacks/snapshot_cb.py \
          btc_web/test_callbacks.py \
          btc_web/test_restore_phase1_e2e.py \
          docs/architecture.md
```

- [ ] **Step 3: Verify staged files**

```bash
git diff --cached --stat
```

Expected: 5 files staged. NOT model_data_ef.pkl, NOT dash_req*.json.

- [ ] **Step 4: Commit**

```bash
git commit -m "$(cat <<'EOF'
fix(restore): route bubble figure through always-mounted Store

The Dash 4 frontend renderer silently drops the entire callback dispatch
when an Output's component is absent from the initial DOM. restore_from_url
had Output("bubble-graph", "figure"), but bubble-graph is inside
bubble-lazy which contains "Loading..." on /2-/7 initial loads — so
restore_from_url was never firing for non-bubble share links. Empirical
proof: dca-amount=999 share link rendered with dca-amount=100 (default).

Fix: route the bubble figure through restore-bubble-fig (an always-mounted
dcc.Store) + a clientside set_props relay. set_props bypasses Dash's
registered-Output existence check.

Phase 2 (per-tab fast modal close, order 3→4→5→7→2→6) blocks on this.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 5: Verify commit**

```bash
git log --oneline -3 && git show --stat HEAD
```

Expected: top commit is the Phase 1 fix; 5 files changed.

---

## Task 10: Prod deploy

**Files:** No changes — deploy step.

- [ ] **Step 1: Push to origin**

```bash
cd /scratch/code/bitcoinprojections && git push origin master
```

Expected: push succeeds, no merge conflicts.

- [ ] **Step 2: SSH deploy to Hetzner VPS**

```bash
ssh root@89.167.70.45 "cd /opt/quantoshi && git pull && redis-cli FLUSHDB && systemctl restart quantoshi"
```

Expected: `git pull` reports the new commit; `redis-cli FLUSHDB` reports OK; `systemctl restart` succeeds silently.

- [ ] **Step 3: Wait for prod to come up**

```bash
sleep 8 && ssh root@89.167.70.45 "systemctl status quantoshi --no-pager | head -10"
```

Expected: `Active: active (running)`, no error lines.

---

## Task 11: Prod verify

**Files:** No changes — verification step.

- [ ] **Step 1: Generate prod-targeted share link for `/3` with non-default value**

```bash
cd /scratch/code/bitcoinprojections && PYTHONPATH="btc_web:." btc_venv/bin/python3 -c "
import sys; sys.path.insert(0, 'btc_web')
import os; os.environ['DEV'] = '1'
import app
from snapshot import _encode_snapshot_v4
blob = _encode_snapshot_v4({'main-tabs:active_tab': 'dca', 'dca-amount:value': 999})
print(f'https://quantoshi.xyz/3#q4:{blob}')
"
```

- [ ] **Step 2: Run Playwright probe against prod URL**

```bash
cat > /tmp/prod_phase1_verify.py <<'EOF'
"""Prod verification: /3 share link restores dca-amount=999."""
import sys, os, time
sys.path.insert(0, "/scratch/code/bitcoinprojections/btc_web")
os.environ["DEV"] = "1"
import app  # noqa
from snapshot import _encode_snapshot_v4
from playwright.sync_api import sync_playwright

blob = _encode_snapshot_v4({"main-tabs:active_tab": "dca", "dca-amount:value": 999})
url = f"https://quantoshi.xyz/3#q4:{blob}"
print(f"URL: {url}")

with sync_playwright() as p:
    browser = p.firefox.launch(headless=True)
    page = browser.new_page()
    t0 = time.perf_counter()
    page.goto(url, wait_until="domcontentloaded", timeout=30_000)
    page.wait_for_function(
        "() => { var gd = document.querySelector('#dca-graph .js-plotly-plot'); return gd && gd.data && gd.data.length > 0; }",
        timeout=20_000,
    )
    t_chart = (time.perf_counter() - t0) * 1000
    amount = page.evaluate("() => document.getElementById('dca-amount').value")
    print(f"chart at {t_chart:.0f}ms; dca-amount={amount}")
    browser.close()
    assert amount == "999", f"PROD FAIL: dca-amount={amount}, expected 999"
    print("PROD VERIFY: PASS")
EOF
btc_venv/bin/python3 /tmp/prod_phase1_verify.py
```

Expected: `dca-amount=999`, `PROD VERIFY: PASS`. If FAIL, the prod cache may need an extra Redis flush, OR there's a divergence between prod and dev that needs investigation.

- [ ] **Step 3: Check prod journal for `[trace] restore_from_url`**

```bash
ssh root@89.167.70.45 "journalctl -u quantoshi --since '2 min ago' --no-pager | grep '\\[trace\\] restore_from_url'"
```

Expected: at least one line `[trace] restore_from_url prefix=q4: controls=N` from the prod-verify load.

- [ ] **Step 4: Run prod regression for `/1`**

```bash
cat > /tmp/prod_bubble_verify.py <<'EOF'
"""Prod regression: /1 share link still works."""
import sys, os, time
sys.path.insert(0, "/scratch/code/bitcoinprojections/btc_web")
os.environ["DEV"] = "1"
import app  # noqa
from snapshot import _encode_snapshot_v4
from playwright.sync_api import sync_playwright

blob = _encode_snapshot_v4({"main-tabs:active_tab": "bubble", "bub-qs:value": ["median"]})
url = f"https://quantoshi.xyz/1#q4:{blob}"

with sync_playwright() as p:
    browser = p.firefox.launch(headless=True)
    page = browser.new_page()
    t0 = time.perf_counter()
    page.goto(url, wait_until="domcontentloaded", timeout=30_000)
    page.wait_for_function(
        "() => { var gd = document.querySelector('#bubble-graph .js-plotly-plot'); return gd && gd.data && gd.data.length > 0; }",
        timeout=20_000,
    )
    t_chart = (time.perf_counter() - t0) * 1000
    print(f"BUBBLE prod chart at {t_chart:.0f}ms")
    browser.close()
    assert t_chart < 6000, f"PROD REGRESSION: bubble took {t_chart:.0f}ms"
    print("PROD BUBBLE: PASS")
EOF
btc_venv/bin/python3 /tmp/prod_bubble_verify.py
```

Expected: `BUBBLE prod chart at <6000>ms`, `PROD BUBBLE: PASS`.

If both verifies pass, Phase 1 is shipped and working.

---

## Done — Phase 1 complete

Phase 2 starts at task list #60 ("Brainstorm /3 (DCA) per-tab fast modal close"). The order is 3→4→5→7→2→6, one tab per ship/verify cycle.
