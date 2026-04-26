# MC Spaghetti Fan on Tab 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render Markov MC simulation paths as a spaghetti fan on Tab 1 (Bubble), driven by a reused `_mc_controls("bub", ...)` panel placed after the Projection Quantiles card, with full payment parity to Tabs 3-5.

**Architecture:** Reuse existing MC plumbing — append `"bub"` to `_MC_TABS`, extend the 7 prefix loops in `mc_controls.py`, add a new `_get_mc_bubble_fig` wrapper through `_get_mc_or_cached`, extend `update_bubble` with MC Inputs/States and a new `_add_mc_spaghetti(fig, paths, t_axis, n_display=100)` rendering helper that produces 100 thin RdYlGn-graded lines. Snapshot defaults append at the end of the bubble section to minimize bit-drift. Tab 1 gains its first paywall surface (MC default off, opt-in).

**Tech Stack:** Plotly Dash 4.0.0, dash_bootstrap_components, plotly.graph_objects, BTCPay Greenfield API, matplotlib (RdYlGn colormap), Markov MC engine, Redis L0/L2 cache.

**Spec:** `docs/superpowers/specs/2026-04-26-mc-spaghetti-fan-tab1-design.md` (architect-reviewed; commit `5b6e8d0`)

---

## Task 0: Inspect existing precedents

Read these files before starting (skim, take notes — no edits):
- `btc_web/callbacks/mc_controls.py` — find every `for _mc_*` loop iterating `("dca", "ret", "hm", "sc", "cp")`. Audit notes for Task 5 should list each line number + variable name.
- `btc_web/callbacks/mc_payment.py` lines 25–145 — `_MC_TABS` and `_mc_payment_initiate` positional indexing.
- `btc_web/callbacks/charts/__init__.py` `update_dca` (lines ~990–1070) — canonical MC chart-callback pattern.
- `btc_web/utils.py` lines 145–185 — `_get_mc_or_cached` and `_get_dca_fig` precedent.
- `btc_web/figures/dca.py` — `build_dca_figure` signature returns `(fig, mc_result)` tuple.
- `btc_web/figures/bubble.py` — `build_bubble_figure` current signature.

No deliverable; just orientation.

---

## Task 1: Pin current snapshot fingerprint before editing defaults

**Files:**
- Run: `tools/update_defaults_registry.py`

- [ ] **Step 1: Verify clean working tree**

```bash
cd /scratch/code/bitcoinprojections
git status --short | grep -v '^??' | grep -v '^.M model_data' | grep -v 'diagnostics.json'
```

Expected: empty output (no tracked changes besides the always-dirty model_data files).

- [ ] **Step 2: Pin current fingerprint**

```bash
btc_venv/bin/python3 tools/update_defaults_registry.py
```

Expected: `fingerprint <8-char-hex> already in registry; no change` OR `appended fingerprint <hex>; registry now has N entries`. Either is fine — the goal is to ensure the current defaults are pinned before we change them.

---

## Task 2: Add `bub-mc-*` snapshot defaults

**Files:**
- Modify: `btc_web/snapshot_defaults.py` — alphabetical `bub-mc-*` keys

- [ ] **Step 1: Read existing dca-mc-* defaults to mirror**

```bash
grep "'dca-mc-" btc_web/snapshot_defaults.py
```

Expected output (these are the keys we mirror):

```
    'dca-mc-advanced:value': [],
    'dca-mc-amount:value': 100,
    'dca-mc-bins:value': 5,
    'dca-mc-enable:value': ['yes'],
    'dca-mc-entry-q:value': 10,
    'dca-mc-freq:value': 'Monthly',
    'dca-mc-infl:value': 4,
    'dca-mc-loaded:data': None,
    'dca-mc-model-src:value': 'bub',
    'dca-mc-regime:value': [0, 1, 2, 3, 4],
    'dca-mc-rendered-key:data': None,
    'dca-mc-results:data': None,
    'dca-mc-sims:value': 200,
    'dca-mc-start-yr:value': 2031,
    'dca-mc-stack:value': 1.0,
    'dca-mc-unblocked:data': None,
    'dca-mc-window:value': [2010, 2026],
    'dca-mc-years:value': 40,
```

- [ ] **Step 2: Add `bub-mc-*` keys to snapshot_defaults.py**

Locate the alphabetical insertion point (after `'bub-legend-pos:value'` and before `'bub-model-show:value'`). Add these 18 keys mirroring the dca-mc-* set, with one critical default change: **`bub-mc-enable:value` is `[]` (off by default)** to preserve Tab 1's clean first-paint UX:

```python
    'bub-mc-advanced:value': [],
    'bub-mc-amount:value': 100,
    'bub-mc-bins:value': 5,
    'bub-mc-enable:value': [],
    'bub-mc-entry-q:value': 10,
    'bub-mc-freq:value': 'Monthly',
    'bub-mc-infl:value': 4,
    'bub-mc-loaded:data': None,
    'bub-mc-model-src:value': 'bub',
    'bub-mc-regime:value': [0, 1, 2, 3, 4],
    'bub-mc-rendered-key:data': None,
    'bub-mc-results:data': None,
    'bub-mc-sims:value': 200,
    'bub-mc-start-yr:value': 2031,
    'bub-mc-stack:value': 1.0,
    'bub-mc-unblocked:data': None,
    'bub-mc-window:value': [2010, 2026],
    'bub-mc-years:value': 40,
```

- [ ] **Step 3: Commit incremental progress**

Don't commit yet — Tasks 2–4 form one snapshot-defaults bundle. Continue.

---

## Task 3: Add `bub-mc-*` to `_SNAPSHOT_CONTROLS` and `_TAB_CONTROLS["bubble"]`

**Files:**
- Modify: `btc_web/snapshot.py` — append at END of bubble section
- Modify: `btc_web/callbacks/routing.py` — add to `_TAB_CONTROLS["bubble"]` set

- [ ] **Step 1: Locate the end of the bubble section in `_SNAPSHOT_CONTROLS`**

```bash
grep -n '("bub-\|("scan-\|("hm-' btc_web/snapshot.py | head -25
```

Find the last `("bub-..."`, "value")` tuple before the heatmap section starts.

- [ ] **Step 2: Append `bub-mc-*` tuples in `snapshot.py`**

Right before the heatmap section starts, append these 18 tuples (positional order matches the snapshot_defaults additions; bit positions for downstream entries shift uniformly):

```python
    # ── Tab 1 (Bubble) MC controls — appended at END of bubble section to
    # minimize positional bit-drift for the dca/ret/sc/hm sections that follow.
    ("bub-mc-advanced",     "value"),
    ("bub-mc-amount",       "value"),
    ("bub-mc-bins",         "value"),
    ("bub-mc-enable",       "value"),
    ("bub-mc-entry-q",      "value"),
    ("bub-mc-freq",         "value"),
    ("bub-mc-infl",         "value"),
    ("bub-mc-loaded",       "data"),
    ("bub-mc-model-src",    "value"),
    ("bub-mc-regime",       "value"),
    ("bub-mc-rendered-key", "data"),
    ("bub-mc-results",      "data"),
    ("bub-mc-sims",         "value"),
    ("bub-mc-start-yr",     "value"),
    ("bub-mc-stack",        "value"),
    ("bub-mc-unblocked",    "data"),
    ("bub-mc-window",       "value"),
    ("bub-mc-years",        "value"),
```

- [ ] **Step 3: Add `bub-mc-*` to `_TAB_CONTROLS["bubble"]` in routing.py**

Locate the bubble entry in `_TAB_CONTROLS` (around line 138). Append the bub-mc-* component IDs:

```python
    "bubble": {"bub-qs","bub-qs-adv","bub-toggles","bub-bubble-toggles",
               # ... existing entries ...
               # MC controls appended 2026-04-26
               "bub-mc-advanced","bub-mc-amount","bub-mc-bins","bub-mc-enable",
               "bub-mc-entry-q","bub-mc-freq","bub-mc-infl","bub-mc-loaded",
               "bub-mc-model-src","bub-mc-regime","bub-mc-rendered-key",
               "bub-mc-results","bub-mc-sims","bub-mc-start-yr","bub-mc-stack",
               "bub-mc-unblocked","bub-mc-window","bub-mc-years"},
```

- [ ] **Step 4: Continue (no commit yet — Task 4 finishes the snapshot bundle)**

---

## Task 4: Add MC keys to `BUBBLE` dict + `bubble_defaults()`

**Files:**
- Modify: `btc_web/tab_defaults.py` — `BUBBLE` dict + `bubble_defaults()`

- [ ] **Step 1: Read existing DCA pattern**

```bash
grep -n "dca-mc-\|DCA\[" btc_web/tab_defaults.py | head -20
```

DCA's pattern: `_build_dca_dict()` reads `sd("dca-mc-*:value", default)` to build the figure params dict. `bubble_defaults()` builds the prewarm cache key. Both must match the runtime callback's params dict in `update_bubble`.

- [ ] **Step 2: Add MC keys to `BUBBLE` dict**

Locate `_build_bubble_dict()` (around line 22). Inside the returned dict, append:

```python
        # MC controls — prewarm-cache parity with update_bubble runtime
        "mc_enabled":     bool(sd("bub-mc-enable:value", []) or []),
        "mc_amount":      sd("bub-mc-amount:value", 100),
        "mc_bins":        sd("bub-mc-bins:value", 5),
        "mc_entry_q":     sd("bub-mc-entry-q:value", 10),
        "mc_freq":        sd("bub-mc-freq:value", "Monthly"),
        "mc_infl":        sd("bub-mc-infl:value", 4),
        "mc_model_src":   sd("bub-mc-model-src:value", "bub"),
        "mc_regime":      list(sd("bub-mc-regime:value", [0, 1, 2, 3, 4]) or []),
        "mc_sims":        sd("bub-mc-sims:value", 200),
        "mc_start_yr":    sd("bub-mc-start-yr:value", 2031),
        "mc_window":      list(sd("bub-mc-window:value", [2010, 2026]) or []),
        "mc_years":       sd("bub-mc-years:value", 40),
```

- [ ] **Step 3: Verify `bubble_defaults()` exposes the same keys**

```bash
grep -n "def bubble_defaults" btc_web/tab_defaults.py
```

If `bubble_defaults()` returns `BUBBLE` directly (or builds from it), no further change. If it strips keys, mirror the same MC additions.

- [ ] **Step 4: Pin the new fingerprint**

```bash
btc_venv/bin/python3 tools/update_defaults_registry.py
```

Expected: `appended fingerprint <new-hex>; registry now has N+1 entries`.

- [ ] **Step 5: Run snapshot-defaults consistency tests**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_snapshot.py -v -k 'defaults' 2>&1 | tail -15
```

Expected: ALL pass. The tests cross-check `SNAPSHOT_DEFAULTS` ↔ widget defaults ↔ `_SNAPSHOT_CONTROLS`. If any fail, the diagnostic message names the divergent key — fix and re-run.

- [ ] **Step 6: Commit the snapshot-defaults bundle**

```bash
git add btc_web/snapshot_defaults.py btc_web/snapshot.py btc_web/callbacks/routing.py btc_web/tab_defaults.py btc_web/snapshot_defaults_registry.json
git commit -m "feat(mc-tab1): snapshot defaults + controls list for bub-mc-* (Task 2-4)

18 new bub-mc-* keys appended at end of bubble section to minimize
bit-drift. Old + new fingerprints both pinned in the registry.
Default mc-enable=[] preserves Tab 1's opt-in UX."
```

---

## Task 5: Add `"bub"` to all 7 prefix loops in `mc_controls.py`

**Files:**
- Modify: `btc_web/callbacks/mc_controls.py` — 7 explicit prefix iterations

- [ ] **Step 1: Locate every loop iterating MC-tab prefixes**

```bash
grep -n 'for _mc_\|for _cost_\|for _path_\|"dca", "ret"\|("dca", "ret"' btc_web/callbacks/mc_controls.py
```

Expected: 7 hits. Note each line number.

- [ ] **Step 2: Add `"bub"` to each loop's tuple**

Identify each location precisely from Step 1, then for each occurrence, append `"bub"` to the prefix tuple. Examples (exact tuples may vary slightly — match what the file has):

| Location | Original tuple | New tuple |
|---|---|---|
| Body-toggle loop | `("dca", "ret", "hm", "sc", "cp")` | `("dca", "ret", "hm", "sc", "cp", "bub")` |
| Display-Models-MC-injection | `("dca", "ret", "sc")` | `("dca", "ret", "sc", "bub")` |
| Advanced-toggle | `("dca", "ret", "hm", "sc", "cp")` | `("dca", "ret", "hm", "sc", "cp", "bub")` |
| Regime-options | `("dca", "ret", "hm", "sc", "cp")` | `("dca", "ret", "hm", "sc", "cp", "bub")` |
| `_MC_MATCH_JS_TPL` | `("dca", "ret", "hm", "sc", "cp")` | `("dca", "ret", "hm", "sc", "cp", "bub")` |
| Restore button loop | per file | append `"bub"` |
| `_cost_pfx` MC cost | `("dca", "ret", "hm", "sc", "cp")` | `("dca", "ret", "hm", "sc", "cp", "bub")` |

- [ ] **Step 3: Verify import smoke**

```bash
PYTHONPATH=btc_web:. btc_venv/bin/python3 -c "import sys; sys.path.insert(0, 'btc_web'); import os; os.environ['DEV']='1'; import app; print('OK')"
```

Expected: `[resqr] bound 87 model bundles  _HAS_RESQR=True` then `OK`. Any error means a callback registration failed (likely a missing `bub-mc-*` ID in layout — handled in Task 6).

- [ ] **Step 4: Don't commit yet — needs layout (Task 6) before clean smoke**

---

## Task 6: Add `_mc_controls("bub", ...)` to bubble layout

**Files:**
- Modify: `btc_web/layout/bubble.py` — insert after Projection Quantiles card

- [ ] **Step 1: Locate the Projection Quantiles section in `_bubble_controls()`**

```bash
grep -n "Projection Quantiles\|q_panel_with_mode\|bub-qs\b" btc_web/layout/bubble.py
```

The Projection Quantiles card is built via `_q_panel_with_mode` or a direct `_section_card` invocation. Find the line after which it ends.

- [ ] **Step 2: Insert MC controls call**

Add the import at the top of the file if not already present:

```python
from layout.mc_controls import _mc_controls
```

Then after the Projection Quantiles card, insert:

```python
        # Tab-1 MC: opt-in (default off). Reuses the panel from Tabs 3-5.
        # show_amount/show_inflation/show_stack=False because Tab 1 is
        # price-space (no withdrawal amount, no inflation, no accumulation).
        _mc_controls("bub",
                     show_amount=False,
                     show_inflation=False,
                     show_stack=False,
                     show_mc_entry_q=True,
                     default_entry_q=10,
                     amount_default=100,
                     mc_enabled_default=False),
```

- [ ] **Step 3: Verify import smoke**

```bash
PYTHONPATH=btc_web:. btc_venv/bin/python3 -c "import sys; sys.path.insert(0, 'btc_web'); import os; os.environ['DEV']='1'; import app; print('OK')"
```

Expected: `OK`. If any "nonexistent object" or "Component not found" error → a `bub-mc-*` ID is referenced by a callback but missing from layout. Cross-check Task 5's loop additions vs this Task's controls.

- [ ] **Step 4: Commit Tasks 5+6 together**

```bash
git add btc_web/callbacks/mc_controls.py btc_web/layout/bubble.py
git commit -m "feat(mc-tab1): wire MC controls panel into Tab 1 layout (Task 5-6)

- 7 prefix loops in mc_controls.py extended to include 'bub'
- _mc_controls('bub', show_amount=False, show_inflation=False,
  show_stack=False) inserted after Projection Quantiles card"
```

---

## Task 7: Append `"bub"` to `_MC_TABS` and 5 new States in `mc_payment.py`

**Files:**
- Modify: `btc_web/callbacks/mc_payment.py` — payment callback

- [ ] **Step 1: Append `"bub"` at the END of `_MC_TABS`**

In `mc_payment.py` near line 25:

```python
_MC_TABS = ("dca", "ret", "hm", "sc", "cp", "bub")
```

**CRITICAL:** Must be at the END. Inserting in the middle shifts the positional `tab_idx * 3` arithmetic and breaks payment for all tabs (this is the e634b54 → 7820a84 footgun).

- [ ] **Step 2: Append 5 new States to the callback's State block**

Find the `State("dca-mc-years", ...)` ... `State("cp-mc-entry-q", "value")` block (around lines 44–53). Add the bub triplet AFTER the cp triplet:

```python
    State("cp-mc-years", "value"), State("cp-mc-start-yr", "value"),
    State("cp-mc-entry-q", "value"),
    State("bub-mc-years", "value"), State("bub-mc-start-yr", "value"),
    State("bub-mc-entry-q", "value"),
    State("mc-pay-trigger", "data"),
```

The model-src and price-val States are generated via `*(State(...) for pfx in _MC_TABS ...)` comprehensions — they auto-extend because they iterate `_MC_TABS` (now 6-tuple). No code change needed there.

- [ ] **Step 3: Smoke-import**

```bash
PYTHONPATH=btc_web:. btc_venv/bin/python3 -c "import sys; sys.path.insert(0, 'btc_web'); import os; os.environ['DEV']='1'; import app; print('OK')"
```

Expected: `OK`. If error mentions `bub-mc-price-val`, that Store may not exist in the layout — verify `_mc_controls("bub", ...)` from Task 6 created it.

- [ ] **Step 4: Commit**

```bash
git add btc_web/callbacks/mc_payment.py
git commit -m "feat(mc-tab1): append 'bub' to _MC_TABS payment callback (Task 7)

Appended at END of _MC_TABS to preserve positional tab_idx*3
arithmetic in _mc_payment_initiate (avoids the e634b54 footgun)."
```

---

## Task 8: New `_get_mc_bubble_fig` wrapper in `utils.py`

**Files:**
- Modify: `btc_web/utils.py` — new function parallel to `_get_dca_fig`

- [ ] **Step 1: Locate the existing wrappers**

```bash
grep -n "_get_dca_fig\|_get_retire_fig\|_get_bubble_fig\|_get_mc_or_cached" btc_web/utils.py
```

`_get_dca_fig` (line ~177) is the precedent: 1-line wrapper through `_get_mc_or_cached`.

- [ ] **Step 2: Add `_get_mc_bubble_fig`**

Right after the existing `_get_bubble_fig` definition (or near the other `_get_*_fig` wrappers — match the local convention), add:

```python
def _get_mc_bubble_fig(p: dict):
    """MC-aware bubble cache wrapper. Routes through _get_mc_or_cached
    so mc_cached dicts are stripped from the JSON cache key when MC is
    disabled and passed directly to the builder when MC is enabled.
    update_bubble switches to this wrapper only when mc_enabled is truthy;
    the non-MC fast path stays on _get_bubble_fig."""
    return _get_mc_or_cached(p, build_bubble_figure, _cached_bubble_fig)
```

- [ ] **Step 3: Verify symbol presence**

```bash
btc_venv/bin/python3 -c "import sys, os; sys.path.insert(0, 'btc_web'); os.environ['DEV']='1'; import app; from utils import _get_mc_bubble_fig; print(_get_mc_bubble_fig)"
```

Expected: `<function _get_mc_bubble_fig at 0x...>`.

- [ ] **Step 4: Don't commit yet — Task 9 may require build_bubble_figure signature update**

---

## Task 9: Confirm `build_bubble_figure` returns `(fig, mc_result)` tuple

**Files:**
- Modify (conditionally): `btc_web/figures/bubble.py` — `build_bubble_figure` signature

- [ ] **Step 1: Check current signature**

```bash
grep -n "def build_bubble_figure\|def build_dca_figure" btc_web/figures/bubble.py btc_web/figures/dca.py
```

`_get_mc_or_cached` (utils.py:149) calls `builder_fn(_app_ctx.M, p_q)` and stores its return as the figure. `build_dca_figure` returns `(fig, mc_result)`. If `build_bubble_figure` currently returns just `go.Figure`, we need to update it.

- [ ] **Step 2: If signature differs, update `build_bubble_figure`**

If current signature is `def build_bubble_figure(m, p) -> go.Figure:`, change the body to:

```python
def build_bubble_figure(m: ModelData, p: dict[str, Any]) -> tuple[go.Figure, dict | None]:
    # ... existing body ...
    return fig, None   # No MC overlay path on bubble yet — the spaghetti
                       # branch in update_bubble adds it post-build.
                       # Returning None as mc_result keeps the (fig, mc) tuple
                       # contract _get_mc_or_cached expects.
```

Update every internal `return fig` to `return fig, None`. There may be early-exit paths.

- [ ] **Step 3: Update `_get_bubble_fig` consumers if signature changed**

The existing `_get_bubble_fig` is called from many callsites. If `build_bubble_figure` now returns a tuple, `_cached_bubble_fig` (which calls `build_bubble_figure(_app_ctx.M, p)`) returns a tuple too. Audit consumers:

```bash
grep -n "_get_bubble_fig(" btc_web/ -r --include='*.py' | head
```

For each call site, the previous `fig = _get_bubble_fig(p)` becomes `fig, _ = _get_bubble_fig(p)` OR `result = _get_bubble_fig(p); fig = result[0] if isinstance(result, tuple) else result`.

- [ ] **Step 4: Run all bubble-related tests**

```bash
btc_venv/bin/python3 -m pytest btc_web/ -k bubble -v --ignore-glob='*_e2e.py' 2>&1 | tail -20
```

Expected: all pass. If new failures show "tuple object has no attribute 'data'", fix the call sites.

- [ ] **Step 5: Commit Tasks 8+9 together**

```bash
git add btc_web/utils.py btc_web/figures/bubble.py
git commit -m "feat(mc-tab1): _get_mc_bubble_fig wrapper + tuple return contract (Task 8-9)

build_bubble_figure now returns (fig, mc_result) like build_dca_figure
so _get_mc_or_cached's contract is honored. Non-MC consumers updated
to unpack the tuple."
```

---

## Task 10: `_add_mc_spaghetti` rendering helper (TDD)

**Files:**
- Create test: `btc_web/test_figures.py` — new test in existing module
- Modify: `btc_web/figures/bubble.py` — add `_add_mc_spaghetti`

- [ ] **Step 1: Write failing test for trace count**

Append to `btc_web/test_figures.py`:

```python
def test_add_mc_spaghetti_returns_n_traces():
    """100 sample paths from a 2000-path array, deterministic stride."""
    from figures.bubble import _add_mc_spaghetti
    import numpy as np
    import plotly.graph_objects as go

    fig = go.Figure()
    n_sims, n_steps = 2000, 60
    rng = np.random.default_rng(0)
    paths = np.cumsum(rng.normal(0, 0.05, size=(n_sims, n_steps)), axis=1)
    t_axis = np.arange(n_steps)

    n_initial = len(fig.data)
    _add_mc_spaghetti(fig, paths, t_axis, n_display=100)
    n_final = len(fig.data)
    assert n_final - n_initial == 100, f"expected 100 new traces, got {n_final - n_initial}"


def test_add_mc_spaghetti_color_gradient_terminal_order():
    """RdYlGn cmap: lowest terminal = red, highest terminal = green."""
    from figures.bubble import _add_mc_spaghetti
    import numpy as np
    import plotly.graph_objects as go

    fig = go.Figure()
    # 3 sims with monotone-increasing terminals
    paths = np.array([
        [0, 0, 0, 0],   # lowest final
        [0, 1, 2, 3],
        [0, 2, 4, 6],   # highest final
    ])
    t_axis = np.arange(4)
    _add_mc_spaghetti(fig, paths, t_axis, n_display=3)
    # Plotly stores rgba string in line.color
    colors = [t.line.color for t in fig.data]
    assert len(colors) == 3
    # Red component decreases (red→green); first trace is reddest
    def red(s): return int(s.split("(")[1].split(",")[0])
    assert red(colors[0]) > red(colors[-1]), \
        f"first trace should be redder than last: {colors}"


def test_add_mc_spaghetti_handles_empty_paths():
    """None / empty array → no-op, no exception."""
    from figures.bubble import _add_mc_spaghetti
    import plotly.graph_objects as go
    import numpy as np

    fig = go.Figure()
    _add_mc_spaghetti(fig, None, np.arange(5))
    _add_mc_spaghetti(fig, np.array([]).reshape(0, 5), np.arange(5))
    assert len(fig.data) == 0
```

- [ ] **Step 2: Verify tests fail (function doesn't exist)**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_figures.py -k spaghetti -v 2>&1 | tail -10
```

Expected: 3 FAILED, all with `ImportError` or `AttributeError` because `_add_mc_spaghetti` doesn't exist.

- [ ] **Step 3: Implement `_add_mc_spaghetti`**

In `btc_web/figures/bubble.py`, after the existing helpers, add:

```python
import matplotlib.cm as _mpl_cm  # Top of file if not already imported

def _add_mc_spaghetti(fig, paths, t_axis, n_display=100):
    """Add up to n_display sample paths from a (n_sims, n_steps) array.

    Subsamples deterministically (stride-based) for reproducibility, then
    color-grades each path by its terminal value using the RdYlGn cmap
    (lowest = red, highest = green). Each trace has line.width=0.6 and
    hoverinfo='skip' so the underlying price scatter remains the
    authoritative hover target.

    Args:
        fig: go.Figure to mutate.
        paths: np.ndarray (n_sims, n_steps), or None / 0-row → no-op.
        t_axis: np.ndarray (n_steps,) — x values matching paths' columns.
        n_display: target trace count. Stride = n_sims // n_display.
    """
    import numpy as np
    import plotly.graph_objects as go

    if paths is None or paths.size == 0 or paths.shape[0] == 0:
        return
    stride = max(1, paths.shape[0] // n_display)
    sample = paths[::stride][:n_display]
    finals = sample[:, -1]
    span = max(np.ptp(finals), 1e-12)
    norm = (finals - finals.min()) / span
    cmap = _mpl_cm.RdYlGn
    for i, path in enumerate(sample):
        rgba = cmap(float(norm[i]))
        color = f"rgba({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)},0.45)"
        fig.add_trace(go.Scatter(
            x=t_axis, y=path,
            mode="lines",
            line=dict(color=color, width=0.6),
            showlegend=(i == 0),
            name=("MC paths" if i == 0 else None),
            hoverinfo="skip",
            legendgroup="mc-spaghetti",
        ))
```

- [ ] **Step 4: Verify tests pass**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_figures.py -k spaghetti -v 2>&1 | tail -10
```

Expected: 3 PASSED.

- [ ] **Step 5: Commit**

```bash
git add btc_web/figures/bubble.py btc_web/test_figures.py
git commit -m "feat(mc-tab1): _add_mc_spaghetti helper + 3 unit tests (Task 10)

100 RdYlGn-graded thin lines, deterministic stride for reproducibility,
hoverinfo=skip so price scatter remains the authoritative hover target."
```

---

## Task 11: Extend `update_bubble` callback with MC plumbing

**Files:**
- Modify: `btc_web/callbacks/charts/__init__.py` — `update_bubble` callback signature, body, post-restore triggers

- [ ] **Step 1: Add MC Inputs/States to `@callback` decorator**

After the existing `lppl-no-13` Input (around line 100), insert these MC Inputs (mirroring `update_dca`'s shape):

```python
    # ── MC controls (Tab 1 added 2026-04-26) ──
    Input("bub-mc-enable",  "value"),
    Input("bub-mc-bins",    "value"),
    Input("bub-mc-regime",  "value"),
    Input("bub-mc-sims",    "value"),
    Input("bub-mc-years",   "value"),
    Input("bub-mc-window",  "value"),
    Input("bub-mc-start-yr", "value"),
    Input("bub-mc-entry-q",  "value"),
    Input("bub-mc-loaded",   "data"),
    Input("bub-mc-model-src", "value"),
    State("mc-pay-trigger",  "data"),
    State("bub-mc-results",  "data"),
    State("bub-mc-rendered-key", "data"),
    State("mc-pay-token",    "data"),
    State("bub-mc-unblocked", "data"),
    State("btc-price-store", "data"),
```

Update the `@callback` Output list to include:

```python
    Output("bub-mc-results",     "data",       allow_duplicate=True),
    Output("bub-mc-rendered-key", "data",      allow_duplicate=True),
    Output("bub-mc-unblocked",   "data",       allow_duplicate=True),
```

- [ ] **Step 2: Add new params to function signature**

Append to `def update_bubble(...)` parameter list (positional order matching the @callback order):

```python
                  mc_enable, mc_bins, mc_regime, mc_sims, mc_years,
                  mc_window, mc_start_yr, mc_entry_q, _mc_loaded, mc_model_src,
                  pay_trigger, mc_cached, mc_auth, pay_token, mc_unblocked,
                  price_data, ...):  # rest of existing params unchanged
```

- [ ] **Step 3: Add bub-mc-* IDs to `_POST_RESTORE_TRIGGERS`**

Locate `_POST_RESTORE_TRIGGERS = {...}` (around line 192). Add:

```python
        "bub-mc-enable", "bub-mc-bins", "bub-mc-regime", "bub-mc-sims",
        "bub-mc-years", "bub-mc-window", "bub-mc-start-yr", "bub-mc-entry-q",
        "bub-mc-model-src",
```

(`bub-mc-loaded` is async-completion driven and intentionally omitted, mirroring DCA's `dca-mc-loaded` exclusion.)

- [ ] **Step 4: Add MC computation block before figure build**

Before the existing `fig = _get_bubble_fig(...)` call, add:

```python
    # ── MC setup (Tab 1) ──
    from callbacks.mc_helpers import _mc_setup, _mc_finalize
    mc_model_src = _resolve_mc_model_src(
        mc_model_src,
        lppl_n_freqs, lppl_weighted, lppl_no_13,
        hyb_a_nlog, hyb_a_ncal, hyb_a_log1d, hyb_a_log2d,
        hyb_a_cal1d, hyb_a_cal2d,
        ep_a_nlog, ep_a_ncal, ep_a_log1d, ep_a_log2d,
        ep_a_cal1d, ep_a_cal2d)
    mc_ok, is_free, mc_p, blocked = _mc_setup(
        "bub", mc_enable, mc_years, mc_start_yr, mc_entry_q,
        mc_bins, mc_sims, "Monthly",  # bub has no freq picker; default Monthly
        mc_window, 100, 0,             # bub has no amount/inflation; defaults
        mc_cached, _cf(price_data, 0), mc_regime, mc_unblocked, pay_token,
        mc_auth=mc_auth,
        stack=None, amount_default=100, infl_default=0.0,
        start_yr_default=2031,
        mc_model_src=mc_model_src or "bub")
    mc_visible = bool(mc_enable) and "yes" in (mc_enable or [])
```

- [ ] **Step 5: Switch figure builder based on `mc_enabled`**

Replace the existing `fig = _get_bubble_fig(params)` call. The non-MC path stays as is; the MC path goes through the new wrapper:

```python
    from utils import _get_mc_bubble_fig
    if mc_visible and mc_ok:
        params = dict(params, **mc_p)  # merge mc_* keys
        result = _get_mc_bubble_fig(params)
        fig, mc_result = result if isinstance(result, tuple) else (result, None)
    else:
        result = _get_bubble_fig(params)
        fig, _ = result if isinstance(result, tuple) else (result, None)
        mc_result = None
```

- [ ] **Step 6: Add spaghetti render after figure build**

Right after the figure is built but before the function returns, add:

```python
    # MC spaghetti fan — paths sourced from mc_p (cached) or mc_result (live)
    if mc_visible and mc_ok:
        from figures.bubble import _add_mc_spaghetti
        from mc_cache import get_cached_paths
        paths = None
        if is_free:
            paths = get_cached_paths(
                mc_p["mc_model_src"], mc_p["mc_start_yr"],
                _ci(mc_p["mc_entry_q"], 10) / 100.0,
                mc_p["mc_years"])
        elif mc_result and isinstance(mc_result, dict):
            paths = mc_result.get("price_paths")
        if paths is not None and len(paths) > 0:
            # t_axis: use the figure's existing x-axis
            import numpy as np
            n_steps = paths.shape[1] if hasattr(paths, "shape") else len(paths[0])
            t_start = mc_p["mc_start_yr"]
            t_axis = np.linspace(t_start, t_start + mc_p["mc_years"], n_steps)
            _add_mc_spaghetti(fig, paths, t_axis, n_display=100)
```

- [ ] **Step 7: Update return tuple with new outputs**

The existing `return fig` becomes (matching the new Output list):

```python
    fig, store_val, status, rendered_key, show_modal, ub_val = _mc_finalize(
        "bub", fig, mc_result, mc_cached, mc_enable, mc_ok,
        is_free, blocked, mc_p["mc_years"], mc_p["mc_start_yr"],
        mc_p["mc_entry_q"], toggles, mc_stale=mc_p.get("mc_stale", False),
        mc_p=mc_p)
    return fig, store_val, rendered_key, ub_val
```

(Adjust output count to match the @callback Output decorator from Step 1.)

- [ ] **Step 8: Run unit tests**

```bash
btc_venv/bin/python3 -m pytest btc_web/ --ignore-glob='*_e2e.py' -q 2>&1 | tail -8
```

Expected: ≥1543 passed, the pre-existing `test_no_hex_literals_outside_colors_module` failure persists (unrelated). Any new failures need fixing inline.

- [ ] **Step 9: Commit**

```bash
git add btc_web/callbacks/charts/__init__.py
git commit -m "feat(mc-tab1): extend update_bubble with MC plumbing + spaghetti render (Task 11)

- 13 MC Inputs/States added; 3 new Outputs
- _resolve_mc_model_src + _mc_setup before figure build
- _get_mc_bubble_fig branch when mc_visible and mc_ok
- _add_mc_spaghetti called with cached/live path arrays
- All new bub-mc-* IDs added to _POST_RESTORE_TRIGGERS"
```

---

## Task 12: Gate `_build_bubble_figure_from_state` on `bub-mc-enable`

**Files:**
- Modify: `btc_web/restore_builder.py` — gate at top of bubble fast-path builder

- [ ] **Step 1: Locate the existing CTA gate**

```bash
grep -n "cta-active\|_build_bubble_figure_from_state" btc_web/restore_builder.py
```

The CTA gate (`if "yes" in _v(state, "cta-active") return None`) is the model — we add a parallel gate.

- [ ] **Step 2: Add MC gate**

Right after the CTA gate in `_build_bubble_figure_from_state`, add:

```python
    # MC gate (Tab 1, 2026-04-26): when MC is enabled in the snapshot
    # state, the fast restore path can't render the spaghetti fan
    # (it has no cfg-modal state, no mc_cached, no payment token).
    # Punt to cascade — update_bubble re-fires after the apply_globals
    # cascade and renders MC properly.
    if "yes" in (_v(state, "bub-mc-enable") or []):
        return None
```

- [ ] **Step 3: Add unit test**

In `btc_web/test_restore_builder.py`, append:

```python
def test_bubble_mc_enabled_returns_none():
    """bub-mc-enable=['yes'] in state → fall back to cascade."""
    from restore_builder import _build_bubble_figure_from_state
    state = {"bub-mc-enable:value": ["yes"]}
    assert _build_bubble_figure_from_state(state) is None
```

- [ ] **Step 4: Run test**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_restore_builder.py::test_bubble_mc_enabled_returns_none -v 2>&1 | tail -5
```

Expected: PASSED.

- [ ] **Step 5: Run full restore_builder test file**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_restore_builder.py -v 2>&1 | tail -10
```

Expected: all PASSED.

- [ ] **Step 6: Commit**

```bash
git add btc_web/restore_builder.py btc_web/test_restore_builder.py
git commit -m "feat(mc-tab1): gate _build_bubble_figure_from_state on MC (Task 12)

When bub-mc-enable=['yes'] in the snapshot state, fast restore returns
None and falls back to the chart-callback cascade (which has the
cfg-modal state and payment token to render MC properly)."
```

---

## Task 13: Full unit suite + cold-cache benchmark

**Files:**
- (No code changes; verification only)

- [ ] **Step 1: Run full unit suite**

```bash
btc_venv/bin/python3 -m pytest btc_web/ --ignore-glob='*_e2e.py' -q 2>&1 | tail -8
```

Expected: 1545+ passed (1544 baseline + new spaghetti tests + new MC gate test). The `test_no_hex_literals_outside_colors_module` pre-existing failure is the only acceptable failure.

- [ ] **Step 2: Restart dev server**

```bash
lsof -ti :8050 | xargs --no-run-if-empty kill -9 2>&1
DEV=1 nohup bash run_web.sh > /tmp/quantoshi_dev.log 2>&1 &
disown
sleep 6
echo "PID: $(lsof -ti :8050)"
```

- [ ] **Step 3: Cold-cache benchmark with `?trace=1`**

Save to `/tmp/bench_mc_tab1.py`:

```python
import sys, time, os
sys.path.insert(0, 'btc_web')
os.environ.setdefault('DEV', '1')
import app
from snapshot import _encode_snapshot_v4
from playwright.sync_api import sync_playwright

# Cached params (must hit free-tier cache: bub × 2031 × 40yr × q=10)
state = {
    'main-tabs:active_tab': 'bubble',
    'bub-mc-enable:value': ['yes'],
    'bub-mc-model-src:value': 'bub',
    'bub-mc-start-yr:value': 2031,
    'bub-mc-years:value': 40,
    'bub-mc-entry-q:value': 10,
}
url = f'http://localhost:8050/1?trace=1#q4:{_encode_snapshot_v4(state)}'

with sync_playwright() as p:
    b = p.firefox.launch(headless=True)
    pg = b.new_page()
    t0 = time.perf_counter()
    pg.goto(url, wait_until='domcontentloaded', timeout=30_000)
    pg.wait_for_function(
        "() => { var gd = document.querySelector('#bubble-graph .js-plotly-plot'); "
        "return gd && gd.data && gd.data.length > 30; }",  # 30+ traces means MC fan present
        timeout=20_000,
    )
    dt = (time.perf_counter() - t0) * 1000
    n_traces = pg.evaluate("() => document.querySelector('#bubble-graph .js-plotly-plot').data.length")
    b.close()
print(f'Total chart paint: {dt:.0f}ms')
print(f'Traces: {n_traces}')
```

Run:

```bash
btc_venv/bin/python3 /tmp/bench_mc_tab1.py
```

- [ ] **Step 4: Check journal for `[trace] BUILT` line**

```bash
grep -E "BUILT|spaghetti" /tmp/quantoshi_dev.log | tail -5
```

Expected: `[trace] update_bubble BUILT <X>ms`. **If X > 1500ms**, reduce `n_display` from 100 to 50 in `_add_mc_spaghetti` callsite (Task 11 Step 6) and re-run.

- [ ] **Step 5: Kill dev server**

```bash
lsof -ti :8050 | xargs --no-run-if-empty kill -9
```

- [ ] **Step 6: Commit any tuning if needed**

If `n_display` was reduced:

```bash
git add btc_web/callbacks/charts/__init__.py
git commit -m "perf(mc-tab1): tune n_display=50 for cold-cache target <1500ms"
```

---

## CHECKPOINT — User review before prod deploy

**Stop here. Show the user:**
- Total commits in the feature branch (`git log --oneline origin/master..HEAD`)
- Test count delta
- Cold-cache benchmark result
- Any deviation from the spec

**Wait for explicit approval to proceed to Task 14.**

---

## Task 14: Prod deploy

**Files:**
- (No code changes; deploy only)

- [ ] **Step 1: Push to origin**

```bash
git push origin master 2>&1 | tail -3
```

- [ ] **Step 2: Deploy + flush + restart + verify**

```bash
ssh root@89.167.70.45 "cd /opt/quantoshi && git pull && redis-cli FLUSHDB && systemctl restart quantoshi && sleep 12 && systemctl is-active quantoshi" 2>&1 | tail -5
```

Expected: `active`.

- [ ] **Step 3: Prod smoke**

```bash
curl -s -o /dev/null -w "HTTP %{http_code} (%{time_total}s)\n" https://quantoshi.xyz/1
```

Expected: `HTTP 200 (<3s)`.

- [ ] **Step 4: Prod trace verification**

```bash
ssh root@89.167.70.45 "journalctl -u quantoshi --since '2 minutes ago' --no-pager | grep -E 'BUILT|spaghetti|update_bubble' | tail -10"
```

Expected: clean startup, no exceptions.

- [ ] **Step 5: Update memory + close out**

```bash
# Append to memory/restore_callback_architecture.md noting the MC-on-Tab-1
# extension under "Complexity-reduction work" section, similar to D1/D2 entries.
```

Final commit (if needed):

```bash
# (Memory updates are not git-tracked; just save the file.)
```

---

## Self-review checklist (filled by plan author)

**1. Spec coverage:**
- ✅ Layout placement after Projection Quantiles → Task 6
- ✅ `_mc_controls("bub", ...)` reuse → Task 6
- ✅ `_MC_TABS` append at END → Task 7
- ✅ 7 prefix loops in mc_controls.py → Task 5
- ✅ `_get_mc_bubble_fig` wrapper → Task 8
- ✅ `build_bubble_figure` tuple signature → Task 9
- ✅ `_add_mc_spaghetti` rendering helper + tests → Task 10
- ✅ `update_bubble` MC plumbing → Task 11
- ✅ `_POST_RESTORE_TRIGGERS` extension → Task 11 Step 3
- ✅ `_build_bubble_figure_from_state` MC gate → Task 12
- ✅ Snapshot defaults + fingerprint pin → Tasks 1-4
- ✅ `_TAB_CONTROLS["bubble"]` extension → Task 3 Step 3
- ✅ `bubble_defaults()` mirror → Task 4
- ✅ Cold-cache benchmark with target → Task 13
- ✅ CHECKPOINT before prod → after Task 13
- ✅ Prod deploy → Task 14

**2. Placeholder scan:**
- No "TBD", no "implement later".
- One soft spot: Task 5 Step 2 says "match what the file has" for some loop tuples — this is intentional because the audit happens in Step 1 and the engineer applies them mechanically. Acceptable.

**3. Type consistency:**
- `_add_mc_spaghetti(fig, paths, t_axis, n_display)` signature consistent across Tasks 10 and 11.
- `_get_mc_bubble_fig(p) → (fig, mc_result)` tuple contract consistent Tasks 8, 9, 11.
- `_resolve_mc_model_src(...)` signature unchanged from existing usage in update_dca / mc_payment.
- `bub-mc-*` ID set consistent across snapshot_defaults.py, _SNAPSHOT_CONTROLS, _TAB_CONTROLS, mc_payment, layout, callback, _POST_RESTORE_TRIGGERS.
