# Snapshot Defaults SSOT — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate every "default value of a control" into one file (`btc_web/snapshot_defaults.py`), migrate `tab_defaults.py` and every layout `value=`/`data=` literal to read from it. No share-link encoding change in this phase — `q3:` ships unchanged.

**Architecture:** New `SNAPSHOT_DEFAULTS: dict[str, Any]` flat dict keyed by `"{cid}:{prop}"` covering all 206 entries in `_SNAPSHOT_CONTROLS`. `tab_defaults.py` becomes a thin widget→figure-params adapter. Layout files import from `SNAPSHOT_DEFAULTS`. The `_compute_defaults_hash()` is replaced by the broader `_compute_snapshot_defaults_fingerprint()`.

**Tech Stack:** Python 3.14 dev / 3.12 prod; Dash 4.0; pytest; gunicorn.

**Spec:** `docs/superpowers/specs/2026-04-24-snapshot-defaults-ssot-and-v4-encoding-design.md`

**Hard gates:** `dash-callback-reviewer` agent on the diff before every push that touches layout literals (Tasks 6–10) AND before final deploy (Task 14).

**Deploy command:**
```bash
git push origin master && ssh root@89.167.70.45 "cd /opt/quantoshi && git pull && redis-cli FLUSHDB && systemctl restart quantoshi"
```

**Working tree:** `/scratch/code/bitcoinprojections` on `master`. Autonomous deploy delegated.

**Pre-existing test failure note:** `test_colors_central.py::test_no_hex_literals_outside_colors_module` fails on master and is unrelated to this work. All "expect green" steps below mean "no NEW failures."

---

## File Map

| File | Status | Responsibility |
|---|---|---|
| `btc_web/snapshot_defaults.py` | NEW | Flat `SNAPSHOT_DEFAULTS` dict (206 entries) + `ALWAYS_ENCODE` + `_compute_snapshot_defaults_fingerprint()` |
| `btc_web/snapshot_defaults_registry.json` | NEW | Versioned defaults snapshots, capped at 20 |
| `tools/update_defaults_registry.py` | NEW | Regenerator script (computes current fp, appends if missing) |
| `tools/_extract_layout_defaults.py` | NEW (transient) | One-shot discovery helper used in Task 2; deleted at end of phase |
| `btc_web/tab_defaults.py` | MOD | `BUBBLE`/`HEATMAP`/etc. derived from `SNAPSHOT_DEFAULTS`; `_compute_defaults_hash` replaced |
| `btc_web/layout/bubble.py` | MOD | `value=...` literals → `SNAPSHOT_DEFAULTS[...]` lookups |
| `btc_web/layout/heatmap.py` | MOD | Same |
| `btc_web/layout/sim_tabs.py` | MOD | Same (DCA + Retire) |
| `btc_web/layout/supercharge.py` | MOD | Same |
| `btc_web/layout/citadel.py` | MOD | Same |
| `btc_web/layout/citadel_tax.py` | MOD | Same |
| `btc_web/layout/leverage.py` | MOD | Same |
| `btc_web/layout/stack.py` | MOD | Same (lots default price) |
| `btc_web/layout/display_models.py` | MOD | Same (palette, sigma-mode radio) |
| `btc_web/layout/custom_time.py` | MOD | Same (CTA controls) |
| `btc_web/layout/__init__.py` | MOD | Same (palette-store) |
| `btc_web/test_snapshot_defaults.py` | NEW | Structural invariants + adapter parity |
| `btc_web/test_snapshot.py` | MOD | Adds registry-fingerprint-present test |

---

## Task 1: Create `snapshot_defaults.py` skeleton

**Files:**
- Create: `btc_web/snapshot_defaults.py`

This task creates the module with the fingerprint function and the `ALWAYS_ENCODE` set, but leaves `SNAPSHOT_DEFAULTS = {}`. Population happens in Task 2.

- [ ] **Step 1.1: Write the file**

```python
# btc_web/snapshot_defaults.py
"""Single source of truth for control default values across all 206 entries
in _SNAPSHOT_CONTROLS.

Phase 1 (2026-04-24): consolidation only. q3: encoding unchanged.

Conventions
-----------
- Keys are "{component_id}:{property}" matching _SNAPSHOT_CONTROLS exactly.
- Values are the WIDGET representation (what dcc.X(value=...) accepts).
- Translation widget→figure-params lives in tab_defaults.py adapters
  (e.g. bub-xrange:value [2010,2033] → xmin=2010, xmax=2033;
        bub-yrange:value [-1.5, 6.05] → ymin=10**-1.5, ymax=10**6.05).
- Dynamic-default fields (current year, today's date, live BTC price)
  use a static placeholder here; ALWAYS_ENCODE forces emission in q4:
  encoding so the link author's value at link-creation time is preserved.
- See spec docs/superpowers/specs/2026-04-24-snapshot-defaults-ssot-
  and-v4-encoding-design.md.
"""

from __future__ import annotations
import hashlib
import json
from typing import Any, Mapping

# Populated in Task 2 from the live layout. Order is irrelevant for
# fingerprinting (the fingerprint function iterates _SNAPSHOT_CONTROLS).
SNAPSHOT_DEFAULTS: dict[str, Any] = {}

# Controls whose default at link-creation time is genuinely dynamic
# (current_year, today, live BTC price). q4: encoder emits these
# unconditionally even when matching the static placeholder in
# SNAPSHOT_DEFAULTS. Phase 1 does not act on this set; it exists here
# so the SSOT module is self-contained for Phase 2.
ALWAYS_ENCODE: frozenset[str] = frozenset({
    "hm-entry-yr:value",
    "hm-entry-q:value",
    "lev-date:date",
    "lev-price:value",
})


def _compute_snapshot_defaults_fingerprint() -> str:
    """8-char SHA256 over SNAPSHOT_DEFAULTS values, ordered by
    _SNAPSHOT_CONTROLS. Stable under benign dict-literal reorderings.

    Imported lazily to avoid a circular dependency with snapshot.py."""
    from snapshot import _SNAPSHOT_CONTROLS
    h = hashlib.sha256()
    for cid, prop in _SNAPSHOT_CONTROLS:
        val = SNAPSHOT_DEFAULTS.get(f"{cid}:{prop}")
        h.update(json.dumps(val, sort_keys=True).encode())
        h.update(b"\x00")
    return h.hexdigest()[:8]


def get(key: str, fallback: Any = None) -> Any:
    """Convenience accessor used by layout files to source their
    value=/data= literals. Returns fallback if the key is absent."""
    return SNAPSHOT_DEFAULTS.get(key, fallback)
```

- [ ] **Step 1.2: Syntax-check**

```bash
cd /scratch/code/bitcoinprojections/btc_web
PYTHONPATH=".:.." ../btc_venv/bin/python3 -c "import snapshot_defaults; print('OK')"
cd ..
```

Expected: `OK`. (Importing the module on its own works; the fingerprint function is not called yet because `SNAPSHOT_DEFAULTS` is empty.)

- [ ] **Step 1.3: Commit**

```bash
git add btc_web/snapshot_defaults.py
git commit -m "feat(snapshot-defaults): module skeleton (Phase 1 task 1)

ALWAYS_ENCODE + fingerprint function in place; SNAPSHOT_DEFAULTS
populated in next task."
```

---

## Task 2: Bootstrap `SNAPSHOT_DEFAULTS` via a one-shot discovery helper

**Files:**
- Create: `tools/_extract_layout_defaults.py` (transient — deleted in Task 14)
- Modify: `btc_web/snapshot_defaults.py` (add 206 entries)

The realistic way to populate 206 entries without manual hunting is to ask the live layout. This script imports the app, calls `_serve_layout()`, walks every component in `_SNAPSHOT_CONTROLS`, and prints a Python dict literal that can be pasted directly into `snapshot_defaults.py`.

- [ ] **Step 2.1: Write the discovery script**

```python
# tools/_extract_layout_defaults.py
"""One-shot helper for Phase 1 of the SNAPSHOT_DEFAULTS migration.

Renders the live layout, walks the component tree, and emits a Python
dict literal mapping each (cid, prop) in _SNAPSHOT_CONTROLS to the
initial value found in the layout. Manual review required —
unresolved entries are emitted with `# TODO` markers.
"""
from __future__ import annotations
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "btc_web"))

# DEV mode skips heavy startup paths.
os.environ["DEV"] = "1"

import app  # noqa: F401 — registers callbacks
import _app_ctx
from snapshot import _SNAPSHOT_CONTROLS

import flask
with _app_ctx.app.server.test_request_context("/"):
    layout_obj = _app_ctx.app.layout()


def _walk(node):
    """Yield every component in the rendered tree."""
    yield node
    children = getattr(node, "children", None)
    if children is None:
        return
    if isinstance(children, list):
        for c in children:
            if hasattr(c, "_traceable"):  # is a Dash component
                yield from _walk(c)
    elif hasattr(children, "_traceable"):
        yield from _walk(children)


by_id: dict[str, object] = {}
for n in _walk(layout_obj):
    nid = getattr(n, "id", None)
    if isinstance(nid, str):
        by_id[nid] = n

resolved: dict[str, object] = {}
unresolved: list[tuple[str, str]] = []
for cid, prop in _SNAPSHOT_CONTROLS:
    comp = by_id.get(cid)
    if comp is None:
        unresolved.append((cid, prop))
        continue
    val = getattr(comp, prop, None)
    # MappingProxyType / tuples → JSON-friendly forms
    if isinstance(val, tuple):
        val = list(val)
    resolved[f"{cid}:{prop}"] = val

print("# === RESOLVED (paste into SNAPSHOT_DEFAULTS) ===")
for k in sorted(resolved):
    v = resolved[k]
    print(f"    {k!r}: {v!r},")
print()
print(f"# === UNRESOLVED ({len(unresolved)}) — review manually ===")
for cid, prop in unresolved:
    print(f"    {cid!r:30s} {prop!r}")
```

- [ ] **Step 2.2: Run the script and capture output**

```bash
cd /scratch/code/bitcoinprojections
btc_venv/bin/python3 tools/_extract_layout_defaults.py > /tmp/snapshot_defaults_seed.txt 2>&1
head -40 /tmp/snapshot_defaults_seed.txt
echo "---"
grep "UNRESOLVED" /tmp/snapshot_defaults_seed.txt
```

Expected: a `# === RESOLVED ===` block with ~190+ entries followed by `# === UNRESOLVED ===` with the remainder. Unresolved entries are typically components rendered conditionally (lazy tab content) or stores that don't have an id-bearing initial render.

- [ ] **Step 2.3: Manually resolve the unresolved entries**

For each entry under `UNRESOLVED`, find its initial value by grepping the layout source:

```bash
grep -rn "id=[\"']<CID>[\"']" /scratch/code/bitcoinprojections/btc_web/layout/
```

For Stores, the value is the `data=` arg. For lazy-tab controls, the value is the `value=`/`data=` literal in the tab's layout function.

Common cases (handle these specifically):

- **`hm-active-model:data`** → `"bub"` (default pill).
- **`palette-store:data`** → `"default"`.
- **`cp-tax-config:data`** → `None` (tax disabled by default).
- **`lev-date:date`** → `None` (today resolved live).
- **`lev-price:value`** → `None` (live BTC resolved live).
- **`hm-entry-yr:value`** → 2026 (current year placeholder; bumped each January).
- **`hm-entry-q:value`** → 50.0 (live percentile placeholder).
- **`bub-xrange:value`** → `[2010, 2033]` (already resolved by script — verify).
- **`bub-yrange:value`** → `[-1.5, 6.05]` (already resolved — verify).
- **`bub-toggles:value`** → `["shade", "show_data", "show_today"]`.
- **`dca-yr-range:value`** → `None` (year-derived sentinel; `dca_defaults()` substitutes `[yr_now, yr_now+10]`).

- [ ] **Step 2.4: Paste the merged dict into `snapshot_defaults.py`**

Replace the `SNAPSHOT_DEFAULTS: dict[str, Any] = {}` line in `btc_web/snapshot_defaults.py` with the merged dict (resolved entries from the script + manually-resolved entries).

The dict literal should be exactly 206 entries — one per `_SNAPSHOT_CONTROLS` row. Verify count:

```bash
btc_venv/bin/python3 -c "
import sys; sys.path.insert(0, 'btc_web')
from snapshot import _SNAPSHOT_CONTROLS
print('expected:', len(_SNAPSHOT_CONTROLS))
import snapshot_defaults
print('actual:', len(snapshot_defaults.SNAPSHOT_DEFAULTS))
"
```

Expected: both numbers identical (206).

- [ ] **Step 2.5: Commit**

```bash
git add btc_web/snapshot_defaults.py tools/_extract_layout_defaults.py
git commit -m "feat(snapshot-defaults): bootstrap SNAPSHOT_DEFAULTS from live layout

206 entries seeded via tools/_extract_layout_defaults.py. The
discovery script is retained until end of Phase 1 then deleted."
```

---

## Task 3: Add structural tests

**Files:**
- Create: `btc_web/test_snapshot_defaults.py`

- [ ] **Step 3.1: Write the test file**

```python
# btc_web/test_snapshot_defaults.py
"""Structural invariants for snapshot_defaults.SSOT. Phase 1.

Tests #6 (bub-toggles round-trip), #7 (always_encode_members),
and #8 (mc_enable_defaults_are_disabled) anticipate Phase 2 but are
cheap to keep here from the start."""
import pytest


def test_every_snapshot_control_has_default():
    from snapshot import _SNAPSHOT_CONTROLS
    from snapshot_defaults import SNAPSHOT_DEFAULTS
    missing = [(c, p) for c, p in _SNAPSHOT_CONTROLS
               if f"{c}:{p}" not in SNAPSHOT_DEFAULTS]
    assert not missing, (
        f"{len(missing)} control(s) missing from SNAPSHOT_DEFAULTS: "
        f"{missing[:5]}{'...' if len(missing) > 5 else ''}")


def test_no_phantom_keys_in_snapshot_defaults():
    from snapshot import _SNAPSHOT_CONTROLS
    from snapshot_defaults import SNAPSHOT_DEFAULTS
    valid = {f"{c}:{p}" for c, p in _SNAPSHOT_CONTROLS}
    phantom = [k for k in SNAPSHOT_DEFAULTS if k not in valid]
    assert not phantom, (
        f"SNAPSHOT_DEFAULTS contains keys not in _SNAPSHOT_CONTROLS: "
        f"{phantom}")


def test_checklist_defaults_are_lists():
    from snapshot import _CHECKLIST_OPTIONS
    from snapshot_defaults import SNAPSHOT_DEFAULTS
    for cid in _CHECKLIST_OPTIONS:
        for prop in ("value",):
            key = f"{cid}:{prop}"
            if key not in SNAPSHOT_DEFAULTS:
                continue
            v = SNAPSHOT_DEFAULTS[key]
            assert v is None or isinstance(v, list), (
                f"Checklist default {key} must be list or None; "
                f"got {type(v).__name__}: {v!r}")


def test_fingerprint_is_8_hex_chars():
    from snapshot_defaults import _compute_snapshot_defaults_fingerprint
    fp = _compute_snapshot_defaults_fingerprint()
    assert len(fp) == 8
    int(fp, 16)  # valid hex


def test_fingerprint_is_deterministic():
    from snapshot_defaults import _compute_snapshot_defaults_fingerprint
    assert (_compute_snapshot_defaults_fingerprint() ==
            _compute_snapshot_defaults_fingerprint())


def test_bub_toggles_default_round_trips_through_bitmask():
    """If layout's bub-toggles literal drifts from the bitmask schema,
    snapshot encode/decode silently corrupts the toggle state."""
    from snapshot import _CHECKLIST_OPTIONS, _list_to_mask, _mask_to_list
    from snapshot_defaults import SNAPSHOT_DEFAULTS
    val = SNAPSHOT_DEFAULTS["bub-toggles:value"]
    if val is None:
        pytest.skip("bub-toggles default is None")
    opts = _CHECKLIST_OPTIONS["bub-toggles"]
    mask = _list_to_mask(val, opts)
    round_tripped = _mask_to_list(mask, opts)
    assert sorted(round_tripped) == sorted(val), (
        f"bub-toggles default {val} does not round-trip: {round_tripped}")


def test_always_encode_members_in_snapshot_controls():
    from snapshot import _SNAPSHOT_CONTROLS
    from snapshot_defaults import ALWAYS_ENCODE
    valid = {f"{c}:{p}" for c, p in _SNAPSHOT_CONTROLS}
    missing = [k for k in ALWAYS_ENCODE if k not in valid]
    assert not missing, (
        f"ALWAYS_ENCODE contains keys not in _SNAPSHOT_CONTROLS: {missing}")


def test_mc_enable_defaults_are_disabled():
    """Required by Phase 2 _mc_null_out_diffs_v4: every *-mc-enable
    default must be the disabled state, AND every expected prefix must
    exist in _SNAPSHOT_CONTROLS (so next(...) cannot StopIteration)."""
    from snapshot import _SNAPSHOT_CONTROLS
    from snapshot_defaults import SNAPSHOT_DEFAULTS
    prefixes = ("dca-mc-", "ret-mc-", "hm-mc-", "sc-mc-", "cp-mc-")
    cids = {c for c, _ in _SNAPSHOT_CONTROLS}
    for pfx in prefixes:
        cid = f"{pfx}enable"
        assert cid in cids, (
            f"_SNAPSHOT_CONTROLS missing {cid} — "
            f"_mc_null_out_diffs_v4 lookup would StopIteration")
        val = SNAPSHOT_DEFAULTS.get(f"{cid}:value")
        assert val in (None, [], 0), (
            f"{cid}:value default must be disabled state; got {val!r}")
```

- [ ] **Step 3.2: Run tests**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_snapshot_defaults.py -v 2>&1 | tail -15
```

Expected: all 8 tests pass. If `test_every_snapshot_control_has_default` fails, return to Task 2.4 and add the missing entries.

- [ ] **Step 3.3: Run cache-key-alignment test (must still pass — pure additive change)**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_cache_key_alignment.py -v 2>&1 | tail -5
```

Expected: PASS unchanged (no consumer is reading from `snapshot_defaults.py` yet).

- [ ] **Step 3.4: Commit**

```bash
git add btc_web/test_snapshot_defaults.py
git commit -m "test(snapshot-defaults): structural invariants (Phase 1 task 3)"
```

---

## Task 4: Migrate `tab_defaults.BUBBLE` to derive from `SNAPSHOT_DEFAULTS`

**Files:**
- Modify: `btc_web/tab_defaults.py`

The principle: the frozen `BUBBLE` MappingProxyType stays (callers import `BUBBLE["xscale"]`), but the values come from `SNAPSHOT_DEFAULTS` via the adapter `bubble_defaults()`. The translation logic (widget→figure-params for `bub-xrange`, `bub-yrange`, etc.) stays in this file.

- [ ] **Step 4.1: Read the current `bubble_defaults()` function**

```bash
grep -n "^def bubble_defaults\|^BUBBLE = " /scratch/code/bitcoinprojections/btc_web/tab_defaults.py
sed -n '$(grep -n "^def bubble_defaults" /scratch/code/bitcoinprojections/btc_web/tab_defaults.py | cut -d: -f1),+50p' /scratch/code/bitcoinprojections/btc_web/tab_defaults.py
```

Note the function's outputs — what figure-params it constructs from BUBBLE.

- [ ] **Step 4.2: Refactor `BUBBLE` to derive from `SNAPSHOT_DEFAULTS`**

Replace the `BUBBLE = MappingProxyType({...})` literal in `tab_defaults.py` with:

```python
def _build_bubble_dict():
    """Derive BUBBLE figure-params dict from SNAPSHOT_DEFAULTS widget
    values. Translation layer lives here so callers can keep importing
    `BUBBLE["xscale"]` etc."""
    from snapshot_defaults import SNAPSHOT_DEFAULTS as S
    sd = lambda k, default=None: S.get(k, default)
    qs_list = sd("bub-qs:value", []) or []
    selected_qs = tuple(_qs_label_to_value(q) for q in qs_list)
    yr = sd("bub-yrange:value", [-1.5, 6.05])
    toggles = set(sd("bub-toggles:value", []) or [])
    return {
        "selected_qs": selected_qs,
        "xscale": sd("bub-xscale:value", "log"),
        "yscale": sd("bub-yscale:value", "log"),
        "ymin": 10 ** float(yr[0]),
        "ymax": 10 ** float(yr[1]),
        "shade":       "shade"       in toggles,
        "show_data":   "show_data"   in toggles,
        "show_today":  "show_today"  in toggles,
        "show_legend": "show_legend" in toggles,
        "minor_grid":  "minor_grid"  in toggles,
        "show_ols":    "show_ols"    in toggles,
        "show_ucl":    "show_ucl"    in toggles,
        "show_comp":   "show_comp"   in toggles,
        "show_sup":    "show_sup"    in toggles,
        "n_future":    sd("bub-future-n:value", 3),
        "pt_size":     sd("bub-pt-size:value", PT_SIZE_DEFAULT),
        "pt_alpha":    sd("bub-alpha:value",   PT_ALPHA_DEFAULT),
        "stack":       sd("bub-stack:value", 0),
        "show_stack":  False,
        "use_lots":    False,
        "legend_pos":  sd("bub-legend-pos:value", "top-left"),
        "comp_color":  sd("bub-comp-color:value", LOT_MARKER_COLOR),
        "comp_lw":     sd("bub-comp-lw:value",    TRACE_WIDTH_COMPOSITE),
        "sup_color":   sd("bub-sup-color:value",  FALLBACK_MODEL_GRAY),
        "sup_lw":      sd("bub-sup-lw:value",     TRACE_WIDTH_SUPPORT),
        "active_models": tuple(sd("bub-model-show:value", []) or ["bub"]),
        "palette":     sd("palette-store:data", "default"),
        "scanner_lines": tuple(sd("bub-scanner-show:value", []) or []),
        "qs_mode":     tuple(sd("bub-qs-mode:value", []) or []),
        "sigma_mode":  sd("bub-sigma-mode:value", "resqr"),
        # Derived-from-State keys (callbacks always inject these into params)
        "decomp_model": "",
        "decomp_mode":  "individual",
        "decomp_components": (),
        "lppl_n_freqs": (),
        "lppl_weighted": (),
        "lppl_no_13": (),
        "config_b_keys": (),
    }


def _qs_label_to_value(label):
    """Translate bub-qs checklist label ('inner','outer','median','custom')
    to a quantile float (0.05, 0.95, 0.50, ...) — must match the
    callback's existing logic."""
    return {"median": 0.5, "outer": 0.95, "inner": 0.75}.get(label, 0.5)


BUBBLE = MappingProxyType(_build_bubble_dict())
```

**Important:** the `_qs_label_to_value` function must produce IDENTICAL values to the existing bub-qs handling in `callbacks/charts/__init__.py`. Locate the existing handler:

```bash
grep -n "bub-qs\|selected_qs" /scratch/code/bitcoinprojections/btc_web/callbacks/charts/__init__.py | head -20
```

Update `_qs_label_to_value` if the actual label→value mapping differs.

If the bub-qs widget stores quantile FLOATS directly (not labels), simplify to `selected_qs = tuple(sd("bub-qs:value", [0.5]) or [0.5])`.

- [ ] **Step 4.3: Run tests — BUBBLE consumers must still pass**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_defaults.py btc_web/test_cache_key_alignment.py btc_web/test_snapshot_defaults.py -v 2>&1 | tail -20
```

Expected: PASS. If `test_inner_collections_are_tuples` fails because some BUBBLE values are now lists instead of tuples, wrap them in `tuple(...)` in `_build_bubble_dict()`.

- [ ] **Step 4.4: Run the bubble figure smoke test**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_figures.py -v -k bubble 2>&1 | tail -10
```

Expected: PASS unchanged.

- [ ] **Step 4.5: Commit**

```bash
git add btc_web/tab_defaults.py
git commit -m "refactor(tab_defaults): BUBBLE now derived from SNAPSHOT_DEFAULTS

Translation layer (widget [2010,2033] → xmin/xmax, log-space yrange,
toggle list → boolean fields) preserved in _build_bubble_dict()."
```

---

## Task 5: Migrate remaining tabs (HEATMAP, DCA, RETIRE, SUPERCHARGE, CITADEL)

**Files:**
- Modify: `btc_web/tab_defaults.py`

Repeat the Task 4 pattern for each remaining tab. **One sub-commit per tab.**

For each tab:
1. Read the current `{tab}_defaults()` function and the corresponding frozen dict.
2. Identify which figure-params are derived from widget keys vs. callback-injected derived keys vs. live-derived (year, price).
3. Build a `_build_{tab}_dict()` helper that calls `SNAPSHOT_DEFAULTS.get(...)` for each widget key, applies translation, and substitutes for None-sentinel year-derived fields at request time (in `{tab}_defaults()`, not at module import).
4. Run the cache-alignment test + the tab's figure tests.
5. Commit with message `refactor(tab_defaults): {TAB} derived from SNAPSHOT_DEFAULTS`.

- [ ] **Step 5.1: Migrate HEATMAP**

Translation points:
- `hm-entry-yr:value` is a static placeholder (e.g. 2026); `heatmap_defaults()` resolves to `current_year` if the placeholder is stale.
- `hm-entry-q:value` is the live-percentile placeholder; resolved via the live ticker callback at request time, not in defaults.
- `hm-active-model:data` → `BUBBLE["hm_model"]` field.
- `hm_palette` is UI-only — stripped from cache key in `utils._get_heatmap_fig`. Keep as-is.

Run after migration: `btc_venv/bin/python3 -m pytest btc_web/test_figures.py -v -k heatmap btc_web/test_cache_key_alignment.py`. Commit.

- [ ] **Step 5.2: Migrate DCA**

Translation points:
- `dca-yr-range:value` may be `None` (year-derived sentinel) → `dca_defaults()` substitutes `[yr_now, yr_now+10]`.
- `dca-amount:value`, `dca-freq:value` are direct widget values.

Run after migration: `pytest btc_web/test_figures.py -v -k dca`. Commit.

- [ ] **Step 5.3: Migrate RETIRE**

Translation points: `ret-yr-range:value` is `[2031, 2075]` static (NOT year-derived per spec).

Run after migration: `pytest btc_web/test_figures.py -v -k retire`. Commit.

- [ ] **Step 5.4: Migrate SUPERCHARGE**

Translation points: `sc-d0..sc-d4:value` are five separate scalars; `_build_supercharge_dict()` packs them into `delays = (d0, d1, d2, d3, d4)`.

Run after migration: `pytest btc_web/test_figures.py -v -k super`. Commit.

- [ ] **Step 5.5: Migrate CITADEL**

Translation points: `cp-tax-config:data` default is `None`. Citadel has the most controls (~40 widget keys + ~16 tax keys).

Run after migration: `pytest btc_web/test_citadel.py btc_web/test_citadel_diag.py btc_web/test_citadel_steps.py btc_web/test_cache_key_alignment.py 2>&1 | tail -8`. Commit.

- [ ] **Step 5.6: Final tab_defaults test pass**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_defaults.py btc_web/test_cache_key_alignment.py btc_web/test_snapshot_defaults.py -v 2>&1 | tail -10
```

Expected: PASS.

---

## Task 6: Migrate `layout/bubble.py` value= literals

**Files:**
- Modify: `btc_web/layout/bubble.py`

- [ ] **Step 6.1: Survey current literals**

```bash
grep -n "value=\|data=" /scratch/code/bitcoinprojections/btc_web/layout/bubble.py | head -40
```

For each `value=...` or `data=...` that is a literal (not a function call, not `_app_ctx.X`, not `BUBBLE["k"]`), replace with `SNAPSHOT_DEFAULTS["{cid}:{prop}"]` lookup.

- [ ] **Step 6.2: Add the import**

At top of `btc_web/layout/bubble.py`:

```python
from snapshot_defaults import SNAPSHOT_DEFAULTS as _SD
```

- [ ] **Step 6.3: Replace literals**

Pattern: for each control declaration in the layout function, locate its `id="..."`. Find the matching `value=`/`data=` arg. Replace the literal with `_SD["<id>:<prop>"]`.

Example:

Before:
```python
dcc.RadioItems(id="bub-xscale", value="log", options=[...])
```

After:
```python
dcc.RadioItems(id="bub-xscale", value=_SD["bub-xscale:value"], options=[...])
```

Leave alone:
- `value=BUBBLE["xscale"]` — already an indirect lookup; works because BUBBLE now derives from SNAPSHOT_DEFAULTS. (Optional cleanup: replace with `_SD["bub-xscale:value"]` for consistency. Pick one style and apply uniformly.)
- `value=_app_ctx._DEF_QS` — live-derived, leave as-is.
- `value=yr_now` (or similar) — live-derived, leave as-is.

- [ ] **Step 6.4: Syntax-check**

```bash
cd /scratch/code/bitcoinprojections/btc_web && PYTHONPATH=".:.." ../btc_venv/bin/python3 -c "import layout.bubble; print('OK')"; cd ..
```

- [ ] **Step 6.5: Visual smoke**

```bash
cd /scratch/code/bitcoinprojections
lsof -ti :8050 2>/dev/null | xargs -r kill -9
DEV=1 nohup bash run_web.sh > /tmp/quantoshi_dev.log 2>&1 &
sleep 12
curl -s -o /dev/null -w "/1: %{http_code}\n" http://localhost:8050/1
```

Expected: HTTP 200. Open `http://localhost:8050/1` in a browser; verify every control on Tab 1 has the same default state as before this task (RadioItems selection, slider position, checklist boxes, color pickers).

- [ ] **Step 6.6: Run callback + figure tests**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_callbacks.py btc_web/test_figures.py -v -k "bubble or scanner" 2>&1 | tail -10
```

Expected: PASS unchanged.

- [ ] **Step 6.7: Commit**

```bash
git add btc_web/layout/bubble.py
git commit -m "refactor(layout): bubble.py reads value=/data= from SNAPSHOT_DEFAULTS"
```

---

## Task 7: Migrate `layout/heatmap.py`

**Files:**
- Modify: `btc_web/layout/heatmap.py`

Same pattern as Task 6.

- [ ] **Step 7.1: Add import + replace literals.** Use `from snapshot_defaults import SNAPSHOT_DEFAULTS as _SD`. Live-derived (`_app_ctx._HM_ENTRY_Q_DEFAULT`, `yr_now`) stays as-is.
- [ ] **Step 7.2: Syntax-check** `import layout.heatmap`.
- [ ] **Step 7.3: Visual smoke** `/2` paints identical.
- [ ] **Step 7.4: Tests** `pytest btc_web/ -v -k heatmap --ignore-glob='*_e2e.py'`.
- [ ] **Step 7.5: Commit** `git commit -m "refactor(layout): heatmap.py reads from SNAPSHOT_DEFAULTS"`.

---

## Task 8: Migrate `layout/sim_tabs.py` (DCA + Retire)

**Files:**
- Modify: `btc_web/layout/sim_tabs.py`

Same pattern. DCA's year-range slider value=None ⇒ leave as `value=[yr_now, yr_now+10]` if the layout currently inlines that — the sentinel lives in `SNAPSHOT_DEFAULTS`, not in the layout `value=` (which must always render to a real range so the slider widget paints correctly).

- [ ] **Step 8.1–8.5:** same shape as Task 7. Visual smoke `/3` and `/4`. Commit `refactor(layout): sim_tabs.py reads from SNAPSHOT_DEFAULTS`.

---

## Task 9: Migrate `layout/supercharge.py`

**Files:**
- Modify: `btc_web/layout/supercharge.py`

Same pattern. Five `sc-d0..sc-d4` inputs each get their own `_SD` lookup.

- [ ] **Step 9.1–9.5:** as Task 7. Visual smoke `/5`. Commit `refactor(layout): supercharge.py reads from SNAPSHOT_DEFAULTS`.

---

## Task 10: Migrate remaining layout files (citadel, citadel_tax, leverage, stack, display_models, custom_time, root layout)

**Files:**
- Modify: `btc_web/layout/citadel.py`
- Modify: `btc_web/layout/citadel_tax.py`
- Modify: `btc_web/layout/leverage.py`
- Modify: `btc_web/layout/stack.py`
- Modify: `btc_web/layout/display_models.py`
- Modify: `btc_web/layout/custom_time.py`
- Modify: `btc_web/layout/__init__.py` (palette-store, snapshot stores)

Same pattern, **one commit per file**.

- [ ] **Step 10.1: citadel.py** — most controls; ~40 widget keys. Visual smoke `/6`. Commit.
- [ ] **Step 10.2: citadel_tax.py** — `cp-tax-*`, `cp-td-*`, `cp-tf-*` prefixes. Visual smoke: open Citadel tab, click "Tax" toggle, confirm modal opens with default values. Commit.
- [ ] **Step 10.3: leverage.py** — `lev-*`. Visual smoke `/7`. Note: `lev-date` and `lev-price` are live-derived; layout keeps `value=today_str` / `value=live_price`. Commit.
- [ ] **Step 10.4: stack.py** — minimal (lots default price). Commit.
- [ ] **Step 10.5: display_models.py** — `bub-sigma-mode` already lives here (we touched it earlier today); ensure it now reads from `SNAPSHOT_DEFAULTS["bub-sigma-mode:value"]`. Same for `palette-select-*` controls. Commit.
- [ ] **Step 10.6: custom_time.py** — CTA `cta-*` controls. Commit.
- [ ] **Step 10.7: layout/__init__.py** — `dcc.Store(id="palette-store", data=...)`, `dcc.Store(id="snapshot-applied-tabs"...)`, etc. The tab routing stores stay as-is; only id-bearing user-facing stores migrate. Commit.

- [ ] **Step 10.8: Final layout test pass**

```bash
btc_venv/bin/python3 -m pytest btc_web/ --ignore-glob='*_e2e.py' -q 2>&1 | tail -5
```

Expected: 1 pre-existing failure (`test_no_hex_literals_outside_colors_module`); zero new failures.

---

## Task 11: Replace `_compute_defaults_hash()` with `_compute_snapshot_defaults_fingerprint()`

**Files:**
- Modify: `btc_web/tab_defaults.py`

- [ ] **Step 11.1: Locate the existing hash function**

```bash
sed -n '329,342p' /scratch/code/bitcoinprojections/btc_web/tab_defaults.py
```

- [ ] **Step 11.2: Replace its body**

Replace `_compute_defaults_hash()` with:

```python
def _compute_defaults_hash() -> str:
    """Cache-invalidation fingerprint. Now sourced from
    snapshot_defaults._compute_snapshot_defaults_fingerprint() —
    broader scope (covers all 206 _SNAPSHOT_CONTROLS entries, not
    just the 6 frozen tab dicts), more accurate L0 invalidation."""
    from snapshot_defaults import _compute_snapshot_defaults_fingerprint
    return _compute_snapshot_defaults_fingerprint()
```

- [ ] **Step 11.3: Run cache-alignment test**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_cache_key_alignment.py -v 2>&1 | tail -5
```

Expected: PASS.

- [ ] **Step 11.4: Commit**

```bash
git add btc_web/tab_defaults.py
git commit -m "refactor(tab_defaults): _compute_defaults_hash delegates to snapshot fingerprint

Broader scope catches default changes in lev-*, cp-tax-*, navbar
controls that the old hash ignored. L0 cache hash will change once
on deploy — redis-cli FLUSHDB in the deploy script handles this."
```

---

## Task 12: Generate baseline registry + add registry test

**Files:**
- Create: `tools/update_defaults_registry.py`
- Create: `btc_web/snapshot_defaults_registry.json`
- Modify: `btc_web/test_snapshot.py`

- [ ] **Step 12.1: Write the updater script**

```python
# tools/update_defaults_registry.py
"""Idempotent registry updater. Phase 1.

Computes today's fingerprint from SNAPSHOT_DEFAULTS. If the
fingerprint is already in btc_web/snapshot_defaults_registry.json,
exits 0 without modification. Else appends a new entry and trims
oldest if length > 20.

Usage:
    btc_venv/bin/python3 tools/update_defaults_registry.py
"""
from __future__ import annotations
import copy
import datetime as dt
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "btc_web"))

from snapshot_defaults import (SNAPSHOT_DEFAULTS,
                               _compute_snapshot_defaults_fingerprint)

REGISTRY_PATH = os.path.join(ROOT, "btc_web",
                              "snapshot_defaults_registry.json")
CAP = 20


def main() -> int:
    fp = _compute_snapshot_defaults_fingerprint()
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH) as f:
            registry = json.load(f)
    else:
        registry = []
    fps = {entry["fp"] for entry in registry}
    if fp in fps:
        print(f"fingerprint {fp} already in registry; no change")
        return 0
    entry = {
        "fp": fp,
        "created_at": dt.date.today().isoformat(),
        "defaults": copy.deepcopy(SNAPSHOT_DEFAULTS),
    }
    registry.append(entry)
    if len(registry) > CAP:
        dropped = registry[: len(registry) - CAP]
        registry = registry[-CAP:]
        print(f"dropped {len(dropped)} oldest entries")
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2, sort_keys=True)
    print(f"appended fingerprint {fp}; registry now has {len(registry)} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 12.2: Run it to create the baseline registry**

```bash
btc_venv/bin/python3 tools/update_defaults_registry.py
ls -lh btc_web/snapshot_defaults_registry.json
head -3 btc_web/snapshot_defaults_registry.json
```

Expected: file size 4–10 KB; first lines show `[\n  {\n    "created_at": "2026-04-24",`.

- [ ] **Step 12.3: Add the registry test**

Append to `btc_web/test_snapshot.py` inside an existing test class (or new class `TestSnapshotDefaultsRegistry`):

```python
class TestSnapshotDefaultsRegistry:
    def test_current_fingerprint_in_registry(self):
        """If this fails, run: btc_venv/bin/python3 tools/update_defaults_registry.py"""
        import json, os, pathlib
        here = pathlib.Path(os.path.dirname(__file__))
        from snapshot_defaults import _compute_snapshot_defaults_fingerprint
        fp = _compute_snapshot_defaults_fingerprint()
        with open(here / "snapshot_defaults_registry.json") as f:
            registry = json.load(f)
        fps = [e["fp"] for e in registry]
        assert fp in fps, (
            f"current fingerprint {fp} not in registry {fps}; run "
            "tools/update_defaults_registry.py and commit the result")

    def test_registry_capped_at_20(self):
        import json, os, pathlib
        here = pathlib.Path(os.path.dirname(__file__))
        with open(here / "snapshot_defaults_registry.json") as f:
            registry = json.load(f)
        assert len(registry) <= 20, f"registry has {len(registry)} entries (cap=20)"
```

- [ ] **Step 12.4: Run tests**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_snapshot.py::TestSnapshotDefaultsRegistry -v 2>&1 | tail -10
```

Expected: PASS.

- [ ] **Step 12.5: Commit**

```bash
git add tools/update_defaults_registry.py btc_web/snapshot_defaults_registry.json btc_web/test_snapshot.py
git commit -m "feat(snapshot-defaults): baseline registry + test (Phase 1 task 12)"
```

---

## Task 13: dash-callback-reviewer hard gate

- [ ] **Step 13.1: Dispatch reviewer on the full Phase 1 diff**

Use the `dash-callback-reviewer` agent. Prompt:

```
Review /scratch/code/bitcoinprojections diff from <BASE_SHA>..HEAD
(<BASE_SHA> = git merge-base of master with the commit immediately
before Task 1 in this Phase 1 plan).

Files modified: btc_web/snapshot_defaults.py (new), btc_web/tab_defaults.py,
btc_web/layout/{bubble,heatmap,sim_tabs,supercharge,citadel,citadel_tax,
leverage,stack,display_models,custom_time,__init__}.py,
btc_web/test_snapshot_defaults.py (new), btc_web/test_snapshot.py,
tools/update_defaults_registry.py (new), btc_web/snapshot_defaults_registry.json (new).

Verify:
1. No layout literal silently changed value during migration
   (every replaced literal must equal the entry it now reads from).
2. No callback signature changed.
3. tab_defaults adapters still produce dicts with the same keys
   the chart callbacks expect (no missing keys, no new keys).
4. _compute_defaults_hash now delegates to fingerprint function;
   call sites unchanged.
5. Cache-key alignment invariant holds: prewarm key still equals
   first-render callback key.
6. No new Output/Input causing allow_duplicate / prevent_initial_call
   collision.
7. The discovery script tools/_extract_layout_defaults.py is still
   present (deletion deferred to Task 14).

Flag BLOCKING issues only. Under 500 words.
```

- [ ] **Step 13.2: Fix any BLOCKING findings, re-dispatch as needed.**

Only proceed past this gate when reviewer returns zero BLOCKING findings.

---

## Task 14: Deploy + verify + clean up discovery helper

- [ ] **Step 14.1: Dev smoke**

```bash
cd /scratch/code/bitcoinprojections
lsof -ti :8050 2>/dev/null | xargs -r kill -9
DEV=1 nohup bash run_web.sh > /tmp/quantoshi_dev.log 2>&1 &
sleep 12
for path in / /1 /2 /3 /4 /5 /6 /7 /8 /9 /10; do
  printf "%s " "$path"
  curl -s -o /dev/null -w "%{http_code}\n" "http://localhost:8050$path"
done
tail -10 /tmp/quantoshi_dev.log
```

Expected: every path returns 200. No tracebacks in the log.

- [ ] **Step 14.2: Manual visual sweep**

In a browser, visit each tab `/1`–`/10`. For each tab, confirm every control's initial state matches the pre-Phase-1 state:
- Radio selections
- Slider positions and value labels
- Checklist boxes (toggled on/off)
- Dropdown values
- Color pickers
- Numeric input values

Also: generate a share link with default settings, paste into a fresh browser, confirm restore is identical.

If anything looks off, identify which control + tab, look up its `SNAPSHOT_DEFAULTS` entry, fix the value, re-run `tools/update_defaults_registry.py`, commit.

- [ ] **Step 14.3: Delete the discovery helper**

```bash
rm tools/_extract_layout_defaults.py
git add tools/_extract_layout_defaults.py
git commit -m "chore: remove transient discovery helper (Phase 1 done)"
```

- [ ] **Step 14.4: Push + deploy**

```bash
git push origin master
ssh root@89.167.70.45 'cd /opt/quantoshi && git pull && redis-cli FLUSHDB && systemctl restart quantoshi'
```

- [ ] **Step 14.5: Prod smoke**

```bash
sleep 8
for path in / /1 /2 /3 /4 /5 /6 /7 /8 /9 /10; do
  printf "%s " "$path"
  curl -s -o /dev/null -w "%{http_code}\n" "https://quantoshi.xyz$path"
done
ssh root@89.167.70.45 'journalctl -u quantoshi --since "60 seconds ago" --no-pager | grep -iE "error|traceback|critical|nonexistent" | head -10'
```

Expected: every path 200; zero error/traceback lines.

- [ ] **Step 14.6: Verify share-link backward compat on prod**

In a fresh browser, paste a known-good `q3:` share link from earlier this week. Confirm restore works identically. (Use one from `link-history` localStorage if available, or generate fresh on prod with non-default settings.)

- [ ] **Step 14.7: Soak**

Phase 1 is "done" once it has been deployed for at least 1 day with no error reports and no observed visual regressions. Phase 2 plan-writing begins after this soak period.

---

## Self-Review

**Spec coverage:**

| Spec section | Implemented in |
|---|---|
| §2 Architecture (two layers) | Tasks 1–5 |
| §3 Live-derived sentinel handling | Task 2 (Step 2.3 enumerates), Task 5 |
| §4 Fingerprint scheme | Task 1, Task 11 |
| §4b ALWAYS_ENCODE | Task 1 (set defined; Phase 2 uses) |
| §5 v4 encoding | (Phase 2 — out of scope) |
| §6 Registry | Task 12 |
| §7 File layout | File map at top of plan |
| §8 Tests | Tasks 3, 12 |
| §9 Phase 1 migration sequence | Tasks 1–14 |
| §11 Trapdoors 1–12 | Tasks 4 (#1, #2), 5.4 (#3), 5.5 (#4), 5.2 (#5), 4 (#6), Task 11 (#7, #8), 12 (#9, #10), Task 14.4 (#11) |

**Placeholder scan:**
- No "TBD" / "implement later" / "similar to Task N".
- Every step shows the exact code, command, or expected output.
- Migration tasks (5.1–5.6, 7–10) are pattern-applications of Task 4/Task 6; the pattern is shown explicitly in the leading task and applied per-tab. This is intentional repetition of the same shape, not a placeholder.

**Type consistency:**
- `SNAPSHOT_DEFAULTS: dict[str, Any]` — same name and shape across all tasks.
- `_compute_snapshot_defaults_fingerprint()` — named identically in Tasks 1, 11, 12.
- `ALWAYS_ENCODE` — frozenset; same name across Task 1, Task 3, spec.
- Registry path `btc_web/snapshot_defaults_registry.json` — identical in Task 12 + spec §6.

---

## Execution choice

User has delegated autonomous execution through prod deploy. Use **superpowers:executing-plans** inline. Task 13 (`dash-callback-reviewer`) is the hard gate before push — do not skip.

Phase 2 plan written separately AFTER Phase 1 deploys and soaks for ≥1 day.
