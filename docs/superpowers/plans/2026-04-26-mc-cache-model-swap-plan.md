# MC Cache Model Swap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the MC cache "what's cached" knowledge into a single-source-of-truth pattern, then use the new pattern to drop LPPL, drop `exp` (not in any dropdown — wasted cache space), and add EPPL→`ecfg_1d_1u`. Cache rebuild deferred.

**Architecture:** Two declarative inputs in `btc_web/mc_cache.py` (`_INTENDED_KEYS`, `intended_models(M)`, `MASTER_TO_CACHED_FALLBACK`) replace today's drift-prone duplicate hardcoded lists. `_CACHED_MODEL_KEYS` becomes disk-derived via a new `_parse_cache_filename` helper. The resolver fallback in `_resolve_mc_model_src` becomes generic via `MASTER_TO_CACHED_FALLBACK.get(master, resolved)`. UI bolding factors into `is_master_cached`. Stash/commit/restore helpers make the rebuild script interrupt-safe.

**Tech Stack:** Python 3.14 (dev) / 3.12 (prod), Plotly Dash 4.0.0, pytest, numpy. No new dependencies.

**Spec:** `/scratch/code/bitcoinprojections/docs/superpowers/specs/2026-04-26-mc-cache-model-swap-design.md` (commit `2103b19`).

**File map:**
- Modify `btc_web/mc_cache.py` — most of the new code; literal replaced by derived
- Modify `btc_web/callbacks/charts/_resolvers.py` — `_resolve_mc_model_src` simplification (lines 213–250)
- Modify `btc_web/layout/mc_controls.py` — `_mc_model_src_options` uses `is_master_cached` (lines 25–47)
- Modify `tools/rebuild_caches.sh` — inline Python heredoc replaced (lines 55–77)
- Modify `docs/cache_architecture.md` — new "Swapping a cached MC model" section
- Create `btc_web/test_mc_cache.py` — 9 unit tests + 1 integration test

**Runtime invariant for today:** Behavior identical to before the change. LPPL still bold, picking LPPL still hits cache, picking EPPL silent-misses (cache files don't yet exist for `ecfg_1d_1u`).

---

### Task 1: Add `_parse_cache_filename` to `btc_web/mc_cache.py`

**Files:**
- Create: `btc_web/test_mc_cache.py`
- Modify: `btc_web/mc_cache.py` (add after line 56 — `CACHE_DIR` definition)

- [ ] **Step 1: Create the test file with two failing tests**

Create `btc_web/test_mc_cache.py`:

```python
"""Tests for MC cache SSOT layer (_parse_cache_filename, _INTENDED_KEYS,
intended_models, MASTER_TO_CACHED_FALLBACK, is_master_cached, stash/commit/
restore)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))


def test_parse_cache_filename_handles_multi_underscore():
    """Model keys with internal underscores (ecfg_1d_1u, cfg_2d_1u) parse correctly."""
    from mc_cache import _parse_cache_filename

    assert _parse_cache_filename("paths_ecfg_1d_1u_2028.npz") == ("paths", "ecfg_1d_1u", 2028)
    assert _parse_cache_filename("overlays_cfg_2d_1u_2031.npz.bak") == ("overlays", "cfg_2d_1u", 2031)
    assert _parse_cache_filename("paths_bub_2028.npz") == ("paths", "bub", 2028)
    assert _parse_cache_filename("overlays_lppl_2035.npz") == ("overlays", "lppl", 2035)


def test_parse_cache_filename_rejects_garbage():
    """Non-cache filenames return None."""
    from mc_cache import _parse_cache_filename

    assert _parse_cache_filename("paths_2028.npz") is None       # no model key
    assert _parse_cache_filename("random.txt") is None
    assert _parse_cache_filename("") is None
    assert _parse_cache_filename("paths_bub_2028") is None       # missing .npz
    assert _parse_cache_filename("paths_bub_abcd.npz") is None   # year not 4 digits
```

- [ ] **Step 2: Run tests — confirm they fail with ImportError**

Run: `cd btc_web && PYTHONPATH=".:.." ../btc_venv/bin/python3 -m pytest test_mc_cache.py -v`
Expected: FAIL with `ImportError: cannot import name '_parse_cache_filename' from 'mc_cache'`

- [ ] **Step 3: Add the parser to `mc_cache.py`**

Edit `btc_web/mc_cache.py`. Find the line `CACHE_DIR = Path(__file__).parent / "mc_cache"` (line 56). Add immediately after:

```python
import re

_CACHE_FILE_RE = re.compile(r"^(paths|overlays)_(.+)_(\d{4})\.npz(?:\.bak)?$")


def _parse_cache_filename(name: str) -> tuple[str, str, int] | None:
    """Parse a cache filename into (kind, model_key, year), or None if not a cache file.

    Pure string parser — does not touch the filesystem. The 4-digit-year
    right-anchor disambiguates greedy `(.+)` capture so multi-underscore
    keys like `ecfg_1d_1u` parse correctly.
    """
    m = _CACHE_FILE_RE.match(name)
    return (m.group(1), m.group(2), int(m.group(3))) if m else None
```

- [ ] **Step 4: Run tests — confirm they pass**

Run: `cd btc_web && PYTHONPATH=".:.." ../btc_venv/bin/python3 -m pytest test_mc_cache.py -v`
Expected: PASS (2/2)

- [ ] **Step 5: Commit**

```bash
git add btc_web/test_mc_cache.py btc_web/mc_cache.py
git commit -m "feat(mc-cache): add _parse_cache_filename helper + tests

Pure string parser with anchored regex that handles multi-underscore
model keys like ecfg_1d_1u. Foundation for disk-derived
_CACHED_MODEL_KEYS in subsequent task.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Add `_INTENDED_KEYS`, `_ef_pkl_path`, and `intended_models(M)`

**Files:**
- Modify: `btc_web/mc_cache.py` (add after `_parse_cache_filename`)
- Modify: `btc_web/test_mc_cache.py` (append tests)

- [ ] **Step 1: Append failing tests**

Append to `btc_web/test_mc_cache.py`:

```python
def test_intended_models_keys_match_intended_set():
    """Drift guard: intended_models(M).keys() == _INTENDED_KEYS."""
    import _app_ctx, app  # noqa: F401  - boots the app, registers PRICE_MODELS
    from btc_core import load_model_data
    from mc_cache import intended_models, _INTENDED_KEYS

    M = load_model_data(str(Path(__file__).resolve().parent.parent / "model_data.pkl"))
    assert set(intended_models(M).keys()) == _INTENDED_KEYS


def test_intended_models_post_swap_keys():
    """Spec contract: _INTENDED_KEYS reflects the post-swap target."""
    from mc_cache import _INTENDED_KEYS

    assert _INTENDED_KEYS == frozenset({"bub", "qr", "pl", "ecfg_1d_1u", "ef"})


def test_intended_models_short_names_match_dict_keys():
    """Each instantiated model's .short_name must equal its dict key
    (cache files are named from short_name; mismatch would break lookup)."""
    import _app_ctx, app  # noqa: F401
    from btc_core import load_model_data
    from mc_cache import intended_models

    M = load_model_data(str(Path(__file__).resolve().parent.parent / "model_data.pkl"))
    for key, model in intended_models(M).items():
        assert model.short_name == key, (
            f"short_name mismatch: dict key {key!r} vs model.short_name {model.short_name!r}")
```

- [ ] **Step 2: Run — confirm failure**

Run: `cd btc_web && PYTHONPATH=".:.." ../btc_venv/bin/python3 -m pytest test_mc_cache.py -v`
Expected: FAIL with `ImportError: cannot import name '_INTENDED_KEYS' from 'mc_cache'`

- [ ] **Step 3: Add `_ef_pkl_path`, `_INTENDED_KEYS`, and `intended_models(M)` to `mc_cache.py`**

Edit `btc_web/mc_cache.py`. Add immediately after `_parse_cache_filename`:

```python
def _ef_pkl_path() -> Path:
    """Resolve <repo_root>/model_data_ef.pkl. Mirrors app.py:343."""
    return Path(__file__).parent.parent.parent / "model_data_ef.pkl"


# ── Single source of truth for what the next rebuild will cache ──
# Edited by hand. Used as a runtime sanity oracle (no model_data.pkl load).
# Compare _CACHED_MODEL_KEYS (disk-derived; what's actually on disk now).
_INTENDED_KEYS = frozenset({"bub", "qr", "pl", "ecfg_1d_1u", "ef"})


def intended_models(M) -> dict:
    """Return {short_name: model_instance} for the next rebuild target.

    Single source of truth consumed by tools/rebuild_caches.sh.
    Must instantiate exactly _INTENDED_KEYS — enforced by
    test_intended_models_keys_match_intended_set.
    """
    from btc_core import (BubbleModel, PowerLawModel,
                          EmpiricalFloorModel, QuantileRegressionModel,
                          EPPLConfigModel)
    return {
        "bub":         BubbleModel(M),
        "qr":          QuantileRegressionModel(M),
        "pl":          PowerLawModel(M.ols_intercept, M.ols_slope,
                                     M.price_years, M.price_prices,
                                     M.genesis, M.QR_QUANTILES),
        "ecfg_1d_1u":  EPPLConfigModel("ecfg_1d_1u",
                                       M.price_years, M.price_prices,
                                       M.QR_QUANTILES),
        "ef":          EmpiricalFloorModel(str(_ef_pkl_path())),
    }
```

The `parent.parent.parent` triple climb: `mc_cache.py` lives in `btc_web/` which lives in repo root. `__file__` → `btc_web/mc_cache.py` → `.parent` is `btc_web/`, `.parent.parent` is repo root.

- [ ] **Step 4: Verify the path helper from a quick smoke test**

Run:
```bash
cd btc_web && PYTHONPATH=".:.." ../btc_venv/bin/python3 -c "
from mc_cache import _ef_pkl_path
print(_ef_pkl_path())
print('exists:', _ef_pkl_path().exists())
"
```
Expected output (paths will differ for your checkout):
```
/scratch/code/bitcoinprojections/model_data_ef.pkl
exists: True
```
If `exists: False`, the parent-climb count is wrong; recount the directory levels.

- [ ] **Step 5: Run tests — confirm they pass**

Run: `cd btc_web && PYTHONPATH=".:.." ../btc_venv/bin/python3 -m pytest test_mc_cache.py -v`
Expected: PASS (5/5)

- [ ] **Step 6: Commit**

```bash
git add btc_web/test_mc_cache.py btc_web/mc_cache.py
git commit -m "feat(mc-cache): add _INTENDED_KEYS + intended_models(M) SSOT

Public function instantiates the post-swap target set. Drift guard
test ensures the dict and the frozenset stay in sync.

Reconciles existing 'ef' drift (was in _CACHED_MODEL_KEYS but missing
from the rebuild-script literal).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Add `MASTER_TO_CACHED_FALLBACK` + tests

**Files:**
- Modify: `btc_web/mc_cache.py` (add after `_INTENDED_KEYS`)
- Modify: `btc_web/test_mc_cache.py` (append tests)

- [ ] **Step 1: Append failing tests**

Append to `btc_web/test_mc_cache.py`:

```python
def test_master_to_cached_fallback_keys_in_dropdown():
    """Every fallback key must be a valid dropdown master."""
    from mc_cache import MASTER_TO_CACHED_FALLBACK
    from layout.heatmap import _HM_PILL_MODELS_BASE

    valid_masters = set(_HM_PILL_MODELS_BASE) | {"ef", "u1"}
    for master in MASTER_TO_CACHED_FALLBACK:
        assert master in valid_masters, (
            f"{master!r} in MASTER_TO_CACHED_FALLBACK but not in dropdown options")


def test_master_to_cached_fallback_values_in_intended():
    """Every fallback value must be a member of _INTENDED_KEYS (so it
    will exist on disk after the next rebuild)."""
    from mc_cache import MASTER_TO_CACHED_FALLBACK, _INTENDED_KEYS

    for master, fallback in MASTER_TO_CACHED_FALLBACK.items():
        assert fallback in _INTENDED_KEYS, (
            f"MASTER_TO_CACHED_FALLBACK[{master!r}] = {fallback!r} "
            f"is not in _INTENDED_KEYS = {_INTENDED_KEYS}")
```

The `lppl → lppl` entry will fail `test_master_to_cached_fallback_values_in_intended` because `_INTENDED_KEYS` no longer contains `"lppl"`. This is intentional — the spec calls the lppl entry a transition artifact. We add an exemption.

Append a third test that documents the exemption:

```python
def test_master_to_cached_fallback_lppl_transition_exemption():
    """The 'lppl' entry is a transition artifact: lppl is no longer in
    _INTENDED_KEYS but the entry stays until the rebuild purges
    paths_lppl_*.npz from prod. After purge, edit the dict to remove
    this entry (and delete this test).
    """
    from mc_cache import MASTER_TO_CACHED_FALLBACK, _INTENDED_KEYS

    # Today: "lppl" → "lppl" remains. This test passes iff the comment
    # invariant in mc_cache.py is upheld.
    assert MASTER_TO_CACHED_FALLBACK.get("lppl") == "lppl"
    assert "lppl" not in _INTENDED_KEYS
```

And update `test_master_to_cached_fallback_values_in_intended` to allow the `lppl` exemption:

```python
def test_master_to_cached_fallback_values_in_intended():
    """Every fallback value must be in _INTENDED_KEYS, EXCEPT the
    'lppl' transition artifact (see _lppl_transition_exemption)."""
    from mc_cache import MASTER_TO_CACHED_FALLBACK, _INTENDED_KEYS

    for master, fallback in MASTER_TO_CACHED_FALLBACK.items():
        if master == "lppl":
            continue  # transition artifact; tracked by separate test
        assert fallback in _INTENDED_KEYS, (
            f"MASTER_TO_CACHED_FALLBACK[{master!r}] = {fallback!r} "
            f"is not in _INTENDED_KEYS = {_INTENDED_KEYS}")
```

- [ ] **Step 2: Run — confirm failure**

Run: `cd btc_web && PYTHONPATH=".:.." ../btc_venv/bin/python3 -m pytest test_mc_cache.py -v`
Expected: FAIL with `ImportError: cannot import name 'MASTER_TO_CACHED_FALLBACK' from 'mc_cache'`

- [ ] **Step 3: Add `MASTER_TO_CACHED_FALLBACK` to `mc_cache.py`**

Edit `btc_web/mc_cache.py`. Add immediately after `intended_models(M)`:

```python
# Maps user-facing master keys (dropdown values) to their preferred
# cached variant. Each entry is a transition aid OR a post-rebuild target.
# REMOVE the "lppl" entry after rebuild confirms paths_lppl_*.npz are
# purged from prod (it becomes dead noise, would mislead a future editor).
MASTER_TO_CACHED_FALLBACK: dict[str, str] = {
    "lppl": "lppl",            # transition: kept until LPPL purged from disk
    "eppl": "ecfg_1d_1u",      # post-rebuild target
}
```

- [ ] **Step 4: Run tests — confirm they pass**

Run: `cd btc_web && PYTHONPATH=".:.." ../btc_venv/bin/python3 -m pytest test_mc_cache.py -v`
Expected: PASS (8/8)

- [ ] **Step 5: Commit**

```bash
git add btc_web/test_mc_cache.py btc_web/mc_cache.py
git commit -m "feat(mc-cache): add MASTER_TO_CACHED_FALLBACK SSOT dict

Maps user-facing dropdown masters (lppl, eppl) to their preferred
cached variants. Replaces inline LPPL fallback logic in resolver
(commit b28b8f4) — generalization comes in a later task.

The 'lppl' entry is a transition artifact; comment + dedicated
test track it for cleanup post-rebuild.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Convert `_CACHED_MODEL_KEYS` to disk-derived

**Files:**
- Modify: `btc_web/mc_cache.py` (replace literal at line 51 with derived computation)
- Modify: `btc_web/test_mc_cache.py` (append integration test)

- [ ] **Step 1: Append integration test**

Append to `btc_web/test_mc_cache.py`:

```python
@pytest.mark.skipif(
    not (Path(__file__).resolve().parent / "mc_cache").exists(),
    reason="mc_cache/ dir does not exist (fresh clone or test env)",
)
def test_cached_model_keys_match_disk_glob():
    """Live _CACHED_MODEL_KEYS matches what _parse_cache_filename
    extracts from real overlays_*.npz files in mc_cache/."""
    from mc_cache import _CACHED_MODEL_KEYS, CACHE_DIR, _parse_cache_filename

    expected = set()
    for f in CACHE_DIR.glob("overlays_*.npz"):
        parsed = _parse_cache_filename(f.name)
        if parsed is not None and parsed[0] == "overlays":
            expected.add(parsed[1])
    assert _CACHED_MODEL_KEYS == frozenset(expected), (
        f"disk has {expected}, _CACHED_MODEL_KEYS = {set(_CACHED_MODEL_KEYS)}")
```

- [ ] **Step 2: Run — confirm test passes against the existing literal**

The existing literal `frozenset(["bub", "qr", "pl", "lppl", "exp", "ef"])` happens to match what's on disk. Test should pass before the change.

Run: `cd btc_web && PYTHONPATH=".:.." ../btc_venv/bin/python3 -m pytest test_mc_cache.py::test_cached_model_keys_match_disk_glob -v`
Expected: PASS

- [ ] **Step 3: Replace the literal with disk-derived computation**

Edit `btc_web/mc_cache.py`. Find:

```python
_CACHED_MODEL_KEYS = frozenset(["bub", "qr", "pl", "lppl", "exp", "ef"])
```

Replace with:

```python
def _derive_cached_model_keys() -> frozenset[str]:
    """Build _CACHED_MODEL_KEYS from on-disk cache files.

    Globs overlays_*.npz (NOT paths_*) — overlay files are written second
    in generate_cache, so their presence implies both halves exist
    (avoids partial-write UI lie).

    Returns frozenset() if CACHE_DIR doesn't exist (fresh clone).
    """
    if not CACHE_DIR.exists():
        return frozenset()
    keys = set()
    for f in CACHE_DIR.glob("overlays_*.npz"):
        parsed = _parse_cache_filename(f.name)
        if parsed is not None and parsed[0] == "overlays":
            keys.add(parsed[1])
    return frozenset(keys)


_CACHED_MODEL_KEYS = _derive_cached_model_keys()
```

**Note:** `_derive_cached_model_keys` must come AFTER `_parse_cache_filename` and AFTER `CACHE_DIR` are defined. The replaced line was at line 51 in the original file; the new function and assignment go at the same approximate location BUT only after the parser (which Task 1 placed after `CACHE_DIR`). Check the file: parser should be ~line 60, the assignment `_CACHED_MODEL_KEYS = ...` was at line 51 — needs to MOVE to after the parser.

Concrete edit: delete the original line 51, then add the new function + assignment near the top of the helpers section (after `_parse_cache_filename`, before `_path_key_str`).

- [ ] **Step 4: Run all tests — confirm everything still passes**

Run: `cd btc_web && PYTHONPATH=".:.." ../btc_venv/bin/python3 -m pytest test_mc_cache.py test_callbacks.py -v`
Expected: PASS (no regressions). The integration test now exercises the same logic (parser + glob) that the runtime uses, so it's effectively a self-consistency check.

- [ ] **Step 5: Commit**

```bash
git add btc_web/test_mc_cache.py btc_web/mc_cache.py
git commit -m "refactor(mc-cache): _CACHED_MODEL_KEYS now disk-derived

Was a hardcoded frozenset literal; now globs mc_cache/overlays_*.npz
at module load and parses keys via _parse_cache_filename. Reflects
what's actually on disk (eliminates drift between intent and reality).

Behavior identical today — disk currently matches the old literal.
After rebuild day, this auto-flips without code changes (UI bolding,
resolver fallback, is_cached() all see the new set).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Add `is_master_cached` helper + tests

**Files:**
- Modify: `btc_web/mc_cache.py` (add after `MASTER_TO_CACHED_FALLBACK`)
- Modify: `btc_web/test_mc_cache.py` (append tests)

- [ ] **Step 1: Append failing tests**

Append to `btc_web/test_mc_cache.py`:

```python
def test_is_master_cached_direct_key(monkeypatch):
    """Sanity baseline: bub is bold when bub is in _CACHED_MODEL_KEYS."""
    import mc_cache
    monkeypatch.setattr(mc_cache, "_CACHED_MODEL_KEYS",
                        frozenset({"bub", "pl"}))
    assert mc_cache.is_master_cached("bub") is True


def test_is_master_cached_via_fallback(monkeypatch):
    """Master 'eppl' is bold when its alias 'ecfg_1d_1u' is on disk."""
    import mc_cache
    monkeypatch.setattr(mc_cache, "_CACHED_MODEL_KEYS",
                        frozenset({"ecfg_1d_1u"}))
    assert mc_cache.is_master_cached("eppl") is True


def test_is_master_cached_returns_false_when_uncached(monkeypatch):
    """Master with no direct cache and no usable fallback → not bold."""
    import mc_cache
    monkeypatch.setattr(mc_cache, "_CACHED_MODEL_KEYS", frozenset())
    assert mc_cache.is_master_cached("eppl") is False
    assert mc_cache.is_master_cached("hybppl") is False  # no fallback entry
    assert mc_cache.is_master_cached("bub") is False     # direct miss
```

- [ ] **Step 2: Run — confirm failure**

Run: `cd btc_web && PYTHONPATH=".:.." ../btc_venv/bin/python3 -m pytest test_mc_cache.py -v`
Expected: FAIL with `AttributeError: module 'mc_cache' has no attribute 'is_master_cached'`

- [ ] **Step 3: Add `is_master_cached` to `mc_cache.py`**

Edit `btc_web/mc_cache.py`. Add immediately after the `MASTER_TO_CACHED_FALLBACK` dict:

```python
def is_master_cached(key: str) -> bool:
    """True if a dropdown value has a usable cached variant on disk.

    True when either the key itself is in _CACHED_MODEL_KEYS (direct
    cache key like 'bub'), or its master-alias maps to one (e.g.
    'eppl' → 'ecfg_1d_1u').

    Used by the MC source dropdown bolding logic in
    layout/mc_controls.py::_mc_model_src_options.
    """
    return key in _CACHED_MODEL_KEYS or \
           MASTER_TO_CACHED_FALLBACK.get(key) in _CACHED_MODEL_KEYS
```

- [ ] **Step 4: Run tests — confirm they pass**

Run: `cd btc_web && PYTHONPATH=".:.." ../btc_venv/bin/python3 -m pytest test_mc_cache.py -v`
Expected: PASS (12/12, including 1 skipped if `mc_cache/` exists; 1 passed otherwise).

- [ ] **Step 5: Commit**

```bash
git add btc_web/test_mc_cache.py btc_web/mc_cache.py
git commit -m "feat(mc-cache): add is_master_cached helper

Factored bolding logic into a reusable helper so the MC source dropdown
(and any future UI surface) can ask 'is this dropdown value usable?'
in one call. Implements: direct key in _CACHED_MODEL_KEYS OR alias
target via MASTER_TO_CACHED_FALLBACK is.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Generalize `_resolve_mc_model_src` (master pin + generic fallback)

**Files:**
- Modify: `btc_web/callbacks/charts/_resolvers.py` (lines 213–250)
- Modify: `btc_web/test_mc_cache.py` (append sharpened test)

- [ ] **Step 1: Append the doubly-falsifiable test**

Append to `btc_web/test_mc_cache.py`:

```python
def test_resolver_eppl_master_falls_back_when_variant_uncached(monkeypatch):
    """Doubly falsifiable: catches both the dict lookup AND the master=src pin.

    Setup: monkey-patch _CACHED_MODEL_KEYS to NOT contain 'ecfg_1d_1u'.
    Stub `_resolve_hm_eppl_master` to return a different variant
    (e.g. 'ecfg_2dd_2dd') so the chain mutates the input.

    Expected: resolver returns 'ecfg_1d_1u' via MASTER_TO_CACHED_FALLBACK
    keyed on the ORIGINAL master 'eppl' (not the chain's mutated value).

    Falsification points:
        1. Remove "eppl" from MASTER_TO_CACHED_FALLBACK → returns the
           chain's mutated 'ecfg_2dd_2dd' instead of 'ecfg_1d_1u'.
        2. Remove the `master = src` pin → dict lookup keys on the
           mutated 'ecfg_2dd_2dd' (not 'eppl') → misses → returns
           'ecfg_2dd_2dd' instead of 'ecfg_1d_1u'.
    """
    import _app_ctx, app  # noqa: F401
    import mc_cache
    from callbacks.charts import _resolvers

    monkeypatch.setattr(mc_cache, "_CACHED_MODEL_KEYS",
                        frozenset({"bub", "pl"}))  # no ecfg_1d_1u

    # Stub the EPPL resolver to return a NON-default variant so we can
    # detect whether the chain's mutation leaks into the dict lookup.
    def fake_eppl_resolver(src, *_args, **_kwargs):
        return "ecfg_2dd_2dd" if src == "eppl" else src

    monkeypatch.setattr(_resolvers, "_resolve_hm_eppl_master",
                        fake_eppl_resolver)

    result = _resolvers._resolve_mc_model_src(
        "eppl",                          # master
        [], [], [],                      # lppl_n_freqs, weighted, no_13
        1, 1, "d", "d", "u", "u",        # hyb_a_*
        1, 1, "d", "d", "u", "u",        # ep_a_*
    )
    assert result == "ecfg_1d_1u", (
        f"Expected fallback to 'ecfg_1d_1u' via MASTER_TO_CACHED_FALLBACK['eppl']; "
        f"got {result!r}. Either the `master = src` pin is missing, or the dict "
        f"entry was lost.")
```

- [ ] **Step 2: Run — confirm it fails on the current LPPL-specific resolver**

Run: `cd btc_web && PYTHONPATH=".:.." ../btc_venv/bin/python3 -m pytest test_mc_cache.py::test_resolver_eppl_master_falls_back_when_variant_uncached -v`
Expected: FAIL — current code only special-cases `src == "lppl"`, so the EPPL master falls through to the chain's mutated value `"ecfg_2dd_2dd"`.

- [ ] **Step 3: Replace the inline LPPL block with the generic pattern**

Edit `btc_web/callbacks/charts/_resolvers.py`. Find the function body of `_resolve_mc_model_src` (lines 213–250). Replace lines 233–250 (everything inside the function body, after the docstring) with:

```python
    # Pin the master before the chain mutates `src` — required for the
    # MASTER_TO_CACHED_FALLBACK lookup below to key on the original input.
    # See test_resolver_eppl_master_falls_back_when_variant_uncached.
    from mc_cache import _CACHED_MODEL_KEYS, MASTER_TO_CACHED_FALLBACK
    master = src

    resolved = _resolve_hm_lppl_master(master, lppl_n_freqs, lppl_weighted, lppl_no_13)
    resolved = _resolve_hm_hybppl_master(
        resolved,
        hyb_a_nlog, hyb_a_ncal, hyb_a_log1d, hyb_a_log2d,
        hyb_a_cal1d, hyb_a_cal2d)
    resolved = _resolve_hm_eppl_master(
        resolved,
        ep_a_nlog, ep_a_ncal, ep_a_log1d, ep_a_log2d,
        ep_a_cal1d, ep_a_cal2d)
    if resolved not in _CACHED_MODEL_KEYS:
        # `.get(master, resolved)` default-to-resolved means: masters without
        # a fallback entry (e.g. hybppl) keep their resolved variant — silent
        # cache miss falls through to live compute on paid tier.
        resolved = MASTER_TO_CACHED_FALLBACK.get(master, resolved)
    return resolved
```

The docstring above the function should also be updated. Replace:

```
    LPPL special case: if the user picks the 'lppl' master without engaging
    the LPPL config modal, prefer the 1-frequency variant ('lppl') because
    it is in the precomputed MC cache. The default n_freqs=[3] would
    otherwise resolve to 'lp3' which is NOT cached → empty MC trace.
```

With:

```
    Master fallback: if the chain resolves to a variant not on disk, look
    up the master's preferred cached alias in MASTER_TO_CACHED_FALLBACK
    (e.g. 'lppl' → 'lppl', 'eppl' → 'ecfg_1d_1u'). Masters with no entry
    keep their resolved variant — silent cache miss falls through to live
    compute on paid tier.
```

- [ ] **Step 4: Run all resolver-related tests**

Run: `cd btc_web && PYTHONPATH=".:.." ../btc_venv/bin/python3 -m pytest test_mc_cache.py test_callbacks.py -v`
Expected: PASS (no regressions; the new test passes).

- [ ] **Step 5: Commit**

```bash
git add btc_web/test_mc_cache.py btc_web/callbacks/charts/_resolvers.py
git commit -m "refactor(mc-resolver): generic fallback via MASTER_TO_CACHED_FALLBACK

Replaces the LPPL-specific branch in _resolve_mc_model_src (commit
b28b8f4) with a generic dict lookup. Critical: master = src must be
pinned before the chain mutates src — without it, the dict lookup
keys on the wrong value.

Doubly-falsifiable test injects a chain mutation to detect both
the dict lookup AND the pin in a single test.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Update `_mc_model_src_options` to use `is_master_cached`

**Files:**
- Modify: `btc_web/layout/mc_controls.py` (lines 24–47, the `_mc_model_src_options` function)

- [ ] **Step 1: Inspect the current bolding logic**

Read `btc_web/layout/mc_controls.py` lines 24–47. Confirm the function uses inline `if k in _CACHED_MODEL_KEYS:` and the import at line 11 (`from mc_cache import (...)`) already includes `_CACHED_MODEL_KEYS`.

- [ ] **Step 2: Update the import + replace the inline check with `is_master_cached`**

Edit `btc_web/layout/mc_controls.py`. Find the current import block (around line 9–11):

```python
from mc_cache import (CACHED_START_YRS, WD_AMOUNTS,
                      ENTRY_PCT_BINS, MC_YEARS_OPTIONS, INFL_OPTIONS,
                      _CACHED_MODEL_KEYS)
```

Replace `_CACHED_MODEL_KEYS` with `is_master_cached` in the import:

```python
from mc_cache import (CACHED_START_YRS, WD_AMOUNTS,
                      ENTRY_PCT_BINS, MC_YEARS_OPTIONS, INFL_OPTIONS,
                      is_master_cached)
```

Then find the function body of `_mc_model_src_options` and replace its current bolding check. The current code is:

```python
    out = []
    for k in keys:
        if k not in _app_ctx.PRICE_MODELS:
            continue
        name = _app_ctx.PRICE_MODELS[k].name
        if k in _CACHED_MODEL_KEYS:
            label = html.Span(f" {name}", style={"fontWeight": "bold"})
        else:
            label = f" {name}"
        out.append({"label": label, "value": k})
    return out
```

Replace with:

```python
    out = []
    for k in keys:
        if k not in _app_ctx.PRICE_MODELS:
            continue
        name = _app_ctx.PRICE_MODELS[k].name
        if is_master_cached(k):
            label = html.Span(f" {name}", style={"fontWeight": "bold"})
        else:
            label = f" {name}"
        out.append({"label": label, "value": k})
    return out
```

- [ ] **Step 3: Smoke-test the dropdown options manually**

Run:
```bash
cd btc_web && PYTHONPATH=".:.." ../btc_venv/bin/python3 -c "
import _app_ctx, app
from layout.mc_controls import _mc_model_src_options
opts = _mc_model_src_options()
for o in opts:
    label = o['label']
    if hasattr(label, 'children'):
        bold = label.style.get('fontWeight') == 'bold'
        print(f\"{'**' if bold else '  '} {o['value']:8s} → {label.children!r}\")
    else:
        print(f\"   {o['value']:8s} → {label!r}\")
"
```

Expected: today (LPPL on disk, exp on disk, no eppl files), `bub`, `pl`, `lppl`, `ef` are bolded; `eppl`, `hybppl`, `pca`, `grdy`, `gomp`, `bpl`, `plo`, `sexp`, `logi` are not. (`exp` is in `_CACHED_MODEL_KEYS` today but not in the dropdown options — unaffected by bolding.)

If `eppl` shows bold today, the `_CACHED_MODEL_KEYS` derivation is wrong (no eppl files exist).
If `lppl` shows non-bold today, the disk derivation is failing to find LPPL files.

- [ ] **Step 4: Run the full test suite**

Run: `cd btc_web && PYTHONPATH=".:.." ../btc_venv/bin/python3 -m pytest test_mc_cache.py test_callbacks.py test_palette_roundtrip.py -v`
Expected: PASS, no regressions.

- [ ] **Step 5: Commit**

```bash
git add btc_web/layout/mc_controls.py
git commit -m "refactor(mc-ui): bolding uses is_master_cached helper

UI dropdown logic now matches the resolver's cache-availability check
(both consult MASTER_TO_CACHED_FALLBACK when the master isn't a direct
cache key). Behavior identical today; auto-flips on rebuild day without
further UI code changes.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Add `stash_stale_files` / `commit_stale_files` / `restore_stale_files`

**Files:**
- Modify: `btc_web/mc_cache.py` (add after `intended_models`)
- Modify: `btc_web/test_mc_cache.py` (append tests)

- [ ] **Step 1: Append failing tests**

Append to `btc_web/test_mc_cache.py`:

```python
def _make_fake_cache(tmp_path, model_keys, years=(2028,)):
    """Helper: create empty path/overlay files for given (model, year) combos."""
    cache = tmp_path / "mc_cache"
    cache.mkdir()
    for k in model_keys:
        for y in years:
            (cache / f"paths_{k}_{y}.npz").write_bytes(b"x")
            (cache / f"overlays_{k}_{y}.npz").write_bytes(b"x")
    return cache


def test_stash_commit_restore_roundtrip(tmp_path, monkeypatch):
    """Full lifecycle: stash → generate → commit deletes .bak.
    And: stash → restore reverts to original state."""
    import mc_cache

    # Mix of intended (bub) and stale (lppl) files.
    cache = _make_fake_cache(tmp_path, model_keys=("bub", "lppl"))
    monkeypatch.setattr(mc_cache, "CACHE_DIR", cache)
    monkeypatch.setattr(mc_cache, "_INTENDED_KEYS", frozenset({"bub"}))

    # Phase 1: stash
    mc_cache.stash_stale_files()
    assert (cache / "paths_bub_2028.npz").exists()         # untouched
    assert (cache / "overlays_bub_2028.npz").exists()
    assert not (cache / "paths_lppl_2028.npz").exists()    # renamed
    assert (cache / "paths_lppl_2028.npz.bak").exists()
    assert (cache / "overlays_lppl_2028.npz.bak").exists()

    # Phase 2a: commit (success path) → .bak deleted
    mc_cache.commit_stale_files()
    assert not (cache / "paths_lppl_2028.npz.bak").exists()
    assert not (cache / "overlays_lppl_2028.npz.bak").exists()
    assert (cache / "paths_bub_2028.npz").exists()         # still untouched

    # Reset and verify Phase 2b (restore)
    cache2 = _make_fake_cache(tmp_path / "second", model_keys=("bub", "lppl"))
    monkeypatch.setattr(mc_cache, "CACHE_DIR", cache2)
    mc_cache.stash_stale_files()
    assert (cache2 / "paths_lppl_2028.npz.bak").exists()
    mc_cache.restore_stale_files()
    assert (cache2 / "paths_lppl_2028.npz").exists()       # restored
    assert not (cache2 / "paths_lppl_2028.npz.bak").exists()


def test_full_cleanup_sequence_noop_on_missing_dir(tmp_path, monkeypatch):
    """All three phases must handle a nonexistent CACHE_DIR cleanly."""
    import mc_cache

    missing = tmp_path / "does_not_exist"
    monkeypatch.setattr(mc_cache, "CACHE_DIR", missing)

    # Each phase must return cleanly with no exception.
    mc_cache.stash_stale_files()
    mc_cache.commit_stale_files()
    mc_cache.restore_stale_files()

    assert not missing.exists()  # no side-effect creation
```

- [ ] **Step 2: Run — confirm failure**

Run: `cd btc_web && PYTHONPATH=".:.." ../btc_venv/bin/python3 -m pytest test_mc_cache.py -v -k "stash or restore or cleanup"`
Expected: FAIL with `AttributeError: module 'mc_cache' has no attribute 'stash_stale_files'`

- [ ] **Step 3: Add the three functions to `mc_cache.py`**

Edit `btc_web/mc_cache.py`. Add immediately after the `intended_models(M)` function:

```python
def stash_stale_files() -> None:
    """Rename cache files for models NOT in _INTENDED_KEYS to *.npz.bak.

    Atomic and reversible — call commit_stale_files() on success or
    restore_stale_files() on interrupt. Idempotent w.r.t. missing CACHE_DIR.
    Logs each rename via print() (consistent with existing build progress).
    """
    if not CACHE_DIR.exists():
        return
    for f in list(CACHE_DIR.glob("*.npz")):
        parsed = _parse_cache_filename(f.name)
        if parsed is None:
            continue
        _kind, model_key, _year = parsed
        if model_key not in _INTENDED_KEYS:
            bak = f.with_suffix(f.suffix + ".bak")
            f.rename(bak)
            print(f"  stash: {f.name} -> {bak.name}")


def commit_stale_files() -> None:
    """Delete all *.npz.bak files. Call only after generate succeeds."""
    if not CACHE_DIR.exists():
        return
    for bak in list(CACHE_DIR.glob("*.npz.bak")):
        bak.unlink()
        print(f"  commit: deleted {bak.name}")


def restore_stale_files() -> None:
    """Rename *.npz.bak back to *.npz. Call on interrupt or error.
    Idempotent — running twice is a no-op."""
    if not CACHE_DIR.exists():
        return
    for bak in list(CACHE_DIR.glob("*.npz.bak")):
        original = bak.with_suffix("")  # strips .bak, leaves .npz
        if original.exists():
            print(f"  restore: skipping {bak.name} (target {original.name} exists)")
            continue
        bak.rename(original)
        print(f"  restore: {bak.name} -> {original.name}")
```

- [ ] **Step 4: Run tests — confirm they pass**

Run: `cd btc_web && PYTHONPATH=".:.." ../btc_venv/bin/python3 -m pytest test_mc_cache.py -v`
Expected: PASS (all tests).

- [ ] **Step 5: Commit**

```bash
git add btc_web/test_mc_cache.py btc_web/mc_cache.py
git commit -m "feat(mc-cache): atomic stash/commit/restore for stale files

Two-phase rename pattern: stash renames files for non-_INTENDED_KEYS
models to .bak; commit deletes them on success; restore reverts on
interrupt. Makes the 2-4 hour rebuild interrupt-safe.

All three functions guard 'if not CACHE_DIR.exists()'.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: Update `tools/rebuild_caches.sh`

**Files:**
- Modify: `tools/rebuild_caches.sh` (lines 55–77)

- [ ] **Step 1: Read the current script body**

Confirm the current heredoc structure at lines 55–77.

- [ ] **Step 2: Replace the inline literal with the new flow**

Edit `tools/rebuild_caches.sh`. Find the block:

```bash
if [ "$BUILD_MC" = "true" ]; then
    echo "▶ Building MC cache (expect 2-4 hours)..."
    if [ -f btc_web/mc_cache.py ]; then
        PYTHONPATH=".:btc_web" btc_venv/bin/python3 -c "
import _app_ctx
from btc_core import load_model_data
import btc_web.mc_cache as mc
M = load_model_data('model_data.pkl')
from btc_core import BubbleModel, PowerLawModel, LPPLModel, ExponentialModel, S2FModel, EmpiricalFloorModel, QuantileRegressionModel
models = {
    'bub': BubbleModel(M),
    'qr':  QuantileRegressionModel(M),
    'pl':  PowerLawModel(M.ols_intercept, M.ols_slope, M.price_years,
                        M.price_prices, M.genesis, M.QR_QUANTILES),
    'lppl': LPPLModel(M.price_years, M.price_prices, M.QR_QUANTILES),
    'exp': ExponentialModel(M.price_years, M.price_prices, M.QR_QUANTILES),
}
mc.generate_all_caches(M, models)
"
    else
        echo "MC cache builder not found — skipping."
    fi
fi
```

Replace with:

```bash
if [ "$BUILD_MC" = "true" ]; then
    echo "▶ Building MC cache (expect 2-4 hours)..."
    if [ -f btc_web/mc_cache.py ]; then
        PYTHONPATH=".:btc_web" btc_venv/bin/python3 -c "
import _app_ctx
from btc_core import load_model_data
import btc_web.mc_cache as mc

M = load_model_data('model_data.pkl')
models = mc.intended_models(M)

mc.stash_stale_files()
try:
    mc.generate_all_caches(M, models)
    mc.commit_stale_files()
except BaseException:
    mc.restore_stale_files()
    raise
"
    else
        echo "MC cache builder not found — skipping."
    fi
fi
```

- [ ] **Step 3: Lint-check the shell script**

Run: `bash -n tools/rebuild_caches.sh`
Expected: no output (silent success).

- [ ] **Step 4: Confirm the script's `--help` output is unchanged**

Run: `bash tools/rebuild_caches.sh --help | head -20`
Expected: standard help output (just confirming the script still parses).

- [ ] **Step 5: Commit**

```bash
git add tools/rebuild_caches.sh
git commit -m "refactor(rebuild): use intended_models + stash/commit/restore

No more inline 'models = {...}' literal. Single source of truth lives
in mc_cache.intended_models(M). Stash renames stale files before
generate; commit deletes .bak on success; restore reverts on interrupt.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: Add workflow doc section to `docs/cache_architecture.md`

**Files:**
- Modify: `docs/cache_architecture.md`

- [ ] **Step 1: Inspect the current doc to find an appropriate insertion point**

Read `docs/cache_architecture.md`. Find a section like "MC cache" or "Rebuilding". Identify the heading hierarchy (e.g., `##` vs `###`) so the new section fits.

- [ ] **Step 2: Append the new section at end of file**

Append to `docs/cache_architecture.md`:

```markdown
## Swapping a cached MC model

The MC cache is keyed by `model.short_name`. To swap (e.g., drop `lppl`,
add `ecfg_1d_1u`):

1. **Edit `btc_web/mc_cache.py`**:
   - Update `_INTENDED_KEYS` (frozenset of `short_name`s for the next rebuild).
   - Update `intended_models(M)` to instantiate exactly those keys.
   - If introducing a master alias (a dropdown value that maps to a different cached variant), add an entry to `MASTER_TO_CACHED_FALLBACK`.

   **Don't confuse**: `_INTENDED_KEYS` is hand-edited (intent for next rebuild). `_CACHED_MODEL_KEYS` is disk-derived (what's actually on disk now). They will differ during the transition window between code change and rebuild.

2. **Run tests**: `btc_venv/bin/python3 -m pytest btc_web/test_mc_cache.py -v`. All must pass before invoking the rebuild.

3. **Rebuild**: `bash tools/rebuild_caches.sh --mc` (2–4 hours, interrupt-safe via stash/commit/restore).

4. **Verify locally — split by transition state**:
   - **Code-change day** (before rebuild): tests pass; app starts; the MC source dropdown bolds the *old* cached set (disk hasn't changed); picking the *new* master falls through to silent live-compute miss on free tier (acceptable transient).
   - **Rebuild day** (after generate completes): app starts; dropdown bolding auto-flips; picking the new master renders a trace.

5. **Deploy**: `tools/rebuild_caches.sh` rsyncs the new files to prod and emits a reminder to restart. Manual: `ssh root@... "redis-cli FLUSHDB && systemctl restart quantoshi"`.

6. **Smoke test on prod**: `/1`, enable MC, pick the new master, confirm a trace renders.

After rebuild lands, edit `MASTER_TO_CACHED_FALLBACK` to remove any transition-only entries (look for the `# transition: ...` comment) — they otherwise become dead noise that misleads future editors.
```

- [ ] **Step 3: Verify markdown renders**

Run: `head -200 docs/cache_architecture.md | tail -50` to spot-check the appended section.

- [ ] **Step 4: Commit**

```bash
git add docs/cache_architecture.md
git commit -m "docs(cache): add 'Swapping a cached MC model' workflow

Six-step procedure for adding/removing a model in the precomputed MC
cache. Captures the SSOT pattern (_INTENDED_KEYS, intended_models,
MASTER_TO_CACHED_FALLBACK) and the transition-state caveat (UI bolding
reflects on-disk state, not intent).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 11: Full-suite verification + CHECKPOINT

**Files:** None (verification + user gate)

- [ ] **Step 1: Run the full non-E2E test suite**

Run: `cd btc_web && PYTHONPATH=".:.." ../btc_venv/bin/python3 -m pytest --timeout=60 --ignore-glob='*_e2e.py' 2>&1 | tail -5`
Expected: PASS with the same count as before the change, plus the new tests in `test_mc_cache.py`. The two pre-existing failures (`test_fingerprint_is_stable`, `test_no_hex_literals_outside_colors_module`) remain unchanged — they are unrelated to this work.

- [ ] **Step 2: Boot the app and check log for unexpected errors**

Run: `cd btc_web && PYTHONPATH=".:.." ../btc_venv/bin/python3 -c "import _app_ctx, app; print('OK')"`
Expected: `OK` printed, no traceback.

- [ ] **Step 3: Confirm the dropdown behavior didn't change today**

Re-run the smoke from Task 7 step 3:
```bash
cd btc_web && PYTHONPATH=".:.." ../btc_venv/bin/python3 -c "
import _app_ctx, app
from layout.mc_controls import _mc_model_src_options
for o in _mc_model_src_options():
    bold = hasattr(o['label'], 'children')
    print(f\"{'BOLD' if bold else '----'} {o['value']}\")"
```
Expected: bub, pl, lppl, ef bold; eppl, hybppl, pca, grdy, gomp, bpl, plo, sexp, logi not bold. (Same as before this work.)

- [ ] **Step 4: CHECKPOINT — present diff and test results to user**

Show the user:
- `git log --oneline origin/master..HEAD` (the new commits in order)
- `git diff origin/master --stat` (cumulative line counts)
- Test output: total passes/fails

Wait for user approval before pushing.

---

### Task 12: Push to origin + standard deploy

**Files:** None (deployment)

- [ ] **Step 1: Push to origin**

Run:
```bash
git push origin master
```

- [ ] **Step 2: Pull, FLUSHDB, restart prod**

Run:
```bash
ssh root@89.167.70.45 "cd /opt/quantoshi && git pull && redis-cli FLUSHDB && systemctl restart quantoshi"
```
Expected: `Already up to date.` or fast-forward summary, then `OK` from redis-cli.

- [ ] **Step 3: Regenerate Citadel cache (per CLAUDE.md ops convention)**

Run:
```bash
ssh root@89.167.70.45 "PYTHONPATH=/opt/quantoshi:/opt/quantoshi/btc_web /opt/quantoshi/btc_venv/bin/python3 /opt/quantoshi/btc_web/generate_citadel_cache.py"
```
Expected: `Done: 19/19 results in ~6s`.

- [ ] **Step 4: Verify prod**

Curl the homepage:
```bash
curl -sI https://quantoshi.xyz/1 | head -1
```
Expected: `HTTP/2 200`.

Optional manual: open `https://quantoshi.xyz/1`, enable MC, pick LPPL master, confirm trace still renders (proves nothing regressed for the cached path). Pick EPPL master, confirm silent miss with no trace (transition-state expected behavior).

---

## Self-review checklist (run after writing the plan)

**Spec coverage:** All 7 deliverables from the spec's "Today's deliverable" section have a corresponding task: parser (T1), `_INTENDED_KEYS`+`intended_models` (T2), `MASTER_TO_CACHED_FALLBACK` (T3), disk-derived `_CACHED_MODEL_KEYS` (T4), `is_master_cached` (T5), resolver simplification (T6), UI bolding update (T7), stash/commit/restore (T8), build script (T9), workflow doc (T10). ✓

**Placeholder scan:** No "TBD", "TODO", "implement later" in any task. Every code step shows complete code. ✓

**Type consistency:** `_parse_cache_filename` returns `tuple[str, str, int] | None` everywhere. `intended_models(M)` returns `dict` consistently. `MASTER_TO_CACHED_FALLBACK` is `dict[str, str]` everywhere. `is_master_cached(key: str) -> bool` consistent. ✓

**Critical pin documented:** Task 6 step 3 explicitly notes the `master = src` pin must be the first statement before any chain call. The doubly-falsifiable test in Task 6 step 1 catches both the dict lookup AND the pin in a single test. ✓
