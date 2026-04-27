# Time-Basis Toggle — Phase 1 (Plumbing) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the site-wide time-basis configuration plumbing (TOML + `time_basis.py` module + cache/snapshot fingerprint extensions + pkl schema fields + a JSON metadata sidecar) without changing any model fits or chart behavior. Default `time_basis = "calendar"`. Site stays green for normal users; admin gains a future flip switch.

**Architecture:** New `quantoshi.toml` at repo root (parsed via stdlib `tomllib`). New `btc_web/time_basis.py` exposes `TIME_BASIS`, axis constants, and `calendar_to_t()` / `t_to_calendar()` helpers. `cache.py::_fp()` and the L0 fingerprint include `TIME_BASIS` in the keyspace prefix so calendar/block caches will never collide once Phase 2 produces a block pkl. `snapshot_defaults.py::_compute_snapshot_defaults_fingerprint()` adds `TIME_BASIS` to its hash input — calendar mode gets a new 8-char fingerprint (defensive only; cross-axis decode rejection arrives in Phase 3). The model build pipeline emits a small `model_data_meta.json` sidecar carrying `time_basis`, `t_label`, `t_per_year`, `t_origin` so consumers (and ops) can introspect the active axis without loading the pkl.

**Tech Stack:** Python 3.14 (dev) / 3.12 (prod), `tomllib` (stdlib), pytest, Plotly Dash 4.0.0, existing test suite (~1456 tests). All work on branch `time-basis-toggle`.

**Spec:** [`docs/superpowers/specs/2026-04-26-time-basis-toggle-design.md`](../specs/2026-04-26-time-basis-toggle-design.md). This plan implements §4 Phase 1 only. Phases 2–5 are separate plans created after Phase 1 ships and Phase 2's offline R²/AIC comparison decides whether to proceed.

**Note on pkl schema fields**: the spec (§3.5) lists `time_basis`, `t_label`, `t_per_year`, `t_origin` inside `model_data.pkl`. This plan writes them inside the pkl AND mirrors them in `model_data_meta.json` next to the pkl. The JSON sidecar is the queryable interface (used by tests, ops, and any non-Python consumer); the in-pkl fields are kept for in-process consumers that already hold an open pkl handle.

---

## File Structure

**Create:**
- `quantoshi.toml` (repo root) — admin preferences. Three fields for now: `time_basis`, `block_origin`, `blocks_per_year`.
- `btc_web/time_basis.py` — single source of truth for axis-derived constants + conversion helpers. Loads `quantoshi.toml` at import time. Graceful fallback to calendar if file missing.
- `btc_web/test_time_basis.py` — unit tests for the new module: load behavior, constants under each basis, conversion round-trips.
- `btc_web/test_time_basis_integration.py` — integration tests: cache fingerprint includes axis, snapshot fingerprint reserves slot, JSON sidecar shape, `engines/custom_fit.py` does not import the module.
- `model_data_meta.json` (repo root, written by build pipeline) — JSON mirror of pkl axis metadata.

**Modify:**
- `btc_web/_app_ctx.py` — re-export the time_basis constants for callers that already import from `_app_ctx` for singletons.
- `btc_web/cache.py` — extend cache key prefixes (line 35), L0 fingerprint (line 133), and Citadel L2 keys (lines 106, 118) to include `TIME_BASIS`.
- `btc_web/snapshot_defaults.py` — extend `_compute_snapshot_defaults_fingerprint()` (line 385) hash input to include `TIME_BASIS`.
- `btc_web/snapshot_defaults_registry.json` — regenerated via existing `tools/update_defaults_registry.py` workflow.
- `tools/model_toolkit/export.py` — write the four schema fields into both the pkl and a sidecar `model_data_meta.json`.
- `tools/build_bm_model.py` — pass schema metadata through to `export.py`.

**Untouched (deliberate):**
- `btc_web/engines/custom_fit.py` — CTA stays per-fit user-controlled; Phase 1 verifies via test that it does not import `time_basis`.
- `btc_core/*` — no model class changes in Phase 1. The `T_MIN` sweep is Phase 2 work.
- `BitcoinBlocksDaily.csv` — no backfill needed for Phase 1; `T_ORIGIN_BLOCK` is resolved in Task 1 and pinned literally.

---

## Task 1: Resolve T_ORIGIN_BLOCK and create quantoshi.toml

**Files:**
- Create: `quantoshi.toml`
- Read-only: `BitcoinBlocksDaily.csv`, `tools/build_block_map.py`

**Goal:** Determine the block height at 2009-07-25 UTC and pin it as a literal constant in the new config file. Both price and block CSVs start at 2010-07-17, so 2009-07-25 is pre-CSV and must come from authoritative chain data.

- [ ] **Step 1: Determine the block height at 2009-07-25 UTC**

Pick whichever method fits the local environment:

**Method A (preferred — uses existing tooling):** if a local `bitcoind` is running with full chain history, query it directly:

```bash
TARGET=$(date -u -d "2009-07-26 00:00:00" +%s)
for h in $(seq 17000 18000); do
  hash=$(bitcoin-cli getblockhash "$h")
  ts=$(bitcoin-cli getblockheader "$hash" | btc_venv/bin/python3 -c \
       'import sys,json; print(json.load(sys.stdin)["time"])')
  if [ "$ts" -ge "$TARGET" ]; then
    echo "Block $((h-1)) is the last 2009-07-25 block"
    break
  fi
done
```

**Method B (no bitcoind):** consult two independent block explorers (e.g., blockchair.com, mempool.space) for "the highest block height with timestamp on 2009-07-25 UTC." Both must agree. Record the URL in a TOML comment for traceability.

**Method C (lazy):** the spec's placeholder `17448` is within ±100 of the true value; for an A/B comparison this precision is irrelevant (≈8 hours of chain time vs the 16-year fit window). If no rigorous lookup is possible, accept `17448` and document the imprecision in a TOML comment.

Acceptance: a single integer `<block>` whose source is documented (RPC output, URL, or "spec placeholder").

- [ ] **Step 2: Write the failing test for `quantoshi.toml` existence and field shape**

Create `btc_web/test_time_basis.py`:

```python
"""Phase 1 plumbing tests for time_basis configuration."""
from __future__ import annotations
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TOML = _REPO_ROOT / "quantoshi.toml"


def test_quantoshi_toml_exists():
    assert _TOML.exists(), f"{_TOML} should exist after Phase 1"


def test_quantoshi_toml_has_required_fields():
    with open(_TOML, "rb") as f:
        cfg = tomllib.load(f)
    assert cfg["time_basis"] in ("calendar", "block")
    assert isinstance(cfg["block_origin"], int)
    assert 17000 <= cfg["block_origin"] <= 18000  # sanity bounds
    assert cfg["blocks_per_year"] == 52596


def test_quantoshi_toml_default_is_calendar():
    """Default ships as calendar so Phase 1 changes nothing user-visible."""
    with open(_TOML, "rb") as f:
        cfg = tomllib.load(f)
    assert cfg["time_basis"] == "calendar"
```

- [ ] **Step 3: Run tests to verify they fail (file does not exist yet)**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_time_basis.py -v
```

Expected: 3 FAILs with `FileNotFoundError` or `AssertionError: <quantoshi.toml> should exist`.

- [ ] **Step 4: Create `quantoshi.toml`**

Substitute `<block>` from Step 1 into the `block_origin` line:

```toml
# quantoshi.toml — site-wide admin preferences.
# Read by btc_web/ + tools/build_*.py via stdlib tomllib at import time.
# Edit-then-rebuild semantics: changing time_basis here requires
# rebuilding model_data.pkl + restarting the app.

# Canonical training axis for site models.
# "calendar" — t = years since 2009-07-25 (current site default)
# "block"    — t = blocks since block_origin (Phase 2+ only; do not flip
#              until model_data_block.pkl exists)
time_basis = "calendar"

# Block height at 2009-07-25 UTC (the calendar time origin).
# Pinned for reproducibility — DO NOT derive at runtime.
# Source: <RPC | blockchair URL | mempool URL | "spec placeholder ±100">
block_origin = <block>

# Forward block-rate constant: 144 blocks/day × 365.25 days/year.
# Used only to project FUTURE block heights from calendar dates.
# Training data uses real observed heights from BitcoinBlocksDaily.csv.
blocks_per_year = 52596
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_time_basis.py -v
```

Expected: 3 PASS.

- [ ] **Step 6: Commit**

```bash
git add quantoshi.toml btc_web/test_time_basis.py
git commit -m "feat(time-basis): add quantoshi.toml admin config

Pins time_basis (default calendar), block_origin at 2009-07-25,
and blocks_per_year=52596 (144 blocks/day × 365.25). Phase 1 of
the time-basis toggle spec — no behavior change.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Create `btc_web/time_basis.py` module

**Files:**
- Create: `btc_web/time_basis.py`
- Modify: `btc_web/test_time_basis.py` (append tests)

**Goal:** A small, importable module that loads the TOML once at import, exposes axis-derived constants, and provides `calendar_to_t()` / `t_to_calendar()` helpers. Block-mode uses `T_PER_YEAR` (52596) for the conversion since Phase 1 has no real chain data dependency yet.

- [ ] **Step 1: Append failing tests for the module API**

Append to `btc_web/test_time_basis.py`:

```python
import datetime as _dt


def test_module_imports_with_calendar_default():
    from btc_web import time_basis as tb
    assert tb.TIME_BASIS == "calendar"
    assert tb.T_LABEL == "years"
    assert tb.T_PER_YEAR == 1.0
    assert tb.T_MIN == 1.0
    assert tb.T_ORIGIN_DATE == _dt.date(2009, 7, 25)
    assert isinstance(tb.T_ORIGIN_BLOCK, int)


def test_calendar_to_t_calendar_mode():
    from btc_web import time_basis as tb
    assert tb.TIME_BASIS == "calendar"
    # 2009-07-25 → t=0
    assert tb.calendar_to_t(_dt.date(2009, 7, 25)) == 0.0
    # 2010-07-25 → t≈1.0 (one year)
    t = tb.calendar_to_t(_dt.date(2010, 7, 25))
    assert abs(t - 1.0) < 1e-6


def test_t_to_calendar_calendar_mode():
    from btc_web import time_basis as tb
    assert tb.t_to_calendar(0.0) == _dt.date(2009, 7, 25)
    # t=1 year → close to 2010-07-25 (within 1 day for 365.25 rounding)
    d = tb.t_to_calendar(1.0)
    assert abs((d - _dt.date(2010, 7, 25)).days) <= 1


def test_round_trip_calendar_mode():
    from btc_web import time_basis as tb
    for d in [_dt.date(2010, 1, 1), _dt.date(2024, 12, 31),
              _dt.date(2050, 6, 15)]:
        t = tb.calendar_to_t(d)
        d2 = tb.t_to_calendar(t)
        assert abs((d - d2).days) <= 1


def test_block_mode_constants(monkeypatch):
    """Verify block-mode constants without rewriting the TOML."""
    from btc_web import time_basis as tb
    monkeypatch.setattr(tb, "TIME_BASIS", "block")
    monkeypatch.setattr(tb, "T_LABEL", "blocks")
    monkeypatch.setattr(tb, "T_PER_YEAR", 52596.0)
    monkeypatch.setattr(tb, "T_MIN", 52596.0)
    # Recompute conversions under patched globals
    t = tb.calendar_to_t(_dt.date(2010, 7, 25))
    assert abs(t - 52596.0) < 1.0  # one year ≈ 52596 blocks


def test_load_config_returns_dict_with_required_keys():
    from btc_web import time_basis as tb
    cfg = tb._load_config()
    assert "time_basis" in cfg
    assert "block_origin" in cfg
    assert "blocks_per_year" in cfg


def test_load_config_with_missing_file_falls_back_to_default(tmp_path):
    from btc_web import time_basis as tb
    cfg = tb._load_config(tmp_path / "nonexistent.toml")
    assert cfg["time_basis"] == "calendar"
    assert cfg["blocks_per_year"] == 52596
```

- [ ] **Step 2: Run tests to verify they fail (module does not exist)**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_time_basis.py -v
```

Expected: 7 FAILs (`ModuleNotFoundError: No module named 'btc_web.time_basis'`) on the new tests; original 3 still pass.

- [ ] **Step 3: Create `btc_web/time_basis.py`**

```python
"""Site-wide time-axis configuration (Phase 1 plumbing).

Loads quantoshi.toml at import time and exposes axis-derived constants
plus calendar-to-t conversion helpers. This module is the single source
of truth for which axis the canonical site model was trained on.

Imported by: btc_core, tools/build_*.py, btc_web/_app_ctx, btc_web/cache,
btc_web/snapshot_defaults.

NOT imported by: btc_web/engines/custom_fit (CTA stays per-fit
user-controlled; verified by integration test).

Spec: docs/superpowers/specs/2026-04-26-time-basis-toggle-design.md §3.2
"""
from __future__ import annotations

import datetime as _dt
import logging
import sys
from pathlib import Path
from typing import Literal, Optional

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover — prod is Py 3.12, dev is 3.14
    import tomli as tomllib

_LOG = logging.getLogger("time_basis")

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TOML_PATH = _REPO_ROOT / "quantoshi.toml"

_DEFAULTS = {
    "time_basis": "calendar",
    "block_origin": 17448,        # spec placeholder; pinned in TOML
    "blocks_per_year": 52596,     # 144 × 365.25
}


def _load_config(path: Optional[Path] = None) -> dict:
    """Load quantoshi.toml, falling back to _DEFAULTS if missing.

    Public for testing — production callers should use the module-level
    constants below, which are computed once at import time.
    """
    p = path if path is not None else _TOML_PATH
    if not p.exists():
        _LOG.warning("time_basis: %s not found; using defaults (calendar)", p)
        return dict(_DEFAULTS)
    with open(p, "rb") as f:
        cfg = tomllib.load(f)
    return {**_DEFAULTS, **cfg}


_cfg = _load_config()

TIME_BASIS: Literal["calendar", "block"] = _cfg["time_basis"]
T_ORIGIN_DATE: _dt.date = _dt.date(2009, 7, 25)
T_ORIGIN_BLOCK: int = int(_cfg["block_origin"])
T_LABEL: str = "years" if TIME_BASIS == "calendar" else "blocks"
T_PER_YEAR: float = 1.0 if TIME_BASIS == "calendar" else float(_cfg["blocks_per_year"])
T_MIN: float = 1.0 if TIME_BASIS == "calendar" else float(_cfg["blocks_per_year"])

_LOG.info(
    "time_basis: %s (T_LABEL=%s, T_PER_YEAR=%g, T_ORIGIN_BLOCK=%d)",
    TIME_BASIS, T_LABEL, T_PER_YEAR, T_ORIGIN_BLOCK,
)


def calendar_to_t(d: _dt.date) -> float:
    """Convert a calendar date to t in the active basis.

    Calendar mode: t = years since 2009-07-25 (`days / 365.25`).
    Block mode: t = projected block offset, using protocol-target rate
                (T_PER_YEAR blocks per year) since the calendar origin.

    Used at the simulator boundary (Citadel, MC year-stepping) to convert
    user-entered calendar years into model-axis t.
    """
    days = (d - T_ORIGIN_DATE).days
    years = days / 365.25
    return years if TIME_BASIS == "calendar" else years * T_PER_YEAR


def t_to_calendar(t: float) -> _dt.date:
    """Convert t (active basis) back to calendar date for chart display.

    Inverse of calendar_to_t (modulo integer-day rounding).
    """
    years = t if TIME_BASIS == "calendar" else t / T_PER_YEAR
    days = years * 365.25
    return T_ORIGIN_DATE + _dt.timedelta(days=days)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_time_basis.py -v
```

Expected: 10 PASS (3 from Task 1 + 7 new).

- [ ] **Step 5: Commit**

```bash
git add btc_web/time_basis.py btc_web/test_time_basis.py
git commit -m "feat(time-basis): add btc_web/time_basis.py module

Loads quantoshi.toml at import. Exposes TIME_BASIS, T_ORIGIN_DATE,
T_ORIGIN_BLOCK, T_LABEL, T_PER_YEAR, T_MIN, plus calendar_to_t /
t_to_calendar helpers. Phase 1 plumbing — no consumers yet.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Re-export from `_app_ctx.py`

**Files:**
- Modify: `btc_web/_app_ctx.py` (append imports near other singletons)
- Modify: `btc_web/test_time_basis.py` (append integration test)

**Goal:** Existing callers that already import singleton flags from `_app_ctx` (e.g., `_HAS_REDIS`, `_MODEL_FP`) get a one-line path to the new constants too. Pure convenience re-export — `btc_web.time_basis` remains the canonical home.

- [ ] **Step 1: Append failing test**

Append to `btc_web/test_time_basis.py`:

```python
def test_app_ctx_re_exports_time_basis_constants():
    from btc_web import _app_ctx
    from btc_web import time_basis as tb
    assert _app_ctx.TIME_BASIS == tb.TIME_BASIS
    assert _app_ctx.T_LABEL == tb.T_LABEL
    assert _app_ctx.T_PER_YEAR == tb.T_PER_YEAR
    assert _app_ctx.T_MIN == tb.T_MIN
    assert _app_ctx.T_ORIGIN_DATE == tb.T_ORIGIN_DATE
    assert _app_ctx.T_ORIGIN_BLOCK == tb.T_ORIGIN_BLOCK
```

- [ ] **Step 2: Run test to verify it fails**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_time_basis.py::test_app_ctx_re_exports_time_basis_constants -v
```

Expected: FAIL with `AttributeError: module 'btc_web._app_ctx' has no attribute 'TIME_BASIS'`.

- [ ] **Step 3: Add re-export to `_app_ctx.py`**

Read `btc_web/_app_ctx.py` to find the singleton-flag block (near `_HAS_REDIS`, `_MODEL_FP`). Append after line 161 (or wherever `_MODEL_FP = _compute_model_fingerprint()` lands), grouped with the other module-level singletons:

```python
# ─────────────────────────────────────────────────────────────────
# Time basis (re-exported from btc_web.time_basis for caller convenience).
# Canonical home is btc_web/time_basis.py; this is just a singleton alias.
# ─────────────────────────────────────────────────────────────────
from time_basis import (  # noqa: E402
    TIME_BASIS,
    T_ORIGIN_DATE,
    T_ORIGIN_BLOCK,
    T_LABEL,
    T_PER_YEAR,
    T_MIN,
)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_time_basis.py::test_app_ctx_re_exports_time_basis_constants -v
```

Expected: PASS.

- [ ] **Step 5: Run the full new test file to confirm no regressions**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_time_basis.py -v
```

Expected: 11 PASS.

- [ ] **Step 6: Commit**

```bash
git add btc_web/_app_ctx.py btc_web/test_time_basis.py
git commit -m "feat(time-basis): re-export constants from _app_ctx

Existing callers that already pull singletons from _app_ctx
(_HAS_REDIS, _MODEL_FP, etc.) can now reach time_basis without
changing import shape.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Extend `cache.py` fingerprint to include `TIME_BASIS`

**Files:**
- Modify: `btc_web/cache.py` lines 35, 106, 118, 133
- Create: `btc_web/test_time_basis_integration.py`

**Goal:** Cache keys (L1 LRU prefix, L0 pinned fingerprint, Citadel L2 keys) include the active axis so that calendar and block caches will never collide once Phase 2 produces a block pkl. Calendar-mode keys gain a `calendar:` prefix; this is a **breaking change for in-flight Redis caches** (FLUSHDB on next deploy is already part of the project's deploy ritual).

- [ ] **Step 1: Read the existing fingerprint code**

```bash
sed -n '30,45p;100,140p' btc_web/cache.py
```

The four spots:
- Line 35: `return f"fig:{_MODEL_FP}:{prefix}:{h}"`
- Line 106: `data = _REDIS.get(f"citadel:{_MODEL_FP}:{cache_key}")`
- Line 118: `_REDIS.set(f"citadel:{_MODEL_FP}:{cache_key}", ...)`
- Line 133: `_L0_FINGERPRINT = hashlib.md5(f"{_MODEL_FP}:{_DEFAULTS_HASH}".encode()).hexdigest()[:8]`

- [ ] **Step 2: Write failing integration test**

Create `btc_web/test_time_basis_integration.py`:

```python
"""Phase 1 integration tests — time_basis plumbing across modules."""
from __future__ import annotations
import hashlib

import pytest


def test_cache_l1_prefix_includes_time_basis():
    """Cache key prefix carries the axis so calendar/block won't collide.

    cache.py uses neighbor-import (`import _app_ctx`, no btc_web. prefix)
    because gunicorn runs with btc_web/ on sys.path. Tests must match.
    """
    import sys
    sys.path.insert(0, "btc_web")
    import cache  # noqa: E402
    from time_basis import TIME_BASIS  # noqa: E402
    key = cache._cache_key("bub", '{"a": 1, "b": 2}')
    assert key.startswith(f"fig:{TIME_BASIS}:"), (
        f"cache key {key!r} should start with fig:{TIME_BASIS}:")


def test_cache_l0_fingerprint_includes_time_basis():
    """L0 pinned fingerprint hash input includes TIME_BASIS.

    Slice is [:12] (matches existing cache.py:134 — do NOT shorten to [:8]).
    """
    import sys
    sys.path.insert(0, "btc_web")
    import cache  # noqa: E402
    from time_basis import TIME_BASIS  # noqa: E402
    from tab_defaults import _DEFAULTS_HASH  # noqa: E402
    expected_input = f"{TIME_BASIS}:{cache._MODEL_FP}:{_DEFAULTS_HASH}"
    expected_fp = hashlib.md5(expected_input.encode()).hexdigest()[:12]
    assert cache._L0_FINGERPRINT == expected_fp


def test_calendar_block_cache_keys_differ(monkeypatch):
    """Same params but different TIME_BASIS yield different cache keys."""
    import sys
    sys.path.insert(0, "btc_web")
    import cache  # noqa: E402
    monkeypatch.setattr(cache, "TIME_BASIS", "calendar", raising=False)
    cal = cache._cache_key("bub", '{"a": 1}')
    monkeypatch.setattr(cache, "TIME_BASIS", "block", raising=False)
    blk = cache._cache_key("bub", '{"a": 1}')
    assert cal != blk
    assert ":calendar:" in cal
    assert ":block:" in blk
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_time_basis_integration.py -v
```

Expected: 3 FAILs — cache key does not yet contain `TIME_BASIS`.

- [ ] **Step 4: Modify `btc_web/cache.py`**

At the top of the file (after the `import _app_ctx` line at line 18), add (matching the neighbor's no-prefix style — gunicorn runs with `btc_web/` on `sys.path`):

```python
from time_basis import TIME_BASIS
```

At line 35, change:

```python
    return f"fig:{_MODEL_FP}:{prefix}:{h}"
```

to:

```python
    return f"fig:{TIME_BASIS}:{_MODEL_FP}:{prefix}:{h}"
```

At lines 106 and 118, change:

```python
        data = _REDIS.get(f"citadel:{_MODEL_FP}:{cache_key}")
```
```python
        _REDIS.set(f"citadel:{_MODEL_FP}:{cache_key}",
```

to:

```python
        data = _REDIS.get(f"citadel:{TIME_BASIS}:{_MODEL_FP}:{cache_key}")
```
```python
        _REDIS.set(f"citadel:{TIME_BASIS}:{_MODEL_FP}:{cache_key}",
```

At line 133, change:

```python
_L0_FINGERPRINT = hashlib.md5(
    f"{_MODEL_FP}:{_DEFAULTS_HASH}".encode()
).hexdigest()[:12]
```

to:

```python
_L0_FINGERPRINT = hashlib.md5(
    f"{TIME_BASIS}:{_MODEL_FP}:{_DEFAULTS_HASH}".encode()
).hexdigest()[:12]
```

**Slice MUST stay `[:12]`** — that's the existing format and downstream consumers expect 12 chars.

- [ ] **Step 5: Run tests to verify they pass**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_time_basis_integration.py -v
```

Expected: 3 PASS.

- [ ] **Step 6: Verify the existing cache-alignment test suite still passes**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_cache_key_alignment.py -v
```

Expected: PASS. Calendar mode is the only mode in CI; the `calendar:` prefix change is additive in the keyspace.

- [ ] **Step 7: Commit**

```bash
git add btc_web/cache.py btc_web/test_time_basis_integration.py
git commit -m "feat(time-basis): include TIME_BASIS in cache fingerprint

L1 prefix, L0 fingerprint hash, and Citadel L2 keys now carry the
active axis so calendar/block caches will never collide. Calendar-mode
keys grow a ':calendar:' prefix — Redis FLUSHDB on next deploy.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Reserve `TIME_BASIS` slot in snapshot fingerprint

**Files:**
- Modify: `btc_web/snapshot_defaults.py` line 385 (`_compute_snapshot_defaults_fingerprint`)
- Modify: `btc_web/snapshot_defaults_registry.json` (regenerated via existing tool)
- Modify: `btc_web/test_time_basis_integration.py` (append tests)

**Goal:** The 8-char fingerprint embedded in `q4:` share-links extends to include `TIME_BASIS`. In calendar mode (default), the value `"calendar"` flows into the hash input; the fingerprint **changes** from its current value, so existing share-links built before this commit fall through to the historical-defaults registry path (already-handled fallback). Phase 3 enforces strict cross-axis decode rejection; Phase 1 only reserves the slot.

- [ ] **Step 1: Read the existing fingerprint helper**

```bash
sed -n '385,400p' btc_web/snapshot_defaults.py
```

The actual function (lines 385–394) uses **streaming sha256 over `_SNAPSHOT_CONTROLS`**, not md5 over a `repr()` blob:

```python
def _compute_snapshot_defaults_fingerprint() -> str:
    """8-char SHA256 over SNAPSHOT_DEFAULTS values, ordered by
    _SNAPSHOT_CONTROLS. Stable under benign dict-literal reorderings."""
    from snapshot import _SNAPSHOT_CONTROLS
    h = hashlib.sha256()
    for cid, prop in _SNAPSHOT_CONTROLS:
        val = SNAPSHOT_DEFAULTS.get(f"{cid}:{prop}")
        h.update(json.dumps(val, sort_keys=True).encode())
        h.update(b"\x00")
    return h.hexdigest()[:8]
```

The surgical change: feed `TIME_BASIS` into the hash *before* the loop. Keep algorithm (sha256), keep slice (`[:8]`), keep streaming pattern.

- [ ] **Step 2: Append failing test**

Append to `btc_web/test_time_basis_integration.py`:

```python
def test_snapshot_fingerprint_changes_when_time_basis_changes(monkeypatch):
    """Reserving the TIME_BASIS slot in the snapshot fingerprint hash."""
    from btc_web import snapshot_defaults as sd
    from btc_web import time_basis as tb
    monkeypatch.setattr(tb, "TIME_BASIS", "calendar")
    monkeypatch.setattr(sd, "TIME_BASIS", "calendar", raising=False)
    cal_fp = sd._compute_snapshot_defaults_fingerprint()
    monkeypatch.setattr(tb, "TIME_BASIS", "block")
    monkeypatch.setattr(sd, "TIME_BASIS", "block", raising=False)
    blk_fp = sd._compute_snapshot_defaults_fingerprint()
    assert cal_fp != blk_fp
    assert len(cal_fp) == 8
    assert len(blk_fp) == 8


def test_snapshot_fingerprint_calendar_value_is_stable():
    """Calendar mode fingerprint is deterministic given the current registry."""
    from btc_web import snapshot_defaults as sd
    fp1 = sd._compute_snapshot_defaults_fingerprint()
    fp2 = sd._compute_snapshot_defaults_fingerprint()
    assert fp1 == fp2
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_time_basis_integration.py::test_snapshot_fingerprint_changes_when_time_basis_changes -v
```

Expected: FAIL — fingerprints are equal because the hash input does not include `TIME_BASIS`.

- [ ] **Step 4: Modify `_compute_snapshot_defaults_fingerprint`**

In `btc_web/snapshot_defaults.py`, near the top imports, add:

```python
from time_basis import TIME_BASIS
```

In `_compute_snapshot_defaults_fingerprint()`, add the two `h.update(...)` calls **before** the existing `for cid, prop in _SNAPSHOT_CONTROLS:` loop. Algorithm and slice are preserved:

```python
def _compute_snapshot_defaults_fingerprint() -> str:
    """8-char SHA256 over SNAPSHOT_DEFAULTS values, ordered by
    _SNAPSHOT_CONTROLS. Stable under benign dict-literal reorderings.

    Phase 1: TIME_BASIS hashed in first so calendar/block links never
    collide. Phase 3 enforces cross-axis decode rejection (spec §3.4).
    """
    from snapshot import _SNAPSHOT_CONTROLS
    h = hashlib.sha256()
    h.update(TIME_BASIS.encode())
    h.update(b"\x00")
    for cid, prop in _SNAPSHOT_CONTROLS:
        val = SNAPSHOT_DEFAULTS.get(f"{cid}:{prop}")
        h.update(json.dumps(val, sort_keys=True).encode())
        h.update(b"\x00")
    return h.hexdigest()[:8]
```

That's the only behavioral change. Do **not** rewrite the function as md5 over a repr() blob — that would silently change the fingerprint algorithm site-wide and invalidate existing registry entries.

- [ ] **Step 5: Run tests to verify they pass**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_time_basis_integration.py -v
```

Expected: 5 PASS (3 from Task 4 + 2 new).

- [ ] **Step 6: Update the snapshot defaults registry**

The fingerprint of the live `SNAPSHOT_DEFAULTS` has changed. Per CLAUDE.md, run:

```bash
btc_venv/bin/python3 tools/update_defaults_registry.py
```

This pins the new fp into `btc_web/snapshot_defaults_registry.json` so existing `q4:` links built before this commit have a defined fallback path (the registry's pre-existing eviction handling kicks in).

- [ ] **Step 7: Run the snapshot test suite to verify no regressions**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_snapshot.py -v
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add btc_web/snapshot_defaults.py btc_web/snapshot_defaults_registry.json \
        btc_web/test_time_basis_integration.py
git commit -m "feat(time-basis): reserve TIME_BASIS slot in snapshot fp

The 8-char fingerprint embedded in q4: share-links now hashes
TIME_BASIS alongside SNAPSHOT_DEFAULTS. Phase 1 reserves the slot
defensively (calendar mode keeps decoding); Phase 3 enforces strict
cross-axis decode rejection.

Registry updated via tools/update_defaults_registry.py per
CLAUDE.md workflow.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Widen `model_data.pkl` schema + JSON sidecar

**Files:**
- Modify: `tools/model_toolkit/export.py` — add fields to `build_bm_pkl_dict()` + `build_ef_pkl_dict()`, add sidecar emission to `write_pkl()`
- No change: `tools/build_bm_model.py` (calls `build_bm_pkl_dict` + `write_pkl` already; new fields flow through transparently)
- Create: `model_data_meta.json` (build artifact, emitted by `write_pkl()`)
- Create: `model_data_ef_meta.json` (build artifact, emitted by `write_pkl()` when EF rebuilds)
- Modify: `btc_web/test_time_basis_integration.py` (append test)

**Goal:** New pkls carry `time_basis`, `t_label`, `t_per_year`, `t_origin` metadata so consumers can sanity-check what axis the fits live on. The same four fields are written to a JSON sidecar next to the pkl — `model_data.pkl` ↔ `model_data_meta.json`, `model_data_ef.pkl` ↔ `model_data_ef_meta.json` — so tests, ops, and any non-Python consumer can introspect without unpickling. Calendar-mode pkl rebuilds remain back-compat: `price_years` stays as the legacy alias for `t`.

**Note:** `tools/model_toolkit/export.py` actual structure is three top-level functions:
- `build_bm_pkl_dict(price_data, support, composite, comp_by_n, qr, sigma, genesis_date=...)` — returns 19-key dict
- `build_ef_pkl_dict(support, composite, comp_by_n, sigma, fitted, price_years, price_prices, quantiles, genesis_date=...)` — returns 16-key dict
- `write_pkl(data, path, protocol=4)` — writes pickle + prints summary; `path` is a `str` (uses `os.path.dirname`, not `pathlib`)

The plan adds 4 fields to BOTH builder dicts and adds JSON-sidecar emission to `write_pkl` so any path passing through gets a sidecar written next to it.

- [ ] **Step 1: Read the current export schema**

```bash
sed -n '1,80p' tools/model_toolkit/export.py
```

Locate the three functions: `build_bm_pkl_dict` (lines 8–35), `build_ef_pkl_dict` (lines 37–63), `write_pkl` (lines 66–71).

- [ ] **Step 2: Append failing test**

Append to `btc_web/test_time_basis_integration.py`:

```python
def test_model_data_meta_json_exists_and_has_required_fields():
    """Phase 1 emits a JSON metadata sidecar alongside model_data.pkl."""
    import json
    from pathlib import Path
    repo_root = Path(__file__).resolve().parent.parent
    meta_path = repo_root / "model_data_meta.json"
    assert meta_path.exists(), (
        f"{meta_path} should be written by tools/build_bm_model.py")
    with open(meta_path) as f:
        meta = json.load(f)
    assert meta["time_basis"] in ("calendar", "block")
    assert meta["t_label"] in ("years", "blocks")
    assert meta["t_per_year"] in (1.0, 52596.0)
    assert meta["t_origin"] is not None
    # Default deployment is calendar
    if meta["time_basis"] == "calendar":
        assert meta["t_label"] == "years"
        assert meta["t_per_year"] == 1.0
        assert meta["t_origin"] == "2009-07-25"


def test_model_data_meta_matches_active_time_basis():
    """The on-disk pkl metadata reflects the current TIME_BASIS config."""
    import json
    from pathlib import Path
    from time_basis import TIME_BASIS
    repo_root = Path(__file__).resolve().parent.parent
    meta_path = repo_root / "model_data_meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    assert meta["time_basis"] == TIME_BASIS, (
        f"sidecar reports {meta['time_basis']!r} but TIME_BASIS is "
        f"{TIME_BASIS!r} — rebuild model_data.pkl after editing "
        f"quantoshi.toml")
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_time_basis_integration.py::test_model_data_meta_json_exists_and_has_required_fields -v
```

Expected: FAIL — `model_data_meta.json` does not exist.

- [ ] **Step 4: Rewrite `tools/model_toolkit/export.py`**

Replace the file's contents to add the schema fields to **both** builder functions and emit a sidecar from `write_pkl`. The full new file:

```python
# tools/model_toolkit/export.py
"""Assemble and write model pkl files."""
from __future__ import annotations
import json
import os
import pickle
import sys
from pathlib import Path

# Match prod sys.path layout — btc_web/ on path, no btc_web. prefix.
_BTC_WEB = str(Path(__file__).resolve().parent.parent.parent / "btc_web")
if _BTC_WEB not in sys.path:
    sys.path.insert(0, _BTC_WEB)

from time_basis import (  # noqa: E402
    TIME_BASIS, T_LABEL, T_PER_YEAR, T_ORIGIN_DATE, T_ORIGIN_BLOCK,
)


def _axis_meta() -> dict:
    """The four Phase 1 schema fields, derived from time_basis at call time."""
    return {
        "time_basis": TIME_BASIS,
        "t_label": T_LABEL,
        "t_per_year": T_PER_YEAR,
        "t_origin": (
            T_ORIGIN_DATE.isoformat() if TIME_BASIS == "calendar"
            else T_ORIGIN_BLOCK
        ),
    }


def build_bm_pkl_dict(price_data, support, composite, comp_by_n, qr, sigma,
                       genesis_date="2009-07-25"):
    """17 keys + 4 axis-metadata keys. String keys for qr_fits.
    float() wrappers on scalars.

    E6: price_dates/years/prices from df_full (date>=fit_min_date).
    """
    d = {
        "qr_fits": {str(k): dict(v) for k, v in qr.fits.items()},
        "QR_QUANTILES": list(qr.fits.keys()),
        "ols_intercept": float(qr.ols_intercept),
        "ols_slope": float(qr.ols_slope),
        "GENESIS_DATE": genesis_date,
        "years_plot_bm": list(composite.t_grid),
        "support_plot_bm": list(composite.support_grid),
        "bm_support_intercept": composite.support_intercept,
        "bm_support_slope": composite.support_slope,
        "bm_comp_by_n": comp_by_n,
        "bm_r2_comp": float(composite.r2),
        "bm_n_future_max": len(comp_by_n) - 1,
        "bm_sigma0_up": float(sigma.sigma0_up),
        "bm_sigma0_down": float(sigma.sigma0_down),
        "bm_alpha_up": float(sigma.alpha_up),
        "bm_alpha_down": float(sigma.alpha_down),
        "price_dates": price_data.df_full["date"].dt.strftime("%Y-%m-%d").tolist(),
        "price_years": price_data.df_full["years"].tolist(),
        "price_prices": price_data.df_full["price"].tolist(),
    }
    d.update(_axis_meta())
    return d


def build_ef_pkl_dict(support, composite, comp_by_n, sigma, fitted,
                       price_years, price_prices, quantiles,
                       genesis_date="2009-07-25"):
    """EF pkl. Different key names from BM. Plus 4 axis-metadata keys."""
    fitted_params = []
    for b in sorted(fitted, key=lambda b: b["t_rise"]):
        fitted_params.append({k: b.get(k, 0.0) for k in
            ["t_rise", "r", "t_plateau", "t_decay", "d", "K",
             "plat_pow", "dur_rise", "dur_plateau"]})
    d = {
        "ef_support_slope": support.slope,
        "ef_support_intercept": support.intercept,
        "genesis": genesis_date,
        "years_plot": composite.t_grid.tolist(),
        "support_plot": composite.support_grid.tolist(),
        "comp_by_n": comp_by_n,
        "bm_r2": float(composite.r2),
        "n_future_max": len(comp_by_n) - 1,
        "sigma0_up": float(sigma.sigma0_up),
        "sigma0_down": float(sigma.sigma0_down),
        "alpha_up": float(sigma.alpha_up),
        "alpha_down": float(sigma.alpha_down),
        "price_years": price_years,
        "price_prices": price_prices,
        "QR_QUANTILES": list(quantiles),
        "fitted_bubbles": fitted_params,
    }
    d.update(_axis_meta())
    return d


def _sidecar_path(pkl_path: str) -> str:
    """Derive sidecar filename from pkl path. Same dir, _meta.json suffix.
    model_data.pkl    -> model_data_meta.json
    model_data_ef.pkl -> model_data_ef_meta.json
    """
    base, _ = os.path.splitext(pkl_path)
    return f"{base}_meta.json"


def write_pkl(data, path, protocol=4):
    """Write pkl file + JSON sidecar with axis metadata."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=protocol)
    print(f"Wrote {path}  ({os.path.getsize(path) // 1024} KB, {len(data)} keys)")
    # Sidecar JSON for ops + tests — query active axis without unpickling.
    meta = {k: data[k] for k in ("time_basis", "t_label", "t_per_year", "t_origin")
            if k in data}
    if meta:
        sidecar = _sidecar_path(path)
        with open(sidecar, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Wrote {sidecar}  ({len(meta)} keys)")
```

The existing `"price_years"` field stays as-is in calendar mode (back-compat). `tools/build_bm_model.py` does not need changes — it already calls `build_bm_pkl_dict(...)` then `write_pkl(...)`, and the new fields flow through transparently.

- [ ] **Step 5: Rebuild `model_data.pkl`**

```bash
btc_venv/bin/python3 tools/build_bm_model.py
```

Expected output ends with `Wrote model_data.pkl` (or equivalent). Sigma fitting included per CLAUDE.md.

- [ ] **Step 6: Confirm sidecar exists**

```bash
cat model_data_meta.json
```

Expected: JSON with the four fields, all reflecting calendar mode.

- [ ] **Step 7: Run tests to verify they pass**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_time_basis_integration.py -v
```

Expected: 7 PASS (5 from prior tasks + 2 new).

- [ ] **Step 8: Verify no regressions in model load**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_models.py btc_web/test_resqr_build.py -v
```

Expected: PASS. The added schema fields are purely additive in the pkl.

- [ ] **Step 9: Commit**

```bash
git add tools/model_toolkit/export.py model_data.pkl model_data_meta.json \
        btc_web/test_time_basis_integration.py
git commit -m "feat(time-basis): add schema metadata + JSON sidecar

New top-level fields in model_data.pkl: time_basis, t_label,
t_per_year, t_origin. Same fields mirrored in model_data_meta.json
next to the pkl so tests + ops can introspect the active axis
without unpickling.

Calendar-mode back-compat: price_years alias preserved. Phase 2
will parameterize the build pipeline to also produce
model_data_block.pkl + model_data_block_meta.json.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: CTA isolation test (`engines/custom_fit.py` does not import `time_basis`)

**Files:**
- Modify: `btc_web/test_time_basis_integration.py` (append test)

**Goal:** Codify the spec's design decision (§3.2): the Custom Time Axis panel on Tab 1 has its own per-fit calendar/block toggle and must not be coupled to the site-wide `TIME_BASIS`. A static-import test catches accidental coupling at PR time.

- [ ] **Step 1: Append regression-guard test**

Append to `btc_web/test_time_basis_integration.py`:

```python
def test_custom_fit_does_not_import_time_basis():
    """CTA stays per-fit user-controlled — must not couple to site axis.

    Spec §3.2 + §3.7. Design decision: the per-fit scale dropdown on
    Tab 1's Custom Time Axis panel is independent of the site-wide
    TIME_BASIS. Coupling would mean changing the admin TOML would
    silently change CTA fits — surprising and wrong.

    Two-layer check:
      1. AST walker catches static `import` / `from … import` forms.
      2. Substring scan catches dynamic forms (importlib.import_module,
         __import__, getattr(sys.modules,…)) that the AST walker misses.
    """
    import ast
    from pathlib import Path
    repo_root = Path(__file__).resolve().parent.parent
    src = (repo_root / "btc_web" / "engines" / "custom_fit.py").read_text()

    # Layer 1: static AST scan
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            assert mod != "time_basis" and mod != "btc_web.time_basis", (
                f"engines/custom_fit.py must not import time_basis "
                f"(found 'from {mod} import …') — spec §3.2"
            )
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "time_basis" not in alias.name, (
                    f"engines/custom_fit.py must not import time_basis "
                    f"(found 'import {alias.name}')"
                )

    # Layer 2: substring scan (catches dynamic imports the AST misses).
    # Strip comments and docstrings first so the assertion message itself
    # — which mentions time_basis — doesn't trip the test.
    import io, tokenize
    code_only = []
    for tok in tokenize.tokenize(io.BytesIO(src.encode()).readline):
        if tok.type not in (tokenize.COMMENT, tokenize.STRING,
                            tokenize.ENCODING, tokenize.NL,
                            tokenize.NEWLINE):
            code_only.append(tok.string)
    code_str = " ".join(code_only)
    assert "time_basis" not in code_str, (
        "engines/custom_fit.py must not reference time_basis even "
        "dynamically (importlib, __import__, etc.)"
    )
```

- [ ] **Step 2: Run test — should pass on the current tree**

`engines/custom_fit.py` already does not import `time_basis` (verified during planning). The test exists as a regression-guard.

```bash
btc_venv/bin/python3 -m pytest btc_web/test_time_basis_integration.py::test_custom_fit_does_not_import_time_basis -v
```

Expected: PASS.

- [ ] **Step 3: Verify the test would FAIL if the import were added**

Manually inject the offending import temporarily (don't commit), re-run the test, confirm it fails, then revert:

```bash
# Temporarily prepend the offending import
btc_venv/bin/python3 -c "
import pathlib
f = pathlib.Path('btc_web/engines/custom_fit.py')
src = f.read_text()
f.write_text('from time_basis import TIME_BASIS\n' + src)
"
btc_venv/bin/python3 -m pytest btc_web/test_time_basis_integration.py::test_custom_fit_does_not_import_time_basis -v
# Expected: FAIL with the assertion message about CTA staying per-fit.

# Revert:
git checkout btc_web/engines/custom_fit.py
btc_venv/bin/python3 -m pytest btc_web/test_time_basis_integration.py::test_custom_fit_does_not_import_time_basis -v
# Expected: PASS again.
```

- [ ] **Step 4: Commit**

```bash
git add btc_web/test_time_basis_integration.py
git commit -m "test(time-basis): CTA isolation guard

engines/custom_fit.py must not import btc_web.time_basis. CTA's
calendar/block scale dropdown is per-fit user-controlled and
independent of site-wide TIME_BASIS (spec §3.2, §3.7).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Phase 1 acceptance — full test suite green + Phase 1 marker commit

**Files:** none modified — verification + tagging only

**Goal:** Confirm Phase 1 changes nothing user-visible. Full test suite (including Playwright E2E if the engineer chooses) passes. A marker commit makes the cherry-pick boundary obvious for the eventual master merge.

- [ ] **Step 1: Run the fast test suite**

```bash
btc_venv/bin/python3 -m pytest btc_web/ -v --ignore-glob='*_e2e.py' 2>&1 | tail -40
```

Expected: all tests pass (count should match master's count + the new tests added across Tasks 1–7, currently 19 new tests: 11 in `test_time_basis.py` (3 + 7 + 1) + 8 in `test_time_basis_integration.py` (3 + 2 + 2 + 1)).

- [ ] **Step 2: Smoke-start the dev server**

```bash
DEV=1 nohup bash run_web.sh > /tmp/quantoshi_dev.log 2>&1 &
sleep 5
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8050/
# Expected: 200
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8050/6
# Expected: 200
lsof -ti :8050 | xargs kill -9
```

If 200 on both, site bootstraps fine with the new schema fields and cache-key prefix.

- [ ] **Step 3: Sanity-check the boot log**

```bash
grep -E "time_basis|TIME_BASIS|fingerprint" /tmp/quantoshi_dev.log | head -10
```

Expected output includes the `time_basis: calendar (T_LABEL=years, …)` log line from `time_basis.py` and the existing `Model fingerprint:` line (with the new fp — different from before due to the calendar prefix change in `_L0_FINGERPRINT`).

- [ ] **Step 4: Run Playwright E2E (optional but recommended)**

```bash
DEV=1 nohup bash run_web.sh > /tmp/quantoshi_dev.log 2>&1 &
sleep 5
btc_venv/bin/python3 -m pytest btc_web/test_plot_appearance_e2e.py -v
btc_venv/bin/python3 -m pytest btc_web/test_scanner_e2e.py -v
lsof -ti :8050 | xargs kill -9
```

Expected: PASS. Phase 1 changes no visual or behavioral surface, so E2E should be a no-op delta.

- [ ] **Step 5: Phase 1 acceptance commit (empty marker)**

```bash
git commit --allow-empty -m "phase1(time-basis): plumbing complete — calendar default unchanged

Tasks 1–7 landed:
  - quantoshi.toml admin config
  - btc_web/time_basis.py module + helpers
  - _app_ctx re-export
  - cache.py L0/L1/L2 keys carry TIME_BASIS
  - snapshot fingerprint reserves TIME_BASIS slot
  - model_data.pkl schema widened + model_data_meta.json sidecar
  - CTA isolation guard

Default time_basis = calendar; no fits change, no charts change.
Phase 2 (parameterize build pipeline + parallel block pkl) is a
separate plan with its own decision gate.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 6: Push the branch**

```bash
git push -u origin time-basis-toggle
```

Expected: branch is on `origin`. Open a PR for review or merge fast-forward to master per the project's deploy workflow (`feedback_deploy_workflow.md` — for non-UI work, master merge is fine).

---

## Phase 1 done

After Task 8: re-read the spec's Phase 2 section (§4 Phase 2, ~25-file scope) and decide whether to proceed. Phase 2 produces `model_data_block.pkl` as a parallel artifact, computes the R² / AIC / OOS-RMSE / calendar-osc-amplitude comparison report, and lands the **Decision Point** (§4 Phase 2) — bail here if block-axis doesn't outperform.

The Phase 2 plan is its own document: `docs/superpowers/plans/<date>-time-basis-toggle-phase2.md`. Do not pre-write it; let Phase 1's results inform Phase 2's task list.
