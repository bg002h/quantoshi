# Time-Basis Toggle — Phase 2a (Refactor + Parameterize Build) Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `btc_core/` and `tools/model_toolkit/` so the build pipeline can accept `--time-basis={calendar,block}`. Calendar-mode behavior **unchanged**; block-mode is enabled but not exercised yet (Phase 2b builds the actual block pkl).

**Architecture:** Add axis-aware `T_MIN`, `year_to_t`, `today_t` to `btc_web/time_basis.py`. Bridge via sys.path so `btc_core/` submodules can import from it. Replace 13 hardcoded `price_years >= 1.0` masks with `>= T_MIN`. Parameterize `tools/model_toolkit/data.py::load_prices` and `tools/model_toolkit/fitting.py` to thread `time_basis` through. Add `--time-basis` CLI flag to `tools/build_bm_model.py` and `tools/build_ef_model.py`. **Calendar-mode (default) builds produce numerically equivalent pkls to current.**

**Tech Stack:** Python 3.14 (dev) / 3.12 (prod). pandas, numpy. stdlib `tomllib` (already wired). Existing test suite (~1600 tests).

**Spec:** [`docs/superpowers/specs/2026-04-26-time-basis-toggle-design.md`](../specs/2026-04-26-time-basis-toggle-design.md) §4 Phase 2.
**Decisions log:** [`docs/superpowers/plans/2026-04-26-decisions-log.md`](2026-04-26-decisions-log.md) (D11 — pivot: block becomes canonical default; A/B comparison report deferred).

**Branch:** Continue on `time-basis-toggle` (currently at `0176148`).

**Phase 2 sequencing (this plan covers 2a only):**
- **2a (this plan):** Refactor — calendar build still works identically.
- **2b (next plan):** Build `model_data_block.pkl` + bound-rescale LPPL/EPPL/HybPPL `W_cal` from rad/yr to rad/block. No comparison report (per D11).
- **2c:** Runtime axis loader — `_app_ctx` loads pkl matching configured `time_basis`. Snapshot fp enforcement.
- **2d:** Heavy caches — block-mode MC + Citadel.
- **2e:** Flip `quantoshi.toml` to `time_basis = "block"`, redeploy.

---

## File Structure

**Modify:**
- `btc_core/__init__.py` — add sys.path bridge to `btc_web/` so submodules can `from time_basis import T_MIN, year_to_t`.
- `btc_web/time_basis.py` — add `year_to_t(cal_year)` and `today_t()` axis-aware helpers.
- `btc_core/_simple.py` — 8 `price_years >= 1.0` sites (lines 70, 129, 183, 223, 276, 331, 376, 489).
- `btc_core/_lppl.py` — 1 site (line 41).
- `btc_core/_basis.py` — 1 site (line 41).
- `btc_core/_hybppl_eppl.py` — 2 sites (lines 836, 1063).
- `btc_core/_helpers.py` — 1 site in `compute_model_r2` (line 291).
- `tools/model_toolkit/data.py` — `load_prices()` accepts `time_basis` param; in block mode joins with `BitcoinBlocksDaily.csv`.
- `tools/model_toolkit/fitting.py` — `find_peaks()` `t_center` calculation axis-aware; `date_rise/plat/decay/end` conversion axis-aware.
- `tools/build_bm_model.py` — add `argparse` `--time-basis` flag; thread to `load_prices`.
- `tools/build_ef_model.py` — add `argparse` `--time-basis` flag; thread to `load_prices`.

**Create:**
- `btc_web/test_time_basis_phase2a.py` — Phase 2a unit tests for new `time_basis` helpers + T_MIN sweep regression.

**Untouched (deliberate Phase 2a non-goals):**
- `btc_core/_simple.py:571` — `UserModel` uses `>= 0.5` not `>= 1.0`. UserModel is per-user click-to-draw, axis-exempt.
- `btc_core/_simple.py:486-520` — `S2FModel`. Spec §2 axis-exempt; calendar-native always.
- `btc_core/_helpers.py::yr_to_t/today_t` — kept calendar-only in Phase 2a. The new axis-aware helpers in `time_basis.py` are deliberately named `year_to_t` (different from `_helpers.yr_to_t`) to avoid an immediate naming collision. Phase 2c will consolidate: rename or have `_helpers.yr_to_t` delegate to `time_basis.year_to_t`. `_helpers.fit_qr_from_csv` likewise stays calendar-only in 2a.
- `btc_web/engines/custom_fit.py:492` — `mask = fi.t > (1.0 / 365.25)`. CTA isolation per Phase 1 Task 7 — must NOT touch.
- `btc_web/_app_ctx.py:302-340` — model registration block uses `M.price_years` extensively. **`price_years` rename is NOT in Phase 2a.** In block mode, the `price_years` attribute carries block offsets (semantic shift, no rename). Phase 2c+ rename target.
- `btc_web/figures/`, `mc_overlay`, `mc_cache`, etc. — all consumers of `M.price_years`. Untouched in 2a.

**Phase 2a invariant:** rebuilding `model_data.pkl` with `--time-basis=calendar` (default) produces a pkl whose model R² and AIC values match current within float-rounding tolerance (≤ 1e-6 relative). The full sigma/composite/QR/resqr pipeline outputs are numerically equivalent.

---

## Task 1: sys.path bridge in `btc_core/__init__.py`

**Files:**
- Modify: `btc_core/__init__.py` (insert at top of file, before existing submodule imports).
- Create: `btc_web/test_time_basis_phase2a.py`.

**Goal:** Allow `btc_core/*.py` submodules to `from time_basis import T_MIN, year_to_t` without each adding its own sys.path hack. The bridge runs once, when the package is first imported.

- [ ] **Step 1: Write failing test for the bridge**

Create `btc_web/test_time_basis_phase2a.py`:

```python
"""Phase 2a tests — refactor + parameterize build pipeline.

Tests the btc_core → time_basis bridge, T_MIN sweep, and build-pipeline
parameterization. Does NOT exercise block-mode end-to-end (Phase 2b builds
the actual block pkl).
"""
from __future__ import annotations
import sys
from pathlib import Path

import pytest


def test_btc_core_bridges_time_basis_into_sys_path():
    """Importing btc_core makes time_basis importable as a top-level module."""
    import btc_core  # noqa: F401
    # After btc_core is imported, time_basis should be importable bare.
    import time_basis as tb  # would fail without the bridge
    assert tb.TIME_BASIS in ("calendar", "block")
    assert tb.T_MIN in (1.0, 52596.0)
```

- [ ] **Step 2: Run test to verify it passes-or-fails depending on prior state**

```bash
cd /scratch/code/bitcoinprojections
btc_venv/bin/python3 -m pytest btc_web/test_time_basis_phase2a.py::test_btc_core_bridges_time_basis_into_sys_path -v
```

If pytest already has `btc_web/` on sys.path (which it does — verified in earlier Phase 1 work), this test passes trivially. Force the failing state by running it in a fresh interpreter:

```bash
btc_venv/bin/python3 -c "
import sys
sys.path[:] = [p for p in sys.path if 'btc_web' not in p]
import btc_core
import time_basis
print('OK:', time_basis.TIME_BASIS)
"
```

Expected (before Phase 2a edits): `ModuleNotFoundError: No module named 'time_basis'`.

- [ ] **Step 3: Add the bridge to `btc_core/__init__.py`**

At the **top** of `btc_core/__init__.py` (before any `from btc_core._helpers import …` lines), add:

```python
# ── Phase 2a sys.path bridge ──────────────────────────────────────────────
# btc_core needs `time_basis` (lives in btc_web/) for axis-aware constants
# (T_MIN, T_PER_YEAR, year_to_t). Adding btc_web/ to sys.path here means
# every btc_core submodule can `from time_basis import …` without its own
# path manipulation. This runs once when btc_core is first imported.
#
# TODO(phase2c): move time_basis to a more neutral location (e.g.
# btc_core/time_basis.py) or make btc_core a proper package. Either
# obviates this bridge.
import sys as _sys
from pathlib import Path as _Path
_BTC_WEB = str(_Path(__file__).resolve().parent.parent / "btc_web")
if _BTC_WEB not in _sys.path:
    _sys.path.insert(0, _BTC_WEB)
del _sys, _Path, _BTC_WEB
# ──────────────────────────────────────────────────────────────────────────
```

- [ ] **Step 4: Run the verification command from Step 2 again**

```bash
btc_venv/bin/python3 -c "
import sys
sys.path[:] = [p for p in sys.path if 'btc_web' not in p]
import btc_core
import time_basis
print('OK:', time_basis.TIME_BASIS)
"
```

Expected: `OK: calendar`.

- [ ] **Step 5: Run the pytest unit test**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_time_basis_phase2a.py::test_btc_core_bridges_time_basis_into_sys_path -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add btc_core/__init__.py btc_web/test_time_basis_phase2a.py
git commit -m "feat(phase2a): btc_core sys.path bridge to time_basis

btc_core needs T_MIN / year_to_t from btc_web/time_basis. The bridge in
btc_core/__init__.py adds btc_web/ to sys.path once per process so
submodules can 'from time_basis import …' without each doing a path hack.
TODO(phase2c) marker for cleanup once time_basis finds a permanent home.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Add `year_to_t` and `today_t` to `time_basis.py`

**Files:**
- Modify: `btc_web/time_basis.py` (append two functions to the bottom).
- Modify: `btc_web/test_time_basis_phase2a.py` (append tests).

**Goal:** Axis-aware calendar-year → t conversion for `tools/model_toolkit/fitting.py` (`find_peaks` t_center) and downstream callers in Phase 2b+. In calendar mode behaves like `(year - 2009) → years-since-origin`. In block mode returns the same value scaled by `T_PER_YEAR` (block offset for the same calendar moment).

- [ ] **Step 1: Append failing tests**

Append to `btc_web/test_time_basis_phase2a.py`:

```python
def test_time_basis_year_to_t_calendar():
    """year_to_t in calendar mode returns years since 2009-07-25."""
    import time_basis as tb
    if tb.TIME_BASIS != "calendar":
        pytest.skip("calendar-only test")
    # 2010 January 1 → 0.439 years past 2009-07-25 (160 days / 365.25).
    t = tb.year_to_t(2010)
    assert 0.4 < t < 0.5
    # 2024 January 1 → 14.439 years past 2009-07-25.
    t = tb.year_to_t(2024)
    assert 14.4 < t < 14.5
    # Fractional year: 2024.5 = July 1 2024 → 14.939
    t = tb.year_to_t(2024.5)
    assert 14.9 < t < 15.0


def test_time_basis_year_to_t_block(monkeypatch):
    """year_to_t in block mode scales the calendar-mode result by T_PER_YEAR."""
    import time_basis as tb
    monkeypatch.setattr(tb, "TIME_BASIS", "block")
    monkeypatch.setattr(tb, "T_PER_YEAR", 52596.0)
    # 2024 January 1 → ~14.439 years × 52596 ≈ 759,406 blocks since origin.
    t = tb.year_to_t(2024)
    assert 759_000 < t < 760_000


def test_time_basis_today_t_positive_and_in_range():
    """today_t returns a sensible value in either basis."""
    import time_basis as tb
    t = tb.today_t()
    if tb.TIME_BASIS == "calendar":
        # Today is at least 16 years past 2009-07-25, less than 30.
        assert 16.0 < t < 30.0
    else:
        # Block mode: 16 years × 52596 ≈ 841,536; less than 30 × 52596.
        assert 800_000 < t < 1_600_000
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_time_basis_phase2a.py -v -k "year_to_t or today_t"
```

Expected: FAILs with `AttributeError: module 'time_basis' has no attribute 'year_to_t'` (and `today_t`).

- [ ] **Step 3: Add helpers to `btc_web/time_basis.py`**

Append after the existing `t_to_calendar` function:

```python
def year_to_t(cal_year: float) -> float:
    """Convert a (possibly fractional) calendar year to t in the active basis.

    Calendar mode: t = years since 2009-07-25 (`(date - origin).days / 365.25`).
    Block mode:    t = projected block offset, using T_PER_YEAR.

    `cal_year` may be fractional (e.g. 2024.5 ≈ July 1, 2024). Integer part
    is treated as January 1 of that calendar year; the fractional part adds
    `frac × T_PER_YEAR` to the result (1 calendar year worth of t-units).

    Used by tools/model_toolkit/fitting.py::find_peaks to convert bubble-year
    centers (e.g. 2017, 2021) to t for peak-finding masks.
    """
    yr = int(cal_year)
    frac = float(cal_year) - yr
    base_date = _dt.date(yr, 1, 1)
    base_t = calendar_to_t(base_date)
    return base_t + frac * T_PER_YEAR


def today_t() -> float:
    """Today's date converted to t in the active basis."""
    return calendar_to_t(_dt.date.today())
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_time_basis_phase2a.py -v -k "year_to_t or today_t"
```

Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add btc_web/time_basis.py btc_web/test_time_basis_phase2a.py
git commit -m "feat(phase2a): add year_to_t and today_t helpers to time_basis

Axis-aware calendar-year-to-t conversion. Used by Task 5 (model_toolkit
fitting) and downstream Phase 2 callers.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: T_MIN sweep across `btc_core/` (13 sites)

**Files:**
- Modify: `btc_core/_simple.py` (8 sites: lines 70, 129, 183, 223, 276, 331, 376, 489 — and add import).
- Modify: `btc_core/_lppl.py` (line 41 — and add import).
- Modify: `btc_core/_basis.py` (line 41 — and add import).
- Modify: `btc_core/_hybppl_eppl.py` (lines 836, 1063 — and add import).
- Modify: `btc_core/_helpers.py` (line 291, in `compute_model_r2` — and add import).
- Modify: `btc_web/test_time_basis_phase2a.py` (append regression test).

**Goal:** Replace 13 hardcoded `price_years >= 1.0` masks with axis-aware `price_years >= T_MIN`. Calendar mode unchanged (`T_MIN = 1.0`); block mode will exclude the first `T_MIN = 52596` blocks (~1 year) — the analog of "first calendar year".

- [ ] **Step 1: Append failing regression test**

Append to `btc_web/test_time_basis_phase2a.py`:

```python
def test_t_min_sweep_calendar_mode_unchanged():
    """All 13 mask sites still exclude the same rows in calendar mode."""
    import numpy as np
    from time_basis import T_MIN
    assert T_MIN == 1.0  # this test is calendar-only
    # The mask `>= T_MIN` with T_MIN=1.0 must produce the same boolean
    # array as the old `>= 1.0` literal. Pick a synthetic price_years
    # array that straddles the threshold.
    price_years = np.array([0.5, 0.99, 1.0, 1.01, 5.0, 14.0])
    new_mask = price_years >= T_MIN
    old_mask = price_years >= 1.0
    np.testing.assert_array_equal(new_mask, old_mask)


def test_t_min_block_mode_threshold():
    """In block mode, T_MIN = T_PER_YEAR (one year's worth of blocks)."""
    import time_basis as tb
    if tb.TIME_BASIS == "block":
        assert tb.T_MIN == tb.T_PER_YEAR == 52596.0
    else:
        assert tb.T_MIN == tb.T_PER_YEAR == 1.0
```

- [ ] **Step 2: Run tests — should pass on the current tree**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_time_basis_phase2a.py -v -k "t_min"
```

Expected: 2 PASS (regression guard tests). The threshold equivalence holds for calendar mode regardless of whether the sweep landed.

- [ ] **Step 3: Sweep `btc_core/_simple.py`**

At the top of the file (after existing imports, before any class definitions), add:

```python
from time_basis import T_MIN
```

Then for each of the 9 lines (verify with `grep -n "price_years >= 1.0" btc_core/_simple.py` first), change:

```python
mask = price_years >= 1.0
```

to:

```python
mask = price_years >= T_MIN
```

Lines: 70, 129, 183, 223, 276, 331, 376, 489 (8 sites).

**Do NOT touch line 571** (`UserModel.from_points` uses `>= 0.5` — different threshold, axis-exempt).

- [ ] **Step 4: Sweep `btc_core/_lppl.py`**

At the top of the file (after existing imports), add:

```python
from time_basis import T_MIN
```

Change line 41:

```python
mask = price_years >= 1.0
```

to:

```python
mask = price_years >= T_MIN
```

- [ ] **Step 5: Sweep `btc_core/_basis.py`**

At the top of the file (after existing imports), add:

```python
from time_basis import T_MIN
```

Change line 41:

```python
mask = price_years >= 1.0
```

to:

```python
mask = price_years >= T_MIN
```

- [ ] **Step 6: Sweep `btc_core/_hybppl_eppl.py`**

At the top of the file (after existing imports), add:

```python
from time_basis import T_MIN
```

Change lines 836 and 1063:

```python
mask = price_years >= 1.0
```

to:

```python
mask = price_years >= T_MIN
```

- [ ] **Step 7: Sweep `btc_core/_helpers.py::compute_model_r2`**

At the top of the file (after existing imports), add:

```python
from time_basis import T_MIN
```

In `compute_model_r2` at line 291, change:

```python
mask = price_years >= 1.0
```

to:

```python
mask = price_years >= T_MIN
```

- [ ] **Step 8: Verify no other `>= 1.0` price_years patterns slipped through**

```bash
grep -rn "price_years >= 1\." btc_core/
```

Expected output: empty (all 13 sites converted). The `>= 0.5` outlier in `_simple.py:571` (UserModel) does not match this pattern.

- [ ] **Step 9: Run unit tests**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_time_basis_phase2a.py btc_web/test_models.py -v 2>&1 | tail -20
```

Expected: all pass. Calendar mode `T_MIN = 1.0` so the masks are equivalent; existing model tests should not see any behavior change.

- [ ] **Step 10: Commit**

```bash
git add btc_core/_simple.py btc_core/_lppl.py btc_core/_basis.py \
        btc_core/_hybppl_eppl.py btc_core/_helpers.py \
        btc_web/test_time_basis_phase2a.py
git commit -m "feat(phase2a): T_MIN sweep across btc_core (13 sites)

Replace hardcoded \`price_years >= 1.0\` with axis-aware \`>= T_MIN\` so
block-mode fits exclude the first T_PER_YEAR blocks (analog of 'first
calendar year'). Calendar mode unchanged: T_MIN = 1.0.

Files: _simple.py (8), _lppl.py (1), _basis.py (1), _hybppl_eppl.py (2),
_helpers.py compute_model_r2 (1).

UserModel (_simple.py:571) untouched — \`>= 0.5\` is per-user click-to-draw,
axis-exempt.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Parameterize `tools/model_toolkit/data.py::load_prices`

**Files:**
- Modify: `tools/model_toolkit/data.py` (the `load_prices` function).
- Modify: `btc_web/test_time_basis_phase2a.py` (append test).

**Goal:** `load_prices` accepts a `time_basis` parameter (default `"calendar"`). In `"block"` mode, joins with `BitcoinBlocksDaily.csv` and computes `df["years"]` as `(blockheight - T_ORIGIN_BLOCK)` — the **column is still named `years`** for back-compat with the rest of the toolkit, even though it carries block offsets in block mode. Document this.

- [ ] **Step 1: Append failing test**

Append to `btc_web/test_time_basis_phase2a.py`:

```python
def test_load_prices_calendar_mode_unchanged():
    """Calendar-mode load_prices produces the same df['years'] as before."""
    import sys
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root / "tools"))
    from model_toolkit.data import load_prices

    pd_calendar = load_prices(str(repo_root / "BitcoinPricesDaily.csv"))
    # First row in df_full is 2010-07-17 → ~0.978 years past 2009-07-25.
    first_years = pd_calendar.df_full["years"].iloc[0]
    assert 0.97 < first_years < 1.0
    # Last row is in the future relative to 2009; should be > 14 years.
    last_years = pd_calendar.df_full["years"].iloc[-1]
    assert last_years > 14.0


def test_load_prices_block_mode_uses_block_offsets():
    """Block-mode load_prices joins with BitcoinBlocksDaily.csv and
    computes years = blockheight - T_ORIGIN_BLOCK."""
    import sys
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root / "tools"))
    sys.path.insert(0, str(repo_root / "btc_web"))
    from model_toolkit.data import load_prices
    from time_basis import T_ORIGIN_BLOCK

    pd_block = load_prices(
        str(repo_root / "BitcoinPricesDaily.csv"),
        time_basis="block",
    )
    # First row date is 2010-07-17, which the block CSV maps to block 68779.
    # Block offset = 68779 - 20188 = 48591.
    first_offset = pd_block.df_full["years"].iloc[0]
    assert 48000 < first_offset < 49500
    # Last row offset must be much larger (block_origin is at 2009-07-25).
    last_offset = pd_block.df_full["years"].iloc[-1]
    assert last_offset > 700_000  # ~13 years past origin in blocks
```

- [ ] **Step 2: Run tests to verify the block-mode one fails**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_time_basis_phase2a.py -v -k "load_prices"
```

Expected: calendar test PASS (existing behavior); block test FAIL (`time_basis` param not yet accepted).

- [ ] **Step 3: Modify `tools/model_toolkit/data.py`**

Replace the file contents with:

```python
"""Load and prepare Bitcoin price data."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd


@dataclass
class PriceData:
    df: pd.DataFrame          # filtered by years>=1 AND date>=fit_min_date (for support/bubble fitting)
    df_full: pd.DataFrame     # filtered by date>=fit_min_date only (for QR fitting + price export)
    years: np.ndarray         # from df (years>=1 subset) — IN BLOCK MODE: block offsets, not years
    log_years: np.ndarray
    prices: np.ndarray
    log_prices: np.ndarray
    dates: list               # date strings from df


def load_prices(csv_path, genesis_date="2009-07-25", fit_min_date="2010-07-17",
                time_basis="calendar"):
    """Load CSV, compute t-axis (years or block offset), log columns, filter.

    Parameters
    ----------
    csv_path : str or Path
        Path to BitcoinPricesDaily.csv (columns: date, price).
    genesis_date : str
        Calendar time origin for the power-law fit (default 2009-07-25).
        Used in calendar mode; ignored in block mode (block origin is pinned
        in quantoshi.toml).
    fit_min_date : str
        Earliest date to include in the fitting dataset (default 2010-07-17).
    time_basis : {"calendar", "block"}
        Axis on which `df["years"]` is computed. Default "calendar" preserves
        existing behavior. In block mode, joins the price CSV with
        BitcoinBlocksDaily.csv and computes `df["years"] = blockheight -
        T_ORIGIN_BLOCK`. **The column is still named `years`** even in block
        mode (back-compat with downstream toolkit code that reads
        `df["years"]`).

    Returns
    -------
    PriceData
        Dataclass with filtered df, full df, and convenience arrays.

    Notes
    -----
    Two datasets match the notebook's dual filtering:
    - Cell 0 (support/bubble): years >= T_MIN AND date >= fit_min_date
    - Cell 1 (QR/OLS/export): date >= fit_min_date only

    In block mode, T_MIN equals T_PER_YEAR (≈52596 blocks ≈ 1 calendar year),
    keeping the "exclude first year" semantics.
    """
    df = pd.read_csv(csv_path)
    date_col = next((c for c in df.columns if "date" in c.lower()), df.columns[0])
    price_col = next((c for c in df.columns if "price" in c.lower()), df.columns[1])
    df = df.rename(columns={date_col: "date", price_col: "price"})
    df["date"] = pd.to_datetime(df["date"])
    df["price"] = df["price"].astype(float)
    df = df.sort_values("date").reset_index(drop=True)

    fit_min = pd.Timestamp(fit_min_date)

    if time_basis == "calendar":
        genesis = pd.Timestamp(genesis_date)
        df["years"] = (df["date"] - genesis).dt.days / 365.25
        t_min_value = 1.0
    elif time_basis == "block":
        # Need btc_web on sys.path for time_basis import (callers ensure this).
        import sys
        _btc_web = str(Path(__file__).resolve().parent.parent.parent / "btc_web")
        if _btc_web not in sys.path:
            sys.path.insert(0, _btc_web)
        from time_basis import T_ORIGIN_BLOCK, T_PER_YEAR

        block_csv = Path(__file__).resolve().parent.parent.parent / "BitcoinBlocksDaily.csv"
        if not block_csv.exists():
            raise RuntimeError(
                f"block-mode load_prices needs {block_csv} — "
                "rerun tools/build_block_map.py if missing."
            )
        blocks_df = pd.read_csv(block_csv, parse_dates=["date"])
        # Inner-join on date so misalignments (price row without block, or
        # vice versa) get dropped. The CSVs are normally tight day-by-day.
        df = df.merge(blocks_df, on="date", how="inner")
        df["years"] = (df["blockheight"] - T_ORIGIN_BLOCK).astype(float)
        t_min_value = float(T_PER_YEAR)
    else:
        raise ValueError(f"unknown time_basis {time_basis!r}")

    df["log_years"] = np.log10(df["years"].clip(lower=1e-10))
    df["log_price"] = np.log10(df["price"].clip(lower=1e-10))

    # df_full: date >= fit_min_date (for QR/OLS fitting + price export)
    df_full = df[df["date"] >= fit_min].reset_index(drop=True)

    # df: years >= T_MIN AND date >= fit_min_date (for support/bubble fitting)
    mask = (df["years"] >= t_min_value) & (df["date"] >= fit_min)
    df = df[mask].reset_index(drop=True)

    return PriceData(
        df=df, df_full=df_full,
        years=df["years"].values, log_years=df["log_years"].values,
        prices=df["price"].values, log_prices=df["log_price"].values,
        dates=df["date"].dt.strftime("%Y-%m-%d").tolist(),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_time_basis_phase2a.py -v -k "load_prices"
```

Expected: 2 PASS (both calendar and block).

- [ ] **Step 5: Run downstream toolkit tests**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_models.py btc_web/test_resqr_build.py -v 2>&1 | tail -10
```

Expected: PASS. Calendar-mode `load_prices` behavior is unchanged.

- [ ] **Step 6: Commit**

```bash
git add tools/model_toolkit/data.py btc_web/test_time_basis_phase2a.py
git commit -m "feat(phase2a): parameterize model_toolkit load_prices(time_basis=)

Calendar mode (default) unchanged. Block mode joins with
BitcoinBlocksDaily.csv and computes df['years'] as blockheight -
T_ORIGIN_BLOCK. Column kept named 'years' for back-compat with
downstream toolkit code; documented in docstring.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Parameterize `tools/model_toolkit/fitting.py`

**Files:**
- Modify: `tools/model_toolkit/fitting.py` (`find_peaks` + the `date_rise/plat/decay/end` block in `fit_one_bubble`).
- Modify: `btc_web/test_time_basis_phase2a.py` (append test).

**Goal:** `find_peaks` `t_center` calculation uses `time_basis.year_to_t` (axis-aware) instead of the hardcoded `(date - GENESIS).days / 365.25`. The `date_rise/plat/decay/end` conversion in `fit_one_bubble` uses `time_basis.t_to_calendar` so block-mode bubble metadata still produces real calendar dates for downstream chart annotations. Calendar mode behavior unchanged.

- [ ] **Step 1: Append failing test**

Append to `btc_web/test_time_basis_phase2a.py`:

```python
def test_find_peaks_t_center_axis_aware():
    """find_peaks computes t_center via time_basis.year_to_t, not hardcoded
    pd.Timestamp arithmetic."""
    import sys
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root / "tools"))
    sys.path.insert(0, str(repo_root / "btc_web"))
    from model_toolkit import fitting as fmod
    import time_basis as tb
    import numpy as np

    if tb.TIME_BASIS != "calendar":
        pytest.skip("calendar-only sanity test")

    # Synthetic data: 1 fake bubble year at 2017.
    # In calendar mode, year_to_t(2017) ≈ 7.44; window is [6.69, 8.19].
    # Inject the peak inside that window so find_peaks can locate it.
    years = np.linspace(0.5, 16.0, 1000)
    log_excess = np.zeros_like(years)
    target_t = 7.5  # well inside the [6.69, 8.19] window for yr=2017
    peak_idx = np.argmin(np.abs(years - target_t))
    log_excess[peak_idx] = 1.0

    peaks = fmod.find_peaks(log_excess, years, [2017], window=0.75)
    assert len(peaks) == 1
    # Peak should be found at approximately the injected location.
    assert abs(peaks[0]["peak_t"] - target_t) < 0.1


def test_date_conversion_calendar_mode_unchanged():
    """The date_rise/plat/decay/end fields produce the same Timestamps
    as the hardcoded GENESIS + Timedelta path in calendar mode."""
    import sys
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root / "btc_web"))
    import pandas as pd
    import time_basis as tb

    if tb.TIME_BASIS != "calendar":
        pytest.skip("calendar-only test")

    # New axis-aware conversion: t -> calendar date via time_basis.t_to_calendar
    # Old: GENESIS + Timedelta(days=t * 365.25)
    GENESIS = pd.Timestamp("2009-07-25")
    for t in [1.0, 5.5, 14.123, 25.0]:
        old_ts = GENESIS + pd.Timedelta(days=t * 365.25)
        new_date = tb.t_to_calendar(t)
        # Must agree to within 1 day (rounding from day-floor).
        assert abs((pd.Timestamp(new_date) - old_ts).days) <= 1
```

- [ ] **Step 2: Run tests — should pass on the current tree (sanity)**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_time_basis_phase2a.py -v -k "find_peaks or date_conversion"
```

Expected: PASS (the tests verify equivalences that should hold both before and after the refactor).

- [ ] **Step 3: Hoist axis-aware imports to top of `tools/model_toolkit/fitting.py`**

After the existing imports (near line 1-15), add:

```python
# Axis-aware helpers for find_peaks t_center + bubble date conversion.
# Mirrors the sys.path bridge in btc_core/__init__.py so this module also
# works when imported standalone (build_bm_model.py inserts ROOT to sys.path
# before importing model_toolkit).
import sys as _sys
from pathlib import Path as _Path
_BTC_WEB = str(_Path(__file__).resolve().parent.parent.parent / "btc_web")
if _BTC_WEB not in _sys.path:
    _sys.path.insert(0, _BTC_WEB)
from time_basis import year_to_t, t_to_calendar  # noqa: E402
del _sys, _Path, _BTC_WEB
```

- [ ] **Step 4: Modify `find_peaks` `t_center` calculation**

Find the function `find_peaks`. Original body (around lines 50-80):

```python
    for yr in bubble_years:
        t_center = (pd.Timestamp(f"{yr}-01-01") - GENESIS).days / 365.25
        t_lo = t_center - window
```

Change to:

```python
    for yr in bubble_years:
        # Phase 2a: year_to_t is axis-aware (calendar: years; block: blocks).
        t_center = year_to_t(float(yr))
        t_lo = t_center - window
```

- [ ] **Step 5: Modify the `date_rise/plat/decay/end` conversion**

Find the dict-construction block in `fit_one_bubble` around lines 195-210. Original:

```python
        "date_rise":    GENESIS + pd.Timedelta(days=tr    * 365.25),
        "date_plat":    GENESIS + pd.Timedelta(days=tplat * 365.25),
        "date_decay":   GENESIS + pd.Timedelta(days=tdec  * 365.25),
        "date_end":     GENESIS + pd.Timedelta(days=t_end * 365.25),
```

Replace with:

```python
        # Phase 2a: t → calendar date via time_basis (axis-aware).
        # In calendar mode equivalent to the old GENESIS + Timedelta(days=t*365.25).
        # In block mode, t is a block offset; t_to_calendar projects via
        # T_PER_YEAR to a calendar date for chart annotations.
        "date_rise":    pd.Timestamp(t_to_calendar(tr)),
        "date_plat":    pd.Timestamp(t_to_calendar(tplat)),
        "date_decay":   pd.Timestamp(t_to_calendar(tdec)),
        "date_end":     pd.Timestamp(t_to_calendar(t_end)),
```

- [ ] **Step 6: Run tests**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_time_basis_phase2a.py btc_web/test_models.py -v 2>&1 | tail -15
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add tools/model_toolkit/fitting.py btc_web/test_time_basis_phase2a.py
git commit -m "feat(phase2a): axis-aware t_center and date conversion in fitting

find_peaks uses time_basis.year_to_t for t_center; fit_one_bubble uses
time_basis.t_to_calendar for date_rise/plat/decay/end. Calendar mode
behavior unchanged (verified by date_conversion test).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Add `--time-basis` flag to `tools/build_bm_model.py`

**Files:**
- Modify: `tools/build_bm_model.py` (add `argparse` block, thread `time_basis` through `load_prices` call).
- Modify: `btc_web/test_time_basis_phase2a.py` (append CLI invocation test).

**Goal:** `tools/build_bm_model.py [--time-basis={calendar,block}]` controls which axis the rebuild uses. Default `calendar`. The flag is threaded through `load_prices(...)`. Block-mode rebuild is **not** exercised in this task — Task 8 (acceptance) only verifies calendar-mode rebuild works.

- [ ] **Step 1: Append failing test**

Append to `btc_web/test_time_basis_phase2a.py`:

```python
def test_build_bm_model_accepts_time_basis_flag():
    """tools/build_bm_model.py --help shows --time-basis flag."""
    import subprocess
    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [str(repo_root / "btc_venv/bin/python3"),
         str(repo_root / "tools/build_bm_model.py"), "--help"],
        capture_output=True, text=True, cwd=repo_root, timeout=30,
    )
    # exit 0 with help text on stdout
    assert result.returncode == 0, f"stderr: {result.stderr!r}"
    assert "--time-basis" in result.stdout
    assert "calendar" in result.stdout
    assert "block" in result.stdout
```

- [ ] **Step 2: Run test to verify it fails**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_time_basis_phase2a.py -v -k "build_bm_model"
```

Expected: FAIL — `build_bm_model.py` currently has no argparse, so `--help` likely errors or returns no `--time-basis` substring.

- [ ] **Step 3: Modify `tools/build_bm_model.py`**

Replace the start of `def main():`. Original:

```python
def main():
    print("Loading prices...")
    prices = load_prices("BitcoinPricesDaily.csv")
```

Change to:

```python
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Build model_data.pkl via model_toolkit.")
    parser.add_argument(
        "--time-basis", choices=["calendar", "block"], default="calendar",
        help="Time axis for fits. 'calendar' (default) computes years since "
             "2009-07-25; 'block' computes blocks since T_ORIGIN_BLOCK "
             "(read from quantoshi.toml). Block mode requires "
             "BitcoinBlocksDaily.csv. Phase 2a: parameterized but block "
             "rebuild is Phase 2b's job.",
    )
    args = parser.parse_args()
    print(f"time_basis: {args.time_basis}")

    print("Loading prices...")
    prices = load_prices("BitcoinPricesDaily.csv", time_basis=args.time_basis)
```

The rest of `main()` is unchanged — `prices.df["years"]` flows transparently through the toolkit.

- [ ] **Step 4: Run test to verify it passes**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_time_basis_phase2a.py -v -k "build_bm_model"
```

Expected: PASS.

- [ ] **Step 5: Verify `--help` output manually**

```bash
btc_venv/bin/python3 tools/build_bm_model.py --help
```

Expected: argparse help text including `--time-basis` and the description.

- [ ] **Step 6: Commit**

```bash
git add tools/build_bm_model.py btc_web/test_time_basis_phase2a.py
git commit -m "feat(phase2a): tools/build_bm_model.py --time-basis flag

argparse adds --time-basis={calendar,block} (default calendar) and
threads it through load_prices(...). Block-mode rebuild itself is
Phase 2b — this task only adds the CLI surface.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Add `--time-basis` flag to `tools/build_ef_model.py`

**Files:**
- Modify: `tools/build_ef_model.py` (mirror Task 6 pattern).
- Modify: `btc_web/test_time_basis_phase2a.py` (append CLI test).

**Goal:** Same as Task 6 but for the EF model build. Spec §2 declares EF axis-exempt — calendar-native always — so the flag exists for symmetry but block-mode produces a calendar-equivalent EF artifact. Document this.

- [ ] **Step 1: Append failing test**

```python
def test_build_ef_model_accepts_time_basis_flag():
    """tools/build_ef_model.py --help shows --time-basis flag."""
    import subprocess
    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [str(repo_root / "btc_venv/bin/python3"),
         str(repo_root / "tools/build_ef_model.py"), "--help"],
        capture_output=True, text=True, cwd=repo_root, timeout=30,
    )
    assert result.returncode == 0, f"stderr: {result.stderr!r}"
    assert "--time-basis" in result.stdout
    assert "axis-exempt" in result.stdout.lower() or "calendar-native" in result.stdout.lower()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_time_basis_phase2a.py -v -k "build_ef_model"
```

Expected: FAIL.

- [ ] **Step 3: Modify `tools/build_ef_model.py`**

Replace the `def main():` start. Original:

```python
def main():
    print("=" * 70)
    print("Building Empirical Floor model")
    print("=" * 70)

    prices = load_prices("BitcoinPricesDaily.csv")
```

Change to:

```python
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Build model_data_ef.pkl (Empirical Floor model). "
                    "Note: EF is axis-exempt (calendar-native) per "
                    "time-basis spec section 2; the --time-basis flag exists for "
                    "symmetry with build_bm_model.py but block mode is "
                    "expected to remain calendar-equivalent.")
    parser.add_argument(
        "--time-basis", choices=["calendar", "block"], default="calendar",
        help="Time axis (default calendar). EF is axis-exempt; block "
             "mode produces the same artifact as calendar mode.",
    )
    args = parser.parse_args()
    print(f"time_basis: {args.time_basis} "
          f"(EF is axis-exempt — calendar-native by design)")

    print("=" * 70)
    print("Building Empirical Floor model")
    print("=" * 70)

    prices = load_prices("BitcoinPricesDaily.csv", time_basis=args.time_basis)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_time_basis_phase2a.py -v -k "build_ef_model"
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/build_ef_model.py btc_web/test_time_basis_phase2a.py
git commit -m "feat(phase2a): tools/build_ef_model.py --time-basis flag

Mirrors build_bm_model.py CLI for symmetry. EF is axis-exempt per
spec §2 — calendar-native always; the flag is documented as such.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Phase 2a acceptance — calendar-mode rebuild byte-similar; suite green; smoke

**Files:** none modified — verification + marker commit.

**Goal:** Confirm Phase 2a refactor changed nothing user-visible. Calendar-mode rebuild produces a `model_data.pkl` whose model R² and AIC values match the pre-2a pkl within float-rounding tolerance. Full suite green. Site smoke 200/200.

- [ ] **Step 1: Snapshot the pre-rebuild pkl R² fingerprint**

Use `btc_core.load_model_data()` (project loader, no direct pickle handling in test code):

```bash
cd /scratch/code/bitcoinprojections
btc_venv/bin/python3 -c "
import btc_core as bc
m = bc.load_model_data()
# ModelData exposes the BM scalars as attributes.
keys = ['ols_slope', 'ols_intercept',
        'support_slope', 'support_intercept',
        'bm_r2', 'bm_alpha_up', 'bm_alpha_down',
        'bm_sigma0_up', 'bm_sigma0_down']
for k in keys:
    print(f'{k}: {getattr(m, k)}')" | tee /tmp/phase2a_pre.txt
```

(If `ModelData` does not expose all of these attributes, query via `m._raw[k]` or `m._dict[k]` per the project's accessor convention — check `btc_core/_model_data.py`.)

Save the output to `/tmp/phase2a_pre.txt`.

- [ ] **Step 2: Rebuild with the (now Phase 2a-aware) build pipeline, calendar mode**

```bash
btc_venv/bin/python3 tools/build_bm_model.py --time-basis=calendar 2>&1 | tail -10
```

Expected: completes in 3–5 minutes, ends with `Wrote model_data.pkl` and `Wrote model_data_meta.json`.

- [ ] **Step 3: Capture the post-rebuild fingerprint**

```bash
btc_venv/bin/python3 -c "
import btc_core as bc
m = bc.load_model_data()
keys = ['ols_slope', 'ols_intercept',
        'support_slope', 'support_intercept',
        'bm_r2', 'bm_alpha_up', 'bm_alpha_down',
        'bm_sigma0_up', 'bm_sigma0_down']
for k in keys:
    print(f'{k}: {getattr(m, k)}')" | tee /tmp/phase2a_post.txt
```

- [ ] **Step 4: Diff pre and post, verify within 1e-6 relative tolerance**

```bash
diff /tmp/phase2a_pre.txt /tmp/phase2a_post.txt
```

Expected: empty diff (same line-by-line) OR all numeric values match to ~6 decimal places.

Also verify the meta sidecar is correct:

```bash
cat model_data_meta.json
```

Expected: `{"time_basis": "calendar", "t_label": "years", "t_per_year": 1.0, "t_origin": "2009-07-25"}`.

If the diff shows numeric differences > 1e-6 relative, **STOP and investigate.** Phase 2a's invariant is calendar-mode unchanged.

- [ ] **Step 5: Run the full test suite**

```bash
btc_venv/bin/python3 -m pytest btc_web/ --ignore-glob='*_e2e.py' 2>&1 | tail -10
```

Expected: counts match Phase 1's count + the new Phase 2a tests (~10 new tests across `test_time_basis_phase2a.py`). The 2 pre-existing master failures (BTCPay tier + colors hex-literal) remain; nothing else fails.

- [ ] **Step 6: Smoke-start the dev server**

```bash
DEV=1 nohup bash run_web.sh > /tmp/quantoshi_dev.log 2>&1 &
sleep 8
curl -s -o /dev/null -w "GET / -> %{http_code}\n" http://127.0.0.1:8050/
curl -s -o /dev/null -w "GET /6 -> %{http_code}\n" http://127.0.0.1:8050/6
lsof -ti :8050 | xargs -r kill -9
```

Expected: 200/200.

- [ ] **Step 7: Verify clean working tree**

```bash
git status --short
```

Expected: clean OR only `model_data.pkl` and `model_data_meta.json` modified (rebuild artifacts).

If those two files are modified, commit them:

```bash
git add model_data.pkl model_data_meta.json
git commit -m "build(phase2a): rebuild model_data.pkl via parameterized pipeline

Numerically equivalent to the pre-2a pkl (R²/AIC/sigma values match
within float-rounding tolerance per Step 4 diff). Captures the new
build pipeline output as the canonical artifact for the branch.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 8: Phase 2a marker commit**

```bash
git commit --allow-empty -m "phase2a(time-basis): refactor + parameterize build complete

Tasks 1–7 landed:
  - btc_core/__init__.py sys.path bridge to btc_web/time_basis
  - time_basis.year_to_t and today_t helpers
  - T_MIN sweep (13 sites across _simple, _lppl, _basis, _hybppl_eppl, _helpers)
  - load_prices(time_basis=…) — block mode joins BitcoinBlocksDaily.csv
  - find_peaks t_center + date_rise/plat/decay/end axis-aware
  - tools/build_bm_model.py --time-basis flag
  - tools/build_ef_model.py --time-basis flag

Calendar-mode behavior unchanged; rebuild produces numerically
equivalent pkl. Block-mode is parameterized but not exercised yet —
Phase 2b builds the actual model_data_block.pkl.

Per decisions log D11: A/B comparison report deferred; Phase 2 path
is 2a → 2b → 2c → 2d → 2e (block becomes prod default).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 9: Verify final state**

```bash
git log --oneline -12
git status --short
```

Expected:
- Marker commit at HEAD
- Working tree clean
- ~10 new commits ahead of Phase 1 marker (`c322549`)

- [ ] **Step 10: DO NOT push or deploy without explicit user consent**

Phase 2a is complete and ready for Phase 2b. The user will decide when to push and whether to deploy 2a alone (which is safe — calendar behavior unchanged) or wait for 2b/2c.

---

## Phase 2a done

After Task 8: Phase 2b is the next plan. It will:
- Add bound rescaling for HybPPL/LinPPL/EPPL `W_cal` from rad/yr to rad/block in 6 `tools/fit_*.py` scripts.
- Run `tools/build_bm_model.py --time-basis=block` to produce `model_data_block.pkl`.
- Run `tools/refit_all_ppl.py --time-basis=block` (or per-script `tools/fit_*.py --time-basis=block --update`) to produce block-mode fits for the 22+ parametric models.
- Sanity-check fits don't NaN; spot-check R²/AIC for at least 3 representative models.

The Phase 2b plan is its own document: `docs/superpowers/plans/<date>-time-basis-toggle-phase2b.md`. Do not pre-write it; let Phase 2a's results inform Phase 2b's task list.
