# Unified Citadel MC — Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create preset definitions for Quick Scenarios, a cache generation script that runs N-sim Citadel simulations for 1,620 parameter combos, and shared-memory storage/loading so the web app can serve pre-computed percentile bands instantly.

**Architecture:** 5 tasks, bottom-up: (1) Preset definitions module, (2) Cache key + storage format, (3) Cache generation script, (4) Shared memory loader for band data, (5) Integration tests. The cache generation script runs offline on a multi-core machine, producing `.npz` files. The web app loads these into RAM at startup (via `/dev/shm` pickle for fast restarts), same pattern as the existing `mc_cache.py`.

**Tech Stack:** Python 3.14, dataclasses, numpy, multiprocessing

**Spec:** `docs/superpowers/specs/2026-03-31-unified-citadel-mc-design.md`

**Test command:** `PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py -x -q --tb=short`

**Full suite:** `PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/ --tb=short 2>&1 | tail -5`

---

## File Structure

### New Files
| File | Responsibility |
|------|---------------|
| `btc_web/citadel_presets.py` | All preset definitions: wealth levels, macro regimes, rule sets, allocation percentages, cache dimensions. Pure data — no logic beyond `build_config()`. |
| `btc_web/citadel_band_cache.py` | Cache key helpers, npz storage format, shared-memory loader, lookup functions. Follows `mc_cache.py` patterns exactly. |
| `tools/generate_citadel_bands.py` | Offline cache generation script. Iterates all 1,620 combos, runs `simulate()` with 800 paths per combo, stores band aggregation as `.npz`. Parallelized via `multiprocessing.Pool`. |
| `btc_web/load_citadel_band_cache.py` | Standalone loader script for `/dev/shm` preload (systemd oneshot). |

### Modified Files
| File | Change |
|------|--------|
| `btc_web/test_web.py` | All new tests |

---

### Task 1: Preset definitions module (`citadel_presets.py`)

**Files:**
- Create: `btc_web/citadel_presets.py`
- Test: `btc_web/test_web.py`

This module is pure data — preset values that map to `SimConfig` fields. The `build_config()` function constructs a `SimConfig` from preset selections. All constants are easily editable without touching logic.

- [ ] **Step 1: Write failing tests**

```python
class TestCitadelPresets:
    def test_wealth_levels_exist(self):
        from citadel_presets import WEALTH_LEVELS
        assert set(WEALTH_LEVELS.keys()) == {"starter", "full", "bitcoin"}

    def test_wealth_level_has_required_keys(self):
        from citadel_presets import WEALTH_LEVELS
        required = {"label", "dollar_assets", "btc", "monthly_spend",
                    "spend_growth", "inflation", "allocation"}
        for key, wl in WEALTH_LEVELS.items():
            assert required.issubset(wl.keys()), f"{key} missing {required - wl.keys()}"

    def test_allocation_sums_to_100(self):
        from citadel_presets import WEALTH_LEVELS
        for key, wl in WEALTH_LEVELS.items():
            total = sum(wl["allocation"].values())
            assert abs(total - 100) < 0.01, f"{key} allocation sums to {total}"

    def test_macro_regimes_exist(self):
        from citadel_presets import MACRO_REGIMES
        assert set(MACRO_REGIMES.keys()) == {"bear", "neutral", "bull"}
        assert MACRO_REGIMES["bear"]["bin"] == 0
        assert MACRO_REGIMES["neutral"]["bin"] == 2
        assert MACRO_REGIMES["bull"]["bin"] == 4

    def test_rule_sets_exist(self):
        from citadel_presets import RULE_SETS
        assert set(RULE_SETS.keys()) == {"no_rebal", "cautious", "aggressive"}

    def test_cache_dimensions(self):
        from citadel_presets import (BTC_MODELS, BTC_ENTRY_QS, START_YEARS,
                                     SIMS_PER_SCENARIO, WEALTH_LEVELS,
                                     MACRO_REGIMES, RULE_SETS, TAX_STATUSES)
        assert BTC_MODELS == ["bub", "qr", "pl", "lppl", "ef"]
        assert BTC_ENTRY_QS == [1, 10, 50]
        assert START_YEARS == [2028, 2035]
        assert SIMS_PER_SCENARIO == 800
        total = (len(BTC_MODELS) * len(BTC_ENTRY_QS) * len(MACRO_REGIMES) *
                 len(WEALTH_LEVELS) * len(RULE_SETS) * len(START_YEARS) *
                 len(TAX_STATUSES))
        assert total == 1620

    def test_build_config_returns_simconfig(self):
        from citadel_presets import build_config
        from engines.citadel_types import SimConfig
        cfg = build_config(
            wealth="starter", regime="neutral", rules="no_rebal",
            start_year=2035, tax_status="single",
        )
        assert isinstance(cfg, SimConfig)

    def test_build_config_starter_values(self):
        from citadel_presets import build_config
        cfg = build_config(
            wealth="starter", regime="neutral", rules="no_rebal",
            start_year=2035, tax_status="single",
        )
        assert cfg.start_stack == 0.5
        assert cfg.monthly_spend == 5000
        assert cfg.cash_initial == 50_000  # 10% of $500k
        assert cfg.start_yr == 2035
        assert cfg.end_yr == 2075
        assert cfg.freq == "Monthly"
        assert cfg.inflation == 4.0

    def test_build_config_regime_sets_initial_regimes(self):
        from citadel_presets import build_config
        cfg = build_config(
            wealth="starter", regime="bull", rules="no_rebal",
            start_year=2035, tax_status="single",
        )
        assert cfg.initial_equity_regime == 4
        assert cfg.initial_bond_regime == 4
        assert cfg.initial_res_short_regime == 4
        assert cfg.initial_res_med_regime == 4
        assert cfg.initial_res_long_regime == 4

    def test_build_config_tax_status_mfj(self):
        from citadel_presets import build_config
        cfg = build_config(
            wealth="starter", regime="neutral", rules="no_rebal",
            start_year=2035, tax_status="mfj",
        )
        assert cfg.tax_enabled is True
        assert cfg.filing_status == "mfj"

    def test_build_config_tax_status_single(self):
        from citadel_presets import build_config
        cfg = build_config(
            wealth="starter", regime="neutral", rules="no_rebal",
            start_year=2035, tax_status="single",
        )
        assert cfg.tax_enabled is True
        assert cfg.filing_status == "single"

    def test_build_config_loads_asset_matrices(self):
        from citadel_presets import build_config
        cfg = build_config(
            wealth="starter", regime="neutral", rules="no_rebal",
            start_year=2035, tax_status="single",
        )
        assert cfg.asset_matrices is not None
        assert "equity" in cfg.asset_matrices
        assert "bond" in cfg.asset_matrices
        assert "tres_short" in cfg.asset_matrices
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestCitadelPresets -x -q --tb=short`
Expected: FAIL with "No module named 'citadel_presets'"

- [ ] **Step 3: Create `btc_web/citadel_presets.py`**

```python
"""Citadel Planner — Quick Scenario preset definitions.

All preset values in one place. Edit values here, not in logic modules.
"""
from __future__ import annotations

from engines.citadel_types import SimConfig
from data.asset_matrices import load_asset_matrices

_CACHED_MATRICES: dict | None = None

def _get_asset_matrices() -> dict:
    """Load asset matrices once, cache for reuse."""
    global _CACHED_MATRICES
    if _CACHED_MATRICES is None:
        _CACHED_MATRICES = load_asset_matrices(n_bins=5)
    return _CACHED_MATRICES

__all__ = [
    "WEALTH_LEVELS", "MACRO_REGIMES", "RULE_SETS",
    "BTC_MODELS", "BTC_ENTRY_QS", "START_YEARS", "TAX_STATUSES",
    "SIMS_PER_SCENARIO", "END_YEAR", "FREQ",
    "build_config",
]

# ── Cache dimensions ─────────────────────────────────────────────────────────

BTC_MODELS = ["bub", "qr", "pl", "lppl", "ef"]
BTC_ENTRY_QS = [1, 10, 50]       # percentile (0-100 scale)
START_YEARS = [2028, 2035]
TAX_STATUSES = ["single", "mfj"]
SIMS_PER_SCENARIO = 800
END_YEAR = 2075
FREQ = "Monthly"

# ── Wealth levels ────────────────────────────────────────────────────────────

# allocation: percentage of dollar_assets assigned to each bucket
_DEFAULT_ALLOCATION = {
    "cash": 10, "res_short": 10, "res_med": 10, "res_long": 10,
    "inv_eq": 40, "inv_bd": 20,
}

WEALTH_LEVELS = {
    "starter": {
        "label": "Starter Citadel",
        "dollar_assets": 500_000, "btc": 0.5,
        "monthly_spend": 5_000, "spend_growth": 1.0, "inflation": 4.0,
        "allocation": dict(_DEFAULT_ALLOCATION),
    },
    "full": {
        "label": "Full Citadel",
        "dollar_assets": 2_500_000, "btc": 2.5,
        "monthly_spend": 25_000, "spend_growth": 2.0, "inflation": 4.0,
        "allocation": dict(_DEFAULT_ALLOCATION),
    },
    "bitcoin": {
        "label": "Bitcoin Citadel",
        "dollar_assets": 2_500_000, "btc": 12.5,
        "monthly_spend": 50_000, "spend_growth": 4.0, "inflation": 4.0,
        "allocation": dict(_DEFAULT_ALLOCATION),
    },
}

# ── Macro regimes ────────────────────────────────────────────────────────────

MACRO_REGIMES = {
    "bear":    {"label": "Bear",    "bin": 0},
    "neutral": {"label": "Neutral", "bin": 2},
    "bull":    {"label": "Bull",    "bin": 4},
}

# ── Rule sets ────────────────────────────────────────────────────────────────

RULE_SETS = {
    "no_rebal": {
        "label": "No Rebal",
        "cash_floor": 0.0,
        "high_q_trigger": 0.99,   # effectively disabled
        "low_q_trigger": 0.01,    # effectively disabled
    },
    "cautious": {
        "label": "Cautious",
        "cash_floor": 50_000.0,
        "high_q_trigger": 0.90,
        "low_q_trigger": 0.10,
    },
    "aggressive": {
        "label": "Aggressive",
        "cash_floor": 100_000.0,
        "high_q_trigger": 0.75,
        "low_q_trigger": 0.25,
    },
}


# ── Config builder ───────────────────────────────────────────────────────────

def build_config(
    wealth: str,
    regime: str,
    rules: str,
    start_year: int,
    tax_status: str,
) -> SimConfig:
    """Build a SimConfig from preset selections.

    Args:
        wealth: key into WEALTH_LEVELS
        regime: key into MACRO_REGIMES
        rules: key into RULE_SETS
        start_year: calendar year (e.g. 2035)
        tax_status: "single" or "mfj"
    """
    wl = WEALTH_LEVELS[wealth]
    mr = MACRO_REGIMES[regime]
    rs = RULE_SETS[rules]
    alloc = wl["allocation"]
    da = wl["dollar_assets"]

    cfg = SimConfig(
        start_stack=wl["btc"],
        cash_initial=da * alloc["cash"] / 100,
        cash_rate=4.0,
        reserve_bins=[
            {"label": "Short (T-Bills)", "initial": da * alloc["res_short"] / 100,
             "rate": 5.0, "volatility": 2.0},
            {"label": "Medium (T-Notes)", "initial": da * alloc["res_med"] / 100,
             "rate": 4.5, "volatility": 8.0},
            {"label": "Long (T-Bonds)", "initial": da * alloc["res_long"] / 100,
             "rate": 4.0, "volatility": 15.0},
        ],
        invest_bins=[
            {"label": "Equities", "initial": da * alloc["inv_eq"] / 100,
             "return_rate": 10.0, "volatility": 16.0},
            {"label": "Bonds", "initial": da * alloc["inv_bd"] / 100,
             "return_rate": 5.0, "volatility": 7.0},
        ],
        monthly_spend=wl["monthly_spend"],
        spend_growth=wl["spend_growth"],
        inflation=wl["inflation"],
        start_yr=start_year,
        end_yr=END_YEAR,
        freq=FREQ,
        asset_return_model="markov",
        # Regime presets: all 5 dollar-asset regime fields set to same bin
        initial_equity_regime=mr["bin"],
        initial_bond_regime=mr["bin"],
        initial_res_short_regime=mr["bin"],
        initial_res_med_regime=mr["bin"],
        initial_res_long_regime=mr["bin"],
        # Rule set
        cash_floor=rs["cash_floor"],
        high_q_trigger=rs["high_q_trigger"],
        low_q_trigger=rs["low_q_trigger"],
        # Tax
        tax_enabled=True,
        filing_status=tax_status,
        state_code="TX",  # no state tax for cached presets
        # Markov transition matrices for dollar assets
        asset_matrices=_get_asset_matrices(),
    )
    return cfg
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestCitadelPresets -x -q --tb=short`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/ --tb=short 2>&1 | tail -5`
Expected: All existing tests still pass

- [ ] **Step 6: Commit**

```bash
git add btc_web/citadel_presets.py btc_web/test_web.py
git commit -m "feat(citadel): add citadel_presets.py with wealth/regime/rule preset definitions"
```

---

### Task 2: Band cache key + storage format (`citadel_band_cache.py`)

**Files:**
- Create: `btc_web/citadel_band_cache.py`
- Test: `btc_web/test_web.py`

This module defines the cache key format, npz storage layout, and lookup functions. Follows `mc_cache.py` patterns: npz files on disk, pickle in `/dev/shm` for fast restart, fingerprint-based invalidation.

- [ ] **Step 1: Write failing tests**

```python
class TestCitadelBandCache:
    def test_band_cache_key_format(self):
        from citadel_band_cache import band_cache_key
        key = band_cache_key("bub", 10, "neutral", "starter",
                             "no_rebal", 2035, "single")
        assert key == "bub_q10_neutral_starter_no_rebal_2035_single"

    def test_band_cache_key_all_combos_unique(self):
        from citadel_band_cache import band_cache_key
        from citadel_presets import (BTC_MODELS, BTC_ENTRY_QS, MACRO_REGIMES,
                                     WEALTH_LEVELS, RULE_SETS, START_YEARS,
                                     TAX_STATUSES)
        keys = set()
        for model in BTC_MODELS:
            for eq in BTC_ENTRY_QS:
                for regime in MACRO_REGIMES:
                    for wealth in WEALTH_LEVELS:
                        for rules in RULE_SETS:
                            for yr in START_YEARS:
                                for tax in TAX_STATUSES:
                                    k = band_cache_key(model, eq, regime,
                                                       wealth, rules, yr, tax)
                                    keys.add(k)
        assert len(keys) == 1620

    def test_pack_unpack_bands_roundtrip(self):
        import numpy as np
        from citadel_band_cache import pack_bands, unpack_bands
        from engines.citadel_bands import BAND_PERCENTILES, BAND_SERIES
        n_periods = 480
        bands = {}
        for pct in BAND_PERCENTILES:
            bands[pct] = {}
            for series in BAND_SERIES:
                bands[pct][series] = np.random.rand(n_periods).astype(np.float32)
        packed = pack_bands(bands)
        assert isinstance(packed, np.ndarray)
        assert packed.dtype == np.float32
        unpacked = unpack_bands(packed)
        for pct in BAND_PERCENTILES:
            for series in BAND_SERIES:
                np.testing.assert_array_almost_equal(
                    unpacked[pct][series], bands[pct][series], decimal=5)

    def test_store_and_lookup(self, tmp_path):
        import numpy as np
        from citadel_band_cache import store_entry, lookup_entry
        from engines.citadel_bands import BAND_PERCENTILES, BAND_SERIES
        n_periods = 24
        bands = {}
        for pct in BAND_PERCENTILES:
            bands[pct] = {s: np.ones(n_periods, dtype=np.float32) * pct
                          for s in BAND_SERIES}
        store_entry("bub", 10, "neutral", "starter", "no_rebal",
                    2035, "single", bands, cache_dir=tmp_path)
        result = lookup_entry("bub", 10, "neutral", "starter", "no_rebal",
                              2035, "single", cache_dir=tmp_path)
        assert result is not None
        for pct in BAND_PERCENTILES:
            np.testing.assert_array_almost_equal(
                result[pct]["total"], np.ones(n_periods) * pct, decimal=5)

    def test_lookup_missing_returns_none(self, tmp_path):
        from citadel_band_cache import lookup_entry
        result = lookup_entry("bub", 10, "neutral", "starter", "no_rebal",
                              2035, "single", cache_dir=tmp_path)
        assert result is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestCitadelBandCache -x -q --tb=short`
Expected: FAIL with "No module named 'citadel_band_cache'"

- [ ] **Step 3: Create `btc_web/citadel_band_cache.py`**

```python
"""Citadel band cache: storage, loading, and lookup for pre-computed percentile bands.

Cache structure on disk:
    citadel_band_cache/
        bands_{model}_{start_yr}.npz   -- all bands for one (model, start_yr) combo

Each npz entry: key = band_cache_key string, value = packed float32 array.
Packed layout: (7 percentiles x 11 series x n_periods) flattened to 1D.

Fast restart via /dev/shm (same pattern as mc_cache.py):
    After first load from npz, the entire cache dict is pickled to
    /dev/shm/quantoshi_citadel_bands.pkl. Subsequent restarts load from
    there (~10x faster). Uses pickle intentionally (trusted local data,
    same as existing mc_cache.py pattern).
"""
from __future__ import annotations

import pickle
import time
import numpy as np
from pathlib import Path

from engines.citadel_bands import BAND_PERCENTILES, BAND_SERIES

__all__ = [
    "band_cache_key", "pack_bands", "unpack_bands",
    "store_entry", "lookup_entry",
    "load_band_caches", "load_startup_band_cache",
    "BAND_CACHE_DIR", "SHM_BAND_CACHE_PATH",
]

BAND_CACHE_DIR = Path(__file__).parent / "citadel_band_cache"
SHM_BAND_CACHE_PATH = Path("/dev/shm/quantoshi_citadel_bands.pkl")

_N_PCTS = len(BAND_PERCENTILES)   # 7
_N_SERIES = len(BAND_SERIES)      # 11

# In-memory cache: {band_cache_key_str: packed_ndarray}
_BAND_CACHE: dict[str, np.ndarray] = {}
_FULL_LOADED = False


def band_cache_key(model: str, entry_q: int, regime: str, wealth: str,
                   rules: str, start_yr: int, tax_status: str) -> str:
    """Deterministic string key for a band cache entry."""
    return f"{model}_q{entry_q}_{regime}_{wealth}_{rules}_{start_yr}_{tax_status}"


def pack_bands(bands: dict[int, dict[str, np.ndarray]]) -> np.ndarray:
    """Pack a bands dict into a flat float32 array for storage.

    Input: {pct: {series_name: ndarray(n_periods,)}}
    Output: float32 array of shape (7 * 11 * n_periods,)
    """
    first_pct = BAND_PERCENTILES[0]
    first_series = BAND_SERIES[0]
    n_periods = len(bands[first_pct][first_series])

    packed = np.zeros(_N_PCTS * _N_SERIES * n_periods, dtype=np.float32)
    for pi, pct in enumerate(BAND_PERCENTILES):
        for si, series in enumerate(BAND_SERIES):
            offset = (pi * _N_SERIES + si) * n_periods
            packed[offset:offset + n_periods] = bands[pct][series]
    return packed


def unpack_bands(packed: np.ndarray) -> dict[int, dict[str, np.ndarray]]:
    """Unpack a flat float32 array back into a bands dict.

    Input: float32 array of shape (7 * 11 * n_periods,)
    Output: {pct: {series_name: ndarray(n_periods,)}}
    """
    n_periods = len(packed) // (_N_PCTS * _N_SERIES)
    bands: dict[int, dict[str, np.ndarray]] = {}
    for pi, pct in enumerate(BAND_PERCENTILES):
        bands[pct] = {}
        for si, series in enumerate(BAND_SERIES):
            offset = (pi * _N_SERIES + si) * n_periods
            bands[pct][series] = packed[offset:offset + n_periods].copy()
    return bands


def store_entry(model: str, entry_q: int, regime: str, wealth: str,
                rules: str, start_yr: int, tax_status: str,
                bands: dict[int, dict[str, np.ndarray]],
                cache_dir: Path | None = None) -> None:
    """Store a single band entry to npz on disk."""
    cdir = cache_dir or BAND_CACHE_DIR
    cdir.mkdir(parents=True, exist_ok=True)
    key = band_cache_key(model, entry_q, regime, wealth, rules,
                         start_yr, tax_status)
    packed = pack_bands(bands)

    npz_file = cdir / f"bands_{model}_{start_yr}.npz"
    existing = {}
    if npz_file.exists():
        with np.load(npz_file) as npz:
            existing = dict(npz)
    existing[key] = packed
    np.savez(npz_file, **existing)


def lookup_entry(model: str, entry_q: int, regime: str, wealth: str,
                 rules: str, start_yr: int, tax_status: str,
                 cache_dir: Path | None = None) -> dict | None:
    """Look up a pre-computed band entry. Returns unpacked bands or None."""
    key = band_cache_key(model, entry_q, regime, wealth, rules,
                         start_yr, tax_status)

    # Check in-memory cache first
    if key in _BAND_CACHE:
        return unpack_bands(_BAND_CACHE[key])

    # Try disk
    cdir = cache_dir or BAND_CACHE_DIR
    npz_file = cdir / f"bands_{model}_{start_yr}.npz"
    if not npz_file.exists():
        return None
    with np.load(npz_file) as npz:
        if key in npz:
            packed = npz[key]
            _BAND_CACHE[key] = packed
            return unpack_bands(packed)
    return None


# ── Shared memory persistence ────────────────────────────────────────────────

def _npz_fingerprint(cache_dir: Path | None = None) -> tuple[float, int]:
    """Return (max_mtime, total_size) of source npz files."""
    cdir = cache_dir or BAND_CACHE_DIR
    if not cdir.exists():
        return (0.0, 0)
    max_mt, total_sz = 0.0, 0
    for f in cdir.glob("bands_*.npz"):
        st = f.stat()
        max_mt = max(max_mt, st.st_mtime)
        total_sz += st.st_size
    return (max_mt, total_sz)


def _try_load_shm() -> bool:
    """Try loading from /dev/shm pickle. Returns True on success."""
    if not SHM_BAND_CACHE_PATH.exists():
        return False
    try:
        with open(SHM_BAND_CACHE_PATH, "rb") as f:
            saved = pickle.load(f)
        fp_saved = saved.pop("_fingerprint", None)
        fp_now = _npz_fingerprint()
        if fp_saved != fp_now:
            print("[CITADEL-BANDS] /dev/shm fingerprint mismatch, reloading from npz")
            return False
        _BAND_CACHE.update(saved)
        return True
    except Exception as exc:
        print(f"[CITADEL-BANDS] /dev/shm load failed: {exc}")
        return False


def _save_shm() -> None:
    """Persist cache to /dev/shm for fast restart."""
    try:
        blob = dict(_BAND_CACHE)
        blob["_fingerprint"] = _npz_fingerprint()
        with open(SHM_BAND_CACHE_PATH, "wb") as f:
            pickle.dump(blob, f, protocol=pickle.HIGHEST_PROTOCOL)
        sz_mb = SHM_BAND_CACHE_PATH.stat().st_size / 1e6
        print(f"[CITADEL-BANDS] Saved /dev/shm pickle ({sz_mb:.0f} MB)")
    except Exception as exc:
        print(f"[CITADEL-BANDS] /dev/shm save failed: {exc}")


def load_band_caches(cache_dir: Path | None = None) -> None:
    """Load all band cache data into RAM.

    Tries /dev/shm pickle first, falls back to npz parsing.
    After npz load, saves pickle to /dev/shm for next restart.
    """
    global _FULL_LOADED
    t0 = time.perf_counter()

    if cache_dir is None and _try_load_shm():
        elapsed = time.perf_counter() - t0
        print(f"[CITADEL-BANDS] Loaded from /dev/shm in {elapsed:.2f}s")
        _FULL_LOADED = True
        return

    cdir = cache_dir or BAND_CACHE_DIR
    if not cdir.exists():
        _FULL_LOADED = True
        return

    for npz_file in sorted(cdir.glob("bands_*.npz")):
        with np.load(npz_file) as npz:
            for key in npz.files:
                _BAND_CACHE[key] = npz[key]

    elapsed = time.perf_counter() - t0
    n = len(_BAND_CACHE)
    print(f"[CITADEL-BANDS] Loaded {n} entries from npz in {elapsed:.2f}s")
    _FULL_LOADED = True

    if cache_dir is None:
        _save_shm()


def load_startup_band_cache() -> None:
    """Load default entries for fast startup (lazy-loads rest on miss)."""
    load_band_caches()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestCitadelBandCache -x -q --tb=short`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/ --tb=short 2>&1 | tail -5`

- [ ] **Step 6: Commit**

```bash
git add btc_web/citadel_band_cache.py btc_web/test_web.py
git commit -m "feat(citadel): add citadel_band_cache.py with pack/unpack/store/lookup/shm"
```

---

### Task 3: Cache generation script

**Files:**
- Create: `tools/generate_citadel_bands.py`
- Test: `btc_web/test_web.py`

Offline script that iterates all 1,620 preset combos, runs `simulate()` with 800 price paths per combo, computes bands, and stores results via `citadel_band_cache.store_entry()`. Uses `multiprocessing.Pool` for parallelism. Requires the Cython `markov` module for BTC price path generation.

- [ ] **Step 1: Write tests**

These tests validate the end-to-end flow using already-created modules (they should pass immediately).

```python
class TestCitadelBandGeneration:
    def test_generate_single_entry(self, tmp_path):
        """Smoke test: generate one combo with 5 sims (fast)."""
        import numpy as np
        from citadel_presets import build_config
        from engines.citadel_sim import simulate
        from engines.citadel_bands import compute_bands, BAND_PERCENTILES, BAND_SERIES
        from citadel_band_cache import store_entry, lookup_entry

        cfg = build_config(
            wealth="starter", regime="neutral", rules="no_rebal",
            start_year=2035, tax_status="single",
        )
        cfg.end_yr = 2036  # 1 year = 12 periods (fast)
        n_sims = 5
        n_periods = 12
        rng = np.random.default_rng(42)
        base = np.linspace(80000, 120000, n_periods)
        paths = np.array([base * (1 + rng.normal(0, 0.05, n_periods))
                          for _ in range(n_sims)])
        result = simulate(cfg, model=None, price_paths=paths)
        bands = compute_bands(result)
        store_entry("bub", 10, "neutral", "starter", "no_rebal",
                    2035, "single", bands, cache_dir=tmp_path)
        loaded = lookup_entry("bub", 10, "neutral", "starter", "no_rebal",
                              2035, "single", cache_dir=tmp_path)
        assert loaded is not None
        assert set(loaded.keys()) == set(BAND_PERCENTILES)
        assert set(loaded[50].keys()) == set(BAND_SERIES)
        assert len(loaded[50]["total"]) == n_periods

    def test_generate_preserves_band_ordering(self, tmp_path):
        """P5 <= P50 <= P95 in generated bands."""
        import numpy as np
        from citadel_presets import build_config
        from engines.citadel_sim import simulate
        from engines.citadel_bands import compute_bands
        from citadel_band_cache import store_entry, lookup_entry

        cfg = build_config(
            wealth="full", regime="bear", rules="cautious",
            start_year=2028, tax_status="mfj",
        )
        cfg.end_yr = 2029
        n_sims = 20
        n_periods = 12
        rng = np.random.default_rng(99)
        base = np.linspace(50000, 100000, n_periods)
        paths = np.array([base * (1 + rng.normal(0, 0.15, n_periods))
                          for _ in range(n_sims)])
        result = simulate(cfg, model=None, price_paths=paths)
        bands = compute_bands(result)
        store_entry("pl", 50, "bear", "full", "cautious",
                    2028, "mfj", bands, cache_dir=tmp_path)
        loaded = lookup_entry("pl", 50, "bear", "full", "cautious",
                              2028, "mfj", cache_dir=tmp_path)
        for t in range(n_periods):
            assert loaded[5]["total"][t] <= loaded[50]["total"][t] + 1e-6
            assert loaded[50]["total"][t] <= loaded[95]["total"][t] + 1e-6
```

- [ ] **Step 2: Run tests**

Run: `PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestCitadelBandGeneration -x -q --tb=short`
Expected: PASS

- [ ] **Step 3: Create `tools/generate_citadel_bands.py`**

```python
#!/usr/bin/env python3
"""Generate pre-computed Citadel band cache for Quick Scenarios.

Iterates all 1,620 preset combos (5 models x 3 entry Qs x 3 regimes x
3 wealth levels x 3 rule sets x 2 start years x 2 tax statuses).
Each combo: 800 simulation paths -> percentile bands -> stored as npz.

Workers return packed band arrays to the main process, which batch-writes
npz files after all workers complete (avoids concurrent npz write races).

Requires the Cython `markov` module for BTC price path generation.

Usage:
    PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 tools/generate_citadel_bands.py
    PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 tools/generate_citadel_bands.py --workers 8
    PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 tools/generate_citadel_bands.py --dry-run
"""
from __future__ import annotations

import argparse
import sys
import time
from itertools import product
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Setup import paths
_ROOT = Path(__file__).parent.parent
for _p in (str(_ROOT), str(_ROOT / "btc_web"), str(_ROOT / "archive" / "btc_app")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from citadel_presets import (
    BTC_MODELS, BTC_ENTRY_QS, MACRO_REGIMES, WEALTH_LEVELS,
    RULE_SETS, START_YEARS, TAX_STATUSES, SIMS_PER_SCENARIO,
    build_config,
)
from citadel_band_cache import band_cache_key, pack_bands, BAND_CACHE_DIR
from engines.citadel_sim import simulate
from engines.citadel_bands import compute_bands


# ── Per-worker initialization (model loading) ────────────────────────────────
# multiprocessing workers don't inherit _app_ctx state from the main process.
# Each worker must load ModelData and construct price models independently.
# Follows the pattern from generate_mc_cache.py.

_worker_M = None
_worker_models = None


def _init_worker():
    """Initialize ModelData and price models in this worker process."""
    global _worker_M, _worker_models
    # Ensure import paths are set (needed for spawn start method)
    for _p in (str(_ROOT), str(_ROOT / "btc_web"), str(_ROOT / "archive" / "btc_app")):
        if _p not in sys.path:
            sys.path.insert(0, _p)

    import _app_ctx
    from btc_core import (load_model_data, BubbleModel, PowerLawModel, LPPLModel,
                          ExponentialModel, EmpiricalFloorModel,
                          QuantileRegressionModel)

    M = load_model_data()
    _app_ctx.M = M
    _worker_M = M

    models = {}
    models["bub"]  = BubbleModel(M)
    models["qr"]   = QuantileRegressionModel(M)
    models["pl"]   = PowerLawModel(M.ols_intercept, M.ols_slope,
                                   M.price_years, M.price_prices,
                                   M.genesis, M.QR_QUANTILES)
    models["lppl"] = LPPLModel(M.price_years, M.price_prices, M.QR_QUANTILES)
    models["exp"]  = ExponentialModel(M.price_years, M.price_prices, M.QR_QUANTILES)

    ef_pkl = _ROOT / "archive" / "btc_app" / "model_data_ef.pkl"
    if ef_pkl.exists():
        models["ef"] = EmpiricalFloorModel(str(ef_pkl))

    _app_ctx.PRICE_MODELS = models
    _worker_models = models


def _generate_btc_paths(model_key, entry_q, start_yr, n_sims, n_periods):
    """Generate BTC price paths using the Markov module.

    Returns ndarray of shape (n_sims, n_periods).
    """
    from markov import build_transition_matrix, monte_carlo_prices
    from btc_core import yr_to_t

    model = _worker_models.get(model_key)
    if model is None:
        raise ValueError(f"Unknown model: {model_key}")

    m = _worker_M
    genesis = m.genesis
    t_start = yr_to_t(start_yr, genesis)

    import pandas as pd
    yr_now = pd.Timestamp.today().year
    window_end = min(start_yr, yr_now)
    ws_yr = yr_to_t(2010, genesis)
    we_yr = yr_to_t(window_end, genesis)

    trans, bin_edges, _ = build_transition_matrix(
        m.price_prices, m.price_years, model,
        n_bins=5, window_start_yr=ws_yr, window_end_yr=we_yr,
        step_days=30,
    )

    pct_bin = entry_q / 100.0  # convert 1/10/50 to 0.01/0.10/0.50
    dt = 1.0 / 12  # Monthly
    price_paths, _ = monte_carlo_prices(
        trans, bin_edges, pct_bin, n_periods, n_sims,
        model, t_start, dt,
    )
    return price_paths.astype(np.float32)


def _worker_task(args):
    """Process one combo. Returns (cache_key, packed_bands) or (cache_key, None, error)."""
    model_key, entry_q, regime, wealth, rules, start_yr, tax_status, n_sims = args
    key = band_cache_key(model_key, entry_q, regime, wealth, rules,
                         start_yr, tax_status)
    try:
        cfg = build_config(wealth, regime, rules, start_yr, tax_status)
        n_periods = int((cfg.end_yr - cfg.start_yr) * 12)  # Monthly

        paths = _generate_btc_paths(model_key, entry_q, start_yr, n_sims, n_periods)
        result = simulate(cfg, model=None, price_paths=paths)
        bands = compute_bands(result)
        packed = pack_bands(bands)

        return key, packed, ""
    except Exception as exc:
        return key, None, str(exc)


def main():
    parser = argparse.ArgumentParser(description="Generate Citadel band cache")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: cpu_count - 2)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print combos without generating")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="Override cache directory")
    args = parser.parse_args()

    import os
    n_workers = args.workers or max(1, (os.cpu_count() or 4) - 2)
    cache_dir = Path(args.cache_dir) if args.cache_dir else BAND_CACHE_DIR

    combos = list(product(
        BTC_MODELS, BTC_ENTRY_QS, MACRO_REGIMES.keys(), WEALTH_LEVELS.keys(),
        RULE_SETS.keys(), START_YEARS, TAX_STATUSES,
    ))
    n_total = len(combos)
    print(f"Total combos: {n_total}")
    print(f"Sims per combo: {SIMS_PER_SCENARIO}")
    print(f"Workers: {n_workers}")
    print(f"Cache dir: {cache_dir}")

    if args.dry_run:
        for c in combos[:10]:
            print(f"  {c}")
        print(f"  ... and {n_total - 10} more")
        return

    cache_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    done, failed = 0, 0

    # Accumulate results in main process to avoid concurrent npz write races.
    # Key: (model, start_yr) -> dict of {cache_key: packed_ndarray}
    npz_batches: dict[tuple, dict] = {}

    tasks = [(*combo, SIMS_PER_SCENARIO) for combo in combos]

    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker) as pool:
        for key, packed, err in pool.map(_worker_task, tasks):
            done += 1
            if packed is not None:
                # Extract (model, start_yr) from the key for npz file grouping
                parts = key.split("_")
                model = parts[0]
                start_yr = int(parts[-2])
                npz_batches.setdefault((model, start_yr), {})[key] = packed
                if done % 50 == 0 or done == n_total:
                    elapsed = time.perf_counter() - t0
                    rate = done / elapsed
                    eta = (n_total - done) / rate if rate > 0 else 0
                    print(f"[{done}/{n_total}] {key}  "
                          f"({rate:.1f}/s, ETA {eta/60:.0f}min)")
            else:
                failed += 1
                print(f"[FAIL] {key}: {err}")

    # Batch-write npz files (one per model x start_yr)
    for (model, start_yr), entries in npz_batches.items():
        npz_file = cache_dir / f"bands_{model}_{start_yr}.npz"
        np.savez(npz_file, **entries)
        print(f"  Saved {npz_file.name}: {len(entries)} entries, "
              f"{npz_file.stat().st_size / 1e6:.1f} MB")

    elapsed = time.perf_counter() - t0
    print(f"\nDone: {done - failed}/{n_total} OK, {failed} failed, "
          f"{elapsed:.0f}s ({elapsed/60:.1f} min)")

    total_sz = sum(f.stat().st_size for f in cache_dir.glob("bands_*.npz"))
    print(f"Total cache size: {total_sz / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Verify script parses**

Run: `PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -c "import py_compile; py_compile.compile('tools/generate_citadel_bands.py', doraise=True); print('OK')"`
Expected: OK

- [ ] **Step 5: Run full test suite**

Run: `PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/ --tb=short 2>&1 | tail -5`

- [ ] **Step 6: Commit**

```bash
git add tools/generate_citadel_bands.py btc_web/test_web.py
git commit -m "feat(citadel): add generate_citadel_bands.py for offline cache generation"
```

---

### Task 4: Shared memory loader script

**Files:**
- Create: `btc_web/load_citadel_band_cache.py`
- Test: `btc_web/test_web.py`

Standalone script (systemd oneshot) that loads band cache into `/dev/shm`. Mirrors `load_shm_cache.py`.

- [ ] **Step 1: Write tests**

```python
class TestCitadelBandCacheLoader:
    @pytest.fixture(autouse=True)
    def _clear_band_cache(self):
        """Isolate tests from shared module state."""
        from citadel_band_cache import _BAND_CACHE
        _BAND_CACHE.clear()
        yield
        _BAND_CACHE.clear()

    def test_load_band_caches_from_disk(self, tmp_path):
        import numpy as np
        from citadel_band_cache import store_entry, load_band_caches, _BAND_CACHE
        from engines.citadel_bands import BAND_PERCENTILES, BAND_SERIES
        n_periods = 12
        bands = {}
        for pct in BAND_PERCENTILES:
            bands[pct] = {s: np.ones(n_periods, dtype=np.float32) * pct
                          for s in BAND_SERIES}
        store_entry("bub", 10, "neutral", "starter", "no_rebal",
                    2035, "single", bands, cache_dir=tmp_path)
        load_band_caches(cache_dir=tmp_path)
        assert len(_BAND_CACHE) == 1
        key = "bub_q10_neutral_starter_no_rebal_2035_single"
        assert key in _BAND_CACHE

    def test_load_empty_dir(self, tmp_path):
        from citadel_band_cache import load_band_caches, _BAND_CACHE
        load_band_caches(cache_dir=tmp_path)
        assert len(_BAND_CACHE) == 0
```

- [ ] **Step 2: Run tests**

Run: `PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestCitadelBandCacheLoader -x -q --tb=short`
Expected: PASS (uses already-created `citadel_band_cache.py`)

- [ ] **Step 3: Create `btc_web/load_citadel_band_cache.py`**

```python
#!/usr/bin/env python3
"""Pre-load Citadel band cache into /dev/shm pickle for fast app startup.

Usage:  python3 load_citadel_band_cache.py [PROJECT_ROOT]

Designed to run as a oneshot systemd service at boot.
"""
import sys
import time
from pathlib import Path

project_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent

for p in [project_root, project_root / "btc_web", project_root / "archive" / "btc_app"]:
    sys.path.insert(0, str(p))

from citadel_band_cache import (
    load_band_caches, _BAND_CACHE, BAND_CACHE_DIR, SHM_BAND_CACHE_PATH,
)

if not BAND_CACHE_DIR.exists():
    print(f"[citadel-band-loader] Cache dir not found: {BAND_CACHE_DIR}")
    sys.exit(1)

if SHM_BAND_CACHE_PATH.exists():
    print(f"[citadel-band-loader] {SHM_BAND_CACHE_PATH} already exists "
          f"({SHM_BAND_CACHE_PATH.stat().st_size / 1e6:.0f} MB), validating...")

t0 = time.perf_counter()
load_band_caches()
elapsed = time.perf_counter() - t0

n_entries = len(_BAND_CACHE)
sz = SHM_BAND_CACHE_PATH.stat().st_size / 1e6 if SHM_BAND_CACHE_PATH.exists() else 0
print(f"[citadel-band-loader] Done in {elapsed:.2f}s -- "
      f"{n_entries} entries, {sz:.0f} MB in /dev/shm")
```

- [ ] **Step 4: Verify script parses**

Run: `PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -c "import py_compile; py_compile.compile('btc_web/load_citadel_band_cache.py', doraise=True); print('OK')"`

- [ ] **Step 5: Run full test suite**

Run: `PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/ --tb=short 2>&1 | tail -5`

- [ ] **Step 6: Commit**

```bash
git add btc_web/load_citadel_band_cache.py btc_web/test_web.py
git commit -m "feat(citadel): add load_citadel_band_cache.py for /dev/shm preload"
```

---

### Task 5: Integration tests

**Files:**
- Test: `btc_web/test_web.py`

- [ ] **Step 1: Write integration tests**

```python
class TestCitadelBandCacheIntegration:
    def test_full_pipeline_build_simulate_store_lookup(self, tmp_path):
        """End-to-end: build_config -> simulate -> compute_bands -> store -> lookup."""
        import numpy as np
        from citadel_presets import build_config
        from engines.citadel_sim import simulate
        from engines.citadel_bands import compute_bands, BAND_PERCENTILES, BAND_SERIES
        from citadel_band_cache import store_entry, lookup_entry

        cfg = build_config(
            wealth="bitcoin", regime="bull", rules="aggressive",
            start_year=2028, tax_status="mfj",
        )
        cfg.end_yr = 2029  # 12 periods for speed
        n_sims = 10
        n_periods = 12
        rng = np.random.default_rng(77)
        base = np.linspace(60000, 200000, n_periods)
        paths = np.array([base * (1 + rng.normal(0, 0.1, n_periods))
                          for _ in range(n_sims)])

        result = simulate(cfg, model=None, price_paths=paths)
        bands = compute_bands(result)

        store_entry("bub", 50, "bull", "bitcoin", "aggressive",
                    2028, "mfj", bands, cache_dir=tmp_path)

        loaded = lookup_entry("bub", 50, "bull", "bitcoin", "aggressive",
                              2028, "mfj", cache_dir=tmp_path)

        assert loaded is not None
        assert set(loaded.keys()) == set(BAND_PERCENTILES)
        for pct in BAND_PERCENTILES:
            assert set(loaded[pct].keys()) == set(BAND_SERIES)
            assert len(loaded[pct]["total"]) == n_periods

        # Verify band ordering
        for t in range(n_periods):
            assert loaded[5]["total"][t] <= loaded[50]["total"][t] + 1e-6
            assert loaded[50]["total"][t] <= loaded[95]["total"][t] + 1e-6

    def test_multiple_entries_same_npz(self, tmp_path):
        """Multiple entries for same (model, start_yr) share one npz."""
        import numpy as np
        from engines.citadel_bands import BAND_PERCENTILES, BAND_SERIES
        from citadel_band_cache import store_entry, lookup_entry

        n_periods = 12
        for regime in ["bear", "neutral", "bull"]:
            bands = {}
            for pct in BAND_PERCENTILES:
                bands[pct] = {s: np.full(n_periods, float(pct), dtype=np.float32)
                              for s in BAND_SERIES}
            store_entry("qr", 10, regime, "starter", "no_rebal",
                        2035, "single", bands, cache_dir=tmp_path)

        # All three in same npz, all retrievable
        for regime in ["bear", "neutral", "bull"]:
            loaded = lookup_entry("qr", 10, regime, "starter", "no_rebal",
                                  2035, "single", cache_dir=tmp_path)
            assert loaded is not None
            assert loaded[50]["total"][0] == 50.0

    def test_cache_key_uniqueness_across_all_dimensions(self):
        """All 1620 combos produce unique cache keys."""
        from itertools import product as iproduct
        from citadel_presets import (BTC_MODELS, BTC_ENTRY_QS, MACRO_REGIMES,
                                     WEALTH_LEVELS, RULE_SETS, START_YEARS,
                                     TAX_STATUSES)
        from citadel_band_cache import band_cache_key
        keys = set()
        for m, eq, reg, wl, rs, yr, ts in iproduct(
            BTC_MODELS, BTC_ENTRY_QS, MACRO_REGIMES.keys(),
            WEALTH_LEVELS.keys(), RULE_SETS.keys(), START_YEARS, TAX_STATUSES,
        ):
            keys.add(band_cache_key(m, eq, reg, wl, rs, yr, ts))
        assert len(keys) == 1620
```

- [ ] **Step 2: Run integration tests**

Run: `PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py::TestCitadelBandCacheIntegration -x -q --tb=short`

- [ ] **Step 3: Run full test suite**

Run: `PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/ --tb=short 2>&1 | tail -5`

- [ ] **Step 4: Commit**

```bash
git add btc_web/test_web.py
git commit -m "test(citadel): integration tests for Phase 2 band cache pipeline"
```

---

## Verification Checklist

After all 5 tasks, run:

```bash
# Full test suite
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/ --tb=short 2>&1 | tail -10

# Import check
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -c "
from citadel_presets import (WEALTH_LEVELS, MACRO_REGIMES, RULE_SETS,
                             BTC_MODELS, BTC_ENTRY_QS, START_YEARS,
                             TAX_STATUSES, SIMS_PER_SCENARIO, build_config)
from citadel_band_cache import (band_cache_key, pack_bands, unpack_bands,
                                store_entry, lookup_entry, load_band_caches)
from engines.citadel_bands import BAND_PERCENTILES, BAND_SERIES

print(f'Wealth levels: {list(WEALTH_LEVELS.keys())}')
print(f'Regimes: {list(MACRO_REGIMES.keys())}')
print(f'Rule sets: {list(RULE_SETS.keys())}')
n = (len(BTC_MODELS) * len(BTC_ENTRY_QS) * len(MACRO_REGIMES) *
     len(WEALTH_LEVELS) * len(RULE_SETS) * len(START_YEARS) *
     len(TAX_STATUSES))
print(f'Total cache combos: {n}')
print(f'Sims per combo: {SIMS_PER_SCENARIO}')
print(f'Band percentiles: {BAND_PERCENTILES}')
print(f'Band series: {len(BAND_SERIES)}')

cfg = build_config('starter', 'neutral', 'no_rebal', 2035, 'single')
print(f'Config start_stack={cfg.start_stack}, monthly_spend={cfg.monthly_spend}')
print('Phase 2 OK')
"

# Script syntax check
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m py_compile tools/generate_citadel_bands.py && echo "generate OK"
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m py_compile btc_web/load_citadel_band_cache.py && echo "loader OK"
```
