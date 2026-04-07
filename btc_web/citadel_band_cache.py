"""Citadel band cache: storage, loading, and lookup for pre-computed percentile bands.

Cache structure on disk:
    citadel_band_cache/
        bands_{model}_{start_yr}.npz   -- all bands for one (model, start_yr) combo

Each npz entry: key = band_cache_key string, value = packed float32 array.
Packed layout: (7 percentiles x 11 series x n_periods) flattened to 1D.

Fast restart via /dev/shm (same pattern as mc_cache.py):
    After first load from npz, the entire cache dict is serialized to
    /dev/shm/quantoshi_citadel_bands.pkl. Subsequent restarts load from
    there (~10x faster). Uses pickle intentionally (trusted local data,
    same as existing mc_cache.py pattern).
"""
from __future__ import annotations

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
    """Unpack a flat float32 array back into a bands dict."""
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

    # Check in-memory cache first (only for default production cache_dir)
    use_default_dir = cache_dir is None
    if use_default_dir and key in _BAND_CACHE:
        return unpack_bands(_BAND_CACHE[key])

    # Try disk
    cdir = cache_dir or BAND_CACHE_DIR
    npz_file = cdir / f"bands_{model}_{start_yr}.npz"
    if not npz_file.exists():
        return None
    with np.load(npz_file) as npz:
        if key in npz:
            packed = npz[key]
            if use_default_dir:
                _BAND_CACHE[key] = packed
            return unpack_bands(packed)
    return None


# ── Shared memory persistence ────────────────────────────────────────────────

from shm_helpers import npz_fingerprint, try_load_shm, save_shm

def _npz_fingerprint(cache_dir: Path | None = None) -> tuple[float, int]:
    """Return (max_mtime, total_size) of source npz files."""
    return npz_fingerprint(cache_dir or BAND_CACHE_DIR, "bands_*.npz")

def _try_load_shm() -> bool:
    return try_load_shm(
        SHM_BAND_CACHE_PATH, _BAND_CACHE, BAND_CACHE_DIR,
        "bands_*.npz", label="CITADEL-BANDS",
    )

def _save_shm() -> None:
    save_shm(
        SHM_BAND_CACHE_PATH, _BAND_CACHE, BAND_CACHE_DIR,
        "bands_*.npz", label="CITADEL-BANDS",
    )


def load_band_caches(cache_dir: Path | None = None) -> None:
    """Load all band cache data into RAM."""
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
