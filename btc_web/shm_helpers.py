"""Shared /dev/shm persistence helpers for npz-backed caches.

Used by mc_cache.py and citadel_band_cache.py — both follow the same
pattern: load from npz (slow), persist to /dev/shm pickle (fast restart),
invalidate when source npz files change.
"""
from __future__ import annotations

import pickle  # noqa: S403 - trusted local data
from pathlib import Path


def npz_fingerprint(cache_dir: Path, glob_pattern: str = "*.npz") -> tuple[float, int]:
    """Return (max_mtime, total_size) of source npz files in *cache_dir*."""
    if not cache_dir.exists():
        return (0.0, 0)
    max_mt, total_sz = 0.0, 0
    for f in cache_dir.glob(glob_pattern):
        st = f.stat()
        max_mt = max(max_mt, st.st_mtime)
        total_sz += st.st_size
    return (max_mt, total_sz)


def try_load_shm(
    shm_path: Path,
    cache_dict: dict,
    cache_dir: Path,
    glob_pattern: str = "*.npz",
    *,
    label: str = "CACHE",
) -> bool:
    """Try loading *cache_dict* from a /dev/shm pickle. Returns True on success."""
    if not shm_path.exists():
        return False
    try:
        with open(shm_path, "rb") as f:
            saved = pickle.load(f)  # noqa: S301 - trusted local file
        fp_saved = saved.pop("_fingerprint", None)
        fp_now = npz_fingerprint(cache_dir, glob_pattern)
        if fp_saved != fp_now:
            print(f"[{label}] /dev/shm fingerprint mismatch, reloading from npz")
            return False
        cache_dict.update(saved)
        return True
    except Exception as exc:
        print(f"[{label}] /dev/shm load failed: {exc}")
        return False


def save_shm(
    shm_path: Path,
    cache_dict: dict,
    cache_dir: Path,
    glob_pattern: str = "*.npz",
    *,
    label: str = "CACHE",
) -> None:
    """Persist *cache_dict* to a /dev/shm pickle for fast restart."""
    try:
        blob = dict(cache_dict)
        blob["_fingerprint"] = npz_fingerprint(cache_dir, glob_pattern)
        with open(shm_path, "wb") as f:
            pickle.dump(blob, f, protocol=pickle.HIGHEST_PROTOCOL)
        sz_mb = shm_path.stat().st_size / 1e6
        print(f"[{label}] Saved /dev/shm pickle ({sz_mb:.0f} MB)")
    except Exception as exc:
        print(f"[{label}] /dev/shm save failed: {exc}")
