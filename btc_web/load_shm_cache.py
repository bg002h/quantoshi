#!/usr/bin/env python3
"""Pre-load MC cache into /dev/shm pickle for fast app startup.

Usage:  python3 load_shm_cache.py [PROJECT_ROOT]

If PROJECT_ROOT is omitted, auto-detects from script location.
Designed to run as a oneshot systemd service at boot.
"""
import sys
import time
from pathlib import Path

project_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent

# Set up import paths (same as run_web.sh PYTHONPATH)
for p in [project_root, project_root / "btc_app", project_root / "btc_web"]:
    sys.path.insert(0, str(p))

from mc_cache import load_caches, SHM_CACHE_PATH, _CACHE, CACHE_DIR

if not CACHE_DIR.exists():
    print(f"[shm-loader] Cache dir not found: {CACHE_DIR}")
    sys.exit(1)

if SHM_CACHE_PATH.exists():
    print(f"[shm-loader] {SHM_CACHE_PATH} already exists "
          f"({SHM_CACHE_PATH.stat().st_size / 1e6:.0f} MB), validating...")

t0 = time.perf_counter()
load_caches()
elapsed = time.perf_counter() - t0

n_years = len(_CACHE)
n_entries = sum(len(v) for yr in _CACHE.values() for v in yr.values())
sz = SHM_CACHE_PATH.stat().st_size / 1e6 if SHM_CACHE_PATH.exists() else 0
print(f"[shm-loader] Done in {elapsed:.2f}s — {n_years} years, "
      f"{n_entries} entries, {sz:.0f} MB in /dev/shm")
