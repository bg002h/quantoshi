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
