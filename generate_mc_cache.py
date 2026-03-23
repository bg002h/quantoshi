#!/usr/bin/env python3
"""Generate multi-model MC cache using parallel processes.

Usage:
    btc_venv/bin/python3 generate_mc_cache.py [--workers N]

Generates paths_{model}_{year}.npz and overlays_{model}_{year}.npz files
for all 6 quantized models x 3 start years = 18 jobs.

Uses multiprocessing (not threading) for true CPU parallelism — each worker
process initializes its own model objects and runs independently.

Default: 12 workers. Use --workers to adjust.
"""

import sys
import os
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# ── Setup paths (before any project imports) ─────────────────────────────────
ROOT = Path(__file__).parent
for p in (str(ROOT), str(ROOT / "btc_web"), str(ROOT / "archive" / "btc_app")):
    if p not in sys.path:
        sys.path.insert(0, p)

from mc_cache import CACHED_START_YRS, _CACHED_MODEL_KEYS

# ── Per-worker initialization ────────────────────────────────────────────────
# Each worker process calls this once to load models into its own memory space.
# Model objects can't cross process boundaries, so each process needs its own.

_worker_M = None
_worker_models = None

def _init_worker():
    """Initialize ModelData and all price models in this worker process."""
    global _worker_M, _worker_models
    import _app_ctx
    from btc_core import (load_model_data, BubbleModel, PowerLawModel, LPPLModel,
                          ExponentialModel, S2FModel, EmpiricalFloorModel,
                          QuantileRegressionModel)
    from figures.common import _build_thermal_colors

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

    ef_pkl = ROOT / "btc_app" / "model_data_ef.pkl"
    if ef_pkl.exists():
        models["ef"] = EmpiricalFloorModel(str(ef_pkl))

    # Thermal colors for bubble model
    thermal = _build_thermal_colors(M.QR_QUANTILES)
    models["bub"].colors.update(thermal)

    _app_ctx.PRICE_MODELS.update(models)
    _app_ctx.DEFAULT_MODEL = models["bub"]
    _worker_models = models


def _run_one(args):
    """Generate cache for one (model_key, start_yr) combo. Runs in worker process."""
    model_key, start_yr = args
    from mc_cache import generate_cache

    model = _worker_models[model_key]
    label = f"{model_key}/{start_yr}"
    pid = os.getpid()
    t0 = time.perf_counter()
    try:
        pf, of = generate_cache(start_yr, _worker_M, model,
                                progress_cb=lambda msg: print(f"  [pid {pid}] [{label}] {msg}",
                                                              flush=True))
        elapsed = time.perf_counter() - t0
        psz = pf.stat().st_size / 1e6
        osz = of.stat().st_size / 1e6
        print(f"  [{label}] DONE in {elapsed:.1f}s  paths={psz:.1f}MB  overlays={osz:.1f}MB",
              flush=True)
        return label, True, elapsed
    except Exception as e:
        import traceback
        elapsed = time.perf_counter() - t0
        print(f"  [{label}] FAILED after {elapsed:.1f}s: {e}", flush=True)
        traceback.print_exc()
        return label, False, elapsed


def main():
    parser = argparse.ArgumentParser(description="Generate multi-model MC cache")
    parser.add_argument("--workers", type=int, default=12,
                        help="Number of parallel worker processes (default: 12)")
    args = parser.parse_args()

    # Build job list
    jobs = []
    for mk in sorted(_CACHED_MODEL_KEYS):
        for yr in CACHED_START_YRS:
            jobs.append((mk, yr))

    n_workers = min(args.workers, len(jobs))
    print(f"Generating {len(jobs)} cache files using {n_workers} worker processes")
    print(f"Models: {sorted(_CACHED_MODEL_KEYS)}")
    print(f"Start years: {CACHED_START_YRS}")
    print(flush=True)

    t0 = time.perf_counter()
    results = []

    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker) as pool:
        futures = {pool.submit(_run_one, job): job for job in jobs}
        for future in as_completed(futures):
            results.append(future.result())

    elapsed = time.perf_counter() - t0
    ok = sum(1 for _, s, _ in results if s)
    fail = sum(1 for _, s, _ in results if not s)

    print(f"\n{'='*60}")
    print(f"Completed: {ok}/{len(jobs)} succeeded, {fail} failed")
    print(f"Total time: {elapsed:.1f}s")

    # Report total cache size
    cache_dir = Path("btc_web/mc_cache")
    if cache_dir.exists():
        total = sum(f.stat().st_size for f in cache_dir.glob("*.npz")
                    if any(m in f.name for m in _CACHED_MODEL_KEYS))
        print(f"New cache size: {total / 1e6:.0f} MB")

    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
