#!/usr/bin/env python3
"""Generate multi-model MC cache using parallel threads.

Usage:
    PYTHONPATH=".:btc_web:archive/btc_app" btc_venv/bin/python3 generate_mc_cache.py [--threads N]

Generates paths_{model}_{year}.npz and overlays_{model}_{year}.npz files
for all 6 quantized models × 3 start years = 18 jobs.

Default: 18 threads (one per job). Use --threads to limit.
"""

import sys
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ── Setup paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "btc_web"))
sys.path.insert(0, str(ROOT / "archive" / "btc_app"))

import _app_ctx
from btc_core import (load_model_data, BubbleModel, PowerLawModel, LPPLModel,
                      ExponentialModel, S2FModel, EmpiricalFloorModel,
                      QuantileRegressionModel)
from mc_cache import generate_cache, CACHED_START_YRS, _CACHED_MODEL_KEYS
from figures.common import _build_thermal_colors

# ── Load models (same as app.py startup) ─────────────────────────────────────
print("Loading model data...")
M = load_model_data()
_app_ctx.M = M

# Register all models (mirrors app.py lines 141-157)
_app_ctx.PRICE_MODELS["bub"] = BubbleModel(M)
_app_ctx.PRICE_MODELS["qr"]  = QuantileRegressionModel(M)
_app_ctx.PRICE_MODELS["pl"]  = PowerLawModel(
    M.ols_intercept, M.ols_slope, M.price_years, M.price_prices,
    M.genesis, M.QR_QUANTILES)
_app_ctx.PRICE_MODELS["lppl"] = LPPLModel(M.price_years, M.price_prices, M.QR_QUANTILES)
_app_ctx.PRICE_MODELS["exp"]  = ExponentialModel(M.price_years, M.price_prices, M.QR_QUANTILES)

# EF model (conditional)
_ef_pkl = ROOT / "btc_app" / "model_data_ef.pkl"
if _ef_pkl.exists():
    _app_ctx.PRICE_MODELS["ef"] = EmpiricalFloorModel(M, str(_ef_pkl))

# Set thermal colors on bubble model
_thermal = _build_thermal_colors(M.QR_QUANTILES)
_app_ctx.PRICE_MODELS["bub"].colors.update(_thermal)
_app_ctx.DEFAULT_MODEL = _app_ctx.PRICE_MODELS["bub"]

# S2F is not quantized — skip
_app_ctx.PRICE_MODELS["s2f"] = S2FModel(M.price_years, M.price_prices, M.genesis)


def _run_one(model_key, start_yr):
    """Generate cache for one (model, start_yr) combo."""
    model = _app_ctx.PRICE_MODELS[model_key]
    label = f"{model_key}/{start_yr}"
    t0 = time.perf_counter()
    try:
        pf, of = generate_cache(start_yr, M, model,
                                progress_cb=lambda msg: print(f"  [{label}] {msg}"))
        elapsed = time.perf_counter() - t0
        psz = pf.stat().st_size / 1e6
        osz = of.stat().st_size / 1e6
        print(f"  [{label}] DONE in {elapsed:.1f}s  paths={psz:.1f}MB  overlays={osz:.1f}MB")
        return label, True, elapsed
    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"  [{label}] FAILED after {elapsed:.1f}s: {e}")
        return label, False, elapsed


def main():
    parser = argparse.ArgumentParser(description="Generate multi-model MC cache")
    parser.add_argument("--threads", type=int, default=18,
                        help="Max parallel threads (default: 18)")
    args = parser.parse_args()

    # Build job list: all quantized models × start years
    jobs = []
    for mk in sorted(_CACHED_MODEL_KEYS):
        if mk not in _app_ctx.PRICE_MODELS:
            print(f"WARNING: model '{mk}' not registered, skipping")
            continue
        mdl = _app_ctx.PRICE_MODELS[mk]
        if not mdl.quantized:
            continue
        for yr in CACHED_START_YRS:
            jobs.append((mk, yr))

    n_threads = min(args.threads, len(jobs))
    print(f"\nGenerating {len(jobs)} cache files using {n_threads} threads...")
    print(f"Models: {sorted(_CACHED_MODEL_KEYS)}")
    print(f"Start years: {CACHED_START_YRS}")
    print()

    t0 = time.perf_counter()
    results = []

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        futures = {pool.submit(_run_one, mk, yr): (mk, yr) for mk, yr in jobs}
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
        total = sum(f.stat().st_size for f in cache_dir.glob("*.npz"))
        print(f"Total cache size: {total / 1e6:.0f} MB")

    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
