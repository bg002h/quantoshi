#!/usr/bin/env python3
"""Generate pre-computed Citadel band cache for Quick Scenarios.

Iterates all 1,620 preset combos (5 models x 3 entry Qs x 3 regimes x
3 wealth levels x 3 rule sets x 2 start years x 2 tax statuses).
Each combo: 800 simulation paths -> percentile bands -> stored as npz.

Workers return packed band arrays to the main process, which batch-writes
npz files after all workers complete (avoids concurrent npz write races).

Requires the Cython `markov` module for BTC price path generation.

Usage:
    PYTHONPATH=".:btc_web" btc_venv/bin/python3 tools/generate_citadel_bands.py
    PYTHONPATH=".:btc_web" btc_venv/bin/python3 tools/generate_citadel_bands.py --workers 8
    PYTHONPATH=".:btc_web" btc_venv/bin/python3 tools/generate_citadel_bands.py --dry-run
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
for _p in (str(_ROOT), str(_ROOT / "btc_web"), str(_ROOT)):
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

_worker_M = None
_worker_models = None


def _init_worker():
    """Initialize ModelData and price models in this worker process."""
    global _worker_M, _worker_models
    for _p in (str(_ROOT), str(_ROOT / "btc_web"), str(_ROOT)):
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

    ef_pkl = _ROOT / "model_data_ef.pkl"
    if ef_pkl.exists():
        models["ef"] = EmpiricalFloorModel(str(ef_pkl))

    _app_ctx.PRICE_MODELS = models
    _worker_models = models


def _generate_btc_paths(model_key, entry_q, start_yr, n_sims, n_periods):
    """Generate BTC price paths using the Markov module."""
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

    pct_bin = entry_q / 100.0
    dt = 1.0 / 12
    price_paths, _ = monte_carlo_prices(
        trans, bin_edges, pct_bin, n_periods, n_sims,
        model, t_start, dt,
    )
    return price_paths.astype(np.float32)


def _worker_task(args):
    """Process one combo. Returns (cache_key, packed_bands, error)."""
    model_key, entry_q, regime, wealth, rules, start_yr, tax_status, n_sims = args
    key = band_cache_key(model_key, entry_q, regime, wealth, rules,
                         start_yr, tax_status)
    try:
        cfg = build_config(wealth, regime, rules, start_yr, tax_status)
        n_periods = int((cfg.end_yr - cfg.start_yr) * 12)

        paths = _generate_btc_paths(model_key, entry_q, start_yr, n_sims, n_periods)

        # Pass model so _initial_state gets correct BTC price and
        # rebalancing triggers can compute quantile position
        from figures.citadel import _ModelAdapter
        model = _ModelAdapter(_worker_M, model_key)
        result = simulate(cfg, model=model, price_paths=paths)
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
    parser.add_argument("--sims", type=int, default=SIMS_PER_SCENARIO,
                        help=f"Sims per combo (default: {SIMS_PER_SCENARIO})")
    parser.add_argument("--flush-every", type=int, default=50,
                        help="Write npz files every N completed combos (default: 50)")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="Override cache directory")
    args = parser.parse_args()

    import os
    n_workers = args.workers or max(1, (os.cpu_count() or 4) - 2)
    n_sims = args.sims
    flush_every = args.flush_every
    cache_dir = Path(args.cache_dir) if args.cache_dir else BAND_CACHE_DIR

    combos = list(product(
        BTC_MODELS, BTC_ENTRY_QS, MACRO_REGIMES.keys(), WEALTH_LEVELS.keys(),
        RULE_SETS.keys(), START_YEARS, TAX_STATUSES,
    ))
    n_total = len(combos)
    print(f"Total combos: {n_total}")
    print(f"Sims per combo: {n_sims}")
    print(f"Workers: {n_workers}")
    print(f"Flush every: {flush_every}")
    print(f"Cache dir: {cache_dir}")

    if args.dry_run:
        for c in combos[:10]:
            print(f"  {c}")
        print(f"  ... and {n_total - 10} more")
        return

    cache_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    done, failed = 0, 0

    npz_batches: dict[tuple, dict] = {}

    def _flush_batches():
        """Write accumulated results to npz files."""
        for (model, start_yr), entries in npz_batches.items():
            npz_file = cache_dir / f"bands_{model}_{start_yr}.npz"
            # Merge with existing file if present
            existing = {}
            if npz_file.exists():
                with np.load(npz_file) as npz:
                    existing = dict(npz)
            existing.update(entries)
            np.savez(npz_file, **existing)
            print(f"  Saved {npz_file.name}: {len(existing)} entries, "
                  f"{npz_file.stat().st_size / 1e6:.1f} MB")
        npz_batches.clear()

    tasks = [(*combo, n_sims) for combo in combos]

    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker) as pool:
        for key, packed, err in pool.map(_worker_task, tasks):
            done += 1
            if packed is not None:
                parts = key.split("_")
                model = parts[0]
                start_yr = int(parts[-2])
                npz_batches.setdefault((model, start_yr), {})[key] = packed
                if done % flush_every == 0:
                    elapsed = time.perf_counter() - t0
                    rate = done / elapsed
                    eta = (n_total - done) / rate if rate > 0 else 0
                    print(f"[{done}/{n_total}] ({rate:.2f}/s, ETA {eta/60:.0f}min)")
                    _flush_batches()
            else:
                failed += 1
                print(f"[FAIL] {key}: {err}")

    # Final flush
    if npz_batches:
        _flush_batches()

    elapsed = time.perf_counter() - t0
    print(f"\nDone: {done - failed}/{n_total} OK, {failed} failed, "
          f"{elapsed:.0f}s ({elapsed/60:.1f} min)")

    total_sz = sum(f.stat().st_size for f in cache_dir.glob("bands_*.npz"))
    print(f"Total cache size: {total_sz / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
