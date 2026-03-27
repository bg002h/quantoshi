#!/usr/bin/env python3
"""Generate pre-computed Citadel Planner cache for instant Tab 9 loads.

Computes deterministic simulation results for common parameter combos
and stores them in Redis. Runs at boot via systemd or manually.

Cache matrix:
  - 3 price models (bub, pl, s2f) × 3 quantiles (Q10%, Q25%, Q50%)
  - Default spending ($5K/mo), inflation (4%), start 2031, end 2075
  - Target: ~9 deterministic results, ~2 MB in Redis

Usage:
    PYTHONPATH=".:btc_web:archive/btc_app" python3 btc_web/generate_citadel_cache.py
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

# Setup paths
_ROOT = Path(__file__).parent.parent
for _p in (str(_ROOT), str(_ROOT / "btc_web"), str(_ROOT / "archive" / "btc_app")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Cache dimensions
_MODELS = ["bub", "pl", "s2f"]
_QUANTILES = [0.10, 0.25, 0.50]
_START_YR = 2031
_END_YR = 2075
_MONTHLY_SPEND = 5000
_INFLATION = 4.0


def _cache_key(model: str, quantile: float) -> str:
    return f"default:{model}:q{quantile:.2f}"


def generate():
    """Generate all default Citadel results and store in Redis."""
    import os
    os.environ.setdefault("TESTING", "1")

    # Import app to initialize models
    from btc_web.app import app
    import _app_ctx
    from figures.citadel import _ModelAdapter, _build_sim_config, build_citadel_figure
    from cache import set_citadel_cached, redis_available

    if not redis_available():
        logger.error("Redis not available — cannot generate cache")
        return

    logger.info("Generating Citadel pre-computed cache...")
    logger.info("  Models: %s", _MODELS)
    logger.info("  Quantiles: %s", _QUANTILES)
    logger.info("  Start: %d, End: %d", _START_YR, _END_YR)

    total = 0
    t0 = time.time()

    for model_key in _MODELS:
        if model_key not in _app_ctx.PRICE_MODELS:
            logger.warning("  Model '%s' not available, skipping", model_key)
            continue

        for q in _QUANTILES:
            key = _cache_key(model_key, q)
            logger.info("  Computing %s...", key)

            t1 = time.time()
            try:
                # Build params matching default UI settings
                p = {
                    "start_stack": 1.0, "use_lots": False, "lots": [],
                    "cash_initial": 50000, "cash_rate": 4.0,
                    "res_short_init": 50000, "res_short_rate": 5.0, "res_short_vol": 2.0,
                    "res_med_init": 100000, "res_med_rate": 4.5, "res_med_vol": 8.0,
                    "res_long_init": 50000, "res_long_rate": 4.0, "res_long_vol": 15.0,
                    "inv_eq_init": 200000, "inv_eq_rate": 10.0, "inv_eq_vol": 16.0,
                    "inv_bd_init": 100000, "inv_bd_rate": 5.0, "inv_bd_vol": 7.0,
                    "monthly_spend": _MONTHLY_SPEND, "inflation": _INFLATION,
                    "spend_growth": 0.0,
                    "high_q_trigger": 95, "high_q_mode": "gradual",
                    "high_q_rate": 2.0, "high_q_dur": 6,
                    "high_q_split_cash": 20, "high_q_split_rs": 20,
                    "high_q_split_rm": 20, "high_q_split_rl": 10,
                    "high_q_split_eq": 20, "high_q_split_bd": 10,
                    "low_q_trigger": 5, "low_q_mode": "lump",
                    "low_q_rate": 10.0, "low_q_dur": 1,
                    "low_q_split_cash": 10, "low_q_split_rs": 10,
                    "low_q_split_rm": 10, "low_q_split_rl": 10,
                    "low_q_split_eq": 40, "low_q_split_bd": 20,
                    "lump_cooldown": 12,
                    "cash_floor": 50000, "cash_floor_growth": 0,
                    "res_short_floor": 0, "res_med_floor": 0,
                    "res_long_floor": 0, "reserve_floor_growth": 0,
                    "scf_enabled": False, "scf_amount": 0,
                    "scf_type": "term", "scf_rate": 8.0,
                    "scf_term": 60, "scf_repay_trigger": 1.0,
                    "start_yr": _START_YR, "end_yr": _END_YR,
                    "freq": "Monthly", "price_model": model_key,
                    "n_sims": 1, "selected_qs": [q],
                    "asset_return_model": "lognormal",
                    "disp_mode": "usd_per_asset",
                    "log_y": False, "annotate": True,
                    "show_legend": True, "legend_pos": "bottom-right",
                    "minor_grid": True, "palette": "default",
                }

                fig, mc_result = build_citadel_figure(_app_ctx.M, p)
                fig_json = fig.to_json()

                # Store in Redis
                data = {"figure": fig_json, "mc_result": mc_result}
                set_citadel_cached(key, data, ttl=86400 * 7)  # 1 week TTL

                elapsed = time.time() - t1
                size_kb = len(fig_json) / 1024
                logger.info("    OK: %.1fs, %.0f KB", elapsed, size_kb)
                total += 1

            except Exception as e:
                logger.error("    FAILED: %s", e)

    elapsed_total = time.time() - t0
    logger.info("\nDone: %d/%d results in %.1fs",
                total, len(_MODELS) * len(_QUANTILES), elapsed_total)


if __name__ == "__main__":
    generate()
