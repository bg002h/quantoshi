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
_MODELS = ["bub", "pl", "qr", "s2f", "lppl", "exp", "ef"]
_QUANTILES = [0.10, 0.25, 0.50]
# Non-quantized models (e.g., S2F) only support Q50% (median)
_NON_QUANTIZED_ONLY_Q50 = {"s2f"}
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
    from tab_defaults import citadel_defaults

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

        quantiles = [0.50] if model_key in _NON_QUANTIZED_ONLY_Q50 else _QUANTILES
        for q in quantiles:
            key = _cache_key(model_key, q)
            logger.info("  Computing %s...", key)

            t1 = time.time()
            try:
                # Build params matching default UI settings
                p = citadel_defaults()
                p["price_model"] = model_key
                p["selected_qs"] = [q]
                p["n_sims"] = 1

                fig, mc_result = build_citadel_figure(_app_ctx.M, p)
                fig_json = fig.to_json()

                # Store in Redis
                data = {"figure": fig_json, "mc_result": mc_result}
                set_citadel_cached(key, data)  # no TTL — invalidated by model fingerprint

                elapsed = time.time() - t1
                size_kb = len(fig_json) / 1024
                logger.info("    OK: %.1fs, %.0f KB", elapsed, size_kb)
                total += 1

            except Exception as e:
                logger.error("    FAILED: %s", e)

    elapsed_total = time.time() - t0
    expected = sum(1 if mk in _NON_QUANTIZED_ONLY_Q50 else len(_QUANTILES)
                   for mk in _MODELS if mk in _app_ctx.PRICE_MODELS)
    logger.info("\nDone: %d/%d results in %.1fs", total, expected, elapsed_total)


if __name__ == "__main__":
    generate()
