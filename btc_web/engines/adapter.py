"""Simulation submission adapter — in-process or Celery background task.

In-process (default): runs simulation synchronously, returns SimResult.
Celery (when available + price_paths provided): submits async task,
returns AsyncResult (caller polls for completion).
"""
from __future__ import annotations

import base64
import dataclasses
import logging

import numpy as np

from engines.citadel import SimConfig, SimResult, simulate, PriceModel

logger = logging.getLogger(__name__)

try:
    from celery_app import celery_app
    _HAS_CELERY = True
except Exception:
    _HAS_CELERY = False


def _encode_paths(paths: np.ndarray) -> tuple[str, list[int]]:
    """Encode numpy array as base64 string + shape for JSON transport."""
    return base64.b64encode(paths.astype(np.float64).tobytes()).decode(), list(paths.shape)


def _decode_paths(b64: str, shape: list[int]) -> np.ndarray:
    """Decode base64 string back to numpy array."""
    return np.frombuffer(base64.b64decode(b64), dtype=np.float64).reshape(shape)


def submit_simulation(config: SimConfig, model: PriceModel,
                      rng_seed: int = 42,
                      price_paths: np.ndarray | None = None,
                      use_celery: bool = False):
    """Run simulation — in-process or via Celery.

    Returns:
        SimResult (in-process) or celery.result.AsyncResult (Celery).
        Caller checks isinstance to determine which.
    """
    if use_celery and _HAS_CELERY and price_paths is not None:
        # Submit to Celery worker
        paths_b64, paths_shape = _encode_paths(price_paths)
        config_dict = dataclasses.asdict(config)
        config_dict.pop("asset_matrices", None)

        # Serialize model's price grids for the Celery worker
        # so it can compute correct quantile_at() for rebalancing
        model_data = None
        if hasattr(model, '_q_grid') and hasattr(model, '_price_grid_cache'):
            # _ModelAdapter with precomputed grids
            model_data = {
                "q_grid": model._q_grid.tolist(),
                "price_grids": {str(k): v.tolist() for k, v in model._price_grid_cache.items()},
                "genesis": getattr(model, 'genesis', 0.0),
            }
        elif hasattr(model, 'fits') and model.fits:
            # Generic model — build grids for the sim time range
            from btc_core import yr_to_t
            q_grid = sorted(model.fits.keys())
            t0 = yr_to_t(config.start_yr)
            ppy = 12  # monthly
            n_periods = int((config.end_yr - config.start_yr) * ppy)
            price_grids = {}
            for step in range(0, n_periods, 12):  # one grid per year
                t = t0 + step / ppy
                t_key = round(t * 12) / 12
                price_grids[str(t_key)] = [float(model.price_at(q, t)) for q in q_grid]
            model_data = {
                "q_grid": [float(q) for q in q_grid],
                "price_grids": price_grids,
                "genesis": getattr(model, 'genesis', 0.0),
            }

        # Check if a Celery worker is actually running before submitting
        try:
            insp = celery_app.control.inspect(timeout=1)
            workers = insp.active_queues()
            if not workers:
                raise ConnectionError("No Celery workers available")
            task = celery_app.send_task(
                'btc_web.tasks.run_citadel_simulation',
                kwargs={
                    "config_dict": config_dict,
                    "price_paths_b64": paths_b64,
                    "price_paths_shape": paths_shape,
                    "model_data": model_data,
                },
            )
            logger.info("Citadel MC task submitted: %s", task.id)
            return task
        except Exception as e:
            logger.info("Celery unavailable (%s) — falling through to in-process", e)

    # In-process (deterministic, no Celery, or Celery unavailable)
    return simulate(config, model, rng_seed=rng_seed, price_paths=price_paths)
