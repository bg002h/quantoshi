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
        # Remove non-serializable fields
        config_dict.pop("asset_matrices", None)
        task = celery_app.send_task(
            'btc_web.tasks.run_citadel_simulation',
            kwargs={
                "config_dict": config_dict,
                "price_paths_b64": paths_b64,
                "price_paths_shape": paths_shape,
            },
        )
        logger.info("Citadel MC task submitted: %s", task.id)
        return task

    # In-process (deterministic or no Celery)
    return simulate(config, model, rng_seed=rng_seed, price_paths=price_paths)
