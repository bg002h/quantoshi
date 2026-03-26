# btc_web/engines/adapter.py
"""Simulation submission adapter. v1: in-process. v2: Celery task."""
from engines.citadel import SimConfig, SimResult, simulate, PriceModel


def submit_simulation(config: SimConfig, model: PriceModel,
                      rng_seed: int = 42) -> SimResult:
    """Run simulation in-process. Returns SimResult directly.
    v2: replace with celery_app.send_task() returning job_id."""
    return simulate(config, model, rng_seed=rng_seed)
