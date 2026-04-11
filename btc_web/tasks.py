"""Celery tasks for Quantoshi background computation.

These tasks run in Celery worker processes, not in gunicorn workers.
They receive serialized inputs and return serialized outputs via Redis.
"""
from __future__ import annotations

import base64
import json
import logging
import urllib.request

import numpy as np

from celery_app import celery_app
from colors import SPARKLINE_UP_2, CITADEL_SPENDING as _SPARKLINE_DOWN_2

logger = logging.getLogger(__name__)


class _SerializedModel:
    """Model reconstructed from serialized price grids in Celery worker.

    The gunicorn worker pre-computes price grids (quantile → price at each time)
    and passes them as task arguments. This avoids loading _app_ctx in the Celery
    worker while producing correct quantile_at() results for rebalancing triggers.
    """
    def __init__(self, q_grid, price_grids, genesis=0.0):
        """
        q_grid: list of quantile values (e.g., [0.001, 0.01, ..., 0.999])
        price_grids: dict of {t_key: list of prices} — one per cached time point
        genesis: model genesis time
        """
        self.fits = {q: None for q in q_grid}
        self.genesis = genesis
        self._q_grid = np.array(q_grid)
        self._price_grids = {float(k): np.array(v) for k, v in price_grids.items()}

    def price_at(self, q: float, t: float) -> float:
        # Find nearest cached time
        t_key = round(t * 12) / 12
        grid = self._price_grids.get(t_key)
        if grid is not None:
            return float(np.interp(q, self._q_grid, grid))
        # Fallback: use nearest available grid
        if self._price_grids:
            nearest_t = min(self._price_grids.keys(), key=lambda k: abs(k - t_key))
            return float(np.interp(q, self._q_grid, self._price_grids[nearest_t]))
        return 100000.0  # absolute fallback

    def quantile_at(self, price: float, t: float) -> float:
        t_key = round(t * 12) / 12
        grid = self._price_grids.get(t_key)
        if grid is None and self._price_grids:
            nearest_t = min(self._price_grids.keys(), key=lambda k: abs(k - t_key))
            grid = self._price_grids[nearest_t]
        if grid is not None:
            q = float(np.interp(price, grid, self._q_grid))
            return max(0.001, min(q, 0.999))
        return 0.5


@celery_app.task(bind=True, name='btc_web.tasks.run_citadel_simulation')
def run_citadel_simulation(self, config_dict: dict,
                           price_paths_b64: str | None = None,
                           price_paths_shape: list | None = None,
                           model_data: dict | None = None) -> dict:
    """Run Citadel simulation in background Celery worker.

    Args:
        config_dict: SimConfig fields as plain dict (via dataclasses.asdict).
        price_paths_b64: base64-encoded numpy float64 array (MC mode)
        price_paths_shape: [n_sims, n_periods] shape for decoding
        model_data: serialized model grids from gunicorn worker:
                    {"q_grid": [...], "price_grids": {"t_key": [prices]}, "genesis": float}

    Returns:
        SimResult as dict (via result.to_dict())
    """
    from engines.citadel import SimConfig, simulate
    import dataclasses

    # Reconstruct config
    valid_fields = {f.name for f in dataclasses.fields(SimConfig)}
    filtered = {k: v for k, v in config_dict.items() if k in valid_fields}
    config = SimConfig(**filtered)

    # Reconstruct price paths
    price_paths = None
    if price_paths_b64 and price_paths_shape:
        raw = base64.b64decode(price_paths_b64)
        price_paths = np.frombuffer(raw, dtype=np.float64).reshape(price_paths_shape)

    # Reconstruct model from serialized grids
    if model_data:
        model = _SerializedModel(
            q_grid=model_data["q_grid"],
            price_grids=model_data["price_grids"],
            genesis=model_data.get("genesis", 0.0),
        )
    else:
        # Fallback: basic model (only for non-rebalancing sims)
        model = _SerializedModel(q_grid=[0.5], price_grids={}, genesis=0.0)

    result = simulate(config, model, price_paths=price_paths)
    return result.to_dict()


@celery_app.task(name='btc_web.tasks.fetch_btc_price')
def fetch_btc_price() -> dict | None:
    """Fetch BTC price and store in Redis for all workers to read."""
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0)

        # Use the same fetch logic as utils._fetch_btc_price
        import urllib.request
        url = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
        req = urllib.request.Request(url, headers={"User-Agent": "Quantoshi/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            price = float(data["price"])

        r.setex("btc:price", 120, json.dumps({"price": price, "source": "binance"}))
        logger.info("BTC price updated: $%.0f", price)
        return {"price": price}
    except Exception as e:
        logger.warning("Price fetch failed: %s", e)
        return None


@celery_app.task(name='btc_web.tasks.fetch_sparkline')
def fetch_sparkline() -> str | None:
    """Fetch 24h sparkline SVG and store in Redis."""
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0)

        url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=1"
        req = urllib.request.Request(url, headers={"User-Agent": "Quantoshi/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            prices = [p[1] for p in data.get("prices", [])]

        if not prices:
            return None

        # Build simple SVG sparkline
        mn, mx = min(prices), max(prices)
        rng = mx - mn or 1
        w, h = 60, 16
        pts = " ".join(f"{i*w/len(prices):.1f},{h - (p-mn)/rng*h:.1f}"
                       for i, p in enumerate(prices))
        color = SPARKLINE_UP_2 if prices[-1] >= prices[0] else _SPARKLINE_DOWN_2
        svg = (f'<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">'
               f'<polyline points="{pts}" fill="none" stroke="{color}" '
               f'stroke-width="1.5"/></svg>')

        r.setex("btc:sparkline", 600, svg)
        return svg
    except Exception as e:
        logger.warning("Sparkline fetch failed: %s", e)
        return None
