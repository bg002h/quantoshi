"""Celery tasks for Quantoshi background computation.

These tasks run in Celery worker processes, not in gunicorn workers.
They receive serialized inputs and return serialized outputs via Redis.
"""
from __future__ import annotations

import base64
import json
import logging

import numpy as np

from celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name='btc_web.tasks.run_citadel_simulation')
def run_citadel_simulation(self, config_dict: dict,
                           price_paths_b64: str | None = None,
                           price_paths_shape: list | None = None) -> dict:
    """Run Citadel simulation in background.

    Args:
        config_dict: SimConfig as plain dict (via dataclasses.asdict)
        price_paths_b64: base64-encoded numpy float64 array (optional, for MC)
        price_paths_shape: [n_sims, n_periods] shape for decoding

    Returns:
        SimResult as dict (via result.to_dict())
    """
    from engines.citadel import SimConfig, simulate, PriceModel

    # Reconstruct config
    config = SimConfig(**config_dict)

    # Reconstruct price paths if provided
    price_paths = None
    if price_paths_b64 and price_paths_shape:
        raw = base64.b64decode(price_paths_b64)
        price_paths = np.frombuffer(raw, dtype=np.float64).reshape(price_paths_shape)

    # Create a simple model adapter (no _app_ctx access in Celery worker)
    class _CeleryModel:
        fits = {}
        genesis = 0.0
        def price_at(self, q, t):
            return 100000.0 * q * max(t, 0.1)
        def quantile_at(self, price, t):
            q = price / (100000.0 * max(t, 0.1))
            return max(0.001, min(q, 0.999))

    model = _CeleryModel()
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
        color = "#2ecc71" if prices[-1] >= prices[0] else "#e74c3c"
        svg = (f'<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">'
               f'<polyline points="{pts}" fill="none" stroke="{color}" '
               f'stroke-width="1.5"/></svg>')

        r.setex("btc:sparkline", 600, svg)
        return svg
    except Exception as e:
        logger.warning("Sparkline fetch failed: %s", e)
        return None
