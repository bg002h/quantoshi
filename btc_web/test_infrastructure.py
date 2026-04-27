"""Tests for Redis cache, Celery tasks, and serialization infrastructure."""

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
for _p in (str(_ROOT), str(_ROOT / "btc_web"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import base64
import json
import dataclasses

import numpy as np
import pytest


# ══════════════════════════════════════════════════════════════════════════════
# Cache module
# ══════════════════════════════════════════════════════════════════════════════

class TestModelFingerprint:
    def test_fingerprint_is_stable(self):
        """Same pkl produces same fingerprint."""
        from _app_ctx import _compute_model_fingerprint
        fp1 = _compute_model_fingerprint()
        fp2 = _compute_model_fingerprint()
        assert fp1 == fp2
        assert len(fp1) == 8  # md5 hex[:8]

    def test_fingerprint_in_cache_key(self):
        from cache import _cache_key, _MODEL_FP
        from time_basis import TIME_BASIS
        key = _cache_key("test", '{"a": 1}')
        assert _MODEL_FP in key
        assert key.startswith(f"fig:{TIME_BASIS}:{_MODEL_FP}:")

    def test_different_params_different_keys(self):
        from cache import _cache_key
        k1 = _cache_key("bub", '{"q": 0.25}')
        k2 = _cache_key("bub", '{"q": 0.50}')
        assert k1 != k2

    def test_different_prefix_different_keys(self):
        from cache import _cache_key
        k1 = _cache_key("bub", '{"q": 0.25}')
        k2 = _cache_key("ret", '{"q": 0.25}')
        assert k1 != k2

    def test_hash_length_is_32(self):
        """SHA-256 truncated to 32 hex chars (128 bits)."""
        from cache import _cache_key, _MODEL_FP
        key = _cache_key("x", '{}')
        # Format: fig:{TIME_BASIS}:{fp}:{prefix}:{hash32}
        parts = key.split(":")
        assert len(parts) == 5
        assert len(parts[4]) == 32


class TestCacheGracefulDegradation:
    def test_get_returns_none_without_redis(self):
        """get_cached returns None when Redis unavailable (no error)."""
        from cache import get_cached
        result = get_cached("test", '{"a": 1}')
        # Might be None (no local Redis) or actual value (Redis running)
        # Either way, no exception
        assert result is None or isinstance(result, dict)

    def test_set_doesnt_raise_without_redis(self):
        """set_cached is non-fatal when Redis unavailable."""
        from cache import set_cached
        # Should not raise
        set_cached("test", '{"a": 1}', {"figure": "test"})

    def test_citadel_cache_roundtrip_graceful(self):
        from cache import get_citadel_cached, set_citadel_cached
        set_citadel_cached("test_key", {"hello": "world"})
        result = get_citadel_cached("test_key")
        # None if no Redis, or the dict if Redis available
        assert result is None or result == {"hello": "world"}


# ══════════════════════════════════════════════════════════════════════════════
# Celery task serialization
# ══════════════════════════════════════════════════════════════════════════════

class TestSimConfigSerialization:
    def test_asdict_roundtrip(self):
        """SimConfig survives dataclasses.asdict → SimConfig(**d) roundtrip."""
        from engines.citadel import SimConfig
        cfg = SimConfig.default()
        d = dataclasses.asdict(cfg)
        # Remove non-serializable fields
        d.pop("asset_matrices", None)
        # JSON roundtrip (simulates Celery transport)
        j = json.dumps(d, default=str)
        d2 = json.loads(j)
        # Reconstruct
        valid_fields = {f.name for f in dataclasses.fields(SimConfig)}
        filtered = {k: v for k, v in d2.items() if k in valid_fields}
        cfg2 = SimConfig(**filtered)
        assert cfg2.start_stack == cfg.start_stack
        assert cfg2.cash_initial == cfg.cash_initial
        assert cfg2.monthly_spend == cfg.monthly_spend
        assert cfg2.high_q_trigger == cfg.high_q_trigger
        assert len(cfg2.reserve_bins) == 3
        assert len(cfg2.invest_bins) == 2

    def test_nested_dicts_survive_roundtrip(self):
        from engines.citadel import SimConfig
        cfg = SimConfig.default()
        d = dataclasses.asdict(cfg)
        d.pop("asset_matrices", None)
        j = json.dumps(d, default=str)
        d2 = json.loads(j)
        # Check nested dicts preserved
        assert d2["high_q_action"]["mode"] == "gradual"
        assert d2["high_q_action"]["split"]["cash"] == 0.20
        assert d2["reserve_bins"][0]["label"] == "Short (T-Bills)"


class TestPricePathSerialization:
    def test_encode_decode_roundtrip(self):
        """Price paths survive base64 encode → decode roundtrip."""
        from engines.adapter import _encode_paths, _decode_paths
        paths = np.random.default_rng(42).uniform(50000, 200000, (10, 48))
        b64, shape = _encode_paths(paths)
        recovered = _decode_paths(b64, shape)
        assert np.allclose(paths, recovered)

    def test_shape_preserved(self):
        from engines.adapter import _encode_paths, _decode_paths
        paths = np.ones((5, 100))
        b64, shape = _encode_paths(paths)
        assert shape == [5, 100]
        recovered = _decode_paths(b64, shape)
        assert recovered.shape == (5, 100)


# ══════════════════════════════════════════════════════════════════════════════
# Serialized model for Celery workers
# ══════════════════════════════════════════════════════════════════════════════

class TestSerializedModel:
    def test_price_at_interpolates(self):
        from tasks import _SerializedModel
        q_grid = [0.1, 0.25, 0.5, 0.75, 0.9]
        prices_at_t20 = [50000, 80000, 120000, 200000, 350000]
        model = _SerializedModel(
            q_grid=q_grid,
            price_grids={"20.0": prices_at_t20},
            genesis=0.0,
        )
        # Exact grid point
        assert abs(model.price_at(0.5, 20.0) - 120000) < 1
        # Interpolated
        p = model.price_at(0.375, 20.0)
        assert 80000 < p < 120000

    def test_quantile_at_inverts(self):
        from tasks import _SerializedModel
        q_grid = [0.1, 0.25, 0.5, 0.75, 0.9]
        prices = [50000, 80000, 120000, 200000, 350000]
        model = _SerializedModel(
            q_grid=q_grid,
            price_grids={"20.0": prices},
        )
        q = model.quantile_at(120000, 20.0)
        assert abs(q - 0.5) < 0.01

    def test_empty_model_returns_defaults(self):
        from tasks import _SerializedModel
        model = _SerializedModel(q_grid=[0.5], price_grids={})
        assert model.price_at(0.5, 20.0) == 100000.0  # fallback
        assert model.quantile_at(50000, 20.0) == 0.5   # no grids


# ══════════════════════════════════════════════════════════════════════════════
# L0 pinned cache
# ══════════════════════════════════════════════════════════════════════════════

class TestPinnedCache:
    def test_pin_and_retrieve(self):
        """Pinned entries are returned without hitting L1 or L2."""
        from utils import _cached_bubble_fig
        # Pin a fake result
        _cached_bubble_fig.pin("test_pinned_key", "fake_result")
        assert _cached_bubble_fig.pinned["test_pinned_key"] == "fake_result"
        # Clean up
        del _cached_bubble_fig.pinned["test_pinned_key"]


# ══════════════════════════════════════════════════════════════════════════════
# Celery app config
# ══════════════════════════════════════════════════════════════════════════════

class TestCeleryConfig:
    def test_celery_app_loads(self):
        from celery_app import celery_app
        assert celery_app.main == "quantoshi"

    def test_beat_schedule_has_price_task(self):
        from celery_app import celery_app
        schedule = celery_app.conf.beat_schedule
        assert "fetch-btc-price" in schedule
        assert schedule["fetch-btc-price"]["schedule"] == 1200.0

    def test_beat_schedule_has_sparkline_task(self):
        from celery_app import celery_app
        schedule = celery_app.conf.beat_schedule
        assert "fetch-sparkline" in schedule

    def test_task_serializer_is_json(self):
        from celery_app import celery_app
        assert celery_app.conf.task_serializer == "json"

    def test_task_includes_module(self):
        from celery_app import celery_app
        assert "btc_web.tasks" in celery_app.conf.include
