# Celery + Scalability Overhaul — Design Spec (v2, post-review)

**Date:** 2026-03-26
**Branch:** `Celery`
**Goal:** Support many concurrent users on modest hardware by eliminating synchronous blocking in expensive computations and optimizing resource usage.
**Hardware:** Single Hetzner VPS, 4 vCPU. RAM to be determined (4GB tight, 8GB recommended — see RAM Budget section).

---

## Problem Statement

Current: 5 synchronous gunicorn workers. Any expensive computation blocks a worker completely. 5 concurrent expensive requests = entire site unresponsive.

| # | Bottleneck | Duration | Impact |
|---|-----------|----------|--------|
| 1 | MC simulation (live, non-cached) | 2-10s | Worker blocked |
| 2 | Citadel simulation (n_sims>1 with price_paths) | 1-3s | Worker blocked |
| 3 | Price fetching (all 5 API sources slow) | Up to 25s | Worker blocked |
| 4 | Per-worker LRU cache fragmentation | N/A | 1/5th effective hit rate |
| 5 | Gunicorn serving static assets | ~50ms each | Workers wasted |

---

## Architecture

### Before
```
Browser → nginx → gunicorn (5 sync workers) → {compute + I/O + static files}
```

### After
```
Browser → nginx ──→ /assets/ [direct from disk, no gunicorn]
                └─→ gunicorn (3-5 workers) → {fast callbacks, Redis cache hits}
                        ↕ Redis (cache + broker)
                        ↕ Celery worker(s) → {MC sims, Citadel sims}
                        ↕ Celery Beat → {price fetch every 20min}
```

---

## RAM Budget (honest accounting)

### Option A: Stay on 4GB (tight, requires optimization)

| Component | RAM |
|-----------|-----|
| OS + nginx | 300 MB |
| Redis (cache + broker) | 256 MB (`maxmemory 256mb`) |
| MC cache (shared via `--preload` COW) | 834 MB |
| gunicorn master + 3 workers (private heap) | 600 MB |
| 1 Celery worker (NO MC cache — see below) | 200 MB |
| Celery Beat | 50 MB |
| **Total** | **~2.2 GB** |
| **Remaining** | **~1.8 GB** for buffers, spikes |

**With numpy.memmap refactor:** MC cache in `/dev/shm` is shared via OS page cache across ALL processes (gunicorn + Celery). No duplication. 834 MB MC + 200 MB Citadel cache = 1.0 GB in /dev/shm, shared by all processes.

**Without memmap (fallback):** Celery workers receive pre-generated price paths via Redis task arguments (~1-5 MB per task). Gunicorn generates paths from its cached data, Celery worker runs only the simulation loop.

### Option B: Upgrade to 8GB (recommended for growth)

| Component | RAM |
|-----------|-----|
| OS + nginx | 300 MB |
| Redis | 512 MB |
| MC cache (shared) | 834 MB |
| gunicorn master + 5 workers | 1.2 GB |
| 2 Celery workers (with MC cache via COW within Celery process group) | 1.0 GB |
| Celery Beat | 50 MB |
| **Total** | **~3.9 GB** |
| **Remaining** | **~4.1 GB** |

With 8GB, Celery workers can load the MC cache independently (or via their own `--preload` equivalent) and generate price paths locally, simplifying the task interface.

**Recommendation:** Start with Option A (4GB, Celery workers receive price paths). Upgrade to 8GB when traffic justifies it. The architecture supports both — only the task argument changes.

---

## Changes

### Phase 0: Nginx Static Asset Serving (zero risk, immediate win)

Configure nginx to serve `/assets/` directly from disk:

```nginx
location /assets/ {
    alias /opt/quantoshi/btc_web/assets/;
    expires 7d;
    add_header Cache-Control "public, immutable";
}
```

**Skip `/_dash-component-suites/`** — these URLs map to multiple Python packages (dash, dash_core_components, dash_bootstrap_components, plotly) and cannot be served with a single `alias` directive. Let gunicorn continue serving these (they're aggressively cached by the browser via fingerprinted URLs anyway).

### Phase 1: Redis + Shared Figure Cache

**Install Redis:**
```bash
apt install redis-server
# /etc/redis/redis.conf:
maxmemory 256mb        # 4GB VPS
maxmemory-policy allkeys-lru
```

**New file: `btc_web/cache.py`**

Redis-backed shared figure cache with LRU fallback:

```python
import json, hashlib
try:
    import redis
    _REDIS = redis.Redis(host='localhost', port=6379, db=0, socket_timeout=1)
    _REDIS.ping()
    _HAS_REDIS = True
except Exception:
    _REDIS = None
    _HAS_REDIS = False

_FIGURE_TTL = 3600  # 1 hour

def cache_key(prefix: str, params_json: str) -> str:
    return f"fig:{prefix}:{hashlib.sha256(params_json.encode()).hexdigest()[:16]}"

def get_cached(prefix, params_json):
    if not _HAS_REDIS:
        return None
    try:
        data = _REDIS.get(cache_key(prefix, params_json))
        return json.loads(data) if data else None
    except Exception:
        return None

def set_cached(prefix, params_json, figure_data):
    if not _HAS_REDIS:
        return
    try:
        _REDIS.setex(cache_key(prefix, params_json), _FIGURE_TTL,
                     json.dumps(figure_data, default=str))
    except Exception:
        pass  # non-fatal — fall back to per-worker LRU
```

**Serialization strategy:**
- Figures: `fig.to_json()` for storage, `plotly.io.from_json()` for retrieval
- Metadata dict: `json.dumps()` / `json.loads()`
- The `(figure, mc_result)` tuple is stored as `{"figure": fig_json, "mc_result": mc_dict}`

**Migration:** `utils.py`'s `_get_mc_or_cached()` checks Redis first, then falls back to per-worker LRU. Per-worker LRU remains as hot L1 cache; Redis is shared L2.

### Phase 2: Celery for MC Simulations

**New files:**
- `btc_web/celery_app.py` — Celery configuration
- `btc_web/tasks.py` — Task definitions

**Celery config:**
```python
# celery_app.py
from celery import Celery

celery_app = Celery('quantoshi',
    broker='redis://localhost:6379/1',
    backend='redis://localhost:6379/2')
celery_app.conf.update(
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    task_soft_time_limit=60,
    task_time_limit=120,
    worker_max_tasks_per_child=50,
)
```

**Task for MC simulation:**
```python
# tasks.py
@celery_app.task(bind=True, name='run_mc_simulation')
def run_mc_simulation(self, price_paths_b64, sim_config_dict, model_key):
    """Run MC simulation with pre-generated price paths.

    price_paths: base64-encoded numpy array (generated by gunicorn worker
    from cached MC data, serialized for transport).
    sim_config_dict: SimConfig as plain dict (from SimConfig.to_dict()).
    """
    import numpy as np, base64
    price_paths = np.frombuffer(base64.b64decode(price_paths_b64),
                                dtype=np.float64).reshape(...)
    config = SimConfig.from_dict(sim_config_dict)
    result = simulate(config, model, price_paths=price_paths)
    return result.to_dict()
```

**Callback pattern:**

The existing `_mc_setup` / `_mc_finalize` already handles "result not ready" via the `mc_cached` store and `mc-pay-trigger` polling. The Celery integration extends this:

1. Gunicorn worker generates price paths from MC cache (fast, ~100ms)
2. Submits Celery task with price_paths + config (returns task_id immediately)
3. Stores task_id in a `dcc.Store`
4. Existing `mc-pay-poll` `dcc.Interval` (3-second polling, already exists) checks task status
5. When complete, result is stored in `{prefix}-mc-results` Store
6. Chart callback re-fires, finds cached result, renders fan bands

This reuses the existing BTCPay polling pattern (mc_payment.py lines 163+) without adding new polling infrastructure.

**Polling parameters** (matching existing BTCPay pattern):
- Interval: 3 seconds (`_MC_POLL_INTERVAL_MS = 3000`)
- Max intervals: 300 (`_MC_POLL_MAX = 300` = 15 minute timeout)
- Cleanup: task results auto-expire in Redis via `result_expires=3600`

### Phase 3: Celery for Citadel Simulations

Modify `engines/adapter.py`:

```python
def submit_simulation(config, model, rng_seed=42, price_paths=None):
    if _HAS_CELERY and price_paths is not None:
        from tasks import run_mc_simulation
        return run_mc_simulation.delay(
            _encode_paths(price_paths),
            config.to_dict(),
            model_key
        )
    return simulate(config, model, rng_seed, price_paths)
```

The Citadel callback already uses the run-on-click pattern (State-based inputs, Run button trigger). The Celery integration adds task submission and polling.

### Phase 4: Celery Beat for Price Fetching

Move price fetching from per-worker `dcc.Interval` callbacks to a single Celery Beat task:

```python
# celery_app.py
celery_app.conf.beat_schedule = {
    'fetch-btc-price': {
        'task': 'tasks.fetch_btc_price',
        'schedule': 1200.0,  # every 20 minutes (matching frontend interval)
    },
    'fetch-sparkline': {
        'task': 'tasks.fetch_sparkline',
        'schedule': 300.0,  # every 5 minutes
    },
}
```

The ticker callback changes from "fetch from API" to "read from Redis" (sub-millisecond, never blocks).

### Phase 5: Gevent Workers (highest risk, do last)

**Prerequisite validation:** Before committing to gevent, run a compatibility test:
```bash
pip install gevent
gunicorn btc_web.app:server --worker-class gevent --workers 1 -b 0.0.0.0:8051
# Test: all 9 tabs render, MC controls work, chart updates fire
```

**Known risks:**
- gevent monkey-patching may conflict with Python 3.14 (very new)
- The `threading.Thread` for MC prewarm becomes a greenlet
- `--preload` + gevent requires careful ordering (preload before monkey-patch)

**If gevent works:** Switch from `--workers 5` (sync) to `--workers 3 --worker-class gevent --worker-connections 100`. This handles 300 concurrent connections (vs 5 with sync) while using less memory.

**If gevent doesn't work:** Stay with sync workers. The Celery offloading (Phases 2-3) already eliminates the main blocking problem. Sync workers handle 5 concurrent requests, but since expensive work is offloaded, each request completes in <100ms (cache lookup + return). 5 sync workers at 100ms/request = 50 requests/second throughput.

---

## Degradation Modes

| Failure | User-Visible Behavior |
|---------|----------------------|
| Redis down | App falls back to per-worker LRU cache. MC sims run in-process (blocking). Price fetch runs in-process. Performance degrades to pre-Celery levels but app stays functional. |
| Celery worker dies | Polling callback times out after 15 minutes, shows error message. User can retry. Gunicorn stays responsive. |
| Celery worker busy | Task queues in Redis. User sees "Computing..." status. Other users unaffected (gunicorn handles their requests normally). |
| Redis full (256MB) | LRU eviction kicks in. Oldest cache entries evicted. New entries still cached. No errors. |
| Task exceeds time limit | Celery kills the task after 120s. Result backend gets failure state. Polling callback shows timeout error. |

---

## Serialization Strategy

| Object | Serialize | Deserialize | Transport |
|--------|-----------|-------------|-----------|
| `go.Figure` | `fig.to_json()` | `plotly.io.from_json()` | Redis figure cache |
| `SimConfig` | `dataclasses.asdict()` | `SimConfig(**d)` | Celery task args |
| `SimResult` | `result.to_dict()` (already exists) | `SimResult.from_dict(d)` (already exists) | Celery task result |
| `np.ndarray` (price_paths) | `base64.b64encode(arr.tobytes())` + shape metadata | `np.frombuffer(base64.b64decode(s)).reshape(shape)` | Celery task args |
| MC metadata dict | `json.dumps()` | `json.loads()` | Redis figure cache |

---

## MC Cache: numpy.memmap Refactor

### Problem
The MC cache is currently loaded via Python deserialization into heap objects (~834 MB). No sharing between gunicorn and Celery process groups.

### Solution: numpy.memmap
Convert cache from serialized Python dicts to structured binary files backed by `numpy.memmap`. Data stays in `/dev/shm` (tmpfs RAM), mapped directly into every process's virtual address space via OS page cache. Zero deserialization, zero duplication.

**File structure:**
```
/dev/shm/quantoshi_mc/
    index.json          # {path_key: {offset, shape, dtype}, ...}
    paths.bin           # concatenated price path arrays
    overlays.bin        # concatenated overlay arrays
```

**Access:** `np.memmap('/dev/shm/quantoshi_mc/paths.bin', dtype='float64', mode='r')` — lazy, zero-copy, shared physical RAM across all processes (gunicorn + Celery).

**Benefits:** 834 MB shared regardless of process count. Near-instant startup. No serialization security concerns.

---

## Citadel Pre-Computed Cache

### Goal
Pre-compute Citadel results for common parameters. Target: ~200 MB cache. Tab 9 returns instantly for default/popular settings.

### Cache Matrix

| Dimension | Values | Count |
|-----------|--------|-------|
| Price model | bub, pl, s2f | 3 |
| Start year | 2031 | 1 |
| Entry quantile | Q10%, Q25%, Q50% | 3 |
| Spending | $5K/mo (default) | 1 |

**Base:** 3 models x 3 quantiles = 9 deterministic results (~2 MB)

**Extended (200 MB budget):**
- MC fan bands (200 sims per combo): 9 combos x ~6 MB = ~50 MB
- Extra quantiles (Q1%, Q5%, Q75%, Q90%): +12 combos = ~3 MB
- Extra start years (2028, 2035): x3 = ~15 MB
- Spending variations ($3K, $8K, $10K/mo): x3 = ~45 MB
- **Total: ~100 combos, ~115 MB**

### Storage
Store in `/dev/shm/quantoshi_citadel/` as mmap files (matching MC pattern), or in Redis with 1-week TTL. Generation script: `btc_web/generate_citadel_cache.py`, runs at boot via systemd oneshot (~2-5 minutes).

### Free tier
All cached combos are free. Non-cached MC parameters require payment.

---

## Performance Targets

| Metric | Current | Target |
|--------|---------|--------|
| Max concurrent responsive users | 5 | 30-50+ |
| MC simulation (worker blocking) | 2-10s | 0s (offloaded to Celery) |
| Citadel simulation (worker blocking) | 1-3s | 0s (offloaded to Celery) |
| Price fetch (worker blocking) | 0-25s | 0s (Celery Beat background) |
| Static asset serving | gunicorn | nginx direct |
| Figure cache hit rate | ~20% (per-worker) | ~60-80% (shared Redis + LRU L1) |
| Gunicorn request latency (cache hit) | <100ms | <50ms |

---

## Deployment (systemd services)

| Service | Description |
|---------|-------------|
| `quantoshi.service` | gunicorn (3-5 workers depending on gevent) |
| `quantoshi-celery.service` | `celery -A btc_web.celery_app worker -c 2 --max-tasks-per-child=50` |
| `quantoshi-celery-beat.service` | `celery -A btc_web.celery_app beat` |
| `redis.service` | Redis server (system package) |
| `quantoshi-cache.service` | Oneshot: pre-load MC cache to /dev/shm (existing) |

---

## Testing

| Test | What |
|------|------|
| Redis cache roundtrip | Figure set/get produces identical output |
| Celery task execution | MC sim task completes, returns valid SimResult dict |
| SimConfig serialization | `asdict()` → `SimConfig(**d)` roundtrip |
| price_paths serialization | base64 encode/decode preserves array exactly |
| Polling callback | Result available after task completion via existing mc-pay-poll |
| Fallback: Redis down | App works with per-worker LRU only |
| Fallback: Celery down | MC sims run in-process (blocking but functional) |
| Load test | 20 concurrent users, measure response times |
| Gevent validation | All 9 tabs render correctly under gevent workers |
| nginx static serving | `/assets/*.css` served by nginx, not gunicorn |

---

## Not In Scope

- Horizontal scaling (multiple VPS) — single-server only
- Database — no database needed
- CDN — nginx caching sufficient
- WebSocket — Dash uses HTTP polling
- Docker/Kubernetes
- numpy.memmap is IN scope (see MC Cache section above)
