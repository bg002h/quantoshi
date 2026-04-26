# Cache Architecture

Quantoshi uses three distinct cache systems. This document explains each
layer, how invalidation works, and when manual rebuilds are needed.

## 1. Figure caches (automatic, three layers)

The main figure caches for all chart tabs. Managed by
[`btc_web/utils.py`](../btc_web/utils.py) and [`btc_web/cache.py`](../btc_web/cache.py).

| Layer | Storage | Size | TTL | Invalidation |
|-------|---------|------|-----|--------------|
| **L0** | Redis, pinned defaults | Small | 7 days | Model fingerprint + defaults hash |
| **L1** | `@lru_cache(maxsize=64)` per worker | ~32 MB/worker | None (LRU) | Reset on worker restart |
| **L2** | Redis, shared across workers | Varies | None (Redis LRU) | Model fingerprint |

**Model fingerprint**: `md5(model_data.pkl mtime + size)`, computed at startup.
When `model_data.pkl` is rebuilt, the fingerprint changes, and all Redis
keys become orphaned. New requests populate fresh cache entries.

**Gotcha**: LPPL parameter changes in `btc_core.py` do NOT bump the model
fingerprint (which is based only on `model_data.pkl`). After LPPL refits,
Redis must be flushed manually — this is handled in both the monthly
refit systemd service and in standard deploys.

## 2. MC cache (Markov chain simulations)

Pre-computed Monte Carlo simulations for the free-tier scenario grid.
Code in [`btc_web/mc_cache.py`](../btc_web/mc_cache.py).

- **Location**: `btc_web/mc_cache/*.npz` (~1.2 GB, not git-tracked)
- **Content**: ~45,000 scenarios (6 models × 3 start years × 3 entry
  percentile bins × 6 withdrawal amounts × 7 inflation rates × 6 stack sizes)
- **Simulations per combo**: 200 price paths, 480 monthly steps each
- **Build time**: 2-4 hours, CPU-bound
- **Runtime**: loaded into RAM at startup (~834 MB), persisted to
  `/dev/shm/quantoshi_mc.pkl` for fast restart (~0.7s vs 7s from npz)
- **Snapshot invalidation**: mtime+size fingerprint of source npz files;
  stale `/dev/shm` snapshot automatically regenerates

## 3. Citadel band cache

Pre-computed Citadel Planner simulation bands.

- **Location**: `btc_web/citadel_band_cache/*.npz` (~200 MB, not git-tracked)
- **Content**: 4 models × 2 start years × ~162 combos = ~1,296 entries
- **Build time**: ~4 hours with 18 `ProcessPoolExecutor` workers
- **Generator**: [`tools/generate_citadel_bands.py`](../tools/generate_citadel_bands.py)

## Refit schedules

| Schedule | What | Where | Trigger |
|----------|------|-------|---------|
| **Daily (manual)** | Prices + BM + EF + LPPL 1-4 params | Dev | User runs `update_prices.py` |
| **Daily (manual deploy)** | `redis-cli FLUSHDB` | Prod | Part of deploy command |
| **Monthly (automatic)** | LPPL weighted + no-13 variants, Redis flush, restart | Prod | `quantoshi-lppl-refit.timer` on 1st of month |
| **Quarterly (automatic)** | MC cache + Citadel bands | Dev | `quantoshi-cache-rebuild.timer` every 90 days |

## Cache age health monitoring

The `/health` endpoint exposes cache staleness:

```json
{
  "cache_age_days": 16.5,
  "cache_warn_45d": false,
  "cache_stale_90d": false,
  ...
}
```

Computed as the oldest `.npz` across both `mc_cache/` and `citadel_band_cache/`.

The [`scripts/quantoshi-health`](../scripts/quantoshi-health) checker reads
this and escalates:

- **< 45 days**: ALL OK, cache age shown in verbose output
- **45-90 days**: Warning error — "Cache aging, consider rebuild"
- **> 90 days**: Stale error — "Cache STALE, run rebuild_caches.sh"

Both error levels trigger the existing PyQt6 fullscreen popup alert when
the health check runs with `--popup` (called by a local systemd timer).

## Manual rebuild

Use [`tools/rebuild_caches.sh`](../tools/rebuild_caches.sh) to regenerate
both heavy caches and rsync to prod:

```bash
bash tools/rebuild_caches.sh              # Both caches + deploy
bash tools/rebuild_caches.sh --mc         # MC cache only
bash tools/rebuild_caches.sh --citadel    # Citadel bands only
bash tools/rebuild_caches.sh --no-deploy  # Build only, skip rsync
```

## Automatic quarterly rebuild (systemd)

Two variants in [`tools/`](../tools/):

**System-level** (recommended, no linger needed):
- `quantoshi-cache-rebuild-system.service` + `.timer`
- Install to `/etc/systemd/system/` (sudo required)
- Runs as `User=bcg`, logs to journald

**User-level**:
- `quantoshi-cache-rebuild.service` + `.timer`
- Install to `~/.config/systemd/user/`
- Requires `loginctl enable-linger` for scheduled runs when logged out
- Logs to `~/.local/log/quantoshi-cache-rebuild.log`

**Install only ONE variant** — they conflict at the same systemd path.

## Invalidation cheat sheet

| Event | What invalidates |
|-------|-----------------|
| `model_data.pkl` rebuild (daily `update_prices.py`) | All figure caches (L0/L1/L2) |
| `btc_core.py` LPPL param refit | Nothing automatic — must flush Redis |
| MC/Citadel `.npz` files change | `/dev/shm` snapshot regenerates on next restart |
| Manual deploy | Explicit `redis-cli FLUSHDB` wipes everything |

## Swapping a cached MC model

The MC cache is keyed by `model.short_name`. To swap (e.g., drop `lppl`,
add `ecfg_1d_1u`):

1. **Edit `btc_web/mc_cache.py`**:
   - Update `_INTENDED_KEYS` (frozenset of `short_name`s for the next rebuild).
   - Update `intended_models(M)` to instantiate exactly those keys.
   - If introducing a master alias (a dropdown value that maps to a different cached variant), add an entry to `MASTER_TO_CACHED_FALLBACK`.

   **Don't confuse**: `_INTENDED_KEYS` is hand-edited (intent for next rebuild). `_CACHED_MODEL_KEYS` is disk-derived (what's actually on disk now). They will differ during the transition window between code change and rebuild.

2. **Run tests**: `btc_venv/bin/python3 -m pytest btc_web/test_mc_cache.py -v`. All must pass before invoking the rebuild.

3. **Rebuild**: `bash tools/rebuild_caches.sh --mc` (2–4 hours, interrupt-safe via stash/commit/restore). The script holds a `flock` on `btc_web/mc_cache/.rebuild.lock` to prevent concurrent invocations.

4. **Verify locally — split by transition state**:
   - **Code-change day** (before rebuild): tests pass; app starts; the MC source dropdown bolds the *old* cached set (disk hasn't changed); picking the *new* master falls through to silent live-compute miss on free tier (acceptable transient).
   - **Rebuild day** (after generate completes): app starts; dropdown bolding auto-flips; picking the new master renders a trace.

5. **Deploy**: `tools/rebuild_caches.sh` rsyncs the new files to prod and emits a reminder to restart. Manual: `ssh root@... "redis-cli FLUSHDB && systemctl restart quantoshi"`.

6. **Smoke test on prod**: `/1`, enable MC, pick the new master, confirm a trace renders.

After rebuild lands, edit `MASTER_TO_CACHED_FALLBACK` to remove any transition-only entries (look for the `# transition: ...` comment) — they otherwise become dead noise that misleads future editors.
