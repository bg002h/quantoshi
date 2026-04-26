# MC Cache Model Swap — Design

**Status:** Approved (architect-reviewed across all 4 sections)
**Author:** bg + Claude
**Date:** 2026-04-26

## Goal

Make swapping a model in the precomputed MC cache a mechanical, single-source-of-truth operation. First application: drop LPPL (1-frequency variant) and add EPPL→`ecfg_1d_1u` (1-log entropy-damped + 1-cal undamped variant of the EPPL config family). Code/script changes today; the 2–4 hour cache rebuild is deferred.

## Architecture

### Section 1 — Source-of-truth refactor

Today, two independent hardcoded lists describe "what's cached":

- `btc_web/mc_cache.py:51`: `_CACHED_MODEL_KEYS = frozenset([...])` — runtime check for UI bolding and resolver fallback.
- `tools/rebuild_caches.sh:64-71`: inline Python `models = {…}` literal — the rebuild script's actual instantiations.

Drift hazard: edit one and forget the other. `ef` already drifted (in `_CACHED_MODEL_KEYS` but not in the rebuild-script literal).

The new design uses **two declarative inputs and two derived views**:

**Inputs (in `btc_web/mc_cache.py`):**

- `_INTENDED_KEYS: frozenset[str]` — the model `short_name`s the next rebuild will produce. Edited by hand. Used at runtime as a sanity-check oracle (no `model_data.pkl` load).

  Post-swap value:
  ```python
  _INTENDED_KEYS = frozenset({"bub", "qr", "pl", "ecfg_1d_1u", "exp", "ef"})
  ```
  (Replaces today's `{"bub", "qr", "pl", "lppl", "exp", "ef"}`. `ef` retained — reconciles existing drift where `ef` was in `_CACHED_MODEL_KEYS` but missing from the rebuild-script literal.)

- `intended_models(M) -> dict[str, PriceModel]` — public function returning `{short_name: model_instance}` for `_INTENDED_KEYS`. Called only by `tools/rebuild_caches.sh` (which always has `M` loaded). The shell script imports it; no inline literal.
- `MASTER_TO_CACHED_FALLBACK: dict[str, str]` — maps each user-facing master ("lppl", "eppl") to its preferred cached variant ("lppl", "ecfg_1d_1u"). Single source of truth for the resolver fallback that previously lived as inline code in `_resolve_mc_model_src`.

**Derived (computed at module load):**

- `_CACHED_MODEL_KEYS: frozenset[str]` — built by globbing `mc_cache/overlays_*.npz` filenames and parsing the model key from each. Reflects what's *on disk now*, independent of intent. Globs `overlays_*` (NOT `paths_*`) because overlay files are written *second* in `generate_cache`, so their presence implies both halves exist (avoids the partial-write UI lie).

**Filename parser** — anchored regex handles multi-underscore keys like `ecfg_1d_1u`:

```python
_CACHE_FILE_RE = re.compile(r"^(paths|overlays)_(.+)_(\d{4})\.npz(?:\.bak)?$")

def _parse_cache_filename(name: str) -> tuple[str, str, int] | None:
    """Return (kind, model_key, year) or None. Pure string parser."""
    m = _CACHE_FILE_RE.match(name)
    return (m.group(1), m.group(2), int(m.group(3))) if m else None
```

The 4-digit-year right-anchor disambiguates greedy `(.+)` capture of the model key.

### Section 2 — Cache key for EPPL + resolver fallback

**Cache key** is `model.short_name`. EPPL slot in `intended_models(M)` is:

```python
EPPLConfigModel("ecfg_1d_1u", M.price_years, M.price_prices, M.QR_QUANTILES)
```

Files end up named `paths_ecfg_1d_1u_2028.npz`, etc. Replaces `LPPLModel(...)`.

**`MASTER_TO_CACHED_FALLBACK`:**

```python
MASTER_TO_CACHED_FALLBACK = {
    "lppl": "lppl",            # transition: kept until LPPL purged
    "eppl": "ecfg_1d_1u",      # post-rebuild target
}
```

Skip HybPPL/PCA/etc. — YAGNI. Only ship entries whose alias becomes real this rebuild cycle.

**Resolver simplification** in `_resolve_mc_model_src` — pin master before the chain mutates it:

```python
master = src                                              # capture original
resolved = _resolve_hm_lppl_master(master, ...)
resolved = _resolve_hm_hybppl_master(resolved, ...)
resolved = _resolve_hm_eppl_master(resolved, ...)
if resolved not in _CACHED_MODEL_KEYS:
    resolved = MASTER_TO_CACHED_FALLBACK.get(master, resolved)
return resolved
```

The `.get(master, resolved)` default-to-resolved pattern means: masters absent from the dict (e.g. HybPPL) keep their resolved variant, which silent-misses cleanly when uncached. No explicit `if fallback in _CACHED_MODEL_KEYS` guard needed. Replaces the inline LPPL-specific block (commit `b28b8f4`).

Function signature keeps `src` as parameter name for caller compat; body uses `master`/`resolved` for clarity. The `master = src` pin must be the first statement before any resolver call (caught during architect review — without it, the chain mutates `src` and the fallback dict lookup misses).

**No change** to `_STARTUP_PATH_ENTRIES` / `_STARTUP_OVERLAY_ENTRIES` — they iterate `_CACHED_MODEL_KEYS` and inherit disk-derived behavior automatically.

### Section 3 — Build script + UI bolding + safe stale-file cleanup

**`tools/rebuild_caches.sh`** — replaces the inline `models = {…}` literal with a call to `intended_models(M)`:

```bash
PYTHONPATH=".:btc_web" btc_venv/bin/python3 -c "
import _app_ctx
from btc_core import load_model_data
import btc_web.mc_cache as mc
M = load_model_data('model_data.pkl')
models = mc.intended_models(M)
mc.stash_stale_files()         # rename stale → .bak (atomic, reversible)
try:
    mc.generate_all_caches(M, models)
    mc.commit_stale_files()    # delete .bak on success
except BaseException:
    mc.restore_stale_files()   # interrupt/error → undo the stash
    raise
"
```

**Two-phase purge** (replaces a naive single-step delete; the architect flagged that a 2–4 hour rebuild interrupted between purge and generate would leave the system worse off than before):

- `stash_stale_files()` — renames `paths_X_*.npz`/`overlays_X_*.npz` to `*.npz.bak` for any `X` not in `_INTENDED_KEYS`. Logs each rename. Guards with `if not CACHE_DIR.exists(): return`.
- `commit_stale_files()` — deletes all `*.npz.bak`. Called only after `generate_all_caches` succeeds.
- `restore_stale_files()` — renames `*.npz.bak` → `*.npz`. Called on interrupt/error. Idempotent.

The disk-derived `_CACHED_MODEL_KEYS` glob deliberately excludes `.bak` files (the regex matches optional `.bak` suffix to *parse* the names, but the filename pattern used at module load is `overlays_*.npz`, not `*.bak`).

**`intended_models(M)`** — public name (no leading underscore; the shell is the canonical consumer):

```python
def intended_models(M) -> dict[str, "PriceModel"]:
    """Return {short_name: model_instance} for the next rebuild target.

    Single source of truth consumed by tools/rebuild_caches.sh.
    Must instantiate exactly _INTENDED_KEYS.
    """
    from btc_core import (BubbleModel, PowerLawModel, ExponentialModel,
                          EmpiricalFloorModel, QuantileRegressionModel)
    from btc_core import EPPLConfigModel
    return {
        "bub":          BubbleModel(M),
        "qr":           QuantileRegressionModel(M),
        "pl":           PowerLawModel(M.ols_intercept, M.ols_slope,
                                       M.price_years, M.price_prices,
                                       M.genesis, M.QR_QUANTILES),
        "ecfg_1d_1u":   EPPLConfigModel("ecfg_1d_1u",
                                         M.price_years, M.price_prices,
                                         M.QR_QUANTILES),
        "exp":          ExponentialModel(M.price_years, M.price_prices,
                                          M.QR_QUANTILES),
        "ef":           EmpiricalFloorModel(str(_ef_pkl_path())),
    }
```

Helper `_ef_pkl_path()` returns the absolute path to `model_data_ef.pkl` at the repo root, mirroring the conditional registration at `app.py:343`. If the pkl is missing, `intended_models(M)` raises — making the dependency explicit and visible to anyone running the rebuild script. Reconciles the existing `ef` discrepancy (was in `_CACHED_MODEL_KEYS` but missing from the rebuild-script literal).

**UI bolding helper** — factored into `mc_cache.py` so any future surface can call it:

```python
def is_master_cached(key: str) -> bool:
    """True if this dropdown value has a usable cached variant on disk."""
    return key in _CACHED_MODEL_KEYS or \
           MASTER_TO_CACHED_FALLBACK.get(key) in _CACHED_MODEL_KEYS
```

`_mc_model_src_options` in `btc_web/layout/mc_controls.py` imports and uses it. Behavior over time:

- Today: `lppl` bold (lppl files on disk), `eppl` not bold (ecfg_1d_1u files don't exist yet).
- Post-rebuild: `lppl` not bold (purged), `eppl` bold (ecfg_1d_1u files written). Auto-flips.

**No other UI surface needs touching.** The heatmap pill bar (`_HM_PILL_MODELS_BASE`) doesn't bold cached entries. Other `_bold_opts` calls bold years/percentiles, not models.

### Section 4 — Tests + workflow doc

**Tests in new `btc_web/test_mc_cache.py`** — split between pure unit (no disk) and integration:

Pure unit:
- `test_parse_cache_filename_handles_multi_underscore` — inline strings: `paths_ecfg_1d_1u_2028.npz`, `overlays_cfg_2d_1u_2031.npz.bak`, `paths_bub_2028.npz`.
- `test_parse_cache_filename_rejects_garbage` — `paths_2028.npz`, `random.txt`, empty → `None`.
- `test_intended_models_keys_match_intended_set` — drift guard: `set(intended_models(M).keys()) == _INTENDED_KEYS`.
- `test_master_to_cached_fallback_keys_in_dropdown` — keys are valid masters in `_HM_PILL_MODELS_BASE`.
- `test_master_to_cached_fallback_values_in_intended` — values are in `_INTENDED_KEYS` (so they exist post-rebuild).
- `test_is_master_cached_direct_key` — `is_master_cached("bub")` returns True when `bub` is in a mocked `_CACHED_MODEL_KEYS`. Sanity baseline.
- `test_resolver_eppl_master_falls_back_when_variant_uncached` — monkey-patch `_CACHED_MODEL_KEYS` to a set NOT containing `ecfg_1d_1u`; assert resolver returns `ecfg_1d_1u` via the `MASTER_TO_CACHED_FALLBACK` branch. Falsifiable: remove `"eppl"` from the dict and the test breaks.
- `test_stash_commit_restore_roundtrip` — `tmp_path` fixture, fake cache with mixed intended + stale files: stash renames stale → `.bak`, commit deletes `.bak`, restore reverts.
- `test_full_cleanup_sequence_noop_on_missing_dir` — full three-phase against a nonexistent path. No exception.

Integration (skippable):
- `test_cached_model_keys_match_disk_glob` — `@pytest.mark.skipif(not CACHE_DIR.exists(), ...)`. Asserts live `_CACHED_MODEL_KEYS` matches `_parse_cache_filename` against the real cache dir.

**Workflow doc** — new section "Swapping a cached MC model" in `docs/cache_architecture.md`:

1. **Edit `btc_web/mc_cache.py`**: update `_INTENDED_KEYS`, `intended_models(M)`, and `MASTER_TO_CACHED_FALLBACK` if introducing a master alias. Note that `_INTENDED_KEYS` is the hand-edited literal; `_CACHED_MODEL_KEYS` is disk-derived — don't confuse them.
2. **Run tests**: `btc_venv/bin/python3 -m pytest btc_web/test_mc_cache.py -v`. All must pass before rebuild.
3. **Rebuild**: `bash tools/rebuild_caches.sh --mc` (2-4 hours, interrupt-safe via stash/restore).
4. **Verify locally — split by transition state**:
   - **Code-change day** (before rebuild): tests pass; app starts; MC source dropdown bolds the *old* cached set (disk hasn't changed yet); picking the *new* master falls through to silent live-compute miss on free tier (acceptable transient).
   - **Rebuild day** (after generate completes): app starts; dropdown bolding auto-flips; picking the new master renders a trace.
5. **Deploy** — handled by script (rsync + FLUSHDB + restart) unless `--no-deploy`.
6. **Smoke test on prod**: `/1`, enable MC, pick the new master, confirm trace renders.

## Today's deliverable (no rebuild)

Single coherent commit, all additions, nothing deleted from `mc_cache.py` until rebuild day:

1. Add `_INTENDED_KEYS`, `intended_models(M)`, `MASTER_TO_CACHED_FALLBACK`, `_parse_cache_filename`, `is_master_cached`, `stash_stale_files`/`commit_stale_files`/`restore_stale_files` to `btc_web/mc_cache.py`.
2. Replace `_CACHED_MODEL_KEYS` literal with disk-derived computation using the new parser.
3. Update `_resolve_mc_model_src` in `btc_web/callbacks/charts/_resolvers.py` to use `master = src` pin + generic `MASTER_TO_CACHED_FALLBACK` fallback (replaces commit `b28b8f4`'s LPPL-specific branch).
4. Update `_mc_model_src_options` in `btc_web/layout/mc_controls.py` to use `is_master_cached` helper.
5. Replace inline `models = {…}` literal in `tools/rebuild_caches.sh` with `intended_models(M)` call + stash/commit/restore flow.
6. Add `btc_web/test_mc_cache.py` with the test set listed above.
7. Add "Swapping a cached MC model" section to `docs/cache_architecture.md`.

Today's runtime behavior: identical to before the change (LPPL still bold, picking LPPL still hits cache). Tomorrow (rebuild day): no code change required; dropdown bolding flips, EPPL renders, LPPL silent-misses.

## Acknowledged residual risks (out of scope)

- **Rsync mid-transfer at gunicorn worker startup** can leave half-written `.npz` files; existing risk, not amplified by this change. Mitigation belongs in the deploy procedure (rsync to staging dir + atomic rename), not this design.
- **`_resolve_mc_model_src` parameter order** is fixed by current callers; changing it is out of scope.

## References

- Architect review feedback incorporated through 4 review cycles.
- Prior commit b28b8f4 (LPPL-specific resolver fallback) is replaced by the generic mechanism.
- `tools/rebuild_caches.sh` invocation pattern preserved (still 2–4 hours, still rsyncs, still optional `--no-deploy`).
- `/dev/shm` snapshot mtime+size fingerprint mechanism (in `shm_helpers`) handles post-rebuild invalidation automatically.
