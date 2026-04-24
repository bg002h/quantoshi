# Snapshot Defaults SSOT + Share-Link Encoding v4 — Design Spec

**Date:** 2026-04-24
**Status:** Approved (post-architect review)
**Implementation:** Two phases, two separate plans

---

## 1. Goal

Two coupled changes to the share-link subsystem:

**A. Defaults SSOT consolidation.** Today the "default value of a control" is fragmented across `btc_web/tab_defaults.py`, inlined `value=`/`data=` literals in every `btc_web/layout/*.py` file, and a handful of startup helpers / live-sync callbacks. CLAUDE.md already flags this as a maintenance hazard ("cache key alignment" footgun). One file owns the default for every control in `_SNAPSHOT_CONTROLS` (206 entries).

**B. Share-link encoding v4 (`q4:`).** Today's `q3:` encoding is a positional array of all 206 control values, gzip+base64. v4 encodes only diffs from defaults plus an 8-char fingerprint of the defaults snapshot. Old links (`q1:`, `q2:`, `q3:`) continue to decode unchanged. Server keeps a small in-repo registry of recent default-fingerprints so old `q4:` links always restore correctly.

**Result:** typical share links shrink from ~250–400 chars to ~80–150 chars; identity links (no controls deviate from defaults) compress to under 50.

---

## 2. Architecture

Two layers, two phases.

### Layer 1 — `btc_web/snapshot_defaults.py` (NEW)

Flat dict keyed by `"{component_id}:{property}"` covering exactly the 206 entries in `_SNAPSHOT_CONTROLS`:

```python
SNAPSHOT_DEFAULTS: dict[str, Any] = {
    "bub-qs:value":         ["inner", "outer", "median"],
    "bub-xscale:value":     "log",
    "bub-yscale:value":     "log",
    "bub-xrange:value":     [2010, 2033],
    "bub-yrange:value":     [-1.5, 6.05],          # log-space exponents
    "bub-toggles:value":    ["shade", "show_data", "show_today"],
    "bub-sigma-mode:value": "resqr",
    "bub-pt-size:value":    3,
    "bub-alpha:value":      0.3,
    "bub-future-n:value":   3,
    # ... 196 more entries ...
    "cp-tax-config:data":   None,                  # None = tax disabled
    "hm-active-model:data": "bub",                 # Store default
}

# Sentinel for "live-derived; resolved by *_defaults() at request time"
LIVE_DERIVED = None
```

Plus a fingerprint function:

```python
def _compute_snapshot_defaults_fingerprint() -> str:
    """8-char SHA256 over values, ordered by _SNAPSHOT_CONTROLS.
    Stable under benign dict-literal reorderings."""
    import hashlib, json
    h = hashlib.sha256()
    for cid, prop in _SNAPSHOT_CONTROLS:
        val = SNAPSHOT_DEFAULTS.get(f"{cid}:{prop}")
        h.update(json.dumps(val, sort_keys=True).encode())
        h.update(b"\x00")
    return h.hexdigest()[:8]
```

### Layer 2 — `btc_web/tab_defaults.py` (RETAINED, ROLE NARROWED)

Stays as the **figure-builder-params adapter**. Its `*_defaults()` functions translate widget-level values → figure-builder params:

- `bub-xrange:value = [2010, 2033]` → `xmin=2010, xmax=2033`
- `bub-yrange:value = [-1.5, 6.05]` → `ymin=10**-1.5, ymax=10**6.05`
- `sc-d0..sc-d4:value = 5 scalars` → `delays = (d0, d1, d2, d3, d4)`
- `dca-yr-range:value = None` → `[yr_now, yr_now+10]` at request time
- `hm-entry-q:value = 50.0` (static fallback) → live ticker callback overwrites in browser

The frozen `BUBBLE`, `HEATMAP`, `DCA`, `RETIRE`, `SUPERCHARGE`, `CITADEL` `MappingProxyType` dicts stay (callers import `BUBBLE["xscale"]` directly), but their values are now derived from `SNAPSHOT_DEFAULTS` at module load. Callback-internal derived keys (`xmin`, `xmax`, `lots`, `is_mobile`, `sc_live_price`) stay in `tab_defaults.py` — they are NOT widgets and do not belong in `SNAPSHOT_DEFAULTS`.

### Phase split

| Phase | Scope | Encoding shipped |
|-------|-------|------------------|
| 1 | Create `snapshot_defaults.py`, migrate `tab_defaults.py` to derive from it, migrate layout literals to import from it. | `q3:` (unchanged) |
| 2 | Add `q4:` encoder/decoder, registry, dev tool, registry test. | `q4:` (new); `q1:`/`q2:`/`q3:` continue to decode. |

Phase 1 ships independently and must produce visually-identical first paint on every tab. Phase 2 begins only after Phase 1 is deployed and stable.

---

## 3. Live-Derived Default Handling

Three categories of "default":

| Category | Examples | Strategy |
|---|---|---|
| **Pure static** | `bub-xscale:value = "log"`, `bub-pt-size:value = 3` | Plain entry in `SNAPSHOT_DEFAULTS`. |
| **Static fallback for live-derived** | `hm-entry-q:value = 50.0`, `hm-entry-yr:value = current_year` | Static placeholder in `SNAPSHOT_DEFAULTS`; existing live-ticker callback overwrites in browser shortly after first paint. Used by prewarm + `q4:` decode when omitted. |
| **None-sentinel year-derived** | `dca-yr-range:value = None` | None means "translate at request time in the adapter" (e.g. `dca_defaults()` substitutes `[yr_now, yr_now+10]`). Documented inline in `snapshot_defaults.py`. |

`current_year` for `hm-entry-yr` falls into the static fallback category — the value committed to `SNAPSHOT_DEFAULTS` is a literal year (e.g. 2026); the registry updater bumps it as part of the New Year deploy ritual.

**Rejected: post-mount callback layer.** Would either trigger spurious chart re-renders (server round-trips) or require new circular-storm guards. The existing live-sync callbacks already do the right thing on first paint.

---

## 4. Fingerprint Scheme

8-char hex (32 bits) of SHA256 over `SNAPSHOT_DEFAULTS` values, iterated in `_SNAPSHOT_CONTROLS` order. Null-byte separator between entries to prevent value-boundary collisions.

**Properties:**
- Stable under dict-literal reorderings in `snapshot_defaults.py`.
- Changes when any default value changes.
- Does NOT change when a new control is appended to `_SNAPSHOT_CONTROLS` with a `None` default (or no entry in `SNAPSHOT_DEFAULTS`) — the fingerprint hashes `None` for the new position, identical to "key absent".
- Does NOT change when `_CHECKLIST_OPTIONS` grows (e.g. new model added) — `_CHECKLIST_OPTIONS` is not in the fingerprint input.
- 32 bits is overkill for a registry of 20 entries; collision probability negligible.

**Reuse for L0 cache.** The existing `_compute_defaults_hash()` in `tab_defaults.py` is replaced by importing `_compute_snapshot_defaults_fingerprint` from `snapshot_defaults.py`. One hash for both the L0 pinned-cache invalidation and the `q4:` registry lookup.

---

## 5. v4 Encoding Format

URL prefix: `q4:<8-char-fp>:<base64-gzip-payload>`

Payload (after gzip+base64 decode) is a 3-tuple JSON array:

```json
[
  "a3c87f12",
  {"4": "lin", "12": [0.5], "47": 1500},
  null
]
```

- **Element 0**: 8-char fingerprint (duplicated from URL prefix for self-contained validation).
- **Element 1**: sparse `{position_index_str: value, ...}` of controls that differ from defaults at that fingerprint. Indices are stringified ints (JSON requires string keys). Checklist values still use bitmask encoding (`_CHECKLIST_OPTIONS`).
- **Element 2**: `lots_data` (same as `q3:`) or `null`.

**Encoder algorithm:**
```python
def _encode_snapshot_v4(state_dict, tab_filter=None):
    fp = _compute_snapshot_defaults_fingerprint()
    diffs = {}
    for i, (cid, prop) in enumerate(_SNAPSHOT_CONTROLS):
        if tab_filter is not None and cid != "main-tabs" and cid not in tab_filter:
            continue
        key = f"{cid}:{prop}"
        val = state_dict.get(key)
        default = SNAPSHOT_DEFAULTS.get(key)
        if val is None or val == default:
            continue
        if cid in _CHECKLIST_OPTIONS:
            val = _list_to_mask(val, _CHECKLIST_OPTIONS[cid])
        diffs[str(i)] = val
    # MC null-out for disabled tabs (preserve q3 behavior)
    _mc_null_out_diffs(diffs, state_dict)
    payload = [fp, diffs, state_dict.get("_lots")]
    blob = gzip.compress(json.dumps(payload, separators=(',', ':')).encode())
    return f"q4:{fp}:{base64.urlsafe_b64encode(blob).decode()}"
```

**Decoder algorithm:**
```python
def _decode_snapshot_v4(encoded_with_fp_prefix):
    # encoded_with_fp_prefix = "<8-char-fp>:<base64-blob>"
    fp_url, blob_b64 = encoded_with_fp_prefix.split(":", 1)
    payload = json.loads(gzip.decompress(base64.urlsafe_b64decode(blob_b64)))
    fp_payload, diffs, lots = payload
    if fp_url != fp_payload:
        return None  # tampered
    historical_defaults = _registry_lookup(fp_payload) or SNAPSHOT_DEFAULTS
    state = {}
    for i, (cid, prop) in enumerate(_SNAPSHOT_CONTROLS):
        key = f"{cid}:{prop}"
        if str(i) in diffs:
            v = diffs[str(i)]
            if cid in _CHECKLIST_OPTIONS and isinstance(v, int):
                v = _mask_to_list(v, _CHECKLIST_OPTIONS[cid])
            state[key] = v
        else:
            v = historical_defaults.get(key)
            if v is not None:
                state[key] = v
    if lots is not None:
        state["_lots"] = lots
    return state
```

**Backward compatibility.** `_decode_snapshot_by_prefix` in `snapshot.py` gains a fourth branch for `q4:`. Order tried: `q4:` → `q3:` → `q2:` → `q1:`. No change to v1/v2/v3 encoders or decoders.

---

## 6. Defaults Registry

**File:** `btc_web/snapshot_defaults_registry.json` (in-repo, git-tracked).

**Format:** JSON array, oldest-last, capped at 20 entries:

```json
[
  {
    "fp": "a3c87f12",
    "created_at": "2026-04-24",
    "defaults": { "bub-qs:value": ["inner","outer","median"], ... }
  },
  {
    "fp": "9b1d4e07",
    "created_at": "2026-04-18",
    "defaults": { ... }
  }
]
```

**Updater script:** `tools/update_defaults_registry.py`

- Computes current fingerprint from `SNAPSHOT_DEFAULTS`.
- If already in registry, exits 0 with no change.
- Else appends `{fp, today, deepcopy(SNAPSHOT_DEFAULTS)}`. Drops oldest if length > 20.
- Writes back with `json.dumps(..., indent=2, sort_keys=True)`.

**Test enforcement:** `test_snapshot.py::test_current_fingerprint_in_registry` asserts the current fingerprint is present. Forces the dev to run the updater after changing defaults — same pattern as the existing color-artifact regen.

**Cap behavior:** when an old fingerprint is dropped, links signed with that fingerprint fall through to current `SNAPSHOT_DEFAULTS`. Some omitted fields may restore wrong if the default changed since — acceptable degradation, logged as a warning at decode time.

---

## 7. File Layout

```
btc_web/
├── snapshot_defaults.py            # NEW — Layer 1 SSOT + fingerprint
├── snapshot_defaults_registry.json # NEW — versioned defaults snapshots
├── snapshot.py                     # MOD — q4 encoder/decoder added; q1/q2/q3 unchanged
├── tab_defaults.py                 # MOD — derives BUBBLE/HEATMAP/... from SNAPSHOT_DEFAULTS
├── layout/
│   ├── bubble.py                   # MOD — value=SNAPSHOT_DEFAULTS["bub-..."] for static defaults
│   ├── heatmap.py                  # MOD — same
│   ├── sim_tabs.py                 # MOD — same
│   ├── supercharge.py              # MOD — same
│   ├── citadel.py                  # MOD — same
│   ├── citadel_tax.py              # MOD — same
│   └── ... (other layout files)
├── test_snapshot_defaults.py       # NEW — covers SSOT structural invariants
└── test_snapshot.py                # MOD — adds q4 round-trip + registry tests

tools/
└── update_defaults_registry.py     # NEW — regenerator + cap enforcement
```

---

## 8. Tests

### Phase 1 — `test_snapshot_defaults.py` (new)

1. **Coverage**: every `(cid, prop)` in `_SNAPSHOT_CONTROLS` has a key in `SNAPSHOT_DEFAULTS`.
2. **No phantoms**: every key in `SNAPSHOT_DEFAULTS` corresponds to an entry in `_SNAPSHOT_CONTROLS`.
3. **Checklist representation**: for every key where `_CHECKLIST_OPTIONS` exists, `SNAPSHOT_DEFAULTS[key]` is a list (not bitmask, not None unless intentional sentinel).
4. **Fingerprint stability**: `_compute_snapshot_defaults_fingerprint()` is 8 hex chars; calling twice returns identical value.
5. **Translation parity**: `bubble_defaults()`, `heatmap_defaults()`, `dca_defaults()`, `retire_defaults()`, `supercharge_defaults()`, `citadel_defaults()` produce dicts whose widget-derived keys round-trip through the SSOT.

### Phase 1 — modifications

- `test_cache_key_alignment.py`: continues to pass unchanged. The prewarm path now sources values from `SNAPSHOT_DEFAULTS` via `*_defaults()` adapters.
- `test_defaults.py`: existing `test_inner_collections_are_tuples` may need adjustment (BUBBLE values that came from `SNAPSHOT_DEFAULTS` lists now flow through tuple-wrap in the adapter, OR the test relaxes to accept both — pick the former for minimal blast).
- `test_resqr_snapshot.py::test_bubble_default_sigma_mode_is_resqr`: passes (we already flipped this default).

### Phase 2 — `test_snapshot.py` additions

1. **Registry has current fingerprint** (forces updater script).
2. **Registry capped at 20**.
3. **q4 round-trip**: encode → decode → state matches original (within `null vs default` equivalence).
4. **q4 against historical fingerprint**: simulate "old link" by encoding against an earlier registry entry's defaults, decode against current `SNAPSHOT_DEFAULTS`, confirm omitted fields fall back to historical defaults.
5. **q4 with unknown fingerprint** (registry-evicted): decode falls back to current `SNAPSHOT_DEFAULTS`, logs warning, returns valid state.
6. **q4 fingerprint tampering**: URL-prefix fingerprint mismatching payload fingerprint returns None (same handling as malformed `q3:`).
7. **Backward compat**: existing `q1:`, `q2:`, `q3:` decoder tests continue to pass — exact same outputs as before Phase 2.

---

## 9. Phase 1 Migration Sequence (one-paragraph granularity)

The detailed Phase 1 plan goes in a separate plan document. Sequence is per-tab incremental commits, each independently deployable:

1. **Create `snapshot_defaults.py`** with all 206 entries plus fingerprint function. Zero callers yet — additive only.
2. **Add `test_snapshot_defaults.py`** with the structural invariants. Verifies the SSOT is self-consistent before any consumer migrates.
3. **Migrate `tab_defaults.py`** to derive `BUBBLE`/`HEATMAP`/etc. from `SNAPSHOT_DEFAULTS` (one tab per sub-commit). After each, full test suite + `test_cache_key_alignment` must pass.
4. **Migrate layout literals** (`value=...`, `data=...`) to import from `SNAPSHOT_DEFAULTS`. One tab per sub-commit; visual smoke (`DEV=1 bash run_web.sh`, walk every tab, confirm first paint identical) after each.
5. **Replace `_compute_defaults_hash()`** in `tab_defaults.py` with `_compute_snapshot_defaults_fingerprint`. L0 cache hash changes — `redis-cli FLUSHDB` on deploy (already in deploy script).
6. **Generate baseline registry**: run `tools/update_defaults_registry.py` for the first time; commit `snapshot_defaults_registry.json` with one entry.
7. **Deploy + verify**: full `git push` → `git pull` → `redis-cli FLUSHDB` → `systemctl restart quantoshi`. Verify every tab paints identical to pre-deploy. Old `q3:` links continue to decode and restore identically. Confirm cache hits on second visit (L0 + L1 should warm in normal time).

`dash-callback-reviewer` gate after step 5 (every layout literal touched). Hard gate before push.

---

## 10. Phase 2 Plan (one-paragraph granularity)

After Phase 1 is shipped and stable for ≥1 day:

1. **Add `q4:` encoder** in `snapshot.py`. New helper `_encode_snapshot_v4`. `_encode_snapshot` (current `q3:` builder) renamed to `_encode_snapshot_v3` for clarity, kept as fallback when `SNAPSHOT_DEFAULTS` is unavailable (it always is now, so dead code; keep one release for revert safety, then remove).
2. **Add `q4:` decoder + registry lookup** in `snapshot.py`. `_decode_snapshot_v4` with the algorithm above. `_registry_lookup` reads `snapshot_defaults_registry.json` once at module load (cached).
3. **Wire prefix detection** in `_decode_snapshot_by_prefix`: try `q4:` first, then existing `q3:`/`q2:`/`q1:`.
4. **Switch the Generate-link button** to emit `q4:` (one-line change in `manage_snapshot` callback).
5. **Add `tools/update_defaults_registry.py`**. Hooked into `test_snapshot.py::test_current_fingerprint_in_registry`.
6. **`test_snapshot.py` round-trip + historical-fingerprint tests** added.
7. **Deploy + verify**: every old link still decodes; new links are visibly shorter; old + new `q4:` links from before-deploy and after-deploy both restore correctly.

`dash-callback-reviewer` gate before push, plus a manual end-to-end test: generate `q4:` link on Tab 1 with non-default settings, paste into a fresh browser, confirm restore.

---

## 11. Trapdoors / Risks

These were surfaced by the architect agent and are checked off explicitly in the implementation plans.

| # | Trap | Where | Mitigation |
|---|------|-------|------------|
| 1 | `bub-xrange` widget = `[2010, 2033]` ≠ `xmin/xmax` figure params | `tab_defaults.bubble_defaults()`, `layout/bubble.py:47` | Translation in `bubble_defaults()` preserved; `SNAPSHOT_DEFAULTS["bub-xrange:value"]` stores widget representation; comment inline. |
| 2 | `bub-yrange` is log-space exponent | `tab_defaults.bubble_defaults()` | Translation `ymin = 10 ** val[0]` preserved; comment inline. |
| 3 | `sc-d0..sc-d4` are 5 scalars; `SUPERCHARGE["delays"]` is one tuple | `tab_defaults.supercharge_defaults()` | Adapter unpacks 5 widget keys into `delays = (d0, d1, d2, d3, d4)`. |
| 4 | `cp-tax-config:data` default must be `None` not `{}` | `SNAPSHOT_DEFAULTS["cp-tax-config:data"]` | Audit Citadel tax callback `None` handling before set; document. |
| 5 | `dca-yr-range` is year-derived | `tab_defaults.dca_defaults()` | None-sentinel; adapter substitutes `[yr_now, yr_now+10]`. Document convention. |
| 6 | `bub-toggles` value list ≠ `BUBBLE` boolean fields | `layout/bubble.py:75`, `tab_defaults.BUBBLE` | `SNAPSHOT_DEFAULTS["bub-toggles:value"] = ["shade","show_data","show_today"]`; adapter splits to `shade=True, show_data=True, show_today=True`. |
| 7 | Model-show checklists grow over time | `_CHECKLIST_OPTIONS["bub-model-show"]` etc. | Bitmask grows; old links missing high bits decode to "not selected" for new model, which is correct. Fingerprint NOT bumped on `_CHECKLIST_OPTIONS` change. |
| 8 | L0 cache invalidates on Phase 1 deploy | `_compute_defaults_hash` → `_compute_snapshot_defaults_fingerprint` | `redis-cli FLUSHDB` already in deploy script. First-request cache miss is one-time; benign. |
| 9 | Registry-evicted fingerprint | `_decode_snapshot_v4` | Falls back to current `SNAPSHOT_DEFAULTS` + warning log. Some fields may restore wrong; acceptable. |
| 10 | Falsy-zero in callbacks | Multiple chart callbacks (`callbacks/charts/__init__.py`) | Pre-existing latent bug. Migration must NOT introduce new instances of `float(x or default)` where 0 is valid. |
| 11 | gunicorn rolling restart with mixed hashes | Production deploy | Full restart (5-worker simultaneous) is atomic; not an issue with current deploy command. |
| 12 | Pre-existing test failure (`test_no_hex_literals_outside_colors_module`) | `static_pages.py` | Unrelated; ignore. |

---

## 12. Out of Scope

- Reordering or compacting `_SNAPSHOT_CONTROLS` — explicit "APPEND-ONLY" convention preserved.
- Changing `_CHECKLIST_OPTIONS` values (would invalidate every existing link).
- Web-Crypto-style URL signing or tamper detection beyond the existing payload-vs-URL fingerprint check.
- Per-control TTLs, link expiry, or server-side link storage.
- Server-side "view as user X" personalization defaults.
- Compression of the URL prefix below 8 fingerprint chars (collision risk for a tiny win).

---

## 13. Success Criteria

- Phase 1 ships with no visual regressions. Every tab paints identical to pre-deploy on first load. All 1493 existing tests continue to pass (modulo the unrelated hex-literal pre-existing failure). Old `q3:` share links restore identically.
- Phase 2 ships with `q4:` links averaging ≤ 50% the length of equivalent `q3:` links across a sample of representative configurations. Old `q1:`/`q2:`/`q3:` links continue to decode without behavior change. New `q4:` links signed against the current registry round-trip exactly. New `q4:` links signed against an evicted fingerprint decode with current defaults filling in (warning logged).

---

## 14. Open Questions

None. All decisions resolved.
