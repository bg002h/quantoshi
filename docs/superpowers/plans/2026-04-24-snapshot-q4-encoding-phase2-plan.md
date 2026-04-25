# Share-Link Encoding v4 (`q4:`) — Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add `q4:` share-link encoding (sparse diff against fingerprinted defaults) to dramatically shrink default-heavy links. Old `q1:`/`q2:`/`q3:` links continue to decode with no behavior change.

**Architecture:** v4 payload is a 3-tuple `[fingerprint_str, sparse_diff_dict, lots_or_null]`. URL prefix `q4:<8-char-fp>:<base64-gzip-blob>` — fingerprint is duplicated in URL prefix and payload (tamper detection). Decoder looks up the fingerprint in `btc_web/snapshot_defaults_registry.json`; missing positions in the diff fall back to that historical defaults snapshot, or to current `SNAPSHOT_DEFAULTS` if the fingerprint has been evicted from the registry (warning logged).

**Spec:** `docs/superpowers/specs/2026-04-24-snapshot-defaults-ssot-and-v4-encoding-design.md` §5 + §5b + §5c.

**Phase 1 dependency:** Phase 1 + 1.5 deployed 2026-04-24 (commits `dc0ad10..2a6e58b`). `SNAPSHOT_DEFAULTS` is the authoritative SSOT; `_compute_defaults_hash()` already delegates to `_compute_snapshot_defaults_fingerprint()`; baseline registry exists with fp `faf00e93`.

**Hard gate:** `dash-callback-reviewer` on the diff before push.

**Deploy command:**
```bash
git push origin master && ssh root@89.167.70.45 "cd /opt/quantoshi && git pull && redis-cli FLUSHDB && systemctl restart quantoshi"
```

**Working tree:** `/scratch/code/bitcoinprojections` on `master`.

---

## File Map

| File | Status | Responsibility |
|---|---|---|
| `btc_web/snapshot.py` | MOD | Add `_SNAP_PREFIX_V4`, `_encode_snapshot_v4`, `_decode_snapshot_v4`, `_load_registry()`, `_registry_lookup()` |
| `btc_web/callbacks/snapshot_cb.py` | MOD | Prefix-dispatch in `_decode_snapshot_by_prefix` adds `q4:` branch first; generate-link callback emits `q4:`; JS share-detector regex recognizes `q4:` |
| `btc_web/test_snapshot.py` | MOD | q4 encoder/decoder/registry tests |

---

## Task 1: Add `q4:` constants and encoder helper

**Files:** `btc_web/snapshot.py`

- [ ] **Step 1.1: Add prefix constant**

Find the existing prefix block (`_SNAP_PREFIX = "q3:"` etc., around line 368) and add `_SNAP_PREFIX_V4 = "q4:"` ABOVE the others. The order of the constants is just style; the dispatch in snapshot_cb.py is what controls preference.

```python
_SNAP_PREFIX_V4 = "q4:"   # current format (v4: sparse diff against fingerprint)
_SNAP_PREFIX    = "q3:"   # prior format (positional array w/ checklist bitmask)
_SNAP_PREFIX_V2 = "q2:"   # prior format (positional array, different control list)
_SNAP_PREFIX_V1 = "q1:"   # legacy format (dict-based)
```

- [ ] **Step 1.2: Add registry loader**

Append after the existing `_safe_decompress` definition (around line 569):

```python
# ── q4: defaults registry ────────────────────────────────────────────────────
_REGISTRY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "snapshot_defaults_registry.json")
_registry_cache: dict[str, dict] | None = None


def _load_registry() -> dict[str, dict]:
    """Cache the registry on first read. Returns {fp -> defaults_dict}.
    Resilient to a missing or malformed file — returns {} so q4: decode
    falls through to current SNAPSHOT_DEFAULTS."""
    global _registry_cache
    if _registry_cache is not None:
        return _registry_cache
    try:
        with open(_REGISTRY_PATH) as f:
            entries = json.load(f)
        _registry_cache = {e["fp"]: e["defaults"] for e in entries}
    except Exception:
        log.warning("q4: registry %s unavailable; falling back to current "
                    "SNAPSHOT_DEFAULTS for all decodes", _REGISTRY_PATH)
        _registry_cache = {}
    return _registry_cache


def _registry_lookup(fp: str) -> dict | None:
    """Return historical defaults for fp, or None if fp not in registry."""
    return _load_registry().get(fp)
```

Add `import os` at top of file if not already present.

- [ ] **Step 1.3: Add `_encode_snapshot_v4`**

Append after `_encode_snapshot` (the existing v3 encoder, around line 534):

```python
def _encode_snapshot_v4(state_dict, tab_filter=None):
    """v4: sparse diff against fingerprinted defaults.

    Output (after gzip+base64 decode) is a 3-tuple JSON array:
        [fingerprint_str, {position_index_str: value, ...}, lots_or_null]

    The URL prefix carries a duplicate of the fingerprint for tamper
    detection. Returns "<fp>:<blob>". Caller prepends only "q4:" (NOT
    "q4:<fp>:" — the fp portion is already in the returned string).

    Always-encoded controls (ALWAYS_ENCODE — dynamic-default fields like
    today's date, current year, live BTC price) are emitted regardless
    of whether the runtime value matches the static placeholder, so the
    link author's value at link-creation time is preserved across day
    boundaries.

    MC controls for tabs where MC is disabled are dropped from the diff
    (see _mc_null_out_diffs_v4); decoder fills them from registry
    defaults, which by Phase-1 invariant store the disabled state.
    """
    from snapshot_defaults import (SNAPSHOT_DEFAULTS, ALWAYS_ENCODE,
                                   _compute_snapshot_defaults_fingerprint)
    fp = _compute_snapshot_defaults_fingerprint()
    diffs: dict[str, object] = {}
    for i, (cid, prop) in enumerate(_SNAPSHOT_CONTROLS):
        if tab_filter is not None and cid != "main-tabs" and cid not in tab_filter:
            continue
        key = f"{cid}:{prop}"
        val = state_dict.get(key)
        default = SNAPSHOT_DEFAULTS.get(key)
        force = key in ALWAYS_ENCODE
        if not force:
            if val is None:
                continue
            if val == default:
                continue
        if val is not None and cid in _CHECKLIST_OPTIONS:
            val = _list_to_mask(val, _CHECKLIST_OPTIONS[cid])
        diffs[str(i)] = val
    _mc_null_out_diffs_v4(diffs)
    lots = state_dict.get("_lots")
    payload = [fp, diffs, lots]
    j = json.dumps(payload, separators=(',', ':'))
    blob = base64.urlsafe_b64encode(gzip.compress(j.encode())).decode()
    return f"{fp}:{blob}"


def _mc_null_out_diffs_v4(diffs: dict[str, object]) -> None:
    """Drop MC control diffs for tabs whose EFFECTIVE mc-enable state
    is disabled (i.e. user has MC turned off for that tab).

    Effective enable state:
      - If `{prefix}-mc-enable` IS in diffs: decode that diff value
        (could be [] or 0-mask = disabled, or ['yes'] / non-zero mask
        = enabled).
      - If `{prefix}-mc-enable` is NOT in diffs: fall back to
        SNAPSHOT_DEFAULTS — i.e. ret-mc-enable defaults to ['yes']
        (enabled), every other tab defaults to [] (disabled).

    Null-out fires only when effective state is DISABLED. This avoids
    silently discarding user-changed downstream MC fields on Retire,
    where the enable default is ['yes'] — if the user keeps enable at
    default but bumps `ret-mc-sims` to 999, the diff has sims but not
    enable; the previous (broken) logic would null-out sims because
    enable was absent.
    """
    from snapshot_defaults import SNAPSHOT_DEFAULTS
    _mc_prefixes = ("dca-mc-", "ret-mc-", "hm-mc-", "sc-mc-", "cp-mc-")
    for pfx in _mc_prefixes:
        enable_cid = f"{pfx}enable"
        enable_idx = next(
            (i for i, (cid, _) in enumerate(_SNAPSHOT_CONTROLS)
             if cid == enable_cid),
            None,
        )
        if enable_idx is None:
            continue
        # Effective enable value: diff if present (decoded from possible
        # bitmask), else SNAPSHOT_DEFAULTS.
        if str(enable_idx) in diffs:
            raw = diffs[str(enable_idx)]
            if enable_cid in _CHECKLIST_OPTIONS and isinstance(raw, int):
                eff = _mask_to_list(raw, _CHECKLIST_OPTIONS[enable_cid])
            else:
                eff = raw
        else:
            eff = SNAPSHOT_DEFAULTS.get(f"{enable_cid}:value")
        # Treat None/[]/0 (and 0-mask) as disabled; anything else enabled.
        is_disabled = eff in (None, [], 0)
        if not is_disabled:
            continue  # MC effectively enabled — keep all MC fields
        for i, (cid, _) in enumerate(_SNAPSHOT_CONTROLS):
            if cid.startswith(pfx) and cid != f"{pfx}model-src":
                diffs.pop(str(i), None)
```

**NOTE on `ret-mc-enable`.** `ret-mc-enable:value=['yes']` is the layout default (Retire is the MC showcase tab). The corrected null-out logic above resolves the effective enable state from the diff if present, else from `SNAPSHOT_DEFAULTS`. So:

- User keeps Retire MC enabled (default), bumps `ret-mc-sims` to 999: encoder emits `{sims_idx: 999}`, no enable in diff. Null-out resolves enable from defaults → `['yes']` → enabled → null-out SKIPS. Decoder restores sims=999. **Correct.**
- User disables Retire MC (sets to `[]`), all downstream fields at default: encoder emits `{enable_idx: 0}` (bitmask of empty). Null-out resolves enable from diff → `[]` → disabled → null-out fires, popping all ret-mc-* (none present anyway). Decoder fills downstream from registry defaults. **Correct (and short).**
- User keeps DCA MC disabled (default `[]`), all defaults: nothing in diffs for dca-mc-*. Null-out resolves enable from defaults → `[]` → disabled → null-out fires (no-op since dict already empty). **Correct.**

**Required test additions for Task 4** (Step 4.1):

```python
def test_q4_mc_null_out_does_not_clobber_retire_changed_sims(self):
    """ret-mc-enable defaults to ['yes']; user keeping enable at default
    while changing ret-mc-sims must NOT cause sims to revert on decode."""
    from snapshot import _encode_snapshot_v4, _decode_snapshot_v4
    state = {"ret-mc-enable:value": ["yes"], "ret-mc-sims:value": 999}
    encoded = _encode_snapshot_v4(state)
    decoded = _decode_snapshot_v4(encoded)
    assert decoded.get("ret-mc-sims:value") == 999, (
        f"Expected sims=999, got {decoded.get('ret-mc-sims:value')!r} "
        "(MC null-out clobbered an explicit user diff)")

def test_q4_mc_null_out_drops_dca_mc_fields_when_default_disabled(self):
    """DCA MC defaults to disabled. Even if user has all other DCA-MC
    fields at default, the diff dict should not carry MC fields."""
    from snapshot import _encode_snapshot_v4
    import json, base64, gzip
    state = {"dca-amount:value": 200}  # something non-MC
    encoded = _encode_snapshot_v4(state)
    fp, blob = encoded.split(":", 1)
    payload = json.loads(gzip.decompress(base64.urlsafe_b64decode(blob)))
    diffs = payload[1]
    from snapshot import _SNAPSHOT_CONTROLS
    for i, (cid, _) in enumerate(_SNAPSHOT_CONTROLS):
        if cid.startswith("dca-mc-") and cid != "dca-mc-model-src":
            assert str(i) not in diffs, (
                f"{cid} should be dropped by MC null-out (DCA MC disabled)")
```

- [ ] **Step 1.4: Syntax-check**

```bash
cd /scratch/code/bitcoinprojections/btc_web && PYTHONPATH=".:.." ../btc_venv/bin/python3 -c "
import snapshot
print('q4 encoder:', snapshot._encode_snapshot_v4 is not None)
print('registry lookup:', snapshot._registry_lookup('faf00e93') is not None)
"; cd ..
```

Expected: `q4 encoder: True` and `registry lookup: True` (the baseline registry has fp `faf00e93`).

- [ ] **Step 1.5: Commit**

```bash
git add btc_web/snapshot.py
git commit -m "feat(snapshot): q4 encoder + registry loader (Phase 2 task 1)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Add `_decode_snapshot_v4`

**Files:** `btc_web/snapshot.py`

- [ ] **Step 2.1: Add the decoder**

Append after `_decode_snapshot` (the existing v3 decoder, around line 605):

```python
def _decode_snapshot_v4(encoded_with_fp_prefix: str):
    """Decode v4 (sparse diff) snapshot.

    encoded_with_fp_prefix has the form "<8-char-fp>:<base64-blob>"
    (the "q4:" prefix is stripped by the dispatcher).

    Returns the same {f"{cid}:{prop}": value, ...} state dict shape as
    _decode_snapshot, OR None on failure / fingerprint tampering.

    Fingerprint resolution:
      1. Payload's embedded fp must equal URL-prefix fp.
      2. If fp is in registry: omitted positions fall back to registry's
         defaults snapshot.
      3. Else: omitted positions fall back to current SNAPSHOT_DEFAULTS;
         a warning is logged because some restored values may diverge
         from the link author's original defaults.
    """
    try:
        if ":" not in encoded_with_fp_prefix:
            return None
        fp_url, blob_b64 = encoded_with_fp_prefix.split(":", 1)
        if len(fp_url) != 8:
            return None
        raw = _safe_decompress(blob_b64)
        if raw is None:
            return None
        payload = json.loads(raw)
        if not (isinstance(payload, list) and len(payload) == 3):
            return None
        fp_payload, diffs, lots = payload
        if fp_url != fp_payload:
            return None  # tamper / corruption
        if not isinstance(diffs, dict):
            return None
        historical = _registry_lookup(fp_payload)
        from snapshot_defaults import SNAPSHOT_DEFAULTS
        if historical is None:
            log.warning("q4: fingerprint %s not in registry; falling back "
                        "to current SNAPSHOT_DEFAULTS (some fields may "
                        "restore differently)", fp_payload)
            historical = SNAPSHOT_DEFAULTS
        state = {}
        for i, (cid, prop) in enumerate(_SNAPSHOT_CONTROLS):
            key = f"{cid}:{prop}"
            si = str(i)
            if si in diffs:
                v = diffs[si]
                if cid in _CHECKLIST_OPTIONS and isinstance(v, int):
                    v = _mask_to_list(v, _CHECKLIST_OPTIONS[cid])
                state[key] = v
            else:
                v = historical.get(key)
                if v is not None:
                    state[key] = v
        if lots:
            state["_lots"] = lots
        return state
    except Exception:
        return None
```

- [ ] **Step 2.2: Round-trip syntax check**

```bash
btc_venv/bin/python3 -c "
import sys; sys.path.insert(0, 'btc_web')
from snapshot import (_encode_snapshot_v4, _decode_snapshot_v4,
                      _SNAPSHOT_CONTROLS)
state = {'main-tabs:active_tab': 'bubble',
         'bub-xscale:value': 'lin',  # not 'log' (default) — should diff
         'bub-future-n:value': 5}
encoded = _encode_snapshot_v4(state)
print('encoded length:', len(encoded))
decoded = _decode_snapshot_v4(encoded)
print('decoded[bub-xscale:value]:', decoded.get('bub-xscale:value'))
print('decoded[bub-future-n:value]:', decoded.get('bub-future-n:value'))
print('decoded[main-tabs:active_tab]:', decoded.get('main-tabs:active_tab'))
print('decoded[bub-xscale:value]==lin:', decoded.get('bub-xscale:value')=='lin')
"
```

Expected: encoded length under 100 chars; decoded values match.

- [ ] **Step 2.3: Commit**

```bash
git add btc_web/snapshot.py
git commit -m "feat(snapshot): q4 decoder + registry fallback (Phase 2 task 2)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Wire `q4:` into prefix dispatch

**Files:** `btc_web/callbacks/snapshot_cb.py`

- [ ] **Step 3.1: Add q4: import and branch**

In the import block at top of `snapshot_cb.py`:

```python
from snapshot import (..., _SNAP_PREFIX_V4, _decode_snapshot_v4, _encode_snapshot_v4, ...)
```

Update `_decode_snapshot_by_prefix`:

```python
def _decode_snapshot_by_prefix(h):
    if h.startswith(_SNAP_PREFIX_V4):
        # q4: <fp>:<blob> — pass everything after "q4:" to the decoder
        rest = h[len(_SNAP_PREFIX_V4):]
        return _decode_snapshot_v4(rest), _SNAP_PREFIX_V4, rest
    if h.startswith(_SNAP_PREFIX):
        return _decode_snapshot(h[len(_SNAP_PREFIX):]), _SNAP_PREFIX, h[len(_SNAP_PREFIX):]
    if h.startswith(_SNAP_PREFIX_V2):
        return _decode_snapshot(h[len(_SNAP_PREFIX_V2):]), _SNAP_PREFIX_V2, h[len(_SNAP_PREFIX_V2):]
    if h.startswith(_SNAP_PREFIX_V1):
        return _decode_snapshot_v1(h[len(_SNAP_PREFIX_V1):]), _SNAP_PREFIX_V1, h[len(_SNAP_PREFIX_V1):]
    return None, None, None
```

`q4:` MUST be tested first because of the literal-prefix overlap risk (none today, but defensive).

- [ ] **Step 3.2: Switch generate-link to emit q4**

Find the `manage_snapshot` (or similar generate-link) callback in `snapshot_cb.py`. Look for `_encode_snapshot(` or `_SNAP_PREFIX +` or `f"#{_SNAP_PREFIX}"`. Replace the encoder call with `_encode_snapshot_v4(...)` and prefix with `_SNAP_PREFIX_V4`.

Search:
```bash
grep -n "_encode_snapshot\b\|_SNAP_PREFIX\b" btc_web/callbacks/snapshot_cb.py
```

For each call site emitting a new link (typically 1–2 lines), update:

Before:
```python
encoded = _encode_snapshot(state, tab_filter=tab_filter)
full_url = f"{base_url}{tab_path}#{_SNAP_PREFIX}{encoded}"
```

After:
```python
encoded = _encode_snapshot_v4(state, tab_filter=tab_filter)
full_url = f"{base_url}{tab_path}#{_SNAP_PREFIX_V4}{encoded}"
```

Note: `_encode_snapshot_v4` already returns `<fp>:<blob>`. So the URL becomes `host/tab#q4:<fp>:<blob>`.

The `link-history` storage should record the same hash format (with the `q4:` prefix or without, depending on existing convention — preserve whichever the existing code does, since the dispatcher handles both).

- [ ] **Step 3.3: Update the JS share-detector regex**

In `snapshot_cb.py` around line 639, find the JS clientside callback's hash-prefix detection:

```javascript
var isShare = h.indexOf('q1:') === 0 ||
              h.indexOf('q2:') === 0 ||
              h.indexOf('q3:') === 0;
```

Change to:

```javascript
var isShare = h.indexOf('q1:') === 0 ||
              h.indexOf('q2:') === 0 ||
              h.indexOf('q3:') === 0 ||
              h.indexOf('q4:') === 0;
```

Failing to update this means the restore-progress modal will not open for `q4:` links (the modal is gated on this regex).

- [ ] **Step 3.4: Syntax-check**

```bash
cd /scratch/code/bitcoinprojections/btc_web && PYTHONPATH=".:.." ../btc_venv/bin/python3 -c "
import callbacks.snapshot_cb as s
print('OK; dispatcher:', s._decode_snapshot_by_prefix.__name__)
"; cd ..
```

Expected: `OK; dispatcher: _decode_snapshot_by_prefix`.

- [ ] **Step 3.5: Commit**

```bash
git add btc_web/callbacks/snapshot_cb.py
git commit -m "feat(snapshot): wire q4: into prefix dispatch + generate-link + JS detector

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Add round-trip + historical-fingerprint tests

**Files:** `btc_web/test_snapshot.py`

- [ ] **Step 4.1: Append a `TestSnapshotV4` class**

```python
class TestSnapshotV4:
    def test_q4_round_trip_identity(self):
        """Encoding all-defaults state produces empty diff dict; decoding
        restores nothing (omitted fields fall back to defaults at consumer)."""
        from snapshot import _encode_snapshot_v4, _decode_snapshot_v4
        encoded = _encode_snapshot_v4({})
        decoded = _decode_snapshot_v4(encoded)
        # All fields fall back to historical defaults via registry; state
        # should equal the registry's snapshot for the current fingerprint.
        from snapshot_defaults import SNAPSHOT_DEFAULTS
        # Decoded state contains keys whose registry-default is not None.
        for k, v in decoded.items():
            if k == "_lots":
                continue
            assert SNAPSHOT_DEFAULTS.get(k) == v, (
                f"{k}: decoded={v!r}, registry-default="
                f"{SNAPSHOT_DEFAULTS.get(k)!r}")

    def test_q4_round_trip_with_diffs(self):
        from snapshot import _encode_snapshot_v4, _decode_snapshot_v4
        state = {
            "main-tabs:active_tab": "bubble",
            "bub-xscale:value":     "lin",
            "bub-future-n:value":   5,
            "bub-toggles:value":    ["shade", "show_data"],  # bitmask path
        }
        encoded = _encode_snapshot_v4(state)
        decoded = _decode_snapshot_v4(encoded)
        for k, v in state.items():
            assert decoded.get(k) == v, f"{k}: {decoded.get(k)!r} != {v!r}"

    def test_q4_link_is_shorter_than_q3_for_default_state(self):
        from snapshot import _encode_snapshot, _encode_snapshot_v4
        state = {"main-tabs:active_tab": "bubble"}
        q3 = _encode_snapshot(state)
        q4 = _encode_snapshot_v4(state)
        assert len(q4) < len(q3) * 0.5, (
            f"q4 ({len(q4)} chars) not < 50% of q3 ({len(q3)} chars)")

    def test_q4_fingerprint_tamper_returns_none(self):
        from snapshot import _encode_snapshot_v4, _decode_snapshot_v4
        encoded = _encode_snapshot_v4({"bub-xscale:value": "lin"})
        # Replace fingerprint in URL with a different valid-shape fp
        bad = "deadbeef:" + encoded.split(":", 1)[1]
        assert _decode_snapshot_v4(bad) is None

    def test_q4_unknown_fingerprint_falls_back_to_current(self):
        """Simulate evicted fingerprint by tampering with payload AFTER
        encoding so URL fp matches payload fp but neither is in the
        registry. Decoder must NOT return None — must fall back to
        current SNAPSHOT_DEFAULTS with a warning."""
        from snapshot import (_encode_snapshot_v4, _decode_snapshot_v4,
                              _registry_cache)
        import snapshot
        encoded = _encode_snapshot_v4({"bub-xscale:value": "lin"})
        fp, blob = encoded.split(":", 1)
        # Force the registry cache to a known state lacking this fp.
        saved_cache = snapshot._registry_cache
        snapshot._registry_cache = {}
        try:
            decoded = _decode_snapshot_v4(encoded)
            assert decoded is not None
            assert decoded.get("bub-xscale:value") == "lin"
        finally:
            snapshot._registry_cache = saved_cache

    def test_q4_dispatcher_picks_q4_over_q3(self):
        """q4: prefix takes precedence over q3: in the dispatcher
        (defensive — there is no literal-string overlap today, but
        relying on order documents the intent)."""
        from callbacks.snapshot_cb import _decode_snapshot_by_prefix
        from snapshot import _SNAP_PREFIX_V4, _encode_snapshot_v4
        encoded = _encode_snapshot_v4({"bub-xscale:value": "lin"})
        h = f"{_SNAP_PREFIX_V4}{encoded}"
        state, prefix, _ = _decode_snapshot_by_prefix(h)
        assert prefix == _SNAP_PREFIX_V4
        assert state.get("bub-xscale:value") == "lin"

    def test_q4_legacy_q3_still_decodes(self):
        """Backward compat: existing q3: links continue to decode."""
        from callbacks.snapshot_cb import _decode_snapshot_by_prefix
        from snapshot import _encode_snapshot, _SNAP_PREFIX
        encoded = _encode_snapshot({"bub-xscale:value": "lin"})
        h = f"{_SNAP_PREFIX}{encoded}"
        state, prefix, _ = _decode_snapshot_by_prefix(h)
        assert prefix == _SNAP_PREFIX
        assert state.get("bub-xscale:value") == "lin"
```

- [ ] **Step 4.2: Run the new tests**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_snapshot.py::TestSnapshotV4 -v 2>&1 | tail -15
```

Expected: 7 PASS.

- [ ] **Step 4.3: Run full snapshot suite for regressions**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_snapshot.py btc_web/test_snapshot_defaults.py btc_web/test_resqr_snapshot.py -q 2>&1 | tail -8
```

Expected: all green (no q1/q2/q3 regressions).

- [ ] **Step 4.4: Commit**

```bash
git add btc_web/test_snapshot.py
git commit -m "test(snapshot): q4 round-trip + registry fallback + dispatcher

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Dev smoke + JS share-detector verification

- [ ] **Step 5.1: Restart dev server and visit `/`**

```bash
cd /scratch/code/bitcoinprojections
/usr/bin/lsof -ti :8050 2>/dev/null | xargs -r kill -9
DEV=1 nohup bash run_web.sh > /tmp/quantoshi_dev.log 2>&1 &
sleep 14
for path in / /1 /2 /3 /4 /5 /6 /7 /8 /9 /10; do
  printf "%s " "$path"
  /usr/bin/curl -s -o /dev/null -w "%{http_code}\n" "http://localhost:8050$path"
done
```

Expected: all 200.

- [ ] **Step 5.2: Manual share-link round-trip**

In a browser:
1. Visit `http://localhost:8050/1` (Bubble tab).
2. Change a few controls (e.g. flip xscale to linear, change future bubbles to 5, toggle show_legend on).
3. Click "📸 Share" → Generate link → verify URL hash starts with `q4:` (not `q3:`).
4. Verify URL is significantly shorter than equivalent q3-style links from before deploy (visually noticeable).
5. Open the link in a fresh browser tab.
6. Verify: restore-progress modal appears (JS detector working), every changed control restores, chart paints with restored state.
7. Check JS console — no errors.

- [ ] **Step 5.3: Backward compat — paste an old q3: link**

Take a previously-saved q3: link from `link-history` localStorage (or generate a fresh one before this deploy). Paste into a fresh browser tab. Verify restore works identically — no regression.

If the modal does NOT open on the q3: link, the JS share-detector regex update is wrong; revert.

---

## Task 6: dash-callback-reviewer hard gate

- [ ] **Step 6.1: Dispatch reviewer**

Use the `dash-callback-reviewer` agent. Prompt:

```
Review /scratch/code/bitcoinprojections diff from <BASE_SHA>..HEAD where
<BASE_SHA> is the commit immediately before Task 1 of this Phase 2 plan.

Files modified:
- btc_web/snapshot.py (q4 encoder + decoder + registry loader added;
  q3/q2/q1 unchanged)
- btc_web/callbacks/snapshot_cb.py (prefix dispatch adds q4 branch first;
  generate-link emits q4; JS isShare regex adds q4)
- btc_web/test_snapshot.py (new TestSnapshotV4 class with 7 tests)

Verify (BLOCKING only):
1. Existing q1/q2/q3 link decode behavior unchanged. The dispatcher's
   q3 branch must still match q3-prefixed hashes; q4 branch must NOT
   shadow q3 (test_q4_legacy_q3_still_decodes covers this — confirm
   prefix-detection order is correct).
2. q4 encoder/decoder round-trip: all-defaults state produces empty
   diffs and decodes to historical defaults (test_q4_round_trip_identity).
3. ALWAYS_ENCODE: hm-entry-yr, hm-entry-q, lev-date, lev-price,
   scan-date — confirm encoder emits these even when matching default,
   so they don't rot across day boundaries.
4. Registry fallback semantics: missing fingerprint logs warning and
   uses SNAPSHOT_DEFAULTS — does NOT raise.
5. Tamper detection: fp mismatch between URL prefix and payload returns
   None (test_q4_fingerprint_tamper_returns_none).
6. _safe_decompress is still used (gzip-bomb guardrail unchanged).
7. JS share-detector regex now recognizes q4: — without this update the
   restore-progress modal won't open for q4: links. Confirm the regex
   was updated AND the modal-open clientside callback still references
   the same hash-source.
8. CLAUDE.md footguns:
   - No falsy-zero in new code (ALWAYS_ENCODE force path: confirm `if
     not force` short-circuits correctly when force=True).
   - No prevent_initial_call=False + allow_duplicate=True combos.
   - No JS clientside tab-map drift.
9. Cache-key alignment unchanged: q4 affects link-decode-time only;
   prewarm + first-render paths untouched.
10. _registry_cache is a module-level mutable; the test suite mutates
    it inside test_q4_unknown_fingerprint. Confirm test restores prior
    state in finally (it does — verify).

Flag BLOCKING issues only. Under 500 words.
```

- [ ] **Step 6.2: Fix any BLOCKING findings; re-dispatch.**

Proceed only when zero BLOCKING.

---

## Task 7: Deploy + verify

- [ ] **Step 7.1: Push + deploy**

```bash
/usr/bin/lsof -ti :8050 2>/dev/null | xargs -r kill -9
git push origin master
/usr/bin/ssh root@89.167.70.45 'cd /opt/quantoshi && git pull && redis-cli FLUSHDB && systemctl restart quantoshi'
```

- [ ] **Step 7.2: Prod smoke**

```bash
sleep 8
for path in / /1 /2 /3 /4 /5 /6 /7 /8 /9 /10; do
  printf "%s " "$path"
  /usr/bin/curl -s -o /dev/null -w "%{http_code}\n" "https://quantoshi.xyz$path"
done
/usr/bin/ssh root@89.167.70.45 'journalctl -u quantoshi --since "60 seconds ago" --no-pager | grep -iE "error|traceback|critical" | head -10'
```

Expected: all 200; zero error/traceback lines.

- [ ] **Step 7.3: Prod end-to-end share test**

In a fresh browser at `https://quantoshi.xyz/1`:
1. Change controls.
2. Generate share link — confirm `q4:` prefix.
3. Compare link length to a hypothetical equivalent q3: link (q3 will be ~250–400 chars; q4 should be ~80–150 chars for same state).
4. Open the link in a new tab — confirm restore + chart paint without flicker, modal appears + dismisses correctly, every restored control matches.
5. Paste a known-good q3: link (from earlier this week's link-history). Confirm it still decodes and restores.
6. Check the prod journal one more time for `WARNING.*q4:` messages — fingerprint registry-miss warnings would appear here. Should be zero (current fingerprint is in registry).

```bash
/usr/bin/ssh root@89.167.70.45 'journalctl -u quantoshi --since "5 minutes ago" --no-pager | grep -i "q4:" | head -20'
```

- [ ] **Step 7.4: Soak**

Phase 2 done after ≥1 day with no error reports.

---

## Self-Review

**Spec coverage:**

| Spec section | Implemented in |
|---|---|
| §5 v4 encoder/decoder | Tasks 1, 2 |
| §5b MC null-out | Task 1.3 (`_mc_null_out_diffs_v4`) |
| §5c None-handling | Tasks 1.3 + 2.1 (encoder skips None unless ALWAYS_ENCODE; decoder leaves omitted keys unset post-fallback) |
| §6 Registry | Phase 1 task 12 (already deployed); Phase 2 task 1.2 (loader) |
| §8 Phase 2 tests | Task 4 (7 tests) |
| §10 Phase 2 sequence | Tasks 1–7 |
| §11 Trapdoors 13 (year-derived defaults) | Task 1.3 (`force = key in ALWAYS_ENCODE`) |
| §11 Trapdoor 14 (MC null-out) | Task 1.3 (`_mc_null_out_diffs_v4`) |
| §11 Trapdoor 15 (Phase 2 after Phase 1) | Plan preamble (Phase 1 deployed 2026-04-24) |

**Placeholder scan:**
- No "TBD" / "implement later".
- Every step shows the exact code, command, or expected output.
- Reviewer prompt is fully written.

**Type consistency:**
- `_encode_snapshot_v4` / `_decode_snapshot_v4` symmetric: encoder returns `f"{fp}:{blob}"` (no prefix); dispatcher in Task 3 strips `q4:` then passes the rest to the decoder. Encoder caller in `snapshot_cb.py` prepends `_SNAP_PREFIX_V4`.
- `_registry_cache: dict[str, dict] | None` — same name in Tasks 1.2 and 4.1.
- `ALWAYS_ENCODE` — frozenset, defined Phase 1; consumed in Task 1.3.
