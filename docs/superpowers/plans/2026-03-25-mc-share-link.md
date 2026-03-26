# MC Share Link Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 37 new entries to `_SNAPSHOT_CONTROLS` so MC simulation settings are preserved in share links, with hybrid encoding that skips MC controls when MC is disabled.

**Architecture:** Append 36 MC controls (9 per tab x 4 tabs) + 1 heatmap palette to `_SNAPSHOT_CONTROLS`. Add bitmask encoding for MC checklists. Add hybrid encoding in `_encode_snapshot` that nulls MC controls when MC is off. Update `_TAB_CONTROLS` for single-tab scope. The `restore_from_url` callback auto-expands to new outputs. All changes in `snapshot.py` and `callbacks/nav.py` must be atomic (single commit) to avoid import assertion crash.

**Tech Stack:** Python (Dash), gzip+base64 URL encoding, bitmask checklist encoding

**Spec:** `docs/superpowers/specs/2026-03-25-mc-share-link-design.md`

---

### Task 1: Write tests for MC snapshot roundtrip

**Files:**
- Modify: `btc_web/test_web.py`

- [ ] **Step 1: Write MC snapshot roundtrip test**

Add to `TestSnapshotRoundtrip` class (after existing roundtrip tests, ~line 360):

```python
def test_mc_controls_roundtrip(self):
    """MC controls survive encode → decode roundtrip."""
    from snapshot import _SNAPSHOT_CONTROLS, _encode_snapshot, _decode_snapshot
    state = {f"{cid}:{prop}": None for cid, prop in _SNAPSHOT_CONTROLS}
    # Enable MC on DCA tab with non-default settings
    state["dca-mc-enable:value"] = ["yes"]
    state["dca-mc-start-yr:value"] = 2035
    state["dca-mc-entry-q:value"] = 30
    state["dca-mc-years:value"] = 20
    state["dca-mc-bins:value"] = 7
    state["dca-mc-regime:value"] = [0, 2, 4]  # skip bins 1,3
    state["dca-mc-sims:value"] = 1600
    state["dca-mc-window:value"] = [2012, 2024]
    state["dca-mc-advanced:value"] = ["yes"]
    encoded = _encode_snapshot(state)
    decoded = _decode_snapshot(encoded)
    assert decoded["dca-mc-enable:value"] == ["yes"]
    assert decoded["dca-mc-start-yr:value"] == 2035
    assert decoded["dca-mc-entry-q:value"] == 30
    assert decoded["dca-mc-years:value"] == 20
    assert decoded["dca-mc-bins:value"] == 7
    assert decoded["dca-mc-regime:value"] == [0, 2, 4]
    assert decoded["dca-mc-sims:value"] == 1600
    assert decoded["dca-mc-window:value"] == [2012, 2024]
    assert decoded["dca-mc-advanced:value"] == ["yes"]
```

- [ ] **Step 2: Write MC hybrid encoding test (disabled tab nulled)**

```python
def test_mc_hybrid_encoding_nulls_disabled_tabs(self):
    """MC controls encode as null when MC is not enabled on that tab."""
    from snapshot import _SNAPSHOT_CONTROLS, _encode_snapshot, _decode_snapshot
    state = {f"{cid}:{prop}": None for cid, prop in _SNAPSHOT_CONTROLS}
    # MC disabled on DCA (default: mc-enable=[])
    state["dca-mc-enable:value"] = []
    state["dca-mc-start-yr:value"] = 2035  # set but should be nulled
    state["dca-mc-bins:value"] = 7
    # MC enabled on Retire
    state["ret-mc-enable:value"] = ["yes"]
    state["ret-mc-start-yr:value"] = 2028
    encoded = _encode_snapshot(state)
    decoded = _decode_snapshot(encoded)
    # DCA MC controls should be None (nulled by hybrid encoding)
    assert decoded.get("dca-mc-start-yr:value") is None
    assert decoded.get("dca-mc-bins:value") is None
    # Retire MC controls should be preserved
    assert decoded["ret-mc-enable:value"] == ["yes"]
    assert decoded["ret-mc-start-yr:value"] == 2028
```

- [ ] **Step 3: Write MC regime bitmask test**

```python
def test_mc_regime_bitmask_roundtrip(self):
    """MC regime checklist with int values survives bitmask encode/decode."""
    from snapshot import _SNAPSHOT_CONTROLS, _encode_snapshot, _decode_snapshot
    state = {f"{cid}:{prop}": None for cid, prop in _SNAPSHOT_CONTROLS}
    state["sc-mc-enable:value"] = ["yes"]
    state["sc-mc-regime:value"] = [0, 1, 3]  # bins 0,1,3 selected
    encoded = _encode_snapshot(state)
    decoded = _decode_snapshot(encoded)
    assert sorted(decoded["sc-mc-regime:value"]) == [0, 1, 3]
```

- [ ] **Step 4: Write hm-palette roundtrip test**

```python
def test_hm_palette_roundtrip(self):
    """Heatmap palette name survives encode → decode."""
    from snapshot import _SNAPSHOT_CONTROLS, _encode_snapshot, _decode_snapshot
    state = {f"{cid}:{prop}": None for cid, prop in _SNAPSHOT_CONTROLS}
    state["hm-palette:value"] = "ocean"
    encoded = _encode_snapshot(state)
    decoded = _decode_snapshot(encoded)
    assert decoded["hm-palette:value"] == "ocean"
```

- [ ] **Step 5: Write backward compat test (old 100-entry link decoded by new code)**

```python
def test_old_link_pads_mc_to_none(self):
    """Old links with 100 entries decode correctly — MC defaults to None."""
    from snapshot import _SNAPSHOT_CONTROLS, _encode_snapshot, _decode_snapshot
    import json, gzip, base64
    # Pre-condition: new controls exist in the list
    assert len(_SNAPSHOT_CONTROLS) >= 137, "MC controls not yet added"
    # Simulate old 100-entry link
    old_values = [None] * 100
    old_values[0] = [0.5]  # bub-qs
    payload = [old_values, None]
    encoded = base64.urlsafe_b64encode(
        gzip.compress(json.dumps(payload, separators=(',', ':')).encode())).decode()
    decoded = _decode_snapshot(encoded)
    # Old controls decoded
    assert decoded.get("bub-qs:value") == [0.5]
    # MC controls padded to None
    assert decoded.get("dca-mc-enable:value") is None
    assert decoded.get("hm-palette:value") is None
```

- [ ] **Step 6: Run tests to verify they fail**

```bash
PYTHONPATH=".:btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py -k "test_mc_controls_roundtrip or test_mc_hybrid or test_mc_regime_bitmask or test_hm_palette_roundtrip or test_old_link_pads" -v --timeout=60
```

Expected: 5 FAILED — 4 from missing component IDs, 1 from `len(_SNAPSHOT_CONTROLS) >= 137` pre-condition. The backward compat test (`test_old_link_pads_mc_to_none`) uses a pre-condition assert to ensure it only runs meaningfully after the `_SNAPSHOT_CONTROLS` entries are added.

**Note:** MC placeholder components are always rendered in the layout (even when `_HAS_MARKOV=False`) — `layout/mc_controls.py` line 65-103 creates hidden placeholders for all MC component IDs. The new `Output()` declarations in `restore_from_url` will find their targets.

- [ ] **Step 7: Commit test file**

```bash
git add btc_web/test_web.py
git commit -m "test: add MC snapshot roundtrip, hybrid encoding, regime bitmask, hm-palette tests"
```

---

### Task 2: Add MC entries to `_SNAPSHOT_CONTROLS` + `_CHECKLIST_OPTIONS` (atomic)

**Files:**
- Modify: `btc_web/snapshot.py:124` (end of `_SNAPSHOT_CONTROLS`)
- Modify: `btc_web/snapshot.py:172` (end of `_CHECKLIST_OPTIONS`)

**CRITICAL:** Both changes must be in the same commit. Adding `_CHECKLIST_OPTIONS` entries without matching `_SNAPSHOT_CONTROLS` entries triggers the validation assertion at line 175 and crashes gunicorn.

- [ ] **Step 1: Append 37 entries to `_SNAPSHOT_CONTROLS`**

Before the closing `]` at line 124, add:

```python
    # ── MC controls (4 tabs x 9 controls) ────────────────────────────────
    # DCA MC
    ("dca-mc-enable",    "value"),   # 100
    ("dca-mc-start-yr",  "value"),   # 101
    ("dca-mc-entry-q",   "value"),   # 102
    ("dca-mc-years",     "value"),   # 103
    ("dca-mc-bins",      "value"),   # 104
    ("dca-mc-regime",    "value"),   # 105
    ("dca-mc-sims",      "value"),   # 106
    ("dca-mc-window",    "value"),   # 107
    ("dca-mc-advanced",  "value"),   # 108
    # Retire MC
    ("ret-mc-enable",    "value"),   # 109
    ("ret-mc-start-yr",  "value"),   # 110
    ("ret-mc-entry-q",   "value"),   # 111
    ("ret-mc-years",     "value"),   # 112
    ("ret-mc-bins",      "value"),   # 113
    ("ret-mc-regime",    "value"),   # 114
    ("ret-mc-sims",      "value"),   # 115
    ("ret-mc-window",    "value"),   # 116
    ("ret-mc-advanced",  "value"),   # 117
    # Heatmap MC
    ("hm-mc-enable",     "value"),   # 118
    ("hm-mc-start-yr",   "value"),   # 119
    ("hm-mc-entry-q",    "value"),   # 120
    ("hm-mc-years",      "value"),   # 121
    ("hm-mc-bins",       "value"),   # 122
    ("hm-mc-regime",     "value"),   # 123
    ("hm-mc-sims",       "value"),   # 124
    ("hm-mc-window",     "value"),   # 125
    ("hm-mc-advanced",   "value"),   # 126
    # Supercharger MC
    ("sc-mc-enable",     "value"),   # 127
    ("sc-mc-start-yr",   "value"),   # 128
    ("sc-mc-entry-q",    "value"),   # 129
    ("sc-mc-years",      "value"),   # 130
    ("sc-mc-bins",       "value"),   # 131
    ("sc-mc-regime",     "value"),   # 132
    ("sc-mc-sims",       "value"),   # 133
    ("sc-mc-window",     "value"),   # 134
    ("sc-mc-advanced",   "value"),   # 135
    # ── Heatmap palette ──────────────────────────────────────────────────
    ("hm-palette",       "value"),   # 136
```

- [ ] **Step 2: Add 12 entries to `_CHECKLIST_OPTIONS`**

Before the closing `}` at line 172, add:

```python
    # MC enable/advanced checklists (1 bit each)
    "dca-mc-enable":    ["yes"],
    "dca-mc-advanced":  ["yes"],
    "ret-mc-enable":    ["yes"],
    "ret-mc-advanced":  ["yes"],
    "hm-mc-enable":     ["yes"],
    "hm-mc-advanced":   ["yes"],
    "sc-mc-enable":     ["yes"],
    "sc-mc-advanced":   ["yes"],
    # MC regime checklists (5 bits each — int values 0-4)
    "dca-mc-regime":    [0, 1, 2, 3, 4],
    "ret-mc-regime":    [0, 1, 2, 3, 4],
    "hm-mc-regime":     [0, 1, 2, 3, 4],
    "sc-mc-regime":     [0, 1, 2, 3, 4],
```

- [ ] **Step 3: Syntax check**

```bash
btc_venv/bin/python3 -m py_compile btc_web/snapshot.py && echo "OK"
```

- [ ] **Step 4: Commit atomically**

```bash
git add btc_web/snapshot.py
git commit -m "feat: add 37 MC + hm-palette entries to _SNAPSHOT_CONTROLS + _CHECKLIST_OPTIONS"
```

---

### Task 3: Add hybrid encoding to `_encode_snapshot`

**Files:**
- Modify: `btc_web/snapshot.py:213` (inside `_encode_snapshot`, after `values.append(val)` loop, before `lots = ...`)

- [ ] **Step 1: Add MC-disable null-out logic**

Insert between line 213 (`values.append(val)` end of loop) and line 214 (`lots = state_dict.get("_lots")`):

```python
    # ── Hybrid MC encoding: null-out MC controls for disabled tabs ────────
    _mc_prefixes = {"dca": "dca-mc-", "ret": "ret-mc-", "hm": "hm-mc-", "sc": "sc-mc-"}
    for _pfx_tab, _pfx_mc in _mc_prefixes.items():
        enable_idx = next(i for i, (cid, _) in enumerate(_SNAPSHOT_CONTROLS)
                          if cid == f"{_pfx_mc}enable")
        mc_on = values[enable_idx] not in (None, [], 0)
        if not mc_on:
            for i, (cid, _) in enumerate(_SNAPSHOT_CONTROLS):
                if cid.startswith(_pfx_mc) and cid != f"{_pfx_mc}model-src":
                    values[i] = None
```

- [ ] **Step 2: Syntax check**

```bash
btc_venv/bin/python3 -m py_compile btc_web/snapshot.py && echo "OK"
```

- [ ] **Step 3: Run the MC tests**

```bash
PYTHONPATH=".:btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py -k "test_mc_controls_roundtrip or test_mc_hybrid or test_mc_regime_bitmask or test_hm_palette_roundtrip or test_old_link_pads" -v --timeout=60
```

Expected: 5 PASSED.

- [ ] **Step 4: Commit**

```bash
git add btc_web/snapshot.py
git commit -m "feat: hybrid MC encoding — null-out disabled tabs in _encode_snapshot"
```

---

### Task 4: Update `_TAB_CONTROLS` for single-tab scope

**Files:**
- Modify: `btc_web/callbacks/nav.py:56-84` (`_TAB_CONTROLS` definition)

- [ ] **Step 1: Add MC component IDs to each tab's control set**

After the existing `_TAB_CONTROLS` dict (after the palette-adding loop at ~line 87), add:

```python
# MC controls per tab (for single-tab share links)
_TAB_CONTROLS["dca"].update({
    "dca-mc-enable", "dca-mc-start-yr", "dca-mc-entry-q", "dca-mc-years",
    "dca-mc-bins", "dca-mc-regime", "dca-mc-sims", "dca-mc-window", "dca-mc-advanced",
})
_TAB_CONTROLS["retire"].update({
    "ret-mc-enable", "ret-mc-start-yr", "ret-mc-entry-q", "ret-mc-years",
    "ret-mc-bins", "ret-mc-regime", "ret-mc-sims", "ret-mc-window", "ret-mc-advanced",
})
_TAB_CONTROLS["heatmap"].update({
    "hm-mc-enable", "hm-mc-start-yr", "hm-mc-entry-q", "hm-mc-years",
    "hm-mc-bins", "hm-mc-regime", "hm-mc-sims", "hm-mc-window", "hm-mc-advanced",
    "hm-palette",
})
_TAB_CONTROLS["supercharge"].update({
    "sc-mc-enable", "sc-mc-start-yr", "sc-mc-entry-q", "sc-mc-years",
    "sc-mc-bins", "sc-mc-regime", "sc-mc-sims", "sc-mc-window", "sc-mc-advanced",
})
```

- [ ] **Step 2: Syntax check**

```bash
btc_venv/bin/python3 -m py_compile btc_web/callbacks/nav.py && echo "OK"
```

- [ ] **Step 3: Commit**

```bash
git add btc_web/callbacks/nav.py
git commit -m "feat: add MC + hm-palette IDs to _TAB_CONTROLS for single-tab scope"
```

---

### Task 5: Run full test suite + deploy

- [ ] **Step 1: Syntax check all modified files**

```bash
for f in btc_web/snapshot.py btc_web/callbacks/nav.py btc_web/test_web.py; do
    btc_venv/bin/python3 -m py_compile $f || echo "FAIL: $f"
done && echo "ALL OK"
```

- [ ] **Step 2: Run full test suite**

```bash
PYTHONPATH=".:btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py --timeout=120
```

Verify no new failures beyond the ~18 pre-existing ones. The new MC snapshot tests should all PASS.

- [ ] **Step 3: Manual verification**

Start dev server: `lsof -ti :8050 | xargs kill -9 2>/dev/null; DEV=1 nohup bash run_web.sh > /tmp/quantoshi_dev.log 2>&1 &`

Test on tab 3 (DCA):
1. Enable MC, set start_yr=2031, entry_q=10%, years=40 (free tier)
2. Click Share → Generate link → copy URL
3. Open URL in new browser tab
4. Verify: MC is enabled, settings restored, MC overlay renders from cache
5. Verify: disable MC → share → new tab → MC is off, controls at defaults

Test on tab 2 (Heatmap):
1. Select "Ocean" palette
2. Share → Generate link → open in new tab
3. Verify: palette dropdown shows "Ocean" (not "custom")

- [ ] **Step 4: Deploy**

```bash
git push origin master
ssh root@89.167.70.45 "cd /opt/quantoshi && git pull && systemctl restart quantoshi"
```

---

## Summary

| Task | What | Risk |
|------|------|------|
| 1 | Write 5 tests (roundtrip, hybrid, bitmask, palette, compat) | None — TDD setup |
| 2 | Add 37 `_SNAPSHOT_CONTROLS` + 12 `_CHECKLIST_OPTIONS` entries (ATOMIC) | Low — append-only |
| 3 | Hybrid encoding in `_encode_snapshot` | Low — 10 lines of logic |
| 4 | Update `_TAB_CONTROLS` with MC + palette IDs | None — additive |
| 5 | Full tests + deploy | None — verification |
