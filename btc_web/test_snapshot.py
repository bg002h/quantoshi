"""Snapshot encode/decode, bitmask, version compat."""
from conftest import (
    ExitStack,
    _ALL_QS,
    _CHECKLIST_OPTIONS,
    _CallbackCtx,
    _SNAPSHOT_CONTROLS,
    _TAB_CONTROLS,
    _TAB_TO_PATH,
    _decode_snapshot,
    _encode_snapshot,
    _list_to_mask,
    _mask_to_list,
    _nearest_quantile,
    _patch_ctx,
    _q3,
    base64,
    gzip,
    json,
    patch,
    pytest,
)


@pytest.mark.skipif(_list_to_mask is None, reason="app.py import failed")
class TestBitmaskEncoding:
    def test_empty_list(self):
        assert _list_to_mask([], ["a", "b", "c"]) == 0

    def test_single_item(self):
        assert _list_to_mask(["b"], ["a", "b", "c"]) == 0b010

    def test_all_items(self):
        assert _list_to_mask(["a", "b", "c"], ["a", "b", "c"]) == 0b111

    def test_roundtrip(self):
        opts = ["shade", "show_ols", "show_data", "show_today", "show_legend"]
        val = ["show_data", "show_today"]
        mask = _list_to_mask(val, opts)
        restored = _mask_to_list(mask, opts)
        assert set(restored) == set(val)

    def test_quantile_roundtrip(self):
        """Test bitmask with actual quantile float values."""
        if not _ALL_QS:
            pytest.skip("No quantiles loaded")
        opts = list(_ALL_QS)
        val = opts[:3]  # first 3 quantiles
        mask = _list_to_mask(val, opts)
        restored = _mask_to_list(mask, opts)
        assert restored == val


# ── Snapshot encode/decode ────────────────────────────────────────────────────


@pytest.mark.skipif(_encode_snapshot is None, reason="app.py import failed")
class TestSnapshotRoundtrip:
    def test_basic_roundtrip(self):
        state = {"bub-xscale:value": "Log", "bub-yscale:value": "Log"}
        encoded = _encode_snapshot(state)
        decoded = _decode_snapshot(encoded)
        assert decoded is not None
        assert decoded.get("bub-xscale:value") == "Log"
        assert decoded.get("bub-yscale:value") == "Log"

    def test_checklist_bitmask_roundtrip(self):
        state = {
            "bub-toggles:value": ["shade", "show_data"],
            "bub-qs:value": list(_ALL_QS)[:2] if _ALL_QS else [],
        }
        encoded = _encode_snapshot(state)
        decoded = _decode_snapshot(encoded)
        assert decoded is not None
        if "bub-toggles:value" in decoded:
            assert set(decoded["bub-toggles:value"]) == {"shade", "show_data"}

    def test_tab_filter(self):
        state = {
            "bub-xscale:value": "Log",
            "dca-amount:value": 500,
            "main-tabs:active_tab": "bubble",
        }
        tab_filter = {"bub-xscale", "bub-yscale"}
        encoded = _encode_snapshot(state, tab_filter=tab_filter)
        decoded = _decode_snapshot(encoded)
        assert decoded is not None
        assert decoded.get("bub-xscale:value") == "Log"
        # dca-amount should be filtered out
        assert "dca-amount:value" not in decoded
        # main-tabs always included
        assert decoded.get("main-tabs:active_tab") == "bubble"

    def test_lots_roundtrip(self):
        state = {
            "bub-xscale:value": "Log",
            "_lots": [{"btc": 1.0, "price": 69420, "date": "2024-01-15"}],
        }
        encoded = _encode_snapshot(state)
        decoded = _decode_snapshot(encoded)
        assert decoded is not None
        assert decoded["_lots"][0]["price"] == 69420

    def test_invalid_decode(self):
        assert _decode_snapshot("not-valid-base64!!!") is None

    def test_mc_controls_roundtrip(self):
        """MC controls survive encode -> decode roundtrip."""
        state = {f"{cid}:{prop}": None for cid, prop in _SNAPSHOT_CONTROLS}
        state["dca-mc-enable:value"] = ["yes"]
        state["dca-mc-start-yr:value"] = 2035
        state["dca-mc-entry-q:value"] = 30
        state["dca-mc-years:value"] = 20
        state["dca-mc-bins:value"] = 7
        state["dca-mc-regime:value"] = [0, 2, 4]
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

    def test_mc_hybrid_encoding_nulls_disabled_tabs(self):
        """MC controls encode as null when MC is not enabled on that tab."""
        state = {f"{cid}:{prop}": None for cid, prop in _SNAPSHOT_CONTROLS}
        state["dca-mc-enable:value"] = []
        state["dca-mc-start-yr:value"] = 2035
        state["dca-mc-bins:value"] = 7
        state["ret-mc-enable:value"] = ["yes"]
        state["ret-mc-start-yr:value"] = 2028
        encoded = _encode_snapshot(state)
        decoded = _decode_snapshot(encoded)
        assert decoded.get("dca-mc-start-yr:value") is None
        assert decoded.get("dca-mc-bins:value") is None
        assert decoded["ret-mc-enable:value"] == ["yes"]
        assert decoded["ret-mc-start-yr:value"] == 2028

    def test_mc_regime_bitmask_roundtrip(self):
        """MC regime checklist with int values survives bitmask encode/decode."""
        state = {f"{cid}:{prop}": None for cid, prop in _SNAPSHOT_CONTROLS}
        state["sc-mc-enable:value"] = ["yes"]
        state["sc-mc-regime:value"] = [0, 1, 3]
        encoded = _encode_snapshot(state)
        decoded = _decode_snapshot(encoded)
        assert sorted(decoded["sc-mc-regime:value"]) == [0, 1, 3]

    def test_hm_palette_roundtrip(self):
        """Heatmap palette name survives encode -> decode."""
        state = {f"{cid}:{prop}": None for cid, prop in _SNAPSHOT_CONTROLS}
        state["hm-palette:value"] = "bwo"
        encoded = _encode_snapshot(state)
        decoded = _decode_snapshot(encoded)
        assert decoded["hm-palette:value"] == "bwo"

    def test_old_link_pads_mc_to_none(self):
        """Old links with 100 entries decode correctly — MC defaults to None."""
        import gzip, base64
        assert len(_SNAPSHOT_CONTROLS) >= 137, "MC controls not yet added"
        old_values = [None] * 100
        old_values[0] = [0.5]
        payload = [old_values, None]
        encoded = base64.urlsafe_b64encode(
            gzip.compress(json.dumps(payload, separators=(',', ':')).encode())).decode()
        decoded = _decode_snapshot(encoded)
        assert decoded.get("bub-qs:value") == [0.5]
        assert decoded.get("dca-mc-enable:value") is None
        assert decoded.get("hm-palette:value") is None


# ── _nearest_quantile ─────────────────────────────────────────────────────────


class TestSnapshotControlsCompleteness:
    """Verify snapshot controls list is self-consistent."""

    @pytest.mark.skipif(_SNAPSHOT_CONTROLS is None, reason="app.py import failed")
    def test_no_duplicate_controls(self):
        ids = [cid for cid, _ in _SNAPSHOT_CONTROLS]
        assert len(ids) == len(set(ids)), f"Duplicate IDs: {[x for x in ids if ids.count(x) > 1]}"

    @pytest.mark.skipif(_CHECKLIST_OPTIONS is None, reason="app.py import failed")
    def test_checklist_options_cover_snapshot(self):
        """Every checklist in snapshot should have options defined."""
        snapshot_ids = {cid for cid, _ in _SNAPSHOT_CONTROLS}
        for cid in _CHECKLIST_OPTIONS:
            assert cid in snapshot_ids, f"{cid} in _CHECKLIST_OPTIONS but not in _SNAPSHOT_CONTROLS"


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5: Callback smoke tests (Phase E)
# ═══════════════════════════════════════════════════════════════════════════════

class _CallbackCtx:
    """Minimal mock for dash.ctx (dash._callback_context)."""
    def __init__(self, triggered_id=None):
        self.triggered_id = triggered_id


def _patch_ctx(triggered_id=None):
    """Context manager that patches dash.ctx across all callback submodules."""
    ctx_obj = _CallbackCtx(triggered_id)
    # After callbacks.py was split into callbacks/, each submodule imports ctx
    # from dash directly.  Patch every submodule that uses ctx.
    from contextlib import ExitStack
    _targets = [
        "callbacks", "callbacks.charts", "callbacks.lots",
        "callbacks.mc_helpers", "callbacks.mc_payment",
        "callbacks.snapshot_cb",
    ]
    stack = ExitStack()
    for t in _targets:
        stack.enter_context(patch.multiple(t, ctx=ctx_obj))
    return stack



@pytest.mark.skipif(_q3 is None, reason="app.py import failed")
class TestTabControlsMappings:
    """Verify _TAB_CONTROLS and _TAB_TO_PATH consistency."""

    def test_all_tabs_have_controls(self):
        for tab in ["bubble", "heatmap", "dca", "retire", "supercharge"]:
            assert tab in _TAB_CONTROLS
            assert len(_TAB_CONTROLS[tab]) > 0

    def test_tab_to_path_complete(self):
        for tab in ["bubble", "heatmap", "dca", "retire", "supercharge", "stack", "faq"]:
            assert tab in _TAB_TO_PATH

    def test_snapshot_controls_covered(self):
        """Every control ID (except main-tabs) should belong to some tab."""
        all_tab_ids = set()
        for ids in _TAB_CONTROLS.values():
            all_tab_ids |= ids
        for cid, _ in _SNAPSHOT_CONTROLS:
            if cid == "main-tabs":
                continue
            assert cid in all_tab_ids, f"{cid} not in any _TAB_CONTROLS set"


# ═══════════════════════════════════════════════════════════════════════════════
# Section 6: Snapshot edge cases (Phase E)
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(_q3 is None, reason="app.py import failed")
class TestSnapshotSingleTabScope:
    """Test single-tab scope filtering in snapshots."""

    def test_tab_filter_encodes_only_matching(self):
        """When tab_filter is set, non-matching controls become null."""
        state = {}
        for cid, prop in _SNAPSHOT_CONTROLS:
            if cid in _TAB_CONTROLS.get("bubble", set()):
                state[f"{cid}:{prop}"] = "test_val"
            elif cid in _TAB_CONTROLS.get("dca", set()):
                state[f"{cid}:{prop}"] = "dca_val"
            elif cid == "main-tabs":
                state[f"{cid}:{prop}"] = "bubble"
        tab_filter = _TAB_CONTROLS["bubble"]
        encoded = _encode_snapshot(state, tab_filter=tab_filter)
        decoded = _decode_snapshot(encoded)
        assert decoded is not None
        # Bubble controls should be present
        assert decoded.get("bub-xscale:value") == "test_val"
        # DCA controls should NOT be present (filtered out)
        assert "dca-amount:value" not in decoded
        # main-tabs always present
        assert decoded.get("main-tabs:active_tab") == "bubble"

    def test_each_tab_filter_roundtrips(self):
        """Each tab's filter should produce a decodable snapshot."""
        for tab, ids in _TAB_CONTROLS.items():
            state = {"main-tabs:active_tab": tab}
            for cid, prop in _SNAPSHOT_CONTROLS:
                if cid in ids:
                    state[f"{cid}:{prop}"] = "val"
            encoded = _encode_snapshot(state, tab_filter=ids)
            decoded = _decode_snapshot(encoded)
            assert decoded is not None, f"Failed to decode {tab} tab snapshot"
            assert decoded.get("main-tabs:active_tab") == tab

    def test_single_tab_shorter_than_all(self):
        """Single-tab snapshot should produce shorter encoded string."""
        state = {}
        for cid, prop in _SNAPSHOT_CONTROLS:
            state[f"{cid}:{prop}"] = "x"
        encoded_all = _encode_snapshot(state)
        encoded_one = _encode_snapshot(state, tab_filter=_TAB_CONTROLS["retire"])
        assert len(encoded_one) < len(encoded_all)



@pytest.mark.skipif(_q3 is None, reason="app.py import failed")
class TestBitmaskEdgeCases:
    """Test bitmask encoding with edge-case states."""

    def test_all_on(self):
        """All options selected → all bits set."""
        for cid, opts in _CHECKLIST_OPTIONS.items():
            mask = _list_to_mask(opts, opts)
            expected = (1 << len(opts)) - 1
            assert mask == expected, f"{cid}: expected {expected}, got {mask}"
            # Roundtrip
            assert _mask_to_list(mask, opts) == opts

    def test_all_off(self):
        """No options selected → mask is 0."""
        for cid, opts in _CHECKLIST_OPTIONS.items():
            assert _list_to_mask([], opts) == 0
            assert _mask_to_list(0, opts) == []

    def test_single_bit_each(self):
        """Each individual option should set exactly one bit."""
        for cid, opts in _CHECKLIST_OPTIONS.items():
            for i, opt in enumerate(opts):
                mask = _list_to_mask([opt], opts)
                assert mask == (1 << i), f"{cid}[{i}]={opt}: expected {1<<i}, got {mask}"
                assert _mask_to_list(mask, opts) == [opt]

    def test_quantile_all_on_roundtrip(self):
        """All bands selected → roundtrip through encode/decode."""
        all_bands = ["inner", "outer", "median"]
        state = {"bub-qs:value": all_bands, "main-tabs:active_tab": "bubble"}
        encoded = _encode_snapshot(state)
        decoded = _decode_snapshot(encoded)
        assert decoded is not None
        restored = decoded.get("bub-qs:value", [])
        assert set(restored) == set(all_bands)

    def test_empty_checklist_roundtrip(self):
        """Empty checklist → 0 bitmask → empty list on decode."""
        state = {"bub-toggles:value": [], "main-tabs:active_tab": "bubble"}
        encoded = _encode_snapshot(state)
        decoded = _decode_snapshot(encoded)
        assert decoded is not None
        # Empty list encodes as 0, which decodes to empty list
        # But 0 might be stored as 0 in JSON; decoder should handle it
        toggles = decoded.get("bub-toggles:value", None)
        # Either not present (null → skipped) or empty list
        assert toggles is None or toggles == []

    def test_high_bit_quantile(self):
        """Last quantile only → highest bit set."""
        opts = _CHECKLIST_OPTIONS["bub-qs"]
        last = opts[-1]
        mask = _list_to_mask([last], opts)
        assert mask == (1 << (len(opts) - 1))
        assert _mask_to_list(mask, opts) == [last]

    def test_mask_to_list_ignores_extra_bits(self):
        """Bits beyond opts length should be ignored."""
        opts = ["a", "b", "c"]
        mask = 0b11111  # 5 bits, but only 3 opts
        result = _mask_to_list(mask, opts)
        assert result == ["a", "b", "c"]



@pytest.mark.skipif(_q3 is None, reason="app.py import failed")
class TestSnapshotLotsSpecialChars:
    """Test lots with special characters survive snapshot roundtrip."""

    def test_unicode_notes(self):
        lots = [{"date": "2024-01-01", "btc": 0.5, "price": 42000,
                 "pct_q": 0.45, "notes": "🚀 Bitcoin — \"to the moon\" ✨"}]
        state = {"_lots": lots, "main-tabs:active_tab": "stack"}
        encoded = _encode_snapshot(state)
        decoded = _decode_snapshot(encoded)
        assert decoded is not None
        assert decoded["_lots"][0]["notes"] == lots[0]["notes"]

    def test_html_in_notes(self):
        lots = [{"date": "2024-01-01", "btc": 1.0, "price": 50000,
                 "pct_q": 0.5, "notes": '<b>bold</b> & "quotes" <script>x</script>'}]
        state = {"_lots": lots, "main-tabs:active_tab": "stack"}
        encoded = _encode_snapshot(state)
        decoded = _decode_snapshot(encoded)
        assert decoded["_lots"][0]["notes"] == lots[0]["notes"]

    def test_empty_notes(self):
        lots = [{"date": "2024-01-01", "btc": 1.0, "price": 50000,
                 "pct_q": 0.5, "notes": ""}]
        state = {"_lots": lots, "main-tabs:active_tab": "stack"}
        encoded = _encode_snapshot(state)
        decoded = _decode_snapshot(encoded)
        assert decoded["_lots"][0]["notes"] == ""

    def test_many_lots_roundtrip(self):
        lots = [{"date": f"2024-{i:02d}-01", "btc": 0.01 * i, "price": 40000 + i * 1000,
                 "pct_q": 0.3 + i * 0.02, "notes": f"lot #{i}"}
                for i in range(1, 13)]
        state = {"_lots": lots, "main-tabs:active_tab": "stack"}
        encoded = _encode_snapshot(state)
        decoded = _decode_snapshot(encoded)
        assert len(decoded["_lots"]) == 12

    def test_no_lots(self):
        state = {"main-tabs:active_tab": "bubble"}
        encoded = _encode_snapshot(state)
        decoded = _decode_snapshot(encoded)
        assert "_lots" not in decoded



@pytest.mark.skipif(_q3 is None, reason="app.py import failed")
class TestSnapshotV1Compat:
    """Legacy v1 snapshot format backward compatibility."""

    def test_v1_decode(self):
        """v1 format is a plain JSON dict, gzip+b64 encoded."""
        state = {"bub-xscale:value": "log", "main-tabs:active_tab": "bubble"}
        j = json.dumps(state, separators=(',', ':'))
        from snapshot import _decode_snapshot_v1
        encoded = base64.urlsafe_b64encode(gzip.compress(j.encode())).decode()
        decoded = _decode_snapshot_v1(encoded)
        assert decoded == state

    def test_v1_invalid(self):
        from snapshot import _decode_snapshot_v1
        assert _decode_snapshot_v1("not-valid-base64!!!") is None



@pytest.mark.skipif(_q3 is None, reason="app.py import failed")
class TestSnapshotVersionCompat:
    """Test that snapshots with different control counts decode gracefully."""

    def test_shorter_snapshot_pads(self):
        """A snapshot with fewer controls than current should pad with None."""
        from snapshot import _SNAPSHOT_CONTROLS, _decode_snapshot
        # Create a truncated snapshot (only first 10 controls)
        short_values = [None] * 10
        payload = [short_values, None]
        j = json.dumps(payload, separators=(',', ':'))
        encoded = base64.urlsafe_b64encode(gzip.compress(j.encode())).decode()
        state = _decode_snapshot(encoded)
        assert state is not None  # should not fail
        # First 10 are None → not in state; rest also None → not in state
        # Key point: no crash

    def test_longer_snapshot_truncates(self):
        """A snapshot with more controls than current should truncate safely."""
        from snapshot import _SNAPSHOT_CONTROLS, _decode_snapshot
        n = len(_SNAPSHOT_CONTROLS)
        long_values = [None] * (n + 20)  # 20 extra
        payload = [long_values, None]
        j = json.dumps(payload, separators=(',', ':'))
        encoded = base64.urlsafe_b64encode(gzip.compress(j.encode())).decode()
        state = _decode_snapshot(encoded)
        assert state is not None


# ═══════════════════════════════════════════════════════════════════════════════
# Section: Price cache tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestDecompSnapshot:
    def test_decomp_fields_in_snapshot_controls(self):
        from snapshot import _SNAPSHOT_CONTROLS
        cids = {cid for cid, _ in _SNAPSHOT_CONTROLS}
        assert "bub-decomp-model" in cids
        assert "bub-decomp-components" in cids

    def test_decomp_fields_in_bubble_tab_controls(self):
        from callbacks.routing import _TAB_CONTROLS
        assert "bub-decomp-model" in _TAB_CONTROLS["bubble"]
        assert "bub-decomp-components" in _TAB_CONTROLS["bubble"]

    def test_decomp_not_bitmask_encoded(self):
        from snapshot import _CHECKLIST_OPTIONS
        assert "bub-decomp-components" not in _CHECKLIST_OPTIONS

    def test_decomp_roundtrip_encode_decode(self):
        from snapshot import _encode_snapshot, _decode_snapshot, _SNAPSHOT_CONTROLS
        state = {f"{cid}:{prop}": None for cid, prop in _SNAPSHOT_CONTROLS}
        state["bub-decomp-model:value"] = "hybppl_dd"
        state["bub-decomp-components:value"] = ["A (constant)", "B\u00b7log\u2081\u2080(t)", "__sum__"]
        encoded = _encode_snapshot(state)
        decoded = _decode_snapshot(encoded)
        assert decoded["bub-decomp-model:value"] == "hybppl_dd"
        assert decoded["bub-decomp-components:value"] == [
            "A (constant)", "B\u00b7log\u2081\u2080(t)", "__sum__"]


# ═══════════════════════════════════════════════════════════════════════════════
# Section: MC model interface verification
# ═══════════════════════════════════════════════════════════════════════════════



class TestSnapshotModelShow:
    """Snapshot roundtrip with 'pl' and 's2f' in model-show checklists."""

    def test_roundtrip_pl_in_model_show(self):
        from snapshot import _encode_snapshot, _decode_snapshot, _SNAPSHOT_CONTROLS
        state = {}
        for cid, prop in _SNAPSHOT_CONTROLS:
            state[f"{cid}:{prop}"] = None
        state["dca-model-show:value"] = ["qr", "pl"]
        encoded = _encode_snapshot(state)
        decoded = _decode_snapshot(encoded)
        assert "pl" in decoded["dca-model-show:value"]
        assert "qr" in decoded["dca-model-show:value"]

    def test_roundtrip_s2f_in_model_show(self):
        from snapshot import _encode_snapshot, _decode_snapshot, _SNAPSHOT_CONTROLS
        state = {}
        for cid, prop in _SNAPSHOT_CONTROLS:
            state[f"{cid}:{prop}"] = None
        state["sc-model-show:value"] = ["qr", "mc", "s2f"]
        encoded = _encode_snapshot(state)
        decoded = _decode_snapshot(encoded)
        assert "s2f" in decoded["sc-model-show:value"]

    def test_old_bitmask_without_pl_decodes_without_pl(self):
        """Old snapshots with only 2-bit model-show (qr+mc) should not have pl/s2f."""
        from snapshot import _encode_snapshot, _decode_snapshot, _SNAPSHOT_CONTROLS
        state = {}
        for cid, prop in _SNAPSHOT_CONTROLS:
            state[f"{cid}:{prop}"] = None
        # Simulate old link: only qr and mc selected (bits 0 and 1)
        state["ret-model-show:value"] = ["qr", "mc"]
        encoded = _encode_snapshot(state)
        decoded = _decode_snapshot(encoded)
        assert "pl" not in decoded["ret-model-show:value"]
        assert "s2f" not in decoded["ret-model-show:value"]




class TestBubbleLazyRelay:
    """Bubble snapshot controls must route through lazy relay store."""

    def test_bubble_controls_not_eager_outputs(self):
        """bub-*, scan-*, cta-* must NOT be direct Outputs of apply_snapshot."""
        from callbacks import snapshot_cb
        for cid, prop in snapshot_cb._EAGER_CONTROLS:
            assert not cid.startswith(("bub-", "scan-", "cta-")), (
                f"{cid} should be routed through snapshot-apply-bubble, "
                f"not written eagerly by apply_snapshot")

    def test_bubble_lazy_controls_populated(self):
        """Relay must include the bubble tab controls."""
        from callbacks import snapshot_cb
        ids = {cid for cid, _ in snapshot_cb._BUBBLE_LAZY_CONTROLS}
        assert "bub-qs" in ids
        assert any(c.startswith("scan-") for c in ids)
        assert any(c.startswith("cta-") for c in ids)

    def test_bubble_controls_still_in_snapshot_controls(self):
        """Bubble controls must still be in _SNAPSHOT_CONTROLS for encode/decode."""
        from snapshot import _SNAPSHOT_CONTROLS
        ids = {cid for cid, _ in _SNAPSHOT_CONTROLS}
        assert "bub-qs" in ids
