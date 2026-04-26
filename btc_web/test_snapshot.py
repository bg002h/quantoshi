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




class TestPostRefactorArchitecture:
    """Tests locking in the new apply_globals + apply_tab_{tab} architecture.
    Written per spec 2026-04-24-drop-all-tabs-snapshot-design.md."""

    def test_no_double_write_partition(self):
        """Union of globals + per-tab equals _SNAPSHOT_CONTROLS;
        intersection is empty."""
        from callbacks.snapshot_cb import _GLOBAL_CONTROLS, _PER_TAB_CONTROLS
        from snapshot import _SNAPSHOT_CONTROLS
        all_tab_cids = set()
        for tab_id, cids_props in _PER_TAB_CONTROLS.items():
            for cid, _prop in cids_props:
                all_tab_cids.add(cid)
        global_cids = {cid for cid, _ in _GLOBAL_CONTROLS}
        assert global_cids.isdisjoint(all_tab_cids), (
            f"Control(s) in both globals and per-tab: "
            f"{global_cids & all_tab_cids}")
        all_cids = {cid for cid, _ in _SNAPSHOT_CONTROLS}
        assert global_cids | all_tab_cids == all_cids, (
            f"Partition mismatch. "
            f"Missing: {all_cids - (global_cids | all_tab_cids)}. "
            f"Extra: {(global_cids | all_tab_cids) - all_cids}")

    def test_apply_globals_writes_only_global_cids(self):
        """apply_globals output count equals len(_GLOBAL_CONTROLS) + 1 (for snapshot-lots)."""
        from callbacks.snapshot_cb import apply_globals, _GLOBAL_CONTROLS
        state = {"main-tabs:active_tab": "heatmap",
                 "palette-store:data": "default",
                 "bub-xscale:value": "Lin"}
        result = apply_globals(state)
        assert len(result) == len(_GLOBAL_CONTROLS) + 1, (
            f"Expected {len(_GLOBAL_CONTROLS) + 1} outputs, got {len(result)}")

    def test_apply_tab_is_noop_when_state_none(self):
        """apply_tab_bubble returns all no_update when state is None."""
        from callbacks.snapshot_cb import apply_tab_bubble
        from dash import no_update
        result = apply_tab_bubble(None, None, None, {})
        assert all(x is no_update for x in result)

    def test_apply_tab_partial_restore_on_legacy_payload(self):
        """Legacy payload with both bubble + heatmap keys — each apply_tab
        callback picks up its own tab's keys when invoked."""
        from callbacks.snapshot_cb import apply_tab_bubble, apply_tab_heatmap
        from dash import no_update
        state = {
            "bub-xscale:value": "Lin",
            "hm-mode:value": "segmented",
            "bub-qs:value": [0.5],
        }
        bub_result = apply_tab_bubble(1, state, "h1", {})
        assert any(v == "Lin" for v in bub_result if v is not no_update), (
            "apply_tab_bubble must write bub-xscale=Lin")
        hm_result = apply_tab_heatmap(1, state, "h1", {})
        assert any(v == "segmented" for v in hm_result if v is not no_update), (
            "apply_tab_heatmap must write hm-mode=segmented")

    def test_no_orphan_relay_stores_in_layout(self):
        """Layout must not contain any snapshot-apply-{tab} Store ids."""
        import layout
        import json
        rendered = layout._serve_layout() if hasattr(layout, "_serve_layout") else None
        serialised = json.dumps(rendered, default=str) if rendered else ""
        banned = ["snapshot-apply-bubble", "snapshot-apply-heatmap",
                  "snapshot-apply-dca", "snapshot-apply-retire",
                  "snapshot-apply-supercharge", "snapshot-apply-citadel",
                  "snapshot-apply-leverage"]
        for b in banned:
            assert b not in serialised, (
                f"Relay store {b} still in layout — not removed by refactor")

    def test_manage_snapshot_signature_has_no_share_scope(self):
        """manage_snapshot should no longer accept share_scope."""
        import inspect
        from callbacks.snapshot_cb import manage_snapshot
        sig = inspect.signature(manage_snapshot)
        param_names = list(sig.parameters.keys())
        assert "share_scope" not in param_names, (
            f"share_scope still a parameter: {param_names}")


class TestSnapshotPendingGate:
    """Tests for the snapshot-pending gate that reduces chart redraws.
    Per spec docs/superpowers/specs/2026-04-24-single-redraw-per-snapshot-design.md."""

    def test_snapshot_pending_in_layout(self):
        """The snapshot-pending Store must exist in the rendered layout."""
        import layout
        import json
        rendered = layout._serve_layout() if hasattr(layout, "_serve_layout") else None
        serialised = json.dumps(rendered, default=str) if rendered else ""
        assert "snapshot-pending" in serialised, (
            "snapshot-pending Store missing from layout")

    def test_restore_from_url_uses_initial_duplicate(self):
        """restore_from_url must use prevent_initial_call='initial_duplicate'
        because it now has an allow_duplicate=True Output.

        Source-level assertion: grep the decorator on the Python source
        file. Dash's internal callback_map doesn't expose this reliably."""
        import os, pathlib
        here = pathlib.Path(os.path.dirname(__file__))
        src = (here / "callbacks" / "snapshot_cb.py").read_text()
        idx = src.find("def restore_from_url(")
        assert idx > 0, "restore_from_url not found in snapshot_cb.py"
        decorator = src[:idx].rsplit("@callback(", 1)[-1]
        assert ("prevent_initial_call='initial_duplicate'" in decorator or
                'prevent_initial_call="initial_duplicate"' in decorator), (
            "restore_from_url must use prevent_initial_call='initial_duplicate'; "
            f"last 300 chars of decorator: {decorator[-300:]}")

    def test_apply_tab_outputs_include_snapshot_pending(self):
        """Each of the 7 apply_tab_{tab} callbacks must release the gate
        by returning False as their last output when called with populated state.

        Behavioral assertion: the callback_map internals are opaque for
        multi-output allow_duplicate callbacks, so test behavior directly."""
        from callbacks.snapshot_cb import (
            apply_tab_bubble, apply_tab_heatmap, apply_tab_dca,
            apply_tab_retire, apply_tab_supercharge, apply_tab_citadel,
            apply_tab_leverage,
        )
        state = {"bub-xscale:value": "Lin", "hm-mode:value": "dca",
                 "dca-amount:value": 100}
        for name, fn in [
            ("bubble", apply_tab_bubble), ("heatmap", apply_tab_heatmap),
            ("dca", apply_tab_dca), ("retire", apply_tab_retire),
            ("supercharge", apply_tab_supercharge),
            ("citadel", apply_tab_citadel), ("leverage", apply_tab_leverage),
        ]:
            result = fn(1, state, "h1", {})
            assert result[-2] is False, (
                f"apply_tab_{name}: gate output (second-to-last) must be "
                f"False; got {result[-2]!r}")

    def test_apply_tab_releases_gate_on_populated_state(self):
        """apply_tab_bubble with populated state returns False (release) as gate output."""
        from callbacks.snapshot_cb import apply_tab_bubble
        state = {"bub-xscale:value": "Lin", "bub-qs:value": [0.5]}
        result = apply_tab_bubble(1, state, "h1", {})
        assert result[-2] is False, (
            f"Gate output must be False to release; got {result[-2]!r}")

    def test_apply_tab_does_not_clear_gate_when_state_none(self):
        """apply_tab_bubble with state=None returns no_update for gate output
        (NOT False) — so non-restore first-render bumps don't accidentally
        clear the gate."""
        from callbacks.snapshot_cb import apply_tab_bubble
        from dash import no_update
        result = apply_tab_bubble(None, None, None, {})
        assert all(x is no_update for x in result), (
            "All outputs must be no_update when state is None")

    def test_snapshot_pending_writers_have_allow_duplicate(self):
        """Every callback that outputs snapshot-pending.data must use
        allow_duplicate=True (the @ marker in the callback_map key)."""
        import _app_ctx
        app = _app_ctx.app
        for cb_key in app.callback_map:
            parts = cb_key.split("...")
            for part in parts:
                base = part.split("@")[0]
                if base == "snapshot-pending.data":
                    assert "@" in part, (
                        f"Callback {cb_key} outputs snapshot-pending without "
                        f"allow_duplicate (part: {part!r})")

    def test_apply_globals_does_not_output_snapshot_pending(self):
        """Guard: apply_globals must NOT output snapshot-pending. If a future
        editor moves the release into apply_globals, it clears the gate before
        apply_tab_{active} runs — breaking the single-redraw invariant."""
        import _app_ctx
        app = _app_ctx.app
        for cb_key in app.callback_map:
            parts = cb_key.split("...")
            clean = [p.split("@")[0] for p in parts]
            if ("main-tabs.active_tab" in clean and "palette-store.data" in clean):
                assert "snapshot-pending.data" not in clean, (
                    "apply_globals must NOT output snapshot-pending "
                    "(would break single-redraw invariant)")

    def test_safety_timer_at_least_3000ms(self):
        """Clientside safety timer must wait at least 4000 ms before
        unconditionally clearing the gate. Bumped from 3000 to 4000
        (2026-04-24) to prevent premature restore-progress modal dismiss
        on cold cache."""
        import os, pathlib, re
        here = pathlib.Path(os.path.dirname(__file__))
        src = (here / "callbacks" / "snapshot_cb.py").read_text()
        idx = src.find("Safety-timer")
        assert idx > 0, "Safety-timer block comment not found in snapshot_cb.py"
        block = src[idx:idx + 2000]
        m = re.search(r"}\s*,\s*(\d+)\s*\)\s*;", block)
        assert m, f"setTimeout delay literal not found in safety-timer block"
        duration = int(m.group(1))
        assert duration >= 4000, (
            f"Safety timer must be >= 4000ms; got {duration}")

    def test_restore_progress_modal_in_layout(self):
        """Modal id must exist in rendered layout."""
        import layout
        import json
        rendered = layout._serve_layout() if hasattr(layout, "_serve_layout") else None
        serialised = json.dumps(rendered, default=str) if rendered else ""
        assert "restore-progress-modal" in serialised, (
            "restore-progress-modal missing from layout")

    def test_restore_progress_modal_backdrop_static(self):
        """Modal must be blocking: backdrop='static', keyboard=False."""
        import os, pathlib
        here = pathlib.Path(os.path.dirname(__file__))
        src = (here / "layout" / "__init__.py").read_text()
        idx = src.find('id="restore-progress-modal"')
        assert idx > 0, "restore-progress-modal not declared in layout"
        block = src[max(0, idx-400):idx+200]
        assert 'backdrop="static"' in block, (
            "restore-progress-modal must use backdrop='static'")
        assert "keyboard=False" in block, (
            "restore-progress-modal must use keyboard=False")

    def test_open_callback_initial_duplicate(self):
        """Open-on-hash clientside must use prevent_initial_call='initial_duplicate'."""
        import os, pathlib
        here = pathlib.Path(os.path.dirname(__file__))
        src = (here / "callbacks" / "snapshot_cb.py").read_text()
        idx = src.find('"restore-progress-modal", "is_open"')
        assert idx > 0, "restore-progress-modal Output not found in snapshot_cb.py"
        opener = src[max(0, idx-3000):idx+500]
        assert 'prevent_initial_call=\'initial_duplicate\'' in opener or \
               'prevent_initial_call="initial_duplicate"' in opener, (
            "open callback must use prevent_initial_call='initial_duplicate'")

    def test_open_callback_debounce_150ms(self):
        """Open callback must debounce 150 ms."""
        import os, pathlib
        here = pathlib.Path(os.path.dirname(__file__))
        src = (here / "callbacks" / "snapshot_cb.py").read_text()
        idx = src.find('__restoreOpenTimer')
        assert idx > 0, "__restoreOpenTimer not found"
        block = src[idx:idx + 500]
        assert '150' in block, "150ms debounce not found in open callback"

    def test_fallback_timer_at_least_5000ms(self):
        """Open callback must arm a hard fallback of at least 5000 ms."""
        import os, pathlib, re
        here = pathlib.Path(os.path.dirname(__file__))
        src = (here / "callbacks" / "snapshot_cb.py").read_text()
        idx = src.find('__restoreFallback')
        assert idx > 0, "__restoreFallback not found"
        block = src[idx:idx + 1500]
        m = re.search(r"}\s*,\s*(\d+)\s*\)\s*;", block)
        assert m and int(m.group(1)) >= 5000, (
            f"hard fallback must be >= 5000 ms; got {m and m.group(1)}")

    def test_close_callback_min_display_500ms(self):
        """Close callback must enforce 500 ms min-display."""
        import os, pathlib
        here = pathlib.Path(os.path.dirname(__file__))
        src = (here / "callbacks" / "snapshot_cb.py").read_text()
        # Locate the close callback by its anchor comment.
        idx = src.find('Restore-progress modal close')
        assert idx > 0, "close-callback anchor not found"
        block = src[idx:idx + 4000]
        assert 'Math.max' in block and '500' in block, (
            "500ms min-display (Math.max) not found in close callback")


class TestSnapshotDefaultsRegistry:
    def test_current_fingerprint_in_registry(self):
        """If this fails, run: btc_venv/bin/python3 tools/update_defaults_registry.py"""
        import json, os, pathlib
        here = pathlib.Path(os.path.dirname(__file__))
        from snapshot_defaults import _compute_snapshot_defaults_fingerprint
        fp = _compute_snapshot_defaults_fingerprint()
        with open(here / "snapshot_defaults_registry.json") as f:
            registry = json.load(f)
        fps = [e["fp"] for e in registry]
        assert fp in fps, (
            f"current fingerprint {fp} not in registry {fps}; run "
            "tools/update_defaults_registry.py and commit the result")

    def test_registry_capped_at_20(self):
        import json, os, pathlib
        here = pathlib.Path(os.path.dirname(__file__))
        with open(here / "snapshot_defaults_registry.json") as f:
            registry = json.load(f)
        assert len(registry) <= 20, f"registry has {len(registry)} entries (cap=20)"


class TestSnapshotDefaultsConsistency:
    """Regression tests for two bug classes discovered during Phase 2:

    1. snapshot_defaults.py declarations can drift from actual widget
       defaults (ret-mc-enable was ['yes'] in snapshot_defaults but [] in
       the placeholder widget; later fixed to '['yes']'/'['yes']' alignment).
    2. _CHECKLIST_OPTIONS must include every value that ever appears in a
       widget value or snapshot_defaults — missing values are silently
       bitmask-stripped to 0 (the 'bub' bug for dca/ret/sc/hm-model-show).

    Both bugs were fixed in commits 9bf85c0 + cda79e0. These tests prevent
    regressions.

    See memory/restore_callback_architecture.md "Lessons learned (Phase 2
    retrospective)" section for context.
    """

    @staticmethod
    def _walk_layout_for_props(component, props_to_capture=("value", "data", "date")):
        """Recursively walk a Dash component tree, returning {id: {prop: val}}.

        Captures only the props in props_to_capture (default: value, data,
        date — the props snapshot_defaults ever uses).
        """
        found = {}

        def _walk(comp):
            if comp is None:
                return
            cid = getattr(comp, "id", None)
            if isinstance(cid, str):
                p = {}
                for attr in props_to_capture:
                    if hasattr(comp, attr):
                        try:
                            v = getattr(comp, attr)
                            if not callable(v):
                                p[attr] = v
                        except Exception:
                            pass
                found[cid] = p
            children = getattr(comp, "children", None)
            if isinstance(children, list):
                for ch in children:
                    _walk(ch)
            elif children is not None:
                _walk(children)

        _walk(component)
        return found

    @classmethod
    def _gather_all_widget_defaults(cls):
        """Build the full layout (top-level + every per-tab content) and
        return {id: {prop: default_value}} for every component.

        Lazy-mounted tab content is materialized via _build_tab_content
        so we can see the real widget defaults (not the "Loading..."
        placeholder).
        """
        from layout import _build_layout
        from callbacks.routing import _build_tab_content

        all_props = {}
        all_props.update(cls._walk_layout_for_props(_build_layout("bubble")))
        for tab_id in ("bubble", "heatmap", "dca", "retire", "supercharge",
                       "citadel", "leverage", "stack", "model_info", "faq"):
            content = _build_tab_content(tab_id)
            if content is not None:
                all_props.update(cls._walk_layout_for_props(content))
        return all_props

    def test_defaults_match_widget_defaults(self):
        """Each entry in SNAPSHOT_DEFAULTS must match the actual widget's
        default value.

        Drift between snapshot_defaults and widgets creates "unreachable"
        code paths and silent encoding bugs. See ret-mc-enable saga
        (commit 9bf85c0, partially reverted).

        Some prop names are aliased: SNAPSHOT_DEFAULTS uses 'value' for
        most controls, 'data' for Stores, 'date' for DatePickerSingle.

        Entries in ALWAYS_ENCODE are SKIPPED — those are dynamic-default
        fields (today's date, live BTC price) where snapshot_defaults
        intentionally declares None as a placeholder; the encoder always
        emits them so the link author's value is captured at share time.
        """
        from snapshot_defaults import SNAPSHOT_DEFAULTS, ALWAYS_ENCODE

        all_widgets = self._gather_all_widget_defaults()
        mismatches = []
        not_found = []
        for key, snap_default in SNAPSHOT_DEFAULTS.items():
            if key in ALWAYS_ENCODE:
                continue  # dynamic default — snapshot_default is intentionally None
            cid, prop = key.split(":", 1)
            if cid not in all_widgets:
                not_found.append(key)
                continue
            widget_props = all_widgets[cid]
            if prop not in widget_props:
                # Component exists but doesn't expose this prop —
                # skip (some controls have None defaults).
                continue
            widget_default = widget_props[prop]
            if widget_default != snap_default:
                mismatches.append(
                    f"{key}: snapshot_defaults={snap_default!r} "
                    f"widget_default={widget_default!r}"
                )

        # Components that don't show up in the layout are usually defunct
        # placeholders or runtime-only stores; not a hard failure.
        # (Print for visibility but don't fail.)
        if not_found:
            print(f"\n[info] {len(not_found)} snapshot_defaults entries "
                  f"have no matching widget in the layout (likely defunct "
                  f"placeholders or runtime stores). First 5: "
                  f"{not_found[:5]}")

        assert mismatches == [], (
            f"\n{len(mismatches)} snapshot_defaults vs widget mismatches. "
            f"Either fix the snapshot_defaults declaration or fix the "
            f"widget default to match. See memory/"
            f"restore_callback_architecture.md 'Lessons learned'. "
            f"Mismatches:\n  " + "\n  ".join(mismatches)
        )

    def test_restore_builder_explicit_defaults_match_snapshot_defaults(self):
        """AST-walk restore_builder.py for `_v(state, cid, [prop,] default=X)`
        calls; assert X equals SNAPSHOT_DEFAULTS[f"{cid}:{prop}"] (or that
        the call is in INTENTIONAL_OVERRIDES).

        Catches the bug class: a builder hardcodes default=Y but
        SNAPSHOT_DEFAULTS says X. New q4 share links get X (decoder fills
        missing keys); legacy q1/q2/q3 links get Y. Same builder produces
        different figures for the same logical state — silently. Phase 2
        had several of these (bub-legend-pos, bub-model-show, etc.).

        After 2026-04-25 the `_v` helper auto-falls-back to SNAPSHOT_DEFAULTS
        when no explicit default is passed, so the cleanest pattern is to
        omit the default entirely. This test enforces that any explicit
        default the developer chooses to keep is in agreement with
        SNAPSHOT_DEFAULTS.
        """
        import ast
        import os
        import pathlib
        from snapshot_defaults import SNAPSHOT_DEFAULTS

        # Calls that intentionally override SNAPSHOT_DEFAULTS. Each entry is
        # `cid:prop` and the reason. Keep this list short — every entry is
        # a place where snapshot_defaults can't be the authority.
        INTENTIONAL_OVERRIDES = {
            # (None currently. The MC=[] gates were architect-flagged as
            # actually-wrong for legacy links. After commit removing those,
            # this set should stay empty.)
        }

        # Module-imported tab_defaults dicts also count as "match" — many
        # builders use `default=BUBBLE["pt_size"]` etc. which is the
        # snapshot-defaults-equivalent value via the prewarm dict.
        from tab_defaults import (BUBBLE, HEATMAP, DCA, RETIRE, SUPERCHARGE,
                                  CITADEL, LEVERAGE_DEFAULTS)
        TAB_DICTS = {
            "BUBBLE": BUBBLE, "HEATMAP": HEATMAP, "DCA": DCA, "RETIRE": RETIRE,
            "SUPERCHARGE": SUPERCHARGE, "CITADEL": CITADEL,
            "LEVERAGE_DEFAULTS": LEVERAGE_DEFAULTS,
        }

        here = pathlib.Path(os.path.dirname(__file__))
        src = (here / "restore_builder.py").read_text()
        tree = ast.parse(src)

        def _eval_constant(node):
            """Evaluate a literal-ish AST node to its Python value.

            Handles constants, lists, tuples, dicts of constants, and
            tab_defaults subscripts like `BUBBLE["pt_size"]`. Returns
            `_NOT_LITERAL` sentinel otherwise.
            """
            if isinstance(node, ast.Constant):
                return node.value
            if isinstance(node, (ast.List, ast.Tuple)):
                vals = [_eval_constant(e) for e in node.elts]
                if any(v is _NOT_LITERAL for v in vals):
                    return _NOT_LITERAL
                return list(vals) if isinstance(node, ast.List) else tuple(vals)
            if isinstance(node, ast.Subscript):
                # BUBBLE["pt_size"] etc.
                if (isinstance(node.value, ast.Name)
                        and node.value.id in TAB_DICTS):
                    key_node = node.slice
                    if isinstance(key_node, ast.Constant):
                        return TAB_DICTS[node.value.id].get(key_node.value, _NOT_LITERAL)
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
                v = _eval_constant(node.operand)
                if isinstance(v, (int, float)):
                    return -v
            return _NOT_LITERAL

        _NOT_LITERAL = object()
        violations = []
        skipped_non_literal = []

        # Walk all _v(...) calls
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not (isinstance(node.func, ast.Name) and node.func.id == "_v"):
                continue
            # Must have a `default=...` keyword to evaluate
            default_kw = next((kw for kw in node.keywords
                               if kw.arg == "default"), None)
            if default_kw is None:
                continue
            # Need cid (first positional after `state`) and optional prop
            args = node.args
            if len(args) < 2 or not isinstance(args[1], ast.Constant):
                continue
            cid = args[1].value
            prop = "value"
            if len(args) >= 3 and isinstance(args[2], ast.Constant):
                prop = args[2].value
            key = f"{cid}:{prop}"
            if key in INTENTIONAL_OVERRIDES:
                continue
            hardcoded = _eval_constant(default_kw.value)
            if hardcoded is _NOT_LITERAL:
                # Non-literal default (e.g., a function call). Skip.
                skipped_non_literal.append((key, ast.unparse(default_kw.value), node.lineno))
                continue
            expected = SNAPSHOT_DEFAULTS.get(key)
            if hardcoded != expected:
                violations.append(
                    f"line {node.lineno}: _v(...{cid!r}..., default={hardcoded!r}) "
                    f"-- SNAPSHOT_DEFAULTS[{key!r}]={expected!r}"
                )

        # Print info on skipped (non-literal) defaults so they're visible
        if skipped_non_literal:
            print(f"\n[info] skipped {len(skipped_non_literal)} non-literal "
                  f"defaults (function calls, etc.). First 5: "
                  f"{skipped_non_literal[:5]}")

        assert violations == [], (
            f"\n{len(violations)} explicit-default drift in restore_builder.py. "
            f"Either remove the explicit default (let _v fall back to "
            f"SNAPSHOT_DEFAULTS) or add to INTENTIONAL_OVERRIDES with reason. "
            f"\n  " + "\n  ".join(violations)
        )

    def test_restore_builders_cover_prewarm_param_keys(self):
        """Every key in tab_defaults' prewarm dict (BUBBLE/DCA/RETIRE/...)
        must also appear in the corresponding _build_{tab}_figure_from_state
        param dict in restore_builder.py.

        Catches drift caused by adding a new control to update_X (which
        also updates tab_defaults' prewarm baseline) but forgetting to
        mirror it in the restore_builder. The 5-failed-attempt saga in
        memory/restore_callback_architecture.md identified exactly this
        kind of missing-key bug as a root cause of restore correctness
        issues.

        Approach: mock each tab's `_get_{tab}_fig` to capture the params
        dict the builder constructs, then compare key sets.

        Citadel intentionally has no restore_builder (Phase 2 ship 6 just
        writes active-chart-committed; the cached default chart loads via
        the existing cascade). Skipped here.
        """
        from unittest.mock import patch
        import plotly.graph_objects as go
        # Use the *_defaults() helpers, not the raw MappingProxyType dicts.
        # The helpers pop UI-only keys (qs_mode, hm_palette, etc.) that
        # appear in the prewarm dict but are stripped before being passed
        # to the figure-builder fn.
        from tab_defaults import (bubble_defaults, heatmap_defaults,
                                  dca_defaults, retire_defaults,
                                  supercharge_defaults, leverage_defaults)
        import restore_builder

        # Tab → (prewarm dict, builder function name, fig fn module path,
        #         fig fn return shape, minimal state to bypass gates)
        # Bubble: state empty (CTA-active=[] by default → builder runs)
        # DCA/Retire/SC/HM: explicitly disable MC so builder doesn't return None
        # Leverage: simple — no MC, no SC complications
        # NOTE: patch path matters. Bubble's _get_bubble_fig is imported at
        # restore_builder.py module-top, so patch restore_builder._get_bubble_fig.
        # The other tabs import function-locally so patching utils._get_*_fig
        # works because each call re-resolves the name from utils.
        cases = [
            ("bubble", bubble_defaults(),
             restore_builder._build_bubble_figure_from_state,
             "restore_builder._get_bubble_fig", "fig",
             {"main-tabs:active_tab": "bubble"}),
            ("dca", dca_defaults(),
             restore_builder._build_dca_figure_from_state,
             "utils._get_dca_fig", "tuple",
             {"main-tabs:active_tab": "dca", "dca-mc-enable:value": []}),
            ("retire", retire_defaults(),
             restore_builder._build_retire_figure_from_state,
             "utils._get_retire_fig", "tuple",
             {"main-tabs:active_tab": "retire", "ret-mc-enable:value": []}),
            ("supercharge", supercharge_defaults(),
             restore_builder._build_supercharge_figure_from_state,
             "utils._get_supercharge_fig", "tuple",
             {"main-tabs:active_tab": "supercharge",
              "sc-mc-enable:value": []}),
            ("heatmap", heatmap_defaults(),
             restore_builder._build_heatmap_figure_from_state,
             "utils._get_heatmap_fig", "fig",
             {"main-tabs:active_tab": "heatmap", "hm-mc-enable:value": [],
              "hm-active-model:data": "bub"}),
            ("leverage", leverage_defaults(),
             restore_builder._build_leverage_figure_from_state,
             "figures.leverage.build_leverage_figure", "fig",
             {"main-tabs:active_tab": "leverage"}),
        ]

        violations = []
        for tab, prewarm, builder, fig_fn_path, return_shape, state in cases:
            captured = {}

            def _capture(p, *args, **kwargs):
                captured["params"] = dict(p)  # snapshot at call-time
                if return_shape == "tuple":
                    return (go.Figure(), None)
                # bubble's _get_bubble_fig returns a Figure directly
                return go.Figure()

            with patch(fig_fn_path, side_effect=_capture):
                result = builder(state)

            assert "params" in captured, (
                f"{tab}: builder didn't reach figure-fn call (likely returned "
                f"None due to a gate; adjust state to bypass)"
            )
            captured_keys = set(captured["params"].keys())
            prewarm_keys = set(prewarm.keys())

            # Some prewarm keys are intentionally not figure params (e.g.,
            # cache-key salt or UI-only flags). The check is one-directional:
            # every prewarm key SHOULD appear in builder's params, OR be in
            # this allowlist of known prewarm-only keys.
            #
            # Currently no allowlist is needed; the prewarm dicts are tightly
            # coupled to the figure-builder param dicts. Add entries here if
            # a prewarm key is intentionally unmapped.
            PREWARM_ONLY_ALLOWLIST = {
                # tab → set of keys that are in prewarm but NOT in builder
                # (e.g., if prewarm needs an extra cache-key field)
                # Bubble MC controls (Task 2-4 scaffolding): keys live in
                # bubble_defaults() so prewarm cache-key matches a future
                # MC-on bubble runtime, but the chart builder doesn't yet
                # consume them — see Task 11 for plumbing.
                "bubble": {
                    "mc_amount", "mc_bins", "mc_enabled", "mc_entry_q",
                    "mc_freq", "mc_infl", "mc_model_src", "mc_regime",
                    "mc_sims", "mc_start_yr", "mc_window", "mc_years",
                },
            }
            allowlist = PREWARM_ONLY_ALLOWLIST.get(tab, set())

            missing = prewarm_keys - captured_keys - allowlist
            if missing:
                violations.append(
                    f"{tab}: prewarm has keys {sorted(missing)} that are "
                    f"NOT in the restore builder's params dict. Either add "
                    f"them to _build_{tab}_figure_from_state or to the "
                    f"PREWARM_ONLY_ALLOWLIST."
                )

        assert violations == [], (
            f"\n{len(violations)} prewarm/restore-builder param key drifts. "
            f"Memory: see lessons-learned section in "
            f"restore_callback_architecture.md.\n  "
            + "\n  ".join(violations)
        )

    def test_checklist_options_cover_default_values(self):
        """For every list-valued entry in SNAPSHOT_DEFAULTS where the
        component is in _CHECKLIST_OPTIONS, every default-list element
        must be in the options list.

        Missing values are silently bitmask-stripped to 0 (the 'bub'
        bug for dca/ret/sc/hm-model-show, fixed in commit cda79e0).
        See memory/restore_callback_architecture.md.
        """
        from snapshot_defaults import SNAPSHOT_DEFAULTS
        from snapshot import _CHECKLIST_OPTIONS

        violations = []
        for key, val in SNAPSHOT_DEFAULTS.items():
            if not isinstance(val, list):
                continue
            cid = key.split(":", 1)[0]
            if cid not in _CHECKLIST_OPTIONS:
                continue
            opts = set(_CHECKLIST_OPTIONS[cid])
            missing = [v for v in val if v not in opts]
            if missing:
                violations.append(
                    f"{key}: default contains {missing!r} but "
                    f"_CHECKLIST_OPTIONS[{cid!r}] does not include "
                    f"these — they will be silently dropped through "
                    f"the bitmask round-trip."
                )

        assert violations == [], (
            f"\n{len(violations)} _CHECKLIST_OPTIONS missing values that "
            f"appear in SNAPSHOT_DEFAULTS. Append the missing values to "
            f"the END of the list (preserves bit indices for legacy share "
            f"links). Violations:\n  " + "\n  ".join(violations)
        )


class TestSnapshotV4:
    def test_q4_round_trip_with_diffs(self):
        from snapshot import _encode_snapshot_v4, _decode_snapshot_v4
        state = {
            "main-tabs:active_tab": "bubble",
            "bub-xscale:value":     "lin",
            "bub-n-future:value":   5,
            "bub-toggles:value":    ["shade", "show_data"],
        }
        encoded = _encode_snapshot_v4(state)
        decoded = _decode_snapshot_v4(encoded)
        for k, v in state.items():
            assert decoded.get(k) == v, f"{k}: {decoded.get(k)!r} != {v!r}"

    def test_q4_link_is_shorter_than_q3_for_realistic_share(self):
        """In production the browser sends VALUES for every control
        (state has all 310 keys populated). q3 then encodes 310 real
        values; q4 encodes only diffs. Test simulates that."""
        from snapshot import (_encode_snapshot, _encode_snapshot_v4,
                              _SNAPSHOT_CONTROLS)
        from snapshot_defaults import SNAPSHOT_DEFAULTS
        state = {f"{c}:{p}": SNAPSHOT_DEFAULTS.get(f"{c}:{p}")
                 for c, p in _SNAPSHOT_CONTROLS}
        state["main-tabs:active_tab"] = "bubble"
        state["bub-xscale:value"] = "lin"
        state["bub-n-future:value"] = 5
        q3 = _encode_snapshot(state)
        q4 = _encode_snapshot_v4(state)
        assert len(q4) < len(q3) * 0.5, (
            f"q4 ({len(q4)}) not < 50% of q3 ({len(q3)})")

    def test_q4_fingerprint_tamper_returns_none(self):
        from snapshot import _encode_snapshot_v4, _decode_snapshot_v4
        encoded = _encode_snapshot_v4({"bub-xscale:value": "lin"})
        bad = "deadbeef:" + encoded.split(":", 1)[1]
        assert _decode_snapshot_v4(bad) is None

    def test_q4_unknown_fingerprint_falls_back_to_current(self):
        import snapshot
        from snapshot import _encode_snapshot_v4, _decode_snapshot_v4
        encoded = _encode_snapshot_v4({"bub-xscale:value": "lin"})
        saved_cache = snapshot._registry_cache
        snapshot._registry_cache = {}
        try:
            decoded = _decode_snapshot_v4(encoded)
            assert decoded is not None
            assert decoded.get("bub-xscale:value") == "lin"
        finally:
            snapshot._registry_cache = saved_cache

    def test_q4_dispatcher_picks_q4_over_q3(self):
        from callbacks.snapshot_cb import _decode_snapshot_by_prefix
        from snapshot import _SNAP_PREFIX_V4, _encode_snapshot_v4
        encoded = _encode_snapshot_v4({"bub-xscale:value": "lin"})
        h = f"{_SNAP_PREFIX_V4}{encoded}"
        state, prefix, _ = _decode_snapshot_by_prefix(h)
        assert prefix == _SNAP_PREFIX_V4
        assert state.get("bub-xscale:value") == "lin"

    def test_q4_legacy_q3_still_decodes(self):
        from callbacks.snapshot_cb import _decode_snapshot_by_prefix
        from snapshot import _encode_snapshot, _SNAP_PREFIX
        encoded = _encode_snapshot({"bub-xscale:value": "lin"})
        h = f"{_SNAP_PREFIX}{encoded}"
        state, prefix, _ = _decode_snapshot_by_prefix(h)
        assert prefix == _SNAP_PREFIX
        assert state.get("bub-xscale:value") == "lin"

    def test_q4_mc_null_out_does_not_clobber_retire_changed_sims(self):
        from snapshot import _encode_snapshot_v4, _decode_snapshot_v4
        state = {"ret-mc-enable:value": ["yes"], "ret-mc-sims:value": 999}
        encoded = _encode_snapshot_v4(state)
        decoded = _decode_snapshot_v4(encoded)
        assert decoded.get("ret-mc-sims:value") == 999, (
            f"Expected sims=999, got {decoded.get('ret-mc-sims:value')!r}")

    def test_q4_mc_null_out_drops_dca_mc_fields_when_default_disabled(self):
        from snapshot import _encode_snapshot_v4, _SNAPSHOT_CONTROLS
        import json, base64, gzip
        state = {"dca-amount:value": 200}
        encoded = _encode_snapshot_v4(state)
        _fp, blob = encoded.split(":", 1)
        payload = json.loads(gzip.decompress(base64.urlsafe_b64decode(blob)))
        diffs = payload[1]
        for i, (cid, _) in enumerate(_SNAPSHOT_CONTROLS):
            if cid.startswith("dca-mc-") and cid != "dca-mc-model-src":
                assert str(i) not in diffs, (
                    f"{cid} should be dropped by MC null-out")
