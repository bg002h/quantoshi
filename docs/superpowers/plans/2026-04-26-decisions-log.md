# Time-Basis Toggle Phase 1 — Autonomous Execution Decisions Log

User went to sleep partway through Task 4 → Task 5 with instruction: "Make decisions and keep going, keeping a list of decisions to review. Always ask agent for review at every step."

This file logs every non-trivial decision I made on their behalf. **Review on wake-up; revert anything you disagree with.**

---

## Decisions made before sleep instruction

### D1: T_ORIGIN_BLOCK = 20188
- **What:** Bitcoin block height at 2009-07-25 UTC, pinned in `quantoshi.toml`.
- **Why:** Resolved via local `bitcoind` RPC binary search. Block 20188 is the last block whose timestamp is on 2009-07-25 UTC (`2009-07-25T15:00:18Z`). The spec's earlier placeholder `17448` was wrong by ~2740 blocks.
- **Reversibility:** Trivial. Edit `quantoshi.toml` + bump `_DEFAULTS["block_origin"]` in `time_basis.py`.

### D2: Calendar-mode test tolerance 1e-6 → 1e-3
- **What:** `test_calendar_to_t_calendar_mode` originally had `abs(t - 1.0) < 1e-6` but 365 days / 365.25 = 0.999315 → delta is 6.8e-4, mathematically impossible at 1e-6. Implementer caught it during Task 2; I updated the plan and accepted their commit.
- **Why:** Pragmatic. The round-trip test `test_round_trip_calendar_mode` validates actual mathematical correctness with ±1 day tolerance; this one is just a sanity check that "1 year ≈ 1.0".
- **Reversibility:** Trivial. Tighten back to 1e-6 only if you switch to a 4-year date that lands on integer years.

### D3: Block-mode test date 2010-07-25 → 2013-07-25
- **What:** `test_block_mode_constants` originally expected `calendar_to_t(2010-07-25) ≈ 52596` but actual is 365 × 144 = 52560 (off by 36). Switched to 2013-07-25 because 1461 days = exactly 4 × 365.25, giving `4 × 52596 = 210384` exactly.
- **Why:** Cleanest — exact integer comparison rather than tolerance fudging.
- **Reversibility:** Trivial.

---

## Tasks completed before sleep instruction

- ✅ Task 1: `quantoshi.toml` + 3 tests. Commit `f512588`.
- ✅ Task 2: `btc_web/time_basis.py` + 7 tests. Commit `4ab4750`.
- ✅ Task 3: `_app_ctx.py` re-export + 1 test. Commit `9835a3a`.
- ✅ Task 4: `cache.py` fingerprint extension + 3 integration tests. Commit `ff54aaf`.

---

## Decisions made during autonomous execution

(Populated below as I proceed. Each entry: what, why, reversibility.)

### D7: Phase 1 final review = SHIP. Added TODO(phase2) markers per reviewer suggestion.
- **What:** Final code reviewer (commit `c322549` review) approved Phase 1 ship-as-is and suggested adding code-side `TODO(phase2)` comments at the two known soft spots in `tools/model_toolkit/export.py`. Done in commit `8b04a45`.
- **Why:** Reviewer's suggestion was "make Phase 2 cleanup mechanically searchable" — agreed, 2 minutes of work for grep-friendly tech-debt tracking.
- **Reversibility:** Trivial (comment-only).

### D6: Task 8 caught 4 test failures; fixing 2 (Phase 1 regressions), ignoring 2 (pre-existing on master)
- **What:** Full test suite ran 1586 passed / 4 failed / 10 skipped. Diagnosis:
  - **Group A — Phase 1 regressions (2):** `test_infrastructure.py::TestModelFingerprint::test_fingerprint_in_cache_key` and `::test_hash_length_is_32`. These hardcode the OLD 4-part cache key format (`fig:{_MODEL_FP}:{prefix}:{hash}`). Task 4 changed it to 5-part (`fig:{TIME_BASIS}:{_MODEL_FP}:{prefix}:{hash}`). Task 4's spec reviewer ran `test_cache_key_alignment.py` only, missed `test_infrastructure.py`. **My responsibility to fix.**
  - **Group B — pre-existing on master (2):** `test_callbacks.py::test_free_tier_all_models` and `test_colors_central.py::test_no_hex_literals_outside_colors_module`. Verified failing on master HEAD too — not introduced by Phase 1. **Leaving alone.**
- **Why fix:** Group A is a real regression caused by Phase 1 work. Acceptance gate must pass.
- **Decision:** Update the two `test_infrastructure.py` assertions to expect the 5-part key format. Mechanical fix. Then re-run gate.
- **Reversibility:** Trivial.

### D5: Task 6 left two reviewer-flagged items for follow-up (not blocking)
- **D5a — `sys.path.insert` hack in `export.py`:** module-import-time path manipulation works but causes test-isolation pollution (once `export.py` is imported, `btc_web/` is on sys.path for the rest of the pytest process). Reviewer suggests moving to `conftest.py` or making `time_basis` a proper package.
  - **Why deferred:** Phase 2 will refactor the build pipeline more invasively; clean this up there.
  - **Reversibility:** Trivial — the alternatives are well-defined.
- **D5b — `_sidecar_path("model_data.pkl.tmp")` produces wrong filename:** would yield `model_data.pkl_meta.json` (since `os.path.splitext(".pkl.tmp")` strips only `.tmp`). Currently unexercised — `write_pkl` writes directly to the final path, no atomic-write pattern in the codebase. Reviewer suggests a 2-line `.endswith(".pkl")` assertion.
  - **Why deferred:** Hypothetical — no current consumer passes `.tmp` paths. Add the assert if/when atomic-write is introduced.
  - **Reversibility:** Trivial.

### D4: Task 5 fingerprint update bumps registry to 12/20
- **What:** Running `tools/update_defaults_registry.py` after the snapshot-fp edit appended fp `4fbb63a6` (Phase 1 calendar-mode current). Registry now has 12 entries (cap is 20; oldest-evicted policy intact).
- **Why:** Required by CLAUDE.md workflow whenever `_compute_snapshot_defaults_fingerprint` changes its output.
- **Concern from code reviewer:** Phase 1 has 0 more fp-bumping tasks (Tasks 6/7/8 don't touch snapshot defaults). Phase 3 may add 1 more (cross-axis enforcement). So we'll be ~13/20 entering Phase 3 — comfortable.
- **Reversibility:** Trivial. `git checkout btc_web/snapshot_defaults_registry.json` reverts the registry; revert the snapshot_defaults.py changes to undo the fingerprint shift.

