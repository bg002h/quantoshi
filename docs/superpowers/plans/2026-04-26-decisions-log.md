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

### D12 (2026-04-27): Phase 2a complete; acceptance gate PASSED with byte-identical fingerprint
- **What:** Phase 2a (refactor + parameterize) shipped in 8 tasks plus a small EF rebuild commit (D5b non-blocking issue handled in same flow). Marker commit at `5d947bc`. Calendar-mode rebuild produced numerically **byte-identical** fingerprint vs the pre-2a pkl (empty `diff`). Test suite: 1600 passed / 2 pre-existing master failures / 10 skipped. Smoke: 200/200.
- **Why this matters:** Validates Phase 2a's invariant ("calendar-mode behavior unchanged"). The refactor was successful — block-mode is now wired into the build pipeline but not exercised yet. Phase 2b builds the actual `model_data_block.pkl`.
- **Branch state:** `time-basis-toggle` is now ~13 commits ahead of master since Phase 2a started. Not pushed. Prod still on Phase 1 marker (`d0638fc`) per the user's prior deploy.
- **Reversibility:** Trivial — `git revert 5d947bc..HEAD` to undo Phase 2a. Phase 2a is purely additive in code paths (calendar mode unchanged, block mode parameterized but not yet shippable since no block pkl exists).

### D11 (2026-04-27): Pivot — "Blocks win." Skipping the formal A/B comparison report; going straight to making block the canonical site default.
- **What:** User declared block-axis fits superior based on their independent research (the `qstar_*`, `crossing_symmetry`, `temporal_sweep`, `blocksweep` tools they checked into master commit `25eecff`). The spec's Phase 2b "decision gate" is now considered cleared.
- **Phase 2 path change:**
  - **Phase 2a (refactor + parameterize)** — unchanged from spec. Mechanical refactor; calendar-mode behavior unchanged.
  - **Phase 2b (build block pkl)** — **drop the R²/AIC/OOS-RMSE/calendar-osc-amplitude comparison report.** Just produce `model_data_block.pkl` + bound-rescale LPPL/EPPL/HybPPL `W_cal`. Sanity-check fits don't NaN.
  - **Phase 2c (runtime axis loader)** — **required**, not optional. Was Phase 3.
  - **Phase 2d (heavy caches)** — **required**, ~6h. Was Phase 4.
  - **Phase 2e (flip + deploy)** — edit `quantoshi.toml` to `time_basis = "block"`, rebuild + redeploy. Was Phase 5.
- **Comparison report deferred** — user said "Run report another day." Logging this so the report TODO doesn't get lost: a future task is to produce the formal R²/AIC/OOS-RMSE/calendar-osc-amplitude comparison after both pkls coexist on disk. Save to `docs/superpowers/specs/time_basis_phase2_results.md`.
- **Why:** User has prior research evidence the controller doesn't have; trust + go.
- **Reversibility:** Whole pivot is reversible at the `quantoshi.toml` level — flip back to `calendar` and rebuild.

### D8: Decisions-log audit caught 3 minor items, all addressed
- **What:** A fresh agent audited this very file (D1–D7) and flagged:
  - **D8a:** No explicit log entry for the snapshot-fp behavior change in commit `bc041d1` (the `h.update(TIME_BASIS.encode())` insertion). D4 covered the registry update only. **Addressed by adding D9 below.**
  - **D8b:** Group B "pre-existing master failures" rested on the controller's word; the auditor wanted independent verification. **Done:** I checked out master HEAD (`d21fe7e`) and ran the two failing tests. Both fail there too. D6 is verified.
  - **D8c:** D5a (`sys.path.insert` hack) reversibility was true but "harmless to defer" understated — the hack is *load-bearing for test isolation*. Once `tools/model_toolkit/export.py` is imported in any pytest run, `btc_web/` stays on `sys.path` for the rest of the process. **Tightened: see D5a-amended below.**
- **Why:** Audit fidelity. The user explicitly asked for a critical review of this log.
- **Reversibility:** This entry is documentation; nothing to reverse.

### D9: Snapshot-fp behavior change (commit bc041d1) — explicit log entry
- **What:** Task 5 modified `_compute_snapshot_defaults_fingerprint()` to feed `TIME_BASIS` + `\x00` into the streaming sha256 hash *before* the `_SNAPSHOT_CONTROLS` loop. Algorithm (sha256), slice (`[:8]`), and lazy `_SNAPSHOT_CONTROLS` import preserved. Calendar-mode fingerprint changed from `60990754` → `4fbb63a6`.
- **Why:** Per spec §3.4, the share-link fingerprint must include the active axis so calendar/block links never collide. Phase 3 will enforce strict cross-axis decode rejection; Phase 1 only reserves the slot.
- **Impact on existing share links:** Old `q4:` links built before this commit decode against the historical-defaults registry (which retains the prior `60990754` entry — see D4). The fallback path is pre-existing infrastructure, not new code.
- **Reversibility:** Three-step revert: (1) remove the two `h.update()` calls in `snapshot_defaults.py`, (2) re-run `tools/update_defaults_registry.py` (it'll re-promote the old fp to current), (3) optionally `git checkout` the registry JSON. Trivial.

### D5a-amended: `sys.path.insert` hack is load-bearing, not just stylistic
- **Original D5a:** Phase 2 cleanup, "harmless to defer."
- **Auditor's correction:** The hack is load-bearing for test isolation — once any test imports `tools/model_toolkit/export.py`, `btc_web/` stays on `sys.path` for the remainder of the pytest process. Currently no observed breakage but it's a real isolation hazard, not just stylistic.
- **Action taken:** No code change, but the `TODO(phase2)` comment in `export.py` (commit `8b04a45`) already documents the hazard.

### D10: Accidental `git stash pop` during D8b verification
- **What:** While verifying D8b on master HEAD, I ran `git stash -u` (stashed nothing — working tree was clean) then later `git stash pop` (popped a *prior* unrelated stash — `WIP on master: d21fe7e fix(mi-deeplink)…`). This created merge conflicts in `btc_web/callbacks/routing.py` and `btc_web/layout/model_info/__init__.py`.
- **Recovery:** `git checkout HEAD -- .` reverted the working tree to the committed state (`feb9bf9`). The stash entry was preserved on the stack (conflict means `pop` becomes `apply`, not removal).
- **Lesson:** `git stash pop` always pops top-of-stack regardless of whether the controlling agent created it. Better practice: check `git stash list` before either command, or commit changes to a throwaway branch instead of stashing.
- **Reversibility:** Already reverted. Original stash@{0} preserved unchanged.

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

