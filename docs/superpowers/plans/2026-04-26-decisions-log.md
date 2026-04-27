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

