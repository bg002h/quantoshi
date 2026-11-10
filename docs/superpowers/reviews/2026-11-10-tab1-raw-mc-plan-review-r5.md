# Review r5 — Tab-1 Raw MC implementation plan after fold 4 (`dd736215`)

**Counts: 0 Critical · 0 Important · 4 Minor · 6 Nit**

Scope: (1) is each round-4 finding resolved in the plan text, (2) did fold 4 introduce a defect,
with the focus on the price-registry contract end to end (Task 1 `set_prices` / `get_prices` /
`months_for` / `raw_prices` fixture; Task 2 `RawRequest.asof` / `simulate(req)`; Task 3
`regime_options` wrapper + inner cache; Task 5 `update_bubble` call, `app.py` registration, tests).
The spec is not re-reviewed; earlier-settled material is not re-audited. Nothing in the repo was
modified (read-only greps and reads; `git status` unchanged: only the pre-existing
`model_data_ef.pkl`).

## Machine-checked (working tree at `dd736215`)

- `app.py`: `M = load_model_data()` at :68, `_app_ctx.M = M` at :343, `import layout` at :547;
  `_app_ctx._MODEL_FP` is fixed at `_app_ctx` import (:161), so a `set_prices` call anywhere
  between :69 and :546 precedes every layout import (r4 already import-hooked this).
- `ModelData.price_dates` is `"%Y-%m-%d"` strings and `price_prices` an ndarray
  (`btc_core/_model_data.py:33-35, 84-85`), so the registration expression is well-formed. The
  CSV's last row (`11/8/26`) is the same day as the pkl's last date, so `FROZEN = 2026-11-08` is
  consistent between the fixture series and the production series.
- The Tab-1 layout is built lazily: `_bubble_tab()` has one production caller
  (`callbacks/routing.py:612`, lazy-tab callback). No non-test module outside `app.py` builds it
  (`static_pages.py` imports only `layout.faq` / `layout.model_info`). So `regime_options` ->
  `get_prices()` raising `RuntimeError` when unset can only bite a test that unsets the registry,
  which is what the restoring fixture is for.
- `conftest.py` imports `app` at module level, so under the suite the registry holds the
  production series (fp = `_MODEL_FP`) unless a test replaces it. `pytest.ini` has no
  `python_files` override, so `raw_mc_testing.py` is not collected.
- No in-process `load_model_data()` reload exists in `btc_web/` runtime code (`app.py:68` only;
  `timemachine.py:77` reads quantile metadata), so a startup-only registration cannot go stale
  within a process.
- Name consistency across the contract: `set_prices` / `get_prices` / `months_for` /
  `raw_prices` / `PRICES` / `FROZEN` / `RawRequest.asof` / `RawRequest.data_fp` /
  `simulate(req)` / `regime_options` (+ inner `_regime_options`) are spelled identically in every
  Interfaces block, test list and Step 3. Callers: `set_prices` <- `app.py` (Task 5 Files) and the
  fixture; `get_prices` <- `update_bubble` (Task 5) and the `regime_options` wrapper; `months_for`
  <- `simulate` and the inner `_regime_options`. Every one has an owner. Callers of
  `load_months` in production: only `months_for`.

## Round-4 findings: disposition

| r4 | Resolved in plan text? | Note |
|---|---|---|
| I-1 price-registry contract split; `months_for` dead; Step 3 contradicts Task 1; no restore | **Yes (blocking substance)** | `months_for` now has two consumers (`simulate` via `req.asof`, `_regime_options`); `simulate(req)` takes no series; Step 3 builds `data_fp = raw_mc.get_prices()[1]`, `asof=None`, calls `raw_mc.simulate(req)`; the colliding "cached per `data_fp`" is gone; `test_prices_built_once` can now fail and pass; `raw_prices` fixture saves/restores the registry. Execution gaps that remain in how the fixture is requested, how "unset" is reached, and the Task 1 -> Task 3 forward reference are filed as M-1..M-3 and N-1 below. |
| M-1 `regime_options` cannot key on `data_fp` | **Yes (design); residual M-1** | Wrapper reads `get_prices()[1]`, inner `lru_cache`d `_regime_options(..., data_fp)`, `set_prices` clears it. Residual: the clear is a forward reference from Task 1 and nothing tests it (new M-1). |
| M-2 round-trip test compares tuples with `==` | Yes | Test is now `got[0] is s and got[1] == "fp1"`, and `set_prices` "stores the series object as given". |
| N-1 Files says 9 ids / no-Markov clause | Yes | Files line says "9 new control/status ids plus `bub-mc-panel`"; the existing-test update says 10 ids and "no `bub-mc-panel` on the other five tabs"; Interfaces list counts to 9. |
| N-2 Task 3 fragment / re-declared registry | Yes | Task 3 no longer re-declares `set_prices`/`get_prices`; sentence is clean. |
| N-3 `test_mc_regime_paths.py` listed twice | Yes | Task 6 Files lists it once. |
| N-4 Task 1 registry tests need a registered series | Yes | Both take `raw_prices`. |

## Important

None.

## Minor

### M-1 Task 1 `set_prices` calls a Task 3 symbol, nobody owns adding the call, and nothing tests the invalidation
- **Where:** Task 1 Interfaces (line 63: "clears the `months_for` dict **and** `regime_options.cache_clear()` (Task 3)"); Task 3 Interfaces (line 147) and Step 3 (line 172); Task 3 test list.
- **Failure:** `regime_options` does not exist until Task 3. Implemented literally, Task 1's `set_prices` raises `NameError`, so every test that uses `raw_prices` (all of Task 1's registry tests, and the fixture's own setup/teardown) errors at Task 1 Step 4 "PASS". The "(Task 3)" tag implies the call is added later, but Task 3 never says "add it to `set_prices`", and no test covers it. r4 M-1's whole point (labels from a previous registry after a swap) therefore has no test that can fail: Task 3 has no `regime_options` test at all (`test_labels_format` reads like `bin_labels`), and Task 5's `test_raw_regime_options_callback` only covers the callback and the exp-failure path.
- **Fix:** Task 1: `set_prices` clears only the `months_for` dict. Task 3 Step 3: add the `regime_options.cache_clear()` call to `set_prices` and attach `regime_options.cache_clear = _regime_options.cache_clear`. Add `test_regime_options_follows_registry` (register series A under fp `"x"`, read options; register a truncated series B under the same fp, options differ).

### M-2 How `raw_prices` is requested is unspecified, and both obvious readings break a listed test
- **Where:** Task 1 Interfaces (line 63: "imported by every `test_raw_mc_*.py`"); Task 2 test stubs (lines 116-132, no fixture); Task 3 `test_steering_targets` (line 156, signature takes none; comment says it "uses the raw_prices fixture"); Task 5 `test_app_registers_prices` (line 225, no fixture) next to `test_prices_built_once(monkeypatch, raw_prices)` in the same module.
- **Failure:** `simulate(req)` now reads the registry and asserts `req.data_fp == get_prices()[1]`; every Task 2/3 test builds `data_fp="csv"`. Under the suite the registry holds the production fp, so a test that does not request the fixture fails with the assertion. Making the fixture `autouse` fixes those but breaks `test_app_registers_prices`, which lives in `test_raw_mc_callback.py` (that module imports the fixture for `test_prices_built_once`) and expects `_app_ctx._MODEL_FP`. Importing a fixture does not apply it. The failure is loud, not silent.
- **Fix:** State in Task 1: fixture is function-scoped, not autouse; Task 2 and Task 3 test modules set `pytestmark = pytest.mark.usefixtures("raw_prices")`; Task 5 tests that need the CSV series take it explicitly, `test_app_registers_prices` and `test_raw_start_price_falls_back_to_last_close` do not. Any test that monkeypatches a fit failure and then calls `regime_options` (Task 5 exp-failure case) must request the fixture or call `regime_options.cache_clear()`, since the inner cache is process-global.

### M-3 The "unset" state has no named handle
- **Where:** Task 1 line 63 ("restores the saved registry (or clears it)") and line 88 ("after clearing -> RuntimeError").
- **Failure:** There is no unset API and the registry attribute is never named, so the test and the fixture each have to guess a private attribute. r4 I-1 asked for it to be named; fold 4 kept the behaviour but not the name. A test that guesses wrong passes vacuously or fails at setup.
- **Fix:** Name it in Task 1 (for example `raw_mc._REGISTRY: tuple[pd.Series, str] | None`); the round-trip test uses `monkeypatch.setattr(raw_mc, "_REGISTRY", None)`; the fixture reads it for the "saved" value.

### M-4 Task 1 commit omits the fixture module its tests import
- **Where:** Task 1 Step 5 (line 96): files are `raw_mc.py`, `test_raw_mc_fit.py`, `tools/raw_mc_cycle_stats.py`; Task 1 Files (line 52) creates `btc_web/raw_mc_testing.py`.
- **Failure:** With the project's scoped-`git add` rule the new module stays untracked, so the Task 1 commit imports a file it does not contain and is red on a clean checkout.
- **Fix:** Add `btc_web/raw_mc_testing.py` to the Step 5 file list.

## Nit

- **N-1** Task 5 Step 3 (line 235) still omits the `app.py` `raw_mc.set_prices(...)` registration (r4 I-1 bullet 4); it is only in Files and Task 1. Add it as the first Step 3 item so `test_app_registers_prices` is not the only prompt.
- **N-2** `_req` has no owner: Task 2 tests run at `asof = today = 2024-04-20`, Task 3's `_req` uses `asof=FROZEN` (2026-11-08). Define it once in `raw_mc_testing.py` with explicit `asof` / `today` parameters.
- **N-3** The inner `_regime_options` in `raw_mc` shares its name with the existing `layout.mc_controls._regime_options` (re-exported from `layout/__init__.py`). Different modules, so no collision today; rename the inner one `_regime_options_cached` to keep grep results unambiguous.
- **N-4** Task 5 Step 3 says `p0` = live price "else last close" without naming the source. Say `get_prices()[0].iloc[-1]` so the fallback uses the same registry as the fan and the test derives its expectation from it.
- **N-5** `simulate` enforces `data_fp` with an `assert` (stripped under `-O`); use a `ValueError`. Also `_app_ctx._MODEL_FP` is `"unknown"` when the process cwd lacks `model_data.pkl` (`_app_ctx.py:155`, relative path), so `data_fp` is a process-lifetime key, not a data identity; harmless here because registration is once per process, but worth a comment.
- **N-6** (carried from r4's machine-check note, not an r4 finding) Step 3 does not say the Raw branch and `mc_visible` must sit above `_mc_setup` (line 357; `mc_visible` is computed at 367). `test_raw_branch_never_calls_payment` pins it, so TDD will find it; one sentence saves a loop.

## Verdict

Green on the gate definition: 0 Critical / 0 Important. The registry contract is now a single path
(`set_prices` -> `get_prices` -> `months_for` -> `simulate(req)` / `regime_options`), names agree
everywhere, and each changed test can fail and pass once M-1..M-3 are folded (all are one-line
plan edits). Fold M-1..M-4 and the Nits, then close without a further full round; re-review
only the changed lines.
