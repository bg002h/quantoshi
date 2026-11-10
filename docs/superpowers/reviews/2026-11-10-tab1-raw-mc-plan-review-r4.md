# Review r4 — Tab-1 Raw MC implementation plan after fold 3 (`6335d361`)

**Counts: 0 Critical · 1 Important · 2 Minor · 4 Nit**

Scope: (1) is each round-3 finding resolved in the plan text, (2) did fold 3 add a wrong
citation, an inconsistent name between Interfaces blocks, a test that cannot fail or cannot pass,
or a Dash 4.0 shape error. No fresh audit; the spec is not re-reviewed. Nothing in the repo was
modified (one read-only `import app` with bytecode writing off, driven from a script in the
scratchpad; `git status` unchanged: only the pre-existing `model_data_ef.pkl`).

## Machine-checked (working tree at `6335d361`)

- **`set_prices` placement is valid.** `M = load_model_data()` is `app.py:68`; `_app_ctx.M = M` is
  `:343`; `import layout` is `:547`. An import hook on a real `import app` (TESTING=1) shows the
  first `layout` import at `app.py:547` with `_app_ctx.M` already set. The only earlier hit for
  "layout" in `figures/heatmap.py:500` is inside a function. `_app_ctx._MODEL_FP` is computed at
  `_app_ctx` import (`_app_ctx.py:161`, pkl mtime+size, independent of `M`), so any line between
  69 and 546 works. `M.price_dates` is a list of date strings (last `2026-11-08`),
  `M.price_prices` an ndarray: `pd.Series(M.price_prices, index=pd.to_datetime(M.price_dates))`
  is well-formed.
- `conftest.py` imports `app` at module level, so every xdist worker registers the production
  series before any test runs. `pytest-randomly` is not installed (only xdist, asyncio, timeout);
  order inside a worker is deterministic, but which tests share a worker under `-n logical` is
  load-dependent.
- `update_bubble` (`callbacks/charts/__init__.py`): 8 Outputs, all `allow_duplicate` except
  `bub-mc-status.children`; short-circuit returns at 229 / 259 are `(dash.no_update,) * 8`; the
  final return (587) is an explicit 8-tuple (the plan's "model mode returns `""` for raw-status"
  covers it). `_mc_setup` is called at line 357 unconditionally, before `mc_visible` (366), so the
  Raw branch must sit above it to satisfy `test_raw_branch_never_calls_payment`; the test pins
  this even though Step 3 does not say it. `_add_mc_fan(label_suffix=)`, `fan_suffix`,
  `_add_mc_spaghetti(..., n_display=)` already exist.
- No id collisions: `bub-mc-lines`, `bub-mc-extremes`, `bub-mc-raw-*`, `bub-mc-panel` appear
  nowhere outside tests; `hm-mc-panel` exists (`callbacks/mc_controls.py:273`).
- Auto-tick callback (`callbacks/mc_controls.py:162-169`) has a primary Output and
  `prevent_initial_call=True`; the Dash 4.0 bug note (`plot_appearance.py:22-28`) is specific to
  `allow_duplicate` + several Inputs, so two Inputs there is consistent. Declared order
  `Input, Input, State, State, State` matches `MC_REGIME_PATHS_JS(checked, rawChecked, bins,
  rawBins, paths)`. `MC_AUTOWIDEN_JS` is defined in `layout/bubble.py`, registered with one Input
  plus States, so a 6th State (`bub-mc-model-src`) is shape-safe.
- Interfaces consistency: the eight new ids/params agree in order across Task 5 (Inputs,
  parameters), Task 7 (`NEW`, defaults) and Task 6; `RAW_LINES_*` (5/200/20) agree with
  `bub-mc-lines`; `MC_EXTREME_*` agree between Task 4 and `colors.py`.
- The `bub-mc-panel` outer Div, the `⚡ Paid` badge in `header_right`, and the no-Markov branch
  (`layout/mc_controls.py:211`) are as the plan describes.

## Round-3 findings: disposition

| r3 | Resolved in plan text? | Note |
|---|---|---|
| I-1 `set_prices`/`get_prices` unowned, untested, no unset behaviour | **Partly** | Resolved: Task 1 owns the registry (line 62) with `test_price_registry_round_trip`; `app.py` is in Task 5 Files and the placement is valid; `test_app_registers_prices` added; unset raises `RuntimeError`. Not resolved: Task 5 Step 3 still contradicts Task 1 (`data_fp`, tuple vs Series), the new `months_for` cache has no consumer, and the fixture-restores-the-registry fix was dropped. Carried as **I-1** below. |
| M-1 `bub-mc-panel` not in the placeholder test | Yes | Test now says "all 10 new ids (the 9 + `bub-mc-panel`)". Residual Nit N-1 (Files line still says 9; "other five tabs in no-Markov mode" clause not added). |
| M-2 `xdist_group` inert | Yes | Marker removed in Task 2 and Task 3; the best-of-3 / 0.6 s bound carries the claim. |
| M-3 Files omit edited tests | Yes | `test_mc_autowiden.py` in Task 6 Files, `test_mc_paths_toggle.py` in Task 7 Files. |
| M-4 `load_months`/`cycle_table` cache has no home or safe key | **Partly** | `months_for` defined, keyed `(data_fp, asof)`, cleared by `set_prices`, test re-worded. But nothing calls it, and Task 5 Step 3 still says "cached per `data_fp` — M-7" (the colliding key r3 flagged). Folded into I-1. |
| N-1 Task 7 paragraph unnumbered | Yes | Now Step 3b. |
| N-2 name the eight parameters | Yes | `mc_raw_sims … mc_extremes`, same order, with defaults. |
| N-3 asset/callback click-stamp test | Yes | `test_raw_asset_and_callback_share_the_click_stamp` added. |

## Important

### I-1 — The price-registry contract is still split across Tasks 1, 3 and 5; `months_for` has no consumer and two tests cannot pass or are order-dependent
- **Where:** Task 1 Interfaces (line 62) and tests (87-88); Task 2 `simulate(req, prices)` (line
  110); Task 3 Interfaces (line 146); Task 5 Step 3 (line 234) and tests (224-225).
- **Failure:**
  1. **`months_for` is dead.** `simulate(req, prices: pd.Series)` takes an explicit series (every
     Task 2/3 test passes `PRICES`) and builds its months itself with
     `load_months(prices, asof=req.today)`. Nothing in Tasks 2, 3 or 5 routes through
     `months_for`. `test_prices_built_once` ("two renders: `raw_mc.load_months` called once")
     therefore counts 2 and cannot pass as written; the only way it passes is a wiring the plan
     does not describe (for example `simulate` testing `prices is get_prices()[0]`).
  2. **Step 3 contradicts Task 1.** Task 1 says "`RawRequest.data_fp` is always
     `get_prices()[1]`"; Step 3 still builds it from `_app_ctx._MODEL_FP`. Step 3 still says
     "`simulate` with `raw_mc.get_prices()`", which reads as passing the `(Series, str)` tuple
     where `simulate` takes a Series (r3 I-1 item 4, not fixed), and keeps "`load_months` /
     `cycle_table` cached per `data_fp`", the key that collides across asofs.
  3. **No restore.** `test_price_registry_round_trip` must reach "unset" (there is no unset API),
     and Task 3's fixture "registers the frozen CSV series"; neither restores the production
     registration. Under `-n logical` any later test that lands on the same worker reads the
     polluted global: `test_app_registers_prices` (fp must equal `_MODEL_FP`), and every test
     that builds the Tab-1 layout, which now calls `regime_options` -> `get_prices()`
     (`test_axes_presets.py`, `test_timemachine_layout.py`, `test_mc_paths_toggle.py::TestLayout`).
     Unset gives `RuntimeError` there; a toy `fp1` series gives an unspecified refusal. The
     failure depends on worker co-location, so it is intermittent. r3 I-1 asked for a restoring
     fixture; fold 3 dropped it.
- **Fix:**
  - Pick one consumer path and say it. Smallest: `simulate(req, prices, months=None)`; Step 3
    does `prices, fp = raw_mc.get_prices()`, `RawRequest(..., data_fp=fp)`,
    `simulate(req, prices, months=raw_mc.months_for(req.today))`. Tests that pass `PRICES`
    omit `months`. Or drop `months_for` and `test_prices_built_once`.
  - Delete "cached per `data_fp` — M-7" from Step 3.
  - Add one shared fixture (conftest or a `raw_mc` test helper) that snapshots the module's
    registry with `monkeypatch` and restores it; require it on every test that calls
    `set_prices`, and reach "unset" with `monkeypatch.setattr(raw_mc, "<registry attr>", None)`
    rather than a public call. Name the attribute in Task 1.
  - Add `btc_web/app.py` to Step 3's bullet list (it is only in Files).

## Minor

- **M-1 `regime_options` cannot key on `data_fp`.** Task 3 (line 146) caches
  `regime_options(bins, window, trend, asof=None)` "on `(bins, window, trend, asof, data_fp)`",
  but `data_fp` is not a parameter, and `set_prices` clears only the `months_for` dict. A
  `functools.lru_cache` on that signature returns labels computed from the previous registry after
  a swap (frozen vs production). Fix: cache an inner function that takes `data_fp` as an argument,
  or have `set_prices` call `regime_options.cache_clear()`.
- **M-2 `test_price_registry_round_trip` compares tuples containing a Series with `==`.**
  `get_prices() == (s, "fp1")` passes only through the identity shortcut (the same object
  returned); if `set_prices` copies or sorts, `Series == Series` makes the tuple comparison raise
  `ValueError` (ambiguous truth value). Fix: assert `got[0] is s and got[1] == "fp1"` and state
  that `set_prices` stores the object as given.

## Nit

- **N-1** Task 5 Files (line 200) still says the no-Markov placeholder test is extended with "the
  9 new ids"; the test list (line 215) says 10 (adds `bub-mc-panel`). r3 M-1's "panel absent on
  the other five tabs in the no-Markov build" clause was not added.
- **N-2** Task 3 Interfaces (line 146) now contains a lowercase fragment ("registration is Task
  1's registry; tests register …") and still re-declares `set_prices`/`get_prices` that Task 1
  owns; keep one declaration and refer to it.
- **N-3** Task 6 Files lists `btc_web/test_mc_regime_paths.py` twice.
- **N-4** `test_price_registry_round_trip` / `test_months_for_cache_keys_on_asof` are Task 1 tests
  that need a registered series, but Task 1 does not say which one; the shared fixture proposed
  in I-1 should supply it.
