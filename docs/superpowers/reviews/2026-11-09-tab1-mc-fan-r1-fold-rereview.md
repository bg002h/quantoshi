Critical: 0 · Important: 0 · Minor: 1 · Nit: 2

# Re-review of fold 7439454b (finding I-1, Tab-1 MC auto-widen)

Scope: only the 255-line fold. Verdict: **I-1 is closed; no new Critical or Important.**
Evidence labels: "ran" = executed here against the dev server on :8050 (which serves the folded
signature: 4 States confirmed in `/_dash-dependencies`) or under node; "read" = by reading code.

## M-1. A tick made outside the Price view is dropped for good, so the fan is a sliver on return

- Where: `btc_web/layout/bubble.py` `MC_AUTOWIDEN_JS` (the new `mode` guard); `bub-mc-exclude-wrap`
  is hidden only by Time Machine, so the MC checkbox is live in CAGR / Residuals / Percentile / Occupancy.
- What: the click stamp is consumed in the other view and nothing re-tries the widen on a view switch.
  Journey: open `/1.2`, tick MC, click the Price pill.
- Evidence (ran): `/1.2` after the tick has `bub-cagr-graph` with 5 traces, 0 MC. The hidden
  `bubble-graph` has 208 traces including `MC 5-95%`, `MC median` and `MC paths`, because
  `update_bubble` is not view-gated. After switching to Price the range is `[2010, 2033]`, so the
  fan shows only as a sliver at the right edge, the state the feature exists to avoid.
- Not a regression: before the fold, the CAGR default `[2025, 2050]` already no-opped
  (`2050 >= 2028+10`), and in the historical views the widen was the I-1 defect.
- Suggested fix, optional: record the missed widen and apply it on the next switch to Price, or
  accept and document. Does not gate.

## N-1. Unknown non-empty view mode suppresses the widen but the UI renders it as Price

- Where: `if (mode && mode !== "price") return NU` in `MC_AUTOWIDEN_JS`.
- `mode_styles` and the sync callback (`T[mode] || T["price"]`) treat an unknown mode as the
  default view. The new guard treats it as "not Price" and does nothing (ran under node: `"bogus"`
  gives NU, `""` and `null` widen).
- Not reachable today: the store holds only `VIEW_MODES` keys, and the layout default is `"price"`.
  Nit only. A fix would be `mode in VIEW_MODES ? mode !== price : false`, not needed.

## N-2. E2E `_open()` leaks the Playwright driver and browser if setup raises

- Where: `btc_web/test_mc_autowiden_e2e.py::_open`. The fixtures call `p.stop()` only after `yield`.
  A `goto` or `wait_for_selector` timeout inside `_open` skips both `browser.close()` and `p.stop()`.
  Effect is a stray firefox on a failed run. Nit.

## Probe answers

- **Nonexistent-object risk: none found.** `bub-view-mode` (`layout/common.py:605`, inside
  `_chart_tab_layout_with_fab`) and `cta-active` (`layout/custom_time.py`, inside
  `bub-cta-exclude-wrap`) are in the same bubble `content_fn()` tree as `bub-mc-enable`,
  `bub-xrange` and `bub-mc-start-yr`, so all five resolve together. Ran the lazy path:
  load `/6`, switch to Price Models, tick MC. The range went `[2010,2033]` to `[2010,2038]` with
  no callback or State error naming the new ids. `_HAS_MARKOV` false (read): neither store depends
  on it, and the placeholder `bub-mc-enable` cannot be clicked, so no stamp. Time Machine hides
  the wrapper by style, it does not unmount it.
- **"Falsy mode means widen" is right.** `bub-view-mode` is `dcc.Store(data="price")`, set by
  `toggle_bub_view`, `deep_link_bub_view` and restore to keys of `VIEW_MODES`. `null` appears only
  before hydration, and `DEFAULT_MODE` is the sensible reading then. Ran: `mode=null` and
  `mode="price"` both widen, `cta=null` widens, `cta=["yes"]` does not.
- **Stamp consumption: every exit path.** `window.__qsMcEnableClickAt = 0` runs before the first
  `return`. Ran a mutant that moves the reset below the mode guard: `test_a_click_in_another_view_is_still_consumed` fails.
- **Test teeth (ran, mutants built in the scratchpad, repo untouched).** Dropping the mode guard
  fails `test_other_views_are_left_alone` (all 3 historical modes) and the consumed test. Dropping
  the CTA guard fails `test_custom_time_axis_is_left_alone`. The `price_widens` cases stay green
  under both mutants, as they should. The three new unit tests are non-vacuous, and each input
  would otherwise yield `[2010, 2038]`. The E2E historical-view test asserts the checkbox became
  checked before asserting the slider is unchanged, so it cannot pass on a missed click. Browser
  controls (ran): Price view widens to 2038, `/1.2` unchanged `[2025,2050]`, CTA on unchanged
  `[2009,2033]`.
- **CAGR exclusion is correct.** `update_bub_cagr` has no `bub-mc-*` Inputs and draws no MC
  (ran, 0 MC traces). In that view `bub-xrange` means the exit-year window, so widening it would
  change the CAGR chart. The cost is M-1.
- **Comment fix in `callbacks/charts/__init__.py` is accurate** (read): `build_bubble_figure`
  ends `return fig, None`, so `mc_result` is always None on Tab 1. The CTA claim is also true:
  `update_bubble` raises `PreventUpdate` when `cta_active` contains `"yes"` (line 265).

## Checked and found sound

- Registration: 1 Input, 4 States in the pinned order; `prevent_initial_call=True`; no
  `allow_duplicate` + multi-Input hazard (States do not count).
- `%`-formatting of the JS: only the added `%(price)s`, no stray `%`; `DEFAULT_MODE` import from
  the import-light `bub_views` adds no cycle.
- The rename to `test_asset_and_callback_share_the_click_stamp` matches what that test asserts.

Out of scope, noted: ticking MC on `/1` logs `nonexistent object ... State dca-mc-years / ret- / sc- / cp-mc-years`
console errors from other tabs' MC callbacks. They do not name the fold's ids; presumed pre-existing (unconfirmed).
