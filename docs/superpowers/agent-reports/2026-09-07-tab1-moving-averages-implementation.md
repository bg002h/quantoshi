# Tab-1 moving averages (`bub-ma`) — implementation report

**Date:** 2026-09-07 · **Branch:** `feat-tab1-ma` (worktree
`.claude/worktrees/tab1-ma`, branched from `master` @ `dc591a5`) ·
**Commit:** `2c25275` · **Not pushed, not deployed, not cherry-picked.**

---

## 1. What was built

A new `bub-ma` checklist inside the existing "Display" `_section_card` on Tab 1,
under a `<hr>` + "Moving averages" divider, drawing simple moving averages of
the daily close on the **Price view only**.

| checklist label | value | window | legend | dash |
|---|---|---|---|---|
| ` 200 week` | `ma200w` | 1400 days | `200W MA` | `solid` |
| ` 52 week` | `ma52w` | 364 days | `52W MA` | `dash` |
| ` 30 day` | `ma30d` | 30 days | `30D MA` | `dot` |
| ` 7 day` | `ma7d` | 7 days | `7D MA` | `dashdot` |

Default `value=["ma200w"]`. Styling matches the sibling `bub-toggles`
(`labelStyle={"display":"block"}`, `inputStyle=_CB_MARGIN`). `bub-toggles`
itself was **not** touched.

### New module — `btc_web/bub_ma.py`

A single import-light SSOT (numpy only, no Dash, no layout — the same shape as
`bub_views.py`), because four separate registries need the same facts and the
snapshot bitmask order is index-addressed:

- `MA_WINDOWS` — the frozen `NamedTuple` table above (value, label, days,
  legend, dash);
- `MA_VALUES` / `MA_BY_VALUE` / `MA_OPTIONS` — derived views;
- `rolling_mean(prices, window)` — O(n) cumulative-sum full-window mean;
  returns `NaN` for the first `window - 1` positions.

### Maths

Plain arithmetic mean of **daily closes**. Window lengths are **days**: the
conventional "200-week MA" is a 1400-day SMA of daily closes, *not* a mean of
200 weekly closes. Machine-verified that the price series is contiguous daily —
5,893 rows, 2010-07-17 .. 2026-09-03, `np.diff(dates)` is 1 day for all 5,892
steps — so one sample is one day and a plain window mean is exact. That check
is now a test (`test_daily_closes_not_weekly`), so it fails loudly if the
series ever gains a gap.

**Full window required.** Nothing is emitted before day *N*; the 200W line
starts at the first date with 1400 days of history behind it. Verified in the
browser: with `xmin=2011` the 52W line's first point is `t = 1.9713`
(2010-07-17 + 364 d), not the left edge of the x-range.

### Rendering (`figures/bubble.py`, immediately after the `show_data` block)

- one trace per selected window, drawn in `MA_WINDOWS` order;
- clipped to `[t_lo, t_hi]` **after** the mean is computed over the full
  series, so the earliest visible value does not depend on the x-range slider;
- multiplied by `stack` when `stack > 0`, exactly as the scatter does;
- downsampled with the scatter's own stride rule (`_MAX_SCATTER_PTS`): ~900
  points per line instead of ~5,900;
- `hovertemplate=_HOVER_FMT_USD`; `_add_date_hover` picks the traces up
  automatically and prepends the calendar date;
- gated only on `p["ma"]`, deliberately **not** on `show_data` — it is its own
  control;
- Price view only comes for free: `build_bubble_figure` serves only the Price
  view, and CAGR / Residuals / Percentile / Occupancy have separate builders
  that were not touched.

### Time Machine correctness (the point that mattered most)

`build_bubble_figure` previously derived the as-of frame boundary `t_D` *inside*
the scatter's `asof` branch. A trailing mean drawn past that date would reveal
prices the frozen model was never fit on, so the derivation was **hoisted to a
single `_t_D`** computed once, above the scatter, and both consumers read it:

```python
_t_D = None
_asof_date_str = None
if asof_idx is not None:
    import timemachine as tm
    _asof_date_str = tm.frames()[asof_idx]
    _t_D = (pd.Timestamp(_asof_date_str) - m.genesis).days / 365.25
```

The MA source series is cut with `_ma_x_all <= _t_D` **before** the rolling mean
runs. A second copy of that derivation is exactly how this leak would come back,
hence one variable rather than two identical expressions.

### Colour — dash pattern, not hue

`colors.py` gained exactly two **new** constants; no existing value in
`CB_BRIAN`, `CB_RG` or `CB_FULL` (or anywhere else) was modified:

- `MA_LINE_COLOR = "#455A64"` (Material Blue Grey 700) — shared by all four;
- `TRACE_WIDTH_MA = 1.6`.

Contrast was measured, not guessed (WCAG relative-luminance ratios):

| against | ratio |
|---|---|
| ivory plot ground `#FAF9F6` | **6.88 : 1** |
| BM gold `#C48209` | **2.26 : 1** |
| price scatter `#1A1A2E` | **2.36 : 1** |
| major grid `#E2E0DB` | 5.49 : 1 |

`#455A64` was picked from four candidates because it is the one that splits its
margin most evenly between the two things it must be told apart from. It is
also separated from the gold on the **blue/yellow** axis, which is the axis
red-green deficiency preserves. The *window* is encoded purely by dash pattern
(solid / dash / dot / dashdot), so the distinction survives every palette and
does not depend on hue discrimination at all.

`tools/generate_color_artifacts.py` was re-run; `_colors_generated.css` and
`_colors_generated.js` are in the commit.

---

## 2. Registry checklist — every item, and how it was verified

| # | Registry | Done | How verified |
|---|---|---|---|
| 1 | `tab_defaults.py` — `"ma"` in `_build_bubble_dict()` (sorted tuple) + promoted to a list in `bubble_defaults()` | ✔ | `bubble_defaults()["ma"] == ["ma200w"]` asserted in `TestCacheKeyAlignment`; the existing `test_defaults.py::test_inner_collections_are_tuples` also passes |
| 1b | `_prewarm_caches()` in sync | ✔ | Nothing to edit — it calls `bubble_defaults()` directly. Alignment proved by `test_cache_key_alignment.py` (whole-suite) plus a targeted assert that `"ma"` appears in the AST-extracted kwargs of **both** `_get_bubble_fig` and `_get_mc_bubble_fig` |
| 2a | `snapshot.py` — `("bub-ma","value")` appended at the **absolute tail** | ✔ | `_SNAPSHOT_CONTROLS[-1] == ("bub-ma","value")`; index 338 of 339. Also asserted that indices 336/337 (`bub-occ-tail`, `bub-occ-window`) and 334/335 (Time Machine) did not move |
| 2b | `_CHECKLIST_OPTIONS["bub-ma"]` with the four values in fixed order | ✔ | `== list(MA_VALUES)` asserted; bitmask round-trip parameterised over 5 selections. Hard-coded in `snapshot.py` (not `list(MA_VALUES)`) for the same reason `_QS_LIST` is — the bitmask layout of shipped links must not be able to move when the table does; the equality test is the drift guard |
| 2c | `_TAB_CONTROLS["bubble"]` (lives in `callbacks/routing.py`) | ✔ | `"bub-ma" in _TAB_CONTROLS["bubble"]` asserted |
| 3 | `snapshot_defaults.py` — `'bub-ma:value': ['ma200w']` | ✔ | Fingerprint pinned on **both** sides per the CLAUDE.md workflow: ran `tools/update_defaults_registry.py` before the edit (reported `3744c860 already in registry; no change`) and again after (`appended fingerprint b98ab5f7; registry now has 16 entries`). The live share link generated in the browser carried `q4:b98ab5f7:` |
| 4a | `callbacks/charts/__init__.py` — `Input("bub-ma","value")` on `update_bubble` | ✔ | AST test asserts `("Input","bub-ma")` is in the decorator sequence, **and** a second test asserts the decorator↔signature positional mapping (83 decorator args = 83 params; index 4 → `ma_sel`, neighbours still `toggles` / `bubble_toggles`). A silently mis-positioned Input is the classic failure here and no figure-level test would catch it |
| 4b | `"bub-ma"` in `_POST_RESTORE_TRIGGERS` | ✔ | AST-parsed out of the function body and asserted |
| 5 | `restore_builder.py` — threaded through the fast restore path | ✔ | Two tests: a state dict with `bub-ma:value == ["ma52w"]` produces exactly the `52W MA` trace; an empty state falls back to the `200W MA` default |
| 6 | `CLAUDE.md` | ✔ | Test count 2994 → **3042**, file count 57 → **58**, `test_bub_moving_avg` added to the inventory list, the Tab-1 row now describes the control, and `bub_ma.py` has a row in the file table. Counts come from `pytest btc_web/ --collect-only -q --ignore-glob='*_e2e.py'`, not by hand |

Cache-key note: the checklist value is `sorted()` in *both* `update_bubble` and
`restore_builder` (and stored sorted in `BUBBLE`), because a Dash checklist can
hand values back in click order and an unsorted list would miss L1/L2 for the
same visual state.

---

## 3. Tests — `btc_web/test_bub_moving_avg.py`, 48 tests

Every test was written **before** the code it covers and observed failing.
First run: `ModuleNotFoundError: No module named 'bub_ma'`. Second run (table +
maths implemented only): **34 failed, 13 passed** — the 13 being exactly the
maths/table group, the 34 being every rendering, as-of and registry assertion.
Final run: 48 passed.

### Maths (`TestRollingMean`, `TestMATable`)

| test | what it catches |
|---|---|
| `test_hand_computed_window_of_three` | wrong divisor / wrong alignment on a trivial case |
| `test_hand_computed_uneven_series` | an **off-by-one window** — deliberately not an arithmetic progression, because every sub-mean of a ramp equals its midpoint, so a window shifted by one still lands on a "nice" number and a ramp-based test passes |
| `test_full_window_rule_nothing_before_day_n` | a partial window being emitted; also pins the exact non-NaN count `n - w + 1` |
| `test_series_shorter_than_window_is_all_nan` | an out-of-range slice / spurious value when the history is shorter than the window |
| `test_window_of_one_is_the_series` | degenerate-window regression |
| `test_matches_naive_loop_on_the_real_price_series` | oracle test — cumulative-sum precision or indexing drift on the real 5.9k-row series, at three probe indices per window for all four windows |
| `test_daily_closes_not_weekly` | the whole premise: window lengths are days (1400/364/30/7) **and** the price series is still contiguous daily. If a gap ever appears in `BitcoinPricesDaily.csv`, "one sample == one day" stops holding and this fails |
| `test_values_and_order`, `test_legend_names`, `test_options_match_table` | table drift against the agreed spec |
| `test_windows_are_distinguished_by_dash_not_hue` | someone "fixing" the colours by giving each window its own hue |

### Rendering (`TestRendering`)

`test_zero_selected_draws_nothing` (`[]`, `None`, and key absent),
`test_one_selected_draws_one`, `test_n_selected_draws_n_with_the_right_names`
(2 and all 4, checking draw order), `test_unknown_value_is_ignored` (a stale
share link carrying a retired token must not raise),
`test_traces_are_lines_with_the_windows_dash` (+ all four share one colour),
`test_values_equal_the_rolling_mean_at_that_date` (every plotted point matched
back to its date's rolling mean — catches an off-by-one introduced by the
clip or the stride), `test_nothing_emitted_before_the_first_full_window` (and
no NaN hole punched mid-line), `test_clipped_to_the_x_range` (with a
non-vacuity check that a wider range really is wider),
`test_downsampled_like_the_scatter` (pins the exact stride-rule count and
asserts ≥4× reduction — catches shipping the full-resolution series),
`test_stack_multiplies_like_the_scatter`,
`test_stack_ignored_when_show_stack_is_off`,
`test_independent_of_show_data`, `test_hover_carries_a_date`.

### Time Machine (`TestAsOfTruncation`)

- `test_no_ma_point_past_the_frame_date` — the leak test: with a mid-range
  as-of frame, no MA x may exceed `t_D`.
- `test_live_view_does_draw_past_that_date` — the **non-vacuity** partner. Without
  it the leak test would still pass if the MA silently drew nothing at all.
- `test_values_still_correct_inside_the_frame` — truncating must not shift or
  corrupt the values that remain.

### Registries

`TestSnapshotRegistry` (tail index, previous tail indices unmoved, bitmask
option order, parameterised bitmask round-trip, full encode/decode round-trip,
empty-selection round-trip — which differs from the default so the sparse diff
must emit it — tab-control membership, default value), `TestCacheKeyAlignment`,
`TestCallbackWiring` (Input registered; decorator↔signature positional
alignment; `_POST_RESTORE_TRIGGERS` membership), `TestRestoreFastPath`,
`TestLayout` (exactly one `bub-ma` checklist with the agreed options and
default; `bub-toggles` options unchanged).

### Two pre-existing tests updated (legitimately invalidated)

- `test_occupancy.py::test_new_fields_appended_at_the_absolute_tail` — asserted
  the occupancy fields sat at `[-2]/[-1]`. Rewritten to **absolute indices
  336/337**, which is what the test actually means (those two must never move);
  the tail now belongs to `bub-ma`, pinned in the new file.
- `test_callbacks.py::TestUpdateBubbleCallback` (3 smoke tests) — call
  `update_bubble` by keyword; `ma_sel` is a required parameter, so each gained
  an explicit value (`["ma200w"]`, `[]`, `["ma52w","ma7d"]`). `ma_sel` was
  deliberately left without a default: a default would mask a positional
  mismatch.

---

## 4. Browser verification

Dev server on **port 8051** (`PORT=8051 DEV=1 bash run_web.sh`), Playwright
against `http://127.0.0.1:8051/1`. Port 8050 was never touched; the 8051 server
was killed afterwards.

1. **Default render** — `/1` loads with `bub-ma` showing the four labels
   ` 200 week` / ` 52 week` / ` 30 day` / ` 7 day`, only `ma200w` checked, and
   exactly one MA trace: `200W MA`, `dash: "solid"`, `color: "#455A64"`.
2. **All four toggle live** — turning the other three on gives
   `["200W MA","52W MA","30D MA","7D MA"]`; turning all four off gives `[]`;
   turning only `7 day` back on gives `["7D MA"]` with `dash: "dashdot"`.
   (Rendered width is 3.75 = `TRACE_WIDTH_MA` 1.6 × the 1.5 desktop multiplier
   from `chart_responsive.js`, as expected.)
3. **Share link round-trip** — selected `52 week` + `30 day`, 📸 Share →
   Generate link produced
   `http://127.0.0.1:8051/1#q4:b98ab5f7:H4sIAAAAAAAC_4tWSrK0SEwyTTNX0qlWMlOyitY1...`
   (note the **new** fingerprint `b98ab5f7`). Loading that URL fresh restored
   the checkboxes to exactly `ma52w`+`ma30d` **and** the chart to exactly
   `52W MA`+`30D MA`. Toggling further on the restored page continued to work.
4. **Wire format** — the traces arrive as `{dtype, bdata}` (numpy → base64),
   decoded by plotly to `Float64Array`; 899–922 points per line rather than
   ~5,900, confirming the stride is in effect on the real payload.
5. **Values** — final 52W MA value $82,712 and final 200W MA $64,795 against a
   last close near $81k: correct ordering and magnitude for trailing means of
   those lengths.
6. **Visual** — screenshot with all four on: the slate-grey lines read clearly
   against both the gold BM composite and the dark price scatter, and the four
   dash patterns are individually distinguishable. (Screenshots were temporary
   and deleted; they were not committed.)

Console errors seen in DEV are the pre-existing "nonexistent object … `State`"
messages for lazy-tab MC controls plus a price-ticker network failure — present
before this change and unrelated to it.

---

## 5. Gate

```
btc_venv/bin/python3 -m pytest btc_web/ -q
  1 failed, 3028 passed, 13 skipped, 18 warnings in 33.24s
```

The single failure is the known, pre-existing, out-of-scope
`test_callbacks.py::TestBTCPayPricing::test_free_tier_all_models`. No other
failure.

```
btc_venv/bin/python3 scripts/sabotage_callback_guards.py
  15/15 checks behaved as expected
```

15/15 both before the change (baseline taken first) and after.

---

## 6. Found but not fixed

**F-12 — "Unfairly Cheap Line" does not survive a share link.** Logged in
`docs/superpowers/followups.md` as Minor, clearly marked found-not-fixed, with
a one-line reproduction. `show_ucl` is an option in the Tab-1 Display checklist
and `figures/bubble.py` reads `p["show_ucl"]`, but it is absent from
`snapshot.py::_CHECKLIST_OPTIONS["bub-toggles"]`, so it is silently dropped on
encode. Verified before filing:

```
_decode_snapshot(_encode_snapshot({'bub-toggles:value': ['show_ucl','shade']}))
  -> {'bub-toggles:value': ['shade']}
```

The entry records *why* it is not a one-word fix: that list is index-addressed
by every shipped link's bitmask, so `show_ucl` must be **appended**, never
inserted beside its Display-card neighbours.

### Other observations, no action taken

- `test_downsampled_like_the_scatter` documents that the scatter's stride rule
  (`stride = n // 800`, then `arange(0, n, stride)`) can return slightly more
  than `_MAX_SCATTER_PTS` — 899 for the 200W MA, 818 for the scatter itself.
  Pre-existing behaviour of the rule the design said to reuse; the test pins
  the rule rather than a hard 800 cap.
- The `restore short-circuit` gate means a control change made
  *programmatically* (`set_props`) on a hash-restored page does not redraw
  until a genuine user event clears the gate. Confirmed the same for
  `bub-toggles`; real clicks work. Pre-existing, by design, not investigated
  further.

---

## 7. Files touched

New: `btc_web/bub_ma.py`, `btc_web/test_bub_moving_avg.py`.

Modified: `btc_web/colors.py`, `btc_web/assets/_colors_generated.{css,js}`,
`btc_web/figures/bubble.py`, `btc_web/layout/bubble.py`,
`btc_web/tab_defaults.py`, `btc_web/snapshot.py`,
`btc_web/snapshot_defaults.py`, `btc_web/snapshot_defaults_registry.json`,
`btc_web/callbacks/charts/__init__.py`, `btc_web/callbacks/routing.py`,
`btc_web/restore_builder.py`, `btc_web/test_callbacks.py`,
`btc_web/test_occupancy.py`, `CLAUDE.md`, `docs/superpowers/followups.md`.

Nothing was refactored beyond the `_t_D` hoist described above. The ticker, the
percentile view and the palette callbacks were not touched. Nothing was pushed
or deployed.
