# Review: cbecb30e "MC example paths ship every other point at 2 significant figures"

**Verdict: GREEN. 0 Critical / 0 Important / 4 Minor / 3 Nit.** `_add_mc_spaghetti` draws the same
example paths as before (same sample, same stack scaling, same right-edge truncation), thinned to every
2nd step plus the last and rounded to 2 significant figures, and every Critical/Important-class mutation
of the feature is killed by at least one test. The Minors are an out-of-range-input edge and test-strength gaps.

Scope: `git diff afb60c42..cbecb30e` (`btc_web/figures/bubble.py`, `btc_web/test_figures.py`,
`btc_web/test_callbacks.py`). Reviewed in an isolated detached worktree of `cbecb30e` (removed afterwards);
the shared tree was not touched. Note the shared tree's HEAD has since moved to `25ae8293` and carries
uncommitted edits to `bubble.py`, `callbacks/charts/__init__.py` and others; none of that is covered here.

Both new test files on the pristine worktree: `1 failed, 418 passed, 5 skipped`; the 1 failure is the known
local-only `TestBTCPayPricing::test_free_tier_all_models`.

---

## Critical

None.

## Important

None.

## Minor

### M-1  `_sig_round_list` silently wraps to INT64_MIN at |v| >= 9.22e18 (and can crash the JSON encoder above 1.8e19)

`btc_web/figures/bubble.py:117` (`return out.astype(np.int64).tolist()`); mixed branch at :118-119.

The all-whole-numbers fast path casts to int64. Values >= 2^63 wrap to `-9223372036854775808` with only a
RuntimeWarning. Reachable only with an absurd Stack (BTC) value: the cached 40-yr scenario's max path is
4.39e7, and `bub-stack` is `min=0, step=0.001` with no max (`layout/bubble.py:579`), so it takes a typed stack
above roughly 2e11 BTC (about 10,000x the supply). Before this commit the same input produced large float32
lines; now the lines silently become negative garbage. The mixed branch returns Python ints without an int64
bound, so `1e300` raises in `to_json_plotly` (`TypeError: Integer exceeds 64-bit range`).

Reproduction (real `update_bubble`, `mc_sims=8`, `show_stack=["yes"]`):

```
stack 1e12  -> min y 68000000000000000      max y 2000000000000000000   serialise OK
stack 1e13  -> min y -9223372036854775808   max y 9200000000000000000   serialise OK   <- wrapped
```
```
_sig_round_list([1.5e19, 2.0])      -> [15000000000000000000, 2]        (ok, < 2^64)
_sig_round_list([1.5e19])           -> [-9223372036854775808]           RuntimeWarning: invalid value encountered in cast
_sig_round_list([1.0, 5.0, 1e300, 5.5]) -> to_json_plotly: TypeError: Integer exceeds 64-bit range
```

Fix: gate the integer conversions on `np.abs(out[safe]).max() < 2**53` and otherwise return `out.tolist()` (floats
serialise as `1e+19`, valid JSON). Optionally also clamp the stack input at the layout level.

### M-2  The "no float noise" assertion cannot fail under the regression it names

`btc_web/test_figures.py:1387-1388`.

`assert repr(ys[1][0]) == "7.4" and repr(ys[1][1]) == "0.037"` is meant to guard the docstring claim that
fractions come back noise-free (division by an exact power of ten, not multiplication by a reciprocal). Both
chosen values happen to be clean under either implementation (`74*0.1 == 7.4`, `37*0.001 == 0.037`).

Mutation (worktree only): replace `* down / up` by `* down * 10.0 ** -np.maximum(dec, 0)`:

```
mutant _sig_round_list([1.23, 0.0123, 0.35, 0.17, 74.0]) -> [1.2000000000000002, 0.012, 0.35000000000000003, 0.17, 74]
pytest -k "add_mc_spaghetti or TestUpdateBubbleCallbackMC"  -> 23 passed
```

The function itself is correct: a sweep of 800,124 values (log-uniform 1e-3..1e13 as float64 and float32, plus
decade edges) found 0 results with more than 2 significant digits in `repr`. Practical impact is nil on real
paths (fractions only appear for stacks below about 1e-4 BTC), hence Minor. Fix: add values that are noisy under
the reciprocal form, e.g. `1.23 -> 1.2`, `0.35 -> 0.35` (`35*0.01 == 0.35000000000000003`), `0.17`, `0.0123 -> 0.012`.

### M-3  Stack-scaling tolerance (rtol=0.11) hides scaling errors up to about 5%

`btc_web/test_callbacks.py:1428-1433`.

The 0.11 bound is mathematically justified (two independent 2-sf roundings differ by at most about 10%; I
computed 9.5/0.952 = 10.0%), and the test is killed by a missing x2.5 (60% off). But it is blind to a small
factor error:

```
mutation: line_paths = line_paths * stack_f * 1.04     -> 23 passed (not detected)
mutation: stack not applied to the lines               -> test_fan_scales_with_stack_like_the_example_lines FAILED
```

Fix: compare against the exact expectation instead of line-vs-line, e.g. build the expected line from the raw
cached sample (`raw[::25][:8][:, :keep][:, steps]`, `*2.5`) and assert
`|got - expected| / expected <= 0.0477` (the 2-sf worst case), which rejects any factor error above about 0.3%.

### M-4  The gzip-ratio test is a loose backstop; it does not discriminate stride 1 or a rounded numpy array

`btc_web/test_figures.py:1429` (`assert got < 0.5 * was`).

Measured on the test's own fixture (float32 reference = 102,614 B):

```
stride2+round+list (shipped)    19387  ratio 0.189  PASSES <0.5
stride1+round+list              34455  ratio 0.336  PASSES <0.5     <- stride regression not caught here
stride2+round+f32 numpy array   25090  ratio 0.245  PASSES <0.5     <- numpy-array regression not caught here
stride2+no round f32 array      52095  ratio 0.508  fails
```

The other tests do catch both (see matrix below), so there is no hole; but this test's name promises more than it
checks. Fix: tighten to `< 0.3 * was` (rejects stride 1; still 1.6x headroom over the shipped 0.19-0.20).

---

## Nit

- **N-1** `test_figures.py:1369-1370` docstring says a numpy array's base64 bytes cannot be shrunk by rounding.
  Measured: for 2-sf prices >= 1e4 a rounded float32 array gzips to 25.1 KB versus 52.1 KB unrounded, because
  2-sf whole numbers leave trailing-zero mantissa bits. The plain list (19.4 KB) is still the smaller choice, so
  the design stands; only the wording is too absolute.
- **N-2** `test_add_mc_spaghetti_rounding_survives_odd_values` (:1391) is described as covering non-finite
  values but feeds only NaN. I checked inf: `_sig_round_list([nan, inf, -inf, 1.0])` -> `[nan, inf, -inf, 1]`,
  serialised as `null,null,null` (valid JSON).
- **N-3** Values below about 1e-21 are not exact (`10.0**dec` loses exactness) and denormals return NaN
  (`_sig_round_list([1e-310]) -> [nan]`, overflow warning). Unreachable for prices.

(Tie handling note, not a finding: `0.995 -> 1` and `9.95 -> 10` where an exact-decimal oracle says `0.99`/`9.9`,
a double-rounding tie artefact; both are within 5% and still 2 significant figures.)

---

## Mutation matrix (worktree copy; each mutation restored from a pristine copy)

Command per row: `pytest btc_web/test_figures.py btc_web/test_callbacks.py -k "add_mc_spaghetti or TestUpdateBubbleCallbackMC" -n0`.

| Mutation | Killed by |
|---|---|
| (a) `MC_PATH_POINT_STRIDE = 1` | `keeps_every_other_point`, `rounds_prices_to_two_sig_figs`, `rounding_survives_odd_values` |
| (b) rounding removed, y = `path.tolist()` | `rounds_prices...`, `rounding_survives_odd_values`, `payload_is_under_half_of_float32` |
| (b2) rounding removed, y = float32 array | same three |
| (c) last step dropped | `keeps_every_other_point`, `TestUpdateBubbleCallbackMC::test_mc_traces_stop_at_the_visible_right_edge` |
| (d) y = rounded float32 numpy array | `rounds_prices...` only (via `type(v) is int`) |
| (d2) y = rounded float64 numpy array | `rounds_prices...` only |
| whole numbers shipped as floats (`75000.0`) | `rounds_prices...` (both fast and mixed branch) |
| steps off by one (`range(1, n, 2)`) | `keeps_every_other_point`, `rounds_prices...`, `rounding_survives_odd_values` |
| x not thinned (x/y length mismatch) | `keeps_every_other_point` |
| `MC_PATH_SIG_FIGS = 3` | `rounds_prices...`, `rounding_survives_odd_values` |
| fan computed from the drawn lines | `test_fan_is_computed_from_every_cached_path...`, `test_regime_filter_on_cached_scenario...` |
| stack not applied to the lines | `test_fan_scales_with_stack_like_the_example_lines` |
| stack applied with a 1.04 error | NOT killed (M-3) |
| float noise (reciprocal multiply) | NOT killed (M-2) |

Requested items (a)-(d) are all killed. (d) rests on a single assertion (`type(v) is int`), which is sound but
thin; (a) and (c) each have two or more independent killers.

## Verified OK (commands run)

All against `cbecb30e` in the isolated worktree, `btc_venv/bin/python3`.

1. **`_sig_round_list` values.** Direct calls:
   `[0.01,0.0123,0.5,1,7.44,74550,123456,1.26e6,1e7,12345678.9,1.5e9] -> [0.01,0.012,0.5,1,7.4,75000,120000,1300000,10000000,12000000,1500000000]`;
   `[99999, 9.95, 9.949999, 0.0995, 99.5, 994.9, 995, 9.5, 0.095, 99500.0, 99499.99] -> [100000, 10, 9.9, 0.1, 100, 990, 1000, 9.5, 0.095, 100000, 99000]`
   (rounding up across a decade stays sane); `[0.0, -0.0, -1234, -0.0123, -99999] -> [0, 0, -1200, -0.012, -100000]`;
   float32 row `[74550.3, 0.037, 99999, 7.44, 123456.7] -> [75000, 0.037, 100000, 7.4, 120000]`; empty list -> `[]`.
2. **Sweep vs `float(f"{x:.1e}")` oracle**, 800,124 values (float64 and float32, 1e-3..1e13, decade edges and
   9.5/9.95 x 10^k): 2 mismatches, both exact-decimal ties (see note); 0 results with more than 2 digits; worst
   relative error vs input 4.76% (< 5%).
3. **Output typing and JSON.** Real `update_bubble` run: all 200 lines' y are `int`; x is `float32` (`dtype f4` in
   JSON). `json.loads(to_json_plotly(fig))` OK; `[0, nan, inf, -inf, 7.44, 75000, -1234] -> y:[0,null,null,null,7.4,75000,-1200]`.
   Dash's serialiser is `plotly.io.json.to_json_plotly` (`dash._utils.to_json`); `pio.to_json` with engines `json` and
   `orjson` both succeed for NaN/inf/ints below 2^64.
4. **`_mc_path_steps`** for n=0..10: `[]`, `[0]`, `[0,1]`, `[0,2]`, `[0,2,3]`, `[0,2,4]`, `[0,2,4,5]`, `[0,2,4,6]`,
   `[0,2,4,6,7]`, `[0,2,4,6,8]`, `[0,2,4,6,8,9]`: sorted, unique, in range, last step always present. Through
   `_add_mc_spaghetti`, x and y lengths are equal and the last x is 1.0 for n=1..11.
5. **Same paths as before.** Real scenario (`bub`, 2028, 10th pct, 40 yr, `mc_sims=200`, cached shape 200x481,
   float32): lines have 73 points over a 145-step fan (steps 0,2,...,144); each line equals the original sampled
   path (`raw[::stride][:200][:, :145][:, steps]`) with max relative error 0.0476 (2-sf bound); last x equals the fan's
   last x (30.43668747 float32 vs 30.43668720 float64). Whole-figure gzip: 72.7 KB with 200 paths vs 50.5 KB with
   `mc_paths=[]`.
6. **Callers (check 3).** `git grep` finds exactly one caller: `btc_web/callbacks/charts/__init__.py:557`. Nothing
   reads spaghetti trace y: `chart_responsive.js:88-91,119` reads only `meta.qs_keep_width` and `line.width`;
   `sc_legend.js` is bound to `supercharge-graph` only; `routing.py:533` reads the CAGR chart's marker trace;
   `figures/common.py:1118,1135` (`_clip_mc_traces`, `_find_mc_median_trace`) operate on the shared-fan traces, not on
   spaghetti; `_add_date_hover` skips `hoverinfo == "skip"` traces (`figures/common.py:913`); `_mc_finalize` and
   `apply_zoom_lock` touch layout only; `restore_builder.py` does not build MC traces. Time Machine, hover and
   legend-group toggles have no reader of the `mc-spaghetti` y arrays (legendgroup is read only by tests and set in
   `bubble.py:163`).
7. **Scaling order (check 4).** `callbacks/charts/__init__.py` scales `line_paths * stack_f` before calling
   `_add_mc_spaghetti`, and rounding is the helper's last step, so scale-then-round holds; the right-edge truncation
   (`keep`) also happens before thinning, and the last kept column is always shipped. Stack 1, 21e6, 1e9, 1e12 serialise
   with whole-dollar ints (1e13: see M-1).
8. **Test run.** `pytest ... -k add_mc_spaghetti` on the pristine worktree: 9 passed; with `TestUpdateBubbleCallbackMC`:
   23 passed; both full files: 1 failed (known), 418 passed, 5 skipped.
9. The rtol=0.06 bound in `test_fan_is_computed_from_every_cached_path_not_the_displayed_few` is justified (median
   of 8 rounded paths differs from the median of 8 exact paths by under 4.8%) and still kills "fan computed from the
   drawn lines" (matrix row above).

## Not covered

- The shared working tree's current state (HEAD `25ae8293` plus uncommitted edits to `bubble.py`,
  `callbacks/charts/__init__.py`, `mc_controls.py` and others) and anything after `cbecb30e`.
- Browser-side rendering (plotly.js handling of mixed `bdata` x with plain-list y, visual lattice) and E2E suites;
  the operator has approved the look from screenshots.
- The live (paid) MC arm, which cannot reach Tab 1 today (F-15), and `mc_sims > 200`.
- Prod serialisation if `orjson` is absent: checked both plotly engines locally, not the prod environment.
- Documentation drift (CLAUDE.md still describes the older RdYlGn spaghetti colouring; pre-existing, unrelated).
