# Re-review: fold of the F-24 / F-25 / Raw redraw cache review (`e69dfd49`)

**Counts: 0 Critical · 0 Important · 0 Minor · 1 Nit**

Scope: the range `9df1aa0f..e69dfd49` does not resolve in this repo, so I reviewed
`git show e69dfd49` (the fold; four source files, the `model_data_ef.pkl` and docs
excluded). Question asked: does each fold change resolve its finding without adding a
defect? Source was not edited. Every prior finding in
`2026-11-10-f24-f25-raw-cache-review.md` is resolved, and the fold adds no defect. The
only new item is a Nit about the M-1 test, and its root cause is old unchanged code.

| Finding | Verdict |
|---|---|
| I-1 (double chart POST on bins raise) | Resolved |
| M-1 (float32 times a large stack) | Resolved as filed. See N-3 for a residual limit |
| M-2 (vacuous cache-clear test) | Resolved |
| N-1 (unlocked `simulate`) | Resolved (docstring only, and it is accurate) |
| N-2 | Left on purpose, still cosmetic |

---

## I-1: resolved

**(a) No other plain writer of any `*-mc-regime.value`.**
`app._callback_list` at import time omits every `@callback(...)` registration,
because Dash merges `GLOBAL_CALLBACK_LIST` only in `_setup_server`. That includes all
`apply_tab_*` and `apply_globals`. A check on the import-time list alone would miss
exactly the restore callbacks. I therefore pulled the full list from the Flask test
client (`GET /_dash-dependencies`, 360 entries, the same data the renderer gets) and
split multi-outputs.

For each of the seven ids (`dca ret hm sc cp bub`-`mc-regime.value` and
`bub-mc-raw-regime.value`) the result is identical:

- Exactly one plain writer: the clientside bins tick.
- Exactly one `allow_duplicate` (`@hash`) writer: the matching `apply_tab_*` restore
  callback.
- Nothing else. `set_props` writes bypass the graph, and the `.options` callbacks write a
  different property.

Across all 360 callbacks, no output id has more than one plain writer. Dash's Python side
does not check this (`validate_duplicate_output` only polices `allow_duplicate` together
with `prevent_initial_call`), so the renderer would be the one to raise. The graph shows
nothing for it to raise on.

Reachability over input-to-output edges shows no cycle through any regime node. The
chain is bins, then tick, then regime, then (`bub-mc-paths` tick and note) then chart.
The tick is one hop ahead of the chart callbacks, as intended.

The comment in `callbacks/mc_controls.py` was updated, and `test_shape` correctly flipped
to `key == f"{regime}.value"`. `layout/mc_controls.py:238` mentions allow_duplicate, but it
describes the paths callback and is not stale.

**E2E on the changed graph.** The fold's gate ran raw_mc, restore_direct and
restore_phase1. The earlier review also named the per-tab restores, which share this
graph. I ran those against a DEV server on `e69dfd49`:
`restore_phase2_{dca,retire,heatmap,supercharge,citadel,leverage}`,
`mc_regime_paths`, `mc_autowiden` and `raw_mc`. **42 passed** (`-p no:randomly -n0`).

A first run on a cold server had 5 failures: four in `restore_phase2_dca`, plus one
timing assert in `restore_phase2_retire` (7996 ms against a 5000 ms limit). They are not
attributable to the fold:

- The same files re-run to 11/11 and then 42/42.
- A second cold restart on the fold passed 6/6 for the DCA file.
- A faithful pre-fold tree (`42fc45f2` with `mc_cache`, `.env` and the markov `.so` in
  place) on a cold server failed `test_dca_mc_share_falls_back` with the same
  `wait_for_function` timeout.

These tests are cold-server timing flakes (no prewarm under `DEV=1`), present before the
fold.

## M-1: resolved as filed

`np.multiply(paths, stack_f, dtype=np.float64)` allocates a new float64 array per
request. The float32 cache stays read-only and untouched, and the transient exists only
when "Show stack" is on. This matches the numerics before the cache diff. The test is
non-vacuous: it uses a fake 1e32 result, and the controller confirmed it fails on the
float32 mutant.

### N-3 (Nit): float64 does not make paths at 9.2e18 or above drawable, and the test cannot see that

`btc_web/test_raw_mc_callback.py::test_huge_stack_does_not_overflow` asserts
`np.isfinite(y).all()`. That correctly pins "no float32 inf", but it passes on garbage.
`figures/bubble.py:125` (`_sig_round_list`, unchanged by this diff) takes the
`out.astype(np.int64)` branch whenever every value in the row is finite and at least 100.
Any value of about 9.22e18 or more then becomes `-9223372036854775808`. Direct check:

```
1e19 -> -9223372036854775808   (RuntimeWarning: invalid value encountered in cast)
1e32 -> -9223372036854775808
1e40 -> -9223372036854775808
[1e5, 1e32, 3e40] -> [100000, -9223372036854775808, -9223372036854775808]
```

The test run itself emits that RuntimeWarning from `bubble.py:125`, which shows the
scenario reaches the cast. So for such paths, M-1's visible symptom (gaps in "Highest
end" and example lines) is not cured. The null becomes a negative value, which a log
axis does not draw. This is not caused by the fold. The earlier review's worst case,
a 1.1e32 raw path, is above 9.2e18 even with no stack. M-1's statement that float64
"could not overflow" was true for float arithmetic but missed this downstream cast.
It only needs absurd inputs (trend "none", 40 years, 3200 sims, or a stack of millions
of BTC), and nothing wrong appears on a normal run.

Suggested follow-up, ownerless residue. In `_sig_round_list`, take the int path only when
`np.abs(out).max() < 2**53`; otherwise fall through to the list branch. Then add a
`> 0` assertion to the huge-stack test so INT64_MIN fails.

## M-2: resolved

After `_run()` the test asserts `_raw._cache` is non-empty. It then calls
`raw_mc.set_prices(*raw_mc.get_prices())` and asserts `not _raw._cache`.
`set_prices` goes through `_set_registry`, which runs every entry in `_CACHE_CLEARERS`
unconditionally. Un-registering `clear_cache` would fail both the membership assert and
the emptiness assert. The test now checks what its name claims. It re-registers the same
series rather than a different one, but that takes the same code path, since
`_set_registry` never compares fingerprints.

## N-1 (prior): resolved

The docstring note on `_simulate_cached` is accurate. `run_web.sh` starts gunicorn with
`--workers 5` and no `--threads` or `--worker-class`, so workers are sync.

---

Housekeeping: I left a DEV server running on :8050 from the working tree at `e69dfd49`.
The scratch pre-fold tree is in the session scratchpad. Nothing in the repo was edited
other than this report.
