# Follow-up burndown — integration review (98cbb4e..5a2c71f)
Reviewer: opus subagent · Date: 2026-09-05

Commits under review:

```
5a2c71f perf(tab1): Occupancy strip hover carries numbers, not rendered text (1.86x smaller)
294a387 refactor(tab1): one VIEW_MODES table drives the pill toggle, clientside sync and deep links
ab963e6 perf(tab1): hidden CAGR / Residuals / Percentile views no longer rebuild on every control change
2896100 docs: refresh the CLAUDE.md test inventory (2887 tests, 50 + 12 files)
72eb16f ops: move the monthly PPL refit from prod to dev (tools/monthly_refit.sh)
98cbb4e chore(btc_core): adopt the 2026-09-01 monthly PPL refit from prod
```

## Verdict: 0 Critical / 0 Important -> GREEN — counts: C=0 I=0 M=2 N=4

Safe to push and deploy. No commit produces a wrong result; no existing Tab-1
view, pill, deep link or share-link restore is broken; the daily auto-deploy
and the new monthly refit job interlock correctly; and the two files two agents
touched in parallel (`btc_web/callbacks/charts/__init__.py`,
`btc_web/test_occupancy.py`) carry both agents' changes intact — the commits
landed sequentially (294a387's diff header is `index 810c2e2..68312df`, i.e.
built on ab963e6's post-image), not as a merge, so there was no reconciliation
to get wrong.

## Findings

### F1 — Occupancy hover label is not byte-identical: 0.67 % of shipped share values move by one digit  [Minor]

**Commit** 5a2c71f · **File:line** `btc_web/figures/occupancy.py:191-196`

**Claim.** Commit message: *"Rendered label is byte-identical (`<b>BM</b> · Aug
14, 2019 / ≥Q90 8.6% · ≤Q10 6.4% / of trailing 4 yr`)"*. Agent report §Headline:
*"with the rendered label byte-identical"*.

**Evidence.** Old code baked the label with Python's `format(v, '.1f')` on the
**raw** share; new code pre-rounds with `np.round` and lets d3 format the
result:

```python
                    customdata=np.round(np.column_stack(
                        (above[pos[hit]], below[pos[hit]])), 1),
                    hovertemplate=(
                        f"<b>{name}</b> · %{{customdata[0]}}<br>"
                        f"≥Q{q_hi} %{{customdata[1]:.1f}}% · "
                        f"≤Q{tail} %{{customdata[2]:.1f}}%"
```

`np.round` scales by 10 and applies round-half-to-**even** on the scaled value;
Python's `.1f` rounds the original double correctly. They disagree at exact
half-boundaries. Measured on real model output (all `above`/`below` line values
for tails 5/10/25 × windows 1/2/4 yr × 8 models, from the live `model_data.pkl`):

```
compared 725808 raw share values; label mismatches: 4871
[('bub', 5, 1, 7.65, '7.7', '7.6'), ...]
```

i.e. a day whose true share is 7.65 % used to hover as `7.7%` and now hovers as
`7.6%`. (Synthetic grid check: 303 mismatches in 1,000,001 values on
`arange(0, 100, 1e-4)`.)

**Failure scenario (concrete).** Operator opens Tab 1 → Occupancy, hovers a day
whose trailing-4-yr above-Q90 share is exactly 7.65 %, and reads `7.6%` where
the pre-deploy build read `7.7%`. Nothing else changes; both are defensible
roundings of the same number and the underlying line trace is untouched.

**Why not Important.** No computation is wrong, no view/link/pipeline is
affected, and the difference is confined to the last displayed digit of a
tooltip. The defect is in the *claim*, which was carried into the commit
message and the agent report and could mislead a future bisect.

**Suggested fix (one sentence).** Either drop the byte-identity claim, or
replace `np.round(...)` with a decimal-correct pre-round
(`np.array([float(format(v, '.1f')) for v in ...])`, or keep float and let d3
round the raw value).

### F2 — `monthly_refit.sh`'s disable-file lock has no staleness guard; a hard kill silently stops the daily auto-deploy indefinitely  [Minor]

**Commit** 72eb16f · **File:line** `tools/monthly_refit.sh:58-64` and
`daily_update.sh:36-39`

**Claim.** Commit message: *"It holds `/tmp/quantoshi-update.disable` while
running so it can never collide with the daily job in the same worktree."*

**Evidence.**

```bash
if [[ -f "$DISABLE_FILE" ]]; then
    echo "$DISABLE_FILE already present — another job holds the worktree; exiting"
    exit 0
fi
run touch "$DISABLE_FILE"
trap 'run rm -f "$DISABLE_FILE"' EXIT
```

and the consumer, which exits successfully and silently:

```bash
if [[ -f /tmp/quantoshi-update.disable ]]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') — daily update disabled via /tmp/quantoshi-update.disable — exiting" >> "$LOG"
    exit 0
fi
```

Nothing anywhere age-checks the file. I verified empirically that bash **does**
run the `EXIT` trap on SIGTERM (`kill -TERM` → exit 143, lock removed) and does
**not** on SIGKILL (lock survives), so the ordinary terminations are covered:
`systemctl --user stop`, session shutdown, and the unit's own
`TimeoutStartSec=4h` expiry all deliver SIGTERM first. `/tmp` is `tmpfs`
(31.3 G), so a reboot also clears a leaked lock.

**Failure scenario (concrete).** The refit is OOM-killed (it runs
`refit_all_ppl.py` for 60–90 min alongside a dev session on the same box) or
`kill -9`'d, the machine is *not* rebooted, and `/tmp/quantoshi-update.disable`
persists. Every subsequent 06:00 `quantoshi-update.service` run exits 0 —
systemd reports success, no `notify-send`, no journal error, only one line in
`/tmp/quantoshi-daily-update.log`. Prod's price data and any pushed commits stop
deploying until someone notices (the repo has a recorded 3-day silent-stall
incident of exactly this shape; see `daily_update_lag1_autodeploy.md`).

Narrow related case: the trap deletes the lock **unconditionally**, so if the
operator touches the disable file to pause the daily job during the ~90-minute
refit window, the refit's exit silently re-enables it.

**Suggested fix (one sentence).** Have `daily_update.sh` treat a disable file
older than ~6 h as stale (log loudly + `notify-send` + proceed), or switch the
refit to `flock` on the worktree so the lock dies with the process.

### F3 — `git clean -fdq btc_core/` no longer removes the `.bak` files it exists to remove  [Nit]

**Commit** 72eb16f · **File:line** `tools/monthly_refit.sh:72` + `.gitignore:83-84`

**Evidence.** The same commit adds

```
# fit tools leave backups next to the patched module; never commit them
btc_core/*.bak
```

and the script's step 1 ends with

```bash
run git clean -fdq btc_core/          # stray .bak from an aborted run
```

`git clean` without `-x` skips **ignored** files, so from this commit forward
that line cannot remove a `btc_core/*.bak`.

**Failure scenario (concrete).** An aborted refit leaves `btc_core/_lppl.py.bak`
in the deploy worktree; the next month's `git clean -fdq btc_core/` leaves it,
contrary to its comment. Harmless in practice: `rm -f btc_core/*.bak` (line 88)
still removes it after the refit, `git add 'btc_core/*.py'` cannot pick it up,
and — importantly — the new ignore rule is what stops `daily_update.sh`'s
`git add btc_core/` from committing and deploying a stray `.bak`, which is the
rule's real value.

**Suggested fix.** `git clean -fdxq btc_core/`, or amend the comment.

### F4 — the new `btc_web/markov.cpython-*.so` ignore pattern also matches a tracked file  [Nit]

**Commit** 72eb16f · **File:line** `.gitignore:85-86`

**Evidence.**

```
$ git ls-files -i -c --exclude-standard
btc_web/markov.cpython-314-x86_64-linux-gnu.so
quantoshi_logo.xcf
```

(`quantoshi_logo.xcf` was already in this state before the commit.)

**Failure scenario (concrete).** `.gitignore` does not affect files already in
the index, so nothing changes today: the tracked dev-built `.so` still diffs,
still deploys, and prod's own `markov.cpython-312-*.so` is correctly ignored —
which is what the rule was for. The one live consequence is that if the
dev `.so` is ever removed and rebuilt, `git add btc_web/markov.cpython-314-*.so`
is refused without `-f`, so a `build_markov.py` refresh could be silently
dropped from a commit.

**Suggested fix.** Either `git rm --cached` the tracked 3.14 `.so` (prod builds
its own and cannot load a 3.14 module anyway), or narrow the pattern to the
Python versions prod builds.

### F5 — CLAUDE.md's test inventory is already stale at HEAD  [Nit]

**Commit** 2896100 · **File:line** `CLAUDE.md:70` and `CLAUDE.md:190`

**Evidence.** The line reads *"2887 tests across 50 non-E2E files"* plus *"12
`*_e2e.py` files"*. Machine-checked against HEAD:

```
listed non-e2e: 50   listed e2e: 12
actual non-e2e: 53
in repo not listed: ['test_bub_view_gating', 'test_bub_view_modes', 'test_date_hover']
listed not in repo: []
```

so 2896100 was factually correct **when written** (50/12, and its 50 names match
the files that existed then), and the three commits stacked on top of it added
three test files without updating it. The commit's own count of 2887 is
likewise superseded by 5a2c71f's message (2942).

**Failure scenario (concrete).** Documentation only — a reader trusting
CLAUDE.md under-counts the suite by 3 files / ~58 tests.

**Suggested fix.** Add the three names and refresh both numbers in the ship
commit, or leave it for the next inventory pass.

### F6 — every step-1 failure in `monthly_refit.sh` exits silently  [Nit]

**Commit** 72eb16f · **File:line** `tools/monthly_refit.sh:69-79`

**Evidence.** `notify_failure` is called on exactly three paths — refit failure,
gate failure, push failure. Everything before that is under bare `set -eo
pipefail`: `cd "$DEPLOY_DIR"`, `git fetch`, `git checkout -f --detach`, `git
reset --hard`, `git clean`, and the symlink loop

```bash
for d in btc_web/mc_cache btc_web/citadel_band_cache; do
    if [[ -d "$MAIN_DIR/$d" && ! -e "$DEPLOY_DIR/$d" ]]; then
        run ln -s "$MAIN_DIR/$d" "$DEPLOY_DIR/$d"
    fi
done
```

**Failure scenario (concrete).** `btc_web/mc_cache` is deleted from the main
checkout while the deploy worktree still holds the symlink to it; `-e` follows
the broken link and is false, `-d "$MAIN_DIR/$d"` is false too so the loop is
skipped — but if only the *target* is restored under a different name the `ln
-s` fails "File exists", `set -e` exits 1, the trap releases the lock, and the
month's refit silently does not happen. Same shape for a transient `git fetch`
failure at 01:00. The only trace is `/tmp/quantoshi-monthly-refit.log`, which
nothing watches. Because the deploy pipeline itself recovers (the daily job
resets the worktree at 06:00), this costs a month of parameter freshness, not
correctness.

**Suggested fix.** Add `trap 'rc=$?; [[ $rc -ne 0 ]] && notify_failure "monthly refit aborted (exit $rc)"; run rm -f "$DISABLE_FILE"' EXIT`, and test `! -L` rather than `! -e` in the symlink loop.

## Checked and clean

**1. Cross-commit interaction in `btc_web/callbacks/charts/__init__.py`.**
Both changes are present at HEAD and neither clobbered the other. ab963e6's
gates survive verbatim at lines 691 / 757 / 810 (`view_mode != "cagr"`,
`!= "resid"`, `!= "percentile"`), alongside the pre-existing
`update_bub_occ` gate at 862 (`!= "occupancy"`) — and all four strings are exact
`VIEW_MODES` keys (`price`, `cagr`, `resid`, `percentile`, `occupancy`). All
four callbacks still carry `Input("bub-view-mode", "data")` as their first
Input and `State("snapshot-pending", "data")` last, so switching pills, deep
links and snapshot restores all re-trigger the build; the gate sits *after* the
snapshot gate, preserving the single-redraw contract. No Input or Output was
lost: `toggle_bub_view` still registers its 16 Outputs / 5 Inputs / 1 State, the
clientside sync still registers its 14 `allow_duplicate` Outputs, and
`deep_link_bub_view` still registers 20 — pinned by
`test_bub_view_modes.py::test_output_order_is_the_table_order` (×3 producers),
which reads the real `app.callback_map` keys rather than the source. Targeted
run: `test_bub_view_modes.py test_bub_view_gating.py test_occupancy.py
test_date_hover.py test_bub_deep_links.py` → **109 passed, 2 skipped**.
I also confirmed the gate cannot strand the CAGR progress bar for a different
reason than the report gives: `bub-cagr-progress-wrap` is a **child of
`bub-cagr-wrap`** (`layout/common.py:641` inside the div opened at `:633`), so
even a bar left running is hidden with the view and is reset to 0 % by the SHOW
callback on the next switch into CAGR.

**2. `btc_web/bub_views.py` against the layout and the deep-link router.**
All 14 driven ids exist in the served layout — 5 wraps + 5 pills at
`layout/common.py:552-556, 603-682`, `bub-scale-controls` at
`layout/bubble.py:309`, `bub-bubble-panel` at `layout/bubble.py:407`,
`bub-cagr-fwd-wrap`/`bub-occ-ctl-wrap` at `layout/common.py:558, 573`. I
hand-checked `mode_styles(mode)` against each of the five deleted literal
tuples in the old `toggle_bub_view`, the old five-branch sync JS and the old
four `startswith` branches in `deep_link_bub_view`: all 5 × 14 values are
identical, including `_inline` (not `{}`) for the active control span and the
`scale_controls=False, bubble_panel=False` pair unique to CAGR. The x-range swap
logic is unchanged, and `_PILL_TO_MODE.get(ctx.triggered_id, DEFAULT_MODE)`
reproduces the old trailing `# Price` fallthrough. `historical_modes_js()` emits
`mode === 'occupancy' || mode === 'percentile' || mode === 'resid'` — the same
three modes as the old chain, same single quotes, so the two historical-only
scripts and the test that greps them are unaffected.
**Ordering hazards, executed rather than reasoned** (`_norm` → `mode_for_path`):
`/1`→None, `/10`→None, `/10.5`→None, `/15`→None, `/9.7`→None, `/leverage`→None,
`/mi.3`→None, `/faq.2`→None, `/1.6`→None, `/1.2`→cagr, `/1.20`→cagr,
`/1.3`→resid, `/1.4`→percentile, `/1.5`→occupancy, `/1.50`→occupancy,
`/1.55`→occupancy, `/1-5-3-1`→occupancy, `/1.2.5.1`→cagr. Every one matches the
old `startswith` chain byte-for-byte (the four prefixes are equal-length and
mutually exclusive, so longest-prefix and first-match agree). `/1.20`→cagr and
`/1.55`→occupancy were also true before the refactor — no regression.
**`deep_link_bub_view` still returns 20 positional values in the registered
Output order**, executed: `/1.2.5.1` → 20 values, mode `cagr`, extras
`([2025, 2050], 20, True, NU, NU)`; `/1.5.3.1` → 20, `occupancy`, extras
`(NU, NU, NU, 25, 1)`; `/1.3` → 20, `resid`, all-NU extras; `/1` → 20 × `NU`.

**3. `_date_customdata` (5a2c71f) and every hovertemplate that indexes
customdata.** I enumerated the call sites independently:
`_add_date_hover` is reached from exactly two places —
`figures/common.py:959` (inside `_apply_final_steps`, used by occupancy /
residuals / percentile directly and by dca / retire / supercharge / citadel via
`_finalize_chart`) and `figures/bubble.py:702` (`recovery=True`). No figure gets
both, so the helper never runs twice on one figure and a date can never be
prepended twice. The `recovery=True` branch is untouched and writes its own
`[[d, r]]` rows without going through `_date_customdata`.
The guard is `if x_min < 0.3 or x_max > 120: continue` (`figures/common.py:924`).
Of the four non-occupancy customdata producers:
- **heatmap** (`figures/heatmap.py:640`, 7-wide tuples read at `customdata[2..6]`
  and again in Python at `:686-687`) — `grep -n '_add_date_hover|_apply_final_steps|_finalize_chart' figures/heatmap.py` returns **nothing**; it builds `go.Figure` and applies only the watermark. Belt-and-braces, its `x=years` are calendar years (>120) so the guard would skip it regardless. **Cannot shift.**
- **supercharge Mode B** (`:607`, `:677`) — `qlbls = [_fmt_q_label(q) for q in sel_qs]` is a flat list of **strings**, and `_date_customdata`'s `all(isinstance(r, (list, tuple, np.ndarray)) for r in rows)` is therefore False, so it takes the historical replace-with-`[[date]]` path *unconditionally* — independent of whether the guard admits it (it does when the lowest selected quantile is ≥ 0.3). This is a proof, not a sample. Pinned by `test_date_hover.py::test_flat_scalar_customdata_is_replaced`.
- **custom_time** (`callbacks/custom_time.py:300, 379, 390`) — `_build_figure` constructs its own `go.Figure` and `_add_date_hover` appears nowhere under `btc_web/callbacks/`. Unaffected.
- **residuals** (`:218`) — hover *format string* only, `customdata[0]` = date. Unaffected.
`grep -rn customdata --include=*.py` over the whole `btc_web/` tree (minus
tests) finds no other producer — in particular none in `mc_overlay.py`.
**Nothing in the app has a shifted `customdata` index.**
**`dy`:** `grep -n 'dy=|y0=' figures/occupancy.py` finds no assignment — only
the explanatory comment. The strip tick traces pass an explicit
`y=[y_row] * days.shape[0]`, and `test_occupancy.py` asserts `t.dy is None and
t.y0 is None` for every strip trace.
**Runtime shape check** (built against the live `model_data.pkl`, 4 parameter
combinations): the full-window hover trace carries 3-wide customdata
(`[date, above, below]`), the pre-window trace 1-wide (`[date]`), `text` is
`None` on both, `len(customdata) == len(x)`, and every share value is already
1-decimal. The float32 `x` narrowing lands after `_add_date_hover` (dtype
`float32` observed on the finished traces while the dates are still correct —
`test_strip_hover_dates_are_the_displayed_days` pins exactly this ordering), it
touches only `yaxis == "y2"` traces, and its ~1e-6 yr error is 3 orders below
the 1-day point spacing.

**4. `tools/monthly_refit.sh` failure paths vs. `daily_update.sh` recovery.**
Traced all four:
- *refit fails* → `notify_failure` + `git checkout -- btc_core/` + exit 1; trap releases the lock. Worktree back to `origin/master`.
- *gate fails* → identical path, nothing committed, nothing pushed.
- *push fails* → `notify_failure`, commit left on the worktree's detached HEAD, lock released. `daily_update.sh:80-83` (`git checkout -f --detach "origin/$PROD_BRANCH"` + `git reset --hard`) orphans it at 06:00; the refit is lost but the pipeline is unharmed and the operator was told.
- *killed mid-way* → SIGTERM runs the trap (verified empirically), so the worktree is left with uncommitted `btc_core/*.py` edits that the daily job's `checkout -f`/`reset --hard` discards; SIGKILL leaks the lock, which is F2.
The interlock itself is sound: the refit checks the lock **before** installing
its trap, so it can never delete a lock it did not create, and `daily_update.sh`
reads the same path at line 36 before doing anything.
**Deploy ride-along confirmed:** `btc_core/` is in `WATCHED_PATHS`
(`daily_update.sh:137`), and because the daily job resets to `origin/master`
first, a refit commit pushed at ~02:30 is already in HEAD at 06:00; the pkl
rebuild that morning runs against the new parameters and its commit carries the
whole thing to prod. `git push origin "HEAD:$BRANCH"` from a detached HEAD is
the same idiom the daily job uses.
**Unit files** (`~/.config/systemd/user/quantoshi-refit.{service,timer}`):
`OnCalendar=*-*-01 01:00:00` + `Persistent=true` fires **once** a month (a
`Persistent` catch-up replays a single missed occurrence, it does not repeat);
`Type=oneshot`, `TimeoutStartSec=4h` comfortably exceeds the 60–90 min refit
plus its gate; `PATH` covers `git`/`python3`/`notify-send`/`systemd-cat` in
`/usr/bin`. `systemctl --user list-timers` shows it armed for
`Thu 2026-10-01 01:00`, five hours ahead of `quantoshi-update.timer`'s 06:00.
No `[Install]` is needed on a timer-activated service.
`--dry-run` exits before any commit/push and never leaves the lock behind.

**5. `.gitignore` additions vs. tracked files.**
`git ls-files | grep -E 'bak$|\.so$|mc_cache|citadel_band_cache'` returns only
`btc_web/citadel_band_cache.py`, `btc_web/load_citadel_band_cache.py`,
`btc_web/mc_cache.py`, `btc_web/test_mc_cache.py`, `generate_mc_cache.py` and
the Markov `.so` — the four `.py` modules are **not** matched by the patterns
`btc_web/mc_cache` / `btc_web/citadel_band_cache` (which have no extension and
so match only the symlink directories the script creates), confirmed by the
authoritative `git ls-files -i -c --exclude-standard`, which lists **only** the
Markov `.so` (F4) and the pre-existing `quantoshi_logo.xcf`. No `.bak` file is
tracked. Nothing tracked is newly hidden.

**6. CLAUDE.md edits.**
The refit description at `CLAUDE.md:153` matches reality on every checkable
point: the timer name (`quantoshi-refit.timer`), the script path, the schedule
(1st, 01:00), that it patches `btc_core/*.py` in the deploy worktree, gates on
model tests, commits and pushes, and that the 06:00 daily job deploys it via a
`model_data.pkl` rebuild that bumps the figure-cache fingerprint (matching
`daily_update.sh:137` `WATCHED_PATHS` and its step-8 comment on why `FLUSHDB` is
omitted). The "prod must stay a pure git consumer" note added at `CLAUDE.md:114`
is consistent with 98cbb4e's rationale. The only factual drift is the test
inventory — F5.

**Also checked, unchanged.** No commit in the range touches
`btc_web/snapshot.py`, `btc_web/snapshot_defaults*.py`, `btc_web/tab_defaults.py`
or `btc_web/colors.py` (`git diff --name-only 98cbb4e~1..5a2c71f` over those
patterns is empty), so `_SNAPSHOT_CONTROLS`, the defaults fingerprint registry
and every existing share link are untouched — no fingerprint needed to be
pinned. `btc_web/bub_views.py` is git-tracked and stdlib-only, so prod's `git
pull` picks it up with no dependency change (`btc_web/requirements.txt` is not
touched). Targeted regression run on the surrounding surface —
`test_no_orphan_callbacks.py test_figures.py test_percentile.py
test_callbacks.py test_snapshot.py test_restore_builder.py` → **499 passed, 5
skipped, 1 failed**, the single failure being the known
`TestBTCPayPricing::test_free_tier_all_models` the operator said to ignore.
