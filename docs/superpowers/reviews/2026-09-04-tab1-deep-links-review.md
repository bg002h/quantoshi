# Tab-1 deep links /1.4, /1.5 — scoped review (commit 4cfdafd)
Reviewer: sonnet subagent · Date: 2026-09-04

## Verdict: 0 Critical / 0 Important -> GREEN — counts: C=0, I=0, M=0, N=0

## Findings
None. No output mis-ordering, no broken existing deep link, no mis-routed path was found.

## Checked and clean

1. **Output order, all branches, element-by-element against the 20-Output decorator list**
   (`btc_web/callbacks/routing.py:373-392` decorator, `:406-455` branches). Indexed the
   Output list as 0=`bub-view-mode.data`, 1-5=five wraps (price/cagr/resid/pctile/occ),
   6-10=five outlines, 11=`bub-scale-controls`, 12=`bub-bubble-panel`,
   13=`bub-cagr-fwd-wrap`, 14=`bub-occ-ctl-wrap`, 15=`bub-xrange.value`,
   16=`bub-cagr-fwd-yrs.value`, 17=`bub-cagr-hover-today.data`, 18=`bub-occ-tail.value`,
   19=`bub-occ-window.value`. Manually mapped every element of the `/1.3`, `/1.4`, `/1.5`,
   `/1.2` return tuples and both `(NU,)*20` fallthroughs onto this list — every position
   lines up with the field its name implies (e.g. `/1.4`'s tuple sets index 9
   `bub-view-pctile.outline = False` and index 4 `bub-pctile-wrap = {}`, nothing else
   shown). No transposition found in any branch.

2. **Interaction with the clientside sync in `charts/__init__.py`** (`toggle_bub_view`,
   lines 542-596, and the mode-sync clientside callback, lines 601-656 — neither file nor
   these functions were touched by 4cfdafd, confirmed via `git show 4cfdafd --stat`).
   Diffed the wrap/outline/scale/panel/cagr-fwd/occ-ctl values (indices 1-14) that
   `deep_link_bub_view` writes directly against what the mode-only clientside sync
   recomputes for the same `bub-view-mode` value (`cagr`/`resid`/`percentile`/`occupancy`)
   — all four modes agree exactly (e.g. cagr: both set `bub-scale-controls`/
   `bub-bubble-panel` hidden and `bub-cagr-fwd-wrap` inline; occupancy: both set
   `bub-occ-ctl-wrap` inline). Since the Python callback's Output list "mirrors
   toggle_bub_view" as the docstring states, the subsequent clientside sync (triggered by
   the `bub-view-mode.data` write) reproduces identical values — no stale/disagreeing
   panel state on any of the 4 deep-linked modes.

3. **`_pick(2, ...)` / `_pick(3, ...)` indexing and `/1.2` parity with pre-commit code.**
   Confirmed `parts = pathname[1:].split(".")` gives `parts[0]="1"`, `parts[1]="5"` (or
   `"2"`/`"3"`/`"4"`), so `_pick(2, ...)` reads T (or N for `/1.2`) and `_pick(3, ...)`
   reads W (or B). Diffed against `git show 4cfdafd^:btc_web/callbacks/routing.py`'s
   `/1.2` branch: old code required `len(parts) >= 3` before touching `parts[2]` and
   `len(parts) >= 4` before `parts[3]`; `_pick`'s `len(parts) <= i: return NU` guard is
   the same condition restated (`i=2` ⇔ old `>= 3` requirement). Same `_FWD_OPTIONS`
   values `[1, 2, 4, 10, 20, 30]` passed to `_pick(2, ...)` for `/1.2`, hover-today logic
   for `parts[3]==1` left untouched. `bub-occ-tail`/`bub-occ-window` dropdown option
   orders in `layout/common.py:574-583` (`[5,10,25]`, `[1,2,4]`) match the lists passed to
   `_pick` exactly. Ran the tail/window math for `/1.5.3.1` by hand and against the
   already-verified Firefox result (tail 25%, window 1yr) — consistent.

4. **JS regex `/^\/1\.\d+/` replacing the two `indexOf` checks**
   (`routing.py:359-360`). Extracted the live tab-map function body and ran it under
   Node against 20 paths including `/1.2.5.1`, `/1-2-5-1` (post `-`→`.` normalization),
   `/1.4`, `/1.5.3.1`, `/1.10.2`, and — per the brief's specific worry — `/10.3` and
   `/10.1`. All resolved correctly (`/1.*` → `bubble`, `/10.*` → `faq`). `/10.3` is
   protected two ways: the `/^\/10\.\d+$/` check runs first in the function and returns
   before the `/1.\d+` regex is reached, and even standing alone the `/1.\d+` regex
   requires a literal `.` immediately after the `1`, which `/10.3` doesn't have (`1`
   is followed by `0`). `/2.N` regex (`/^\/2\.\d+$/`) is byte-identical to pre-commit,
   untouched by this commit's diff (only the `/1.2`/`/1.3` two-line block was replaced).

5. **`prevent_initial_call=True` + `allow_duplicate=True` on all 20 Outputs.** Extracted
   the decorator's Output list from the commit's post-image and regex-checked each of
   the 20 entries for `allow_duplicate=True` — all 20 have it (including the two newly
   split-out original Outputs and the 7 new ones). `prevent_initial_call=True` is
   unchanged from pre-commit. No missing `allow_duplicate` that would raise "Duplicate
   callback outputs" at import.

Also confirmed by direct diff inspection: the commit touches only
`btc_web/callbacks/routing.py` (3 hunks — the JS one-liner, a comment, and the
`deep_link_bub_view` rewrite), `btc_web/test_bub_deep_links.py`, `CLAUDE.md`,
`docs/user_manual.md`, `docs/superpowers/followups.md`. No edits to
`btc_web/callbacks/charts/__init__.py` or `btc_web/layout/common.py` — all referenced
component IDs (`bub-pctile-wrap`, `bub-occ-wrap`, `bub-occ-ctl-wrap`, `bub-view-pctile`,
`bub-view-occ`, `bub-occ-tail`, `bub-occ-window`) already existed pre-commit and are
wired identically on both sides of the diff, so `/9.N`, `/10.N`, `/2.N`, `/faq.N`,
`/mi.N` deep links are untouched by this commit.
