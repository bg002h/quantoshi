# R2 measurement: who bumps `bubble-first-render`, and what it costs

Controller-measured, 2026-09-06. This closes open question §5.1 of
`2026-09-06-restore-burst-recon.md`, which said of R2:

> **This is an estimate:** the fan-out is measured, the bump attribution is
> inferred from the writer/consumer graph, not from a counter. Instrument
> before committing to a number.

Instrumented all five writers of `bubble-first-render` — plus both writers of
the `active-tab-bump-tick` store that feeds writer W4 — with a throwaway
clientside recorder in a scratch worktree, ran Playwright against the dev
server, and read `window.__qsBumps` / `window.__qsTicks` back. The
instrumentation was **discarded, not committed** (worktree
`r2-instrumentation`, removed).

## Headline

| load | POSTs | bumps | POSTs caused by a bump | upload in those |
|---|---|---|---|---|
| plain `/1` | 96 | **2** | **16 (17 %)** | **53.0 KB** |
| share-link restore | 147 | **5** | **30 (20 %)** | **136.3 KB** |

One bump costs ~6 POSTs. The recon's "4–5 bumps" read off timestamp clusters
was right; its ~30-POST saving for R2 was right in magnitude but **wrong in
attribution**, which changes what the fix should be.

## Per-writer attribution (share-link restore, 5 bumps)

| t (ms) | writer | value | verdict |
|---|---|---|---|
| 1735 | **W2** snapshot-state-store | 1→2 | **legitimate** — this *is* the restore |
| 2903 | **W4** active-tab-bump-tick | 2→3 | tick from `palette-store` hydration |
| 3059 | **W5** σ-mode (`scanner.py:328`) | 3→4 | mode=`resqr`, i.e. the framework's own restore write |
| 4405 | **W4** active-tab-bump-tick | 4→5 | tick from `snapshot-lots`+`effective-lots`, **`eff` `[]`→`[]`** |
| 10527 | **W4** active-tab-bump-tick | 5→6 | tick from `lots-store`+`effective-lots` (localStorage hydration) |

W1 (tab switch) and W3 (post-lazy-load) never fired on either load.

**The recon fingered W5 as the culprit; the measurement says W4 is.** W5 fires
exactly once (~6 POSTs). W4 fires **three times** (~18 POSTs). Fixing only
`scanner.py:328-336`, as R2 proposed, would save ~6 of the 30 — a fifth of
the estimate.

## Root cause of the W4 fires — hydration, not user action

`active-tab-bump-tick` (`charts/_clientside.py:511-533`) has two writers, and
both bump unconditionally on *any* write, including the initial hydration
write that carries a value nothing acted on:

* `palette-store.data` — fires once as the store hydrates, value `"default"`,
  i.e. the palette the page already rendered with.
* `effective-lots` / `snapshot-lots` / `lots-store` — fires on each, and the
  cascade means one logical change bumps twice. On the plain load the first
  fire is provably a no-op: triggered by `snapshot-lots.data,effective-lots.data`
  with `eff` going **`[]` → `[]`**.

On a plain load with no share link and no user interaction at all, that is
**2 bumps / 16 POSTs / 53 KB spent re-rendering for values that did not
change.**

## What this implies for an R2 cycle

The shape of the fix moves from "guard the σ-mode writer" to "**don't bump on
a write that changed nothing**" — compare against the previous value in the
two `active-tab-bump-tick` writers (and in W5) and return `no_update` when
equal. That covers 4 of the 5 bumps rather than 1, and it is a clientside
value comparison, not a dispatch change — the same low-risk shape as R1.

Not attempted here. Two cautions for whoever picks it up:

1. **`eff` can be large.** A lots comparison must be on a cheap key (length +
   a hash), not `JSON.stringify` of the whole array on every fire.
2. **The 10.5 s fire is real localStorage data arriving late**, not a no-op —
   suppressing it needs care, and this browser profile carried a saved lot
   from earlier E2E runs, so a clean profile may show only 2 W4 fires.

Numbers are dev-only; see recon §5.2 on the dev/prod gap.
