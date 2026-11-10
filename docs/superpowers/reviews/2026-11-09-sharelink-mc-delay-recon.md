# Recon: MC paths appear ~40 s after a Tab-1 share link loads

**Date:** 2026-11-09 · **Scope:** evidence only — nothing was built or changed.
**Link under test:** `/1#q4:f10ac563:…` (Tab 1, MC on, PL source, regime [0,2],
start 2028, sims 8, x-range 2026–2042).

## Finding

The delay is **browser main-thread time, not server or network time**. Two
causes stack:

1. **The Dash 4 renderer saturates the main thread for the whole load.**
   Every callback response is a Redux store write, and each write re-runs a
   selector for every mounted component (~13–21 k DOM nodes, 10 tabs' worth of
   controls). A load fires ~100–150 callback POSTs, so the page is pinned.
2. **The MC figure is the last thing in that queue.** `restore_builder.py:99`
   declines to build a Tab-1 figure when the snapshot has MC on, so the figure
   comes from the ordinary cascade. `update_bubble` ran 6 times during the
   load; the first 5 returned no-ops (gated by `snapshot-pending`), and only
   the invocation triggered by `bubble-first-render.data` — the final bump of
   the restore sequence — returned the figure.

## Evidence

CPU profile (Chromium CDP sampling profiler, 2 ms) from navigation until the
first `mc-spaghetti` trace is in the graph:

| Run | MC visible | Main thread busy / idle | POSTs | Self time in `dash_renderer` |
|---|---|---|---|---|
| prod, share link | 36.8 s | 35.7 s / 0.7 s | 147 | 26.5 s (74 %) |
| dev (`.dev.js` build), share link | 72.3 s | 72.7 s / 0.2 s | 138 | 56.3 s (78 %) |
| prod, plain `/1` (no hash, to quiescence) | — (figure at 6.5 s) | 22.5 s / 1.6 s | 100 | 14.7 s (65 %) |

Per-request timing on dev (Resource Timing, 138 POSTs): server time-to-first-
byte p50 3 ms, p90 20 ms, max 424 ms, **sum 2.0 s**; connection stall max
7 ms; download max 6 ms; 2.1 MB total. Long tasks: 277, totalling 65.6 s of a
75 s load, spread evenly (7–10 s in every 10 s bucket). The server log agrees:
`restore_from_url … decode_ms=3.5`, `bubble-fig BUILT` 100–950 ms.

Hot functions in the dev build (self ms): Ramda `_path` 7 973,
`getSnapshotWithSelector` 5 422, `checkIfSnapshotChanged` (react-dom) 3 666,
`isFirstLevelPropsChild` 3 502, `memoizedSelector` 2 385,
`checkDashChildrenUpdate` 1 261 — the per-component `useSelector` path in
Dash 4's `DashWrapper`. Plotly itself is ~1 s.

`update_bubble` invocations on dev (ms after navigation → changed props):
3 840 `bub-xrange` · 4 902 `bub-yrange` · 32 397 `bub-qs…` · 50 271
`bub-xrange` · 63 772 `user-model-store…` · **67 209 `bubble-first-render`**
(the only one that returned a figure; MC visible at 75 066).

Earlier network captures that looked like 5–24 s server latency on 28-byte
responses were the browser being unable to *process* the response, not the
server being slow to send it.

## What this rules out

- Server compute, gunicorn worker starvation, nginx rate limiting, and the
  browser's 6-connection limit (stall max 7 ms).
- The MC cache or the new 200-path fan (the link under test uses 8 paths; the
  delay predates Phase 1 and reproduces on prod's pre-Phase-1 build).

## Not established

- Whether cost scales with mounted-component count, callback count, or both
  (the two were not varied independently).
- What the 13 s "first figure" on prod's share-link run is (restored figure
  vs. the pre-injected default) — the probe only counted traces.
- Which of the ~100 POSTs on a plain `/1` load are avoidable.

## Options for a Phase 2 (not started; each needs its own design pass)

| Option | Attacks | Rough size | Note |
|---|---|---|---|
| Build the MC figure in the direct-restore path (lift the `restore_builder.py:99` punt for cached-MC snapshots) | cause 2 | small–medium | MC would arrive with the first restored figure; total page-busy time unchanged |
| Cut callback POSTs during load (lazy-tab second rounds, repeated prefetch rounds, other-tab chart builds) | cause 1 | medium | Helps every page load, not just share links |
| Mount fewer components until a tab is visited | cause 1 | large | Touches the lazy-tab / prefetch architecture |

The first is the cheap, targeted one; the second is where the general
page-load win is. Restore has four failed attempts on record
(`restore_callback_architecture.md`), so any of these wants a spec first.
