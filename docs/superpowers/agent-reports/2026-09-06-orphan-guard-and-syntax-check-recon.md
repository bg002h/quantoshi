# Recon: the orphan guard's blind spot, and the stale syntax-check recipe

**Date:** 2026-09-06
**Scope:** RECON ONLY. No application code, test, CLAUDE.md or doc was edited;
nothing committed; no dev server started. The only file created in the repo is
this report. All throwaway scripts live in the session scratchpad.
**Repo state:** `/scratch/code/bitcoinprojections`, branch `master`.
**Python:** `btc_venv/bin/python3` (3.14.3), Dash 4.0.0.

Every number below is measured, not reasoned. Reproduction commands are the
scratchpad scripts named per section.

---

# A. The orphan guard is checking almost nothing — but there is nothing to find

## A0. Orphan inventory (lead)

**The widened check finds exactly one orphan, and it is already allowlisted.**

| # | Component id | Referenced by | Role(s) | Map | Classification |
|---|---|---|---|---|---|
| 1 | `restore-lots-btn` | snapshot-lots restore banner wiring (already in `_KNOWN_ORPHANS`) | Input/Output | clientside | **Conditionally-rendered** — banner button only mounted when `snapshot-lots` is populated; the static layout walk cannot see it. Correctly allowlisted. |

**New orphans exposed by adding the 96 server callbacks: 0.**

That is the number that decides this. The blind spot is real and large, but it
is currently empty.

## A1. Does the orphan test walk zero server callbacks? — CONFIRMED

`btc_web/test_no_orphan_callbacks.py::_collect_callback_refs` iterates
`_app_ctx.app.callback_map` only.

Measured (`scratchpad/measure_maps.py`, conftest-faithful: `TESTING=1`,
mocked `urlopen`, `import app`):

```
app.callback_map              : 251
GLOBAL_CALLBACK_MAP           : 96
key overlap                   : 0
app._callback_list            : 251

orphan test: distinct ids referenced (via app.callback_map) : 543
orphan test: total ref sites                                : 1111
orphan test: server callbacks inspected                     : 0
```

**0 of 96 server callbacks are inspected.** The docstring's step 4 ("Walk
`app.callback_map` collecting every id referenced") is accurate about what the
code does and silently wrong about what that covers.

Exposure of the blind spot, from `scratchpad/final_numbers.py`:

| | distinct ids | ref sites |
|---|---|---|
| clientside (`app.callback_map`) — walked today | 543 | 1,111 |
| server (`GLOBAL_CALLBACK_MAP`) — never walked | 579 | 2,040 |
| union | 842 | 3,151 |
| **server-only ids, never checked by anything** | **299** | — |
| layout ids collected by the walk | 938 | — |

So **65 % of all callback reference sites** (2,040 / 3,151) and **299 distinct
component ids** have never been checked by this guard since it was written.

`entry["clientside_function"]` is confirmed useless as a discriminator: it is
`None`/absent in **all 251** `app.callback_map` entries and **all 96**
`GLOBAL_CALLBACK_MAP` entries. Map membership is the only reliable signal.

## A2. Running the check it should be running

`scratchpad/orphan_full.py` reuses the shipped `_collect_layout_ids()` and
`_parse_output_key()` verbatim (so the layout side is byte-identical to what
ships) and widens only the callback side to `GLOBAL_CALLBACK_MAP + app.callback_map`:

```
layout ids collected: 938
distinct ids referenced (clientside only) : 543
distinct ids referenced (server only)     : 579
distinct ids referenced (union)           : 842

ORPHANS with current (clientside-only) walk : 1 -> after allowlist: 0
ORPHANS with server map only               : 0
ORPHANS with union walk                    : 1 -> after allowlist: 0

NEW orphans exposed by adding the server map (not seen today): 0

pattern/dict-id deps: server = 0  clientside = 0
```

**Zero real orphans exist today.** There are also zero pattern-matching/dict
ids anywhere in either map, so that classification bucket is empty and the
`cid.startswith("{")` guards are currently dead code (harmless).

### The widened walk is not a false PASS

`scratchpad/sabotage.py` registers a synthetic server `@callback`
(`sabotage-server-out` ← `sabotage-server-in`, plus a real `State`) and a
synthetic `app.clientside_callback`, then runs both walks:

```
CURRENT walk orphans : ['restore-lots-btn', 'sabotage-client-in', 'sabotage-client-out']
WIDENED walk orphans : ['restore-lots-btn', 'sabotage-client-in', 'sabotage-client-out',
                        'sabotage-server-in', 'sabotage-server-out']

server-side sabotage seen by CURRENT walk : False
server-side sabotage seen by WIDENED walk : True
clientside sabotage seen by CURRENT walk  : True
all four seen by WIDENED walk             : True
```

The guard *can* fail when it should; it simply never looks at the server half.
Baseline: `pytest btc_web/test_no_orphan_callbacks.py` → **2 passed** in 0.63 s.

## A3. Classification of each orphan

Only one exists (`restore-lots-btn`, table above): **conditionally-rendered
component the static walk cannot see** — the existing allowlist entry is
correct and its stated reason is accurate. No genuinely-missing components, no
pattern-matching ids, no walker false positives.

## A4. Smallest correct fix — walk both maps, do NOT force the merge

**Recommendation: walk both maps.** One line in `_collect_callback_refs`:

```python
from dash._callback import GLOBAL_CALLBACK_MAP
for key, entry in {**GLOBAL_CALLBACK_MAP, **app.callback_map}.items():
```

Justification:

- The keys are **disjoint** (overlap 0), so the merge is lossless.
- This exact idiom is already the house pattern on master, in three files:
  `test_bub_view_gating.py:108`, `test_bub_view_modes.py:45`,
  `test_bub_deep_links.py:27`. Adopting it costs no new concept.
- `_parse_output_key()` already handles both server-key forms correctly —
  verified against a real multi-output key:
  `_parse_output_key('..bubble-graph.figure...bub-mc-results.data...')`
  → `['bubble-graph', 'bub-mc-results', 'bub-mc-status', …]`.

**Forcing the merge is available but worse.** Dash 4.0.0 merges in
`Dash._setup_server()` (`dash/dash.py:1639-1647`):

```python
for k in list(_callback.GLOBAL_CALLBACK_MAP):
    if k in self.callback_map:
        raise DuplicateCallback(...)
    self.callback_map[k] = _callback.GLOBAL_CALLBACK_MAP.pop(k)
```

Side effects that disqualify it in tests:

1. **It `pop()`s — destructively emptying a process-global.** `GLOBAL_CALLBACK_MAP`
   is module state shared by every test in an xdist worker. Calling
   `_setup_server()` in one test would empty it for the rest of that worker.
   Concretely, `test_callbacks.py::test_restore_from_url_does_not_output_bubble_graph`
   reads `GLOBAL_CALLBACK_MAP` *without* merging and ends in
   `assert found, "restore_from_url callback not found in GLOBAL_CALLBACK_MAP"`
   — it would go red purely from test-ordering. (The pending
   `test_lazy_flag_payload.py` in the r1-state-payload worktree does the same.)
2. It also runs `validate_layout`, `_walk_assets_directory`,
   `_generate_scripts_html`, `_generate_css_dist_html`,
   `validate_background_callbacks`, and **registers extra cancel callbacks** for
   background callbacks — i.e. it mutates the callback graph the test is
   measuring.
3. It is private API guarded by a one-shot `_got_first_request` flag, so it is
   order-dependent and not idempotent across tests.

## A5. Would the fix land red? — No. It lands green with the allowlist unchanged.

- Orphans after widening: **1**, which is the single existing `_KNOWN_ORPHANS`
  entry. `_KNOWN_ORPHANS` needs **0 new entries**.
- `test_known_orphans_still_orphaned` also stays green: `restore-lots-btn` is
  still referenced in the union and still absent from the layout (measured:
  `known orphan still referenced: True`).

No real bugs need fixing first. This is a pure "make the guard actually guard"
change, and the ratchet question does not arise.

## A6. The same blind spot in other tests — and a second, independent defect

Two defects, measured separately (`scratchpad/vacuity.py`, `scratchpad/dig2.py`):

- **D1** — walking `app.callback_map` only, so server callbacks are invisible.
- **D2** — parsing output keys with a naive `key.split("...")`, which leaves the
  wrapping `..` on the **first and last** output of every multi-output callback
  (e.g. `'..main-tabs.active_tab'` instead of `'main-tabs.active_tab'`).
  Measured: **154 of 347** callbacks are multi-output; **222 output entries are
  mis-parsed** by the naive split. `test_no_orphan_callbacks._parse_output_key`
  is the only parser in the repo that gets this right.

| Test | D1 | D2 | Measured effect today | Green after fix? |
|---|---|---|---|---|
| `test_no_orphan_callbacks.py::test_no_orphan_callback_refs` | yes | ok | inspects 0/96 server callbacks | **yes** (0 new orphans) |
| `test_no_orphan_callbacks.py::test_known_orphans_still_orphaned` | yes | ok | same | **yes** |
| `test_callbacks.py::TestNoDuplicateCallbackOutputs::test_no_unguarded_duplicate_outputs` | yes | yes | checks 421 of 819 distinct outputs; a server↔server or server↔clientside collision — the exact class that "blocks ALL callbacks" per its own docstring — is invisible | **yes** (union + correct parse → 0 violations) |
| `test_callbacks.py::TestNoDuplicateCallbackOutputs::test_restore_from_url_uses_intermediate_store` | yes | yes | **VACUOUS**: 0 matching callbacks in `app.callback_map`, so the loop body and its assertion never execute. The target (`restore_from_url`) is a server callback. | **yes** (1 match, `_SNAPSHOT_CONTROLS` overlap = 0) |
| `test_callbacks.py::TestSnapshotPendingChartGate::test_nine_chart_callbacks_have_snapshot_pending_state` | yes | yes | **VACUOUS**: finds **0 of the 9** chart callbacks. All 9 are server callbacks, and 7 of them are multi-output so `outputs[0]` is `'..bubble-graph.figure'`. Union alone lifts it to 2/9; union + correct parse gives 9/9. | **yes** (9/9 found, all 9 already declare `State('snapshot-pending')`) |
| `test_snapshot.py::test_snapshot_pending_writers_have_allow_duplicate` | yes | yes | checks **1 of 9** writers; the 8 server writers (`restore_from_url`, `apply_tab_bubble` … `apply_tab_leverage`) are unchecked | **yes** (9 writers, 0 violations) |
| `test_snapshot.py::test_apply_globals_does_not_output_snapshot_pending` | yes | yes | **VACUOUS on both counts**: 0 matches in *either* map, because the predicate needs `"main-tabs.active_tab" in clean` and the naive parse yields `'..main-tabs.active_tab'`. Fixing only D1 would not revive it. | **yes** (matches `apply_globals`; `snapshot-pending.data` correctly absent from its 32 outputs) |
| `test_axes_presets.py` (`_preset_entries`, `_style_entries`) | — | — | **NOT affected.** Targets are `app.clientside_callback` registrations; measured 4 in `app.callback_map`, 0 in `GLOBAL_CALLBACK_MAP`. Its docstring already explains this correctly. | n/a |
| `test_bub_view_gating.py`, `test_bub_view_modes.py`, `test_bub_deep_links.py`, `test_callbacks.py::test_restore_from_url_does_not_output_bubble_graph` | — | — | **NOT affected.** Already merge or read `GLOBAL_CALLBACK_MAP`. | n/a |

Headline: **three tests are currently vacuous** (their assertion bodies never
execute), and `test_apply_globals_does_not_output_snapshot_pending` is vacuous
for a reason the map fix alone does not repair.

## A. Recommendation — **worth a cycle, but a small one**

Worth it, with eyes open about what it buys. The orphan guard itself finds
**zero** live bugs today, so the round-trip cost cited in
`feedback_nonexistent_input_perf.md` is not currently being paid. What the
cycle actually buys is the **three vacuous tests** — assertions that read as
guarding documented, previously-experienced crash classes (Dash blocking all
callbacks on an unguarded duplicate output; `restore_from_url` writing snapshot
controls directly; the single-redraw chart gate) and that would not fail if
those invariants were violated tomorrow. That is a false-PASS class, which the
project severity rules treat as blocking.

Everything lands **green**, so it is a low-risk cycle with no allowlist growth
and no prerequisite bug-fixing.

**Smallest first slice** (one commit, ~15 lines, no behaviour change):

1. Add a shared helper — the natural home is `test_no_orphan_callbacks.py`
   (it already owns the only correct `_parse_output_key`), exporting
   `_all_callbacks()` returning `{**GLOBAL_CALLBACK_MAP, **app.callback_map}`.
2. Point `_collect_callback_refs` at it, and update the module docstring's
   step 4 (it currently names `app.callback_map` specifically).
3. Fix the four `.split("...")` call sites in `test_callbacks.py` and
   `test_snapshot.py` to use the same parser, and point them at both maps.
4. Add one assertion per revived test that it is **non-vacuous** (e.g.
   `assert len(chart_callbacks) == 9`), so the next silent regression to
   zero matches is red rather than green. Without step 4 the fix is
   undetectable and can silently rot back.

Steps 1–2 alone are a defensible half-slice if budget is tight, but steps 3–4
are where the actual defect is; step 4 is what stops it recurring.

---

# B. The CLAUDE.md syntax-check recipe

## B1. When did it break? — It was **born broken**, 2026-04-16

`CLAUDE.md:103-107`:

```bash
cd btc_web && PYTHONPATH=".:.." ../btc_venv/bin/python3 -c \
  "import layout, figures, callbacks, cache, engines.adapter, engines.citadel, engines.tax, engines.tax_lots, engines.tax_data, data.asset_matrices; print('OK')"
```

Timeline, from `git log -S` (read-only, shared history):

| Date | Commit | Event |
|---|---|---|
| 2026-03-07 | `7a0d9d1` | `btc_web/_app_ctx.py` created with `app = None` (still line 183 today); `btc_web/callbacks/` gains module-level `_app_ctx.app.clientside_callback` in the same commit |
| **2026-04-16 20:55:59** | **`f8e2038`** | `feat(citadel): add WIP warning modal on every /6 page load` — adds a **module-level** `_app_ctx.app.clientside_callback(...)` to `btc_web/layout/citadel.py` (today at line 492). `import layout` now requires `app` to have been imported first. |
| **2026-04-16 21:36:12** | **`fb821bf`** | The recipe is added to `CLAUDE.md`. |

`git merge-base --is-ancestor f8e2038 fb821bf` → **YES**. The commit that broke
`import layout` landed **41 minutes before** the recipe was written, and
`git show fb821bf:btc_web/layout/citadel.py | grep -c clientside_callback` → **1**.

**It was never correct.** Even setting `layout` aside, `callbacks` has required
`app` first since 2026-03-07 — the same commit that introduced `app = None`.
The recipe imports `layout` first, so the observed failure is the citadel one:

```
AttributeError: 'NoneType' object has no attribute 'clientside_callback'
  btc_web/layout/citadel.py:492
```

Reproduced this session while trying to `from layout import _build_layout`
without importing `app`.

## B2. Minimal correct replacement

```bash
TESTING=1 PYTHONPATH="btc_web:." btc_venv/bin/python3 -c \
  "import app, engines.adapter; print('OK')"
```

Measured (`scratchpad/coverage_app.py`):

- `import app` transitively loads **74** `btc_web` modules and reaches **9 of
  the 10** modules the recipe names. The one it does **not** reach is
  **`engines.adapter`** (imported lazily inside a function at
  `mc_overlay.py:1283`), so it must be named explicitly — hence
  `import app, engines.adapter`.
- The command above runs clean: `OK`.

**Is `TESTING=1` needed?** Not for correctness — only for behaviour. It is read
at exactly one place, `app.py:408`:
`if os.environ.get("DEV") != "1" and os.environ.get("TESTING") != "1":` — the
`_prewarm_caches()` guard. Measured: 2.29 s with `TESTING=1`, 1.33 s without
(the /dev/shm MC cache was already warm, so prewarm was cheap here; on a cold
cache it builds every tab's figures). Keep `TESTING=1`: it makes the check a
*syntax/import* check rather than a partial app boot, and keeps it bounded.

## B3. Does it catch anything pytest does not? — **No. Pytest is strictly louder.**

Tested by simulating exactly the failure class the recipe is supposed to catch
(`scratchpad/poison_app.py`: a `sys.meta_path` finder that raises `ImportError`
for `app`, installed via `-p` so it precedes `btc_web/conftest.py`). This is the
`feedback_prod_runtime_imports.md` shape — a prod-missing dependency imported at
module scope.

Result:

```
756 tests collected, 13 errors in 0.91s
!!!!!!!!!!!!!!!!!!! Interrupted: 13 errors during collection !!!!!!!!!!!!!!!!!!!
```

**Pytest does not even start.** It aborts at collection with 13 errored
modules — `test_axes_presets`, `test_bub_deep_links`, `test_bub_view_gating`,
`test_bub_view_modes`, `test_callbacks`, `test_citadel`, `test_core`,
`test_model_registration`, `test_models`, `test_palette_roundtrip`,
`test_resqr_snapshot`, `test_snapshot`, `test_timemachine_callbacks` — several
reporting the *same* `AttributeError: 'NoneType' …` the recipe would print.

A full run with the poison confirms it (`263 failed, 415 passed, 5 skipped,
247 errors` vs. baseline `31 failed, 2943 passed, 12 skipped, 46 errors` on the
same collection — the baseline failures/errors are the E2E files, which need a
dev server, plus the known BTCPay one).

The `try: import app … except Exception:` block in `btc_web/conftest.py` (with
its 27 `@pytest.mark.skipif(_q3 is None)` guards) does **not** mask this: only
5 tests skipped under the poison, versus 12 at baseline, because module-level
`from callbacks import …` in the test files themselves errors first.

Coverage is also complete: all 10 modules the recipe names are exercised by the
suite — 9 via `conftest.py`'s `import app`, and `engines.adapter` directly by
`btc_web/test_infrastructure.py:139,146`.

## B. Recommendation — **drop it (delete the recipe), fold into an existing cycle**

Not worth a cycle of its own. The recipe has never worked, has been wrong in
`CLAUDE.md` for ~5 months, and covers nothing the suite misses — the suite
catches the same class harder and faster (aborts in 0.9 s at collection).
Keeping a fixed version would add a second thing to remember that only
duplicates `pytest`.

**Smallest first slice** — a one-hunk `CLAUDE.md` edit, folded into whatever
doc-touching commit comes next (this has no owning phase and is Nit-severity):

Replace the "### Syntax-check the web app" block with a single line under the
existing "### Run tests" section, e.g.:

> **Import/syntax check:** there is no separate recipe — `pytest btc_web/`
> aborts at collection (13 modules, <1 s) on any import-time failure in
> `app.py` or anything it loads. For a bare import check without the suite:
> `TESTING=1 PYTHONPATH="btc_web:." btc_venv/bin/python3 -c "import app, engines.adapter"`
> (`engines.adapter` is lazily imported and is the only runtime module `import app` misses).

If the preference is to keep a standalone command, use the B2 command verbatim
— but say plainly in `CLAUDE.md` that it is redundant with `pytest`.

---

## Scratchpad reproduction scripts

All under
`/tmp/claude-1000/-scratch-code-bitcoinprojections/1b0dd4e4-5118-4045-9449-48ad6c21eb8d/scratchpad/`:

| Script | Produces |
|---|---|
| `measure_maps.py` | A1 map sizes, disjointness, `clientside_function` uselessness |
| `orphan_full.py` | A2 widened orphan walk (+ `orphans.json`) |
| `sabotage.py` | A2 false-PASS check |
| `vacuity.py` | A6 per-test vacuity, D1 |
| `dig2.py` | A6 D2 mis-parse counts, 9-chart-callback census, duplicate-output check |
| `final_numbers.py` | A1/A2 consolidated counts |
| `poison_app.py` | B3 pytest plugin that makes `import app` fail |
| `coverage_app.py` | B2 `import app` module coverage + timings |
