# Adversarial whole-diff review — R1 "stop uploading component subtrees as State"

**Date:** 2026-09-06 · **Worktree:** `/scratch/code/bitcoinprojections/.claude/worktrees/r1-state-payload`
**Branch:** `worktree-r1-state-payload` (base: `master` @ `7dd7b4b`) · **Diff read with:** `git diff --cached`
**Scope:** this diff and its immediate blast radius only. No application code was edited, nothing committed, nothing deployed.

**The one question:** *is there any path in which a `{tab}-loaded` flag says "loaded" while that tab's `{tab}-lazy` Div still holds the "Loading..." placeholder — or any other way this diff can leave a user staring at a tab that never populates?*

---

## VERDICT: **GREEN** — 0C / 0I / 1M / 5N

**Answer to the one question: no.** I could not construct such a path, and the two halves
of the invariant are not merely *kept* in sync — they are *derived from the same value in
the same function call* on the layout side, and *written from the same `return` statement*
on the callback side. There is no third writer of either half anywhere in the codebase.

The three legs, each machine-verified rather than argued:

1. **Layout side.** `_build_layout(initial_tab)` (`btc_web/layout/__init__.py:261`) uses the
   single local `initial_tab` for both the eager render (`_t()`, `:272`) and the flag
   (`:386-391`). I built the layout for **19 entry paths** — every key of `_PATH_TO_TAB`
   plus `/`, `/nonsense`, `/1.5`, `/9.3`, `/10.4`, `/mi.2` — and asserted, for all 10 tabs
   on each, that `{tab}-loaded.data == (that tab's lazy Div is NOT the "Loading..."
   placeholder)`, that exactly one tab is eager-rendered, that it equals
   `main-tabs.active_tab`, and that `auto-y-grid-loaded == (auto-y-grid.data is not None)`.
   **190 tab-assertions + 19 auto-Y assertions, all pass.**

2. **Callback side.** `_lazy_load` and `_pf` return either `(no_update, no_update)` or
   `(content, True)`. There is no `(no_update, True)` and no `(content, no_update)` branch,
   so the flag cannot be raised without the content or vice versa. Enumerated over the live
   callback map: **exactly 20 server callbacks write `*-lazy.children`, and for all 10 tabs
   the writers of `{tab}-lazy.children` and of `{tab}-loaded.data` are the identical pair
   `[_lazy_load, _pf]`.**

3. **No third writer.** `grep` over `btc_web/**` for `-lazy` finds no other `Output`, and
   no `dash_clientside.set_props` call anywhere targets a `*-lazy` Div (the seven
   `set_props` relays in `snapshot_cb.py` all target `*-graph` figures *inside* those
   Divs, and none of them ever depended on the deleted placeholder detection). Server-side
   references to `*-lazy.children` as Input/State are now **zero**; the 12 remaining
   readers are all clientside and untouched.

Both outputs land in one HTTP response, and both target Stores/Divs that are mounted at
the top level of `dbc.Container` for the life of the page — so Dash 4's "silently drop the
dispatch when an Output component is absent from the DOM" hazard (the one documented at
`btc_web/callbacks/snapshot_cb.py:906`) cannot split the pair.

---

## Hunt list — findings against each item

### 1. Flag/content divergence — **clear**

Only two writers, both write both outputs from one `return`. The interleaving worth
checking is *prefetch fires for tab X while the user clicks tab X*:

* If `_pf`'s request is built while `main-tabs.active_tab` is already X, `_pf` short-circuits
  on `_tid == active` (`routing.py:700`).
* If it is built earlier, both `_pf` and `_lazy_load` may see `loaded == False` and both
  return `(content, True)`. `content` is the **same cached object** from
  `_TAB_CONTENT_CACHE`, so the tab is written twice with identical content and the flag
  ends True. The user sees a populated tab.

Under master the identical double-write existed for exactly the same request-construction
window (`_is_loading_placeholder(current)` was evaluated on the same stale client state).
**The race window is unchanged in both width and outcome** — this diff neither widens nor
narrows it.

The failure direction that matters (flag True, content placeholder) requires a write of the
flag alone. No such write exists.

### 2. Eager-render initialisation — **clear, verified across every entry path**

`_serve_layout` (`layout/__init__.py:187`) computes
`clean = path.rstrip("/").split(".")[0] or "/"` then
`initial_tab = _PATH_TO_TAB.get(clean, "bubble")` and passes **that one value** to
`_build_layout`. Deep links reduce correctly (`/1.5`→`/1`, `/10.4`→`/10`, `/mi.2`→`/mi`,
`/9.3`→`/9`), unknown paths fall back to `bubble`, and every value of the map is a member of
`LAZY_TABS`. The `/_dash-layout` Referer fallback (`:198-203`) can only *change which tab is
initial* — it cannot desynchronise the flag from the render, because both are computed from
the post-fallback value. A stripped `Referer` (Tor Browser) yields `initial_tab = "bubble"`,
a self-consistent layout, and the clientside router then drives the real tab through
`_lazy_load` normally. This is the same self-healing path master had.

`_build_layout` has exactly one non-test caller. `static_pages.py` serves plain Flask HTML
(`/faq`, `/faq.<int>`, `/mi`, `/mi.<int>`, `/Z`, `/mcideas`) and builds no Dash layout, so it
cannot produce a `{tab}-lazy` Div without its Store.

### 3. `auto-y-grid-loaded` — **clear**

`data=(initial_tab == "bubble")` mirrors `auto-y-grid`'s own
`data=AUTO_Y_GRID if initial_tab == "bubble" else None` from the same expression; asserted
equal across all 19 entry paths. A non-bubble initial load leaves the flag `False`, and the
first switch to bubble populates the grid and raises it.

The one latching concern — "could the callback set `loaded=True` while shipping an empty
grid, permanently?" — does not apply: `_app_ctx.AUTO_Y_GRID` is assigned unconditionally at
`btc_web/app.py:460` from a dict literal built at `:430`, so it is never `None` at callback
time, and master's `current is not None` guard would have latched on exactly the same
condition. Behaviour is identical.

### 4. `open_faq_item` trigger change — **clear (behaviour byte-for-byte preserved)**

`faq-loaded.data` is written in the same output batch as `faq-lazy.children` by the same two
callbacks, and its value always transitions `False → True` at the moment the content lands
(the invariant above guarantees it is never `True` while the placeholder is showing), so the
trigger moment is identical. Confirmed against all three cases:

* **FAQ lazily loaded by click** — `_lazy_load` writes both; `open_faq_item` fires with
  `loaded=True`, reads `url.pathname`, opens item N. ✓
* **FAQ populated by background prefetch** — `_pf` writes both; same. This is the path the
  recon report measured firing once per page load (`faq-accordion.active_item`, 1 POST,
  1 empty, **82 KB up** on both plain and share runs) — it will still fire once, now with a
  boolean instead of the 82 KB subtree. ✓
* **FAQ as the eager initial tab** (`/10.N` typed directly) — neither `faq-lazy.children`
  nor `faq-loaded.data` ever changes (`prevent_initial_call=True`, and `_lazy_load`
  no-ops), so `open_faq_item` never fires. **This is true on master too** — the callback
  had the same `prevent_initial_call` and master's `_is_loading_placeholder(current)` was
  equally `False` for eager content. Pre-existing hole, unchanged by this diff, and out of
  scope; recorded in "Observations" below rather than as a finding.

`faq-loaded` cannot be written more than once (both writers guard on it, and nothing writes
`False`), so no double-open. `faq-loaded` is now the **only** new `Input` created by this
diff — every other `{tab}-loaded` reference is a `State` — so it adds no new server fan-out.

### 5. Clientside readers left alone — **confirmed untouched, ordering preserved**

12 clientside callbacks read `{tab}-lazy.children`: the 7 `_register_first_render_bump`
registrations (`routing.py:731`), the 4 per-prefix summary fan-outs
(`callbacks/charts/_clientside.py:122`), and `_mi_spa_open` (`routing.py:797`). None is in
the diff. The diff does not change *when* `{tab}-lazy.children` changes — only what else
changes alongside it in the same response — and dash-renderer applies every output of one
response before notifying observers, so a first-render bump still lands after the content is
in the layout tree. The only new observer edge in the whole diff is
`faq-loaded → faq-accordion.active_item`, which can add a callback but cannot reorder one.

### 6. Snapshot restore relays — **clear**

`snapshot_cb.py:904-1100` mentions `*-lazy` and "Loading..." only in *comments* explaining
why the figure relays use `set_props` on `*-graph` instead of a registered `Output`. None of
the seven relays called `_is_loading_placeholder` or read a lazy Div's children. Zero
references to the deleted helper remain anywhere in `btc_web/` or `docs/` (`grep`, exit 1).

### 7. Missing-Store risk — **clear**

`LAZY_TABS` (`layout/__init__.py:255`) is now the single source for both the Store list and
the two `@callback` registration loops, so the drift this hazard needs is structurally
impossible without editing one tuple. Verified empirically anyway: all 11 new Stores
(`{10 tabs}-loaded` + `auto-y-grid-loaded`) are present in the layout on all 19 entry paths,
at the top level of `dbc.Container` (never inside a tab pane or modal). No import cycle:
`layout/__init__.py` imports nothing from `callbacks`, and `routing.py` already imported the
`layout` package via `layout.faq` / `layout.bubble` before this diff.

Note for the record (**not** a defect in this diff): `test_no_orphan_callbacks.py` walks
`app.callback_map`, which — measured — holds **251 clientside** registrations and has **zero
overlap** with the 96 server callbacks in `dash._callback.GLOBAL_CALLBACK_MAP`. So the
orphan ratchet has never covered server callbacks and does not cover the new `{tab}-loaded`
refs. The new test file's own docstring states this relationship correctly, and
`test_eagerly_rendered_tab_starts_loaded_and_others_do_not` does `stores[f"{tab}-loaded"]`
for all 10 tabs (a `KeyError` if a Store went missing), so the coverage exists — just in the
new file rather than the old ratchet. See N-4 for the one gap in it.

### 8. Lost safety property of `_is_loading_placeholder` — **no live case**

The helper accepted `None`, `"Loading..."`, a `dict` with nested `props.children`, and a
single-element list; it returned `False` (i.e. "populated, skip") for every other shape. The
boolean expresses the same partition as long as it tracks the writes, which it does. What is
genuinely given up is **self-healing**: master's guard re-derived the answer from the actual
content, so a hypothetically corrupt client state would repair itself on the next tab
switch, whereas a wrong flag would latch. Since no path can produce a wrong flag — the
flag's only writes are `False` at layout time (with the placeholder) and `True` alongside
the content — this is a reduction in redundancy, not in correctness. Not a finding.

### 9. Vacuous tests — **none; three nits**

Measured on the live app: `_uploaded_refs()` returns **1,386 refs (986 State, 400 Input)**
across 96 server callbacks, so the two "no offender" tests are not passing over an empty
set. `test_lazy_children_and_loaded_flag_are_written_in_one_batch` asserts `writers` is
non-empty before checking each, and is parametrized over all 10 tabs.
`_callback_for_output` returning `None` is caught by an explicit assert in one test and by
an unpacking `TypeError` in the other two — no silent pass. `GLOBAL_CALLBACK_MAP` is the
correct surface: `grep -rn "app.callback(" btc_web` finds none, and the map holds all 96
server callbacks including every one this diff touches. The `_outputs()` key parser handles
both the single-output `id.prop@hash` and multi-output `..a.p@h...b.q@h..` forms correctly —
verified by the flag-in-same-batch test actually matching. Nits at N-3/N-4.

---

## Findings

### M-1 · Minor · `CLAUDE.md:70` and `CLAUDE.md:190` — test inventory not updated

The repo keeps an explicit inventory of non-E2E test files and their count, added
deliberately in the 2026-09-05 burndown so that drift is a grep. This diff adds
`btc_web/test_lazy_flag_payload.py` (+17 tests) without touching it, so both statements are
now wrong:

* `:70` — "full suite (**2955** non-E2E tests, ~30 s on 24 cores)" → 2960.
* `:190` — "**2955** tests across **53** non-E2E files" → 2960 / 54, and
  `test_lazy_flag_payload` is absent from the alphabetical file list (it belongs between
  `test_infrastructure` and `test_leverage`).

Non-blocking (documentation only). **Fix:** update the count in both places and insert
`test_lazy_flag_payload` into the list in `:190`.

### N-1 · Nit · `btc_web/callbacks/routing.py:683-687` — comment describes the deleted guard

Two sentences in the prefetch block header are now false:

* `:683-684` "Each callback writes only its own lazy-div — **no multi-output payload**, so
  React reconciles one small chunk at a time" — `_pf` is now a two-output callback. (The
  *intent* survives: the second output is one boolean, so React still reconciles one tab's
  chunk per response. The sentence just no longer says that.)
* `:685-687` "Guards: only load when current content is still the 'Loading...' placeholder
  (idempotent with click-triggered lazy load)" — the guard is now the `{tab}-loaded` flag.

**Fix:** reword to "writes only its own lazy-div plus a one-boolean flag" and "Guards: only
load when `{tab}-loaded` is falsy".

### N-2 · Nit · `btc_web/callbacks/routing.py:648-651` — pre-existing docstring inaccuracies left in place

`_register_lazy_tab`'s docstring says the prefetch callback "fires once on
`prefetch-interval.n_intervals` change" — the actual Input is `{tab_id}-prefetch-iv`
(`:692`) — and asserts "The two never race", which is not quite true (see hunt item 1: they
can both fire in a narrow window and both write the same cached content). Both statements
predate this diff and neither affects behaviour; flagged only because the diff rewrote the
neighbouring comments and left these.

### N-3 · Nit · `btc_web/test_lazy_flag_payload.py:81-88` — writer selection depends on registration order

`_callback_for_output` returns the **first** callback whose output set contains
`(cid, prop)`. `bubble-lazy.children` has two writers, so the three behaviour tests
(`:143`, `:152`, `:161`) get `_lazy_load` only because `_register_lazy_tab`'s loop runs
before `_register_prefetch`'s. Verified today that it does select `_lazy_load`
(signature `(tab, loaded, _tid='bubble')`). If the loops were ever swapped the tests would
raise `TypeError` on `fn("bubble", True)` against `_pf(n, ready, loaded, active)` — a loud
failure, not a silent pass, so this is genuinely a nit. **Fix (optional):** filter on
`entry["callback"].__wrapped__.__name__ == "_lazy_load"`, or take an expected-arity
parameter.

### N-4 · Nit · `btc_web/test_lazy_flag_payload.py:191` — only the non-bubble initial tab is asserted

`test_eagerly_rendered_tab_starts_loaded_and_others_do_not` builds `_build_layout("heatmap")`
only, so the assertion set covers `auto-y-grid-loaded is False` but never the bubble branch,
where the invariant that actually matters for auto-Y is `auto-y-grid-loaded is True` **and**
`auto-y-grid.data is not None`. I verified that branch by hand across all 19 entry paths and
it holds. **Fix:** `@pytest.mark.parametrize("initial", ["heatmap", "bubble"])` and assert
`stores["auto-y-grid-loaded"].data == (stores["auto-y-grid"].data is not None)`.

### N-5 · Nit · deploy note — `/_dash-dependencies` signature changes for 21 callbacks

Dash's `create_callback_id` (`dash/_utils.py:138`) hashes **only the Inputs**, so the
`@<hash>` suffixes are unchanged by this diff. But the *key shape* changes for all 21
touched callbacks: single-output `bubble-lazy.children@<h>` becomes multi-output
`..bubble-lazy.children@<h>...bubble-loaded.data@<h>..`, and `auto-y-grid.data` becomes
`..auto-y-grid.data...auto-y-grid-loaded.data..`. A browser holding a stale dependency map
across the deploy will POST the old keys and get 500s on the lazy loaders — memory
`feedback_stale_dash_deps.md`, the "user interactions silently do nothing" failure. The
mitigation is already in place (`@server.after_request` sets `Cache-Control: no-cache` on
`/_dash-layout` and `/_dash-dependencies`), and it resolves on reload; this is the normal
cost of any callback-graph change. Worth one line in the deploy note, nothing more.

---

## Observations (not findings)

* **`/10.N` typed directly never opens FAQ item N — on master as well as on this branch.**
  When FAQ is the eager initial tab, nothing changes `faq-lazy.children` (master) or
  `faq-loaded.data` (branch), so `open_faq_item`'s `prevent_initial_call=True` means it
  never runs. Model Info is immune because `_mi_spa_open` (`routing.py:775-799`) carries a
  second `Input("url", "pathname")`; FAQ has no such fallback and is not covered by any
  test. Pre-existing, unchanged by this diff, out of scope — but a candidate follow-up, and
  the one-line fix is symmetric with Model Info (add `Input("url", "pathname")` +
  `allow_duplicate`).
* **The invariant is now structural rather than observational**, which is a genuine
  improvement in reviewability: `LAZY_TABS` collapsed two hand-maintained 10-tuples into
  one, and a future 11th lazy tab gets its Store, its loader and its prefetcher from a
  single edit instead of three that fail silently apart.
* **`_pf` still uploads `State("main-tabs", "active_tab")`** and all 10 `_lazy_load`
  callbacks still round-trip on every tab switch. Both are correct and cost a few bytes;
  noted only to record that the diff's remaining POST count is deliberate (the recon report
  classified count reduction as R2/R4, explicitly out of this slice).

---

## What I verified with tools, in this worktree, today

Everything below is a fresh measurement against the branch's live app, not a re-run of the
gate results supplied in the brief.

| check | result |
|---|---|
| layout invariant `flag == (content is not placeholder)`, 10 tabs × 19 entry paths | **190/190 pass** |
| `auto-y-grid-loaded == (auto-y-grid.data is not None)`, 19 entry paths | **19/19 pass** |
| exactly one tab eager-rendered per path, and it `== main-tabs.active_tab` | **19/19 pass** |
| server callbacks writing `*-lazy.children` | **20** (10 `_lazy_load` + 10 `_pf`) |
| per tab, writers of `{tab}-lazy.children` == writers of `{tab}-loaded.data` | **10/10 == `[_lazy_load, _pf]`** |
| server Input/State refs to `*-lazy.children` | **0** |
| clientside readers of `*-lazy.children` | **12**, all untouched by the diff |
| consumers of the new flags | 21 `State`, **1 `Input`** (`faq-loaded → faq-accordion.active_item`) |
| `set_props` targets among `*-lazy` Divs | **none** |
| residual references to `_is_loading_placeholder` in `btc_web/` + `docs/` | **0** |
| `_uploaded_refs()` non-vacuity | **1,386 refs** (986 State / 400 Input) over 96 callbacks |
| `_callback_for_output("bubble-lazy","children")` resolves to | `_lazy_load(tab, loaded, _tid='bubble')` |
| `app.callback_map` vs `GLOBAL_CALLBACK_MAP` | 251 vs 96, **overlap 0** — the new file picked the right surface |
| import cycle `routing → layout` | none (`layout/__init__.py` imports nothing from `callbacks`) |

Taken as given from the brief and not re-run: full suite (2960 passed, 1 pre-existing
out-of-scope failure), the 17 new tests' RED-first watch, restore E2E 30/30, the other-E2E
no-regression comparison, and the Playwright payload gate (POST counts and download bytes
identical; upload 1129→191 KB plain, 3826→318 KB share restore, 2294→19 KB per tab switch).

---

## Recommendation

**Ship.** The diff does what the recon report's R1 said it would — it changes *what a
callback carries*, not *when it fires* — and the one guard it replaced is now enforced by
construction on both the layout side and the callback side rather than by re-deriving it
from a megabyte of uploaded DOM. Fold M-1 (a two-number edit in `CLAUDE.md`) and, if
convenient, N-1 (two stale sentences) before the ship commit; N-2 through N-5 are
opportunistic. Add the N-5 line to the deploy note.
