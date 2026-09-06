# Recon: what composes the Tab-1 share-link POST burst, and what is removable

**Date:** 2026-09-06 · **Repo:** `/scratch/code/bitcoinprojections` · **Branch:** `master` @ `7dd7b4b`
**Scope:** RECON ONLY. No application code was edited. Nothing was deployed.
**Question:** a share-link restore of Tab 1 fires ~160 POSTs to `/_dash-update-component`. What exactly composes that burst, and which parts are removable — ranked by POSTs saved against implementation risk?

**Headline (one line):** the POST *count* is dominated by a `bubble-first-render` re-bump loop (39 of 163 POSTs), but the POST *cost* is dominated by something else entirely — **10 lazy-tab callbacks that upload their own serialized component subtree as `State` on every fire, 4.4 MB of the 5.1 MB uploaded on a share restore, 86%, for 43 empty responses** — and that cost recurs at **~2.3 MB per tab switch for the rest of the session**, which no restore-side fix touches.

---

## 1. Method — exactly what was measured

Dev server (`DEV=1`, no reloader, no Redis, no Markov/MC, prewarm skipped):

```bash
cd /scratch/code/bitcoinprojections
lsof -ti :8050 | xargs -r kill -9
DEV=1 nohup bash run_web.sh > /tmp/quantoshi_dev.log 2>&1 &
```

Three Playwright (Firefox, headless, 1400×900) captures. Every `page.on("request")` for
`/_dash-update-component` was recorded with `request.post_data` parsed for `output`,
`inputs`, `changedPropIds`, and request byte length; the paired `page.on("response")`
recorded HTTP status, body byte length, and the set of keys in the JSON `response`
object. A response is counted **empty** (= work done for nothing) when the status is
`204` **or** the body's `response` object has zero keys — i.e. every output was
`no_update` / `PreventUpdate`.

Each run navigates, waits for `#bubble-graph .js-plotly-plot`, then settles
(**poll until 6 s with no new POST, 45 s wall cap**) so the full prefetch tail is
captured. Harness + raw captures live in the session scratchpad
(`capture.py`, `tabswitch.py`, `plain.jsonl`, `share.jsonl`, `share2.jsonl`,
`tabswitch.jsonl`) — **not** in the repo.

| run | URL | POSTs |
|---|---|---|
| `plain` | `http://127.0.0.1:8050/1` | 96 |
| `share` (A) | `/1#q4:3744c860:H4sIAA_AnWoC_…` (Occupancy, tail 5 %, window 1 yr, QR, x 2012–2027) | 163 |
| `share2` (B) | same link, repeat run | 147 |
| `tabswitch` | `/1`, settle 20 s, then click tab 2 → tab 3 → tab 1 | 96 + 14 + 14 + 11 |

**Reproduce:**
```bash
btc_venv/bin/python3 <scratch>/capture.py plain 'http://127.0.0.1:8050/1'
btc_venv/bin/python3 <scratch>/capture.py share '<the q4 link>'
btc_venv/bin/python3 <scratch>/tabswitch.py
btc_venv/bin/python3 <scratch>/final.py     # inventory table
```

### Which numbers are dev-only

* **Absolute POST counts run high vs. the operator's earlier figures** (dev plain 61, dev
  share 96) purely because of the 6 s-quiet settle window — mine captures the prefetch
  tail and the FAQ/Model-Info lazy loads that a shorter wait misses. The **plain : share
  ratio** is the stable quantity: operator 1.57×, mine 1.70× / 1.53×. Prod's 95 → 161 is
  1.69×. The *shape* transfers; treat my absolute counts as dev-with-a-long-tail.
* **2 × HTTP 500 per run** on `citadel-graph.figure` are **dev-only artefacts**:
  `dash.exceptions.MissingLongCallbackManagerError` — `diskcache` is not in the dev venv.
  Prod has it. Not a defect; excluded from all analysis below.
* **No Redis (L0/L2)** in dev, so server compute per POST is worse than prod. Irrelevant
  to counts and byte volumes, which is what this recon measures.
* **No Markov/MC** in dev — the ~8 MC option/upload callbacks still fire and still
  round-trip, so their counts are real; their payloads would be larger on prod.
* Byte volumes are **request/response bodies as sent**, pre-gzip. nginx gzips responses;
  request bodies from the browser are **not** compressed, so the upload numbers are what
  actually crosses the wire.

### Run-to-run stability

Per-callback counts vary by ≤1 between runs A and B for every group. The empty-response
*fraction* is stable at **56 % on both share runs** and 48 % on plain. Totals:

| run | POSTs | empty | upload | download | span | peak |
|---|---|---|---|---|---|---|
| plain | 96 | 46 (48 %) | 1,129 KB | 2,107 KB | 9.4 s | 28/s |
| share A | 163 | 91 (56 %) | 5,096 KB | 2,214 KB | 14.0 s | 36/s |
| share B | 147 | 82 (56 %) | 3,826 KB | 1,597 KB | 12.9 s | 31/s |

---

## 2. Inventory

### 2a. Grouped inventory (share run A, groups are disjoint and exhaustive)

| group | plain n (empty) | share n (empty) | Δ | share upload | share download |
|---|---|---|---|---|---|
| **`*-lazy.children`** (10 lazy + 10 prefetch callbacks) | 30 (21) | **52 (43)** | +22 | **4,399 KB** | 809 KB |
| **all other** (options builders, MC controls, warn dialogs, ticker, lots, scanner…) | 38 (10) | 62 (8) | +24 | 92 KB | 236 KB |
| **hidden Tab-1 views** (`bub-cagr` / `bub-resid` / `bub-pctile`) | 6 (6) | **16 (16)** | +10 | 17 KB | 0 KB |
| **non-active tab charts** (hm/dca/ret/sc/cp/lev) | 12 (2) | 12 (10) | 0 | 61 KB | 8 KB |
| **`update_bubble` [8 outputs]** | 3 (2) | **5 (5)** | +2 | 33 KB | 0 KB |
| **`apply_tab_bubble` [66 outputs]** | 2 (2) | 5 (2) | +3 | 109 KB | 6 KB |
| **`bub-occ-graph.figure`** (the view the user actually sees) | 1 (1) | 4 (2) | +3 | 4 KB | **1,155 KB** |
| **CTA** `[bubble-graph.figure + cta-status + bub-redraw-tick]` | 2 (0) | 4 (2) | +2 | 6 KB | 0 KB |
| **`auto-y-grid.data`** | 1 (1) | 2 (2) | +1 | **292 KB** | 0 KB |
| **`faq-accordion.active_item`** | 1 (1) | 1 (1) | 0 | **82 KB** | 0 KB |
| **TOTAL** | **96 (46)** | **163 (91)** | **+67** | **5,096 KB** | **2,214 KB** |

### 2b. Per-callback detail — the repeat offenders (share A)

| callback (outputs, abbreviated) | plain | share A | share B | share empty | share up | share down |
|---|---|---|---|---|---|---|
| `[4] s2f-warn-dialog.displayed … model-warn-dismissed.data` (×4 tab variants) | 7 (0e) | 9 (0e) | 9 (0e) | 0 | 9 KB | 1 KB |
| `bub-cagr-graph.figure` | 4 (4e) | 7 (7e) | 7 (7e) | **7** | 7 KB | 0 |
| `faq-lazy.children` | 3 (2e) | 6 (5e) | 6 (5e) | 5 | 4 KB | 83 KB |
| `bub-decomp-active-formula.children` | 2 (0e) | 6 (0e) | 6 (0e) | 0 | 12 KB | 0 |
| `model_info-lazy.children` | 3 (2e) | 6 (5e) | 5 (4e) | 5 | 4 KB | 421 KB |
| `[3] scan-output-field + scan-results + scan-price-hint` | 2 (0e) | 6 (0e) | 4 (0e) | 0 | 5 KB | 46 KB |
| `bub-decomp-formula.children` | 2 (0e) | 5 (0e) | 5 (0e) | 0 | 6 KB | 0 |
| `[3] bub-decomp-components.options + warning + body` | 2 (0e) | 5 (0e) | 5 (0e) | 0 | 10 KB | 1 KB |
| `stack / leverage / citadel / supercharge / retire / dca / heatmap -lazy.children` | 3 each | 5 each | 4–5 each | 4 each | 3 KB each | 6–92 KB |
| **`bubble-lazy.children`** | 3 (3e) | **5 (5e)** | 4 (4e) | **5 — all** | **4,370 KB** | **0** |
| `[8] bubble-graph.figure … bub-yrange.value` (`update_bubble`) | 3 (2e) | 5 (5e) | 4 (4e) | **5 — all** | 33 KB | 0 |
| `[66] bub-qs.value … bubble-snap-applied.data` (`apply_tab_bubble`) | 2 (2e) | 5 (2e) | 4 (3e) | 2 | 109 KB | 6 KB |
| `bub-model-show.options` (`_update_bub_swatches`) | 2 (0e) | 5 (0e) | 4 (0e) | 0 | 2 KB | 67 KB |
| `bub-resid-graph.figure` | 1 (1e) | 5 (5e) | 4 (4e) | **5 — all** | 6 KB | 0 |
| `[3] bubble-graph.figure + cta-status + bub-redraw-tick` (CTA) | 2 (0e) | 4 (2e) | 3 (2e) | 2 | 6 KB | 0 |
| `bub-occ-graph.figure` | 1 (1e) | 4 (2e) | 3 (2e) | 2 | 4 KB | **1,155 KB (2 × 577)** |
| `bub-pctile-graph.figure` | 1 (1e) | 4 (4e) | 3 (3e) | **4 — all** | 3 KB | 0 |
| `[2] bub-yrange.max + bub-yrange.marks` | 1 (0e) | 3 (0e) | 3 (0e) | 0 | 1 KB | 0 |
| `auto-y-grid.data` | 1 (1e) | 2 (2e) | 2 (2e) | **2 — all** | **292 KB** | 0 |
| `faq-accordion.active_item` | 1 (1e) | 1 (1e) | 1 (1e) | 1 | **82 KB** | 0 |
| `[10] snapshot-state-store … restore-heatmap-fig` (`restore_from_url`) | 1 (1e) | 1 (0e) | 1 (0e) | 0 | 2 KB | 103 KB |
| `[32] main-tabs.active_tab … snapshot-lots.data` (`apply_globals`) | 1 (1e) | 1 (0e) | 1 (0e) | 0 | 15 KB | 1 KB |

(The remaining ~30 groups fire exactly once on both loads — MC option builders, MC
upload handlers, lots, price ticker, link history, scanner hints, citadel scenario
loader. They are the necessary floor and are not itemised.)

### 2c. Timing / burst shape (share A)

```
POSTs per 500 ms bucket
 t= 3.0s  #                        1     ← page ready
 t= 4.0s  ############            12     ← restore_from_url responds @4.07s
 t= 6.0s  #############           13
 t= 7.0s  ############            12
 t= 7.5s  ################        16
 t= 8.0s  #######################  23    ← peak; occupancy figure built @8.14s
 t= 8.5s  #############           13
 t=10.0s  ##########              10
 t=14.0s  ##########              10
 t=17.5s  #                        1
```

* `restore_from_url` responds at **t = 4.07 s**. **158 of 163 POSTs (97 %) happen after
  that.**
* The figure the user actually sees (`bub-occ-graph`, 577 KB) lands at **t = 8.14 s** —
  and is then **rebuilt identically at t = 8.39 s** (a second 577 KB response).
* **86 POSTs and 4,367 KB of upload happen after the visible chart is already on screen.**
  That is the tail nginx's `limit_req` was rejecting.

---

## 3. Classification of every group

### (i) Genuinely necessary — ~40 POSTs

`restore_from_url` ×1 · `apply_globals` ×1 · first productive `apply_tab_bubble` ×1 ·
first `bub-occ-graph` build ×1 · price ticker ×1 · 9 non-active tab lazy populates ·
first fire of each option/formula builder (~9) · MC option + upload handlers (~8) ·
lots/scanner/link-history/citadel-scenario (~5) · first fire of the 5 non-active chart
callbacks that warm the L1 cache for later tab switches (~5, debatable — see (v)).

### (ii) Already gated but still round-trips — 41 POSTs, 41 empty

Server-side gates fire correctly and return `no_update`; the HTTP round-trip is spent anyway.

* **hidden Tab-1 views** — `bub-cagr` 7, `bub-resid` 5, `bub-pctile` 4 = **16 POSTs, 16 empty.**
  The 2026-09-05 view gating (`if view_mode != "cagr": return no_update`,
  `charts/__init__.py:691/757/810/862`) works exactly as designed — it stops the *rebuild*,
  not the *dispatch*.
* **`update_bubble`** — 5 POSTs, 5 empty. Confirmed in the dev journal:
  `bubble-fig SKIPPED (gate) ×2` + `SKIPPED (restore short-circuit) ×3`. On this share link
  `update_bubble` contributes literally nothing.
* **non-active tab charts** — 10 of 12 empty on the share run.
* **`bubble-lazy.children`** — 5 POSTs, 5 empty, because bubble is the active tab and was
  already populated at layout time. **These five cost 4,370 KB of upload.**
* **`auto-y-grid`** — 2 POSTs, 2 empty (`current is not None` → `no_update`), 292 KB up.

### (iii) Duplicate / redundant fire — ~30 POSTs producing a value already held

Driven by one mechanism. **`bubble-first-render.data` appears in the trigger set of 39 of
163 share POSTs (24 %)** and gets bumped 4–5 times per restore. It has **five writers**:

| # | writer | file |
|---|---|---|
| 1 | tab-switch bump (`if (!cur) out[idx]=1`) | `routing.py:43-72` |
| 2 | post-snapshot bump (`cur+1`, Input `snapshot-state-store`) | `routing.py:92-129` |
| 3 | post-lazy-load bump (Input `{tab}-lazy.children`) | `routing.py:_register_first_render_bump` ~715-732 |
| 4 | palette change → `active-tab-bump-tick` | `callbacks/charts/_clientside.py:480-508` |
| 5 | **`bub-sigma-mode.value` change — `function(mode, cur){return (cur||0)+1;}`, NO guard** | `callbacks/scanner.py:328-336` |

…and **nine consumers**: `update_bubble` (`charts/__init__.py:93`), `bub-cagr-graph`
(`:666`), `_update_bub_swatches` → `bub-model-show.options` (`:1978`), three decomp
callbacks (`_resolvers.py:312/369/455`), the scanner readout (`scanner.py:72`), and —
critically — **`apply_tab_bubble` itself** (`snapshot_cb.py:218`, `_TAB_SPECS`).

**The loop:** first-render bump → `apply_tab_bubble` fires → writes 53 controls including
**`bub-sigma-mode`** (a snapshot control: `snapshot.py:358`, `snapshot_defaults.py:74`) →
writer #5 bumps first-render again → `apply_tab_bubble` fires again → … It terminates only
because the written value converges, after **3 identical `apply_tab_bubble` runs**
(`[trace] apply_tab_bubble controls=53/64 apply_ms=0.0` × 3 in the dev journal) and 4–5
first-render bumps. Each bump fans out to all nine consumers. `bub-model-show.options`
returns the identical 13.4 KB payload five times (67 KB total); `scan-output-field` the
identical 7.6 KB six times.

### (iv) Convertible to clientside — ~3 POSTs but ~4.7 MB of upload

Not count reductions — **payload** reductions. Three callbacks take a large `State`/`Input`
purely to answer a boolean:

* `_register_lazy_tab` / `_register_prefetch` (`routing.py:633-708`) take
  `State(f"{tab_id}-lazy", "children")` **only** to evaluate `_is_loading_placeholder(current)`.
  The browser therefore uploads the tab's entire serialized component subtree — and once
  Tab 1's figure is in that subtree, **1,315 KB per fire** (208 KB before the figure lands,
  1,315 KB after). Measured: 4,370 KB across bubble's 5 fires.
* `_lazy_load_auto_y_grid` (`routing.py:572-583`) takes `State("auto-y-grid","data")` only to
  evaluate `current is not None` — **146 KB uploaded per fire**, and it is an `Input` on
  `main-tabs.active_tab`, so this recurs on **every tab switch, forever**.
* `open_faq_item` (`routing.py:885-900`) takes `Input("faq-lazy","children")` — **82 KB** —
  only to learn "the FAQ mounted", then reads `url.pathname` and returns `no_update` for
  any path that isn't `/10.N`.

### (v) Avoidable by batching — ~2 POSTs

The 2026-04-24 spec's own candidate (a), *batching `apply_globals` + `apply_tab_{active}`*
(`docs/superpowers/specs/2026-04-24-single-redraw-per-snapshot-design.md:148`), removes at
most **1–3 POSTs of 163**. It does not touch the fan-out that produces the other 39, and the
spec itself lists it as requiring the safety-bump pattern to be restructured. **Poor ratio;
do not pursue.** Candidate (b), *pre-warming Redis* (`:146`), saves **0 POSTs** — it changes
CPU per POST, and share links carry arbitrary configs so the hit rate is low. **Both of the
spec's own "if revisiting" candidates are worse than the four found here.**

---

## 4. Ranked candidate reductions

Ordered by (value ÷ blast radius). "Saved" figures are **measured** unless marked *(est.)*.

### R1 — Stop uploading component subtrees as `State`  ★ best ratio in the report

**Change:** in `_register_lazy_tab` and `_register_prefetch`, replace
`State(f"{tab_id}-lazy","children")` with a small always-mounted boolean
`dcc.Store(f"{tab_id}-loaded")` (written in the same output batch as the children). Same for
`auto-y-grid` (`State` → boolean flag) and `open_faq_item` (`Input("faq-lazy","children")` →
the same flag, or move the `/10.N` test clientside).

**Saved (measured):** **0 POSTs**, but **4,744 KB of the 5,096 KB uploaded on a share restore
— 93 %.** Share upload drops ~5.1 MB → ~0.35 MB. Plain load drops 1,129 KB → ~170 KB.
**And it fixes the recurring cost:** a measured tab switch is **14 POSTs / 2,294 KB up /
64 KB down, of which 13 POSTs are empty and 2,242 KB is these blobs** — every switch, all
session, on every device. Full breakdown of one bubble→heatmap switch:

```
 525KB up  0KB dn  EMPTY  supercharge-lazy.children
 419KB up  0KB dn  EMPTY  model_info-lazy.children
 336KB up  0KB dn  EMPTY  retire-lazy.children
 299KB up  0KB dn  EMPTY  bubble-lazy.children
 260KB up  0KB dn  EMPTY  dca-lazy.children
 146KB up  0KB dn  EMPTY  auto-y-grid.data
 108KB up  0KB dn  EMPTY  heatmap-lazy.children
  93KB up  0KB dn  EMPTY  citadel-lazy.children
  82KB up  0KB dn  EMPTY  faq-lazy.children
   7KB up  0KB dn  EMPTY  [hm-entry-yr.value … hm-exit-range.value]
   7KB up  0KB dn  EMPTY  leverage-lazy.children
   7KB up  0KB dn  EMPTY  stack-lazy.children
   4KB up 59KB dn         [heatmap-graph.figure … mc-save-tab.data]   ← real work
   1KB up  5KB dn         [hm-pill-bub.children … ]                    ← real work
```

**Blast radius: LOW — the lowest of any candidate here.** It changes *what a callback
carries*, not *when it fires*. No dispatch order, no gate, no `allow_duplicate`, no
clientside dispatch semantics. Every documented failure in
`restore_callback_architecture.md`, `tabs_2_7_fast_modal_attempt1.md`,
`feedback_task45_clientside_patch_regressed_bubble.md` and
`feedback_slider_commit_debounce_failed.md` was a *timing/dispatch* change; this is not one.

**Prior attempt:** none. No commit in the history has attempted this.
**Caveats:** (a) the guard is a real correctness guard — it stops a re-render clobbering user
state — so the flag must be written in the *same output batch* as the children, never in a
follow-up callback. (b) `_register_first_render_bump` also reads `{tab}-lazy.children`, but
**clientside**, so it costs no POST and must be left alone. (c) `open_faq_item` is
positional-deep-link machinery — verify against `test_static_pages.py` / the `/10.N` tests.

### R2 — Break the `bub-sigma-mode` → `bubble-first-render` feedback loop

**Change:** guard `callbacks/scanner.py:328-336` so a *framework* write of `bub-sigma-mode`
during restore does not bump first-render — e.g. `State("snapshot-pending","data")` and
return `no_update` while armed, or remember the last-seen mode clientside and bump only on
an actual change.

**Saved (est., ~30 POSTs):** removes 1–2 of the 4–5 first-render bumps. Each bump fans out
to 9 consumers, so ~9–18 POSTs directly, plus the second and third `apply_tab_bubble` runs
and their downstream cascade — including, plausibly, **the duplicate 577 KB occupancy
figure build.** Estimated 163 → ~130 POSTs and ~600 KB less download. **This is an estimate:
the fan-out is measured, the bump attribution is inferred from the writer/consumer graph,
not from a counter.** Instrument before committing to a number (see §5).

**Blast radius: MEDIUM.** `bubble-first-render` is load-bearing and the sigma-mode bump
exists deliberately (`bub-sigma-mode` is a `State`, not an `Input`, on `update_bubble`, to
avoid a `/_dash-dependencies` signature change — `feedback_stale_dash_deps.md`). A guard
that is too broad makes the σ-mode radio stop redrawing the chart. Needs a live browser
smoke test of the σ-mode radio, not just the suite.

**Prior attempt:** none at this specific loop. Adjacent territory failed twice —
`restore_callback_architecture.md` Attempt 4 (`ecaa07f`/`dbe264f`, reverted) and the
tabs-2-7 attempt (reverted at **`42acaf9`**) — but both were *adding* a re-trigger, whereas
this *removes* one. Note lesson #5 of that memory: "CTA is load-bearing, don't touch its
tick bump" — this is a different bump, but the same caution applies.

### R3 — Make the fast-restore path view-aware for `/1.2`–`/1.5`

**Finding:** `restore_builder.py` and `snapshot_cb.py` contain **no reference to
`view_mode` / `bub-view-mode`** (verified by grep). For an Occupancy share link,
`restore_from_url` builds and ships the **hidden Price figure** — measured
`[trace] restore-direct-build BUILT 138.7ms`, 105,750-byte response — while the figure the
user is actually looking at (`bub-occ-graph`) does not arrive until **t = 8.14 s**, ~4 s
later, via the ordinary cascade. Then it is built a second time at t = 8.39 s.

**Saved (measured):** 1 wasted 138.7 ms server build + 105.7 KB of download per
non-Price Tab-1 share link, and it would move the visible chart from t≈8.1 s to t≈4.1 s.
Not a POST-count reduction — a latency and CPU reduction.

**Blast radius: MEDIUM-HIGH.** It extends `restore_builder` to a fifth figure kind and
touches the `restore-bubble-fig` Store relay, which is exactly the subsystem
`splash_coldload_dismiss.md` records as "fragile — don't re-investigate". Needs its own
spec. **Listed for completeness; do not bundle it with R1/R2.**

**Prior attempt:** none. This gap was created by the 2026-09-04 Occupancy/deep-link work
landing on top of the 2026-04-25 restore_builder, which predates views.

### R4 — Dispatch-level gating of the three hidden Tab-1 views

**Change:** give `update_bub_cagr` / `_resid` / `_pctile` a per-view redraw tick bumped
clientside only while that view is active, instead of ~20 shared `bub-*` Inputs each.

**Saved (measured):** **16 POSTs, all 16 currently empty** (7 + 5 + 4), ~17 KB up.

**Blast radius: HIGH relative to the 16 POSTs it buys.** It is a dispatch restructure of
four chart callbacks, and it collides with the deep-link surface
(`test_bub_deep_links.py`, `test_bub_view_gating.py`, `test_bub_view_modes.py`,
`bub_views.py`). This is the shape that failed in
`feedback_task45_clientside_patch_regressed_bubble.md` (reverted via the `3b58f09` revert
chain) and `feedback_slider_commit_debounce_failed.md` (reverted at `c19ebd0`) — clientside
rewrites that passed static review and broke live UI. **Not recommended.**

### R5 / R6 — the 2026-04-24 spec's own two candidates

* **Batch `apply_globals` + `apply_tab_{active}`** — ≤3 POSTs saved, restructures the safety
  bump. **Not recommended.**
* **Pre-warm Redis with common share-link configs** — **0 POSTs saved**, low hit rate on
  arbitrary share configs. **Not recommended.**

---

## 5. Open questions / what I could not measure

1. **R2's exact saving is unmeasured.** I proved five writers and nine consumers of
   `bubble-first-render`, and that 39 POSTs carry it. I did **not** capture the store's
   successive *values*, so "4–5 bumps" is read off timestamp clusters, and the
   per-writer attribution is inferred. **Before implementing R2, land a throwaway
   clientside `console.log` on each of the five writers and count.** That is a
   15-minute measurement that turns the only estimate in this report into a number.
2. **Prod is not dev.** Every number here is dev. Prod has Redis (fewer cold builds,
   different response sizes), Markov/MC (more MC callbacks fire), prewarm (different L1
   hit pattern), and 5 gunicorn workers. `tabs_2_7_fast_modal_attempt1.md` records a
   dev 88 % / prod 13 % gap on a restore change — **treat dev agreement as necessary,
   never sufficient.** R1 is the least exposed to this (it changes payloads, not timing).
3. **Response gzip.** Download figures are pre-compression; nginx gzips them. Upload
   figures are **not** compressed and are the honest wire cost.
4. **Mobile.** The lazy-tab `State` upload is worst exactly where it hurts most — a phone
   on cellular uploading 1.3 MB to be told `no_update`. Not measured on a real device.
5. **The duplicate occupancy build** (2 × 577 KB at t=8.14 s and t=8.39 s) is attributed to
   the first-render cascade by triggers (`bub-xscale.value` + `bub-xrange.value`, then
   `bub-model-show.value` + `bub-xrange.value`) but I did not prove R2 removes it.
6. **Not swept:** whether other tabs' share links show the same shape (only `/1` measured),
   and whether the ~30 single-fire "necessary floor" callbacks contain further redundancy.

---

## 6. Recommendation

**Yes — but only R1, and it is not really about the restore burst.**

The framing that started this ("~160 POSTs on restore") points at the wrong quantity. The
POST *count* is mostly cheap: 91 of 163 return nothing and cost a few KB each, and the
count-reduction candidates (R2, R4) all require touching the restore/dispatch machinery
that has a **4-failures-out-of-5-attempts** track record in this repo. Meanwhile the
measurement turned up something the burst framing hides entirely: **the app uploads
2.3 MB per tab switch and 4.4 MB per share restore in order to receive `no_update`**, and
that is a pure `State`-shape bug with no timing component at all.

Against the operator's stated ALARA preference — minimize bandwidth, server RAM, server
CPU — R1 is roughly a **93 % cut in upload bytes** on the exact interaction users perform
most (tab switching), for a change that cannot alter dispatch order because it does not
touch dispatch.

### Smallest first slice

**`auto-y-grid` alone.** One callback, `routing.py:572-583`, seven lines. Replace
`State("auto-y-grid","data")` with a boolean flag written in the same batch. Saves
**146 KB per tab switch and 292 KB per share restore**, measured. If the pattern holds
(it will — it is the same shape), extend it to the ten `_register_lazy_tab` / ten
`_register_prefetch` callbacks in a second commit, then `open_faq_item` in a third.

**Verification gate for the slice** (mirroring the discipline that made the 5th restore
attempt succeed): re-run `capture.py plain` and `tabswitch.py`, assert total upload bytes
fall and **POST counts are unchanged** — an unchanged count is the proof that dispatch was
not perturbed. Then the E2E restore suite (`-n 0`), then prod.

**Do not bundle R2, R3 or R4 into that cycle.** R2 is worth a follow-up *after* the
five-writer instrumentation in §5.1 gives it a real number. R3 is a genuine gap worth its
own spec. R4 is not worth its blast radius.

---

### Reference: files and lines cited

| concern | location |
|---|---|
| lazy tab load + prefetch (`State(children)`) | `btc_web/callbacks/routing.py:633-708` |
| `auto-y-grid` lazy load (`State(data)`, 146 KB) | `btc_web/callbacks/routing.py:572-583` |
| `open_faq_item` (`Input(faq-lazy.children)`, 82 KB) | `btc_web/callbacks/routing.py:885-900` |
| first-render writer 1 (tab switch) | `btc_web/callbacks/routing.py:43-72` |
| first-render writer 2 (post-snapshot) | `btc_web/callbacks/routing.py:92-129` |
| first-render writer 3 (post-lazy-load) | `btc_web/callbacks/routing.py:~715-732` |
| first-render writer 4 (palette tick) | `btc_web/callbacks/charts/_clientside.py:480-508` |
| **first-render writer 5 (σ-mode, unguarded)** | `btc_web/callbacks/scanner.py:328-336` |
| first-render consumers | `charts/__init__.py:93,666,1978` · `_resolvers.py:312,369,455` · `scanner.py:72` · `snapshot_cb.py:218` |
| `bub-sigma-mode` is a snapshot control | `btc_web/snapshot.py:358` · `btc_web/snapshot_defaults.py:74` |
| hidden-view gates (working as designed) | `btc_web/callbacks/charts/__init__.py:691,757,810,862` |
| `apply_globals` / `apply_tab_*` factory | `btc_web/callbacks/snapshot_cb.py:235-265` |
| restore fast path (no view awareness) | `btc_web/restore_builder.py` · `btc_web/callbacks/snapshot_cb.py` |
| spec's own two "if revisiting" candidates | `docs/superpowers/specs/2026-04-24-single-redraw-per-snapshot-design.md:146,148` |
