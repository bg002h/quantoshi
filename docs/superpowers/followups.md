# Follow-ups

Items deferred out of a plan, each with an **owning phase** so burndown is a
grep rather than a memory. An item whose owning phase has passed is overdue,
not deferred.

Severity follows the project rule: **Critical** / **Important** block a gate;
**Minor** / **Nit** are recorded and batched.

---

## Open

### F-1 — Greedy Select is silently skipped by every refit run
**Severity:** Important · **Owning phase:** next `btc_core/` param maintenance
· **Found:** 2026-08-07, during spl Phase 1 Task 3 · **Pre-existing, not
introduced by that work**

`tools/refit_all_ppl.py:61` stores a command-line flag *inside the script path*:

```python
("Greedy Select",  "tools/fit_grdy.py --mode=de"),
```

Line 86 then does `os.path.exists(script_path)`, which cannot resolve
`"tools/fit_grdy.py --mode=de"` as a filename. The entry is skipped with no
error. It is the only entry in `SCRIPTS` whose path fails to resolve.

**Consequence:** `grdy` has never been refit by the monthly job. Its class
attributes in `btc_core/_basis.py` are frozen at whatever the last manual
`tools/fit_grdy.py --update` wrote, while every other model's have moved.

**Verify in one command:**
```bash
btc_venv/bin/python3 tools/refit_all_ppl.py --dry-run | grep MISSING
```

**Fix sketch:** carry the args separately — `("Greedy Select",
"tools/fit_grdy.py", ["--mode=de"])` — and join them at the subprocess call,
so the existence check sees a real path. Any other entry that grows a flag
later then cannot reintroduce the bug.

**After fixing:** `grdy`'s parameters will change on the next run. `btc_core/`
param changes invalidate nothing automatically (see the cache-invalidation
table in `CLAUDE.md`), so a `redis-cli FLUSHDB` on prod is required, followed
by `generate_citadel_cache.py`.

---

### F-2 — `logi` sigma bands are fitted against an unmasked residual set
**Severity:** Important · **Owning phase:** next `btc_core/` param maintenance
· **Found:** 2026-08-07, during spl Phase 1 Task 3 · **Pre-existing**

`LogisticSCurveModel.__init__` masks its fit to `price_years >= T_MIN`, but the
`logi` branch in `tools/fit_shrinking_sigma.py:219` applies no mask. The sigma
bands are therefore fitted against a **different residual set** than the model
itself was fitted on — the pre-`T_MIN` residuals, which are large and are
exactly what the constructor excludes, are included in the sigma fit.

**Consequence:** `logi`'s quantile band widths are wrong. Silent — the bands
render, they are simply fitted to the wrong residuals.

**Verify in one command:**
```bash
btc_venv/bin/python3 -c "
import pathlib, inspect, sys; sys.path.insert(0,'.')
from btc_core import LogisticSCurveModel as L
s = pathlib.Path('tools/fit_shrinking_sigma.py').read_text()
i = s.index('model_name == \"logi\"'); j = s.index('elif model_name ==', i+10)
print('ctor masks T_MIN :', 'T_MIN' in inspect.getsource(L.__init__))
print('branch masks     :', 'T_MIN' in s[i:j])"
```

**Fix sketch:** mirror the `spl` branch at `tools/fit_shrinking_sigma.py:225-240`,
which does this correctly: import `T_MIN`, compute residuals from the model's
own `_model_log10`, and `return np.where(t >= T_MIN, resid, np.nan)`, letting
the existing downstream NaN drop (`:252`) handle the excluded window.

**Check the other branches while you are there.** Only `plo`, `sexp` and `spl`
were confirmed to mask. The remaining branches (`pl`, `exp`, `lppl`, `hybppl`,
`hybppl_dd`, `eppl`, `grdy`, `gomp`, `bpl`) were **not** audited — whether each
needs a mask depends on whether its own constructor applies one, so this is a
per-model check, not a blanket edit.

---

### F-3 — Generalise the `SCRIPTS` path assertion once F-1 lands
**Severity:** Minor · **Owning phase:** immediately after F-1 · **Depends on
F-1**

`btc_web/test_spl_registration.py` asserts that the **`spl` entry's** script
path resolves on disk. It was deliberately scoped to that one entry, because
generalising it today would go red on F-1 — and forcing an unrelated fix
through a failing test is the wrong way to raise it.

Once F-1 is fixed, widen the assertion to every entry in `SCRIPTS`. That closes
the whole class of bug: a flag appended to any script path becomes a red test
instead of a model that silently stops being refit.

Note the shape is worth remembering — the substring assertion `"fit_spl.py" in
src` was blind to precisely the bug (F-1) that was live three lines away in the
same file it was testing.

---

### F-4 — MC free-tier badge disagrees with what the server charges
**Severity:** Important · **Owning phase:** next payment/MC touch · **Found:**
2026-08-07, during spl Phase 1 Task 4 review · **Pre-existing**

`btc_web/assets/_mc_cost_consts.js` hardcodes the free-tier model list, and it
has drifted from the server's derived list. Both directions are wrong:

| | |
|---|---|
| UI says **free**, server **charges** | `exp`, `lppl` |
| UI says **paid**, server gives free | `ecfg_1d_1u` |

**Consequence:** a user selecting `exp` or `lppl` for Monte Carlo sees no
payment prompt in the UI's cost display, then hits a charge. The reverse case
is milder — `ecfg_1d_1u` is free but advertised as paid, so it is simply never
chosen for free use.

`spl` is in neither list and is therefore unaffected — it is correctly paid on
both sides. That is why this was not introduced by the spl work.

**Verify in one command:**
```bash
btc_venv/bin/python3 - <<'EOF'
import sys, json, re, pathlib
sys.path[:0] = ['btc_web', '.']
import mc_cache
js = pathlib.Path('btc_web/assets/_mc_cost_consts.js').read_text()
client = set(json.loads(re.search(r'"CACHED_MODEL_KEYS"\s*:\s*(\[[^\]]*\])', js).group(1)))
server = set(mc_cache._derive_cached_model_keys())
print('UI free / server charges:', sorted(client - server))
print('UI paid / server free   :', sorted(server - client))
EOF
```

**Fix sketch:** the JS list must be *generated* from `mc_cache._derive_cached_model_keys()`,
the way `_colors_generated.js` is generated from `colors.py` — not hand-kept.
Any hand-maintained mirror of a server-side list will drift again. If a
generator is too much for now, at minimum add a test asserting the two sets are
equal, so the next drift is a red suite rather than a billing surprise.

---

### F-5 — `/mi` and `/faq` cache for 24h, so card edits are invisible to returning readers
**Severity:** Minor · **Owning phase:** next static-pages touch · **Found:**
2026-08-07, Task 7 · **Pre-existing**

`btc_web/static_pages.py:328` sets `Cache-Control: public, max-age=86400` on
the static `/mi` and `/faq` routes. The Dash equivalents (`/9`, `/10`) send
`no-cache, no-store, must-revalidate`.

**Consequence:** after a deploy that changes a Model Info card, anyone who
loaded `/mi` in the previous 24 hours keeps seeing the old copy — including
missing images for assets added in the same deploy. Observed during Task 7: a
first `/mi.30` load replayed a pre-Task-6 copy with the profile figure absent;
cache-busting rendered both.

The long TTL is deliberate and worth keeping — these pages are rendered once at
import, so they only change on deploy, and the bandwidth saving matters for the
onion service.

**Fix sketch:** keep the TTL but add a content-derived `ETag`
(`md5(_STATIC_MI_HTML)`) and honour `If-None-Match`. Returning readers then pay
one conditional request and get a 304 when nothing changed, but see new content
immediately after a deploy. Alternatively `max-age=0, must-revalidate` plus the
ETag — same freshness, marginally more requests.

**Interim:** a deploy that must be visible immediately can be announced as
"hard-refresh `/mi`", or the Dash route `/9.N` linked instead, which never
caches.

---

### F-6 — `plo`, `sexp` and `logi` are missing from `_SCANNER_ORDER`
**Severity:** Minor · **Owning phase:** next Display-Models / scanner touch ·
**Found:** 2026-08-07 by `tools/check_model_registration.py` · **Pre-existing**

`btc_web/callbacks/scanner.py:22` lists 15 keys. The scanner's own filters
drop `exp`, the `cfg_*`/`ecfg_*` families, `_HYBPPL_FAMILY_EXTRAS` and the
non-active LPPL variants — but **not** `plo`, `sexp` or `logi`. Those three
reach the ordering step and miss the list, so `_scanner_sort_key`'s
`except ValueError` returns `len(_SCANNER_ORDER)` and sinks them below `mc`
instead of placing them in their Display Models slot.

Silent by construction: the rows still render, just in the wrong place, and
the `except ValueError` is exactly what stops it being an error.

**Verify in one command:**
```bash
btc_venv/bin/python3 tools/check_model_registration.py | grep scanner
```

**Fix sketch:** append `"plo", "sexp", "logi"` next to `"spl"` in
`_SCANNER_ORDER` (order within the deprioritized run is cosmetic), then delete
the three `("<model>", "scanner")` entries from `KNOWN_HOLES` in
`tools/check_model_registration.py`. That deletion is the regression test —
`TestLinterIntegrity::test_known_holes_still_reproduce` goes red if the entries
are left behind.

---

### F-7 — 4 models are instantiated for the resqr fit but never fitted

> **CLOSED 2026-08-08.** All four (`spl`, `plo`, `sexp`, `logi`) promoted to
> `RESQR_FLAGSHIP_MODELS`; 91 models now carry a bundle, up from 87. Measured
> Q10 band change at t=16 (median unchanged in every case — resqr affects
> band width, not the central fit):
>
> | | constant σ | resqr |
> |---|---|---|
> | `spl` | $33,932 | $47,625 |
> | `plo` | $36,756 | $51,843 |
> | `sexp` | $49,542 | $77,078 |
> | `logi` | $13,703 | $28,743 |
>
> The 13 models still without a bundle are **deliberate exclusions**, not
> holes: the LPPL family (10) is excluded because its 3.92yr halving-cycle
> residual periodicity is not something PWL can capture (documented at
> `tools/build_bm_model.py:48`), `qr` *is* the quantile model, `s2f` is
> non-quantized, and `ef` is conditional on its own pkl.
**Severity:** Important · **Owning phase:** next `model_data.pkl` /
`build_bm_model.py` touch · **Found:** 2026-08-07 by
`tools/check_model_registration.py` · **Pre-existing** (3 of the 4 predate
`spl`)

`tools/build_bm_model.py::_build_model_instances` builds 91 instances. Its
only consumer, `_fit_all_resqr`, iterates
`RESQR_FLAGSHIP_MODELS + _HYBPPL_CONFIG_PARAMS + _EPPL_CONFIG_PARAMS` = 87
keys. The four instances built and never used are **`plo`, `sexp`, `logi`,
`spl`** — none appears in `RESQR_FLAGSHIP_MODELS` (`build_bm_model.py:47-52`).

**Consequence:** those four carry no `_resqr` bundle, so
`btc_core/_helpers.py::_resqr_price_at` returns `None` and the caller falls
back to constant-σ *silently* (`:96-101`, documented as a fallback). Tab 1's
sigma mode defaults to `"resqr"` (`btc_web/tab_defaults.py:87`), so on the
default chart these four render constant-σ bands while every other overlay on
the same axes renders residual-QR bands. Nothing logs, nothing fails.

Either they belong in `RESQR_FLAGSHIP_MODELS`, or their construction in
`_build_model_instances` is dead code — it is one or the other, and today the
code asserts both.

**Verify in one command:**
```bash
btc_venv/bin/python3 -c "
import sys, re, pathlib; sys.path[:0]=['.','btc_web']
import tools.build_bm_model as b
src = pathlib.Path('tools/build_bm_model.py').read_text()
built = set(re.findall(r'instances\[\"([a-z0-9_]+)\"\]', src))
print('built but never resqr-fitted:',
      sorted(built - set(b.RESQR_FLAGSHIP_MODELS)))"
# built but never resqr-fitted: ['logi', 'plo', 'sexp', 'spl']
```

**Fix sketch:** decide per model. If a diagnostic model should get bands,
add it to `RESQR_FLAGSHIP_MODELS` and rebuild `model_data.pkl` (the resqr fit
runs inside `build_bm_model.py`, so `update_prices.py` picks it up on the next
daily append). If not, drop its line from `_build_model_instances` so the two
lists cannot disagree. Then add a check to
`tools/check_model_registration.py` asserting
`instances-literals ⊆ resqr scope` — the linter deliberately does **not**
gate on this today, because which of the two fixes is correct is a judgement
call, not a lint rule.

---

### F-8 — 18 fit tools still carry a private copy of the `--update` patcher
**Severity:** Minor · **Owning phase:** next `btc_core/` param maintenance ·
**Found:** 2026-08-08, while building `tools/_patch_class_attrs.py` ·
**Pre-existing**

21 `tools/fit_*.py` scripts support `--update`. Three now use the shared,
tested patcher (`fit_spl`, `fit_gompertz`, `fit_logistic` — the three that
patch `_t0`, the attribute name they collide on). The other **18** still
inline their own regex. They were deliberately left alone: they are believed
correct, none is covered by a test, and a sweeping rewrite risks breaking
`--update` for models nobody is touching.

Two of the 18 are **unscoped** — they patch the first match in the whole file
and rely on the target class happening to be first:

| tool | file patched | scoping |
|---|---|---|
| `tools/fit_lppl.py:110-115` | `btc_core/_lppl.py` | none — `re.subn(..., src, count=1)` over the whole file |
| `tools/fit_grdy.py:346-349` | `btc_core/_basis.py` | none — `re.sub(..., src, count=1)` over the whole file |

**This does not reproduce today** (which is why it is Minor, not Important):
`LPPLModel` is the first class in `_lppl.py`, and `_alpha`/`_beta` appear
exactly once in `_basis.py`. But `_lppl.py` has **11 classes each defining
`_A`, `_B`, `_C`, `_W`, `_PHI`, `_D`**, so inserting or reordering a class
above `LPPLModel` silently redirects `fit_lppl.py --update` onto a different
LPPL variant. Nothing would raise.

**Verify in one command:**
```bash
btc_venv/bin/python3 -c "
import re, pathlib
src = pathlib.Path('btc_core/_lppl.py').read_text()
cls = [(m.start(), m.group(1)) for m in re.finditer(r'^class (\w+)', src, re.M)]
hit = [m.start() for m in re.finditer(r'^    _A\s*=', src, re.M)]
own = lambda p: max((c for c in cls if c[0] < p), key=lambda c: c[0])[1]
print(f'{len(hit)} classes define _A; fit_lppl.py patches {own(hit[0])}')"
# 11 classes define _A; fit_lppl.py patches LPPLModel
```

**Fix sketch:** per tool, delete the private regex and call
`apply_and_report(path, class_name, values)` from
`tools/_patch_class_attrs.py`. Migrate in ones and twos, and for each: run the
tool **without** `--update` and diff its printed fit against the pre-migration
run (must be byte-identical), then exercise `--update` against a *copy* of the
target `btc_core/` file with the tool's `CORE_PATH` repointed at it. Do the
two unscoped tools first; the rest are cleanup.

**A second bug the shared patcher fixes on adoption.** The old
`f"{val:>11.6f}"` rendering pads the value, but the regex's own capture group
already absorbs the padding in the file — so the number moved **3 columns
right on every `--update` run**, plus 2 trailing spaces. Reproduced on
`GompertzModel._K`: col 22 → 25 → 28 → 31 across three runs with the *same*
fitted value. That is why `LogisticSCurveModel`'s attribute block currently
sits further right than `GompertzModel`'s and carries trailing whitespace.
Every unmigrated tool using the `:>Nf` idiom still creeps.

---

### F-11 — A saved colourblind palette does not survive a page reload on the chart
**Type:** defect · **Owning phase:** unscheduled · **Severity:** Important (accessibility)
· **Found:** 2026-09-06, while safety-testing the no-op bump guards ·
**PRE-EXISTING — verified identical on master and on the branch**

Select **Deuteranomaly** (`cb-brian`) on Tab 1, reload the page, and the chart
comes back painted in the **default** palette. Measured on the bubble trace:
`#D8BD65` (CB-Brian BM) before the reload, `#C48209` (default BM) after it.

This matters more than a normal cosmetic bug: the operator is colourblind and
the CB palettes exist for them specifically, so the failure mode is "the
accessibility setting silently does not stick".

**What is NOT the cause.** The bump fires: 27 POSTs on the reload carry
`bubble-first-render` in `changedPropIds`, so the palette hydration DOES
re-trigger the chart callbacks. Something downstream of that re-render is
still using the default palette — the pre-injected figure, the L1/L0 cache
key, or the palette value reaching the figure builder.

**Reproduction (committed):**

```bash
DEV=1 bash run_web.sh
btc_venv/bin/python3 scripts/check_palette_persists.py     # exits 1 today
```

**Where to start:** `palette-store` is `storage_type="local"` and hydrates
correctly (the dropdown shows the right value after reload). Suspect the
figure path: `_serve_layout` pre-injects figures built with
`palette = sd("palette-store:data")` = `"default"` (`tab_defaults.py:85`), and
the re-render after hydration may be served from a cache keyed without the
palette, or the clientside repaint may lose to the pre-injected figure.
Compare with `apply_hm_palette` and the `palette-store` guard described in
CLAUDE.md's "Centralized appearance system".

### F-9 — Time-evolving bubble model
**Type:** research idea, not a defect · **Owning phase:** unscheduled · **Raised:**
2026-08-08, out of the support-phase `spl` trial

BM today has a **fixed** support line — a pure straight line in log-log,
`log10 p = -1.5549 + 5.1215 * log10(t)`, verified straight to 5.6e-17 — with
bubbles superimposed. The idea is a version whose support and/or bubble
parameters evolve with time.

**What motivated it.** The support-phase trial (scratchpad, 2026-08-08) fitted
`spl` and a plain power law to support-phase prices only, and turned up three
things a fixed support line cannot express:

- **The floor's exponent is era-dependent.** Clusters from 2011-12..2012-11 and
  2015-09..2017-02 pooled give 4.8754. Adding the 2026 floor moves it to
  4.95–5.12 depending on the cut. A single straight line is a compromise across
  eras that differ.
- **The recent floor sits ~1.47x ABOVE its own early-era extrapolation.** Fit the
  floor on 2011–2017, project nine years, predict ~$44k; the actual 2026 floor
  is ~1.47x higher, and that ratio held at 1.46–1.48x across every start date
  swept from 2026-06-01 to 2026-07-13. The offset is robust, so it is structure
  the fixed line is missing, not noise.
- **Support phases are becoming rare.** 1,167 support days total, but only 251
  after 2018 and a 3.13-year gap between 2023-06-19 and 2026. If bubble duration
  or amplitude is drifting, a static decomposition will keep mislabelling
  phases — and the trial had to hand-define the 2026 floor by drawdown because
  BM classified none of it as support.

**The hazard to design against, learned the hard way this week.** `spl` added
ONE parameter to a power law and the data could not identify it: L pinned at
whatever bound was imposed, across seven datasets. A time-evolving BM adds
several more. Before building, decide what would falsify it and check that the
data can support that many parameters — the profile-curve method in
`tools/analyze_spl.py` sections [0]/[0b] is the right instrument, and it should
be run on the proposed parameterisation BEFORE the model is written, not after.

**Cheap first step.** Fit the support line separately per era (the three
clusters above) with a common exponent and free intercepts, versus free
exponents. If the free-exponent version does not beat the common-exponent one
on a boundary-corrected LRT, the floor is one power law and only its *level*
drifts — which is a much smaller change than a full time-evolving model.

---

## Closed

### F-10 — Tab-1 view pills Percentile and Occupancy have no deep link
**CLOSED 2026-09-04** (`/1.4`, `/1.5[.T[.W]]`, JS tab map regex, `test_bub_deep_links.py`) · **Severity:** Minor · **Found:**
2026-09-04, while adding the Occupancy pill · **Pre-existing for Percentile**

`callbacks/routing.py::deep_link_bub_view` handles `/1.2` (Forward CAGR) and
`/1.3` (Residuals) only. The Percentile pill (`bub-view-pctile`, added
2026-08-22) never got `/1.4`, and the Occupancy pill (`bub-view-occ`) has no
`/1.5`. Share links still work — `bub-view-mode` is snapshot-encoded — so
only the short URL form is missing.

**Fix sketch:** add `/1.4` → `"percentile"` and `/1.5` → `"occupancy"`
branches; the handler's Output list must grow the two wraps + two outlines
(+ `bub-occ-ctl-wrap`) that `toggle_bub_view` already sets, and the
`/1.5.T.W` suffix could carry tail/window like `/1.2.N.B` carries CAGR years.

