# `spl` — continuity: whole-diff review, then Phase 2

**Written** 2026-08-08 · **Branch** `time-basis-toggle-phase2b` · **Prod is on
this branch** at `098b5e6` (switched from `time-basis-toggle`, which is what it
tracked before).

Phase 1 is **complete, reviewed task-by-task, and deployed live** to
quantoshi.xyz. Two things remain, in this order.

---

# PART 1 — The whole-diff review (do this first)

## Why it is still open

Every task passed its own review, and every fold passed a scoped re-review. But
a **broad review of the whole branch has never run**, and it was deliberately
moved to *after* the prod deploy (user decision, 2026-08-07) so the tier could
be chosen with the deployed behaviour in hand.

**The tier is the user's call: fable, opus, or sonnet.** Ask; do not assume.

## What has already been reviewed — do NOT re-audit

Each of the 7 tasks got an independent review at opus, and each fix round got a
scoped re-review. Findings were folded and re-verified. Handing a reviewer the
whole diff without this list wastes most of its budget re-deriving settled work.

| commits | what | reviewed |
|---|---|---|
| `e22593b` `884bdae` | Task 1 — model class | ✅ + fold |
| `b0e7e94` | Task 2 — `fit_spl.py` | ✅ |
| `89ffb4f` `eed84e6` | Task 3 — pipeline integration | ✅ + fold |
| `069e9c9` | Task 4 — app registration | ✅ |
| `41c1926` `90ea5e2` | Task 5 — Model Info card | ✅ + fold |
| `b06ac22` `ff27b6c` | Task 6 — figures + registries | ✅ + fold |
| `1ec39f5` | Task 7 — verification + 4 follow-ups | ✅ |

Reports and diffs are under `.superpowers/sdd/2026-08-07-spl-phase1/`
(git-ignored, still on disk): `task-N-report.md`, `review-*.diff`, `fold-*.diff`,
`progress.md` (the ledger), `linter-report.md`, `patcher-report.md`.

## What has NOT been reviewed — point the review HERE

These are controller-authored or tooling commits. Nobody has independently read
them:

| commit | what | why it matters |
|---|---|---|
| `f0a1f82` | data: phase2b CSVs/pkl brought up to prod | touched 6 data-path files; also picked up `update_prices.py`, whose absence made `daily_update.sh`'s `SETTLE_LAG=1` a silent no-op on this branch |
| `b21e3c7` `7508f9e` | analysis + spec rebuilt on the instability finding | §3 retracts "fail to reject"; §4's "cap never binds" corrected |
| `c6c5f12` | spec §7.1a — card pinned to two data windows | reverses a requirement the brief AND a passing review had both demanded |
| `7135172` `ef0df7a` | docstring + guard-comment corrections | both were false statements in shipped source |
| `a4e2772` | `style.css` — `mjx-container` overflow | affects every page linking style.css, not just `spl` |
| `d52ba3d` `599a207` | resqr promotion of `spl`, `plo`, `sexp`, `logi` | **changes band rendering for 3 pre-existing models**; rebuilt `model_data.pkl` |
| `8acee12` `36e5f78` `bef2821` | model-registration linter (+1052 assertions) | new test surface; `36e5f78` fixes an unrelated pre-existing ordering bug |
| `2ba202f` `098b5e6` | shared `--update` patcher, adopted in 3 tools | rewrites the write path for `fit_spl`/`fit_gompertz`/`fit_logistic` |
| `b4c2aa8`, `c5a1a76`, `5e22d6a`, `9c69428`, `d984c0c` | docs / follow-ups | low risk |

**Highest-value targets for a fresh reviewer**, in order:

1. **`599a207` / `d52ba3d`** — promoting `plo`/`sexp`/`logi` changed the
   quantile bands of three models that were shipping before this work started.
   Verified: 91/91 fit, medians unchanged, Q10 moves (e.g. `logi` $13,703 →
   $28,743). Nobody has reviewed whether that is *desirable* for those three.
2. **`2ba202f`** — the atomic write path. If it is wrong, `--update` can
   truncate `btc_core/_simple.py`. Task 7's reviewer verified the earlier
   `fit_spl`-local version (same filesystem, cleanup on both `KeyboardInterrupt`
   and `OSError`, mode preserved); the *shared* version is newer.
3. **`f0a1f82`** — the only commit that discarded working-tree content.
4. The **spec's statistical claims** (`7508f9e`) — reviewed by a stats consult
   before the data changed, not since.

## State to hand the reviewer

```
suite    : 2,735 passed, 10 skipped, 2 failed
           BOTH failures are pre-existing and NOT from this work:
             test_callbacks.py::TestBTCPayPricing::test_free_tier_all_models
             test_colors_central.py::test_no_hex_literals_outside_colors_module
linter   : btc_venv/bin/python3 tools/check_model_registration.py spl  -> 10/10 ✓
build gate: btc_venv/bin/python3 tools/spl_spec_build_gate.py          -> 3/3 pass
prod     : https://quantoshi.xyz  all routes 200, no journal errors since restart
           /mi.23 -> mi-s2f (unchanged)   /mi.30 -> mi-spl   both figures 200
```

Range for the review: `963fb60~1..HEAD` (33 commits).

## Open follow-ups (`docs/superpowers/followups.md`)

F-1 `grdy` never refits (flag inside the script path) · F-2 `logi` + 9 other
sigma branches unmasked — **latent in calendar basis, LIVE in block basis** ·
F-3 widen the SCRIPTS assertion once F-1 lands · F-4 MC free-tier badge
disagrees with billing (`exp`/`lppl` show free, server charges) · F-5 `/mi`
caches 24h · F-6 `plo`/`sexp`/`logi` missing from `_SCANNER_ORDER` ·
**F-7 CLOSED** · F-8 18 fit tools still inline their own patcher · F-9
time-evolving bubble model (research).

Each carries a one-command verification that was actually executed.

---

# PART 2 — Phase 2: the user-set ceiling

Design is **spec §8** in
`docs/superpowers/specs/2026-08-07-saturating-power-law-design.md`. Read it
before planning; what follows is orientation, not a substitute.

## The groundwork already exists

`SaturatingPowerLawModel.__init__` takes optional `log10_L`, `t0`, `beta`
overrides that **shadow the class attributes without mutating them**, and
`test_optional_overrides_shadow_class_attrs` exists solely to stop a future
refactor deleting them. That test looks gratuitous in isolation. It is not.

## The two things that are easy to get wrong

1. **A mutated singleton is a cross-user data race.** `_resolve_model`
   (`figures/common.py:1255-1268`) returns `_app_ctx.PRICE_MODELS.get(key)` — one
   module-level object shared by every request in every gunicorn worker. Setting
   `PRICE_MODELS["spl"]._log10_L` from a callback changes the ceiling **for
   everyone currently on the site**. `u1` is the only key with a per-request
   escape hatch (`UserModel.from_store_dict`, `:1263-1267`); add the same branch
   for `spl`. The constructor overrides exist for exactly this.
2. **The ceiling must land in the params dict AND in `tab_defaults`.** The
   params dict is the cache key (`cache.py:35`, `sha256(params_json)`). Omit the
   ceiling → two users with different ceilings share one cached figure. Include
   it but omit it from `bubble_defaults()` → the prewarm key stops matching the
   runtime key and Tab 1 loses its first-visit L1 hit. `user_model` does this
   correctly at `tab_defaults.py:506/540/553/564/573` — mirror it.

## Interaction design (settled in the spec)

Log-scale slider in **total market cap** (`L = cap / 21e6`), marks at
$1T/$10T/$35T/$100T/$1000T, in the Tab 1 controls column, revealed when `spl` is
ticked in Display Models — the U₁ mechanism (`callbacks/user_model.py:127-135`),
a clientside callback on `bub-model-show` toggling `style`. **Deliberately not a
gear modal**: those cover the chart, and the point is watching the fit degrade
while dragging. R0 measured a `_ShrinkingBandsMixin` refit at **1.2–1.8 ms warm**,
so no background-callback machinery — the real budget is one bubble redraw per
mouseup.

## What Phase 2 must NOT break

- **The card is PINNED** (spec §7.1a). Every figure on it is a static constant
  fixed to two dated windows, including the coefficient table. A slider that
  writes back into the card would undo that deliberately.
- `spl`'s class constants are pinned to the **2026-08-06** window. Do not run
  `fit_spl.py --update` casually; the card's numbers are keyed to them.
- `short_name = "spl"` is permanent — a share-link key.

## Also file under Phase 2 (deferred from Phase 1)

- `beta` is only swept to 20 in tests while the code comments claim headroom to
  ~210. If β becomes user-settable it needs a test **and a clamp**.
- `t0 <= 0` silently yields all-NaN prices. No validation today. Same owner.
- The `_spl_profile_table` caption should gain "(profiled over t₀ on a coarse
  grid; the free optimum is $14.8T)" — the card's [4] table is a **t₀**-profile
  while the figure is an **L**-profile, and "best $18T" sits near "$14.8T"
  unlabelled.
- Mobile: the 6-cutoff table still overhangs the viewport by 42px. Not fixed
  because the only reliable CSS fixes change layout for all 65 tables.

---

# Context a fresh session will not otherwise have

**The statistics were rebuilt three times and the earlier framing keeps
resurfacing.** It is in the git history, in superseded spec revisions, and in
any summary written before 2026-08-07. The retracted claim is *"the data fail to
reject a pure power law"* — true of the 2026-06-03 window, **false on current
data** (LRT 13.65 REJECTS). The current framing is **the instability of L**: it
moved $34.3T → $14.8T on 1.1% more data, ranges 93× across six cutoffs, and the
verdict flips non-monotonically. Do not let a well-meaning edit restore the old
story.

**A separate trial ran 2026-08-08** fitting `spl` to support-phase prices only
(scratchpad `spl_support.py`, `spl_floor_v3_fig.py` — deliberately never
committed). Headline: strip the bubbles and saturation vanishes — L pins at
whatever cap is imposed across seven datasets, and the recent floor sits ~1.47×
**above** its own nine-year extrapolation. That work motivated **F-9**. It is
exploratory and has not been reviewed.

**Process note that paid off repeatedly.** Three separate findings this week
were *false statements in durable artifacts* rather than broken logic: a report
claiming "no second optimiser" when there was one, a class docstring asserting a
retracted statistical result, and a guard comment explaining a bug that did not
exist. All three were caught by independent readers; none by tests; none would
have failed anything at runtime. Point reviewers at comments, reports and
docstrings, not only at code.
