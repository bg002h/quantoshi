# `spl` spec — R0 independent architect review (opus)

**Date**: 2026-08-07
**Artifact reviewed**: `docs/superpowers/specs/2026-08-07-saturating-power-law-design.md` @ `4d88ce2`
**Reviewer**: independent opus agent (author ≠ reviewer)
**Brief**: completeness against the 26-step registration checklist; irreversible
decisions; defensibility of the statistics in §3; Phase 2 feasibility.
Machine-verified inputs declared up front (`tools/analyze_spl.py`,
`tools/spl_spec_build_gate.py`) so reviewer budget went to design.

Persisted verbatim, unedited, **before** any fold. Do not amend this file.

---

```
VERDICT: not ready

CHECKLIST COVERAGE
  step 2  — PARTIAL — §5's class body omits `dash_style`. `_ShrinkingBandsMixin` does not
            supply one (btc_core/_base.py:17-83; :98 is a Protocol annotation, not an attr).
            LOUD: AttributeError at btc_web/figures/bubble.py:269 + test_models.py:809
            `test_all_models_have_dash_style` fails. Fix: add `dash_style = "longdash"`
            (unused by gomp/logi/sexp; "dot"/"longdash" are taken by pl/s2f and one other).
  step 3  — PARTIAL — §5 shows no `__init__`. Siblings apply `mask = price_years >= T_MIN`
            (btc_core/_simple.py:332; T_MIN=1.0 from btc_web/time_basis.py:84). Spec must
            state the constructor applies it — see CRITICAL #3.
  step 4  — GAP — `_build_colors()` not mentioned. Mixin does not define it → AttributeError
            in `__init__`. LOUD. Fix: add a `_build_colors` gradient like _simple.py:344-353.
  step 5  — GAP — `btc_core/__init__.py` re-export unmentioned. LOUD (ImportError in app.py).
  step 6  — GAP — `app.py` PRICE_MODELS registration unmentioned. Implied but never stated;
            it is the single step that makes the model exist. LOUD if missed.
  step 8  — GAP — `tools/generate_color_artifacts.py` regeneration unmentioned. This was the
            v1 ⚠️ miss. SILENT: model renders fallback gray in the JS palette path only.
  step 9  — GAP — `_MODEL_INFO_ITEMS` unmentioned, AND the checklist itself is stale: there
            are TWO hand-mirrored copies (btc_web/callbacks/routing.py:537 and
            btc_web/layout/model_info/__init__.py:60) plus the physical order in
            `_items.py::_build_accordion_items`. No test guards the three-way sync.
            See CRITICAL #4 — insertion position is irreversible.
  step 10 — PARTIAL — decided "yes" but no `_HM_PILL_LABELS` label string given, and the
            layout consequence is unconsidered. See IMPORTANT #1.
  step 22 — GAP — no test class specified. Steps 23/24/25 (suite run, smoke test, deploy)
            also unmentioned.
  NEW STEP (not in the 26) — GAP — `docs/architecture.md:339-351` carries a price-model
            registry table and is **served live** at `/docs/architecture` via
            `btc_web/api.py:154-156` (`_render_doc`). SILENT: public docs page goes stale.
  NEW STEP (not in the 26) — GAP — `btc_web/callbacks/scanner.py:22-25` `_SCANNER_ORDER`.
            SILENT and benign (`_scanner_sort_key` sinks unknown keys via `except ValueError`),
            but it is a real per-model registry with a hole.
  NEW STEP (not in the 26) — GAP — `RESQR_FLAGSHIP_MODELS`, tools/build_bm_model.py:55-60,
            mirrored in btc_web/test_resqr_build.py:29-34. See CRITICAL #2.
  Steps 1, 7, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 26 covered.
  (Spec's §4.1 line citations verified correct: build_bm_model.py:96, refit_all_ppl.py:59,
   fit_shrinking_sigma.py:219 + :248.)

CRITICAL
  - **The "1 AIC unit" band is a ΔAIC ≤ 2 band.** tools/analyze_spl.py:132 —
    `thresh = best * (1 + 2.0/n)  # ~1 AIC unit`. With k constant across the profile,
    ΔAIC = n·ln(SSE/SSE_best), so ΔAIC=1 ⇒ threshold `1.0/n`, not `2.0/n`. Measured
    (n=5801, SSE_best=502.306): ΔAIC≤1 admits ΔSSE ≤ 0.0866; ΔAIC≤2 admits ≤ 0.1732.
    Recomputed profile:
      t₀=25 → ΔSSE +0.1392 ($19T)  — in ΔAIC≤2, **out** of ΔAIC≤1
      t₀=35 → +0.0816 ($98T) in both;  t₀=40 → +0.1309 ($192T) — in ≤2, out of ≤1
      t₀=50 → +0.1727 ($594T) — in ≤2, out of ≤1
    So §3.2's headline "**L anywhere from $19T to $594T, a 30× range**" and the "$19T floor"
    are the ΔAIC≤2 numbers wearing a ΔAIC=1 label. Under the stated criterion the band is
    ~$35T–$100T, a **3× range**. Both the floor and the range appear in the Model Info card
    (§7 item 4). Fix: pick one and label it — either change :132 to `1.0/n` and restate the
    band as $35T–$100T, or keep `2.0/n` and call it "within 2 AIC units (Burnham & Anderson
    'substantial support')". Update §3.2, §7, and the :132 comment together.
  - **§4.1 step 19 misstates what `instances["spl"]` buys, and the resqr decision is unmade.**
    resqr is gated by an explicit allow-list, `RESQR_FLAGSHIP_MODELS` (tools/build_bm_model.py:55-60,
    consumed at :130-134) — `logi`/`plo`/`sexp` are NOT in it, and `model_data_resqr_diagnostics.json`
    has no entries for them. Adding `instances["spl"]` alone therefore yields **no** resqr bundle
    and no diagnostics, contradicting §4.1's "no resqr bundle, no diagnostics" justification.
    Consequence is silent: Tab 1's σ-mode radio defaults to `"resqr"`
    (btc_web/layout/display_models.py:262), and `_resqr_price_at` returns None for a model with
    no `_resqr` bundle (btc_core/_helpers.py:99-101) → `spl` silently renders constant-σ bands
    beside flagship models rendering resqr bands, with the UI claiming resqr mode. Fix: spec must
    decide flagship-or-not. If yes, add to `RESQR_FLAGSHIP_MODELS` **and** to the hand-mirrored
    `EXPECTED_FLAGSHIP_MODELS` in btc_web/test_resqr_build.py:29-34. If no, say so and rewrite
    §4.1's step-19 rationale (what step 19 actually buys a non-flagship is close to nothing —
    that materially weakens the "High severity" rating in §9).
  - **§3's inference ignores residual autocorrelation; effective n is not 5801.** Measured on
    the `pl` residuals: Durbin–Watson = 0.0038, lag-1 ρ = 0.9981. Every AIC/BIC quantity in §3
    (ΔAIC −0.25, ΔBIC +6.42, the band width ∝ 1/n, §3.4's ΔBIC +4942, §8's projected "ΔBIC +5900")
    assumes iid Gaussian residuals. Widen the band by even 10× for n_eff and the entire profile
    from t₀=20 to ∞ falls inside it — i.e. **the $19T floor, the headline result, does not survive
    the correction**. §3.3 makes the fallacy explicit: "visible only because there are 5,801 daily
    points" — daily closes of a trending series do not each carry an independent observation.
    Fix: state n_eff explicitly (report DW/ρ), widen or caveat the band accordingly, and soften
    "the data exclude ceilings below ~$19T" to a statement conditioned on the iid assumption.
    The repo already uses Durbin–Watson for origin selection, so the tooling exists.
  - **`mi-spl` must be appended at the END of the accordion, not next to `mi-logi`.** Deep-link
    lookup is strictly positional — `_MODEL_INFO_ITEMS[n - 1]` (btc_web/callbacks/routing.py:730,
    same pattern in static_pages.py:501-503). Inserting after `mi-logi` (index 22) shifts
    `mi-s2f`…`mi-citadel` by one, silently breaking every existing `/mi.23`…`/mi.29` and
    `/9.23`…`/9.29` link. This is the same irreversibility class as `short_name` and the spec
    does not mention it. Fix: append `"mi-spl"` as item 30 in **both** routing.py:537 and
    layout/model_info/__init__.py:60, append the AccordionItem last in `_items.py`, and accept
    that the card sits after `mi-citadel`. Add a test asserting the three orders match — there
    is currently none (`test_static_pages.py:126` covers FAQ only).

IMPORTANT
  - **§4's fit bounds are not the bounds that produced §3's numbers.** tools/analyze_spl.py:85:
    `bounds = [(-10,10), (0.1,20.0), (log10(t.max()), 2.0)]` → t₀ ∈ [16.86, 100], β ∈ [0.1, 20],
    and **no L bound at all** — `CAP_FIT_USD` is computed at :33 but never applied to the spl fit
    (only to the linear-time counterpart at :103). So §4's claims "the $1000T bound never binds"
    and "t₀ [1,100], fitted 28.4 comfortably interior" are untested: the region t₀ ∈ [1, 16.86)
    was never searched. Fix: either align §4's bounds to analyze_spl.py's, or run the §4 bounds
    and confirm they reproduce t₀=28.371 / RMSE 0.294261 before the plan is written.
  - **"L ≥ highest observed price" is a constraint on a derived quantity.** Under the (A, β, log10 t₀)
    parameterisation L = 10^(A + β·log10 t₀); `differential_evolution` takes box bounds on
    coordinates only. §4 does not say whether this is a penalty, a rejection, or a post-hoc check —
    and §4's own rule ("the fit report must state whether any bound is active") cannot be applied
    to a constraint that isn't implemented. Fix: specify the mechanism.
  - **§3.2's heading "bounded below, not above" contradicts §3.2's own conclusion.** The spec says
    both "`pl` itself falls *outside* that band" and "the data cannot distinguish $35T from $594T
    from **no ceiling at all**". By the stated criterion the band is bounded on both sides —
    t₀=60 (+0.1859) and t₀=100 (+0.1942) are excluded along with pl (+0.1948). What is actually
    true: the penalty gradient is steep below (ΔSSE +3.36 at t₀=20) and asymptotic above, so the
    upper edge is an artifact of threshold placement — pl misses by 0.0217 SSE out of 502. Fix:
    say "the lower bound is robust; the upper bound is wherever you set the threshold, because
    SSE flattens rather than rising."
  - **§3.3's percentages are `spl` vs its own asymptote, not vs the fitted `pl`.**
    tools/analyze_spl.py:166 computes `-log10(1 + (tt/t0)**beta)` using spl's β=5.0905 — but §3.1's
    `pl` has slope 5.0734, a different curve. Recomputed against the actual fitted `pl`:
      2020 **+1.19%** (spl is *above* pl), 2024 −0.80%, 2026 **−3.39%**, 2030 −13.33%,
      2038 −48.52%, 2050 −85.27%.
    So "a ~6% shortfall against the power law today" is ~3.4% against the model users will see on
    the chart, and the deviation is a **pivot** (above early, below late), not a monotone shortfall.
    The card would quote a number measured against a curve that is not plotted. Fix: report the
    vs-fitted-`pl` column, or label the existing one "vs its own asymptotic power law".
  - **Heatmap pill bar has no wrap or overflow handling.** `_HM_PILL_MODELS_BASE` is 12
    (btc_web/layout/heatmap.py:26-29) plus conditional `ef`/`u1`/`mc` = up to 15 today; `spl`
    makes 16. Rendered as `dbc.ButtonGroup(size="sm")` (layout/heatmap.py:229-230), which is
    `display:inline-flex` with default `flex-wrap:nowrap`; `style.css` has no rule for it.
    A 16th pill compresses and then overflows the card on narrow viewports. Fix: add
    `flex-wrap:wrap` (or an `overflow-x:auto` container) to the pill bar as part of this change,
    or record the decision to defer with the overflow acknowledged.
  - **§3.4's $124,642 is a bound value, not an estimate.** analyze_spl.py:103 sets the L lower
    bound to `log10(p.max())`, so "pinned to the highest observed price" is guaranteed by
    construction. §4's own rule says a pinned parameter is an artifact. The ΔBIC +4942 rejection
    stands regardless. Fix: state it as "the optimiser drives L down to its lower bound", drop
    the dollar figure, and keep the ΔBIC.
  - **`tools/fit_spl.py` must apply the class's data mask.** analyze_spl.py's `load()` (:36-46)
    keeps all `t > 0` (t_min = 0.9774, n=5801); the class constructor will mask
    `price_years >= T_MIN` with T_MIN = 1.0 (btc_web/time_basis.py:84). Different residual sets →
    the shipped params are not the SSE optimum for the residuals `_init_shrinking_bands` sees, and
    `_sigma` is derived from those residuals. Effect is small (~8 days) but silent. Fix: state that
    fit_spl.py, fit_shrinking_sigma.py's `_eval_model` branch, and the constructor all use the
    same `>= T_MIN` mask.

MINOR / NIT
  - `updatemode="mouseup"` (§8) is already `dcc.Slider`'s default, and no layout module sets
    `updatemode` at all (grep over btc_web/layout/*.py: zero hits). Presenting it as a mitigation
    overstates it; it just means "don't set `updatemode='drag'`".
  - RMSE quoted to 6 decimals (0.294261 vs 0.294318) implies discrimination the data cannot
    support once ρ=0.998 is accounted for. Round for the card.
  - `CLAUDE.md:262-276` model list is a third doc registry that goes stale. Not runtime-served.
  - §3.3's quadratic coefficient reproduces exactly (+0.02853). Honest caveat, correctly stated;
    note only that a global quadratic on autocorrelated data is descriptive, not a test.
  - §6 decides "step 10 — yes" but gives no `_HM_PILL_LABELS["spl"]` string. Pick one (e.g. "SatPL").

STATISTICS REVIEW
  - §3.1 fit table / ΔAIC = −0.25 / ΔBIC = +6.42 — **sound and internally consistent.** Verified
    analytically: n·2·ln(0.294261/0.294318) = −2.247; +2 → −0.247 (AIC), +ln(5801)=8.666 → +6.419 (BIC).
  - §3.1 handling of the AIC/BIC disagreement — **honest, not cherry-picked.** Both are reported and
    the conclusion follows the less favourable one ("the extra parameter does not pay for itself").
    This is the strongest part of §3. Caveat only that ln(n)'s penalty is inflated by the same
    n_eff problem, so the disagreement is even less meaningful than stated.
  - §3.2 "the ceiling is bounded below, not above" — **wrong as stated**, right in spirit. See
    IMPORTANT above: the spec's own criterion bounds it on both sides while the prose says it
    doesn't, and the same paragraph claims pl is excluded.
  - §3.2 1-AIC band construction `SSE_best·(1 + 2/n)` — **wrong criterion for the stated label.**
    That is ΔAIC = 2. See CRITICAL.
  - §3.2 "pl falls outside that band" — **arithmetically correct** (pl ΔSSE +0.19485 vs threshold
    0.17318) but the margin is 0.02 SSE units out of 502, and it is stated as if decisive while the
    adjacent sentence claims the opposite.
  - §3.3 "spl − pl reduces exactly to −log10(1 + (t/t₀)^β)" — **algebraically correct**, verified by
    hand. But "pl" there means spl's own asymptote; the table's header and the prose do not say so.
  - §3.3 "~6% shortfall today, roughly a tenth of the noise" — the ratio arithmetic is right
    (log10(0.9415)/0.294 = 0.089) but the 6% is the wrong comparator (3.4% vs the fitted pl), and
    there is an unremarked tension: a signal at 0.089σ over n=5801 iid points would be ~6.8σ, yet
    ΔAIC is only −0.25. That gap is precisely the autocorrelation problem, and naming it would
    make §3 much stronger rather than weaker.
  - §3.4 linear-time rejection — **conclusion sound** (ΔBIC +4942 is decisive), **evidence
    mis-described** (L is at its lower bound by construction). "Saturation is a log-time phenomenon
    or it is nothing" is rhetorical overreach from one alternative parameterisation; soften.
  - Unremarked but worth one line in the card: fitted t₀ = 28.4 yr against a data span ending at
    t = 16.86 yr. The inflection point is a pure extrapolation — nearly 2× beyond any observation.
    §3.2 shows this implicitly; the card should say it.

PHASE 2 FEASIBILITY
  - **Reveal mechanism works.** `_DEPRIORITIZED` (btc_web/layout/display_models.py:51, used only at
    :212-214) affects **ordering only** — `spl` remains a normal checkable value in
    `bub-model-show`, so the U₁ pattern (`Input("bub-model-show","value")` → panel `style`,
    callbacks/user_model.py:127-135) applies unchanged. Cost: being deprioritized puts `spl` at the
    bottom of a ~20-row checklist, so the ceiling panel is behind a scroll — a discoverability
    tradeoff, not a defect.
  - **NEEDS: a per-request model instance, not a mutated singleton.** `_resolve_model`
    (btc_web/figures/common.py:1255-1268) returns `_app_ctx.PRICE_MODELS.get(model_key)` — the
    module-level singleton shared by every request in every gunicorn worker. Setting
    `PRICE_MODELS["spl"]._log10_L` from a callback is a cross-user data race. `u1` is the only key
    with a per-request escape hatch (`UserModel.from_store_dict(p["user_model"])`, :1263-1267).
    Phase 2 must add the same branch for `spl` and thread the ceiling through the params dict.
  - **NEEDS: the ceiling in the params dict AND in `tab_defaults`.** The params dict is the cache
    key (btc_web/cache.py:35, `sha256(params_json)`). If the ceiling is not in it, two users with
    different ceilings share one cached figure. If it is in it but absent from `bubble_defaults()`,
    the prewarm key stops matching the runtime key and Tab 1 loses its first-visit L1 hit — the
    documented cache-key-alignment rule. `user_model` does exactly this: `d["user_model"] = None`
    at tab_defaults.py:506/540/553/564/573. Mirror it.
  - **No background-callback machinery needed.** Measured: constructing a `_ShrinkingBandsMixin`
    model over the full n=5801 history costs **1.2–1.8 ms** warm (281 ms first call is scipy import).
    A 2-parameter local refit seeded at the known optimum is a few ms more. The spec's "sub-100 ms"
    is right for the refit. But end-to-end latency is dominated by the full `build_bubble_figure`
    rebuild and the Dash dispatch of a large figure — the exact cost recorded in
    `feedback_task45_clientside_patch_regressed_bubble.md`. Frame the budget as
    "one full bubble redraw per mouseup", not "sub-100 ms".
  - **Phase 1 bakes in one thing that makes Phase 2 harder:** `_log10_L` / `_t0` / `_beta` as class
    attributes read directly off the class elsewhere (tools/fit_shrinking_sigma.py:219-221 does
    `m = LogisticSCurveModel; m._K`). Phase 2 needs instance-level overrides. Instance attrs shadow
    class attrs so this works, but the constructor must accept the three params as optional
    arguments defaulting to the class attrs — decide that in Phase 1, not Phase 2, or the
    constructor signature changes after the model ships.
  - §8's "ΔBIC +5900 at $2T" readout inherits the n_eff problem — it will display a number that is
    ~2 orders of magnitude too confident. Qualitatively it still shows the right thing.

CLEARED
  - `short_name = "spl"` free against all 103 registered models — accepted as given, not re-checked.
  - **Snapshot bitmask width is safe.** `_list_to_mask`/`_mask_to_list` (btc_web/snapshot.py:543-554)
    use arbitrary-precision Python ints, and encode/decode is Python-only: `restore_from_url` is a
    server callback (callbacks/snapshot_cb.py:58), the gzip+base64 blob is opaque to the browser,
    and the only bitwise ops in the entire assets tree are a hex-colour decode at
    assets/scanner.js:23. 32→33 and 33→34 bits carry no JS truncation risk.
  - **§6's `SNAPSHOT_DEFAULTS` claim is correct.** No default enumerates the model registry —
    `bub-model-show:value` is `['bub']` (snapshot_defaults.py:58), all `*-model-show` defaults are
    fixed small lists; `PRICE_MODELS` appears nowhere in snapshot_defaults.py / snapshot.py /
    tab_defaults.py. The fingerprint (snapshot_defaults.py:387-402) and `_L0_FINGERPRINT`
    (cache.py:131-135) are unchanged by appending to `_CHECKLIST_OPTIONS`. No registry entry needed.
  - **Append-only treatment of `_CHECKLIST_OPTIONS` is correct and sufficient** — all 5 lists
    (snapshot.py:471-474 at 33 entries, :478 `bub-model-show` at 32), appending preserves bits 0..n-1.
  - Figure-cache staleness: adding a model cannot resurrect a stale figure, because no pre-existing
    `params_json` could contain `"spl"` → guaranteed new sha256 → miss. Deploy's `FLUSHDB` covers
    the rest.
  - `_prewarm_caches()` (app.py:487-527) does not enumerate models; key set and cost unchanged.
  - Startup cost of a 104th model: low single-digit ms (construction) plus one pass each through the
    backgrounded R² loop (app.py:363-389) and the auto-Y grid (app.py:404-409). Negligible.
  - §4.1's three line citations verified: `instances["logi"]` at tools/build_bm_model.py:96,
    `("Logistic","tools/fit_logistic.py")` at tools/refit_all_ppl.py:59, `elif model_name == "logi"`
    at tools/fit_shrinking_sigma.py:219 with `models = [...]` at :248. All correct.
  - §4.1's step-21 severity assessment is right and appropriately low-key: `_ShrinkingBandsMixin._sigma_at`
    (btc_core/_base.py:38-40) returns constant `self._sigma` — the fitted σ₀/α are computed and unused,
    so the offline sweep is bookkeeping, not rendering.
  - §5's `logaddexp` discussion — correct and appropriately de-escalated; the overflow claim was
    already refuted by the gate and the spec says so.
  - Palette `#4C1D95` — accepted as given, not relitigated.
  - `DECOMP_FAMILIES` (step 14, "no") and `_MODEL_CYCLE` (step 15, "no") decisions match how
    logi/plo/sexp/gomp/bpl are handled today.
```

## Reviewer's closing summary

Two things worth pulling out for the user, since they change the shape of the
work rather than just adding steps:

**The statistics section needs a rebuild, not an edit.** The `$19T floor` and
`30× range` — the headline results destined for the Model Info card — are
computed at ΔAIC ≤ 2 but labeled ΔAIC = 1 (`tools/analyze_spl.py:132`). At the
stated criterion the band is ~$35T–$100T, a 3× range. Separately, lag-1
residual autocorrelation is 0.998 (DW = 0.0038), so every information criterion
in §3 assumes an effective sample size roughly three orders of magnitude larger
than the data supply. The floor result does not survive that correction. §3 is
currently the spec's strongest-looking section and its weakest actual one.

**Two silent registration surfaces are outside the 26-step checklist**, meaning
the checklist is stale rather than the spec merely incomplete: the resqr
flagship allow-list (`RESQR_FLAGSHIP_MODELS`, which the spec's §4.1 rationale
mis-describes) and the live-served `/docs/architecture` model table. The Model
Info accordion is a third, and it is *positional* — appending `mi-spl` next to
`mi-logi` rather than at the end silently breaks every existing `/mi.23`–`/mi.29`
deep link, which is the same irreversibility class as `short_name` and is not
mentioned anywhere in the spec.
