# Tab-1 Raw MC spec: builder's open questions (2026-11-10)

Spec: `docs/superpowers/specs/2026-11-10-tab1-raw-mc-design.md`. Checked against
`callbacks/charts/__init__.py` (update_bubble), `callbacks/mc_helpers.py`,
`callbacks/mc_controls.py`, `callbacks/mc_payment.py`, `layout/mc_controls.py`,
`snapshot.py`, `figures/bubble.py`, `assets/mc_autowiden.js`. Read-only.

Ranked by how much a wrong guess would cost.

## 1. What happens to the paid-MC chrome around a Raw fan?

The spec says Raw never opens a payment prompt (goal 2, anti-goal 4) but is silent
on the visible controls that lead there. In the current code, with `raw_iid` or
`raw_cycle` selected:

- **Run MC Simulation button** (`{"type":"mc-run-btn","tab":"bub"}`) stays visible.
  `_mc_payment_initiate` (`callbacks/mc_payment.py` L152-181) calls
  `btcpay.is_free_tier("raw_iid", ...)`, which is False because the key is not in
  `_CACHED_MODEL_KEYS`. On prod it then calls `btcpay.create_invoice` and opens the
  payment modal. In DEV it bumps `mc-pay-trigger`.
- **Cost panel** (`_MC_COST_JS_TPL`) falls into its "Live ... Cost: N sats" branch.
  It also writes `bub-mc-price-val`, which can trip the >50k-sat quant modal.
- The **"Paid" badge** in the MC header still shows.

Question: in Raw mode, do we (a) hide the Run button and replace the cost panel with
a "Free" line, and (b) also add an early return in `_mc_payment_initiate` for raw
keys as a backstop? Or is a visible-but-inert button acceptable?

Why it matters: this is the one place goal 2 can fail on prod. It adds work in
clientside JS (and `assets/_mc_cost_consts.js` if the cost JS is reused),
`callbacks/mc_payment.py` and the layout, each needing tests. "Hide" and "inert"
lead to different tests.

## 2. Cycle mode: is the fallback pool filtered, and what about thin pools?

The spec says: "If a phase has no eligible block under the current window and
filter, that block falls back to the whole eligible pool." It does not say whether
that fallback pool still honours the momentum filter. "Unchecked bins are never
sampled" suggests yes, but the sentence reads as if the filter is dropped.

On the real CSV (window 2015-2026, 5 equal-count bins, my rough count of 3-month
block starts per phase 0-7 and bin):

- With all bins on, each phase has 11-22 candidate starts.
- With only the coldest bin checked, phases 0, 1, 2, 6 and 7 have zero candidates
  (5 of 8 fall back), and phase 3 has exactly one.
- With only the hottest bin checked, phases 0, 5, 6 and 7 have zero and phase 4 has one.

So under a narrow filter, most of the horizon uses the fallback pool, or 200-3200
paths are drawn from one repeated block per phase and the fan collapses.

Questions:
- Does the fallback keep the filter?
- Is "zero candidates" the right trigger, or should there be a minimum pool size (N)
  below which the phase falls back or the status line warns?
- Should the status line report thin phases as well as empty ones?

Why it matters: this is the core sampler of the headline variant. The answer
changes the algorithm, the status-line text and the test vectors. A wrong guess
produces a fan that looks filtered but is not, or looks diverse but is not.

## 3. Where does Raw's feedback appear, and what are the other MC outputs?

The spec says "the MC status line" must say why no fan is drawn, and how many phases
fell back. update_bubble's only text output is `bub-mc-status`. It sits under the
card titled "Saved Simulation", below the Run button, so a "no fan" failure would
show an empty chart plus a line out of sight (especially on mobile).

Related loose ends if Raw skips `_mc_setup` / `_mc_finalize` (it must, to avoid the
payment path):

- `bub-mc-rendered-key` would be None. The clientside match indicator then shows
  "Chart does not include MC overlay" beside a drawn Raw fan.
- If we set a rendered-key instead, which values? Raw ignores start year and entry
  percentile, but the match JS compares those hidden controls.

Questions:
- Is `bub-mc-status` the intended line, or should the "no fan" and "N phases fell
  back" messages also go on the chart (annotation) or in the match indicator?
- What should rendered-key hold in Raw?

Why it matters: a new Output on update_bubble changes the 8-tuple contract
(`(dash.no_update,) * 8` appears at least twice, plus tests). Reusing
`bub-mc-status` changes nothing structural.

## 4. What happens to an existing regime selection when the meaning of the bins changes?

`bub-mc-regime` holds bin indices. In model mode they are price-percentile bins
(Bargain...Bubble); in Raw they are momentum bins. Switching model to Raw, or back,
silently reinterprets a stored filter. For example, "only Bargain" becomes "only
coldest momentum". iid to cycle keeps the meaning but changes the pool.

Complication: `_REGIME_JS` deliberately never resets `value` on a bins change,
because that clobbered share-link restores. The same trap applies to anything keyed
on `bub-mc-model-src`.

Question: on a real user switch of source, do we reset the regime to all-checked
(click-stamped, never on restore) and, if so, bins too? Or leave it and rely on
the relabelled options and the note?

Why it matters: this decides whether the nudge callback gets extra Outputs and
stamp logic, and what the E2E asserts. It also interacts with `window.__qsRegimeK`
in the auto-tick JS.

## 5. Phase for months before the first halving (cycle mode)

Phase is defined as "whole months since the most recent halving", but about 28
history months (Aug 2010 - Nov 2012) precede 2012-11-28, the first entry in
`_HALVING_KNOWN_DATES`. That is roughly 14% of the default 2010 window, the very
years the spec flags as producing absurd prices.

Choices: count from genesis (the first block-reward epoch, giving phases 3-7), drop
them as block starts in cycle mode, or fold them into phase 7.

Why it matters: it fixes the phase function and its test vectors, and changes what
a user who deliberately keeps 2010-2012 in the window gets. Independent mode is
unaffected.

## 6. Are the controls Raw depends on reachable?

The window slider (the spec's only way to include or exclude bubbles), plus sims,
bins and the regime checklist, sit inside "Advanced simulator options"
(`bub-mc-adv-body`, hidden until ticked). The spec's window nudge also moves it
without the user seeing it. Placement of Lines shown and Show extremes is not stated
either.

Question: in Raw mode do we pull window (and the new Lines/Extremes) out of the
Advanced fold, or leave everything behind it and rely on the legend suffix?

Why it matters: it is layout and visibility wiring, and it decides whether the
Raw-mode visibility callback toggles one block or several. Cheap to change early,
awkward after E2E selectors are written.

---

Checked and consistent with the code (no question): `_SNAPSHOT_CONTROLS` has 340
entries so 340/341 are the next free tail indices; `_add_mc_fan(label_suffix=)`,
`_add_mc_spaghetti`, `_mc_path_alpha`, `_HALVING_EPOCHS` (4-year cadence to 2080),
`MC_REGIME_PATHS_JS` and the single-Input clientside constraint all exist as
described. `_TAB_CONTROLS` lives in `callbacks/routing.py`, not `snapshot.py`.
Time Machine already forces MC off, so no as-of interaction.
