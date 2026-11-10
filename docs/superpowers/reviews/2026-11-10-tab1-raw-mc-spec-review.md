# Tab-1 Raw MC spec: adversarial review (2026-11-10)

**Spec:** `docs/superpowers/specs/2026-11-10-tab1-raw-mc-design.md` (at d5d7f7de)
**Counts: 0 Critical · 5 Important · 8 Minor · 2 Nit**

Checked against `callbacks/charts/__init__.py` (update_bubble), `callbacks/mc_helpers.py`,
`callbacks/mc_controls.py`, `layout/mc_controls.py`, `layout/bubble.py`, `figures/bubble.py`,
`btcpay.py`, `snapshot_defaults.py`, `assets/mc_autowiden.js`, `assets/mc_regime_paths.js`, and the
Dash 4.0 `dcc` bundle (`async-dropdown.js`). Read-only. Statistics come from
`/tmp/claude-1000/-scratch-code-bitcoinprojections/3a79ac35-3364-4e4c-9013-baf596cc6ebf/scratchpad/raw_exp.py`
run on `BitcoinPricesDaily.csv` (last close 2026-11-08), using month-end closes with the incomplete
current month dropped.

---

## Important

### I-1. Raw-mode sims and bins carry over into model mode and switch on the paid path

**Where:** Sims, lines and extremes (`bub-mc-sims` = paths, 1–3200); Regime filter (bins 5–10);
Goal 2; Non-goal "Changing the model-based MC on Tab 1 in any way"; anti-goal 4.

**Failure:** `bub-mc-sims` and `bub-mc-bins` are the same components in both modes. Raw invites
users to raise them, because there they are free. `btcpay.is_free_tier` (L68) returns False when
`mc_sims > 200` or `mc_bins != 5`. Scenario:
1. A user selects Raw · independent and sets sims to 3200 (or bins to 8).
2. They switch back to "Bubble".
3. The model-mode MC is no longer free. `_mc_payment_check` fails, the fan disappears, the match
   line says "Chart does not include MC overlay", and the Run button and sats cost come back.
   Clicking Run on prod creates an invoice.

So a Raw selection does lead into the payment path, and the model MC behaves differently after a
Raw visit. The spec resets only `bub-mc-regime` on a switch.

The reverse direction also breaks. In model mode, sims means "example lines", and users set small
values (the Tab-1 sims default was 8 until recently). Switching to Raw then turns sims=8 into an
8-path fan, whose 5–95 % band is meaningless.

**Fix:** Give Raw its own control (e.g. `bub-mc-raw-sims`, default 1000; Lines shown already
covers the line count), or restore the model-mode values on every switch back. Either way:
- pin a test "Raw → model with sims 3200 / bins 8 still renders the free cached fan";
- in Raw, warn in the status line when paths < ~100.

### I-2. The Raw regime filter does the opposite of what its labels say, and its effect is mostly sampling noise

**Where:** Regime filter in Raw mode ("Unchecked bins are never sampled", the checklist labels
"−55 % to −10 %").

**Failure 1: direction.** The filter drops history months by the momentum state that *preceded*
them. It places no limit on the state the simulated path itself reaches. In model mode the same
checklist means "paths cannot leave selected regimes" (the paid-live note). Users will read the
Raw labels the same way.

Measured, window 2015–2026, 5 bins (edges −72 / −33 / +32 / +93 / +168 % trailing-12-month
return):
- **Only the coldest bin ticked:** this samples the rebound months after crashes. Mean next-month
  log return is +0.043, about ×1.66 a year. The simulated paths' own trailing-12-month return has
  a median of **+65 %**, and only **6 %** of simulated years sit in the coldest bin.
- **Bin 1 (−33 % to +32 %):** this is the bearish one, at ×0.83 a year.

A user asking for "cold regimes" gets a bull fan.

**Failure 2: noise.** Each bin holds ~28 months, so the bin mean has a standard error of
≈0.036/month. The spread between bins (−0.015 to +0.070/month) is within ~2.4 SE, consistent with
the spec's own finding that monthly autocorrelation is inside noise since 2015. Compounded over
480 months, the 40-year medians per bin run from **×0.0007 to ×3·10¹⁴**. Different boxes give
wildly different pictures, and the differences come from which 28 months were drawn, not from
regime structure. The fan percentiles show within-sample spread only, so the picture looks
confident when it is not.

**Fix (operator call; pick one):**
- (a) Make the filter act on the **simulated** state: condition each simulated month on the path's
  own trailing-12-month return, and resample or reject draws that would leave the checked bins.
  This matches model-mode semantics.
- (b) Keep the current mechanics but relabel honestly ("months that followed a −72 %…−33 %
  year"), show the per-bin pool size, and add the status line "N months in pool; results are
  dominated by which months these are" whenever the pool is < ~60 months.

Either way, raise the "too small to simulate" floor above 12 months. 12 months iid-resampled for
40 years is pure noise.

### I-3. Thin-phase fallback counts overlapping block starts, not independent episodes (reopens the folded answer)

**Where:** Raw · halving cycle, "Thin phases": fewer than 3 distinct candidate block starts → fall
back.

**Failure:** Consecutive start months in one cycle share 2 of their 3 months, so they are one
episode, not several. Measured, window 2015–2026, as starts per phase / distinct cycles:
- **All bins on:** 15–21 starts per phase, but only **3 distinct cycles** (4 for phase 4). The
  spec's "11–22 starts" overstates the diversity about 6×.
- **Bins 0–1 on:** phase 0 has 4 starts from **1 cycle**. This passes the ≥3 rule, so every
  path's phase-0 blocks copy near-identical three-month returns from a single 2016/2020/2024
  episode. Phases 3 and 6 are similar (7 starts / 2–3 cycles).
- **Hottest bin only:** phase 1 has 7 starts from 2 cycles; phase 2 has 12 from 2.

Result: within those phases the 3200 paths move almost in lockstep, and the fan narrows exactly
where the evidence is thinnest. The fold did not catch this because it counted starts.

**Fix:**
- Define the threshold on **distinct cycles (episodes)**, e.g. fewer than 2 distinct cycles → fall
  back. Report it in the status line as "phase N drew from 1 past cycle".
- State in the spec, and in the variant's caveat, that unfiltered cycle mode replays about 3
  historical cycles per phase under the 2015 window.

### I-4. The 2015 window nudge does not stop absurd projections, and nothing on screen says why the fan is where it is

**Where:** "Default window when switching to Raw" (controller's call). Anti-goal 1. The Years
default (`bub-mc-years` = 40).

**Failure:** The spec moves the window to 2015 because 2010–now "makes Raw absurd". 2015–2026 is
still absurd at the default horizon. Its mean monthly log return is 0.0388, about **+59 % a year**.
Terminal p5 / p50 / p95 per BTC, starting at $75 k:

| Variant, window 2015–2026 | 10 years | 40 years |
|---|---|---|
| iid | $0.25 M / **$7.6 M** / $229 M | $6.9·10⁹ / **$9.5·10¹²** / $8·10¹⁵ |
| cycle | $1.6 M / $29 M / $530 M | median **$1.2·10¹⁴** |

The y-axis slider tops out at $1 B (`bub-yrange` max 9), so most of a 40-year Raw fan cannot be
displayed at any zoom.

The legend suffix "2015–2026" meets anti-goal 1 only on paper. The viewer is told which years were
used, not that the fan is those years' average growth compounded.

**Fix:**
- The Raw status line always states the window's implied growth, e.g. "History 2015–2026 grew
  +59 %/yr on average; the fan compounds that."
- Default Raw's horizon to 10 years on a real switch, as the window nudge does.
- Let the operator reconsider the nudge target. A 2018 start gives ×7 in 10 years instead of ×105.
- Pin a test that the status line carries the growth figure.

### I-5. The click-stamp pattern cannot see a Dash 4 Dropdown selection

**Where:** "Switching meaning" (regime reset) and "Default window when switching to Raw". Both say
"a real user selection … click-stamp pattern".

**Failure:** In Dash 4.0, `dcc.Dropdown` puts the component `id` on the trigger `<button>`. The
options list is a Radix Popover portal rendered into the sibling `.dash-dropdown-wrapper` div
(`async-dropdown.js`: `b.ZL,{container:cn.current}` with `ref:cn` on `dash-dropdown-wrapper`). The
existing pattern (`assets/mc_autowiden.js`, `mc_regime_paths.js`) checks
`t.closest('#bub-mc-enable')`. Copied as `closest('#bub-mc-model-src')`, it stamps only the click
that **opens** the menu, never the option click.

Consequences:
- A user who takes longer than the stamp window to choose gets no window nudge and no regime reset.
  A model-mode "Bargain only" filter then silently turns into "coldest momentum only".
- Keyboard selection (arrows + Enter) never stamps.
- A fast Playwright E2E would pass anyway, so the test would not catch this.

**Fix:** Stamp on `pointerdown` / `keydown` inside `.dash-dropdown-wrapper` elements that contain
`#bub-mc-model-src` (portal content included). Alternatively, use a "previous value + not during
restore" guard (`snapshot-pending` / `loaded-hash-store`) in place of the time window. The E2E
should include a delayed selection and a keyboard selection.

---

## Minor

- **M-1. Partial current month** (History pool: "the last year running to the latest close").
  `resample('ME')` yields a Nov-2026 "month" of 8 days (−4.0 %), which would be sampled as a full
  monthly return. **Fix:** drop the incomplete trailing month; test it.
- **M-2. Seed stability** (Determinism). With 5 gunicorn workers, seeding via Python `hash()`
  (salted per process) gives a different fan per worker. Toggling "Lines shown" would then
  reshuffle the fan, and single-process tests would not notice. Separately, "same share link →
  same fan" holds only on the same day, because the seed includes today and the start price is the
  live ticker. **Fix:** require a stable digest (e.g. `hashlib.blake2b` of canonical JSON), and
  state that the fan scales with the start price and re-draws daily.
- **M-3. Time axis origin** (Starting point). update_bubble's existing `t_axis` runs from
  `yr_to_t(int(mc_start_yr))`, i.e. 1 January. Reusing it with the current year would plot today's
  price ~10 months early. **Fix:** require the Raw `t_axis` to start at today's fractional t, with
  monthly steps.
- **M-4. Premium chrome via the match indicator** (Raw status line / paid chrome). When the Raw
  rendered key matches, `_MC_MATCH_JS_TPL` applies `mc-premium-chart` (gold border) and shows
  `bub-mc-badge`. When it does not match, it shows the overlay and "Restore last settings". The spec
  pins only "no mismatch warning". **Fix:** pin "Raw: no premium class, no badge, no restore
  button".
- **M-5. Callback wiring not stated** (Share links / Constraints). New Inputs `bub-mc-lines` and
  `bub-mc-extremes` must be appended **last** on update_bubble (stale-client rule, as for
  `bub-mc-paths`) and added to `_POST_RESTORE_TRIGGERS`. A `bub-mc-raw-status` Output grows the
  8-tuple; every `(dash.no_update,) * 8` must change with it. **Fix:** list these in Constraints.
- **M-6. Cycle block phase alignment** (Raw · halving cycle). Phases are 6 months wide but blocks
  are 3. A future block at month offset 4–5 of a phase can copy a history block from offset 0, so
  rhythm alignment drifts by up to 5 months, and future blocks that straddle a phase boundary take
  only their first month's phase. **Fix:** match the history start's months-since-halving to the
  future block's within ±2 months, falling back to the phase bucket.
- **M-7. The window nudge persists into model mode** (Default window). The 2015 window stays after
  switching back. Paid live model runs (`mc_window` → transition matrix) and `_ghost_match` then
  use it, behind the Advanced fold. **Fix:** restore the pre-nudge window on a real switch back,
  or state that it persists.
- **M-8. Model-mode labels shown in Raw.** "Markov transition matrix dimension 5×5" (bins) and
  "Simulations (1–3200; ≤200 is free)" are wrong or misleading in Raw. **Fix:** relabel in Raw
  mode, as the spec already does for the regime options.

## Nit

- **N-1.** "Same click-stamp pattern as the window nudge": no window nudge exists today. The
  pattern meant is the MC-enable auto-range stamp (`mc_autowiden.js`).
- **N-2.** Lines are ranked "at the last plotted step", so the sample and the extremes change when
  the user pans the x-range. That is fine, but say so, so that a test does not expect them to be
  stable under zoom.
