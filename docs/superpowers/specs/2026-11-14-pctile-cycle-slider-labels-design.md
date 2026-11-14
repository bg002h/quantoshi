# Percentile cycle fit — slider-driven extrapolation + label switch (design note + plan)

Follows `2026-11-10-percentile-cycle-fit-design.md` (shipped `15b3b926`).
Light process agreed with the operator 2026-11-14: this note (design + plan in
one) → one independent Opus review → TDD build → full suite → one Opus
whole-diff review → prod deploy.

## Intent (operator's words, condensed)

1. "I want the x-range (year) slider to control the plot, including the
   extrapolation… 2033 is a perfectly reasonable end year to inherit, but if
   user changes xrange to reach 2080, then extrapolation must go to 2080."
   Drop the +10/+20/+30 yr menu.
2. A switch to turn the peak/trough text boxes on and off — they sometimes sit
   on top of the **current price** (the next trough is ~late Nov 2026 – early
   Jan 2027 under QR/PL/SatPL/LPPL, i.e. on today). Default **on**.
3. Label text gains the **date range** of each turn (the timing bar's span)
   and the **price at each end** of that range.

Non-goals: no change to the fit, bands, hindcast, gating, model picker or
element menu; no per-pill slider default (the Percentile view inherits the
shared `bub-xrange`, default end 2033); no new label placement/collision logic.

## Decisions

**D1 — Slider drives the forecast.** With a non-weak fit and the slider's right
handle at or past today (`yr_to_t(xmax) >= today_t`, today's rule), the forecast region runs from the last close to `yr_to_t(xmax)`;
the x-axis ends there and the year ticks run to `xmax`. Otherwise (fit off,
weak fit, or `xmax` at/before today) behaviour is unchanged: the axis is
capped at today (`t_hi = min(yr_to_t(xmax), today_t)`). Slider domain is
`layout/bubble.py::XRANGE_MIN/MAX` = 2010/2080. Rationale for keeping weak /
off capped: there is nothing to draw in the future.

**D2 — Band grid reaches the slider's end.** `_cycle_result`'s weekly grid
currently stops at last close + `_GRID_PAST_LAST` (30 yr ≈ 2056). It must
reach `XRANGE_MAX` (+1 yr margin): grid end = `max(last + 30, yr_to_t(2081))`.
`figures/percentile.py` must not import `layout` (layering): define the
constant in `pctile_cycle.py` or `figures/percentile.py` and add a test that
it is ≥ `layout.bubble.XRANGE_MAX + 1`. Cost: ~3700 vs ~2600 grid points,
one-off per (model, σ-mode, censor, element, fp) in the existing
`lru_cache(16)`; payload scales with the visible span (default 2033 is
*smaller* than today's default +20 yr).

**D3 — Horizon menu retired, kept hidden.** `bub-pctile-fit-horizon` stays in
the layout as a hidden component (precedent: `hm-model-show`, kept hidden for
snapshot compat), stays in `_SNAPSHOT_CONTROLS` (index 351, append-only),
`_TAB_CONTROLS["bubble"]`, and `SNAPSHOT_DEFAULTS` (20) — so the defaults
fingerprint does **not** change because of it and old links decode exactly as
before. It is removed as an `Input` of `update_bub_pctile`, from
`_cycle_params` (no `cycle_horizon` key any more) and from
`_PCTILE_SUB_CONTROLS`. `pctile_cycle.HORIZONS/DEFAULT_HORIZON` and
`figures/percentile._LINEAR_LABELS` are deleted (grep for every user first;
tests that pin them are updated). The hidden dropdown keeps its options as a
literal `[10, 20, 30]` in `layout/common.py` (review N-2), so a restored `30`
is not dropped by an option-less dropdown; D7 uses the same literal set.

**D4 — Label switch is a NEW, separate checklist.** `dcc.Checklist(
id="bub-pctile-fit-labels", options=[{"label": "labels", "value": "labels"}],
value=["labels"], inline=True)` placed right after `bub-pctile-fit` in
`_pctile_cycle_controls()`. **Not** a third option of `bub-pctile-fit`: a link
that stored that group as `["fit"]` (or a v4 link whose fingerprint registers
`['fit']`) would restore labels **off**, the opposite of today's behaviour.
As its own field, every older link lacks the key and restores at the layout
default `["labels"]` — which is exactly what prod shows today — so **no**
`_PREDATING_LINK_VALUES` entry. Wiring: append `("bub-pctile-fit-labels",
"value")` at index 352; `_CHECKLIST_OPTIONS["bub-pctile-fit-labels"] =
["labels"]`; `SNAPSHOT_DEFAULTS['bub-pctile-fit-labels:value'] = ['labels']`
via `tools/update_defaults_registry.py` before and after (new fp registered,
oldest evicted at the 20 cap — check the evicted one is not the current
prod-era fp `757574d3`); `_TAB_CONTROLS["bubble"]`; new `Input` on
`update_bub_pctile`; `_cycle_params` gains `cycle_labels: bool` (fit off →
fixed default `True`); add the id to `_PCTILE_SUB_CONTROLS` so toggling it with
the fit off is a no-op. `_TAB_CONTROLS["bubble"]` gains the id; the deep link
(`routing.py`) needs no new outputs (the switch lives inside the existing
`bub-pctile-ctl-wrap`).

**D5 — Labels: content and count.**
- Only **forecast** turns are labelled (as now), and only the **next peak and
  next trough** on both log and linear axes (`n_lab = 1` always). Later turns
  keep marker + timing bar + hover.
- Switch off → no turn annotations; markers, bars, hover, warning boxes and the
  censor tag are unaffected.
- Text (three short lines, `<br>`-separated, existing `_money` / `_mon_yr`):
  `Peak Sep 2028 · Q79 · $P` / `range Mar 2028 – Apr 2029` / `$A – $B`, where
  `$A`, `$B` = the turn's percentile `v` read off the model's fan
  (`_pc.fan_prices`) at the range's start `a` and end `b`. `Trough` likewise.
- Capped end (`b ≥ t_last + _TIMING_CAP`): date reads `Apr 2129 or later` as
  in today's hover, and that end's price is shown as `—` (a fan price 100 yr
  out is meaningless). Same for a start clamped to the first close.
- The marker hover gains the same two range prices.
- Placement (review M-3, M-4): peaks anchor `yanchor="bottom"` with
  `yshift≈+8`, troughs `yanchor="top"` with `yshift≈-8` (a 3-line box centred
  on `yshift=24` would cover its own marker). Raise the y headroom from
  `[-12, 112]` so a Q95 peak's / Q5 trough's 3-line box stays inside the plot
  area (pin with a test on the range bound; exact value the builder's call,
  checked visually on a 450 px and a mobile-width chart). Horizontal: when
  `ax_x(x)` lies in the last 15% of the axis span use `xanchor="right"`, in the
  first 15% `xanchor="left"`, else centre — one conditional, no collision
  logic. Keep font/bg/border as now.

**D7 — Old links keep their forecast length (operator: "I want this link to
still work").** Under D1 alone, a link made before this change would load fine
but draw a shorter forecast than its sender saw. The operator's own link
(`https://quantoshi.xyz/1#q4:757574d3:H4sIAEWp-GoC_w2MMQrEMBAD_6J6bXZtbwz-SkgR7lwEjAkhqcL9_YQKMQzSiupM-WbIi4yGccy-XxA42prUVJKmugkWcrDo4tGJRgbVEsyCFQ5MK5oKkrNx9uvT532MTpUz36zYT-YzxvYHsN3TJXYAAAA=`)
decodes (current code) to `bub-view-mode: percentile`, `bub-xscale: linear`,
`bub-xrange: [2010, 2027]`, `bub-pctile-fit: ['fit']`, model `qr`, basis
`cal`, horizon `20`: prod draws the forecast to today + 20 yr (≈ Nov 2046);
D1 alone would stop it at 1 Jan 2027. Migration, applied inside/right after
`_fill_predating_fields` (both live decoders, `_decode_snapshot` and
`_decode_snapshot_v4`, return through it; v1 links predate the feature): a
decoded state that **lacks** `bub-pctile-fit-labels:value` **and** has view
`percentile` **and** `fit` ticked **and** a horizon `h` in (10, 20, 30) **and**
a list-valued `bub-xrange` whose end `e` satisfies `e >= current year + 1`
(prod's rule: forecast drawn iff `yr_to_t(e) >= today_t`; with integer years
that is `e > current year`) gets `bub-xrange` end set to
`min(current_year + h + 1, XRANGE_MAX)` — the whole year that contains prod's
`today + h` edge, which is where prod's axis ended whatever `e` was (review
M-1: exact, not `max(e, …)`; M-2: clamped to the slider domain, via a
snapshot-local constant pinned by a test to `layout.bubble.XRANGE_MAX`).
States missing `bub-xrange` or with a non-list value are skipped. Gated on
view `percentile` so a link opened on another pill keeps its range. "Current
year" is evaluated at restore time, matching prod (whose edge was `today + h`
at view time, not at link time).

*Why absence is a sound trigger (review I-1).* A v4 link fills omitted fields
from its own registered fingerprint's defaults, so every link made after the
new fingerprint is registered carries the key, and a `757574d3` link does not.
But a link whose fingerprint has been **evicted** from the 20-entry registry
takes the unregistered branch (`historical = dict(SNAPSHOT_DEFAULTS)`), which
would hand it today's key and silently skip D7. Fix: a
`_MIGRATION_SENTINELS = ("bub-pctile-fit-labels:value",)` tuple, included in
that branch's pop loop alongside `_PREDATING_LINK_VALUES` (keys the oldest
registry entry lacks are removed). Positional links pad with `None`, so they
lack the key; they also predate the cycle fit (fit → `[]`), so D7 is inert
for them.

Tests: the operator's link above, decoded, restores `bub-xrange` =
`[2010, min(date.today().year + 21, 2080)]`; the same state with view `price`
keeps `[2010, 2027]`; xrange end ≤ current year → untouched; a state carrying
`bub-pctile-fit-labels` → untouched; `h = 10, e = 2060` → end
`current_year + 11`; a forced end past 2080 is clamped; **an unregistered-fp
link whose oldest registry entry lacks the labels key still triggers D7**.

**D6 — Copy.** Summary text and subtitle carry no horizon wording today; no
change needed beyond any test that asserts horizon-dependent output.

## Plan (TDD; one commit per task; `set -o pipefail` on every gate)

1. **Slider forecast + grid (D1, D2, D3-figure side).** Tests in
   `test_pctile_cycle.py`: forecast trace's last x ≈ `yr_to_t(xmax)` for
   xmax = 2033 and 2080; no forecast trace for xmax = this year − 1, a weak
   model, or fit off; x-axis range ends at `yr_to_t(xmax)` when forecasting;
   band grid end ≥ `yr_to_t(XRANGE_MAX + 1)`; ticks include 2080 at xmax 2080.
   Then implement in `figures/percentile.py` / `pctile_cycle.py`; delete
   `HORIZONS`, `DEFAULT_HORIZON`, `_LINEAR_LABELS`.
2. **Labels (D5).** Tests: 2 forecast-turn annotations (one peak, one trough)
   on linear and log axes at xmax 2080; zero with `cycle_labels=False` while
   markers + error bars remain; text contains the kind word, a ` – ` date
   range, and two `$` prices that equal `_money(fan_prices(...))` at `a`/`b`;
   capped end renders `or later` and `—`; peak `yanchor` bottom / trough top.
3. **Controls + wiring + snapshot (D3, D4).** Tests: horizon dropdown hidden
   (style display none) and not an Input of `update_bub_pctile`; labels
   checklist present with default `["labels"]`; `_cycle_params` keys;
   `_PCTILE_SUB_CONTROLS` no-op with fit off; snapshot index 352 and
   `_CHECKLIST_OPTIONS`; an old v4 link at fp `757574d3` and a positional link
   without index 352 both decode with the labels key **absent**, and the
   layout default is `["labels"]` (review N-3); a new-fp link with labels at
   default decodes **with** the key;
   registry round-trip; D7 migration tests (operator's link verbatim).
   Update count-pinning tests (Input counts etc.).
4. **Gate:** full `pytest btc_web/` (expected: only the known
   `test_free_tier_all_models` failure), `scripts/sabotage_callback_guards.py`,
   py_compile on prod's 3.12 if available. DEV server: Playwright journey on
   `/1.4` — default (2033), slider to 2080, labels off/on, BM (weak), linear
   axis; 0 new console errors. Then whole-diff Opus review → fold → prod
   deploy (`git pull` + restart; no pkl change, no FLUSHDB) → prod journey.

## Review focus (inputs no task pins by default)

- Slider right end just past today (forecast region of a few weeks): labels
  for turns beyond `xmax` must not be drawn off-axis.
- Slider moved while the fit is off, then fit turned on: must use current xmax.
- Linear axis at 2080: ~15 forecast cycles of markers/bars — readable? (labels
  limited to 2 by D5).
- A turn whose timing bar's start is before today (wide spread): range price
  at a past date is still a fan price — acceptable, but must not crash.
- Old share link carrying horizon = 30 on the Percentile view: D7 widens its
  slider end to current year + 31; the restored slider then also applies if
  the viewer switches to the Price pill (accepted: the link's own view is what
  must match).
