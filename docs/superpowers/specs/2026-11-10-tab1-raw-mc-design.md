# Tab 1 "Raw" Monte Carlo — design

Status: draft for review, 2026-11-10. Operator-approved through the
brainstorm; items marked **(builder's call)** or **(controller's call)** were
not decided by the operator.

## Why

Every MC option today simulates transitions between percentile bands of a
chosen price model, so the fan inherits that model's assumptions. The
operator wants a model-free option: a fan made only from Bitcoin's own price
history. The first question a viewer will ask of it is whether it reflects
Bitcoin's momentum and its 4-year boom/bust rhythm. So it ships in two
flavours that answer that question side by side: one with no memory, and one
that follows the halving cycle.

Measured on `BitcoinPricesDaily.csv` (script:
`scratchpad/raw_mc_measure.py`, figures "since 2010 / since 2015"):

- month-to-month return autocorrelation +0.24 / +0.11 (noise ±0.14 / ±0.17);
  P(up | up month) 0.60 vs P(up | down month) 0.55;
- variance ratio at 6–12 months 1.4–1.56: multi-month swings are ~1.5× wider
  than independent months produce;
- volatility clustering (|daily return| lag-1 autocorrelation) +0.46 / +0.38;
- 18–30 months after a halving the mean monthly return was negative in
  cycles 1–3; the current cycle partly broke the pattern;
- bull years shrink (×15, ×55, ×14, ×4): sampling 2010–2013 projects absurd
  prices, so the history window is effectively the model.

## Goals

1. Two new entries in Tab 1's **Model source** dropdown (`bub-mc-model-src`):
   **Raw · independent** (value `raw_iid`) and **Raw · halving cycle**
   (value `raw_cycle`). No other tab gets them.
2. Raw is **free for everyone**, computed live per request, no cache build,
   never a payment prompt, regardless of sims, bins or years.
3. **Sims** (`bub-mc-sims`, 1–3200, unchanged control) is the number of
   simulated paths; the fan is the percentiles of all of them.
4. The example lines are a **small sample spread across end prices**
   (new control **Lines shown**, 5–200, default 20), plus an optional
   **Show extremes** toggle that adds the single highest- and lowest-ending
   paths.
5. The existing **training-window slider** (`bub-mc-window`) picks which
   years of history are sampled; that is how the user includes or excludes
   bubbles (trimming from either end is enough — no need to skip a middle
   one).
6. The existing **regime filter** (`bub-mc-bins` + `bub-mc-regime`) is reused
   with a new meaning in Raw mode: bins of trailing 12-month return.

## Non-goals

- Any tab other than Tab 1.
- Changing the model-based MC on Tab 1 in any way (behaviour, look, the
  "best 50 of 200" regime subset, sims meaning, defaults).
- Skipping a middle bubble (per-cycle checkboxes).
- Daily-step simulation, GARCH or any fitted volatility model.
- A paid tier or cached precompute for Raw.

## Anti-goals (would be a failure even if it "works")

- Raw silently projecting absurd prices with no visible statement of which
  years it learned from.
- Example lines hiding the bands (the reason for goal 4).
- Restyling the approved MC look (amber fan, dark-neutral lines,
  `MC_PATH_ALPHA_BUDGET / n` opacity), the shared `_MC_BANDS` /
  `_GHOST_BANDS`, or any CB palette. The operator is deuteranomalous: never
  tell lines apart by red/green hue.
- A Raw selection leaking into a payment, cache or BTCPay code path.

## Behaviour

### Starting point and time axis

Raw has no model percentile to start from, so a Raw run **always starts
today at the current price** — the live ticker price when available
(`btc-price-store`), else the last daily close **(controller's call)**. Tab 1's
**Start year** and **Entry percentile** controls do not apply in Raw mode:
they are hidden (or disabled) with a one-line hint "Raw starts today at the
current price" **(controller's call; hidden vs disabled is the builder's)**.
The horizon is the existing **Years** control (`bub-mc-years`). Steps are
monthly.

The MC auto-range (real tick of MC sets `bub-xrange` to [current year,
max(old right, MC start + 10)]) uses the current year as the MC start in Raw
mode.

### History pool

Month-end closes from the full price CSV give monthly log returns. A month
is **eligible** when it falls inside the window slider's years (inclusive,
the last year running to the latest close). Each eligible month carries:

- its log return;
- its **momentum state**: the trailing 12-month log return ending at the
  start of that month (months without 12 prior months of data have no state);
- its **halving phase**: whole months since the most recent halving
  (`figures/common.py::_HALVING_KNOWN_DATES`), bucketed into 6-month phases
  0–7 (≥ 48 months folds into phase 7).

### Raw · independent

Each simulated month is an independent draw (with replacement) of one
eligible month's log return. No memory: the label says so (see Labels).

### Raw · halving cycle

The simulated future is walked in **3-month blocks**. Each block's phase is
the phase of its first month relative to the projected halvings
(`_HALVING_EPOCHS`, which extends past 2024 at a 4-year cadence). The block
copies three consecutive real monthly returns starting at a random eligible
month of the **same phase**; the three months must all be eligible (inside
the window). Momentum survives inside a block; the bull/bear rhythm comes
from the phase. If a phase has no eligible block under the current window
and filter, that block falls back to the whole eligible pool, and the status
line says how many phases fell back **(controller's call)**.

### Regime filter in Raw mode

- **Bins**: `bub-mc-bins` equal-count bins of momentum state over the
  eligible months, coldest first.
- **Checklist labels** (`bub-mc-regime` options) change in Raw mode to show
  each bin's trailing-12-month return range, e.g. "−55 % to −10 %", coldest
  to hottest. Model mode keeps today's labels.
- **Unchecked bins are never sampled**: an eligible month (independent) or a
  block's first month (cycle) whose momentum state is in an unchecked bin is
  removed from the pool. Months with no state are dropped only while a
  filter is active.
- Same server rule as today: the filter applies only when
  0 < checked < bins; none checked means all allowed.
- No ranking and no "best N of M" relabel: the fan is the filtered result.
- The Tab-1 regime note (`bub-mc-regime-note`) gets Raw wording, e.g.
  "Raw: history from unchecked momentum bins is never sampled." The
  "best 50 of 200 … not a new simulation" text stays for model mode only.
- The auto-tick of "Show individual price paths" on a real filter click
  (`MC_REGIME_PATHS_JS`) is unchanged and applies in Raw mode too.
- If the filtered pool is too small to simulate (fewer than 12 eligible
  months **(controller's call)**), no fan is drawn and the MC status line
  says why.

### Sims, lines and extremes

- `bub-mc-sims` = number of simulated paths (1–3200). The default stays 200
  (the Tab-1 snapshot default) **(controller's call)**.
- **Lines shown** (`bub-mc-lines`, number input 5–200, step 1, default 20;
  visible only in Raw mode). Paths are ranked by price at the last plotted
  step (the visible right edge) **(controller's call)**; lines are taken at
  evenly spaced ranks from the 2nd to the 98th percentile. If lines ≥ sims,
  all paths are drawn.
- **Show extremes** (`bub-mc-extremes`, checklist `["yes"]`, default off,
  Raw only). Adds the highest- and lowest-ranked paths of the whole run, on
  top of the sample, dashed and slightly bolder in the same dark neutral
  (new constants in `colors.py`), legend entries "Highest end" and
  "Lowest end", no hover.
- Both only matter while **Show individual price paths** (`bub-mc-paths`) is
  ticked; it stays the master switch, default off.
- Lines keep today's payload treatment (every other step, 2 significant
  figures, `_add_mc_spaghetti`). At ≤ 200 + 2 lines that is ≤ ~25 KB gzip on
  top of the ~38 KB fan (measured: `scratchpad/raw_stride.py`).

### Labels

- Fan legend entries carry a suffix naming the variant and window, e.g.
  "MC median · raw independent 2015–2026" (`_add_mc_fan(label_suffix=…)`).
- Independent mode shows a one-line caveat under the dropdown: "No memory:
  understates multi-month swings."

### Default window when switching to Raw

The shared window default is 2010–now, which makes Raw absurd. On a **real
user selection** of a Raw entry in the dropdown, if the window starts before
2015 it is moved to start at 2015 (click-stamp pattern, as for the MC
auto-range: never on share-link restore or programmatic writes)
**(controller's call)**.

### Determinism

Same settings → same picture: the RNG is seeded from the inputs (variant,
window, bins, checked bins, sims, years, start date, price-data
fingerprint). Two requests with the same share link draw the same fan.

### Payment and caching

Raw never calls the BTCPay / free-tier / MC-cache paths and never opens the
payment modal. Nothing new is stored server-side. Target: a 3200-path ×
480-month run in well under 100 ms on prod.

### Share links

- `raw_iid` / `raw_cycle` are new values of the existing
  `bub-mc-model-src` field.
- Append `("bub-mc-lines", "value")` and `("bub-mc-extremes", "value")` at
  the absolute tail of `_SNAPSHOT_CONTROLS` (indices 340 and 341); add
  defaults (20, `[]`) and `bub-mc-extremes` to `_CHECKLIST_OPTIONS`;
  register the old and new defaults fingerprints
  (`tools/update_defaults_registry.py`); add both to
  `_TAB_CONTROLS["bubble"]`.
- Links made before this change decode with the defaults; model-mode links
  are unaffected.

## Constraints (standing project rules that bind the builder)

- TDD; full non-E2E suite green; E2E for real-click behaviour (window
  nudge, auto-range in Raw mode).
- Runtime imports must be in `btc_web/requirements.txt` (numpy/pandas only
  expected).
- Clientside callbacks with one Input when using `allow_duplicate` +
  `prevent_initial_call` (Dash 4.0 silently drops multi-Input ones).
- Colours and widths live in `colors.py`
  (`test_colors_central.py`).
- `_SNAPSHOT_CONTROLS` is append-only.
- Cache-key alignment: if new controls reach the bubble params dict, keep
  `tab_defaults.py` / prewarm aligned (`test_cache_key_alignment.py`); MC
  traces are added on a clone after the cached figure, so ideally they do
  not.

## Left to the builder

Module layout (a numpy-only `btc_web/raw_mc.py` is the obvious home), how
the regime options are relabelled, hidden vs disabled for start-year /
entry-percentile, the exact caveat and note wording, the extremes' exact
width/opacity, and test structure.
