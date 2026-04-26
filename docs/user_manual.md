# Quantoshi -- User Manual

A guided tour of [quantoshi.xyz](https://quantoshi.xyz) for the curious but non-developer visitor.

---

## 1. What Quantoshi is / isn't

Quantoshi is a **Bitcoin price projection toolkit**. It takes ~15 years of
daily closing prices and fits 20+ different statistical models to them --
a power law, a bubble model, quantile regression, several log-periodic
oscillators, and a few experimental forms. You can then visualize where
today's price sits inside each model's projected distribution, play
"what-if" with DCA and retirement simulations, and plan a long-term
multi-asset citadel.

**What it is:**

- A visualization and simulation tool, powered by math.
- Free to use for core features. Monte Carlo simulation (the only truly
  compute-heavy feature) is gated behind a small Lightning payment.
- Privacy-respecting. **No account, no login, no cookies, no server-side
  storage about you.** Your saved lots live in your browser's
  `localStorage` and go nowhere else. The site is fully usable from Tor.

**What it isn't:**

- Not investment advice. Seriously. Do not trade on what you see here.
- Not a prediction engine. The models describe *where historical prices
  have been* and extrapolate those patterns forward. Past performance
  does not guarantee future results.
- Not a news site, not a price tracker, not an exchange. It does show a
  live BTC price ticker (refreshed every 20 min), but that's the only
  live data pulled in.

Quantoshi is available at two endpoints:

- Clearnet: `https://quantoshi.xyz`
- Tor onion: `u5dprelc4ti7xoczb5sbtye6qidlji2l6psmkx35anvxgjyqrkmu32ad.onion`

Both serve the same application. The onion service is the preferred way
to visit if you care about network-level privacy.

<!-- merged from v1: privacy + server-log retention detail -->
### Privacy, in detail

All your data stays in your browser's `localStorage`. Quantoshi stores
nothing server-side -- no accounts, no cookies, no tracking. Your lot
data, settings, and link history never leave your device. Export/import
uses local file downloads.

Server logs are retained for 27 days and contain only standard web
server request data (IP, timestamp, URL path). Tor users can access
the `.onion` address for additional privacy.

---

## 2. How to read the charts

Most Quantoshi charts share a common visual grammar. Once you can read
one, you can read them all.

### Axes and scale

Bitcoin's price has grown roughly along a power law for 15 years. That
means a **logarithmic Y axis** (and often a logarithmic X axis too) is
essential -- a linear axis compresses the interesting early history into a
flat line along the bottom.

- **Y axis (price)**: `Log` by default. `Linear` is available but rarely
  useful except over short time windows.
- **X axis (time)**: `Log` by default on Tab 1. Linear is available.
- **Auto Y**: When checked, the price range auto-fits to whatever
  quantiles and years you've selected. Uncheck if you want to pin the Y
  range manually.

<!-- merged from v1: plainer "what is log scale" prose -->
A move from $1,000 to $10,000 on a log axis looks the same as $10,000 to
$100,000 -- equal vertical distances represent equal percentage changes.
This is the natural scale for Bitcoin's exponential growth. Linear is
useful for seeing absolute dollar differences but makes early history
invisible.

### Scatter points = history

The small dots are actual daily closing prices. One dot per day. Point
size and alpha are tunable under "Plot Appearance."

### Shaded bands = model quantiles

Each model produces a distribution, not a single line. Quantoshi renders
that distribution as shaded bands:

- **Q50%** is the median -- the "if the model is right, the price is
  equally likely to be above or below this line."
- **Q15% / Q85%** bracket the 70% central interval. Below Q15% counts as
  "notably low"; above Q85% counts as "notably high."
- **Q1% / Q99%** are extremes. Historically cheap (Q1%) or historically
  expensive (Q99%).

You can select which quantiles to show under "Projection Quantiles." The
default panel offers three symmetric band checkboxes (median, inner,
outer). Click "Advanced" to pick from the full ~17-quantile menu.

### "Today" line + watermark

A vertical line marks today's date. The Quantoshi logo watermark sits in
the chart corner -- it's faint on purpose. Chart title includes the model
name; hover any trace for a tooltip with exact values.

<!-- merged from v1: recovery-time annotation -->
### Recovery time on hover

For non-monotonic model traces (like the Bubble Model) and historical
price data, hovering a point shows how long until the price recovers
to that level -- e.g., "Recovery: 3.1 yr (Dec 2028)". This tells you
how long a buyer at that price would wait to break even. Points on
rising segments show no recovery annotation (no drawdown to recover
from).

### Plotly toolbar

Hover the chart and a Plotly toolbar appears in the top-right: zoom, pan,
box-select, autoscale, download-as-PNG. There's also a dedicated `PNG`
export button below each chart tab.

---

## 3. Navbar features

Across the top of every page:

- **Logo + Quantoshi wordmark**. Clicking the logo triggers a playful
  easter-egg animation; clicking the wordmark takes you home.
- **Live BTC price ticker** -- `₿ $68,204` style. Refreshes every 20
  minutes from Binance (CoinGecko fallback). Mono-spaced, tabular nums.
- **24-hour sparkline SVG** -- a tiny price-history sparkline next to the
  ticker.
- **Percentile badge** (e.g. `QR 62%`) -- the current price expressed as
  a percentile of the named model. **Tap the badge to cycle through
  models.** Cycle order: `QR -> BM -> PL -> LPPL3 -> HybPPL -> EPPL -> PCA ->
  Greedy -> EF -> U1` (U1 only if you've drawn one). Each model name is
  color-coded to match its chart trace.
- **Sats/$ toggle** -- a small glyph next to the ticker flips the display
  from `₿ $68,204` to `1,467 sats/$`.
- **Stay dark, Anon -> onion** -- clearnet visitors see a link
  inviting them to switch to the onion service.
- **Palette dropdown** (hidden; per-tab selector is the primary way to
  switch palettes -- see below).
- **Share button** -- opens the share-link modal (section 6).

On mobile portrait, the navbar collapses: logo + price on one row, a
menu toggle reveals the onion link and share button.

---

## 4. Color palettes

Quantoshi ships four palettes:

- **Default** -- warm/cool deltaE-optimized scheme with six flagship
  trace colors. Readable for most visitors.
- **CB-Brian** -- hand-tuned for deuteranomaly (red-green colorblindness,
  site author's vision profile). This palette is load-bearing and is not
  changed without explicit approval.
- **CB-RG** -- general-purpose colorblind-safe palette (Wong-Okabe
  inspired).
- **CB-Full** -- maximum-contrast palette for severe vision deficits.

Switching palette updates **everything** at once: chart trace colors,
heatmap CAGR presets, the Display Models swatches, and the Model Info
chart. The heatmap also auto-picks a sensible CAGR gradient preset for
the active palette (red->white->green for Default; blue->white->orange
for the CB palettes).

**Per-tab palette selector**: at the bottom of each chart tab's control
column. This is the canonical switcher -- changing it propagates
site-wide.

<!-- merged from v1: visual-grammar primer, more accessible wording -->
### Chart reading guide at a glance

- **Model traces**: solid colored lines at varying opacity. Median (Q50%)
  is full opacity; extreme quantiles fade. Each model has a unique
  color in the palette.
- **Quantile bands**: semi-transparent shaded regions between quantile
  pairs (inner / outer). Color matches the owning model.
- **Scatter data points**: small semi-transparent dots showing historical
  BTC daily closes. Dense clusters appear darker due to overlapping
  transparency.
- **Grid lines**: faint warm-gray lines for reference. Log-scale charts
  have denser minor gridlines.
- **Watermark**: small "quantoshi.xyz" logo + text in the chart's
  lower-right corner.

### Display Models section

The "Display Models" card in each chart tab's controls lets you toggle
which price models appear. Models with a gear icon open a configuration
modal when you click the gear. The small triangle icon in a model's row
deep-links to that model's entry on the Model Info tab.

### Plot Appearance

The "Plot Appearance" panel at the bottom of Tab 1's controls lets you
customize trace width, grid colors, grid line widths, and data point
color. Changes are stored in `localStorage` and apply across all chart
tabs. Click "Reset" to restore defaults.

---

## 5. Tab walkthroughs

The app has nine tabs. Each is directly reachable via `/N` URL paths
(see section 8 for the cheat sheet).

### Tab 1 -- Price & Model Overlays (`/1`, `bubble`)

The flagship chart. Shows historical prices plus whatever models you
toggle on, with quantile bands and an optional component decomposition.

**View mode pills** -- Price / CAGR / Residuals.
- **Price** -- the default bubble chart.
- **CAGR** -- forward compound-annual-growth-rate view. Pick a horizon
  (1, 2, 4, 10, 20, or 30 years). Shows a progress bar while computing.
  Intra-window excursion bands indicate peak and trough multiples over
  the window.
- **Residuals** -- shows `log10(price) - log10(model)` at historical data
  points for each active Display Model. A trace at 0 = perfect fit;
  positive = price above model; negative = price below. X-range max is
  auto-capped at the current year + 1 (residuals don't exist for future
  data). Turning off a model clears its residual trace.

**Display Models accordion** -- The centerpiece of Tab 1's right column.
Each row is a checkbox for one model:

- **Bubble Model** (`bub`) -- default flagship. Gear icon opens its
  config modal (N future bubbles slider, activate checkbox).
- **Power Law** (`pl`), **Quantile Regression** (`qr`) -- plain rows.
- **Entropy PPL** (`eppl`) -- master entry. Gear opens EPPL config with
  "down / up" frequency count sliders and per-slot damping toggles. The
  summary in parentheses (e.g. "1d+1u") tells you the active
  configuration at a glance.
- **Hybrid PPL** (`hybppl`) -- master entry with gear config modal
  (Config A + optional Config B for combining two hybrid fits).
- **LPPL** (`lppl`) -- master entry. Gear opens the LPPL config modal (N
  frequencies 1-4, weighted/unweighted, no-1/3 constraint toggles).
- Remaining rows: **PCA**, **Greedy**, **Gompertz**, **Broken Power Law**,
  **Offset Power Law**, **Stretched Exponential**, **Logistic**, **S2F**,
  **Exponential**, **Empirical Floor** (conditional), **U1** (User Model;
  session-only).

Each row has a model-info icon that deep-links to that model's Model
Info card (section 5 Tab 8). Clicking the icon uses SPA navigation -- it
does **not** reset your other tabs' configuration.

**Axes & Range panel** -- Log/Linear X and Y toggles, year range slider
(2010-2080), price range slider (hidden when Auto Y is on).

**Display panel (toggles)** -- Show price data, Shade bands, Show OLS
line, "Unfairly Cheap Line," Show today, Show legend, Minor grid, Enable
chart zoom.

**Projection Quantiles** -- 3 default band checkboxes (Q15-85%, Q1-99%,
Q50%) with an "Advanced" toggle to pick from ~17 individual quantiles.
Chart opacity fades with distance from the median so extreme bands look
lighter.

**Sigma mode** -- Choose between "Constant sigma (legacy)" and "Residual
quantile bands." Residual QR bands (when available) come from fitting
quantile regression to each model's residuals rather than assuming a
constant gaussian sigma. Tab 1 only; LPPL family is excluded.

**Custom Time Axis panel** -- Check "Activate Custom Time Axis" to expose
controls that refit PL / QR / BM-floor / Exponential / Gompertz at a
user-chosen origin.

- Time scale: `Calendar (years)` or `Blockheight (blocks)`.
- Calendar `t0`: preset dates (Genesis, v0.1 release, optimal July
  2009, first $0.01 transaction, etc.) or "Custom..." with a date picker
  (2000-01-01 to 2015-12-31).
- Block `t0`: preset block heights or "Custom..." with a block-number
  input.
- Weighting: `Unweighted`, `1/t`, `1/sqrt(t)`, `Uniform log-t density`.
- Model subset: checkbox list.
- Footer: `$1M Projection Table` -- opens a modal with the projected
  month BTC first breaks $1M, computed across every `t0` x weighting
  combination.

CTA changes affect **only** Tab 1; other tabs keep the default
2009-07-25 origin.

**Model Component Decomposition** -- Pick a model family (BM, LPPL,
HybPPL, EPPL, etc.) and the panel breaks it into additive log-space
terms. Each checkbox toggles one term on/off. When all terms are
checked, `log10(price) = sum of terms` and the chart shows the full
model. Uncheck terms to visualize their individual contribution.

<!-- merged from v1: decomposition card details and tip -->
Each checkbox label shows the term's formula expression, the current
fitted coefficient values, and the individual R-squared for that one
term against actual price data. An "active-selection formula" updates
live as you toggle checkboxes. The chart's trace legend shows the
R-squared of the partial model -- watch it grow as you add components.

For the LPPL family, the decomposition checkboxes update to match
whichever LPPL variant is currently selected. If more than one variant
is ticked, a reminder asks you to pick exactly one.

**Tip**: switch to the **Residuals** view mode with decomposition active
to see where in time the partial model diverges from actual price.
Bumps in the residual curve show eras the current component set
doesn't explain.

**Stack (BTC)** -- Enter a BTC amount (or check "Use Stack Tracker
lots"). When enabled, each quantile legend label shows `-> $X` with the
final USD stack value at the right edge of the selected x-range. Full
portfolio at a glance.

**Model Scanner** -- Enter any two of (price, date, quantile) and the
scanner computes the third across every visible model. Bidirectional:
typing in a price+date gives you the implied quantile in every model;
typing in a quantile+date gives the implied price. The "last two
edited" fields become inputs; the third becomes the output column. `Q`
defaults to unbounded (0.1-99.9%); `Price ($)` takes a number or stays
on live ticker.

**User Model (U1)** panel -- Hidden until you right-click the chart to
draw it. See section 10.

**Monte Carlo Simulation** panel -- Opt-in (default off). When you
activate the checkbox, the bubble chart gains a "spaghetti fan" overlay
of N sample MC simulation paths -- thin RdYlGn-colored lines, one per
path, grading by terminal price (red = lowest finish, green = highest).
The number of sims is freely typeable (1-3200) with autocomplete
suggestions [8, 16, 32, 64, 128, 200, 400, 800, 1600, 3200]; ≤200 is
free-tier (cache holds 200 paths), >200 triggers the standard MC
paywall. Default is 8 sims so the fan stays visually tractable on
the busy bubble chart.

The **regime checklist** (Bargain / Cheap / Fair / Pricey / Bubble)
filters the displayed paths: untick a regime to drop paths that spend
more time in that regime than allowed. With `sims=1`, untick all but
one regime to see the single cached path most aligned with that
regime ("only Bargain" picks a stuck-low path, "only Bubble" picks a
ripped-up path). On free-tier cached paths the filter is rank-based
(no cached path NEVER visits any regime over 40 yr; the filter sorts
by alignment and trims to your sim count). Live (paid) sims apply the
filter strictly via the transition matrix.

**Plot Appearance** -- point size (1-20), point alpha (0.1-1.0), plus a
shared block of trace widths, grid visibility, palette override.

**Palette selector** at the bottom.

<!-- merged from v1: tips for Tab 1 quantile selection -->
**Tips:**
- Select a few quantiles that bracket your scenario (e.g., Q10%
  pessimistic, Q50% median, Q85% optimistic).
- The Auto Y checkbox automatically rescales the Y axis to fit your
  selected quantiles within the visible X range.
- Point size and alpha help when zooming into dense data regions.

### Tab 2 -- CAGR Heatmap (`/2`, `heatmap`)

A 2-D grid of compound annual growth rates: entry year + entry
percentile (rows -- actually, fixed because you pick one entry) x exit
year x exit quantile.

**Multi-model pill bar** at the top of the chart. Click a pill -> the
heatmap recomputes for that model. Pill set (as of 2026-04):
`BM, PL, LPPL, HybPPL, PCA, Grdy, EPPL, Gomp, BPL, PL+c, SExp, Logi`
(EF added if `model_data_ef.pkl` is present, U1 added if you drew one,
MC added as an orange "paid" pill if Markov is available).

**Entry Conditions panel** -- Entry year slider (2010-2039), entry
percentile **free numeric input** 0.1-99.9% (not a dropdown), starting
BTC for portfolio display. The entry percentile auto-syncs to the live
BTC price percentile when entry year = current year.

**Projection Quantiles** -- full ~17-quantile checklist (no band
abstraction here, since each quantile becomes an output column).

**Axes & Range** -- exit year range (X axis of the heatmap).

**Display toggles** -- Show colorbar, Enable chart zoom.

**Cell text** dropdown -- choose what each cell shows: CAGR %, Exit
Price, CAGR + Price, CAGR + Portfolio, Portfolio Value, Multiple (x),
CAGR + Multiple, Multiple + Portfolio, or None.

**Heatmap Colorscale panel** --
- **Color mode**: `Segmented` (uses Break 1 + Break 2 integer breakpoints
  to split low/mid/high), `Data-Scaled` (quantile-based), `Diverging`
  (centers at 0% CAGR).
- **Break 1, Break 2** -- integer CAGR %.
- **Color palette** -- `Red -> White -> Green`, `Red -> Black -> Green`,
  `Blue -> White -> Orange`, `Monochrome (gray)`. Auto-suggested for the
  active color palette.
- **Lo / Mid1 / Mid2 / Hi** color pickers for custom schemes.
- **Gradient steps** (2-64) -- cosmetic; actual rendering uses 256-point
  colorscale.

**Monte Carlo controls** panel -- see section 9.

<!-- merged from v1: heatmap reading hint -->
**Reading the heatmap:** Each cell shows the CAGR you'd achieve buying
at the entry point and selling at that exit year / quantile
intersection. Hot colors = high returns, cool colors = low or negative
returns.

### Tab 3 -- BTC Accumulator (DCA) (`/3`, `dca`)

A dollar-cost-averaging simulator. You buy `$X` every period over a
specified year range; the tab computes your BTC stack and USD value
across multiple price quantiles.

- **Purchase amount ($)** -- integer, `step=1`.
- **Frequency** -- Daily / Weekly / Monthly / Quarterly / Annually.
- **Year range** slider.
- **Inflation rate** (0-100% / yr). Optional; purchases scale if on.
- **Starting BTC stack** -- existing holdings at year 0.
- **Display mode** -- USD / BTC / both (dual-y axis; primary BTC stack,
  secondary USD value).
- Chart toggles: Annotate depletion, Log y, Shade bands.

**Stack-celerator ("Enter Saylor Mode")** -- opt-in loan-financed lump
sum overlay. Enable the checkbox to reveal:
- Loan amount ($)
- Entry price source: Live ticker / Model price / Custom price
- Loan type: Interest-only / Amortizing
- Roll over (refinance; no BTC sold between cycles)
- Annual interest rate (0-100% / yr)
- Loan term (months)
- Additional loan cycles (0 = single loan)
- Capital gains tax on repayment (0-99%)

Legend labels for BTC-mode traces include the final USD value in
parentheses.

**Display Models + Projection Quantiles + MC** -- shared with the rest.

<!-- merged from v1: Stack-celerator deep-dive -->
### Stack-celerator deep dive

The Stack-celerator models borrowing USD to front-load BTC purchases
with the loan repaid over time.

1. You borrow `principal` USD and buy BTC immediately at the entry
   price.
2. Your regular DCA amount is reduced by the loan payment each period.
3. At loan maturity (interest-only), you sell BTC to repay the
   principal.

The simulation shows whether the leveraged BTC purchase outperforms the
equivalent un-leveraged DCA -- the "Stack-celeration factor" in the chart
title.

**Amortizing vs interest-only.**
- **Amortizing**: each payment covers interest + principal. No BTC sale
  needed at maturity. Tax has no effect. Safer but higher periodic
  payments.
- **Interest-only**: payments cover only interest. At maturity, you must
  sell BTC to repay principal. Capital gains tax applies to the profit
  on the BTC sold (sell price minus cost basis). Higher risk, lower
  periodic payments.

**Rollover (interest-only only).**
Without rollover, each cycle independently buys BTC at start and sells
at end. With rollover, a new loan pays off the old loan (net zero BTC
movement) and you make a single final repayment at simulation end.
Avoids intermediate tax events and keeps more BTC in your stack.

**Loan cap.** If the loan payment would exceed your DCA amount, the
principal is automatically capped. The info panel notes when this
happens. The cap formulas:

- Amortizing: `max_principal = amount * (1 - (1+r)^-n) / r`
- Interest-only: `max_principal = amount / r`

**When it helps.** Stack-celerator tends to outperform plain DCA when
BTC appreciates significantly during the loan term, interest rates are
moderate relative to BTC's growth rate, or the entry price is
relatively low (lower percentile). It underperforms when BTC is flat
or declining -- you pay interest on borrowed money while your BTC isn't
growing.

### Tab 4 -- BTC RetireMentator (`/4`, `retire`)

Withdrawal simulator. You pull `$X` every period starting from a base
year; the tab tracks how long the BTC lasts under each quantile.

- **Withdrawal amount ($)** per period.
- **Frequency** (same five options).
- **Year range** -- default 2031-2075.
- **Inflation rate** -- withdrawals scale with inflation.
- **Starting BTC stack**.
- **Display mode** -- USD / BTC Remaining.

Depletion annotations mark the year each quantile's stack hits zero,
staggered at multiple heights to avoid overlap. MC default entry
percentile = 10% (conservative).

<!-- merged from v1: tips for Tab 4 -->
**Tips:**
- Select multiple quantiles to see the range of outcomes. Q1% is near
  worst-case, Q50% is median, Q85%+ is optimistic.
- Dual-Y axis shows both BTC stack and USD value simultaneously.

### Tab 5 -- HODL Supercharger (`/5`, `supercharge`)

A two-mode retirement optimizer with delay-offset trajectories to show
the value of waiting.

**Mode A -- Fixed spending (depletion date)** -- You pick `$X/period`;
the tab computes the depletion year for each quantile. Use case: "I
have X BTC, spending Y/yr -- when does it run out?"

**Mode B -- Fixed depletion (max spending)** -- You pick a target
depletion year; the tab computes the max sustainable withdrawal via
binary search, at each selected quantile. Use case: "I have X BTC,
want it to last until year Z -- what's the maximum I can spend?"

**Delay offsets** (5 inputs, years) -- a quirk of this tab. You pick
five different delays (default `[0, 0, 0, 1, 2]`) and the tab draws
five parallel trajectories on the chart, each representing "wait N
years before starting withdrawals." Exposes the opportunity cost (or
benefit) of delaying onset.

**Chart layout** -- a checkbox "Shade quantile bands."
- Checked (default): quantile fan bands visible, depletion year
  annotated per quantile. Display quantile dropdown hidden.
- Unchecked: discrete per-delay lines at the selected display
  quantile. Dropdown visible.

Default quantiles: `Q15%, Q85%` (mode A) or `Q10%` for MC overlay.

<!-- merged from v1: delay colors trivia -->
Delays are colored consistently: blue (0yr), red (1yr), green (2yr),
purple (3yr), orange (4yr). Duplicate delays are automatically
deduplicated.

### Tab 6 -- Citadel Planner (`/6`, `citadel`)

The biggest tab. A multi-asset simulation for long-horizon retirement
planning with Bitcoin as the growth engine plus cash, reserves,
investments, and optional real estate.

When you navigate to `/6` a WIP warning modal appears -- the engine
math is being stress-tested; treat output as illustrative.

**Quick Scenarios panel** (above the sub-tabs) -- preset Wealth levels
(Starter, Middle, Citadel), Macro Regimes (Neutral, Bull, Bear,
Stagflation...), and Rule sets (No rebal, Simple rebal, Aggressive). Pick
a start year 2025-2040. 800 simulations per scenario, delivered from a
pre-computed cache (free).

**Four sub-tabs:**

1. **Assets** --
   - Bitcoin Stack: starting BTC + optional "Use Stack Tracker lots"
     import.
   - Cash Account: initial balance + interest rate.
   - Reserve Fund -- US Treasuries with 3 maturity bins: Short (T-Bills
     <=1yr), Medium (T-Notes 2-10yr), Long (T-Bonds 10-30yr). Each bin
     has Initial $, Return %, Vol %.
   - Investment Account: Equities + Bonds, each with Value, cost
     Basis, Return, Vol.

2. **Spending** -- Monthly spending ($), Inflation rate, Spending
   growth above inflation.

3. **Rules** --
   - **High-Q Trigger (Take Profits)**: enable, threshold percentile,
     Gradual vs Lump mode, rate per action, duration, 6-way proceeds
     split (Cash/Short/Med/Long/Equities/Bonds, must sum to 100%).
   - **Low-Q Trigger (Accumulate BTC)**: same structure but
     accumulating instead of reducing.
   - **Global Lump Cooldown** -- minimum periods between lump actions.
   - **Account Floor Rules** -- minimum balances per account with
     optional annual floor growth.
   - **Saylor Citadel Fortifier** -- optional loan against BTC for
     liquidity. Term or perpetual, interest rate, repayment trigger.

4. **Simulation** -- Year range, Frequency (Monthly/Quarterly/Annually),
   BTC price model source, Dollar Asset Returns mode (Fixed Rates /
   Historical Regimes Markov), **Tax toggle** (section Tax below), BTC
   Price Scenario quantile picker, MC controls.

Below the sub-tabs (so always visible):

- **Display panel** -- Display mode (USD total / USD per asset / BTC),
  chart toggles, legend position, Show All / Hide All bulk-visibility
  buttons (since Citadel uses per-asset traces, not per-model).
- **Plot Appearance**, **Palette selector**.

**Run Simulation button** -- single `Run Simulation` button.
Deterministic projection runs as a background task so it doesn't block
gunicorn workers. Button shows `Computing...` during long runs.

**Save / Load Scenario** -- Download current config as JSON; upload a
previously saved JSON.

**Tax toggle** -- opens a modal with full US tax configuration:
- Master toggle (master off = engine runs with no tax logic)
- State dropdown (51 entries: 50 states + DC, showing top marginal
  rate)
- Filing status, birth year (1900-2099; determines RMD start age:
  73 for 1951-1959, 75 for 1960+)
- TCJA sunset toggle (39.6% top rate, lower standard deduction)
- **Tax-Deferred account** (Traditional IRA/401k) -- BTC, Cash, Reserve
  bins, Investment bins. All withdrawals taxed as ordinary income.
  Subject to RMDs.
- **Tax-Free account** (Roth) -- same asset structure, tax-free
  withdrawals, no RMDs.

When tax is on, the engine uses:
- IRS Sec. 1(h) capital loss netting with $3k deduction + carryforward.
- Progressive federal ordinary brackets (inflation-indexed from 2025).
- Federal LTCG brackets (0/15/20%) stacked above ordinary taxable
  income.
- NIIT 3.8% on min(NII, MAGI - threshold). Threshold not
  inflation-indexed.
- State flat top marginal rate on AGI minus treasury interest.
- **Dynamic cost-ranked withdrawal waterfall** -- each period, the
  engine scores every funding source by `tax_cost + PV-discounted
  opportunity_cost`, draws from the cheapest first, and re-ranks at
  bracket boundaries. Roth BTC is always last (tax-free compounding
  on the highest-growth asset).

A tax summary panel below the chart shows year-by-year tax owed, AGI,
LTCG, NIIT, and withdrawal sources.

<!-- merged from v1: Citadel tips -->
**Tips:**
- Use Fixed Rates for a quick baseline scenario, then switch to
  Historical Regimes to see how sequence-of-returns risk affects the
  outcome.
- Disable all rebalancing rules to model a pure "buy and hold" strategy,
  then re-enable them one by one to see which triggers matter most.
- The Citadel Planner uses the same BTC price model as the other tabs --
  your selected quantile for BTC projections feeds directly into the
  simulation.
- Compare tax ON vs OFF to see the true cost of taxes over a 40-year
  horizon. The drag is often 20-40% of terminal wealth -- a powerful
  motivator for Roth conversions and tax-loss harvesting.

### Tab 7 -- Stack Tracker (`/7`, `stack`)

Purely local lot management. Nothing goes to the server; everything is
stored in your browser's `localStorage`.

**Lot table** (left) -- one row per purchase: Date, BTC, Price $/BTC,
Total Paid, Percentile (of the model at purchase time), Notes. Select
rows via checkboxes in the leftmost column; pages of 20.

**Summary** -- total BTC, total cost basis, current $ value (at live
BTC price), % gain, unrealized gain.

**Add Lot card** (right) -- Date picker, BTC amount, Price ($/BTC),
Notes. `Add Lot` button. Live percentile preview updates as you type
the price.

**Delete selected / Clear all** buttons.

**Export / Import** -- Export JSON (downloaded to your machine); Import
JSON (upload a previously exported file).

**Restore my lots** -- when a share link overrides your lots, a banner
appears at the top with a "Restore my lots" button to revert to your
own data.

<!-- merged from v1: percentile interpretation -->
### Percentile interpretation

- **Low percentile** (e.g., 5%): you bought near the bottom of the
  historical range -- a "cheap" purchase relative to the model.
- **High percentile** (e.g., 90%): you bought near the top -- "expensive"
  relative to the model.
- **Median** (~50%): a "fair value" purchase.

### How to back up your lots

- **Export**: click Export JSON. Your browser downloads a file like
  `lots_2026-04-17.json`. Stash it anywhere -- cloud drive, USB, etc.
- **Import**: click Import JSON and select a previously exported file.
  This overwrites any existing lots in `localStorage`. Make sure you
  really want to replace first.
- When you check "Use Stack Tracker lots" on other tabs, your total BTC
  from all lots becomes the starting stack for that simulation.

### Tab 8 -- Model Info (`/8`, `model_info`)

An accordion of 26 named items -- one per price model plus a few
concept entries.

**Item list (order fixed for /8.N stability):**

1. Bubble Model
2. Quantile Regression
3. Power Law (OLS)
4. LPPL
5. LPPL_2
6. LPPL Weighting & Regime Shifts
7. LinPPL
8. HybPPL
9. HybPPL (DD)
10. HybPPL +2L
11. HybPPL +2C
12. HybPPL +2B
13. HybPPL 4D
14. PCA
15. Greedy Select
16. Entropy PPL
17. Exponential
18. Gompertz
19. Broken Power Law
20. S2F
21. Monte Carlo
22. Empirical Floor
23. User Model
24. Model Comparison
25. Historical Regimes
26. Citadel Planner

Deep-link any item via `/8.N`. Example: `/8.4` opens LPPL.

Each item typically contains: formula (LaTeX, rendered via MathJax on
clearnet), methodology, motivation, **live fitted coefficients** (pulled
from the running model instance -- updates the moment a refit ships,
except for S2F / MC / U1 / Compare / Regimes / Citadel which are
conceptual), and caveats. Images are clickable for an enlarged
lightbox view.

### Tab 9 -- FAQ (`/9`, `faq`)

A flat accordion of Q&A. Covers: what Share does, quantile regression
primer, "is this predicting the price" (no), the 2009-07-25 time
origin and why it matters, halving cycle, R-squared interpretation, and
more. Deep-link any item via `/9.N` or `/faq.N`.

Links in FAQ answers are styled in a blue accent color.

---

## 6. Sharing your view (Share button)

Click the `Share` button in the navbar to open the share modal.

- **Scope radio** -- choose `Current tab only -- shorter link` (default)
  or `All tabs -- full state, longer link`. Single-tab links only
  encode controls that belong to the active tab, via a 20-checkbox
  bitmask encoding for toggle fields.
- **Include Stack Tracker lots** -- optional checkbox. When on, your
  lots are compressed into the link.
- **Generate link** button creates the URL and displays it in a
  readonly input.
- **Clipboard icon** copies the URL.
- **QR code** appears for scanning to another device.
- **Preview thumbnail** shows the chart the recipient will land on.

**Link History** -- your last 50 generated links are saved to browser
`localStorage` with scope + tab metadata. Click any entry to re-copy.
Deduplicates automatically. `Clear history` empties it.

URL format: `host/N#q4:<base64>` where `N` is the tab path. The tab
routes independently of the hash decode so the correct tab opens even
before state is restored.

**Restore speed**: bubble-tab share links typically have the chart
visible in 3-4 seconds and the "Restoring..." modal cleared shortly
after. Other tab types (Citadel, etc.) take a little longer because
their figure builders are heavier and the helper that pre-builds the
figure during decode only handles the bubble tab today.

**Refreshing a shared URL** reverts to defaults (aside from the hash
content) -- this is the "refresh to start over" behavior. Use your
browser history / back button if you want to step back a state.

**Legacy links** (`q1:`, `q2:`, `q3:` prefix) are still decoded; new
links use `q4:`. The `q4:` format embeds a fingerprint of the defaults
that were live when the link was generated and stores only the fields
that differ -- when defaults change later, omitted fields fall back to
the original values, and the link still produces the chart you saw.

<!-- merged from v1: snapshot-lots flow -->
### Snapshot lots

If you have lots in Stack Tracker and tick "Include Stack Tracker lots,"
they're included in the snapshot. Recipients see your lots while
viewing the shared link. A "Restore my lots" button lets them revert
to their own `localStorage` lots.

---

## 7. Static analysis pages

A handful of model-research pages are served as static HTML/SVG at
short capital-letter URLs. These are not interactive tools -- they're
pre-generated reports you can share directly.

- `/B` -- BM sensitivity sweep (SVG).
- `/BB` -- Empirical Floor sensitivity sweep (SVG).
- `/D` -- Residual FFT power spectra -- frequency-domain view of model
  residuals with a calendar-year cycle conversion table (HTML + SVG).
- `/E` -- Historical Regime Shift analysis: multi-model rolling-window
  detection of regime changes (HTML index referencing ~25 SVGs +
  CSVs; legacy single-SVG fallback).
- `/F` -- LPPL family fits to BM-excess: oscillator-only fits against
  the detrended BM floor (HTML + SVGs).
- `/G` -- Non-sinusoidal wave basis comparison (SVG).
- `/H` -- Genesis-date sweep (SVG).

If a page says "Not generated yet" it means the upstream research
artifact hasn't been regenerated on this deployment.

<!-- merged from v1: detailed /E section -->
### /E -- LPPL regime shift detection (detailed)

A single HTML page with anchor navigation to four sections showing how
LPPL parameters evolve over rolling time windows:

1. **LPPL_1** (6 params) -- 5-year windows
2. **LPPL_2** (9 params) -- 5-year windows
3. **LPPL_3** (12 params) -- 7-year windows
4. **LPPL_3** (12 params) -- 9-year windows

Each section has one time-series panel per fitted parameter, plus
panels for residual sigma and R-squared. Vertical dashed lines mark
known Bitcoin regime events: 2013/2017/2021 bubble peaks, the March
2020 Covid crash, the November 2022 FTX collapse, and the January
2024 ETF approval.

**What to look for:**
- W (log-time frequency) saturating or jumping -> structural change
- D (damping) dropping to near-zero -> cycles stopped shrinking
- Residual sigma spikes -> model can't capture new dynamics
- W_2 flipping between ~9 and ~21 -> regime-dependent secondary
  oscillation

---

## 8. Deep links cheat sheet

| URL | Destination |
|---|---|
| `/1` | Tab 1 Price & Model Overlays (bubble) |
| `/1.2` | Tab 1 in Forward CAGR view |
| `/1.2.N` | Tab 1 CAGR with N-year horizon (1-6 -> 1,2,4,10,20,30 yr) |
| `/1.2.N.1` | Same, with "hover today" beacon auto-triggered |
| `/1.3` | Tab 1 in Residuals view |
| `/2` | Tab 2 Heatmap |
| `/2.N` | Tab 2 with Nth pill active (1-indexed) |
| `/3` | Tab 3 DCA |
| `/4` | Tab 4 RetireMentator |
| `/5` | Tab 5 Supercharger |
| `/6` | Tab 6 Citadel Planner |
| `/7` | Tab 7 Stack Tracker |
| `/8` | Tab 8 Model Info |
| `/8.N` | Model Info item N (1-indexed) |
| `/9` | Tab 9 FAQ |
| `/9.N` or `/faq.N` | FAQ item N (1-indexed) |
| `/mi` | Static Model Info page (single round-trip HTML) |
| `/mi.N` | Static Model Info with item N pre-opened |
| `/faq` | Static FAQ page |
| `/faq.N` | Static FAQ with item N pre-opened |
| `/docs/architecture` | Architecture guide (rendered markdown) |
| `/docs/user-manual` | This manual |
| host/N`#q4:...` | Tab N with snapshot state applied (current); `#q3:`, `#q2:`, `#q1:` legacy links still decode |

You can also replace `.` with `-` in deep links -- `/1-2-5-1` is
equivalent to `/1.2.5.1`. Useful in chat contexts where periods get
interpreted as sentence enders.

---

## 9. Monte Carlo simulation (paid)

Monte Carlo is Quantoshi's one truly compute-heavy feature. Running
thousands of stochastic Markov-chain price paths takes real server
cycles, so it's behind a small Lightning paywall. Expect 500-2000 sats
depending on the number of sims and horizon.

**Cached scenarios are free.** The server holds ~45,000 pre-computed
MC scenarios covering common parameter combinations (entry
percentiles in 10% steps, horizons 10/20/30/40 yr, a curated set of
withdrawal amounts, inflation values, stack sizes). When you request
MC with parameters that fall on the cache grid, the result is served
instantly -- no payment, no compute. MC dropdowns **bold** the
cache-aligned values to guide you toward free lookups.

**Custom MC is paid.** Anything off-grid runs a live 100-3200-sim
Markov simulation. The cost is displayed under the cost line before
you click "Run MC Simulation." Payment is via:

- **Lightning** (default; near-instant).
- **On-chain** (slower confirmation; still supported for high-fee
  simulations where the BTCPay backend decides to offer it).

Once paid, a server-side one-time nonce authorizes the compute.

**What you get**: fan bands of simulation paths overlaid on DCA,
Retire, Heatmap, Supercharger, and Citadel charts. Each quantile of
the MC fan is drawn as a translucent band. **Tab 1 (Bubble) renders a
"spaghetti fan"** instead -- N individual sample paths as thin
RdYlGn-colored lines, color-graded by terminal price -- since the
bubble chart already has its own analytical quantile bands and would
be over-cluttered by an additional MC fan. You can `Save` the MC
result as a JSON file and `Load` it later to restore the same view
without re-computing.

**Monte Carlo controls** (appear on Bubble, DCA, Retire, Heatmap,
Supercharger, Citadel tabs):

- **Activate** checkbox (top right, `NEW` badge)
- **MC start year** dropdown (bold = cached, e.g. 2031).
- **Entry percentile** dropdown (bold = cached, e.g. 10%).
- **Model source** -- which model's quantile bands generate the
  starting price distribution. Defaults to `bub`.
- **Amount / inflation / stack** (if applicable) -- often shared with
  the tab's main inputs.
- **Years to model** -- 10 / 20 / 30 / 40 (bold cached).
- **Advanced simulator options** checkbox reveals:
  - Markov transition matrix dimension (5x5 through 10x10).
  - Price regime filter (checklist of regime bins).
  - Simulations count -- typeable Input (1-3200) with HTML5 datalist
    autocomplete: 8, 16, 32, 64, 128, 200, 400, 800, 1600, 3200.
    ≤200 is free-tier (cache subsampling); >200 triggers paid live MC.
  - Frequency (if not shared).
  - Historical window (date range slider).
- **Cost line** -- shows computed cost in sats before run.
- **Run MC Simulation** button.
- **Save / Load** buttons for MC result persistence.

When your MC cost crosses 50,000 sats you see the **Quant Territory**
warning modal ("I see your model costs have left the realm of mere
mortals..."). Click "Proceed, I am Sir Baller" to continue.

<!-- merged from v1: regime-filter explanation + ghost overlay -->
### Regime filter (blocked bins)

You can remove price regimes from the simulation to model scenarios
like "what if we never see another extreme bubble?" or "what if prices
never drop to bargain levels again?" The **ghost overlay** shows the
unfiltered simulation as a faded comparison, so you can see how
blocking bins changes the outcome distribution.

**On Tab 1 (Bubble) free-tier cached paths the filter is rank-based**
rather than strict -- every cached path eventually visits every
regime over a 40-yr horizon, so a strict "drop paths that ever touched
a blocked bin" filter would drop everything. Instead, paths are sorted
by ascending time spent in blocked regimes and trimmed to your
simulation count, so you always see the N paths most aligned with
your regime preference. With `sims=1` and "only Bargain" selected, you
see the single cached path that spent the least time in
Cheap/Fair/Pricey/Bubble regimes (= the most stuck-low). Live (paid)
MC sims apply the filter strictly via the transition matrix.

### Interpreting fan bands

- **Median depletion year**: the year the typical simulation path hits
  zero BTC (for withdrawal tabs).
- **Wide fan bands**: high uncertainty -- outcomes vary widely.
- **Narrow fan bands**: more agreement across simulations -- higher
  confidence.
- **Fan tilting up**: most simulations show growth at that timeframe.
- **Fan tilting down**: most simulations show decline (withdrawal
  exceeds growth).

---

## 10. The User Model (U1)

Tab 1 includes a click-to-draw custom power law. Right-click the
bubble chart to open a context menu with "Set Point 1" and "Set Point
2" entries. Pick two year+price points; Quantoshi fits a power law
through both.

- Once both points are set, U1 auto-draws and is auto-selected in the
  Tab 1 Display Models checklist (U1 (User)).
- Empirical residual quantization produces a full quantile fan around
  your custom line.
- The U1 model appears in the Model Scanner, in the ticker percentile
  cycle (as U1 with an orange badge), and in the heatmap pill bar.
- The User Model is **session-only**. It disappears on page refresh
  and does not persist across tabs (stored only in a memory-typed
  `dcc.Store`).
- Useful for sanity-checking alternative time origins, "what if we
  start from 2012" experiments, or quick two-point regressions.

Clear the model via the `Clear` button in the User Model (U1)
panel, or just reload the page.

---

## 11. Welcome modal / first visit

On your first visit of a session, a splash modal appears with a
randomly picked quote from a rotating collection (Satoshi, Cypherpunk
writings, sound-money quotes, HODL wisdom, freedom/conviction). The
Genesis quote is always first; clicking "More quotes" cycles through
others.

- Logo + Quantoshi brand at top.
- Quote + attribution in italic.
- Journey stats (visits, days, etc.) shown after first visit.
- **Tor users** see an extra "Accept Knighthood" button for an
  onion-service easter egg. Clearnet visitors see the usual dismiss
  flow.
- `Let's go` button to dismiss.

The splash only re-appears after a dismissal gap; repeat visits within
the same day go straight to the app.

<!-- merged from v1: logo-click easter egg note -->
Click the logo 6 times from the splash modal to summon the Genesis Block
easter egg. Many quotes link to their original source (BitcoinTalk
posts, mailing list archives, tweets); click the quote text to follow
the source.

---

## 12. Understanding quantiles

<!-- merged from v1: quantile-semantics primer, plainer wording -->

### What "Q10%" means

Q10% means: "10% of historical trading days, Bitcoin's price was at or
below this line." It represents the 10th percentile of the historical
price distribution, projected forward.

### The quantile spectrum

| Quantile | Interpretation |
|---|---|
| Q0.1%-Q1% | Extreme pessimism -- near worst-case historical scenarios |
| Q5%-Q10% | Very pessimistic -- only 5-10% of history was this low |
| Q25% | Lower quartile -- below-median path |
| Q50% | Median -- half of history was above, half below |
| Q75% | Upper quartile -- moderately optimistic |
| Q85%-Q95% | Optimistic -- only 5-15% of history was this high |
| Q99%-Q99.9% | Extreme optimism -- near best-case historical scenarios |

### Important caveats

- Quantiles describe the **historical distribution**, not predictions.
  Future price behavior may not follow historical patterns.
- The power-law model assumes Bitcoin's growth continues on a similar
  trajectory. This is a modeling assumption, not a guarantee.
- Lower quantiles are useful for conservative planning (retirement,
  withdrawal budgets). Higher quantiles show what's possible but
  shouldn't be relied upon.

### Arbitrary percentiles

The heatmap's entry percentile accepts any value 0.1%-99.9%. Values
between fitted quantiles (e.g., Q7.5%) are interpolated in log-price
space between the two nearest fits.

---

## 13. FAQ-style troubleshooting

<!-- merged from v1: FAQ-style troubleshooting entries for first-time visitors -->

### Why is my chart empty?

- You probably unchecked every model in Display Models or every
  quantile in Projection Quantiles. Tick at least one of each.
- If you are on a share link, the hash might have restored the tab's
  controls to an uncommon setting. Refresh to fall back to defaults.

### The chart looks wrong on mobile

- Rotate to portrait. On very narrow screens columns stack vertically;
  the graph height is pinned to `55vw` so it doesn't leave a blank
  gap. If you still see layout drift, double-tap the active tab header
  to force a re-render.
- Plotly's toolbar can be hard to reach -- use the `PNG` export button
  below the chart instead.

### "Loading..." never finishes on Tab 6

- Citadel simulations run as background tasks. Large MC runs can take
  30-600 seconds. If the button stays `Computing...` for much longer,
  check the server health endpoint (`/health`) or reload; the app will
  serve from cache on the next attempt.

### The live price hasn't updated

- The ticker refreshes every 20 minutes, not continuously. Binance is
  the primary source; CoinGecko is the fallback. If both are failing,
  the ticker stops updating but the rest of the app still works.

### Share links don't work between old and new deploys

- Share links are versioned (`q1:`, `q2:`, `q3:`, `q4:`). Older links
  still decode but may fall back to defaults for controls that have
  been added or removed. `q4:` links additionally carry an 8-character
  defaults fingerprint -- if the server has shipped new defaults since
  your link was generated, the embedded fingerprint lets the decoder
  pick the right baseline so omitted fields restore as you saw them.
  If a link silently loses state, re-generate a fresh one.

---

## 14. Glossary

- **Bai-Perron test** -- a method for finding structural breakpoints in
  time-series data.
- **Box-Cox transformation** -- a family of power transformations
  parameterized by lambda; lambda=0 is log (power law), lambda=1 is
  linear (exponential).
- **Bubble Model (BM)** -- Quantoshi's default flagship model:
  support-line power law floor plus a quasi-periodic bubble amplitude
  term fitted to historical halving-cycle peaks. Median = support +
  mean bubble.
- **CAGR** -- Compound Annual Growth Rate. `(V_f / V_0)^{1/n} - 1` as a
  percentage. Tab 2 is a heatmap of CAGR across entry/exit
  combinations.
- **CUSUM** -- Cumulative Sum of residuals, a test for detecting regime
  changes in time series.
- **Depletion year** -- the year a withdrawal simulation's BTC stack
  reaches zero.
- **Durbin-Watson** -- a statistic measuring residual autocorrelation;
  2.0 = no autocorrelation, near 0 = extreme positive autocorrelation.
- **Entropy PPL (EPPL)** -- Entropy-damped log-periodic oscillator; a
  family variant where damping depends on information-theoretic
  entropy rather than a fixed exponent.
- **Empirical Floor (EF)** -- a support-line model fit to the lowest
  envelope of historical prices.
- **Entry percentile** -- where the current price sits on the quantile
  model (0-100%).
- **Fan band** -- shaded region between MC simulation percentiles
  showing uncertainty.
- **Forward CAGR** -- the compound annual growth rate from a given date
  looking N years into the future, based on model price projections.
- **Genesis block** -- Bitcoin's first block, mined January 3, 2009.
- **Halving** -- Bitcoin's supply-issuance rate halves every ~210,000
  blocks (~4 years). Visible as a calendar-frequency component in
  LinPPL and HybPPL fits (~3.56-3.59 year period).
- **HybPPL** -- Hybrid PPL. Combines log-time LPPL (multiple bubble
  frequencies) with a calendar-time oscillator (halving cycle).
  Quantoshi's strongest fit on residuals.
- **LinPPL** -- Linear PPL. Power law drift + calendar-time cosine.
- **LPPL** -- Log-Periodic Power Law. A family of models where
  log-space oscillations ride on top of a power-law trend. Quantoshi
  supports 1-4 simultaneous frequencies, weighted/unweighted fits,
  and a "no 1/3" constraint (excluding the omega~13 frequency).
- **Log-space** -- the natural working space for exponential-growth
  phenomena. `log10(price)` vs `log10(t)` turns a power law into a
  straight line, making visual intuition much sharper.
- **Markov chain / regime** -- a model where future state depends only
  on current state, not history. A discrete price-regime bin used by
  Quantoshi's Monte Carlo engine. The transition matrix (5x5 default)
  encodes the probability of moving between bins each period.
- **Monte Carlo** -- generating many random simulations to estimate
  probability distributions.
- **NIIT** -- Net Investment Income Tax. 3.8% federal surtax on
  investment income above a MAGI threshold ($200k single / $250k
  joint, not inflation-indexed).
- **Optimal time origin** -- July 25, 2009, the statistically optimal
  start date for the power law fit. All time calculations reference
  this date.
- **Percentile** -- the percentage of observations at or below a value
  (same as quantile x 100).
- **Power Law** -- `price(t) = a * t^b`. In log-space, a straight line.
  Quantoshi fits OLS and quantile regression variants.
- **Quantile / Percentile** -- the Qx% line or band shows where x% of
  the historical distribution falls. Q50% is the median; Q99% is the
  99th percentile.
- **Quantile regression** -- fitting a model to a specific percentile
  rather than the mean.
- **Recovery time** -- how long until a price level is seen again after
  a drawdown -- shown in hover on non-monotonic model traces and
  historical data.
- **Regime** -- a price bin (Bargain / Cheap / Fair / Pricey / Bubble)
  used in MC simulation.
- **Regime filter** -- blocking specific price regimes to model
  constrained scenarios.
- **RMD** -- Required Minimum Distribution. Mandatory annual
  withdrawal from tax-deferred accounts starting at age 73 (born
  1951-1959) or 75 (born 1960+, SECURE 2.0 rule).
- **S2F (Stock-to-Flow)** -- a popular but non-quantized BTC valuation
  model. Kept for reference; not a rigorous quantile-fit model.
- **Sats** -- Satoshis. 1 BTC = 100,000,000 sats. `sats/$` expresses
  dollar value per sat and *rises* as BTC falls, making it an
  everyday-useful number for stackers.
- **Stack** -- your total Bitcoin holdings (measured in BTC).
- **TCJA** -- Tax Cuts and Jobs Act (2017). Current US tax brackets
  run under TCJA rules through 2025. The TCJA-sunset toggle reverts
  to pre-TCJA rates (higher top rate, lower standard deduction).
- **Transition matrix** -- grid of probabilities for moving between
  price regimes.
- **U1 (User Model)** -- your personally-drawn power law. See section
  10.

---

If you find a bug, have a feature request, or want to ask a model
question that isn't covered here, the FAQ tab (`/9`) is the best
starting point. For technical detail on the models themselves, jump
into Model Info (`/8`) -- every item has a formula, methodology, and
live coefficient table.
