# Quantoshi User Manual

A comprehensive guide to using Quantoshi for Bitcoin price analysis, accumulation
planning, and retirement modeling.

---

## 1. Getting Started

### What Quantoshi shows

Quantoshi projects Bitcoin's future price using **quantile regression** — a
statistical method that fits power-law curves to Bitcoin's entire price history
at different confidence levels. These aren't predictions; they describe the
historical distribution of prices and where the current price sits within it.

### How to read the charts

- **Log scale** (default): Equal vertical distances represent equal percentage
  changes. A move from $1,000 to $10,000 looks the same as $10,000 to $100,000.
  This is the natural scale for Bitcoin's exponential growth.
- **Linear scale**: Standard number line. Useful for seeing absolute dollar
  differences but makes early history invisible.
- **Colored lines**: Each line is a quantile (percentile channel). Lower
  quantiles are pessimistic paths, higher quantiles are optimistic paths.
- **Shaded regions**: The filled area between adjacent quantile lines shows
  the range of likely outcomes at that confidence level.

### Privacy

All your data stays in your browser's `localStorage`. Quantoshi stores nothing
server-side — no accounts, no cookies, no tracking. Your lot data, settings, and
link history never leave your device. Export/import uses local file downloads.

Server logs are retained for 27 days and contain only standard web server
request data (IP, timestamp, URL path). Tor users can access the `.onion`
address for additional privacy.

---

## 2. Tab-by-Tab Guide

### Tab 1: Bubble + QR Overlay

The main chart showing Bitcoin's price history overlaid with quantile regression
channels and bubble model projections.

**Key controls:**
- **Quantiles**: Check boxes to show/hide individual price channels. Each
  quantile (e.g., Q10%, Q50%, Q85%) represents a percentile of the historical
  price distribution.
- **X/Y Scale**: Toggle between Log and Linear for each axis. Log-Log is the
  default and shows the power law as straight lines.
- **N Future Bubbles**: How many future bubble cycles to extrapolate (1–10).
  More bubbles extend the projection further.
- **Shade**: Fill the area between adjacent selected quantiles.
- **Stack (BTC)**: Enter your BTC holdings to see projected USD value at each
  quantile on the right edge of the chart.
- **Use Stack Tracker lots**: Pull your BTC amount from the Stack Tracker tab
  instead of entering manually.

**Tips:**
- Select a few quantiles that bracket your scenario (e.g., Q10% pessimistic,
  Q50% median, Q85% optimistic).
- The Auto Y checkbox automatically rescales the Y axis to fit your selected
  quantiles within the visible X range.
- Point size and alpha controls help when zooming into dense data regions.

### Tab 2: CAGR Heatmap

A color-coded grid showing the Compound Annual Growth Rate (CAGR) for every
combination of entry year/percentile and exit year.

**Key controls:**
- **Entry Year**: When you (hypothetically) buy. Current year uses the live
  ticker price; historical years use the model price at your entry percentile.
- **Entry Percentile**: Where on the quantile spectrum you enter. The live
  percentile updates every 20 minutes based on the current BTC price.
- **Exit Years**: Range of years for the right side of the grid.
- **Exit Quantiles**: Which percentile lines to include as exit scenarios.
- **Color Mode**: Segmented (discrete color bands), Data-Scaled (continuous
  gradient), or Diverging (centered on 0% CAGR — green for gains, red for
  losses).

**Reading the heatmap:** Each cell shows the CAGR you'd achieve buying at the
entry point and selling at that exit year/quantile intersection. Hot colors =
high returns, cool colors = low/negative returns.

### Tab 3: BTC Accumulator (DCA)

Simulates dollar-cost averaging — buying a fixed USD amount of Bitcoin on a
regular schedule.

**Key controls:**
- **Amount**: USD per purchase (e.g., $100).
- **Frequency**: Daily, Weekly, Monthly, Quarterly, or Annually.
- **Year Range**: Start and end years for the simulation.
- **Display Mode**: BTC (accumulated stack) or USD (portfolio value).
- **Quantiles**: Select which price paths to simulate along.
- **Starting Stack**: BTC you already own before DCA begins.

**Stack-celerator** ("Enter Saylor Mode"): An advanced feature that simulates
borrowing USD to buy BTC upfront, then reducing your DCA by the loan payment.

- **Loan type**: Amortizing (principal + interest payments) or Interest-only
  (pay interest monthly, repay principal at end).
- **Rate**: Annual interest rate.
- **Term**: Loan duration in months.
- **Repeats**: Number of additional loan cycles (0 = one loan only).
- **Rollover** (interest-only only): Instead of selling BTC to repay at cycle
  end, the new loan pays off the old one. Single repayment at simulation end.
- **Tax rate**: Capital gains tax on BTC sold to repay (interest-only only).
  Only applies to the gain (sell price minus cost basis), not full proceeds.
- **Loan cap**: If your loan payment would exceed your DCA amount, the principal
  is automatically capped so the payment fits within your DCA budget.

### Tab 4: BTC RetireMentator

Simulates retirement withdrawals from a BTC stack over time, accounting for
inflation.

**Key controls:**
- **Withdrawal**: USD amount per period.
- **Frequency**: How often you withdraw.
- **Inflation Rate**: Annual inflation applied to the withdrawal amount.
- **Year Range**: When you start and stop (or when the simulation ends).
- **Display Mode**: BTC (remaining stack) or USD (portfolio value).
- **Annotate**: Show depletion year markers — when each quantile path hits zero.

**Tips:**
- Select multiple quantiles to see the range of outcomes. Q1% is near
  worst-case, Q50% is median, Q85%+ is optimistic.
- Dual-Y axis shows both BTC stack and USD value simultaneously.
- Depletion annotations mark the year each path's stack reaches zero.

### Tab 5: HODL Supercharger

Advanced withdrawal modeling with delay scenarios — "what if I wait N years
before starting withdrawals?"

**Mode A** (Depletion Date): "I have X BTC, spending Y/yr — when does it run
out?" Shows depletion curves for different delay offsets (0, 1, 2, ... years of
waiting before starting withdrawals).

**Mode B** (Max Spending): "I have X BTC, want it to last until year Z — what's
the maximum I can spend?" Uses binary search to find the highest sustainable
withdrawal rate for each quantile.

**Key controls:**
- **Stack**: Your BTC holdings.
- **Delays**: Up to 5 delay offsets (years before starting withdrawals).
- **Chart Layout**: "Shade" toggles between single-line per delay (off) and
  quantile bands per delay (on).
- **Display Q**: Which quantile to show when bands are off.

### Tab 6: Stack Tracker

A simple BTC lot tracker. Add your purchases with price, date, and BTC amount.

**Key controls:**
- **Add Lot**: Enter purchase price (USD), date, and BTC amount.
- **Delete**: Remove individual lots.
- **Export**: Download your lots as a JSON file (browser download).
- **Import**: Upload a previously exported JSON file.
- **Lot Percentile**: Each lot shows where you bought relative to the quantile
  model — a low percentile means you bought "cheap" historically.

**Privacy**: Lot data lives exclusively in your browser's `localStorage`. It is
never sent to the server. The Export feature creates a local file download.

**Cross-tab usage**: When you check "Use Stack Tracker lots" in other tabs, your
total BTC from all lots becomes the starting stack for that simulation.

### Tab 7: FAQ

14 entries covering common questions. Directly linkable via URL paths like
`/7.3` (opens the 3rd FAQ item).

---

## 3. Understanding Quantiles

### What "Q10%" means

Q10% means: "10% of historical trading days, Bitcoin's price was at or below
this line." It represents the 10th percentile of the historical price
distribution, projected forward.

### The quantile spectrum

| Quantile | Interpretation |
|----------|---------------|
| Q0.1%–Q1% | Extreme pessimism — near worst-case historical scenarios |
| Q5%–Q10% | Very pessimistic — only 5–10% of history was this low |
| Q25% | Lower quartile — below-median path |
| Q50% | Median — half of history was above, half below |
| Q75% | Upper quartile — moderately optimistic |
| Q85%–Q95% | Optimistic — only 5–15% of history was this high |
| Q99%–Q99.9% | Extreme optimism — near best-case historical scenarios |

### Important caveats

- Quantiles describe the **historical distribution**, not predictions. Future
  price behavior may not follow historical patterns.
- The power-law model assumes Bitcoin's growth continues on a similar trajectory.
  This is a modeling assumption, not a guarantee.
- Lower quantiles are useful for conservative planning (retirement, withdrawal
  budgets). Higher quantiles show what's possible but shouldn't be relied upon.

### Arbitrary percentiles

The heatmap's entry percentile accepts any value 0.1%–99.9%. Values between
fitted quantiles (e.g., Q7.5%) are interpolated in log-price space between the
two nearest fits.

---

## 4. Monte Carlo Simulations

### What MC does

Monte Carlo simulation generates thousands of possible future Bitcoin price
paths using a Markov chain trained on historical price transitions. Instead of
following a single quantile line, MC shows the range of outcomes when future
prices follow the same transition patterns as the past.

### How it works

1. Bitcoin's price history is divided into bins (regimes): Bargain, Cheap, Fair,
   Pricey, Bubble.
2. A transition matrix records how often price moved between bins historically.
3. The simulator starts at your chosen entry percentile and randomly walks
   forward using the transition probabilities.
4. 100–800 simulations produce a distribution of outcomes.

### Fan bands

The colored fan shape shows percentiles across all simulated paths:
- **P5%–P95%**: Light outer band — 90% of simulations fall here
- **P25%–P75%**: Medium inner band — 50% of simulations fall here
- **P50%**: Median line — the "typical" outcome across simulations

### Regime filter (blocked bins)

You can remove price regimes from the simulation to model scenarios like "what
if we never see another extreme bubble?" or "what if prices never drop to
bargain levels again?"

The **ghost overlay** shows the unfiltered simulation as a faded comparison,
so you can see how blocking bins changes the outcome distribution.

### Free tier vs paid

| Feature | Free | Paid (Lightning) |
|---------|------|-------------------|
| Simulations | 100 | 800 |
| Start years | 2028, 2031 | All cached years |
| Entry percentile | 10% | Any |
| Duration | 10 or 20 years | 10–40 years |

MC simulations appear as an overlay on the Heatmap, DCA, Retirement, and
Supercharger tabs.

### Interpreting results

- **Median depletion year**: The year the typical simulation path hits zero BTC
  (for withdrawal tabs).
- **Wide fan bands**: High uncertainty — outcomes vary widely.
- **Narrow fan bands**: More agreement across simulations — higher confidence.
- **Fan tilting up**: Most simulations show growth at that timeframe.
- **Fan tilting down**: Most simulations show decline (withdrawal exceeds growth).

---

## 5. Stack-celerator Deep Dive

The Stack-celerator is Quantoshi's leverage simulation for the DCA tab. It models
borrowing USD to front-load BTC purchases.

### How it works

1. You borrow `principal` USD and buy BTC immediately at the entry price.
2. Your regular DCA amount is reduced by the loan payment each period.
3. At loan maturity (interest-only), you sell BTC to repay the principal.

The simulation shows whether the leveraged BTC purchase outperforms the
equivalent un-leveraged DCA — the "Stack-celeration factor" in the chart title.

### Amortizing vs interest-only

- **Amortizing**: Each payment covers interest + principal. No BTC sale needed at
  maturity. Tax has no effect. Safer but higher periodic payments.
- **Interest-only**: Payments cover only interest. At maturity, you must sell BTC
  to repay principal. Capital gains tax applies to the profit on the BTC sold
  (sell price minus cost basis). Higher risk, lower periodic payments.

### Rollover (interest-only only)

Without rollover: Each cycle independently buys BTC at start and sells at end.
With rollover: New loan pays off old loan (net zero BTC movement). Single final
repayment at simulation end. This avoids intermediate tax events and keeps more
BTC in your stack.

### Loan cap

If the loan payment would exceed your DCA amount, the principal is automatically
capped. The info panel notes when this happens. The cap formula:

- **Amortizing**: `max_principal = amount * (1 - (1+r)^-n) / r`
- **Interest-only**: `max_principal = amount / r`

This ensures `payment <= DCA amount` at all times.

### When it helps

Stack-celerator tends to outperform plain DCA when:
- BTC appreciates significantly during the loan term (the front-loaded purchase
  captures more upside)
- Interest rates are moderate relative to BTC's growth rate
- The entry price is relatively low (lower percentile)

It underperforms when BTC is flat or declining — you're paying interest on
borrowed money while your BTC isn't growing.

---

## 6. HODL Supercharger Details

### Mode A: Depletion date

You specify: stack size, withdrawal amount, frequency, inflation rate, start
year, and up to 5 delay offsets.

The chart shows when each delay scenario's stack hits zero. Delays let you
compare "start withdrawing now" vs "wait 1 year" vs "wait 3 years" — waiting
often dramatically extends the stack's lifetime because BTC may appreciate
during the delay.

### Mode B: Max spending

You specify: stack size, target end year, frequency, inflation rate, and start
year. The simulator binary-searches for the maximum withdrawal amount that
doesn't deplete your stack before the target year, at each selected quantile.

### Chart layouts

- **Single-line** ("shade" off): One line per delay, colored by delay. Select a
  specific quantile to display via the Display Q dropdown.
- **Quantile bands** ("shade" on): Shaded bands between quantile pairs per
  delay. Shows the full uncertainty range but can be busy with many delays.

### Delay colors

Delays are colored consistently: blue (0yr), red (1yr), green (2yr), purple
(3yr), orange (4yr). Duplicate delays are automatically deduplicated.

---

## 7. Sharing & Snapshots

### How sharing works

1. Click the camera button in the navbar.
2. Choose scope: **Current tab only** (shorter URL) or **All tabs** (full
   cross-tab fidelity).
3. Click **Generate link**. The URL encodes all your control states.
4. Copy and share the URL. Anyone opening it sees your exact configuration.

### What's encoded

The URL hash contains a compressed representation of all UI control values. For
single-tab shares, only that tab's controls are encoded; other tabs use defaults.

### Snapshot lots

If you have lots in Stack Tracker, they're included in the snapshot. Recipients
see your lots while viewing the shared link. A "Restore my lots" button lets
them revert to their own `localStorage` lots.

### Link history

Your last 50 generated share links are stored in `localStorage`. Each entry
records the scope (all tabs / single tab) and which tab was active.

---

## 8. Stack Tracker Usage

### Adding lots

Enter the purchase price (USD), date, and BTC amount for each buy. The lot
appears in the table with its calculated percentile — where that price fell on
the quantile model at that date.

### Percentile interpretation

- **Low percentile** (e.g., 5%): You bought near the bottom of the historical
  range — a "cheap" purchase relative to the model.
- **High percentile** (e.g., 90%): You bought near the top — "expensive"
  relative to the model.
- **Median** (~50%): A "fair value" purchase.

### Export / import

- **Export**: Downloads a JSON file to your device containing all your lots.
- **Import**: Upload a previously exported JSON file to restore your lots.
  This overwrites any existing lots in `localStorage`.

### Cross-tab integration

When "Use Stack Tracker lots" is checked in the Bubble, DCA, Retirement, or
Supercharger tabs, your total BTC from all lots becomes the starting stack for
that simulation. Individual lot prices and dates are used for weighted-average
entry price calculations.

---

## 9. Glossary

| Term | Definition |
|------|-----------|
| **CAGR** | Compound Annual Growth Rate — annualized return between two dates |
| **Depletion year** | The year a withdrawal simulation's BTC stack reaches zero |
| **Entry percentile** | Where the current price sits on the quantile model (0–100%) |
| **Fan band** | Shaded region between MC simulation percentiles showing uncertainty |
| **Genesis block** | Bitcoin's first block, mined January 3, 2009. All time calculations reference this date |
| **Markov chain** | A model where future state depends only on current state, not history |
| **Monte Carlo** | Generating many random simulations to estimate probability distributions |
| **Percentile** | The percentage of observations at or below a value (same as quantile × 100) |
| **Power law** | A relationship where one quantity varies as a power of another: y = ax^b |
| **Quantile** | A cut point dividing a probability distribution (0.10 = 10th percentile) |
| **Quantile regression** | Fitting a model to a specific percentile rather than the mean |
| **Regime** | A price bin (Bargain/Cheap/Fair/Pricey/Bubble) used in MC simulation |
| **Regime filter** | Blocking specific price regimes to model constrained scenarios |
| **Stack** | Your total Bitcoin holdings (measured in BTC) |
| **Transition matrix** | Grid of probabilities for moving between price regimes |
