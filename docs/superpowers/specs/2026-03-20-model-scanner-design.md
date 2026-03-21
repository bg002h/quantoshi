# Model Scanner — Design Spec

## Summary

Add a "Model Scanner" panel to the Bubble tab (tab 1) controls and register
Quantile Regression as a standalone PriceModel. The scanner lets users input
any two of {price, date, quantile} and computes the third across all registered
models.

## Components

### 1. QuantileRegressionModel

Register the raw QR fits as a proper PriceModel:

```python
class QuantileRegressionModel(_FitsBasedModel):
    name = "Quantile Regression"
    short_name = "qr"
    dash_style = "solid"
    def __init__(self, md):
        self.fits = md.qr_fits
        self.colors = dict(md.qr_colors)
        self.quantiles = sorted(md.qr_fits.keys())
```

This is what `BubbleModel` used to be before the shrinking Gaussian conversion.
Straight lines in log-log, model-free, purely empirical. Once registered, it
appears automatically in Display Models toggles on all tabs.

Side benefit: ticker, lots, and MC code can use `PRICE_MODELS["qr"]` instead
of special-casing `M.qr_fits`.

### 2. Model Scanner Panel

Collapsible section in tab 1's control panel. Three input fields:

- **Price ($)** — `dbc.Input(type="number")`, default: live ticker price
- **Date** — `dbc.Input(type="date")` or `dcc.DatePickerSingle`, default: today
- **Quantile (%)** — `dbc.Input(type="number", min=0.1, max=99.9)`, default: computed

#### Interaction model

The two most recently edited fields are inputs; the third is the output.

- **Initial load**: price = live ticker, date = today → quantile computed (output)
- **User edits price**: date stays, quantile recalculates (output)
- **User edits quantile**: date stays (most recent non-quantile), price recalculates (output)
- **User edits date**: the other most-recently-edited field stays, third recalculates

A `dcc.Store` tracks which field is the current output. The output field gets
a distinct visual style (highlighted border/background, `disabled=True`).

#### Results table

Below the three inputs, a table with one row per registered model:

| Model | {computed variable} |
|-------|---------------------|
| Quantile Regression | value |
| Bubble Model | value |
| Power Law | value |
| LPPL | value |
| BM Empirical Floor | value |

The table header changes based on which variable is being computed:
- Solving for quantile → header is "Quantile"
- Solving for price → header is "Price"
- Solving for date → header is "Date"

The table auto-discovers models by iterating `_app_ctx.PRICE_MODELS`.

#### Computation

Three cases, all using `model.price_at(q, t)`:

- **Solve for quantile**: `model.find_percentile(t, price)` — direct call
- **Solve for price**: `model.price_at(q, t)` — direct call
- **Solve for date**: root-find `t` where `model.price_at(q, t) = price`.
  Use `scipy.optimize.brentq` on `log10(model.price_at(q, t)) - log10(price)`
  over the range `[0.5, 72]` (years since genesis). If no root exists (price
  below model's minimum or above maximum in range), show "—".

#### Live ticker sync

When price = live ticker and date = today (the default state), both fields
update every 20 minutes with the ticker callback. A "₿ live" hint appears
below the price field. Once the user manually edits either field, the hint
disappears and auto-update stops for that field. Clearing the field (or a
"reset" link) restores the live default.

### 3. Tab placement

The Model Scanner panel goes in tab 1's control column as a collapsible
`_section_card("Model Scanner", ...)`, placed after the Projection Quantiles
panel (`_q_panel`) and before the Data Point Appearance section. This is the
natural position: the user selects which quantiles to display, then the
scanner tells them what those quantiles mean in terms of price/date across
models. No new tab needed — no URL renumbering.

### 4. Interaction with Projection Quantiles panel

The scanner's quantile input and the Projection Quantiles checklist are
**independent** — toggling quantile lines on the chart does not change the
scanner, and editing the scanner does not change which lines are displayed.

**Convenience feature**: clicking a model row in the scanner results table
toggles that model's overlay on the bubble chart (adds/removes it from the
`bub-model-show` checklist). This connects scanner → chart without the
noisy reverse direction.

### 5. Snapshot/Share

Add the three scanner inputs to `_SNAPSHOT_CONTROLS` so share links preserve
the scanner state. Add to `_TAB_CONTROLS["bubble"]`.

## Out of scope

- Changing other tabs to use the scanner
- MC overlay for the QR model
- Auto-populating scanner quantile from quantile panel clicks
- Chart visualization of scanner results (just the table for now)
