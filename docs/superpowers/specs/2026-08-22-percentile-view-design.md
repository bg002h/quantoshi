# Percentile-vs-Time View (Tab 1) — Design

**Goal:** Add a Tab-1 chart mode that plots, over time, the percentile at which
BTC's actual price sat within a model's quantile fan — a model-relative
valuation / mean-reversion oscillator.

**Architecture:** A 4th pill in the existing `bub-view-mode` group
(Price / Forward CAGR / Residuals → **Percentile**), rendering a new panel that
reuses `find_percentile()` and the site's colorblind-aware color system.

**Tech stack:** Dash/Plotly, existing `btc_core` model protocol
(`find_percentile`), existing `bub-view-mode` panel-swap machinery, `colors.py`
palette system.

---

## Motivation

Each quantized model defines, at every time `t`, a distribution of prices (its
quantile fan). The percentile of the *actual* price within that fan is a
dimensionless "how rich/cheap is BTC vs this model" signal in [0, 100%]. Plotted
over time it becomes an oscillator (cousin of MVRV-Z / "risk metric" charts) that
shows mean-reversion *within* a model's own fan — information the price+fan view
conveys only implicitly. The navbar ticker already surfaces this for *today*;
this feature shows its full history.

## UI / placement

- A **4th pill — "Percentile"** — joins the existing `bub-view-mode` button
  group (`bub-view-price` / `bub-view-cagr` / `bub-view-resid`), with a new value
  `"percentile"` on the `bub-view-mode` store. Selecting it swaps the Tab-1 chart
  panel in place, exactly like Forward CAGR and Residuals already do.
- The lines drawn are the **quantized** models currently ticked in **Display
  Models** (`bub-model-show`). Non-quantized models (only `s2f`, `s2f_inst`) are
  silently skipped — they have no fan, so `find_percentile` is undefined for them.
- One line per model, in that model's palette color; legend by model name.

## The chart (percentile panel)

- **Y-axis:** 0–100% percentile, fixed (linear). No dollar axis.
- **X-axis:** the existing axes/range slider (`bub-xrange`) window, **historical
  only** — lines run through today's last actual price. No future (percentile
  needs a realized price).
- **No price data:** the panel shows *only* percentile lines + zones + reference
  marks. No price scatter, no model fans/composite, no dollar y-axis. (Structural,
  like the CAGR and Residuals panels, which already drop the price scatter.)
- **Colorblind-safe shaded valuation zones** behind the lines: a cool hue (blue/
  teal) "cheap" band at the bottom (< 25%), an amber/warm "rich" band at the top
  (> 75%), neutral middle. **No red/green.** Colors come from the existing
  CB-aware color system (same infrastructure as the heatmap diverging presets), so
  they read by hue *and* luminance and adapt per active palette. Thresholds fixed
  at 25 / 75 for v1.
- **Edge behavior:** when the actual price is outside a model's entire fan, its
  line pins at 100% / 0% (honest "off the top/bottom of the fan"; BM does this at
  extremes since its fan is Q0.1–Q99.9). This is intended, not clamped-away.
- **Consistency:** the last point of each line equals the navbar ticker's current
  percentile for that model.

## Data & computation

- Reuses `mdl.find_percentile(t, price)` (already battle-tested in the ticker;
  returns a fraction in [0, 1], log-space interpolation across the fan).
- Series = `find_percentile(price_years[i], price_prices[i])` for each historical
  row `i`, per selected quantized model. ~6k rows × a handful of selected models
  is cheap (each call is an O(n_quantiles) interpolation) and the result is a
  cached figure variant.
- **Caching:** the percentile figure is a new variant of the Tab-1 figure, keyed
  by `bub-view-mode` (already part of the params) plus the model selection +
  x-range that already key the price figure. No new cache fingerprint concerns.

## Colors

- Model lines: `_get_model_color(key, palette)` (existing).
- Zones: reuse the CB-aware diverging color infrastructure in `colors.py` /
  heatmap presets so "cheap" (cool) vs "rich" (warm) is deuteranomaly-safe and
  palette-aware. **CB palettes must not change** — this only *consumes* existing
  palette colors, adds none to the protected CB palettes.

## Architecture / touch points

| Area | Change |
|------|--------|
| `layout/common.py` | Add `bub-view-pctile` button to the pill group; add a `bub-pctile-wrap` panel (a `dcc.Graph`, hidden by default). |
| `callbacks/charts/__init__.py` | View-mode toggle callback: add the 4th button `Input`, the `bub-pctile-wrap` style `Output`, and the 4th button-outline `Output`; handle `"percentile"` in the show/hide + x-range logic. |
| `figures/` | New `build_percentile_figure(m, p)` (new `figures/percentile.py`) — computes per-model percentile series via `find_percentile`, draws lines + CB-safe zones + reference marks on a 0–100% axis. |
| `figures/common.py` / `colors.py` | Helper for the CB-safe zone colors per palette (reuse diverging-preset infra). |
| chart callback params | Thread `bub-view-mode` (already an input/state) so the builder picks the percentile branch; quantized-only model filter. |
| `snapshot.py` / `snapshot_defaults.py` | **No change** — `bub-view-mode` is already snapshotted (default `'price'`); `'percentile'` is just a new value. Old links keep working; default (hence fingerprint) unchanged. |
| tests | `test_figures` percentile builder (lines present, 0–100% range, non-quantized skipped, zones present); `test_callbacks` view-mode toggle includes percentile; snapshot round-trip of `bub-view-mode='percentile'`. |

## Interaction with other Tab-1 controls in percentile mode

Follows the established per-mode pattern (Residuals/CAGR already do this):
- **Active:** Display Models (drives lines), axes/range slider (x window), palette.
- **Inert / hidden:** quantile selection, Y scale/range, bubble composite, N
  future bubbles, point size/alpha, show-data/today/shade, MC — all price-specific.

## Out of scope (v1)

- Configurable zone thresholds (fixed 25/75).
- Overlaying non-quantized models (no fan → undefined).
- Projecting percentile into the future (needs a realized price).
- A dedicated per-model pill selector (uses Display Models instead).

## Risks / open questions

- **Edge pinning readability:** several models pinning at 100%/0% during extreme
  periods can overlap. Acceptable for v1 (honest); revisit if it reads poorly.
- **Zone vs line contrast:** shaded zones must stay subtle enough that the
  model-colored lines remain legible across all 4 palettes — verify per palette.
- **Legend with many models:** if the user ticks many quantized models the legend
  could crowd; same behavior as the price view's overlay legend, so no new
  problem.

## Testing approach

- Unit: percentile series is in [0, 1], monotonic mapping sanity, non-quantized
  models excluded, last point == `find_percentile(today)`.
- Figure: percentile panel has N lines for N ticked quantized models, 0–100%
  y-axis, zone shapes present, no price/dollar traces.
- Callback: clicking the Percentile pill sets `bub-view-mode='percentile'`, shows
  `bub-pctile-wrap`, hides the others, outlines sync.
- Snapshot: `bub-view-mode='percentile'` round-trips; old links (no percentile)
  still restore.
