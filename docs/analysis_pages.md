# Analysis Pages

Quantoshi serves six custom analysis pages outside the normal Dash tab
system. Each is a static SVG generated offline by a Python script and
served via a Flask route in [`btc_web/api.py`](../btc_web/api.py).

## Page index

| URL | Topic | Generator |
|-----|-------|-----------|
| [`/A`](https://quantoshi.xyz/A) | Color palette builder | `btc_web/assets/palette_picker.html` |
| [`/B`](https://quantoshi.xyz/B) | BTC daily-close percentile vs BM (flat σ) | `tools/plot_bm_percentile.py --flat` |
| [`/BB`](https://quantoshi.xyz/BB) | EF support line sensitivity sweep | `tools/build_sensitivity.py` |
| [`/C`](https://quantoshi.xyz/C) | BM percentile × quantile regression sweep | `tools/build_sensitivity_pq.py` |
| [`/D`](https://quantoshi.xyz/D) | Residual FFT spectrum | `tools/residual_fft.py` |
| [`/E`](https://quantoshi.xyz/E) | Rolling-window LPPL regime detection | `tools/regime_shift.py` |

## What each page shows

### /A — Palette picker

Interactive HTML/JS tool for designing colorblind-safe color palettes.
Drag-and-drop color swatches onto three palette slots, visualized with:

- CIELAB a* × b* colorspace
- Machado 2009 deuteranomaly simulation
- HSL hue wheel
- ΔE (full vision) and ΔE'' (simulated deuteranomaly) distance matrices

No generator script — the HTML file is self-contained.

### /B, /BB — Support line sensitivity

5-panel heatmaps showing how Bubble Model parameters vary as the support
line's slope (x-axis) and intercept (y-axis) change. Grid is 50×50 across
slope [4.0, 7.0] and intercept ±2.5 around the fitted value.

**Panels**: R², predicted next bubble onset year, mean bubble amplitude K,
mean peak-to-peak interval, mean rise-to-rise interval.

**/B** marks the BM reference point (goldenrod) with EF as secondary (blue).
**/BB** flips these — EF as primary, BM as secondary.

**Key finding**: a diagonal R² ridge separates a broad stable region
(below/left) from a chaotic unstable zone (above/right). Both BM and EF
fits sit in the stable zone — moderate perturbations produce smooth,
predictable changes in bubble parameters.

### /C — Percentile × quantile regression sweep

7×7 grid (49 points) varying the two support-line fitting parameters:
percentile filter (X: 5% to 35%) and quantile regression target (Y: 5%
to 95%). Full bubble model pipeline runs at each grid point.

**Panels**: predicted next bubble onset, composite R², support slope, mean
major bubble interval, mean amplitude K, number of detected bubbles.

**Key finding**: onset year varies from ~2027 to ~2029 across the grid;
the default BM choice (0.20, 0.50) sits near the transition boundary at
2029.

### /D — Residual FFT spectrum

FFT power spectrum of model residuals in log-time, with window function
comparison. 6×5 grid = 30 panels.

**Rows**: 6 sampling/window combinations (daily & weekly × None/Hann/Blackman-Harris)
**Columns**: 5 residuals (BM floor, BM composite, LPPL, LPPL₂, LPPL₄)

Blue dashed vertical lines mark ω=7.38 and ω=20.9 (the LPPL₁ and LPPL₂
fitted frequencies). Top 3 peaks per panel annotated in red.

**Key finding**: peaks at ω≈9 and ω≈21 appear in every residual series
(robust signal). The ω≈13 peak is an intermodulation artifact
(W₂ − W₁ ≈ 13.5), not a real oscillation.

### /E — Rolling-window LPPL regime detection

4-panel time series showing LPPL₁ parameters over rolling 5-year windows,
stepped monthly, 129 windows total.

**Panels**: W (log-time frequency), D (damping exponent), residual σ,
R² per window.

Vertical dashed lines mark known regime events: 2013/2017/2021 bubble
peaks, Covid crash, FTX collapse, ETF approval.

**Key finding**: W saturates at the upper bound (15.0) from ~2020 onward,
while D swings wildly — both signals of structural change in Bitcoin's
cycle behavior post-2020.

## Regeneration

Each page's source is a Python script in `tools/`. To refresh after new
price data or parameter changes:

```bash
btc_venv/bin/python3 tools/build_sensitivity.py       # /B + /BB
btc_venv/bin/python3 tools/build_sensitivity_pq.py    # /C
btc_venv/bin/python3 tools/residual_fft.py            # /D
btc_venv/bin/python3 tools/regime_shift.py            # /E
```

Each writes its output to both `.svg` (served) and `.jpg` (preview) in
the project root. Commit the updated files and deploy as usual.

## Design conventions

- **Dark theme**: `facecolor="#1a1a2e"`, axes `#16213e`, text `#cccccc`
- **Deuteranomaly-safe colormaps**: cividis, plasma, inferno
- **Color families**: BM/EF share goldenrod tones (`#DAA520`, `#E8C860`);
  LPPL variants share orange tones (`#FF6D00` → `#FFE0A0`, darker to
  lighter as order increases)
