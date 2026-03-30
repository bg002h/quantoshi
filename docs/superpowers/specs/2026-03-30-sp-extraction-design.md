# SP.ipynb Extraction — Design Spec

**Goal:** Extract model computation from `SP.ipynb` into a standalone build script (`tools/build_bm_model.py`), move visual config into `btc_web/theme.py`, and slim the pkl to model-only keys. The original `SP.ipynb` remains for research/exploration but is no longer source of truth for anything.

**Motivation:** `SP.ipynb` is 2500+ lines across 8 cells, takes 2+ minutes to execute, and bundles model computation, chart generation, visual config, and interactive exploration in one file. The pkl it produces mixes model data with 25 UI constants that don't belong in a model artifact. Editing the notebook requires JSON patch scripts. Price updates (`update_prices.py`) currently re-execute the entire notebook just to regenerate model fits.

---

## Three Phases

### Phase 1: `sp_stripped.ipynb` — minimal notebook, lean pkl

Create a stripped notebook at project root that produces a **lean pkl** containing only model keys. This is the low-risk step — delete code from a copy of the notebook and verify model output is key-by-key identical to the current pkl.

**Notebook structure (3 cells):**

| Cell | Source | Content | Approx lines |
|------|--------|---------|-------------|
| 0 | SP.ipynb Cell 0 | Bubble model computation only — remove all matplotlib plotting (~400 lines of chart drawing after model fitting) | ~700 |
| 1 | SP.ipynb Cell 1 | QR fitting + OLS regression + price data prep only — remove all chart generation (`_draw_channels`, `_draw_ols`, `_draw_data`, `_save_show`, all `fig` creation, ~860 lines) | ~250 |
| 2 | SP.ipynb Cell 3 | Export cell — modified to write only 17 model keys | ~60 |

**Drop entirely:** Empty cells (old 2, 4, 7), PowerPoint cell (old 2 — already `pass`), interactive cells (old 5, 6).

**Lean pkl keys (17):**

```
qr_fits              dict {q: {intercept, slope, r2}} — 27 quantile regression fits
QR_QUANTILES         list — 27 quantile values
ols_intercept        float — OLS intercept
ols_slope            float — OLS slope
GENESIS_DATE         str — '2009-07-25'
years_plot_bm        list (3000) — time grid
support_plot_bm      list (3000) — support curve in USD
bm_comp_by_n         list of lists — precomputed composites (n_future_max+1 × 3000)
bm_r2_comp           float — composite R²
bm_n_future_max      int — max future bubbles
bm_sigma0_up         float — shrinking gaussian upper sigma
bm_sigma0_down       float — shrinking gaussian lower sigma
bm_alpha_up          float — shrinking gaussian upper alpha
bm_alpha_down        float — shrinking gaussian lower alpha
price_dates          list — date strings
price_years          list — years since genesis
price_prices         list — USD prices
```

**Verification:** A test script loads both the current full pkl and the new lean pkl, then for each of the 17 model keys:
- ndarrays/lists: `np.array_equal(np.array(old[k]), np.array(new[k]))`
- dicts: recursive exact equality (float values compared bitwise via `struct.pack('d', v)`)
- floats: bitwise identical via `struct.pack('d', old[k]) == struct.pack('d', new[k])`
- strings/ints: `old[k] == new[k]`

Any mismatch = fail. This is safer than whole-file sha256 because the file format changed (fewer keys).

---

### Phase 2: `btc_web/theme.py` — visual config extraction

Move visual config out of the pkl and model object into a dedicated theme module. Figure builders import directly from `theme.py`.

**Step 1: Empirical key-by-key testing**

Before creating `theme.py`, test which of the 25 visual keys actually affect web app output:

1. Generate baseline figures from all 6 chart builders (bubble, heatmap, DCA, retire, supercharge, citadel) with current full pkl
2. For each visual key, one at a time:
   - Delete the key from the in-memory model object
   - Rebuild all 6 figures
   - Compare against baseline (byte-for-byte on serialized plotly JSON)
   - If identical → key is dead (not just in the web app — truly unused)
   - If different → key is live, needs to go in `theme.py`
3. Keys that are dead everywhere get dropped entirely — no theme.py entry, no pkl entry

**Step 2: Create `btc_web/theme.py`**

Flat module with only the keys proven live by Step 1. Expected contents (subject to empirical results):

```python
"""Quantoshi chart theme — visual constants for figure builders."""

# Chart colors
PLOT_BG_COLOR = "#FFFFFF"
TEXT_COLOR = "#222222"
TITLE_COLOR = "#1A3060"
SPINE_COLOR = "#888888"
GRID_MAJOR_COLOR = "#BBBBBB"

# CAGR heatmap
CAGR_SEG_B1 = 5.0
CAGR_SEG_B2 = 16.0
CAGR_SEG_C_LO = "#2166AC"
CAGR_SEG_C_MID1 = "#F7F7F7"
CAGR_SEG_C_MID2 = "#FF8C00"
CAGR_SEG_C_HI = "#CC1100"

# ... only keys that empirically affect output
```

Keys expected to be dead (loaded but overridden or unused):
- `qr_colors` — overwritten by `_build_thermal_colors()` at startup
- `QR_LINESTYLES` — not used in web app figure code paths
- `ZOOM_YEAR_LO/HI`, `ZOOM_PRICE_LO/HI` — likely standalone app only
- `DATA_COLOR`, `DATA_PT_SIZE`, `DATA_PT_SIZE_ZOOM` — likely standalone app only
- `GRID_MINOR_COLOR` — likely standalone app only
- `CAGR_GRAD_STEPS`, `CAGR_HEATMAP_FONTSIZE` — likely standalone app only

**Step 3: Update consumers**

- `figures/common.py` (`_base_layout`): `m.PLOT_BG_COLOR` → `theme.PLOT_BG_COLOR`
- `figures/heatmap.py`: `m.TEXT_COLOR` → `theme.TEXT_COLOR`, etc.
- `figures/bubble.py`: same pattern
- `figures/supercharge.py`: same pattern
- `btc_core.py` (`BubbleModel.__init__`): Stop loading visual keys from pkl dict. Add fallback for standalone app: if key exists in pkl, load it; otherwise skip. This keeps the standalone app working with old full pkls.

**Step 4: Verify**

Re-run all 6 chart builders. Output must be byte-identical to baseline from Step 1.

---

### Phase 3: `tools/build_bm_model.py` — standalone build script

Standalone Python script following the `tools/build_ef_model.py` pattern. Produces the identical lean pkl from pure Python — no notebook dependency.

**Script structure:**
- Reads `BitcoinPricesDaily.csv`
- Runs bubble model fitting (Cell 0 computation: support line, bubble fitting, composite, predictions)
- Runs QR fitting + OLS regression (Cell 1 computation: quantile regression, linregress)
- Writes lean pkl with 17 model keys
- Same key insertion order and pickle protocol as `sp_stripped.ipynb`

**Dependencies:** pandas, numpy, scipy (differential_evolution, linregress), statsmodels (QuantReg)

**Verification:** Compute both `sha256(build_bm_model.py output)` vs `sha256(sp_stripped.ipynb output)` AND key-by-key value comparison. Report both results. If sha256 matches, byte-identical. If only values match, accept and document why bytes differ.

**Integration:**
- `update_prices.py`: Replace `jupyter nbconvert --execute SP.ipynb` with `python3 tools/build_bm_model.py`
- CLAUDE.md: Update "Run the notebook" and "Full rebuild" sections

---

## What happens to SP.ipynb

`SP.ipynb` stays in the repo as-is for research and exploration. It is no longer source of truth for anything — the model comes from `tools/build_bm_model.py`, visual config from `btc_web/theme.py`. The notebook can be edited freely without risk to production.

`sp_stripped.ipynb` is an intermediate artifact for Phase 1 verification, committed to the repo on the `SPFix` branch. Once Phase 3 produces a byte-identical pkl, `sp_stripped.ipynb` can optionally be kept as a readable reference or deleted.

---

## Dependency Order

Phase 1 and Phase 2 are independent — they can be done in either order.

Phase 3 depends on Phase 1 (needs the stripped computation code to extract into a script, and the lean pkl to verify against).

The web app needs Phase 2 complete before it can use the lean pkl (otherwise it crashes looking for missing visual keys on the model object).

Recommended execution order: Phase 1 → Phase 2 → Phase 3.

---

## Constraints

- **Value-identical model data**: Phase 1 lean pkl model keys must be value-identical to current pkl (key-by-key bitwise float comparison).
- **Zero visual regression**: After Phase 2, all 6 chart builders produce identical figures.
- **Standalone app compatibility**: `BubbleModel` in `btc_core.py` must handle both old full pkls and new lean pkls gracefully (fallback for missing visual keys). `ModelData.__init__` has hard reads on `qr_colors` and `QR_LINESTYLES` — must be patched to use `.get()` with defaults.
- **No matplotlib in build script**: Phase 3 script has no visualization dependencies.
- **Pickle protocol 4**: Same as current Cell 3 export.
- **Deterministic computation**: Cell 0 uses `differential_evolution(seed=42+idx)` — fully seeded. Cell 1 QR/OLS fitting is deterministic. No random state anywhere. Same inputs + same library versions = identical output.

## Sigma keys pipeline

The 4 shrinking gaussian keys (`bm_sigma0_up`, `bm_sigma0_down`, `bm_alpha_up`, `bm_alpha_down`) are NOT written by Cell 3. They are injected into the pkl by `tools/fit_sigma.py` as a post-processing step. The Phase 3 build script must either:
- (a) Incorporate sigma fitting inline, or
- (b) Call `fit_sigma.py` as a post-step, or
- (c) Document a two-step build: `build_bm_model.py` then `fit_sigma.py`

Decision deferred to implementation plan.

## Serialization notes

- `qr_fits` uses **string keys** in the pkl: `{str(q): {intercept, slope, r2}}`. Both the stripped notebook and build script must match this convention.
- `qr_colors` and `QR_LINESTYLES` are removed from the lean pkl. `ModelData.__init__` must use `.get()` fallbacks. `app.py` already overwrites `qr_colors` with thermal palette at startup.

## Verification Summary

All phases compute **both** sha256 whole-file hash and key-by-key value comparison. If sha256 matches, we have byte-identical output. If only value comparison matches, we accept that and document why bytes differ.

| Phase | Verification |
|-------|-------------|
| 1 | Key-by-key value comparison: 13 model keys from lean pkl match current full pkl. (4 sigma keys come from `fit_sigma.py`, not Cell 3 — verified separately.) |
| 2 | All 6 chart figures byte-identical before/after theme.py migration |
| 3 | sha256 of build script output vs. sp_stripped output (report match/mismatch). Key-by-key value comparison as fallback. |
