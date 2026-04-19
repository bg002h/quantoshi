# Leverage Calculator — Design Spec

**Date:** 2026-04-18
**Target page:** `/leverage` on Quantoshi
**Status:** Design approved, ready for implementation plan

## 1. Goal

Interactive page that answers: *"At a given horizon H, target CAGR, borrow rate, opportunity cost, reversion-target model, and floor quantile, what is the maximum price I should pay for BTC today?"*

Core formula, one line:

```
P_max(c, H) = F(t_now + H) / (1 + c)^H
```

where `F(τ)` is the chosen model's quantile-q floor price at time τ, and `c` is the user's target CAGR.

## 2. User stories

- *"I can borrow at 13% and my T-bill yields 4.5%. If BTC reverts to the Q1% floor in 4 years, what's my max pay-price to still hit 20% CAGR?"*
- *"At today's price, how long would I need to hold to break even on my borrow cost?"*
- *"How does switching the floor quantile from Q1% to Q5% change my answer?"*
- *"If I trust LPPL₃ instead of the Bubble Model, does my max price move?"*

## 3. Placement and URL routing

**Option C from brainstorm — hidden 10th Dash tab.** Full tab infrastructure internally (tab_defaults, callbacks, snapshot integration, first-render trigger, palette) but hidden from the visible tab bar. Accessible only via URL `/leverage` (primary) and `/10` (numeric alias for consistency).

To promote to visible later: one-line change in `btc_web/layout/__init__.py` (add `dbc.Tab` to the `dbc.Tabs` children list). No refactor required.

Route wiring — **three tab-map sites plus the first-render plumbing**. Missing any silently no-ops navigation or skips the initial render trigger:

**Tab maps (path → tab_id):**

1. `btc_web/callbacks/routing.py:113` — canonical `_PATH_TO_TAB` Python dict.
2. `btc_web/layout/__init__.py:114` — duplicate `_PATH_TO_TAB` (explicit "keep in sync" comment).
3. `btc_web/callbacks/routing.py:308-311` — hardcoded JS object inside the URL-routing clientside callback.

All three need `"/leverage": "leverage"` and `"/10": "leverage"`. Reverse map in `_TAB_TO_PATH["leverage"] = "/leverage"` (canonical copy).

Note: `routing.py:331-338` has regex dispatch for *deep links* (`/8.N`, `/9.N`, `/faq.N`, `/mi.N`, `/1.2`, `/2.N`), not a tab map. No leverage deep-link syntax is needed for MVP — skip.

**First-render trigger plumbing** (leverage tab's callback fires when the `lev-first-render` store increments — three locations hardcode the 6-tab list and must extend to 7):

4. `btc_web/callbacks/routing.py:40-68` — tab-switch clientside callback. Add `leverage:6` to the JS map, a 7th `NU` slot in the output array, a 7th Output/State pair.
5. `btc_web/callbacks/routing.py:77-106` — snapshot-restore clientside callback. Same edits.
6. `btc_web/layout/__init__.py:275` — hardcoded 6-tuple `("bubble", "heatmap", "dca", "retire", "supercharge", "citadel")` in the list comprehension that creates `*-first-render` Stores. Add `"leverage"`.

**Pre-inject hook:**

7. `btc_web/layout/__init__.py:121` — `_TAB_TO_GRAPH["leverage"] = "lev-graph"`.
8. `btc_web/layout/__init__.py:126` + `:138-143` — `_TAB_TO_FIG_FN["leverage"] = (_get_leverage_fig, leverage_defaults)` inside the lazy-init block.

Tab switching logic and URL→tab callbacks already handle arbitrary tab_ids once all eight sites are updated.

## 4. Input controls

All component IDs prefixed `lev-*`. Layout stacks vertically on mobile (`max-width: 767px`), horizontal rows on desktop.

### 4.1 Scenario context row (defaults live, user-overridable)

| ID | Type | Default | Notes |
|---|---|---|---|
| `lev-date` | `dcc.DatePickerSingle` | `date.today()` | ISO date |
| `lev-price` | `dbc.Input type=number` | latest close from `model_data.pkl` | `min=1, step=0.01` |

### 4.2 Reversion target row

| ID | Type | Default | Notes |
|---|---|---|---|
| `lev-model` | `dcc.Dropdown` | `"bub"` | 10-entry flagship list (see §4.4) |
| `lev-floor-q-store` | `dcc.Store` | `0.01` (Q1%) | pill bar state |
| 6 pill buttons | `html.Button` | Q1% active | `lev-pill-q001`, `lev-pill-q01`, `lev-pill-q05`, `lev-pill-q10`, `lev-pill-q15`, `lev-pill-q20` |

Pill-bar wiring borrows the idea from `layout/heatmap.py::_hm_pill_bar()` + `callbacks/routing.py::_hm_pill_click()` / `_hm_pill_sync()` but does **not** inherit them directly: `_HM_PILL_IDS` is baked in at module import, so the heatmap callbacks cannot be reused as-is. The leverage tab needs parallel `_LEV_PILL_IDS = ["lev-pill-q001", "lev-pill-q01", ...]` plus its own click/sync clientside callbacks. One click callback writes the selected quantile (as a float) to `lev-floor-q-store`; one sync callback updates button outlines when the store changes (e.g., from snapshot restore).

### 4.3 Rate environment row

| ID | Type | Default | Notes |
|---|---|---|---|
| `lev-rb` | `dbc.Input type=number` | `13.0` | `min=0, max=50, step=0.001` — accepts `7.125` etc. |
| `lev-rl` | `dbc.Input type=number` | `4.5` | `min=0, max=50, step=0.001` |

### 4.4 Your-scenario row

| ID | Type | Default | Notes |
|---|---|---|---|
| `lev-horizon` | `dcc.Slider` | `4.0` | `min=0.25, max=20, step=0.25`; marks every 2 yr |
| `lev-cagr` | `dcc.Slider` | `20.0` | `min=0, max=50, step=0.5`; marks at 0, 10, 20, 30, 40, 50 |

### 4.5 Flagship model list (`lev-model` dropdown)

```
Bubble Model (default)
Power Law
Quantile Regression
LPPL₃
Hybrid PPL (default config)
Entropy PPL (default config)
PCA
Greedy Select
Empirical Floor          # conditional: only if "ef" in PRICE_MODELS
User Model (U₁)          # conditional: only if user-model-store non-empty
```

Config for LPPL₃, HybPPL, EPPL: **fixed at each model's default**. Not shared with other tabs' config Stores. Rationale: keeps leverage self-contained (no Store subscriptions, no cache-key churn, no initial-render coupling). If users later want tunable config, add it as a follow-up.

## 5. Math

### 5.1 Derived quantities

Given the inputs:

```python
from datetime import date, timedelta

buy_date   = lev_date                                   # user date or today
P_now      = lev_price                                  # user price or latest close
H_yr       = max(lev_horizon, 0.01)                     # guard H > 0 (see §5.5)
c          = lev_cagr / 100                             # decimal
q          = lev_floor_q                                # decimal (0.001, 0.01, ...)
r_b        = lev_rb / 100
r_l        = lev_rl / 100

sell_date  = buy_date + timedelta(days=round(H_yr * 365.25))
sell_price = floor_price(model_short, q, sell_date)     # see §5.4
```

Origin handling: every `PriceModel` has its own origin (e.g. BM uses `2009-07-25`); the `floor_price` helper resolves calendar date → model-internal `t` (years since origin) internally. Callers pass dates.

**Custom Time Axis interaction:** leverage ignores the CTA state on Tab 1. If the user has activated CTA and re-fit PL/QR/etc. at a custom `t₀`, leverage still uses each model's stock fit from `PRICE_MODELS`. Simpler, decoupled; revisit if users ask.

### 5.2 Max pay-price curves

Plotting grid: `H_grid = np.linspace(0.25, 20, 400)` — 400 evenly spaced horizons.
For each H in the grid, compute `sell_price_H = floor_price(model, q, buy_date + H)` once, then:

```python
def P_max(sell_price_at_H, H, target_cagr):
    return sell_price_at_H / (1 + target_cagr) ** H
```

Four curves on plot: `target_cagr ∈ {0, r_l, r_b, c}`. Vectorized over `H_grid`.

### 5.3 Readout quantities

```python
P_max_at_slider = P_max(sell_price, H_yr, c)
implied_c       = (sell_price / P_now) ** (1 / H_yr) - 1 if P_now > 0 else None
is_buy          = P_now <= P_max_at_slider
delta_pct       = (P_max_at_slider - P_now) / P_now * 100 if P_now > 0 else 0
```

### 5.5 Defensive guards

Snapshot restore can load any value the user ever set, including edge values. Guard at the callback entry:

- `H_yr` clamped to `max(lev_horizon, 0.01)` — protects `(1+c)^H`, `implied_c = ratio**(1/H)`, and `sell_date = buy_date + H`.
- `P_now` clamped to `max(lev_price, 1.0)` — protects `implied_c = sell/P_now` and `delta_pct = (P_max - P_now)/P_now`.
- `c`, `r_b`, `r_l` pass through as-is after standard falsy-zero-safe coerce: `float(x) if x is not None else DEFAULT` (never `float(x or default)` — CLAUDE.md "falsy-zero" footgun).
- `q` always comes from `lev-floor-q-store`, which is only ever set by pill clicks or snapshot restore to one of the 6 preset values — no clamp needed.

### 5.4 Floor access per model

The real `PriceModel` API is **`interp_price(q, t)`** (q first, t second — `t` in float years from the model's origin), defined in `btc_core/_base.py`. Every model in the flagship dropdown implements it via one of:

- `_FitsBasedModel` (BM, QR, PCA, Greedy, EF): log-space interpolation between adjacent QR fits, handled inside `interp_price`.
- `_ShrinkingBandsMixin` or parametric `interp_price` overrides (PL, LPPL₃, HybPPL, EPPL): quantile computed from σ_eff + normal inverse.
- U₁: its own drawn-PL plus empirical residual quantization, exposed through the same `interp_price` signature.

Caution: `S2FModel.interp_price` ignores the `q` argument (by design — non-quantized). S2F is **not** in the leverage dropdown, but leave a comment so no one drops it in later.

Single helper in `figures/leverage.py`. All models in the project share the same `t=0` origin (CLAUDE.md: "All models use 2009-07-25 as their time origin"), so no per-model origin dispatch is needed:

```python
import pandas as pd
import _app_ctx

_GENESIS = pd.Timestamp("2009-07-25")  # shared across all PriceModels

def floor_price(model_short: str, q: float, target_date) -> float:
    """Return the model-q floor price at target_date in USD.
    target_date: datetime.date or datetime.datetime."""
    model = _app_ctx.PRICE_MODELS[model_short]
    t_yr = (pd.Timestamp(target_date) - _GENESIS).days / 365.25
    return model.interp_price(q, t_yr)
```

Utility reference: `btc_core/_helpers.py` exposes `yr_to_t(cal_year, genesis=...)` (calendar-year → t) and `today_t(genesis=...)` (today → t). Neither takes a `date` directly; the raw subtraction above is the correct idiom when going from a `date` object.

## 6. Output

### 6.1 Plot (`lev-graph`)

- **X-axis:** H (years), `[0.25, 20]`, linear; tick marks every 2 yr.
- **Y-axis:** max pay-price ($), log scale; tick labels formatted `$10k, $100k, $1M`.
- **Four traces** (colors from `colors.py` via palette):
  1. `Nominal breakeven (0%)` — thin, reference tone
  2. `Opportunity cost (r_l %)` — thin, cool palette tone
  3. `Borrow cost (r_b %)` — thin, warm palette tone
  4. `Your target (c %)` — **thick (3px), highlighted warm accent**
- **Horizontal dashed line** at `P_now` (current price), color via palette-aware `colors.py` constant (reuse an existing `ALERT_*` or `DASHED_REFERENCE` family; introduce `LEV_CURRENT_PRICE_LINE` only if no existing constant fits).
- **Vertical dashed gray line** at `H = lev-horizon` (slider).
- **Big dot** at the intersection of trace 4 and the vertical line, hover: `"H=4.0 yr, max pay = $87,601"`.
- **Title:** `"Max rational pay-price — reversion to {model_name} Q{q}% floor"`.
- **Subtitle:** `"Current date: {buy_date}  ·  Current price: ${P_now:,.0f}"`.
- **Watermark:** `_apply_watermark(fig)` (standard).

### 6.2 Readout (`lev-readout`, `html.Div`)

Structured block, palette-aware:

```
┌─ Scenario ───────────────────────────────────────┐
│  Buy:  2026-04-18   @ $72,926 (current BTC)      │
│  Sell: 2030-04-18   @ $181,649 (BM Q1% floor)    │
│  Horizon H = 4.0 yr                              │
├─ Your target: 20 % CAGR ─────────────────────────┤
│  Max pay-price today: $87,601                    │
│  ✓ BUY — 17 % under your max                     │
│  Implied CAGR at $72,926: 25.6 %                 │
└──────────────────────────────────────────────────┘
```

Badge logic:

- `is_buy == True` → green check, text: `"BUY — {delta_pct:+.1f} % under your max"`
- `is_buy == False` → red caution, text: `"ABOVE MAX — raise H or lower target CAGR to flip"`

### 6.3 Table (`lev-table`)

`dash_table.DataTable` with 7 columns, 7 rows:

| H (yr) | Sell date | Sell price | Max pay @ 0 % | @ r_l % | @ r_b % | @ your c % |
|---|---|---|---|---|---|---|
| 1 | {buy_date + 1y} | $F(τ+1) | ... | ... | ... | ... |
| 2 | {buy_date + 2y} | ... | ... | ... | ... | ... |
| 3 | {buy_date + 3y} | ... | ... | ... | ... | ... |
| 4 | {buy_date + 4y} | ... | ... | ... | ... | ... |
| 5 | {buy_date + 5y} | ... | ... | ... | ... | ... |
| 8 | {buy_date + 8y} | ... | ... | ... | ... | ... |
| 10 | {buy_date + 10y} | ... | ... | ... | ... | ... |

- Row where `H == round(lev-horizon)` gets soft background highlight (`style_data_conditional`).
- Column headers dynamically reflect `r_l`, `r_b`, `c` values.
- Mobile: `overflow-x: auto` wrapper.

## 7. Files

### 7.1 New files

- `btc_web/layout/leverage.py` — `_leverage_tab()` returning the tab content tree.
- `btc_web/figures/leverage.py` — `build_leverage_figure(md, p)` + `floor_price(model, q, tau)` helper.
- `btc_web/callbacks/leverage_cb.py` — one `@callback` wiring all inputs → `lev-graph.figure`, `lev-readout.children`, `lev-table.data`. Uses `prevent_initial_call=True` and the `lev-first-render` Store pattern.
- `btc_web/test_leverage.py` — unit tests (see §10).

### 7.2 Modified files

- `btc_web/tab_defaults.py` — add `LEVERAGE_DEFAULTS` `MappingProxyType` with all ~8 control defaults.
- `btc_web/layout/__init__.py` — register hidden tab in `_serve_layout`; add `"leverage"` to duplicate `_PATH_TO_TAB` at line 114; add `"leverage": "lev-graph"` to `_TAB_TO_GRAPH` at line 121; extend hardcoded 6-tuple at line 275 to 7 entries so `lev-first-render` Store exists; add `_TAB_TO_FIG_FN["leverage"]` inside the lazy-init block at line 138; pre-inject initial figure when `initial_tab == "leverage"`.
- `btc_web/callbacks/routing.py` — three separate edits: (a) add to canonical `_PATH_TO_TAB` and `_TAB_TO_PATH`; (b) add to hardcoded JS tab-map at 308-311; (c) add 7th entry (`leverage:6`) to both hardcoded 6-tab JS blocks at 40-68 and 77-106 — these drive the first-render trigger on tab switch and snapshot-restore, and each has a parallel 7-wide Output/State list that must also extend.
- `btc_web/snapshot.py` — append ~8 tuples to `_SNAPSHOT_CONTROLS`; add `"leverage"` key to `_TAB_CONTROLS`.
- `btc_web/callbacks/__init__.py` — import `leverage_cb` so its callbacks register.
- `btc_web/assets/style.css` — mobile sizing rule for `#lev-graph`.
- `btc_web/colors.py` — add any missing palette entries (reference-curve colors, badge success/danger). Regenerate artifacts via `tools/generate_color_artifacts.py`.

## 8. Snapshot / share integration

Append to `_SNAPSHOT_CONTROLS` (exact order):

```python
("lev-date",          "date"),
("lev-price",         "value"),
("lev-model",         "value"),
("lev-floor-q-store", "data"),
("lev-rb",            "value"),
("lev-rl",            "value"),
("lev-horizon",       "value"),
("lev-cagr",          "value"),
```

Add to `_TAB_CONTROLS`:

```python
_TAB_CONTROLS["leverage"] = {
    "lev-date", "lev-price", "lev-model", "lev-floor-q-store",
    "lev-rb", "lev-rl", "lev-horizon", "lev-cagr",
}
```

No checklist fields → no changes to `_CHECKLIST_OPTIONS`. All 8 controls are scalar — standard snapshot path.

## 9. Palette and mobile

### 9.1 Palette

All chart colors resolved via `colors.py` + active palette. Reuse existing constants where sensible:

- Reference curves (0%, r_l, r_b): pick from `MODEL_TRACE_COLORS` family or introduce 3 new palette-invariant constants (`LEV_REF_NEUTRAL`, `LEV_REF_COOL`, `LEV_REF_WARM`).
- Your-target curve: palette accent (match bubble-tab accent or introduce `LEV_TARGET_COLOR`).
- Current-price line: existing dashed-reference accent (magenta range).
- Badges: existing `MC_FREE_GREEN` (buy) and a red accent (use existing danger color if available, else new).

Regenerate `_colors_generated.css` + `_colors_generated.js` after any additions. Artifacts are palette-aware automatically.

### 9.2 Mobile

CSS rule in `style.css` mirroring existing chart tabs:

```css
@media (max-width: 767px) {
  #lev-graph { height: 55vw !important; min-height: 280px !important; }
  #lev-table-wrap { overflow-x: auto; }
}
```

Controls stack below chart by default via Bootstrap grid (`col-12 col-md-4` / `col-12 col-md-8` pattern). **Controls go below the chart on mobile** (matching existing site behavior — not above).

## 10. Testing

### 10.1 Unit tests (`test_leverage.py`)

- **Math correctness:**
  - `P_max(sell=181649, H=4, c=0.20) ≈ 87601`
  - `implied_cagr(sell=181649, P_now=72926, H=4) ≈ 0.256`
- **Edge cases:**
  - `H = 0.25` (minimum) — no division blow-up
  - `H = 0` (snapshot-restored pathological) — guarded to `0.01`; `implied_c` returns sentinel
  - `P_now = 0` or `P_now = None` (snapshot-restored) — guarded; no divide-by-zero
  - `c = 0` → `P_max == sell_price`
  - Floor quantile not in `qr_fits` for BM/QR → `interp_price` interpolation works
  - `model = "ef"` when EF absent from `PRICE_MODELS` → dropdown option suppressed at layout build; if forced by snapshot, `floor_price` raises and callback catches → readout shows "Model unavailable"
  - `model = "u1"` when user-model-store is empty → same fallback
- **API shape:**
  - For each flagship model, `floor_price(short, 0.01, date.today())` returns a positive float
  - `S2FModel` is not in the dropdown options list (safety check)
- **Defaults alignment:**
  - `LEVERAGE_DEFAULTS` keys match all `lev-*` controls referenced by the callback
- **Snapshot roundtrip:**
  - Encode with all 8 leverage fields → decode → same values
  - Tab-filtered encode (scope="leverage") → decode → leverage fields match, others are None
  - Restore with `H=0`, `P_now=0` → guarded; readout renders without crash
- **URL routing:**
  - `/leverage` → `active_tab == "leverage"`
  - `/10` → `active_tab == "leverage"`
  - JS tab maps updated: visiting `/leverage` client-side sets `main-tabs.active_tab = "leverage"`

### 10.2 E2E (deferred)

Out of MVP. Follow-up Playwright test once the page ships (`test_leverage_e2e.py`).

## 11. Caching

**No dedicated cache layer.** Each callback recomputes figure + readout + table from scratch. Estimated cost: <500 μs per invocation (four power-law evaluations × ~200 H grid points + 7 table rows). Cheaper than cache plumbing overhead.

Consequence: no entry in `_prewarm_caches()`; no L0/L1/L2 cache keys; no `test_cache_key_alignment` entry.

Pre-injected initial figure via `_serve_layout` still happens (standard pattern) — but it's a live compute, not a cache hit.

## 12. Out of scope for MVP

- Paywall / freemium tier (page is free).
- Gear-based config modal for LPPL₃/HybPPL/EPPL (fixed defaults only).
- Historical backtest overlay (showing theoretical P_max against actual BTC price over past years).
- Multi-target comparison (single target CAGR at a time).
- Time-varying rates over the horizon (r_b and r_l constant).
- Interest capitalization vs. discrete compounding variants (uses `(1+c)^H` — simplest).
- Leveraged-purchase sizing / margin-call risk analysis.
- Non-USD currency.

## 13. Acceptance criteria

- Visiting `/leverage` loads the tab with all defaults.
- Slider moves update plot + readout + table live.
- Model dropdown and floor pill bar recompute everything correctly.
- Rate inputs accept precise typed values (e.g. `7.125`).
- Current date + price override work.
- Share link via 📸 Share round-trips all 8 controls.
- Palette switch changes plot colors.
- Mobile stacks controls below chart; table scrolls horizontally.
- All unit tests pass (`pytest btc_web/test_leverage.py -v`).

## 14. Open questions

None. All resolved in brainstorming session:

- Placement: hidden 10th tab with URL access
- Free tier, no paywall
- Input layout with step=0.001 on rates
- 10-model flagship dropdown with fixed default config
- Snapshot integration included
- Sell date + sell price surfaced in both readout and table
- Mobile: controls below chart

## 15. Review history

**2026-04-18 first-pass review** (general-purpose agent). Addressed in the first revision:

- **Blockers fixed:** Corrected API (`interp_price(q, t)` not `quantile_at(date, q)`), removed ghost reference to `_interp_qr_price`, enumerated tab-map update sites.
- **Concerns fixed:** Flagged pill-bar's import-time hardcoded IDs (can't be inherited directly), listed `_TAB_TO_GRAPH`/`_TAB_TO_FIG_FN` as additional touchpoints, added H=0/P_now=0 snapshot-restore guards (§5.5), explicitly documented CTA non-interaction, replaced raw "magenta" color with named `colors.py` constant requirement.
- Nits (step=0.001 safety, PCA/Greedy API) confirmed as non-issues.

**2026-04-18 second-pass review** (general-purpose agent). Addressed in this revision:

- **Blockers fixed:** Corrected `model.origin` — the project uses a single shared `2009-07-25` genesis (CLAUDE.md), not per-model origins; helper rewritten accordingly. Removed invented `date_to_years` utility. Corrected the "4 JS tab-map sites" claim to 3 tab-map sites plus 2 first-render-trigger JS blocks at `routing.py:40-68` and `77-106` that must extend to include leverage. Added the `layout/__init__.py:275` hardcoded tuple to the edit list — without this, `lev-first-render` Store never exists.
- **Concerns fixed:** Corrected `_TAB_TO_FIG_FN` location (`layout/__init__.py`: declared empty at :126, populated inside the lazy-init try block at :138-143 — the latter is the edit site). Implicit declaration of `lev-first-render` now covered via §3 fix.

**Verdict after third-pass review:** clean, ready for implementation planning.
- Confirmed-clean items (interp_price signature, S2F footgun, pill-bar IDs at line 804, falsy-zero, H guard) retained as-is.
