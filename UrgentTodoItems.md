# Deferred feature ideas

- **Custom Time Axis (Tab 1) — "scan" mode**: user clicks two points on the
  bubble chart, panel auto-fits PL/QR/Exp through a grid of t=0 candidates
  between them, highlights the best R². Research-oriented UX, deferred from
  the initial Custom Time Axis panel spec (2026-04-13). Build as v2 once the
  core panel is stable.

# Urgent TODO — Server Callback Reduction (ALARA)

Goal: minimize network bandwidth, server RAM, and server CPU by porting server callbacks to clientside and gating chatty ones. All refs are in `btc_web/callbacks/`.

## Top wins (highest impact first)

### 1. `mc_controls.py:435-457` — `_update_mc_cost`
5 tabs × 9 inputs of pure arithmetic against static dicts (`_MC_PRICE_LIVE`, `btcpay.is_free_tier`). Fires on every slider nudge.
- **Action:** port `_calc_mc_cost` + `_mc_cost_display` to JS; serialize price tables as JS constants; keep server fallback only for actual invoice creation.
- **Impact:** very high (chattiest non-chart callback in the app).
- **Risk:** medium — free-tier table must be mirrored in JS.

### 2. `mc_controls.py:113-151` — `_mc_yr_sync` / `_mc_sc_yr_sync`
Duplicate the existing clientside `_MC_EXTEND_YR_JS` (line 251-279).
- **Action:** delete server callbacks if dead; otherwise migrate SC path to clientside.
- **Impact:** high — fires on every MC start/years/enable change.
- **Risk:** low — confirm which one actually runs.

### 3. `mc_controls.py:64-84` — `_toggle_mc_adv` × 5, `_update_regime_opts`
Return static, pre-computable options lists (`_MC_ENTRY_Q_OPTIONS`, `_regime_options(n)` for n=2..10).
- **Action:** pre-compute all variants at import, inline into JS, one clientside callback per tab.
- **Impact:** high (10 callbacks × tabs).
- **Risk:** zero.

### 4. `charts.py:68-111` — `compute_family_summaries`
29 Inputs → Store → 3 clientside fan-outs, all pure string formatting (`_format_lppl_summary`, `_format_hybppl_summary`, `_format_eppl_summary`).
- **Action:** port the 3 format helpers to JS; collapse into one clientside callback writing all 12 `-summary-inline` outputs directly; drop the Store.
- **Impact:** high — fires on every LPPL/HybPPL/EPPL knob toggle.
- **Risk:** low.

## Medium wins

### 5. `sc_loan.py:61-68` — `_toggle_custom_price_row`, `_toggle_rollover_row`
Return `{}` or `{"display":"none"}` based on a radio. Pure clientside one-liners. **Risk:** zero.

### 6. `citadel_tax_cb.py:24-30, 34-43, 117-153`
- `_open_tax_modal` — returns `True` on click, clientside one-liner.
- `_update_state_rate` — dict lookup in `STATE_TAX_RATES` (~50 entries); inline in JS.
- `_build_tax_summary` — builds HTML table from a store that already contains JSON; clientside via React descriptors.
- **Impact:** med. **Risk:** low.

### 7. `citadel_cb.py:69-91` — `show_asset_model_info`
Returns a static HTML block when `cp-asset-model == "markov"`. Pure clientside (follow `splash.py` `_PARSE_MD_JS` pattern). **Risk:** zero.

### 8. `scanner.py:68-234` — `update_scanner`
Has `prevent_initial_call=False`; runs `find_percentile` across ~10 models on every page load, even for users landing on tabs 2-9.
- **Action:** set `prevent_initial_call=True` and gate on `bubble-first-render`; verify the scanner placeholder is pre-injected at layout time or triggered on first bubble visit.
- **Impact:** med (every page load currently runs this).
- **Risk:** low — verify layout/bubble.py.

### 9. `snapshot_cb.py:144-165` — `update_effective_lots`, `update_snapshot_banner`
Both fire on page load (no `prevent_initial_call`). Trivial value merge + small `dbc.Alert` build.
- **Action:** port to clientside; skip round trip on cold pages with no snapshot.
- **Risk:** zero.

### 10. `lots.py:130-139` — `sync_table_on_load`
Rebuilds table via pure Python formatting (`_format_lots_for_table`, `_lots_summary`). Clientside candidate. **Risk:** none if row mapping is trivial.

### 11. `user_model.py:100-133` vs `177-197` — overlapping `draw_user_model` / `auto_draw`
Both build a `UserModel` from the same 4 inputs. `auto_draw` already handles updates.
- **Action:** delete `draw_user_model` or make the button purely a clientside `bub-model-show` value-add; preserve the "add u1 to bub-model-show" side effect.
- **Risk:** low.

## Patterns worth a sweep

- **Server-side style toggles:** any callback whose only output is `Output(..., "style")` returning `{}` / `{"display":"none"}` based on a checklist/radio → always clientside. Grep `Output\(.*"style"\)` on server `@callback` decorators.
- **Static-dict lookups:** `STATE_TAX_RATES`, `_MC_PRICE_LIVE`, `HM_PRESET_PALETTES`, `_regime_options` results — serialize once at import, look up in JS.
- **MC control layer:** currently ~8-10 callbacks × 5 tabs (cost display, match indicator, yr sync, regime opts, years opts, adv opts). Consolidate into one per-tab clientside MC-state module; keep server only for the BTCPay invoice path.
- **`prevent_initial_call=False` without a gate:** `scanner.py:82`, `snapshot_cb.py:148,156,179`, `lots.py:22`. Audit every non-chart server callback; default to `True` unless there's a reason.
- **"20+ Inputs → one Store → fan-out" pattern:** collapse into one clientside callback with multiple outputs.

## Skip (looks suspicious, actually fine)

- `charts.py:1001-1153` `update_bubble` and sibling chart callbacks — 50+ Inputs but first-render gate + L0/L1/L2 figure cache make them cheap on repeat hits, and every input legitimately affects the figure. Don't split.
- `snapshot_cb.py:58-72` `apply_snapshot` — ~206 `allow_duplicate` outputs look nasty, but moving to `set_props` duplicates `citadel_save_cb.py` load pattern. Fires only on actual snapshot URLs; tightly coupled with the MC first-render nudge in `routing.py:77-106`. Not worth the refactor risk.
- `mc_payment.py:33-144` `_mc_payment_initiate` — huge input list but genuinely needs server (BTCPay invoice). `ctx.triggered_id` router is correct.
- `ticker.py:27-120` — `dcc.Interval` at 20 min is fine per CLAUDE.md; per-model `find_percentile` loop is cheap; cycling is already clientside.
- `citadel_cb.py:94-465` — background callback via diskcache, doesn't block workers.
- `routing.py:538-561` lazy-loaders (`_lazy_load_auto_y_grid`, `_lazy_load_citadel`, `_lazy_load_model_info`, `_lazy_load_faq`) — firing on tab switch is the whole point; saves far more layout JSON bytes than they cost.
- `plot_appearance.py` — already clientside via generator loop.
- `charts.py:715-935` decomp option callbacks — look fat but `_resolve_decomp_model_key` needs Python-side model metadata (`component_names`, `component_details`); clientside isn't viable without shipping the model registry.
