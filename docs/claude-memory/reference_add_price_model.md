---
name: How to add a new price model
description: Complete checklist of all files that must change when adding a new price model to Quantoshi
type: reference
---

## Checklist: Adding a New Price Model

1. **Implement** `PriceModel` protocol in `archive/btc_app/btc_core.py`:
   - Required fields: `name`, `short_name`, `quantized`, `quantiles`, `colors`, `fits`, `dash_style`
   - Required methods: `price_at(q, t)`, `interp_price(q, t)`, `find_percentile(t, price)`
   - `fits` dict must contain keys for all quantiles — figure builders check `q in model.fits`
   - Composite-median models: follow `LPPLModel` / `EmpiricalFloorModel` pattern
   - Log-linear models: extend `_FitsBasedModel`

2. **Register** in `btc_web/app.py` (register price models block)

3. **Update `btc_web/snapshot.py`** — add `short_name` to `_CHECKLIST_OPTIONS` for all `*-model-show` and `bub-model-show` keys (~lines 164–168). Without this, snapshot/share links can't encode the model. Old links decode safely.

4. **Update `btc_web/test_web.py`** — hardcoded `PRICE_MODELS.keys()` assertion (~line 3783) uses exact set. Use `issubset()` or add the new key. Add model-specific test class.

5. **UI auto-discovers** — `_model_show_checklist()` in `layout/common.py` and heatmap pill bar in `layout/heatmap.py` iterate `PRICE_MODELS`. No layout changes needed.

6. Add accordion item to `btc_web/layout/model_info.py`
7. Add FAQ entry if warranted in `btc_web/layout/faq.py`
8. Update `docs/architecture.md` and `docs/user_manual.md`

**How to apply:** Reference this checklist whenever adding, removing, or modifying a price model. Steps 3 and 4 are easy to miss — they don't cause import errors, just broken snapshots and failing tests.
