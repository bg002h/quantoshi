# Phase 4: Payment Integration — Notes

Phase 4 is partially deferred. The spec has several TBD items:

## Already Done
- Free cached scenarios load instantly (Phase 2 cache + Phase 3 UI)
- DEV bypass verified in tests (Phase 1, Task 6)
- Existing `mc_payment.py` BTCPay flow handles paid MC runs

## Deferred (TBD in spec)
- **Discounted tier tag assignments** — "deferred to post-Phase 3 review" per spec
- **Full sim download format** — "format and serving mechanism TBD" per spec
- **Pricing scaling** — base price calculation not specified

## Remaining Work (when spec is complete)
1. Add pricing tier tags to cached entries in `citadel_band_cache.py`
2. Wire Citadel "Run MC" through BTCPay payment flow (reuse existing `mc_payment.py` pattern)
3. Sim download endpoint (background task + file generation)
4. UI: disable download button for free tier
