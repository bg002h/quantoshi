---
name: update-prices
description: Fetch latest BTC prices, validate the data, append to CSV, and re-run the notebook to regenerate model_data.pkl.
disable-model-invocation: true
---

# Update Bitcoin Prices

Fetch the latest daily BTC/USD closes and update the model.

## Steps

1. **Dry run** — Preview what will be added:
   ```bash
   python3 update_prices.py --dry-run
   ```
   Show the user the preview table of new rows. If no new rows, stop — data is already current.

2. **Confirm with user** — Ask whether to proceed with the live update.

3. **Live update** — Append new rows to CSV and re-execute the notebook:
   ```bash
   python3 update_prices.py
   ```
   This appends to `BitcoinPricesDaily.csv` and runs `SP.ipynb` (takes ~2-4 minutes due to chart generation).

4. **Validate** — Check the updated CSV:
   ```bash
   tail -5 BitcoinPricesDaily.csv
   ```
   Verify dates are contiguous and prices look reasonable (no zeros, no duplicates).

5. **Commit** — Stage and commit the updated files:
   ```bash
   git add BitcoinPricesDaily.csv btc_app/model_data.pkl SP.ipynb
   git commit -m "Daily price update $(date +%Y-%m-%d)"
   ```

## Notes
- The update intentionally skips the 8 most recent days (settling period)
- Binance is the primary source; CoinGecko is the fallback (for US geo-blocked users)
- The notebook timeout is 600s — do not reduce it
- Do NOT push or deploy automatically — the user will say "deploy" when ready
