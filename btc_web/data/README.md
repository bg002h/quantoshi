# Historical Asset Data — Sources & Update Guide

## Data Files

| File | Contents | Source | History |
|------|----------|--------|---------|
| `equity_returns.csv` | S&P 500 monthly total returns | Yahoo Finance (`^SP500TR` or `SPY`) | 1990-present |
| `bond_returns.csv` | US Aggregate Bond monthly returns | Yahoo Finance (`AGG` ETF) | 2003-present |
| `treasury_yields.csv` | Treasury yields (3mo/5yr/30yr) | Yahoo Finance (`^IRX`/`^FVX`/`^TYX`) | 1990-present |
| `treasury_returns.csv` | Treasury total returns (derived) | Computed from yields via duration approx | 1990-present |

## Data Sources Detail

### S&P 500 Equity Returns
- **Primary ticker:** `^SP500TR` (S&P 500 Total Return Index — includes dividends)
- **Fallback ticker:** `SPY` (SPDR S&P 500 ETF — adjusted close includes dividends)
- **API:** Yahoo Finance via `yfinance` Python library
- **Frequency:** Monthly
- **Return calculation:** Simple percentage change on adjusted close prices
- **Known issues:** Yahoo may geo-block or rate-limit. SPY data starts ~1993.

### US Aggregate Bond Returns
- **Ticker:** `AGG` (iShares Core US Aggregate Bond ETF)
- **API:** Yahoo Finance via `yfinance`
- **Frequency:** Monthly (ETF inception: Sep 2003)
- **Return calculation:** Simple percentage change on adjusted close
- **Limitations:** Only ~22 years of history. For longer history, Bloomberg Aggregate Index data would be needed (not freely available).

### Treasury Yields
- **3-month T-Bill yield:** `^IRX` (CBOE 13-Week Treasury Bill Rate)
- **5-year T-Note yield:** `^FVX` (CBOE 5-Year Treasury Note Yield)
- **30-year T-Bond yield:** `^TYX` (CBOE 30-Year Treasury Bond Yield)
- **API:** Yahoo Finance via `yfinance`
- **Units:** Yahoo quotes as percentage (e.g., 4.5 = 4.5%). Script converts to decimals (0.045).

### Treasury Total Returns (Derived)
Computed from yields using **duration approximation**:
```
monthly_return ≈ yield/12 - duration × Δyield
```

| Maturity | Duration (approx) | Description |
|----------|-------------------|-------------|
| 3-month | 0.25 years | Minimal price sensitivity, return ≈ yield/12 |
| 5-year | 4.5 years | Moderate price sensitivity |
| 30-year | 20 years | High price sensitivity, volatile total returns |

This approximation is standard in fixed-income analytics. It captures both income (coupon) and price return (duration × yield change). Accuracy decreases for large yield moves (convexity not modeled).

## How to Update

```bash
# From project root:
PYTHONPATH=".:btc_web:archive/btc_app" btc_venv/bin/python3 btc_web/data/fetch_historical.py

# Preview without saving:
PYTHONPATH=".:btc_web:archive/btc_app" btc_venv/bin/python3 btc_web/data/fetch_historical.py --dry-run
```

The script:
1. Fetches latest data from Yahoo Finance (requires internet)
2. Computes monthly returns
3. Derives Treasury total returns from yields
4. Saves CSV files to `btc_web/data/`
5. Prints summary statistics for verification

**Frequency:** Run monthly or after significant market events. Data is append-only (new months added, historical data unchanged).

**Dependencies:** `yfinance`, `pandas`, `numpy` (all in btc_venv)

## Alternative Data Sources (for future consideration)

| Source | Assets | Access | Notes |
|--------|--------|--------|-------|
| FRED API | Treasury yields, macro data | Free (API key) | More reliable than Yahoo for yields |
| Tiingo | Equities, ETFs | Free tier (500 req/day) | Better adjusted close than Yahoo |
| Quandl/Nasdaq Data Link | Broad asset classes | Free/paid tiers | Historical commodity, FX, etc. |
| Bloomberg Terminal | Everything | Expensive ($$$) | Gold standard for institutional data |

## Expected Return Ranges (sanity check)

| Asset | Ann. Return | Ann. Volatility | Notes |
|-------|------------|-----------------|-------|
| S&P 500 | 10-12% | 14-16% | Long-term average |
| US Agg Bond | 3-5% | 4-6% | Post-2003 (AGG era) |
| 3mo T-Bill | 2-4% | 0.5-1% | Near risk-free rate |
| 5yr T-Note | 3-5% | 4-5% | Moderate duration |
| 30yr T-Bond | 5-8% | 14-18% | High duration, volatile |
