#!/usr/bin/env python3
"""Fetch historical monthly returns for equities, bonds, and treasuries.

Data sources:
  - S&P 500 total return: Yahoo Finance (^SP500TR or SPY adjusted close)
  - US Aggregate Bond: Yahoo Finance (AGG adjusted close)
  - Treasury yields: FRED via Yahoo Finance proxy (^IRX for 3mo, ^FVX for 5yr, ^TYX for 30yr)

Output: CSV files in btc_web/data/ with monthly return series.

Usage:
    python3 btc_web/data/fetch_historical.py [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_DATA_DIR = Path(__file__).parent


def _fetch_yahoo(ticker: str, start: str = "1990-01-01",
                 end: str | None = None) -> pd.Series:
    """Fetch monthly adjusted close prices from Yahoo Finance."""
    import yfinance as yf
    t = yf.Ticker(ticker)
    df = t.history(start=start, end=end, interval="1mo", auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    return df["Close"].dropna()


def fetch_equity_returns(start: str = "1990-01-01") -> pd.DataFrame:
    """Fetch S&P 500 monthly total returns."""
    # Try total return index first, fall back to SPY
    for ticker in ("^SP500TR", "SPY"):
        try:
            prices = _fetch_yahoo(ticker, start=start)
            if len(prices) > 12:
                break
        except Exception:
            continue
    else:
        raise RuntimeError("Could not fetch equity data from Yahoo Finance")

    returns = prices.pct_change().dropna()
    df = pd.DataFrame({
        "date": returns.index,
        "monthly_return": returns.values,
    })
    df["date"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp()
    return df.set_index("date")


def fetch_bond_returns(start: str = "2003-09-01") -> pd.DataFrame:
    """Fetch US Aggregate Bond monthly returns (AGG ETF)."""
    prices = _fetch_yahoo("AGG", start=start)
    returns = prices.pct_change().dropna()
    df = pd.DataFrame({
        "date": returns.index,
        "monthly_return": returns.values,
    })
    df["date"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp()
    return df.set_index("date")


def fetch_treasury_yields(start: str = "1990-01-01") -> pd.DataFrame:
    """Fetch Treasury yields: 3mo (^IRX), 5yr (^FVX), 30yr (^TYX).

    Yahoo Finance quotes these as percentage values (e.g., 4.5 = 4.5%).
    Returns DataFrame with columns: yield_3mo, yield_5yr, yield_30yr (as decimals).
    """
    tickers = {"yield_3mo": "^IRX", "yield_5yr": "^FVX", "yield_30yr": "^TYX"}
    frames = {}
    for col, ticker in tickers.items():
        try:
            prices = _fetch_yahoo(ticker, start=start)
            # Yahoo quotes yields as percentages; convert to decimals
            frames[col] = prices / 100.0
        except Exception as e:
            print(f"  Warning: {ticker} fetch failed: {e}", file=sys.stderr)
            continue

    if not frames:
        raise RuntimeError("Could not fetch any Treasury yield data")

    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index).to_period("M").to_timestamp()
    df.index.name = "date"
    return df.dropna()


def treasury_yields_to_returns(yields_df: pd.DataFrame) -> pd.DataFrame:
    """Convert Treasury yields to approximate monthly total returns.

    Uses duration approximation:
        monthly_return ≈ yield/12 - duration × Δyield

    Durations (approximate):
        3mo T-Bill: ~0.25 yr (minimal price sensitivity)
        5yr T-Note: ~4.5 yr
        30yr T-Bond: ~20 yr
    """
    durations = {"yield_3mo": 0.25, "yield_5yr": 4.5, "yield_30yr": 20.0}
    returns = pd.DataFrame(index=yields_df.index[1:])

    for col, duration in durations.items():
        if col not in yields_df.columns:
            continue
        y = yields_df[col]
        # Income component: yield / 12
        income = y.iloc[:-1].values / 12.0
        # Price component: -duration × change in yield
        dy = y.diff().iloc[1:].values
        ret = income + (-duration * dy)
        ret_col = col.replace("yield_", "return_")
        returns[ret_col] = ret

    return returns.dropna()


def save_all(dry_run: bool = False):
    """Fetch all data and save to CSV files."""
    print("Fetching S&P 500 equity returns...")
    eq = fetch_equity_returns()
    print(f"  {len(eq)} monthly returns ({eq.index.min()} to {eq.index.max()})")
    if not dry_run:
        eq.to_csv(_DATA_DIR / "equity_returns.csv")
        print(f"  Saved to {_DATA_DIR / 'equity_returns.csv'}")

    print("Fetching US Aggregate Bond returns (AGG)...")
    bd = fetch_bond_returns()
    print(f"  {len(bd)} monthly returns ({bd.index.min()} to {bd.index.max()})")
    if not dry_run:
        bd.to_csv(_DATA_DIR / "bond_returns.csv")
        print(f"  Saved to {_DATA_DIR / 'bond_returns.csv'}")

    print("Fetching Treasury yields...")
    yld = fetch_treasury_yields()
    print(f"  {len(yld)} monthly observations ({yld.index.min()} to {yld.index.max()})")
    if not dry_run:
        yld.to_csv(_DATA_DIR / "treasury_yields.csv")
        print(f"  Saved to {_DATA_DIR / 'treasury_yields.csv'}")

    print("Computing Treasury total returns from yields...")
    tret = treasury_yields_to_returns(yld)
    print(f"  {len(tret)} monthly returns")
    for col in tret.columns:
        ann = tret[col].mean() * 12 * 100
        vol = tret[col].std() * np.sqrt(12) * 100
        print(f"    {col}: ann. return={ann:.1f}%, ann. vol={vol:.1f}%")
    if not dry_run:
        tret.to_csv(_DATA_DIR / "treasury_returns.csv")
        print(f"  Saved to {_DATA_DIR / 'treasury_returns.csv'}")

    # Summary statistics
    print("\n=== Summary ===")
    eq_ann = eq["monthly_return"].mean() * 12 * 100
    eq_vol = eq["monthly_return"].std() * np.sqrt(12) * 100
    print(f"  Equities: ann. return={eq_ann:.1f}%, ann. vol={eq_vol:.1f}%")
    bd_ann = bd["monthly_return"].mean() * 12 * 100
    bd_vol = bd["monthly_return"].std() * np.sqrt(12) * 100
    print(f"  Bonds: ann. return={bd_ann:.1f}%, ann. vol={bd_vol:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch historical asset data")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview data without saving")
    args = parser.parse_args()
    save_all(dry_run=args.dry_run)
