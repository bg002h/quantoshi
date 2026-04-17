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

    Uses a duration + convexity approximation:
        monthly_return ≈ yield/12 − duration × Δyield + ½ × convexity × Δyield²

    The convexity term matters for long-duration bonds during sharp
    rate moves (2022 cycle: +100bp/month on the 30yr T-Bond produced
    ≈ +2% of convexity-driven return that the duration-only linear
    approximation missed, systematically overstating realized vol).

    Durations (standard market estimates):
        3mo T-Bill: ~0.25 yr
        5yr T-Note: ~4.5 yr
        30yr T-Bond: ~20 yr

    Convexity is approximated as duration × (duration + 1) — the
    coupon-bond rule of thumb. Units: years².
    """
    durations = {"yield_3mo": 0.25, "yield_5yr": 4.5, "yield_30yr": 20.0}
    returns = pd.DataFrame(index=yields_df.index[1:])

    for col, duration in durations.items():
        if col not in yields_df.columns:
            continue
        convexity = duration * (duration + 1.0)
        y = yields_df[col]
        # Income component: yield / 12
        income = y.iloc[:-1].values / 12.0
        # Price component: -duration × Δy + ½ × convexity × Δy²
        dy = y.diff().iloc[1:].values
        ret = income + (-duration * dy) + 0.5 * convexity * dy * dy
        ret_col = col.replace("yield_", "return_")
        returns[ret_col] = ret

    return returns.dropna()


def _merge_with_committed(fresh_df: pd.DataFrame, path: Path,
                           mode: str, index_name: str = "date") -> pd.DataFrame:
    """Merge freshly fetched data with the committed CSV at *path*.

    Yahoo/FRED occasionally REVISE historical values for already-settled
    months (splits, dividend adjustments, index reconstitutions). The
    committed CSV is the reference-archive; we never overwrite historical
    values that diverge from what's already there.

    Modes:
      - "append": keep every committed row; only append months strictly
        newer than the latest committed index entry. This preserves
        historical integrity across refreshes.
      - "replace": write the fresh dataframe as-is (legacy behaviour;
        used only when explicitly requested via --replace).
      - "verify": compare fresh vs committed for overlapping months and
        warn on any mismatch; return the committed dataframe unchanged.

    Returns the dataframe to write. Caller still decides whether to write.
    """
    if not path.exists() or mode == "replace":
        return fresh_df

    committed = pd.read_csv(path, index_col=0, parse_dates=True)
    committed.index.name = index_name

    if mode == "verify":
        overlap = fresh_df.index.intersection(committed.index)
        drift = []
        for idx in overlap:
            for col in fresh_df.columns:
                if col not in committed.columns:
                    continue
                fv = fresh_df.at[idx, col]
                cv = committed.at[idx, col]
                if pd.notna(fv) and pd.notna(cv):
                    if abs(float(fv) - float(cv)) > 1e-6 * max(abs(float(cv)), 1.0):
                        drift.append((idx, col, cv, fv))
        if drift:
            print(f"  WARNING: {len(drift)} historical revisions in {path.name}:")
            for idx, col, cv, fv in drift[:5]:
                print(f"    {idx.date()} {col}: committed={cv:.6g} fresh={fv:.6g}")
            if len(drift) > 5:
                print(f"    ... and {len(drift) - 5} more")
        return committed

    # "append" mode (default): only add strictly-newer rows.
    last_idx = committed.index.max()
    new_rows = fresh_df[fresh_df.index > last_idx]
    if len(new_rows) == 0:
        print(f"  {path.name}: no new rows (latest committed {last_idx.date()})")
        return committed
    print(f"  {path.name}: appending {len(new_rows)} new rows "
          f"({new_rows.index.min().date()} to {new_rows.index.max().date()})")
    return pd.concat([committed, new_rows])


def save_all(dry_run: bool = False, mode: str = "append") -> None:
    """Fetch all data and save to CSV files.

    Mode controls what happens to the committed CSVs:
      - "append": default. Keep every committed row; append only months
        strictly newer than the latest committed index. Historical
        revisions from Yahoo/FRED are ignored (the reference-archive is
        treated as immutable).
      - "verify": DRY-run style — compare fresh vs committed, warn on
        any historical mismatch, and write NOTHING.
      - "replace": legacy destructive behaviour. Overwrites all CSVs with
        whatever the data sources return today. Use only for a fresh
        bootstrap or when you genuinely want to adopt a revision sweep.
    """
    print(f"save_all mode={mode} dry_run={dry_run}")
    print("Fetching S&P 500 equity returns...")
    eq = fetch_equity_returns()
    print(f"  {len(eq)} monthly returns ({eq.index.min()} to {eq.index.max()})")
    eq_path = _DATA_DIR / "equity_returns.csv"
    eq_to_save = _merge_with_committed(eq, eq_path, mode)
    if not dry_run and mode != "verify":
        eq_to_save.to_csv(eq_path)
        print(f"  Saved to {eq_path}")

    print("Fetching US Aggregate Bond returns (AGG)...")
    bd = fetch_bond_returns()
    print(f"  {len(bd)} monthly returns ({bd.index.min()} to {bd.index.max()})")
    bd_path = _DATA_DIR / "bond_returns.csv"
    bd_to_save = _merge_with_committed(bd, bd_path, mode)
    if not dry_run and mode != "verify":
        bd_to_save.to_csv(bd_path)
        print(f"  Saved to {bd_path}")

    print("Fetching Treasury yields...")
    yld = fetch_treasury_yields()
    print(f"  {len(yld)} monthly observations ({yld.index.min()} to {yld.index.max()})")
    yld_path = _DATA_DIR / "treasury_yields.csv"
    yld_to_save = _merge_with_committed(yld, yld_path, mode)
    if not dry_run and mode != "verify":
        yld_to_save.to_csv(yld_path)
        print(f"  Saved to {yld_path}")

    print("Computing Treasury total returns from yields...")
    tret = treasury_yields_to_returns(yld_to_save)
    print(f"  {len(tret)} monthly returns")
    for col in tret.columns:
        ann = tret[col].mean() * 12 * 100
        vol = tret[col].std() * np.sqrt(12) * 100
        print(f"    {col}: ann. return={ann:.1f}%, ann. vol={vol:.1f}%")
    tret_path = _DATA_DIR / "treasury_returns.csv"
    # Treasury returns are DERIVED from yields; regenerate on every run
    # but still append-only vs the committed file so pre-existing rows
    # don't get their computed values (possibly from older convexity
    # approximation) silently rewritten.
    tret_to_save = _merge_with_committed(tret, tret_path, mode)
    if not dry_run and mode != "verify":
        tret_to_save.to_csv(tret_path)
        print(f"  Saved to {tret_path}")

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
    parser.add_argument(
        "--mode", choices=["append", "verify", "replace"], default="append",
        help=("append (default): only add rows newer than the committed "
              "CSV; verify: warn on historical-value drift and write "
              "nothing; replace: legacy destructive overwrite."),
    )
    args = parser.parse_args()
    save_all(dry_run=args.dry_run, mode=args.mode)
