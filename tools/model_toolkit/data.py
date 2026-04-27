"""Load and prepare Bitcoin price data."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd


@dataclass
class PriceData:
    df: pd.DataFrame          # filtered by years>=1 AND date>=fit_min_date (for support/bubble fitting)
    df_full: pd.DataFrame     # filtered by date>=fit_min_date only (for QR fitting + price export)
    years: np.ndarray         # from df (years>=1 subset) — IN BLOCK MODE: block offsets, not years
    log_years: np.ndarray
    prices: np.ndarray
    log_prices: np.ndarray
    dates: list               # date strings from df


def load_prices(csv_path, genesis_date="2009-07-25", fit_min_date="2010-07-17",
                time_basis="calendar"):
    """Load CSV, compute t-axis (years or block offset), log columns, filter.

    Parameters
    ----------
    csv_path : str or Path
        Path to BitcoinPricesDaily.csv (columns: date, price).
    genesis_date : str
        Calendar time origin for the power-law fit (default 2009-07-25).
        Used in calendar mode; ignored in block mode (block origin is pinned
        in quantoshi.toml).
    fit_min_date : str
        Earliest date to include in the fitting dataset (default 2010-07-17).
    time_basis : {"calendar", "block"}
        Axis on which `df["years"]` is computed. Default "calendar" preserves
        existing behavior. In block mode, joins the price CSV with
        BitcoinBlocksDaily.csv and computes `df["years"] = blockheight -
        T_ORIGIN_BLOCK`. **The column is still named `years`** even in block
        mode (back-compat with downstream toolkit code that reads
        `df["years"]`).

    Returns
    -------
    PriceData
        Dataclass with filtered df, full df, and convenience arrays.

    Notes
    -----
    Two datasets match the notebook's dual filtering:
    - Cell 0 (support/bubble): years >= T_MIN AND date >= fit_min_date
    - Cell 1 (QR/OLS/export): date >= fit_min_date only

    In block mode, T_MIN equals T_PER_YEAR (≈52596 blocks ≈ 1 calendar year),
    keeping the "exclude first year" semantics.
    """
    df = pd.read_csv(csv_path)
    date_col = next((c for c in df.columns if "date" in c.lower()), df.columns[0])
    price_col = next((c for c in df.columns if "price" in c.lower()), df.columns[1])
    df = df.rename(columns={date_col: "date", price_col: "price"})
    df["date"] = pd.to_datetime(df["date"])
    df["price"] = df["price"].astype(float)
    df = df.sort_values("date").reset_index(drop=True)

    fit_min = pd.Timestamp(fit_min_date)

    if time_basis == "calendar":
        genesis = pd.Timestamp(genesis_date)
        df["years"] = (df["date"] - genesis).dt.days / 365.25
        t_min_value = 1.0
    elif time_basis == "block":
        # Need btc_web on sys.path for time_basis import (callers ensure this).
        import sys
        _btc_web = str(Path(__file__).resolve().parent.parent.parent / "btc_web")
        if _btc_web not in sys.path:
            sys.path.insert(0, _btc_web)
        from time_basis import T_ORIGIN_BLOCK, T_PER_YEAR

        block_csv = Path(__file__).resolve().parent.parent.parent / "BitcoinBlocksDaily.csv"
        if not block_csv.exists():
            raise RuntimeError(
                f"block-mode load_prices needs {block_csv} — "
                "rerun tools/build_block_map.py if missing."
            )
        blocks_df = pd.read_csv(block_csv, parse_dates=["date"])
        # Inner-join on date so misalignments (price row without block, or
        # vice versa) get dropped. The CSVs are normally tight day-by-day.
        df = df.merge(blocks_df, on="date", how="inner")
        df["years"] = (df["blockheight"] - T_ORIGIN_BLOCK).astype(float)
        t_min_value = float(T_PER_YEAR)
    else:
        raise ValueError(f"unknown time_basis {time_basis!r}")

    df["log_years"] = np.log10(df["years"].clip(lower=1e-10))
    df["log_price"] = np.log10(df["price"].clip(lower=1e-10))

    # df_full: date >= fit_min_date (for QR/OLS fitting + price export)
    df_full = df[df["date"] >= fit_min].reset_index(drop=True)

    # df: years >= T_MIN AND date >= fit_min_date (for support/bubble fitting)
    mask = (df["years"] >= t_min_value) & (df["date"] >= fit_min)
    df = df[mask].reset_index(drop=True)

    return PriceData(
        df=df, df_full=df_full,
        years=df["years"].values, log_years=df["log_years"].values,
        prices=df["price"].values, log_prices=df["log_price"].values,
        dates=df["date"].dt.strftime("%Y-%m-%d").tolist(),
    )
