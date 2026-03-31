"""Load and prepare Bitcoin price data."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class PriceData:
    df: pd.DataFrame          # filtered by years>=1 AND date>=fit_min_date (for support/bubble fitting)
    df_full: pd.DataFrame     # filtered by date>=fit_min_date only (for QR fitting + price export)
    years: np.ndarray         # from df (years>=1 subset)
    log_years: np.ndarray
    prices: np.ndarray
    log_prices: np.ndarray
    dates: list               # date strings from df


def load_prices(csv_path, genesis_date="2009-07-25", fit_min_date="2010-07-17"):
    """Load CSV, compute years since genesis, log columns, filter.

    Parameters
    ----------
    csv_path : str or Path
        Path to BitcoinPricesDaily.csv (columns: date, price).
    genesis_date : str
        Time origin for the power-law fit (default 2009-07-25).
    fit_min_date : str
        Earliest date to include in the fitting dataset (default 2010-07-17).

    Returns
    -------
    PriceData
        Dataclass with filtered df, full df, and convenience arrays.

    Notes
    -----
    Two datasets match the notebook's dual filtering:
    - Cell 0 (support/bubble): years >= 1.0 AND date >= fit_min_date
    - Cell 1 (QR/OLS/export): date >= fit_min_date only
    """
    genesis = pd.Timestamp(genesis_date)
    fit_min = pd.Timestamp(fit_min_date)
    df = pd.read_csv(csv_path)
    date_col = next((c for c in df.columns if "date" in c.lower()), df.columns[0])
    price_col = next((c for c in df.columns if "price" in c.lower()), df.columns[1])
    df = df.rename(columns={date_col: "date", price_col: "price"})
    df["date"] = pd.to_datetime(df["date"])
    df["price"] = df["price"].astype(float)
    df = df.sort_values("date").reset_index(drop=True)
    df["years"] = (df["date"] - genesis).dt.days / 365.25
    df["log_years"] = np.log10(df["years"].clip(lower=1e-10))
    df["log_price"] = np.log10(df["price"].clip(lower=1e-10))

    # df_full: date >= fit_min_date (for QR/OLS fitting + price export)
    df_full = df[df["date"] >= fit_min].reset_index(drop=True)

    # df: years >= 1.0 AND date >= fit_min_date (for support/bubble fitting)
    mask = (df["years"] >= 1.0) & (df["date"] >= fit_min)
    df = df[mask].reset_index(drop=True)

    return PriceData(
        df=df, df_full=df_full,
        years=df["years"].values, log_years=df["log_years"].values,
        prices=df["price"].values, log_prices=df["log_price"].values,
        dates=df["date"].dt.strftime("%Y-%m-%d").tolist(),
    )
