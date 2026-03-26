"""Build Markov transition matrices for traditional assets (equities, bonds, treasuries).

Each asset's monthly return series is binned into regime states (e.g., 5 bins
from worst to best). The transition matrix captures the probability of moving
from one regime to another each month.

Usage:
    from data.asset_matrices import load_asset_matrices
    matrices = load_asset_matrices()
    # matrices["equity"] = {"trans": ndarray, "bin_edges": ndarray, "stats": dict}
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_DATA_DIR = Path(__file__).parent

# Default: 5 regime bins (matching BTC's default)
_DEFAULT_N_BINS = 5


def _build_return_matrix(returns: np.ndarray, n_bins: int = _DEFAULT_N_BINS,
                         ) -> dict:
    """Build a transition matrix from a monthly return series.

    Args:
        returns: 1D array of monthly returns (as decimals, e.g., 0.05 = 5%)
        n_bins: number of regime bins

    Returns:
        dict with keys:
            trans: (n_bins, n_bins) row-stochastic transition matrix
            bin_edges: (n_bins+1,) array of return percentile boundaries
            bin_means: (n_bins,) mean return within each bin
            bin_vols: (n_bins,) return volatility within each bin
            n_obs: number of observations
            ann_return: annualized mean return
            ann_vol: annualized volatility
    """
    returns = np.asarray(returns, dtype=np.float64)
    returns = returns[np.isfinite(returns)]

    if len(returns) < n_bins * 3:
        raise ValueError(f"Need at least {n_bins * 3} observations, got {len(returns)}")

    # Compute bin edges using percentiles (equal-frequency binning)
    pctiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(returns, pctiles)
    # Ensure edges are strictly increasing
    bin_edges[0] = returns.min() - 1e-10
    bin_edges[-1] = returns.max() + 1e-10

    # Assign each return to a bin
    bins = np.digitize(returns, bin_edges) - 1
    bins = np.clip(bins, 0, n_bins - 1)

    # Build transition matrix
    trans = np.zeros((n_bins, n_bins))
    for i in range(len(bins) - 1):
        trans[bins[i], bins[i + 1]] += 1

    # Normalize rows to get probabilities
    row_sums = trans.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid division by zero
    trans = trans / row_sums

    # Compute per-bin statistics
    bin_means = np.zeros(n_bins)
    bin_vols = np.zeros(n_bins)
    for b in range(n_bins):
        mask = bins == b
        if mask.sum() > 0:
            bin_means[b] = returns[mask].mean()
            bin_vols[b] = returns[mask].std() if mask.sum() > 1 else 0.0

    return {
        "trans": trans,
        "bin_edges": bin_edges,
        "bin_means": bin_means,
        "bin_vols": bin_vols,
        "n_obs": len(returns),
        "ann_return": float(returns.mean() * 12),
        "ann_vol": float(returns.std() * np.sqrt(12)),
    }


def _load_csv_returns(filename: str, col: str = "monthly_return") -> np.ndarray:
    """Load monthly returns from a CSV file."""
    path = _DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run fetch_historical.py first.")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in {filename}. Columns: {list(df.columns)}")
    return df[col].dropna().values


def build_equity_matrix(n_bins: int = _DEFAULT_N_BINS) -> dict:
    """Build transition matrix for S&P 500 equity returns."""
    returns = _load_csv_returns("equity_returns.csv")
    result = _build_return_matrix(returns, n_bins)
    result["asset"] = "equity"
    result["label"] = "S&P 500"
    return result


def build_bond_matrix(n_bins: int = _DEFAULT_N_BINS) -> dict:
    """Build transition matrix for US Aggregate Bond returns."""
    returns = _load_csv_returns("bond_returns.csv")
    result = _build_return_matrix(returns, n_bins)
    result["asset"] = "bond"
    result["label"] = "US Agg Bond"
    return result


def build_treasury_matrices(n_bins: int = _DEFAULT_N_BINS) -> dict[str, dict]:
    """Build transition matrices for Treasury total returns (3 maturities)."""
    results = {}
    for col, key, label in [
        ("return_3mo", "tres_short", "T-Bills (3mo)"),
        ("return_5yr", "tres_med", "T-Notes (5yr)"),
        ("return_30yr", "tres_long", "T-Bonds (30yr)"),
    ]:
        returns = _load_csv_returns("treasury_returns.csv", col=col)
        result = _build_return_matrix(returns, n_bins)
        result["asset"] = key
        result["label"] = label
        results[key] = result
    return results


def load_asset_matrices(n_bins: int = _DEFAULT_N_BINS) -> dict[str, dict]:
    """Load or build all asset transition matrices.

    Returns dict keyed by asset name:
        "equity": S&P 500
        "bond": US Aggregate Bond
        "tres_short": T-Bills (3mo)
        "tres_med": T-Notes (5yr)
        "tres_long": T-Bonds (30yr)

    Each value has keys: trans, bin_edges, bin_means, bin_vols, n_obs,
                          ann_return, ann_vol, asset, label
    """
    matrices = {}
    matrices["equity"] = build_equity_matrix(n_bins)
    matrices["bond"] = build_bond_matrix(n_bins)
    matrices.update(build_treasury_matrices(n_bins))
    return matrices


def print_summary(matrices: dict[str, dict]):
    """Print summary of all transition matrices."""
    for key, m in matrices.items():
        print(f"\n{m['label']} ({key}):")
        print(f"  Observations: {m['n_obs']}")
        print(f"  Ann. return: {m['ann_return']*100:.1f}%")
        print(f"  Ann. vol: {m['ann_vol']*100:.1f}%")
        print(f"  Bin means (monthly): {[f'{x*100:.2f}%' for x in m['bin_means']]}")
        print(f"  Transition matrix:")
        for i, row in enumerate(m["trans"]):
            print(f"    Bin {i}: [{', '.join(f'{p:.2f}' for p in row)}]")


if __name__ == "__main__":
    matrices = load_asset_matrices()
    print_summary(matrices)
