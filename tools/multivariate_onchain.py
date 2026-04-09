#!/usr/bin/env python3
"""Multivariate analysis: do on-chain metrics improve the power law?

Fetches daily hash rate, active addresses, transaction count, UTXO count,
and difficulty from blockchain.com (free, no auth), aligns with price data,
and tests whether any metric adds predictive power beyond log₁₀(t).

Usage:
    btc_venv/bin/python3 tools/multivariate_onchain.py
"""

import json, sys, time, pathlib
from datetime import datetime, timezone
from urllib.request import urlopen, Request

import numpy as np
import pandas as pd
from scipy import stats

# ── paths ──────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "BitcoinPricesDaily.csv"
CACHE_DIR = ROOT / "tools" / "_onchain_cache"
GENESIS = datetime(2009, 7, 25, tzinfo=timezone.utc)

# ── on-chain metrics to fetch ──────────────────────────────────────────
METRICS = {
    "hash_rate":    "hash-rate",
    "active_addr":  "n-unique-addresses",
    "tx_count":     "n-transactions",
    "utxo_count":   "utxo-count",
    "difficulty":   "difficulty",
}


def _fetch_metric(slug: str, name: str) -> pd.Series:
    """Fetch daily metric from blockchain.com, cache locally."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"{name}.json"

    # Use cache if < 1 day old
    if cache_file.exists():
        age_h = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_h < 24:
            with open(cache_file) as f:
                data = json.load(f)
            print(f"  {name}: loaded from cache ({len(data)} points)")
            return _to_daily_series(data, name)

    url = f"https://api.blockchain.info/charts/{slug}?timespan=all&format=json"
    print(f"  {name}: fetching from blockchain.com ...", end=" ", flush=True)
    req = Request(url, headers={"User-Agent": "Quantoshi/1.0"})
    with urlopen(req, timeout=60) as resp:
        raw = json.loads(resp.read())
    values = raw["values"]
    with open(cache_file, "w") as f:
        json.dump(values, f)
    print(f"{len(values)} points")

    return _to_daily_series(values, name)


def _to_daily_series(values: list, name: str) -> pd.Series:
    """Convert sparse timestamped values to daily series via interpolation."""
    dates = [datetime.fromtimestamp(d["x"], tz=timezone.utc).date() for d in values]
    vals = [d["y"] for d in values]
    s = pd.Series(vals, index=pd.DatetimeIndex(dates), name=name)
    s = s[~s.index.duplicated(keep="first")]
    s = s.sort_index()
    # Resample to daily, interpolate
    s = s.resample("D").mean().interpolate(method="linear")
    return s


def load_price_data() -> pd.DataFrame:
    """Load price CSV and compute t (years since genesis) and log10(price)."""
    df = pd.read_csv(CSV_PATH)
    col = "Close" if "Close" in df.columns else df.columns[-1]
    date_col = "Date" if "Date" in df.columns else df.columns[0]
    df["date"] = pd.to_datetime(df[date_col]).dt.date
    df["price"] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["price"])
    df = df[df["price"] > 0].copy()
    df["t_days"] = df["date"].apply(lambda d: (datetime(d.year, d.month, d.day, tzinfo=timezone.utc) - GENESIS).days)
    df = df[df["t_days"] > 0].copy()
    df["t_yr"] = df["t_days"] / 365.25
    df["log_t"] = np.log10(df["t_yr"].values)
    df["log_price"] = np.log10(df["price"].values)
    df.index = pd.DatetimeIndex(df["date"])
    df = df.drop(columns=["date"])
    return df


def run_regression(y, X, label):
    """OLS regression, return R², coefficients, p-values."""
    from numpy.linalg import lstsq
    # Add intercept
    ones = np.ones((len(X), 1))
    Xm = np.hstack([ones, X]) if X.ndim == 2 else np.column_stack([ones, X])
    coeffs, residuals, rank, sv = lstsq(Xm, y, rcond=None)
    y_hat = Xm @ coeffs
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot
    n, k = Xm.shape
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
    # Standard errors
    mse = ss_res / (n - k)
    try:
        cov = mse * np.linalg.inv(Xm.T @ Xm)
        se = np.sqrt(np.diag(cov))
        t_stats = coeffs / se
        p_vals = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - k))
    except np.linalg.LinAlgError:
        se = np.full(k, np.nan)
        t_stats = np.full(k, np.nan)
        p_vals = np.full(k, np.nan)
    return r2, adj_r2, coeffs, se, p_vals


def main():
    print("Loading price data ...")
    price_df = load_price_data()
    print(f"  {len(price_df)} price observations")

    print("\nFetching on-chain metrics ...")
    metrics = {}
    for name, slug in METRICS.items():
        try:
            s = _fetch_metric(slug, name)
            metrics[name] = s
        except Exception as e:
            print(f"  {name}: FAILED ({e})")
        time.sleep(1)  # be polite to API

    # ── align all data ─────────────────────────────────────────────────
    print("\nAligning datasets ...")
    df = price_df[["t_yr", "log_t", "log_price"]].copy()
    for name, s in metrics.items():
        df[name] = df.index.map(s)

    # Drop rows with any missing metric
    n_before = len(df)
    df = df.dropna()
    print(f"  {n_before} → {len(df)} rows after alignment (dropped {n_before - len(df)})")

    if len(df) < 100:
        print("ERROR: too few aligned observations")
        sys.exit(1)

    y = df["log_price"].values
    log_t = df["log_t"].values

    # ── baseline: power law only ───────────────────────────────────────
    print("\n" + "=" * 72)
    print("BASELINE: log₁₀(price) = α + β·log₁₀(t)")
    r2_base, adj_r2_base, coeffs, se, pvals = run_regression(y, log_t.reshape(-1, 1), "baseline")
    print(f"  R² = {r2_base:.6f}   adj R² = {adj_r2_base:.6f}")
    print(f"  β = {coeffs[1]:.4f} ± {se[1]:.4f}   (p = {pvals[1]:.2e})")

    # ── test each metric individually ──────────────────────────────────
    print("\n" + "=" * 72)
    print("INDIVIDUAL METRIC TESTS: log₁₀(price) = α + β·log₁₀(t) + γ·log₁₀(metric)")
    print(f"{'Metric':<16} {'R²':>10} {'adj R²':>10} {'ΔR²':>10} {'γ':>10} {'p(γ)':>12}")
    print("─" * 72)

    results = []
    for name in METRICS:
        raw = df[name].values
        # Log-transform (all metrics are positive)
        with np.errstate(divide="ignore"):
            log_m = np.log10(np.maximum(raw, 1e-30))
        X = np.column_stack([log_t, log_m])
        r2, adj_r2, coeffs, se, pvals = run_regression(y, X, name)
        dr2 = r2 - r2_base
        results.append((name, r2, adj_r2, dr2, coeffs[2], pvals[2]))
        sig = "***" if pvals[2] < 0.001 else "**" if pvals[2] < 0.01 else "*" if pvals[2] < 0.05 else ""
        print(f"{name:<16} {r2:>10.6f} {adj_r2:>10.6f} {dr2:>+10.6f} {coeffs[2]:>+10.4f} {pvals[2]:>12.2e} {sig}")

    # ── test all metrics together ──────────────────────────────────────
    print("\n" + "=" * 72)
    print("FULL MODEL: log₁₀(price) = α + β·log₁₀(t) + Σ γᵢ·log₁₀(metricᵢ)")
    log_metrics = []
    names = []
    for name in METRICS:
        raw = df[name].values
        with np.errstate(divide="ignore"):
            log_m = np.log10(np.maximum(raw, 1e-30))
        log_metrics.append(log_m)
        names.append(name)
    X_all = np.column_stack([log_t] + log_metrics)
    r2_all, adj_r2_all, coeffs, se, pvals = run_regression(y, X_all, "all")
    print(f"  R² = {r2_all:.6f}   adj R² = {adj_r2_all:.6f}   ΔR² = {r2_all - r2_base:+.6f}")
    print(f"\n  {'Variable':<16} {'coeff':>10} {'SE':>10} {'p-value':>12}")
    print("  " + "─" * 52)
    var_names = ["intercept", "log₁₀(t)"] + [f"log₁₀({n})" for n in names]
    for vn, c, s, p in zip(var_names, coeffs, se, pvals):
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {vn:<16} {c:>+10.4f} {s:>10.4f} {p:>12.2e} {sig}")

    # ── Metcalfe test: price ~ addresses² ──────────────────────────────
    print("\n" + "=" * 72)
    print("METCALFE'S LAW: log₁₀(price) = α + γ·log₁₀(active_addr)")
    if "active_addr" in metrics:
        log_addr = np.log10(np.maximum(df["active_addr"].values, 1))
        r2_met, adj_r2_met, coeffs, se, pvals = run_regression(y, log_addr.reshape(-1, 1), "metcalfe")
        print(f"  R² = {r2_met:.6f}   adj R² = {adj_r2_met:.6f}")
        print(f"  γ = {coeffs[1]:.4f} ± {se[1]:.4f}   (p = {pvals[1]:.2e})")
        print(f"  If Metcalfe (value ∝ n²): expect γ ≈ 2.0")
        print(f"  Actual: γ = {coeffs[1]:.2f} → {'super-Metcalfe' if coeffs[1] > 2.0 else 'sub-Metcalfe' if coeffs[1] < 2.0 else 'Metcalfe'}")

        # Controlled: price ~ time + addresses
        print(f"\n  Controlled: log₁₀(price) = α + β·log₁₀(t) + γ·log₁₀(addr)")
        X_ta = np.column_stack([log_t, log_addr])
        r2_ta, adj_r2_ta, coeffs, se, pvals = run_regression(y, X_ta, "time+addr")
        print(f"  R² = {r2_ta:.6f}   adj R² = {adj_r2_ta:.6f}   ΔR² vs baseline = {r2_ta - r2_base:+.6f}")
        print(f"  β(time) = {coeffs[1]:.4f}  p = {pvals[1]:.2e}")
        print(f"  γ(addr) = {coeffs[2]:.4f}  p = {pvals[2]:.2e}")

    # ── hash rate as security proxy ────────────────────────────────────
    print("\n" + "=" * 72)
    print("HASH RATE AS SECURITY PROXY: log₁₀(price) = α + γ·log₁₀(hash_rate)")
    if "hash_rate" in metrics:
        log_hr = np.log10(np.maximum(df["hash_rate"].values, 1e-30))
        r2_hr, _, coeffs, se, pvals = run_regression(y, log_hr.reshape(-1, 1), "hashrate")
        print(f"  R² = {r2_hr:.6f}")
        print(f"  γ = {coeffs[1]:.4f} ± {se[1]:.4f}   (p = {pvals[1]:.2e})")

    # ── summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("─" * 72)
    print(f"  Baseline (time only):      R² = {r2_base:.6f}")
    best = max(results, key=lambda x: x[1])
    print(f"  Best single metric add:    R² = {best[1]:.6f}  (+{best[3]:.6f})  [{best[0]}]")
    print(f"  All metrics combined:      R² = {r2_all:.6f}  (+{r2_all - r2_base:.6f})")
    if "active_addr" in metrics:
        print(f"  Addresses only (Metcalfe): R² = {r2_met:.6f}")
    print()

    if r2_all - r2_base < 0.01:
        print("  CONCLUSION: On-chain metrics add < 1% R² above the power law.")
        print("  The power law is predominantly a function of time.")
        print("  However, collinearity (metrics grow with time) makes it hard to")
        print("  distinguish 'price follows time' from 'price follows adoption")
        print("  which grows with time'.")
    else:
        print("  CONCLUSION: On-chain metrics DO improve the power law!")
        print("  The power law may partially reflect network effects, not just time.")


if __name__ == "__main__":
    main()
