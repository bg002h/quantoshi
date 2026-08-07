#!/usr/bin/env python3
"""Regenerate every quantitative claim in the Saturating Power Law spec.

    btc_venv/bin/python3 tools/analyze_spl.py

Run this instead of trusting the numbers in
docs/superpowers/specs/2026-08-07-saturating-power-law-design.md — the price
history grows daily, so the spec's figures are a snapshot and this is the
source of truth for them.

The model (short_name "spl"):

    price     = L / (1 + (t/t0)^(-beta))
    log10 p   = log10(L) - log10(1 + (t/t0)^(-beta))

t is years since the 2009-07-25 origin. Fitting is done on log10 residuals;
fitting on price would let the last two years dominate every earlier decade.

Internally the fit uses (A, beta, log10 t0) where A = log10 L - beta*log10 t0
is the power-law intercept. That reparameterisation matters: A and beta are
well determined by the data, so the near-degenerate direction collapses onto
the single coordinate log10 t0 instead of smearing across all three.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

GENESIS = pd.Timestamp("2009-07-25")
LN10 = np.log(10.0)
SUPPLY = 21e6                     # BTC max supply, for market-cap framing
CAP_FIT_USD = 1000e12             # $1000T total cap -> fitting bound on L
CAP_SANITY_USD = 100e12           # $100T -> display/sanity annotation only


def load():
    df = pd.read_csv("BitcoinPricesDaily.csv")
    dc = [c for c in df.columns if c.lower() in ("date", "time", "day")][0]
    pc = [c for c in df.columns
          if "price" in c.lower() or "close" in c.lower()][0]
    t = ((pd.to_datetime(df[dc], format="mixed") - GENESIS)
         .dt.total_seconds().to_numpy() / (365.25 * 86400))
    p = df[pc].to_numpy(float)
    m = (t > 0) & (p > 0)
    return t[m], p[m]


def spl_log10(t, A, beta, log10_t0):
    """log10 price. logaddexp keeps (t/t0)^(-beta) from overflowing: at
    t=1, t0=28, beta=5 the raw term is ~1e7 and grows during the search."""
    u = -beta * LN10 * (np.log10(t) - log10_t0)
    return (A + beta * log10_t0) - np.logaddexp(0.0, u) / LN10


def lin_log10(t, log10_L, t0, r):
    """Linear-time counterpart: price = L / (1 + exp(-r(t - t0)))."""
    return log10_L - np.logaddexp(0.0, -r * (t - t0)) / LN10


def ic(sse, k, n):
    ll = n * np.log(sse / n)
    return ll + 2 * k, ll + k * np.log(n)


def main() -> None:
    t, p = load()
    lp, n = np.log10(p), len(t)
    print(f"n={n}  t {t.min():.3f}..{t.max():.3f} yr  "
          f"price ${p.min():,.4g}..${p.max():,.0f}\n")

    # ---- baselines -----------------------------------------------------
    X = np.vstack([np.ones_like(t), np.log10(t)]).T
    c_pl, *_ = np.linalg.lstsq(X, lp, rcond=None)
    sse_pl = float(np.sum((lp - X @ c_pl) ** 2))
    aic_pl, bic_pl = ic(sse_pl, 2, n)
    print(f"[1] power law : slope {c_pl[1]:.4f}  RMSE {np.sqrt(sse_pl/n):.6f}")

    Xe = np.vstack([np.ones_like(t), t]).T
    c_ex, *_ = np.linalg.lstsq(Xe, lp, rcond=None)
    print(f"    exponential: RMSE "
          f"{np.sqrt(np.sum((lp - Xe @ c_ex)**2)/n):.6f}")

    # ---- log-time fit (the model) --------------------------------------
    cap = np.log10(CAP_FIT_USD / SUPPLY)
    bounds = [(-10.0, 10.0), (0.1, 20.0), (np.log10(t.max()), 2.0)]
    res = differential_evolution(
        lambda th: float(np.sum((lp - spl_log10(t, *th)) ** 2)),
        bounds, seed=0, tol=1e-12, maxiter=6000, popsize=25, polish=True)
    A, beta, lt0 = res.x
    log10L = A + beta * lt0
    aic, bic = ic(res.fun, 3, n)
    print(f"\n[2] spl (log-time)")
    print(f"    beta={beta:.4f}  t0={10**lt0:.3f} yr  "
          f"L=${10**log10L:,.0f}/BTC  (cap ${10**cap:,.0f})")
    print(f"    market cap ${10**log10L * SUPPLY / 1e12:,.1f}T"
          f"   RMSE {np.sqrt(res.fun/n):.6f}")
    print(f"    dAIC vs pl {aic-aic_pl:+.2f}   dBIC vs pl {bic-bic_pl:+.2f}"
          f"   -> {'pl' if bic > bic_pl else 'spl'} preferred by BIC")
    print(f"    L at fitting cap? {'YES' if abs(log10L-cap) < 1e-6 else 'no'}")

    # ---- linear-time counterpart (documented negative result) ----------
    rl = differential_evolution(
        lambda th: float(np.sum((lp - lin_log10(t, *th)) ** 2)),
        [(np.log10(p.max()), cap), (1.0, 100.0), (0.01, 30.0)],
        seed=1, tol=1e-12, maxiter=6000, popsize=25, polish=True)
    l10L, t0l, rr = rl.x
    _, bic_lin = ic(rl.fun, 3, n)
    print(f"\n[3] linear-time counterpart (rejected)")
    print(f"    L=${10**l10L:,.0f}/BTC  t0={t0l:.3f} yr  r={rr:.4f}"
          f"   RMSE {np.sqrt(rl.fun/n):.6f}")
    print(f"    dBIC vs pl {bic_lin-bic_pl:+.1f}"
          f"    L pinned to highest observed price? "
          f"{'YES' if abs(10**l10L - p.max()) / p.max() < 0.01 else 'no'}")

    # ---- how well is the ceiling determined? ---------------------------
    print(f"\n[4] profile: best SSE with t0 held fixed")
    print(f"    {'t0 (yr)':>9} {'RMSE':>9} {'L ($/BTC)':>13} {'cap':>9}"
          f"  dSSE vs best")
    rows, best = [], None
    for t0v in (20, 25, 28.4, 35, 50, 100, 1000):
        lt = np.log10(t0v)
        r2 = minimize(lambda ab: float(np.sum((lp - spl_log10(t, ab[0], ab[1], lt))**2)),
                      [c_pl[0], c_pl[1]], method="Nelder-Mead",
                      options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20000})
        L = 10 ** (r2.x[0] + r2.x[1] * lt)
        rows.append((t0v, r2.fun, L))
        best = r2.fun if best is None else min(best, r2.fun)
    for t0v, s, L in rows:
        print(f"    {t0v:>9.4g} {np.sqrt(s/n):>9.6f} {L:>13,.0f} "
              f"{L*SUPPLY/1e12:>8,.0f}T  {s-best:+.5f}")
    thresh = best * (1 + 2.0 / n)          # ~1 AIC unit
    band = [f"{r[0]:g}" for r in rows if r[1] <= thresh]
    print(f"    within 1 AIC unit of the optimum: t0 = {', '.join(band)} yr")
    print(f"    pl inside that band? {'yes' if sse_pl <= thresh else 'no'}")

    # ---- where does the $100T sanity line bind? ------------------------
    lo, hi = 1.0, 200.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lt = np.log10(mid)
        r3 = minimize(lambda ab: float(np.sum((lp - spl_log10(t, ab[0], ab[1], lt))**2)),
                      [c_pl[0], c_pl[1]], method="Nelder-Mead",
                      options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20000})
        L = 10 ** (r3.x[0] + r3.x[1] * lt)
        if L * SUPPLY < CAP_SANITY_USD:
            lo = mid
        else:
            hi = mid
    print(f"\n[5] the $100T sanity line is reached at t0 ~ {lo:.1f} yr")
    print(f"    (fitted t0 is {10**lt0:.1f} yr, so it is currently below it)")

    # ---- what the model actually claims --------------------------------
    print(f"\n[6] spl - pl = -log10(1 + (t/t0)^beta), at the fitted values")
    for yr in (2020, 2024, 2026, 2030, 2038, 2050):
        tt = yr - (GENESIS.year + GENESIS.dayofyear / 365.25)
        if tt <= 0:
            continue
        d = -np.log10(1.0 + (tt / 10**lt0) ** beta)
        print(f"    {yr}: {d:+.4f} log10 = {100*(10**d - 1):+6.2f}% vs power law")


if __name__ == "__main__":
    main()
