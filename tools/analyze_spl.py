#!/usr/bin/env python3
"""Regenerate every quantitative claim in the Saturating Power Law spec.

    btc_venv/bin/python3 tools/analyze_spl.py

Run this instead of trusting the numbers in
docs/superpowers/specs/2026-08-07-saturating-power-law-design.md — the price
history grows daily, so the spec's figures are a snapshot and this is the
source of truth for them.

    price   = L / (1 + (t/t0)^(-beta))
    log10 p = log10(L) - log10(1 + (t/t0)^(-beta))

t is years since the 2009-07-25 origin. Fitting is on log10 residuals.

Internally the fit uses (A, beta, log10 t0) where A = log10 L - beta*log10 t0
is the power-law intercept: A and beta are well determined, so the
near-degenerate direction collapses onto log10 t0 alone.

CORRECTIONS APPLIED after the R0 architect review
(docs/superpowers/reviews/2026-08-07-spl-spec-r0-architect-review.md):

  1. Data mask is now `t >= T_MIN` (1.0), matching what the model class will
     see. The previous `t > 0` fitted a different residual set than the one
     _init_shrinking_bands derives sigma from.
  2. The information-criterion band is reported at BOTH dAIC<=1 and dAIC<=2,
     each correctly labelled. The previous code used thresh = best*(1+2/n) --
     a dAIC<=2 band -- while calling it "1 AIC unit".
  3. Deviations are reported against the FITTED power law, not against spl's
     own asymptote (spl's beta != pl's slope, so they are different curves).
  4. Bounds now match spec section 4: L in [max observed, $1000T cap] enforced
     as a constraint on the derived L, t0 in [1, 100], beta in (0.01, 20].
     Bound activity is reported.
  5. Residual autocorrelation is measured and an effective sample size is
     reported, with every criterion recomputed against it. This is the
     finding that governs how section 3 may be worded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

GENESIS = pd.Timestamp("2009-07-25")
LN10 = np.log(10.0)
SUPPLY = 21e6
T_MIN = 1.0                       # must match btc_web/time_basis.py
CAP_FIT_USD = 1000e12             # fitting bound on L
CAP_SANITY_USD = 100e12           # display annotation, not a bound
BETA_MAX = 20.0
T0_LO, T0_HI = 1.0, 100.0


def load():
    df = pd.read_csv("BitcoinPricesDaily.csv")
    dc = [c for c in df.columns if c.lower() in ("date", "time", "day")][0]
    pc = [c for c in df.columns
          if "price" in c.lower() or "close" in c.lower()][0]
    t = ((pd.to_datetime(df[dc], format="mixed") - GENESIS)
         .dt.total_seconds().to_numpy() / (365.25 * 86400))
    p = df[pc].to_numpy(float)
    m = (t >= T_MIN) & (p > 0)     # correction 1
    return t[m], p[m]


def spl_log10(t, A, beta, log10_t0):
    u = -beta * LN10 * (np.log10(t) - log10_t0)
    return (A + beta * log10_t0) - np.logaddexp(0.0, u) / LN10


def lin_log10(t, log10_L, t0, r):
    return log10_L - np.logaddexp(0.0, -r * (t - t0)) / LN10


def ic(sse, k, n):
    ll = n * np.log(sse / n)
    return ll + 2 * k, ll + k * np.log(n)


def autocorr(resid):
    """Durbin-Watson, lag-1 rho, and the AR(1) effective sample size."""
    n = len(resid)
    dw = float(np.sum(np.diff(resid) ** 2) / np.sum(resid ** 2))
    rho = float(np.corrcoef(resid[:-1], resid[1:])[0, 1])
    n_eff = n * (1.0 - rho) / (1.0 + rho)
    return dw, rho, n_eff


def main() -> None:
    t, p = load()
    lp, n = np.log10(p), len(t)
    print(f"n={n}  t {t.min():.3f}..{t.max():.3f} yr  "
          f"price ${p.min():,.4g}..${p.max():,.0f}   (mask t >= {T_MIN})\n")

    # ---- baselines -----------------------------------------------------
    X = np.vstack([np.ones_like(t), np.log10(t)]).T
    c_pl, *_ = np.linalg.lstsq(X, lp, rcond=None)
    pl_pred = X @ c_pl
    resid_pl = lp - pl_pred
    sse_pl = float(np.sum(resid_pl ** 2))
    aic_pl, bic_pl = ic(sse_pl, 2, n)
    print(f"[1] power law : slope {c_pl[1]:.4f}  RMSE {np.sqrt(sse_pl/n):.6f}")
    Xe = np.vstack([np.ones_like(t), t]).T
    c_ex, *_ = np.linalg.lstsq(Xe, lp, rcond=None)
    print(f"    exponential: RMSE "
          f"{np.sqrt(np.sum((lp - Xe @ c_ex)**2)/n):.6f}")

    # ---- correction 5: how many independent observations are there? ----
    dw, rho, n_eff = autocorr(resid_pl)
    cycles = (t.max() - t.min()) / 4.0        # ~4-year halving cycle
    print(f"\n[2] residual autocorrelation (governs everything below)")
    print(f"    Durbin-Watson {dw:.4f}   lag-1 rho {rho:.6f}")
    print(f"    n_eff (AR1)   {n_eff:.1f}  from n={n}")
    print(f"    sanity check: {t.max()-t.min():.1f} yr of history is "
          f"~{cycles:.1f} four-year cycles")
    print(f"    -> treat ~{max(n_eff, cycles):.0f} as the sample size for "
          f"judging a long-run SHAPE, not {n}")

    # ---- correction 4: fit with the spec's own bounds -------------------
    lo_L, hi_L = np.log10(p.max()), np.log10(CAP_FIT_USD / SUPPLY)

    def objective(th):
        A, beta, lt0 = th
        log10L = A + beta * lt0
        if not (lo_L <= log10L <= hi_L):       # constraint on derived L
            return 1e6 + abs(log10L - np.clip(log10L, lo_L, hi_L)) * 1e3
        return float(np.sum((lp - spl_log10(t, A, beta, lt0)) ** 2))

    res = differential_evolution(
        objective,
        [(-10.0, 10.0), (0.01, BETA_MAX), (np.log10(T0_LO), np.log10(T0_HI))],
        seed=0, tol=1e-12, maxiter=6000, popsize=25, polish=True)
    A, beta, lt0 = res.x
    t0, log10L = 10 ** lt0, A + beta * lt0
    aic, bic = ic(res.fun, 3, n)
    active = []
    if abs(log10L - hi_L) < 1e-6: active.append("L@cap")
    if abs(log10L - lo_L) < 1e-6: active.append("L@floor")
    if abs(t0 - T0_LO) < 1e-3:    active.append("t0@lo")
    if abs(t0 - T0_HI) < 1e-3:    active.append("t0@hi")
    if abs(beta - BETA_MAX) < 1e-6: active.append("beta@max")
    print(f"\n[3] spl fitted under the spec's bounds "
          f"(L<=${CAP_FIT_USD/1e12:,.0f}T, t0 in [{T0_LO},{T0_HI}], beta<={BETA_MAX})")
    print(f"    beta={beta:.4f}  t0={t0:.3f} yr  L=${10**log10L:,.0f}/BTC"
          f"  = ${10**log10L*SUPPLY/1e12:,.1f}T")
    print(f"    RMSE {np.sqrt(res.fun/n):.6f}")
    print(f"    bounds active: {', '.join(active) if active else 'none (interior)'}")
    print(f"    at n={n}     : dAIC {aic-aic_pl:+.2f}  dBIC {bic-bic_pl:+.2f}"
          f"  -> {'pl' if bic > bic_pl else 'spl'} by BIC")
    a2, b2 = ic(res.fun, 3, n_eff)
    a2p, b2p = ic(sse_pl, 2, n_eff)
    print(f"    at n_eff={n_eff:.1f}: dAIC {a2-a2p:+.2f}  dBIC {b2-b2p:+.2f}"
          f"  -> {'pl' if b2 > b2p else 'spl'} by BIC")

    # ---- correction 2: bands at BOTH criteria, correctly labelled ------
    print(f"\n[4] profile: best SSE with t0 fixed")
    rows, best = [], None
    for t0v in (20, 25, 28.4, 35, 40, 50, 100, 1000):
        lt = np.log10(t0v)
        r2 = minimize(lambda ab: float(np.sum((lp - spl_log10(t, ab[0], ab[1], lt))**2)),
                      [c_pl[0], c_pl[1]], method="Nelder-Mead",
                      options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20000})
        rows.append((t0v, r2.fun, 10 ** (r2.x[0] + r2.x[1] * lt)))
        best = r2.fun if best is None else min(best, r2.fun)
    print(f"    {'t0 (yr)':>9} {'RMSE':>9} {'cap':>12}  dSSE")
    for t0v, s, L in rows:
        print(f"    {t0v:>9.4g} {np.sqrt(s/n):>9.6f} "
              f"{L*SUPPLY/1e12:>11,.0f}T  {s-best:+.5f}")
    for lbl, k, nn in (("dAIC<=1, n", 1.0, n), ("dAIC<=2, n", 2.0, n),
                       ("dAIC<=2, n_eff", 2.0, n_eff)):
        th = best * (1 + k / nn)
        inside = [f"{r[0]:g}" for r in rows if r[1] <= th]
        print(f"    {lbl:>16}: t0 = {', '.join(inside) or '(none)'}"
              f"   pl {'inside' if sse_pl <= th else 'outside'}")

    # ---- correction 3: deviation vs the FITTED pl, not spl's asymptote --
    print(f"\n[5] spl vs the FITTED power law (not spl's own asymptote)")
    for yr in (2020, 2024, 2026, 2030, 2038, 2050):
        tt = yr - (GENESIS.year + GENESIS.dayofyear / 365.25)
        if tt <= 0:
            continue
        d = (spl_log10(np.array([tt]), A, beta, lt0)[0]
             - (c_pl[0] + c_pl[1] * np.log10(tt)))
        print(f"    {yr}: {d:+.4f} log10 = {100*(10**d - 1):+6.2f}%")

    # ---- linear-time counterpart --------------------------------------
    rl = differential_evolution(
        lambda th: float(np.sum((lp - lin_log10(t, *th)) ** 2)),
        [(lo_L, hi_L), (1.0, 100.0), (0.01, 30.0)],
        seed=1, tol=1e-12, maxiter=6000, popsize=25, polish=True)
    _, bic_lin = ic(rl.fun, 3, n)
    print(f"\n[6] linear-time counterpart: RMSE {np.sqrt(rl.fun/n):.6f}  "
          f"dBIC {bic_lin-bic_pl:+.1f}")
    print(f"    L at its LOWER BOUND by construction? "
          f"{'yes' if abs(rl.x[0]-lo_L) < 1e-6 else 'no'} "
          f"-- report the dBIC, not the dollar value")

    print(f"\n[7] the $100T sanity line")
    lo, hi = 1.0, 200.0
    for _ in range(60):
        mid = 0.5 * (lo + hi); lt = np.log10(mid)
        r3 = minimize(lambda ab: float(np.sum((lp - spl_log10(t, ab[0], ab[1], lt))**2)),
                      [c_pl[0], c_pl[1]], method="Nelder-Mead",
                      options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20000})
        L = 10 ** (r3.x[0] + r3.x[1] * lt)
        lo, hi = (mid, hi) if L * SUPPLY < CAP_SANITY_USD else (lo, mid)
    print(f"    reached at t0 ~ {lo:.1f} yr (fitted t0 = {t0:.1f})")
    print(f"\n    NOTE: t0={t0:.1f} yr vs data ending at t={t.max():.2f} yr — "
          f"the inflection is extrapolated {t0/t.max():.1f}x beyond any observation.")


if __name__ == "__main__":
    main()
