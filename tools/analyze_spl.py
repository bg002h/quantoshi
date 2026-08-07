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
  5. Residual autocorrelation is measured (Durbin-Watson, lag-1 rho) and
     answered with the two corrections that actually apply: AR(1)-GLS
     (Cochrane-Orcutt) and a moving-block bootstrap. An earlier revision used
     a single scalar n_eff = n(1-rho)/(1+rho) instead; that formula is for the
     mean of a stationary series and is far too pessimistic for regression
     slopes. Measured, the exponent's bootstrap spread is 1.2x while t0's is
     ~1000x -- so there is no single effective sample size.
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
    """Durbin-Watson and lag-1 rho.

    NOTE: an earlier revision also returned n_eff = n(1-rho)/(1+rho) and used
    it as the headline correction. That formula estimates the effective sample
    size for a MEAN of a stationary series. Regression slopes draw their
    precision from the spread in x, which serial correlation does not destroy,
    so the mean formula is far too pessimistic for them -- measured, the
    exponent's bootstrap spread is 1.2x while t0's is ~1000x. A single scalar
    n_eff is the wrong abstraction; see gls_refit() and block_bootstrap().
    """
    dw = float(np.sum(np.diff(resid) ** 2) / np.sum(resid ** 2))
    rho = float(np.corrcoef(resid[:-1], resid[1:])[0, 1])
    return dw, rho


def _qd(v, rho):
    """Quasi-difference v_t - rho*v_(t-1)."""
    return v[1:] - rho * v[:-1]


def gls_refit(t, lp, x):
    """Cochrane-Orcutt feasible GLS for pl and spl.

    With rho ~ 0.998 the quasi-difference is nearly a first difference, so
    this fits day-to-day CHANGES rather than levels. Parameters that survive
    it are carried by the shape of the data, not by its level.
    """
    X = np.vstack([np.ones_like(x), x]).T
    c_ols, *_ = np.linalg.lstsq(X, lp, rcond=None)
    c = c_ols
    for _ in range(30):
        rho = float(np.corrcoef((lp - X @ c)[:-1], (lp - X @ c)[1:])[0, 1])
        Xg = np.vstack([_qd(np.ones_like(x), rho), _qd(x, rho)]).T
        c_new, *_ = np.linalg.lstsq(Xg, _qd(lp, rho), rcond=None)
        if np.allclose(c_new, c, atol=1e-10):
            c = c_new
            break
        c = c_new

    seed = [-1.178, 5.091, np.log10(28.31)]
    r0 = minimize(lambda th: float(np.sum((lp - spl_log10(t, *th)) ** 2)), seed,
                  method="Nelder-Mead",
                  options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 40000})
    th = r0.x
    for _ in range(30):
        res = lp - spl_log10(t, *th)
        rho_s = float(np.corrcoef(res[:-1], res[1:])[0, 1])
        rr = minimize(lambda z: float(np.sum(_qd(lp - spl_log10(t, *z), rho_s) ** 2)),
                      th, method="Nelder-Mead",
                      options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 40000})
        if np.allclose(rr.x, th, atol=1e-9):
            th = rr.x
            break
        th = rr.x
    return c_ols[1], c[1], r0.x, th


def block_bootstrap(t, lp, n_boot=200, block_yr=4.0, seed=0):
    """Moving-block bootstrap preserving autocorrelation within blocks.

    Block length defaults to one halving cycle, the dominant correlation
    scale. Returns per-parameter distributions -- the honest replacement for
    a single n_eff, because the parameters degrade at wildly different rates.
    """
    rng = np.random.default_rng(seed)
    n = len(t)
    blk = int(block_yr * 365.25)
    nb = int(np.ceil(n / blk))
    seed_th = [-1.178, 5.091, np.log10(28.31)]
    expo, betas, t0s = [], [], []
    for _ in range(n_boot):
        st = rng.integers(0, n - blk, size=nb)
        idx = np.sort(np.concatenate([np.arange(s, s + blk) for s in st])[:n])
        tt, ll = t[idx], lp[idx]
        Xb = np.vstack([np.ones_like(tt), np.log10(tt)]).T
        cb, *_ = np.linalg.lstsq(Xb, ll, rcond=None)
        expo.append(cb[1])
        rb = minimize(lambda z: float(np.sum((ll - spl_log10(tt, *z)) ** 2)),
                      seed_th, method="Nelder-Mead",
                      options={"xatol": 1e-9, "fatol": 1e-11, "maxiter": 8000})
        betas.append(rb.x[1])
        t0s.append(10 ** rb.x[2])
    return np.array(expo), np.array(betas), np.array(t0s)


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

    # ---- correction 5: autocorrelation, and what to do about it --------
    dw, rho = autocorr(resid_pl)
    print(f"\n[2] residual autocorrelation (governs everything below)")
    print(f"    Durbin-Watson {dw:.4f}   lag-1 rho {rho:.6f}")
    print(f"    DW is a DIAGNOSTIC, not a correction. The corrections are")
    print(f"    GLS ([2a]) and the block bootstrap ([2b]).")

    # [2a] GLS: fit day-to-day changes instead of levels
    expo_ols, expo_gls, spl_ols, spl_gls = gls_refit(t, lp, np.log10(t))
    A_o, b_o, l_o = spl_ols
    A_g, b_g, l_g = spl_gls
    print(f"\n[2a] AR(1)-GLS (Cochrane-Orcutt) vs OLS point estimates")
    print(f"     pl  exponent : OLS {expo_ols:.4f}  ->  GLS {expo_gls:.4f}"
          f"   ({expo_gls-expo_ols:+.4f})")
    print(f"     spl beta     : OLS {b_o:.4f}  ->  GLS {b_g:.4f}"
          f"   ({b_g-b_o:+.4f})   <- shape survives")
    print(f"     spl t0 (yr)  : OLS {10**l_o:.2f}  ->  GLS {10**l_g:.2f}")
    print(f"     spl ceiling  : OLS ${10**(A_o+b_o*l_o)*SUPPLY/1e12:,.1f}T"
          f"  ->  GLS ${10**(A_g+b_g*l_g)*SUPPLY/1e12:,.1f}T"
          f"   <- moves ~8x; not a real feature")

    # [2b] block bootstrap: per-parameter uncertainty
    expo_b, beta_b, t0_b = block_bootstrap(t, lp)
    print(f"\n[2b] moving-block bootstrap (4-yr blocks = 1 halving cycle)")
    print(f"     n_eff is NOT one number -- parameters degrade at different rates:")
    qq = lambda a, k: float(np.percentile(a, k))
    for nm, a, unit in (("pl exponent", expo_b, ""), ("spl beta", beta_b, ""),
                        ("spl t0", t0_b, " yr")):
        lo, hi = qq(a, 5), qq(a, 95)
        print(f"       {nm:12s} median {np.median(a):9.3f}{unit}"
              f"   5-95%: {lo:.3f} .. {hi:.3f}   spread {hi/max(lo,1e-12):,.1f}x")

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
    print(f"    NB: these assume iid residuals, which [2] refutes. They are")
    print(f"        reported at face value; [2b] is the honest uncertainty.")

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
    for lbl, k, nn in (("dAIC<=1, n", 1.0, n), ("dAIC<=2, n", 2.0, n)):
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
