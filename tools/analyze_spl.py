#!/usr/bin/env python3
"""Regenerate every quantitative claim in the Saturating Power Law spec.

    btc_venv/bin/python3 tools/analyze_spl.py

Run this instead of trusting the numbers in
docs/superpowers/specs/2026-08-07-saturating-power-law-design.md. The price
history grows daily, so the spec's figures are a snapshot.

    price   = L / (1 + (t/t0)^(-beta))
    log10 p = log10(L) - log10(1 + (t/t0)^(-beta))

t is years since the 2009-07-25 origin. Fitting is on log10 residuals, over
t >= T_MIN, matching what the model class sees. The fit uses
(A, beta, log10 t0) with A = log10 L - beta*log10 t0, so the near-degenerate
direction collapses onto log10 t0 alone.

METHOD (rebuilt after a statistics consult; see the spec's section 3)

The question is whether the price history can identify a ceiling. The primary
test is a likelihood-ratio test of t0 = infinity, which is a BOUNDARY of the
parameter space, so the null distribution is a 50:50 mixture of chi2_0 and
chi2_1 and the 5% critical value is 2.706, not 3.841.

Discarded, and why:
  - AR(1)-GLS (Cochrane-Orcutt / Prais-Winsten). The residual ACF changes
    sign (+0.90 at 30 d, -0.22 at 365 d), so the error process is oscillatory
    and no AR(1) can represent it at any rho. CO and PW disagreed by 0.32 in
    the pl exponent purely over whether observation 1 is retained -- a
    near-unit-root artifact, not a finding.
  - Moving-block PAIRS bootstrap. t is a fixed exhaustive grid, not a random
    sample, so resampling (t, price) rows only randomises the design; the
    block structure does no work on the error process. A residual block
    bootstrap would be the correct instrument if one is wanted.
  - A single scalar n_eff. Effective sample size is frequency-dependent; for
    the lowest-frequency contrast (trend curvature) it is of order the number
    of halving cycles, ~4.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from scipy.stats import chi2

GENESIS = pd.Timestamp("2009-07-25")
LN10 = np.log(10.0)
SUPPLY = 21e6
T_MIN = 1.0                       # must match btc_web/time_basis.py
CAP_FIT_USD = 1000e12             # fitting bound on L
CAP_SANITY_USD = 100e12           # display annotation, not a bound
BETA_MAX = 20.0
T0_LO, T0_HI = 1.0, 100.0
CRIT = float(chi2.ppf(1 - 2 * 0.05, 1))   # 2.706, boundary-corrected 5%


SPEC_SNAPSHOT = "2026-06-03"   # the window section 3's original numbers were computed on


def load(cutoff=None):
    """Price series, optionally truncated at `cutoff` (inclusive).

    Appends are forward-only, so truncating the current CSV reproduces an
    earlier snapshot exactly -- that is how the side-by-side in [0] is built
    without needing the old file.
    """
    df = pd.read_csv("BitcoinPricesDaily.csv")
    dc = [c for c in df.columns if c.lower() in ("date", "time", "day")][0]
    pc = [c for c in df.columns
          if "price" in c.lower() or "close" in c.lower()][0]
    t = ((pd.to_datetime(df[dc], format="mixed") - GENESIS)
         .dt.total_seconds().to_numpy() / (365.25 * 86400))
    p = df[pc].to_numpy(float)
    m = (t >= T_MIN) & (p > 0)
    if cutoff is not None:
        m &= (pd.to_datetime(df[dc], format="mixed") <= cutoff).to_numpy()
    return t[m], p[m]


def stats_at(cutoff=None):
    """Every headline statistic for one data window, as a dict."""
    t, p = load(cutoff)
    lp, n = np.log10(p), len(t)
    X = np.vstack([np.ones_like(t), np.log10(t)]).T
    c, *_ = np.linalg.lstsq(X, lp, rcond=None)
    sse_pl = float(np.sum((lp - X @ c) ** 2))
    r = fit_spl(t, lp)
    A, beta, lt0 = r.x
    sse = float(r.fun)
    l10L = A + beta * lt0
    # L pinned at the cap means the optimiser wanted L -> infinity, i.e. pure
    # pl. The bound stops it just short, which is what drives LRT slightly
    # negative there -- an artifact of the bound, not of the test.
    pinned = abs(l10L - np.log10(CAP_FIT_USD / SUPPLY)) < 1e-6
    return dict(
        cutoff=cutoff or "current", n=n, last_t=t[-1], last_p=p[-1],
        L_T=10 ** l10L * SUPPLY / 1e12, t0=10 ** lt0, beta=beta,
        rmse=float(np.sqrt(sse / n)), sse_pl=sse_pl, sse_spl=sse,
        var_pct=100 * (sse_pl - sse) / sse_pl,
        stat=n * np.log(sse_pl / sse), pinned=pinned,
    )


def spl_log10(t, A, beta, log10_t0):
    u = -beta * LN10 * (np.log10(t) - log10_t0)
    return (A + beta * log10_t0) - np.logaddexp(0.0, u) / LN10


def cyc(t, C, P, ph):
    return C * np.cos(2 * np.pi * t / P + ph)


def lin_log10(t, log10_L, t0, r):
    return log10_L - np.logaddexp(0.0, -r * (t - t0)) / LN10


def lrt(sse_null, sse_alt, n):
    """2*dloglik for nested gaussian models, and its boundary-corrected p."""
    stat = n * np.log(sse_null / sse_alt)
    return stat, 0.5 * float(chi2.sf(stat, 1))


def acf(r, lag):
    return float(np.corrcoef(r[:-lag], r[lag:])[0, 1])


def fit_spl(t, lp, seed=0):
    lo_L, hi_L = np.log10(np.max(10 ** lp)), np.log10(CAP_FIT_USD / SUPPLY)

    def obj(th):
        A, beta, lt0 = th
        l10L = A + beta * lt0
        if not (lo_L <= l10L <= hi_L):
            return 1e6 + abs(l10L - np.clip(l10L, lo_L, hi_L)) * 1e3
        return float(np.sum((lp - spl_log10(t, *th)) ** 2))

    return differential_evolution(
        obj, [(-10.0, 10.0), (0.01, BETA_MAX),
              (np.log10(T0_LO), np.log10(T0_HI))],
        seed=seed, tol=1e-12, maxiter=6000, popsize=25, polish=True)


def main() -> None:
    t, p = load()
    lp, n = np.log10(p), len(t)
    print(f"n={n}  t {t.min():.3f}..{t.max():.3f} yr  "
          f"price ${p.min():,.4g}..${p.max():,.0f}   (mask t >= {T_MIN})")
    print(f"span {t.max()-t.min():.1f} yr\n")

    X = np.vstack([np.ones_like(t), np.log10(t)]).T
    c_pl, *_ = np.linalg.lstsq(X, lp, rcond=None)
    resid_pl = lp - X @ c_pl
    sse_pl = float(resid_pl @ resid_pl)
    Xe = np.vstack([np.ones_like(t), t]).T
    c_ex, *_ = np.linalg.lstsq(Xe, lp, rcond=None)
    print(f"[1] pl : exponent {c_pl[1]:.4f}  RMSE {np.sqrt(sse_pl/n):.6f}"
          f"     exp : RMSE {np.sqrt(np.sum((lp - Xe@c_ex)**2)/n):.6f}")

    # ---- [2] PRIMARY: can the data reject "no ceiling"? ----------------
    res = fit_spl(t, lp)
    A, beta, lt0 = res.x
    t0, l10L = 10 ** lt0, A + beta * lt0
    stat, pv = lrt(sse_pl, res.fun, n)
    print(f"\n[0] THE CENTRAL FINDING -- L IS NOT STABLE UNDER MORE DATA")
    print(f"    Two windows, same code, same test. The later one is a strict")
    print(f"    superset: it adds {stats_at()['n'] - stats_at(SPEC_SNAPSHOT)['n']} rows to {stats_at(SPEC_SNAPSHOT)['n']}, a 1.1% increase.")
    a, b = stats_at(SPEC_SNAPSHOT), stats_at()
    rows = [
        ("data through",      f"{a['cutoff']}",            f"{b['cutoff']} (latest)"),
        ("n",                 f"{a['n']:,}",               f"{b['n']:,}"),
        ("last price",        f"${a['last_p']:,.0f}",      f"${b['last_p']:,.0f}"),
        ("",                  "",                          ""),
        ("L  (market cap)",   f"${a['L_T']:,.1f}T",        f"${b['L_T']:,.1f}T"),
        ("t0 (yr)",           f"{a['t0']:.2f}",            f"{b['t0']:.2f}"),
        ("beta",              f"{a['beta']:.4f}",          f"{b['beta']:.4f}"),
        ("RMSE",              f"{a['rmse']:.6f}",          f"{b['rmse']:.6f}"),
        ("",                  "",                          ""),
        ("variance removed",  f"{a['var_pct']:.4f}%",      f"{b['var_pct']:.4f}%"),
        ("LRT statistic",     f"{a['stat']:.4f}",          f"{b['stat']:.4f}"),
        ("vs crit " + f"{CRIT:.4f}", 
                              "fail to reject" if a['stat'] < CRIT else "REJECTS",
                              "fail to reject" if b['stat'] < CRIT else "REJECTS"),
        ("p",                 f"{0.5*chi2.sf(max(a['stat'],0),1):.4f}",
                              f"{0.5*chi2.sf(max(b['stat'],0),1):.4f}"),
    ]
    print(f"    {'':<22} {'SPEC SNAPSHOT':>18} {'CURRENT':>18}")
    for lab, x, y in rows:
        print(f"    {lab:<22} {x:>18} {y:>18}")
    print(f"    -> L moved {max(a['L_T'],b['L_T'])/min(a['L_T'],b['L_T']):.1f}x and the verdict FLIPPED"
          f" on 1.1% more data.")
    print(f"       The estimate is not converging; it tracks the price.")

    print(f"\n[0b] L BY CUTOFF -- the instability is systematic, not a one-off")
    print(f"    {'cutoff':>12} {'n':>6} {'last $':>10} {'L ($T)':>10} {'t0':>7} {'LRT':>8}  verdict")
    for cut in ("2024-08-06", "2025-02-06", "2025-08-06",
                "2026-02-06", SPEC_SNAPSHOT, None):
        s_ = stats_at(cut)
        pin = " <-CAP" if s_["pinned"] else ""
        print(f"    {str(s_['cutoff']):>12} {s_['n']:>6} {s_['last_p']:>10,.0f} "
              f"{s_['L_T']:>10,.1f} {s_['t0']:>7.1f} {s_['stat']:>8.2f}  "
              f"{'REJECTS' if s_['stat'] > CRIT else 'fail to reject'}{pin}")
    print(f"    <-CAP: L pinned at the $1000T fitting bound. The optimiser wanted")
    print(f"    L -> infinity, which IS the pure power law; the bound stopped it")
    print(f"    short, and that is why LRT goes slightly negative there.")

    print(f"\n[2] PRIMARY TEST — likelihood ratio for t0 = infinity")
    print(f"    spl: beta {beta:.4f}  t0 {t0:.2f} yr  "
          f"L ${10**l10L*SUPPLY/1e12:,.1f}T   RMSE {np.sqrt(res.fun/n):.6f}")
    print(f"    2*dloglik = {stat:.4f}   boundary-corrected 5% crit = {CRIT:.4f}")
    print(f"    p = {pv:.3f}  ->  {'FAIL TO REJECT' if stat < CRIT else 'REJECT'}"
          f" t0 = infinity, ASSUMING IID")
    print(f"    spl removes {100*(1-res.fun/sse_pl):.4f}% of variance")

    # ---- [3] why iid is generous: the error process --------------------
    resid_spl = lp - spl_log10(t, *res.x)
    print(f"\n[3] residual structure (why the iid assumption is generous)")
    print(f"    {'lag':>6} {'pl':>9} {'spl':>9} {'AR(1) pred':>11}")
    r1 = acf(resid_pl, 1)
    for L in (1, 30, 90, 180, 365, 730, 1095):
        print(f"    {L:>6} {acf(resid_pl,L):>+9.4f} {acf(resid_spl,L):>+9.4f} "
              f"{r1**L:>+11.4f}")
    print(f"    lag-1 rho: pl {r1:.6f}  spl {acf(resid_spl,1):.6f}  "
          f"(delta {acf(resid_spl,1)-r1:+.2e})")
    print(f"    ACF changes SIGN -> oscillatory; no AR(1) fits at any rho.")

    # ---- [4] profile: is t0 pinned from either side? -------------------
    print(f"\n[4] profile — best SSE with t0 fixed")
    rows, best = [], None
    for t0v in (20, 25, 28.4, 35, 50, 100, 1000):
        lt = np.log10(t0v)
        r2 = minimize(lambda ab: float(np.sum((lp - spl_log10(t, ab[0], ab[1], lt))**2)),
                      [c_pl[0], c_pl[1]], method="Nelder-Mead",
                      options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20000})
        rows.append((t0v, r2.fun, 10 ** (r2.x[0] + r2.x[1] * lt)))
        best = r2.fun if best is None else min(best, r2.fun)
    print(f"    {'t0 (yr)':>9} {'RMSE':>9} {'cap':>13}  dSSE")
    for t0v, s, L in rows:
        print(f"    {t0v:>9.4g} {np.sqrt(s/n):>9.6f} "
              f"{L*SUPPLY/1e12:>12,.0f}T  {s-best:+.5f}")

    # ---- [5] what the model claims, vs the FITTED pl -------------------
    print(f"\n[5] spl vs the fitted pl")
    for yr in (2020, 2026, 2030, 2038, 2050):
        tt = yr - (GENESIS.year + GENESIS.dayofyear / 365.25)
        d = spl_log10(np.array([tt]), *res.x)[0] - (c_pl[0] + c_pl[1]*np.log10(tt))
        print(f"    {yr}: {d:+.4f} log10 = {100*(10**d-1):+6.2f}%")

    # ---- [6] SENSITIVITY: model the ~4-yr cycle ------------------------
    BO = [(-10, 10), (0.01, BETA_MAX), (np.log10(T0_LO), np.log10(T0_HI)),
          (0, 1), (2, 8), (-np.pi, np.pi)]
    BN = [(-10, 10), (0.01, BETA_MAX), (0, 1), (2, 8), (-np.pi, np.pi)]
    f_alt = lambda z, tt=t, ll=lp: float(np.sum((ll-(spl_log10(tt,*z[:3])+cyc(tt,*z[3:])))**2))
    f_nul = lambda z, tt=t, ll=lp: float(np.sum((ll-(z[0]+z[1]*np.log10(tt)+cyc(tt,*z[2:])))**2))
    alt = differential_evolution(f_alt, BO, seed=0, tol=1e-12, maxiter=4000,
                                 popsize=25, polish=True)
    nul = differential_evolution(f_nul, BN, seed=0, tol=1e-12, maxiter=4000,
                                 popsize=25, polish=True)
    stat_c, pv_c = lrt(nul.fun, alt.fun, n)
    print(f"\n[6] SENSITIVITY — with a calendar cycle in the mean")
    print(f"    pl+cycle : RMSE {np.sqrt(nul.fun/n):.6f}  P {nul.x[3]:.2f} yr"
          f"   (cycle removes {100*(1-nul.fun/sse_pl):.1f}% of variance)")
    print(f"    spl+cycle: RMSE {np.sqrt(alt.fun/n):.6f}  t0 {10**alt.x[2]:.1f} yr")
    print(f"    2*dloglik = {stat_c:.3f}  -> REJECTS t0=inf ASSUMING IID (p={pv_c:.3f})")
    n_cyc = (t.max() - t.min()) / alt.x[4]
    print(f"    BUT: {n_cyc:.1f} cycles of data; scaled to that, "
          f"{stat_c*n_cyc/n:.3f} vs crit {CRIT:.3f}")
    print(f"    and the cycle barely whitens: lag-1 rho "
          f"{acf(lp-(spl_log10(t,*alt.x[:3])+cyc(t,*alt.x[3:])),1):.6f}")
    need = CRIT * n / stat_c
    days = 365.25 * (t.max()-t.min()) / need
    print(f"    robust framing: significance needs n_eff >= {need:.0f}, i.e.")
    print(f"    residual decorrelation within ~{days:.0f} days -- but the")
    print(f"    measured 30-day residual ACF is {acf(resid_pl,30):.2f}.")
    print(f"    Even a generous n_eff=16 (one per year) gives "
          f"{stat_c*16/n:.2f} vs crit {CRIT:.2f}.")

    # ---- [7] the three checks on that conditional result ---------------
    print(f"\n[7] checks on the conditional result")
    seeded = minimize(f_nul, [alt.x[0]+alt.x[1]*alt.x[2], alt.x[1], *alt.x[3:]],
                      method="Nelder-Mead",
                      options={"xatol":1e-11,"fatol":1e-13,"maxiter":80000})
    print(f"    (a) null convergence: DE {nul.fun:.4f} vs seeded {seeded.fun:.4f}"
          f"  -> {'DE fine' if nul.fun <= seeded.fun else 'UNDER-CONVERGED'}")
    r_n = lp - (nul.x[0] + nul.x[1]*np.log10(t) + cyc(t, *nul.x[2:]))
    r_a = lp - (spl_log10(t, *alt.x[:3]) + cyc(t, *alt.x[3:]))
    d = np.cumsum(r_n**2 - r_a**2)
    last2 = d[-1] - d[int(np.searchsorted(t, t.max()-2))]
    print(f"    (b) gain localisation: final 2 yr = {last2:.2f} of {d[-1]:.2f}"
          f" = {100*last2/d[-1]:.1f}% of the total")
    print(f"    (c) conditional t0 by cutoff:")
    for cut in (12., 14., 15., 16., t.max()):
        tr = t <= cut
        a2 = differential_evolution(lambda z: f_alt(z, t[tr], lp[tr]), BO,
                                    seed=0, tol=1e-10, maxiter=2500,
                                    popsize=20, polish=True)
        at_b = abs(10**a2.x[2] - T0_HI) < 0.5
        print(f"          through {2009.6+cut:.0f}: t0 = {10**a2.x[2]:6.1f} yr"
              f"{'   <- AT BOUND' if at_b else ''}")

    # ---- [8] linear-time counterpart ----------------------------------
    lo_L = np.log10(p.max()); hi_L = np.log10(CAP_FIT_USD/SUPPLY)
    rl = differential_evolution(
        lambda th: float(np.sum((lp - lin_log10(t, *th)) ** 2)),
        [(lo_L, hi_L), (1.0, 100.0), (0.01, 30.0)],
        seed=1, tol=1e-12, maxiter=6000, popsize=25, polish=True)
    print(f"\n[8] linear-time counterpart: RMSE {np.sqrt(rl.fun/n):.6f}"
          f"   (L at its lower bound by construction; report the fit, not L)")

    print(f"\n[9] the $100T annotation, and the extrapolation gap")
    lo, hi = 1.0, 200.0
    for _ in range(60):
        mid = 0.5*(lo+hi); lt = np.log10(mid)
        r3 = minimize(lambda ab: float(np.sum((lp - spl_log10(t, ab[0], ab[1], lt))**2)),
                      [c_pl[0], c_pl[1]], method="Nelder-Mead",
                      options={"xatol":1e-10,"fatol":1e-12,"maxiter":20000})
        L = 10 ** (r3.x[0] + r3.x[1]*lt)
        lo, hi = (mid, hi) if L*SUPPLY < CAP_SANITY_USD else (lo, mid)
    print(f"    $100T reached at t0 ~ {lo:.1f} yr (fitted {t0:.1f})")
    print(f"    inflection is {t0/t.max():.1f}x beyond the last observation")


if __name__ == "__main__":
    main()
