"""Offline stats exploration for the percentile cycle-fit feature."""
import os, sys, time, pathlib
import numpy as np
REPO = pathlib.Path("/scratch/code/bitcoinprojections")
sys.path.insert(0, str(REPO / "btc_web")); sys.path.insert(0, str(REPO))
os.environ.setdefault("TESTING", "1")
import app  # noqa
import _app_ctx
from figures.percentile import _percentile_series

M = _app_ctx.M
t_all = np.asarray(M.price_years, float)
keep = t_all >= 1.0
t = t_all[keep]; px = np.asarray(M.price_prices, float)[keep]


def fit(t, y, pmin=2.0, pmax=8.0, ngrid=400):
    """Calendar sinusoid, frequency scanned on a grid in [1/pmax, 1/pmin]."""
    fs = np.linspace(1 / pmax, 1 / pmin, ngrid)
    W = 2 * np.pi * np.outer(fs, t)               # (nf, n)
    C, S = np.cos(W), np.sin(W)
    n = len(t)
    # normal equations for [1, cos, sin] per f
    sy, sc, ss = y.sum(), C.sum(1), S.sum(1)
    scc, sss, scs = (C * C).sum(1), (S * S).sum(1), (C * S).sum(1)
    scy, ssy = C @ y, S @ y
    best = (np.inf, None, None)
    for k in range(len(fs)):
        A = np.array([[n, sc[k], ss[k]], [sc[k], scc[k], scs[k]], [ss[k], scs[k], sss[k]]])
        b = np.array([sy, scy[k], ssy[k]])
        coef = np.linalg.solve(A, b)
        sse = ((y - coef[0] - coef[1] * C[k] - coef[2] * S[k]) ** 2).sum()
        if sse < best[0]:
            best = (sse, fs[k], coef)
    # golden refine
    from scipy.optimize import minimize_scalar
    df = fs[1] - fs[0]
    def sse_f(f):
        w = 2 * np.pi * f * t
        X = np.column_stack([np.ones_like(t), np.cos(w), np.sin(w)])
        c, *_ = np.linalg.lstsq(X, y, rcond=None)
        return ((y - X @ c) ** 2).sum()
    f0 = best[1]
    r = minimize_scalar(sse_f, bounds=(max(f0 - df, fs[0]), min(f0 + df, fs[-1])), method="bounded")
    f = r.x
    w = 2 * np.pi * f * t
    X = np.column_stack([np.ones_like(t), np.cos(w), np.sin(w)])
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    A = np.hypot(c[1], c[2]); phi = np.arctan2(-c[2], c[1])
    sst = ((y - y.mean()) ** 2).sum()
    return dict(c=c[0], A=A, f=f, phi=phi, r2=1 - r.fun / sst)


def curve(p, x):
    return p["c"] + p["A"] * np.cos(2 * np.pi * p["f"] * x + p["phi"])


for sm in ("constant", "resqr"):
    for short in ("qr", "pl", "bub", "spl"):
        mdl = _app_ctx.PRICE_MODELS.get(short)
        if mdl is None:
            print(short, "missing"); continue
        y = np.asarray(_percentile_series(mdl, t, px, sigma_mode=sm), float)
        t0 = time.time(); p = fit(t, y); dt = time.time() - t0
        print(f"{sm:8s} {short:4s} period {1/p['f']:.3f} A {p['A']:.1f} c {p['c']:.1f} R2 {p['r2']:.3f}  ({dt*1000:.0f} ms)")

# ---- bootstrap band for QR, resqr ----
mdl = _app_ctx.PRICE_MODELS["qr"]
y = np.asarray(_percentile_series(mdl, t, px, sigma_mode="resqr"), float)
p0 = fit(t, y)
res = y - curve(p0, t)
# residual autocorrelation
r1 = np.corrcoef(res[:-1], res[1:])[0, 1]
print("lag-1 autocorr of residual (daily):", round(r1, 4), " n_eff ~", round(len(res) * (1 - r1) / (1 + r1), 1))
for lag_days in (30, 90, 180, 365, 730):
    print(f"  acf lag {lag_days}d: {np.corrcoef(res[:-lag_days], res[lag_days:])[0,1]:.3f}")

wk = slice(None, None, 7)
tw, yw = t[wk], y[wk]
fitw = curve(p0, tw); resw = yw - fitw
rng = np.random.default_rng(0)
td = t[-1]
xs = np.array([td, td + 5, td + 10, td + 20])
for Lyr in (0.5, 1.0, 2.0):
    L = int(round(Lyr * 52.18))
    n = len(resw)
    periods, curves = [], []
    t0 = time.time()
    for b in range(200):
        nb = int(np.ceil(n / L))
        starts = rng.integers(0, n, nb)
        idx = (starts[:, None] + np.arange(L)[None, :]).ravel()[:n] % n
        ystar = fitw + resw[idx]
        pb = fit(tw, ystar, ngrid=200)
        periods.append(1 / pb["f"]); curves.append(curve(pb, xs))
    curves = np.array(curves)
    lo, hi = np.percentile(curves, [10, 90], axis=0)
    print(f"block {Lyr} yr: period sd {np.std(periods):.3f} (10-90 {np.percentile(periods,10):.2f}-{np.percentile(periods,90):.2f}); "
          f"band width @ now/+5/+10/+20: {np.round(hi-lo,1)}  ({time.time()-t0:.1f}s)")

# ---- rolling-origin hindcast: does the wave beat a flat forecast? ----
print("\nRolling-origin hindcast (QR resqr): fit on data <= cutoff, score next H years")
for H in (1.0, 2.0, 4.0):
    rows = []
    for cut in np.arange(td - 10, td - H + 1e-9, 0.5):
        m_in = t <= cut
        if (t[m_in][-1] - t[m_in][0]) < 5:
            continue
        pc = fit(t[m_in][::7], y[m_in][::7])
        m_out = (t > cut) & (t <= cut + H)
        e_wave = y[m_out] - curve(pc, t[m_out])
        e_flat = y[m_out] - y[m_in].mean()
        e_last = y[m_out] - y[m_in][-1]
        rows.append((np.sqrt((e_wave**2).mean()), np.sqrt((e_flat**2).mean()), np.sqrt((e_last**2).mean())))
    rows = np.array(rows)
    wins = (rows[:, 0] < rows[:, 1]).mean()
    print(f"  H={H}: n_cut={len(rows)}  RMSE wave {rows[:,0].mean():.1f}  flat-mean {rows[:,1].mean():.1f}  last-value {rows[:,2].mean():.1f}  wave beats flat in {wins*100:.0f}% of cutoffs")

# peak timing hindcast: fit thru 2022-12-31 -> predicted next peak vs actual fitted peak
from datetime import date
def yr(d):
    return (np.datetime64(d) - np.datetime64("2009-07-25")).astype(int) / 365.25
for cutd in ("2017-06-30", "2019-06-30", "2021-06-30", "2023-06-30"):
    c_ = yr(cutd)
    m_in = t <= c_
    pc = fit(t[m_in][::7], y[m_in][::7])
    tt = np.arange(c_, c_ + 8, 1 / 365.25)
    yy = curve(pc, tt)
    pk = tt[np.flatnonzero((yy[1:-1] > yy[:-2]) & (yy[1:-1] >= yy[2:])) + 1]
    tr = tt[np.flatnonzero((yy[1:-1] < yy[:-2]) & (yy[1:-1] <= yy[2:])) + 1]
    fmt = lambda v: str(np.datetime64("2009-07-25") + np.timedelta64(int(v * 365.25), "D"))
    print(f"  cut {cutd}: period {1/pc['f']:.2f}  next peaks {[fmt(v) for v in pk[:2]]}  troughs {[fmt(v) for v in tr[:2]]}")
