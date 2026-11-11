"""Calendar-time cycle fit to a model's percentile series (Tab 1 Percentile pill).

One sinusoid, ``y = c + A·cos(2πf·t + φ)``, t in calendar years since the
2009-07-25 origin, fitted to the percentile (0-100) at which each daily close
sat inside ONE model's quantile fan. Everything here is pure numpy/scipy so
tests and the figure builder can import it without Dash.

Why each choice (measured 2026-11-10, spec
``docs/superpowers/specs/2026-11-10-percentile-cycle-fit-design.md``):

* For a fixed frequency the model is linear in (c, a, b) with
  a·cos + b·sin = A·cos(·+φ), so the frequency is scanned on a grid and each
  node solved exactly — no starting guess can strand the fit. The scan is
  limited to periods of 2-8 yr: an a-priori halving-cycle band, which also
  keeps a bootstrap replicate from locking onto a sub-year wiggle.
* Daily residuals have lag-1 autocorrelation ~0.996 (n_eff ~ 12), so an
  i.i.d. bootstrap would understate the uncertainty by an order of magnitude.
  The residual ACF reaches ~0 at one year, so the band comes from a CIRCULAR
  MOVING-BLOCK bootstrap with 1-yr blocks on the weekly subsample.
* The band is parameter uncertainty of the wave (where it is), not the
  percentile's scatter around it; the hindcast RMSE reports that scatter.
* Seeds are fixed so every worker and every reload draws the same band.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

__all__ = [
    "CycleFit", "fit_cycle", "wave", "bootstrap_params", "band", "turns",
    "turn_timing", "hindcast", "fan_prices",
    "PMIN", "PMAX", "B_DEFAULT", "BLOCK_WEEKS", "SEED", "WEAK_R2",
    "CENSOR_WINDOW", "CYCLE_MODEL_KEYS", "CYCLE_MODEL_LABELS",
]

PMIN = 2.0            # shortest period searched (yr)
PMAX = 8.0            # longest period searched (yr)
B_DEFAULT = 150       # bootstrap replicates
BLOCK_WEEKS = 52      # block length on the weekly subsample (= 1 yr)
STEP = 7              # weekly subsample of the daily series
SEED = 20090725       # fixed: same band on every worker and reload
WEAK_R2 = 0.20        # below this the wave is reported as not meaningful
CENSOR_WINDOW = ("2019-04-15", "2020-02-15")   # operator's study window

#: Models offered in the picker (the navbar ticker's cycle, same order).
CYCLE_MODEL_KEYS: tuple[str, ...] = (
    "qr", "bub", "pl", "spl", "lp3", "cfg_1d_1u", "ecfg_1d_1u", "pca",
    "grdy", "ef")
#: Labels for keys whose legend_name is not reader-facing ("1D_1U").
CYCLE_MODEL_LABELS: dict[str, str] = {
    "lp3": "LPPL₃", "cfg_1d_1u": "HybPPL", "ecfg_1d_1u": "EPPL"}


@dataclass(frozen=True)
class CycleFit:
    c: float
    A: float
    f: float          # cycles per year
    phi: float        # radians
    r2: float
    n: int

    @property
    def period(self) -> float:
        return 1.0 / self.f


def wave(c, A, f, phi, x):
    """c + A·cos(2πf·x + φ). Scalars -> x.shape; (B,) arrays -> (B, len(x))."""
    x = np.asarray(x, float)
    c, A, f, phi = (np.asarray(v, float) for v in (c, A, f, phi))
    if c.ndim == 0:
        return c + A * np.cos(2 * np.pi * f * x + phi)
    return (c[:, None] + A[:, None]
            * np.cos(2 * np.pi * f[:, None] * x[None, :] + phi[:, None]))


def _solve(t, y, f):
    """Exact LSQ for fixed f: returns (coef[c, a, b], sse)."""
    w = 2 * np.pi * f * t
    X = np.column_stack([np.ones_like(t), np.cos(w), np.sin(w)])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ coef
    return coef, float(r @ r)


def _grid_best(t, y, fs):
    """Best grid frequency via vectorised 3x3 normal equations."""
    W = 2 * np.pi * np.outer(fs, t)
    C, S = np.cos(W), np.sin(W)
    n = float(t.size)
    sc, ss = C.sum(1), S.sum(1)
    scc, sss, scs = (C * C).sum(1), (S * S).sum(1), (C * S).sum(1)
    sy, scy, ssy = y.sum(), C @ y, S @ y
    G = np.empty((fs.size, 3, 3))
    G[:, 0, 0] = n
    G[:, 0, 1] = G[:, 1, 0] = sc
    G[:, 0, 2] = G[:, 2, 0] = ss
    G[:, 1, 1] = scc
    G[:, 1, 2] = G[:, 2, 1] = scs
    G[:, 2, 2] = sss
    rhs = np.stack([np.full(fs.size, sy), scy, ssy], axis=1)
    coef = np.linalg.solve(G, rhs[..., None])[..., 0]          # (nf, 3)
    # SSE = y·y − coef·rhs  (normal-equation identity)
    sse = float(y @ y) - np.einsum("ij,ij->i", coef, rhs)
    return int(np.argmin(sse))


def fit_cycle(t, y, *, pmin=PMIN, pmax=PMAX, ngrid=400) -> CycleFit:
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    fs = np.linspace(1.0 / pmax, 1.0 / pmin, ngrid)
    k = _grid_best(t, y, fs)
    df = fs[1] - fs[0]
    lo, hi = max(fs[k] - df, fs[0]), min(fs[k] + df, fs[-1])
    r = minimize_scalar(lambda f: _solve(t, y, f)[1], bounds=(lo, hi),
                        method="bounded", options={"xatol": 1e-7})
    f = float(r.x)
    coef, sse = _solve(t, y, f)
    # the grid node can beat a refine that wandered to a bound
    coef_k, sse_k = _solve(t, y, fs[k])
    if sse_k < sse:
        f, coef, sse = float(fs[k]), coef_k, sse_k
    sst = float(((y - y.mean()) ** 2).sum())
    return CycleFit(c=float(coef[0]), A=float(np.hypot(coef[1], coef[2])),
                    f=f, phi=float(np.arctan2(-coef[2], coef[1])),
                    r2=1.0 - sse / sst if sst > 0 else 0.0, n=int(t.size))


def bootstrap_params(t, y, fit: CycleFit, *, B=B_DEFAULT, block=BLOCK_WEEKS,
                     step=STEP, seed=SEED, ngrid=200) -> np.ndarray:
    """(B, 4) array of replicate (c, A, f, phi): circular moving-block
    bootstrap of the central fit's residuals on the weekly subsample."""
    tw = np.asarray(t, float)[::step]
    yw = np.asarray(y, float)[::step]
    base = wave(fit.c, fit.A, fit.f, fit.phi, tw)
    res = yw - base
    n = tw.size
    block = max(1, min(int(block), n))
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    out = np.empty((B, 4))
    for b in range(B):
        starts = rng.integers(0, n, nb)
        idx = ((starts[:, None] + np.arange(block)[None, :]).ravel()[:n]) % n
        fb = fit_cycle(tw, base + res[idx], ngrid=ngrid)
        out[b] = (fb.c, fb.A, fb.f, fb.phi)
    return out


def band(params, x, lo=10.0, hi=90.0):
    """Pointwise lo/hi percentile of the replicate curves, clipped to 0-100."""
    curves = wave(params[:, 0], params[:, 1], params[:, 2], params[:, 3], x)
    b_lo, b_hi = np.percentile(curves, [lo, hi], axis=0)
    return np.clip(b_lo, 0.0, 100.0), np.clip(b_hi, 0.0, 100.0)


def turns(fit: CycleFit, t0: float, t1: float):
    """Peaks and troughs of the central wave in [t0, t1], analytic, by t.

    Peak where 2πf·t + φ = 2πk, trough where it equals π + 2πk (A > 0)."""
    out = []
    w = 2 * np.pi * fit.f
    for offset, kind, val in ((0.0, "peak", fit.c + fit.A),
                              (np.pi, "trough", fit.c - fit.A)):
        k0 = int(np.floor((w * t0 + fit.phi - offset) / (2 * np.pi)))
        k1 = int(np.ceil((w * t1 + fit.phi - offset) / (2 * np.pi)))
        for k in range(k0, k1 + 1):
            tk = (2 * np.pi * k + offset - fit.phi) / w
            if t0 <= tk <= t1:
                out.append((float(tk), float(val), kind))
    out.sort(key=lambda r: r[0])
    return out


def turn_timing(params, t_e: float, kind: str, lo=10.0, hi=90.0):
    """lo-hi percentile of the date of each replicate's same-kind turn
    nearest t_e."""
    offset = 0.0 if kind == "peak" else np.pi
    w = 2 * np.pi * params[:, 2]
    k = np.round((w * t_e + params[:, 3] - offset) / (2 * np.pi))
    tb = (2 * np.pi * k + offset - params[:, 3]) / w
    a, b = np.percentile(tb, [lo, hi])
    return float(min(a, t_e)), float(max(b, t_e))


def hindcast(t, y, *, horizon=2.0, every=0.5, lookback=10.0, min_span=5.0,
             step=STEP) -> dict:
    """Rolling-origin check: fit on data up to each cutoff, score the next
    ``horizon`` years against the flat (in-sample mean) forecast."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    last = t[-1]
    rw, rf = [], []
    for cut in np.arange(last - lookback, last - horizon + 1e-9, every):
        m_in = t <= cut
        if not m_in.any() or (t[m_in][-1] - t[0]) < min_span:
            continue
        m_out = (t > cut) & (t <= cut + horizon)
        if not m_out.any():
            continue
        f = fit_cycle(t[m_in][::step], y[m_in][::step], ngrid=200)
        ew = y[m_out] - wave(f.c, f.A, f.f, f.phi, t[m_out])
        ef = y[m_out] - y[m_in].mean()
        rw.append(float(np.sqrt((ew ** 2).mean())))
        rf.append(float(np.sqrt((ef ** 2).mean())))
    if not rw:
        return dict(n=0, rmse_wave=None, rmse_flat=None, win_rate=None)
    rw, rf = np.array(rw), np.array(rf)
    return dict(n=int(rw.size), rmse_wave=float(rw.mean()),
                rmse_flat=float(rf.mean()), win_rate=float((rw < rf).mean()))


def fan_prices(logfan, qs, pct):
    """USD the fan puts at ``pct`` (0-100) on each date.

    logfan is (n_q, n_t) log10 USD at quantiles ``qs`` (increasing). Each
    column is SORTED first — monotone rearrangement (Chernozhukov,
    Fernández-Val & Galichon 2010) — so a crossed fan still maps forward to a
    ranking; then linear interpolation in quantile, clipped to the outer
    quantiles.
    """
    lf = np.sort(np.asarray(logfan, float), axis=0)
    qs = np.asarray(qs, float)
    q = np.clip(np.asarray(pct, float) / 100.0, qs[0], qs[-1])
    out = np.empty(lf.shape[1])
    for j in range(lf.shape[1]):
        out[j] = np.interp(q[j], qs, lf[:, j])
    return 10.0 ** out
