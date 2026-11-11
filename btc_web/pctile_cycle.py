"""Single-element cycle fit to a model's percentile series (Tab 1 Percentile pill).

One oscillator, ``y = c + A·cos(2πf·g(t) + φ)``, fitted to the percentile
(0-100) at which each daily close sat inside ONE model's quantile fan. ``g`` is
the element's time coordinate: calendar years since the 2009-07-25 origin
(``g = t``) or log-time (``g = ln t``, log-periodic) — the two spaces of the
repo's oscillator dictionary (GreedyModel ``_BASIS``). The operator picks the
element; the default is a calendar sinusoid searched over 2-8 yr periods.
Everything here is pure numpy/scipy so tests and the figure builder can import
it without Dash.

Why each choice (measured 2026-11-10, spec
``docs/superpowers/specs/2026-11-10-percentile-cycle-fit-design.md``):

* For a fixed frequency the model is linear in (c, a, b) with
  a·cos + b·sin = A·cos(·+φ), so the frequency is scanned on a grid and each
  node solved exactly — no starting guess can strand the fit.
* Element choice (tools/pctile_cycle_basis_explore.py): calendar 2-8 yr is
  best in- AND out-of-sample for QR / PL / SatPL; damped and log-time elements
  gain in-sample but lose out-of-sample; for BM and the oscillator models no
  single element beats a flat forecast out-of-sample — which is why
  :func:`is_weak` gates on the hindcast, not on R² alone.
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
    "turn_timing", "hindcast", "fan_prices", "is_weak", "beats_flat",
    "likely_band", "available_models",
    "PMIN", "PMAX", "B_DEFAULT", "BLOCK_WEEKS", "SEED", "WEAK_R2",
    "CENSOR_WINDOW", "CYCLE_MODEL_KEYS", "BASES", "DEFAULT_BASIS",
    "HORIZONS", "DEFAULT_HORIZON",
]

PMIN = 2.0            # shortest period of the default calendar element (yr)
PMAX = 8.0            # longest period of the default calendar element (yr)
B_DEFAULT = 150       # bootstrap replicates
BLOCK_WEEKS = 52      # block length on the weekly subsample (= 1 yr)
STEP = 7              # weekly subsample of the daily series
SEED = 20090725       # fixed: same band on every worker and reload
WEAK_R2 = 0.20        # below this the wave is reported as not meaningful
CENSOR_WINDOW = ("2019-04-15", "2020-02-15")   # operator's study window

#: Models offered in the picker: the SAME keys bub-model-show uses (masters,
#: not the ticker's config variants), so a picked model and its line in the
#: Percentile view are one series under one name (legend_name).
CYCLE_MODEL_KEYS: tuple[str, ...] = (
    "qr", "bub", "pl", "spl", "lppl", "hybppl", "eppl", "pca", "grdy", "ef")
HORIZONS: tuple[int, ...] = (10, 20, 30)


def available_models(price_models) -> list[str]:
    """Picker keys usable on this host, in picker order: registered, quantized,
    with a fan. The ONE filter for the layout options, the callback coercion
    and the figure builder (``ef`` needs model_data_ef.pkl)."""
    out = []
    for k in CYCLE_MODEL_KEYS:
        m = price_models.get(k)
        if m is not None and getattr(m, "quantized", False) and getattr(m, "quantiles", None):
            out.append(k)
    return out
DEFAULT_HORIZON = 20

#: Basis elements on offer: key -> (space, f_min, f_max, label). f is cycles
#: per unit of g: per year for "cal", per unit of ln t for "log" (so the log
#: element's angular frequency is ω = 2πf; ω 1-30 brackets every LPPL fit).
BASES: dict[str, tuple[str, float, float, str]] = {
    "cal": ("cal", 1.0 / PMAX, 1.0 / PMIN, "calendar 2–8 yr"),
    "calw": ("cal", 1.0 / 16.0, 1.0 / 0.75, "calendar 0.75–16 yr"),
    "log": ("log", 1.0 / (2 * np.pi), 30.0 / (2 * np.pi), "log-time"),
}
DEFAULT_BASIS = "cal"


@dataclass(frozen=True)
class CycleFit:
    c: float
    A: float
    f: float          # cycles per unit of g (per yr for "cal", per ln t for "log")
    phi: float        # radians
    r2: float
    n: int
    basis: str = DEFAULT_BASIS
    at_limit: bool = False    # refined f within one grid step of the range end

    @property
    def space(self) -> str:
        return BASES[self.basis][0]

    @property
    def period(self) -> float:
        """Calendar period in years ("cal"), or for "log" the scale ratio
        λ = e^(1/f) between successive same-kind turns (t_next = λ·t)."""
        return 1.0 / self.f if self.space == "cal" else float(np.exp(1.0 / self.f))


def _g(x, space):
    x = np.asarray(x, float)
    return x if space == "cal" else np.log(np.maximum(x, 1e-9))


def _g_inv(u, space):
    return u if space == "cal" else np.exp(u)


def wave(c, A, f, phi, x, space="cal"):
    """c + A·cos(2πf·g(x) + φ). Scalars -> x.shape; (B,) arrays -> (B, len(x))."""
    x = _g(x, space)
    c, A, f, phi = (np.asarray(v, float) for v in (c, A, f, phi))
    if c.ndim == 0:
        return c + A * np.cos(2 * np.pi * f * x + phi)
    return (c[:, None] + A[:, None]
            * np.cos(2 * np.pi * f[:, None] * x[None, :] + phi[:, None]))


def _solve(u, y, f):
    """Exact LSQ for fixed f on coordinate u: returns (coef[c, a, b], sse)."""
    w = 2 * np.pi * f * u
    X = np.column_stack([np.ones_like(u), np.cos(w), np.sin(w)])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ coef
    return coef, float(r @ r)


def _grid_best(u, y, fs):
    """Best grid frequency via vectorised 3x3 normal equations."""
    W = 2 * np.pi * np.outer(fs, u)
    C, S = np.cos(W), np.sin(W)
    n = float(u.size)
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


def fit_cycle(t, y, *, basis=DEFAULT_BASIS, ngrid=400) -> CycleFit:
    space, fmin, fmax, _ = BASES[basis]
    u = _g(t, space)
    y = np.asarray(y, float)
    fs = np.linspace(fmin, fmax, ngrid)
    k = _grid_best(u, y, fs)
    df = fs[1] - fs[0]
    lo, hi = max(fs[k] - df, fs[0]), min(fs[k] + df, fs[-1])
    r = minimize_scalar(lambda f: _solve(u, y, f)[1], bounds=(lo, hi),
                        method="bounded", options={"xatol": 1e-7})
    f = float(r.x)
    coef, sse = _solve(u, y, f)
    # the grid node can beat a refine that wandered to a bound
    coef_k, sse_k = _solve(u, y, fs[k])
    if sse_k < sse:
        f, coef, sse = float(fs[k]), coef_k, sse_k
    sst = float(((y - y.mean()) ** 2).sum())
    return CycleFit(c=float(coef[0]), A=float(np.hypot(coef[1], coef[2])),
                    f=f, phi=float(np.arctan2(-coef[2], coef[1])),
                    r2=1.0 - sse / sst if sst > 0 else 0.0, n=int(u.size),
                    basis=basis,
                    at_limit=bool(f <= fs[0] + df or f >= fs[-1] - df))


def bootstrap_params(t, y, fit: CycleFit, *, B=B_DEFAULT, block=BLOCK_WEEKS,
                     step=STEP, seed=SEED, ngrid=200) -> np.ndarray:
    """(B, 4) array of replicate (c, A, f, phi): circular moving-block
    bootstrap of the central fit's residuals on the weekly subsample, each
    replicate refit with the same basis element."""
    tw = np.asarray(t, float)[::step]
    yw = np.asarray(y, float)[::step]
    base = wave(fit.c, fit.A, fit.f, fit.phi, tw, fit.space)
    res = yw - base
    n = tw.size
    block = max(1, min(int(block), n))
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    out = np.empty((B, 4))
    for b in range(B):
        starts = rng.integers(0, n, nb)
        idx = ((starts[:, None] + np.arange(block)[None, :]).ravel()[:n]) % n
        fb = fit_cycle(tw, base + res[idx], basis=fit.basis, ngrid=ngrid)
        out[b] = (fb.c, fb.A, fb.f, fb.phi)
    return out


def _block_path(res, length, block, rng):
    """One circular moving-block resample of ``res``, ``length`` long, with a
    random block phase (so different paths change block in different weeks)."""
    n = res.size
    block = max(1, min(int(block), n))
    phase = int(rng.integers(0, block))
    total = length + phase
    nb = int(np.ceil(total / block))
    starts = rng.integers(0, n, nb)
    idx = ((starts[:, None] + np.arange(block)[None, :]).ravel()[:total]) % n
    return res[idx[phase:]]


def likely_band(params, t, y, fit: CycleFit, grid, *, lo=10.0, hi=90.0,
                paths=8, block=BLOCK_WEEKS, step=STEP, seed=SEED + 1):
    """Predictive envelope: pointwise lo/hi of (replicate curve + a
    block-resampled residual path) on ``grid`` (a weekly grid), clipped to
    0-100. The curve carries where the wave is; the residual path carries how
    far the percentile strays from it — so in-sample it covers ~(hi-lo) %.

    ``paths`` residual paths per replicate, each with its own random block
    phase: with one synchronised path per replicate every block boundary fell
    in the same week and the band edge stepped 12-15 pp (review r2 I-4)."""
    tw = np.asarray(t, float)[::step]
    res = np.asarray(y, float)[::step] - wave(fit.c, fit.A, fit.f, fit.phi,
                                               tw, fit.space)
    curves = wave(params[:, 0], params[:, 1], params[:, 2], params[:, 3],
                  grid, fit.space)
    rng = np.random.default_rng(seed)
    L = curves.shape[1]
    out = np.empty((curves.shape[0] * paths, L))
    for b in range(curves.shape[0]):
        for k in range(paths):
            out[b * paths + k] = curves[b] + _block_path(res, L, block, rng)
    b_lo, b_hi = np.percentile(out, [lo, hi], axis=0)
    return np.clip(b_lo, 0.0, 100.0), np.clip(b_hi, 0.0, 100.0)


def band(params, x, lo=10.0, hi=90.0, space="cal"):
    """Pointwise lo/hi percentile of the replicate curves, clipped to 0-100."""
    curves = wave(params[:, 0], params[:, 1], params[:, 2], params[:, 3], x,
                  space)
    b_lo, b_hi = np.percentile(curves, [lo, hi], axis=0)
    return np.clip(b_lo, 0.0, 100.0), np.clip(b_hi, 0.0, 100.0)


def turns(fit: CycleFit, t0: float, t1: float):
    """Peaks and troughs of the central wave in [t0, t1], analytic, by t.

    Peak where 2πf·g(t) + φ = 2πk, trough where it equals π + 2πk (A > 0);
    g is monotone, so solving in g and mapping back is exact."""
    out = []
    w = 2 * np.pi * fit.f
    u0, u1 = float(_g(t0, fit.space)), float(_g(t1, fit.space))
    for offset, kind, val in ((0.0, "peak", fit.c + fit.A),
                              (np.pi, "trough", fit.c - fit.A)):
        k0 = int(np.floor((w * u0 + fit.phi - offset) / (2 * np.pi)))
        k1 = int(np.ceil((w * u1 + fit.phi - offset) / (2 * np.pi)))
        for k in range(k0, k1 + 1):
            tk = float(_g_inv((2 * np.pi * k + offset - fit.phi) / w, fit.space))
            if t0 <= tk <= t1:
                out.append((tk, float(val), kind))
    out.sort(key=lambda r: r[0])
    return out


def turn_timing(params, t_anchor: float, j: int, kind: str, t_e: float,
                lo=10.0, hi=90.0, space="cal"):
    """lo-hi percentile of each replicate's turn date for the central turn
    ``t_e``, matched by CYCLE INDEX from an in-sample anchor.

    ``t_anchor`` is a central same-kind turn inside the data (the last one),
    where replicates agree; each replicate's turn nearest it is index 0 and
    its ``j``-th same-kind turn after that is the match (``t_e`` is the
    central fit's j-th). Anchoring in-sample keeps a turn that falls just after
    the last close from matching replicates' NEXT cycle, and keeps a wide
    spread far out from being truncated at half a period. Clamped to contain
    ``t_e``."""
    offset = 0.0 if kind == "peak" else np.pi
    w = 2 * np.pi * params[:, 2]
    u_a = float(_g(t_anchor, space))
    k = np.round((w * u_a + params[:, 3] - offset) / (2 * np.pi)) + j
    tb = _g_inv((2 * np.pi * k + offset - params[:, 3]) / w, space)
    a, b = np.percentile(tb, [lo, hi])
    return float(min(a, t_e)), float(max(b, t_e))


def hindcast(t, y, *, horizon=2.0, every=1.0 / 12, lookback=10.0,
             min_span=5.0, step=STEP, basis=DEFAULT_BASIS, y_at=None,
             fit_mask=None) -> dict:
    """Rolling-origin check: fit on data up to each cutoff, score the next
    ``horizon`` years against the flat (in-sample mean) forecast.

    Cutoffs are monthly (``every``) so a daily price update moves the verdict
    by a sliver, not across a half-yearly cutoff grid (review r2 I-1).
    ``fit_mask`` (optional, aligned with ``t``) removes days from FITTING only
    (the operator's censor window); scoring always uses every day, so leaving
    out a window that misfits cannot flatter the score (review r2 I-3).

    ``y_at(cut)`` (optional) returns the series as it would have been computed
    at the cutoff (against an as-of fan), aligned with ``t``, or None to skip
    the cutoff; without it the series ``y`` — computed against today's fan —
    is used, which leaks the future into the scored values (optimistic)."""
    t = np.asarray(t, float)
    y0 = np.asarray(y, float)
    fm = np.ones(t.size, bool) if fit_mask is None else np.asarray(fit_mask, bool)
    last = t[-1]
    rw, rf = [], []
    for cut in np.arange(last - lookback, last - horizon + 1e-9, every):
        m_in = (t <= cut) & fm
        if not m_in.any() or (t[m_in][-1] - t[0]) < min_span:
            continue
        m_out = (t > cut) & (t <= cut + horizon)
        if not m_out.any():
            continue
        y = y0 if y_at is None else y_at(cut)
        if y is None:
            continue
        y = np.asarray(y, float)
        if not np.all(np.isfinite(y[m_in | m_out])):
            continue
        f = fit_cycle(t[m_in][::step], y[m_in][::step], basis=basis, ngrid=200)
        ew = y[m_out] - wave(f.c, f.A, f.f, f.phi, t[m_out], f.space)
        ef = y[m_out] - y[m_in].mean()
        rw.append(float(np.sqrt((ew ** 2).mean())))
        rf.append(float(np.sqrt((ef ** 2).mean())))
    if not rw:
        return dict(n=0, rmse_wave=None, rmse_flat=None, win_rate=None)
    rw, rf = np.array(rw), np.array(rf)
    return dict(n=int(rw.size), rmse_wave=float(rw.mean()),
                rmse_flat=float(rf.mean()), win_rate=float((rw < rf).mean()))


def is_weak(fit: CycleFit) -> bool:
    """Too weak to extrapolate: R² < WEAK_R2, or the period sits at the edge
    of the searched range (an artefact of the window, not an estimate).

    The hindcast does NOT decide this: QR's out-of-sample edge is ~1.5 pp on
    ~4 independent windows, so a binary gate on it flips with routine price
    updates (review r2 I-1). It is reported instead (:func:`beats_flat`)."""
    return fit.r2 < WEAK_R2 or fit.at_limit


def beats_flat(hind: dict):
    """True / False: mean hindcast RMSE of the wave below / not below the flat
    forecast's; None when there is no hindcast."""
    if not hind.get("n"):
        return None
    return hind["rmse_wave"] < hind["rmse_flat"]


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
