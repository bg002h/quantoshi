#!/usr/bin/env python3
"""Which SINGLE basis element best fits each model's percentile series?

    btc_venv/bin/python3 tools/pctile_cycle_basis_explore.py

Candidates are one element of the repo's oscillator dictionary (GreedyModel
_BASIS): y = c + A·D(t)·cos(ω·g(t) + φ), g ∈ {t (calendar), ln t (log)},
D ∈ {1, t^-d (hybrid)}. Each is scored in-sample (R²) and out-of-sample
(rolling-origin 2-yr hindcast vs a flat forecast) on the weekly series,
because picking an element by in-sample R² alone rewards flexibility.
Scratch measurement for the percentile cycle-fit spec; not used at runtime.
"""
import os
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "btc_web"))
sys.path.insert(0, str(REPO))
os.environ.setdefault("TESTING", "1")

import app  # noqa: E402,F401
import _app_ctx  # noqa: E402
from figures.percentile import _percentile_series  # noqa: E402

ELEMENTS = {
    # name: (g, omega grid, damping exponents d for env t^-d)
    "cal 2-8yr": ("cal", 2 * np.pi / np.linspace(2.0, 8.0, 300), [0.0]),
    "cal 0.75-16yr": ("cal", 2 * np.pi / np.geomspace(0.75, 16.0, 500), [0.0]),
    "cal 2-8yr damped": ("cal", 2 * np.pi / np.linspace(2.0, 8.0, 200),
                         [-0.6, -0.3, 0.3, 0.6, 1.0]),
    "log": ("log", np.linspace(1.0, 30.0, 600), [0.0]),
    "log damped": ("log", np.linspace(1.0, 30.0, 300), [-0.6, -0.3, 0.3, 0.6, 1.0]),
}


def fit(t, y, g, omegas, ds):
    x = t if g == "cal" else np.log(t)
    best = (np.inf, None)
    for d in ds:
        env = t ** (-d)
        for om in omegas:
            X = np.column_stack([np.ones_like(t), env * np.cos(om * x), env * np.sin(om * x)])
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            r = y - X @ coef
            sse = float(r @ r)
            if sse < best[0]:
                best = (sse, (om, d, coef))
    return best


def predict(t, g, p):
    om, d, coef = p
    x = t if g == "cal" else np.log(t)
    env = t ** (-d)
    return coef[0] + env * (coef[1] * np.cos(om * x) + coef[2] * np.sin(om * x))


def main():
    M = _app_ctx.M
    t_all = np.asarray(M.price_years, float)
    keep = t_all >= 1.0
    t = t_all[keep][::7]
    px = np.asarray(M.price_prices, float)[keep][::7]
    keys = ["qr", "bub", "pl", "spl", "lp3", "cfg_1d_1u", "ecfg_1d_1u", "pca", "grdy", "ef"]
    print(f"{'model':11s} {'element':17s} {'R2':>6s} {'period/ω':>9s} {'d':>5s} {'hc win':>7s} {'RMSE w/f':>11s}")
    for k in keys:
        mdl = _app_ctx.PRICE_MODELS.get(k)
        if mdl is None:
            continue
        y = np.asarray(_percentile_series(mdl, t, px, sigma_mode="resqr"), float)
        sst = float(((y - y.mean()) ** 2).sum())
        for name, (g, oms, ds) in ELEMENTS.items():
            sse, p = fit(t, y, g, oms, ds)
            r2 = 1 - sse / sst
            # 2-yr rolling-origin hindcast
            wins, rw, rf = [], [], []
            last = t[-1]
            for cut in np.arange(last - 10, last - 2 + 1e-9, 0.5):
                mi = t <= cut
                if t[mi][-1] - t[0] < 5:
                    continue
                mo = (t > cut) & (t <= cut + 2)
                _, pc = fit(t[mi], y[mi], g, oms[::3], ds)
                ew = y[mo] - predict(t[mo], g, pc)
                ef = y[mo] - y[mi].mean()
                rw.append(np.sqrt((ew ** 2).mean()))
                rf.append(np.sqrt((ef ** 2).mean()))
            rw, rf = np.array(rw), np.array(rf)
            om, d, _ = p
            per = (2 * np.pi / om) if g == "cal" else om
            print(f"{k:11s} {name:17s} {r2:6.3f} {per:9.3f} {d:5.2f} {np.mean(rw < rf):7.0%} {rw.mean():5.1f}/{rf.mean():4.1f}")
        print()


if __name__ == "__main__":
    main()
