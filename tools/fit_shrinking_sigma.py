#!/usr/bin/env python3
"""Fit time-dependent σ(t) = σ₀·t^(-α) for all constant-σ models.

Uses parallel processing to fit all models simultaneously.
Optionally updates btc_core.py with fitted parameters.

Usage:
    btc_venv/bin/python3 tools/fit_shrinking_sigma.py           # dry run
    btc_venv/bin/python3 tools/fit_shrinking_sigma.py --update   # write to btc_core/
"""

import sys, re, pathlib
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

GENESIS_YR = 2009.56  # 2009-07-25


def _load_data():
    """Load price data, return (t_yr, log10_price) arrays."""
    import pandas as pd
    df = pd.read_csv(ROOT / "BitcoinPricesDaily.csv")
    col = "Close" if "Close" in df.columns else df.columns[-1]
    date_col = "Date" if "Date" in df.columns else df.columns[0]
    from datetime import datetime
    df["date"] = pd.to_datetime(df[date_col])
    df["price"] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["price"])
    df = df[df["price"] > 0].copy()
    # t in years since genesis
    df["t_yr"] = (df["date"] - pd.Timestamp("2009-07-25")).dt.days / 365.25
    df = df[df["t_yr"] >= 1.0].copy()
    return df["t_yr"].values, np.log10(df["price"].values)


def _fit_sigma_params(t, residuals):
    """Fit σ(t) = σ₀·t^(-α) to |residuals| using log-linear regression.

    In log space: log|r| = log(σ₀) - α·log(t)
    This is simple OLS on (log(t), log|r|).

    Also fit asymmetric: separate σ for above/below median.
    """
    abs_r = np.abs(residuals)
    # Remove zeros
    mask = abs_r > 1e-10
    t_m, r_m = t[mask], abs_r[mask]

    # Windowed σ estimation (more robust than pointwise)
    # Sort by t, compute rolling std in windows
    order = np.argsort(t_m)
    t_s, r_s = t_m[order], r_m[order]

    # Use overlapping windows of ~500 points
    window = min(500, len(t_s) // 4)
    if window < 50:
        # Not enough data, return constant σ
        sigma0 = float(np.std(residuals))
        return sigma0, 0.0, sigma0, 0.0

    n_windows = max(10, len(t_s) // (window // 2))
    edges = np.linspace(0, len(t_s), n_windows + 1, dtype=int)
    t_mid = []
    sigma_win = []
    sigma_up_win = []
    sigma_dn_win = []
    for i in range(n_windows):
        lo, hi = edges[i], edges[i + 1]
        if hi - lo < 20:
            continue
        chunk_t = t_s[lo:hi]
        chunk_r = residuals[order][lo:hi]  # signed residuals
        t_mid.append(np.median(chunk_t))
        sigma_win.append(np.std(chunk_r))
        # Asymmetric
        up = chunk_r[chunk_r >= 0]
        dn = chunk_r[chunk_r < 0]
        sigma_up_win.append(np.std(up) if len(up) > 5 else np.std(chunk_r))
        sigma_dn_win.append(np.std(np.abs(dn)) if len(dn) > 5 else np.std(chunk_r))

    t_mid = np.array(t_mid)
    sigma_win = np.array(sigma_win)
    sigma_up_win = np.array(sigma_up_win)
    sigma_dn_win = np.array(sigma_dn_win)

    # Fit log(σ) = log(σ₀) - α·log(t)
    def _fit_power(t_arr, s_arr):
        mask = (t_arr > 0) & (s_arr > 0)
        if mask.sum() < 3:
            return float(np.mean(s_arr)), 0.0
        lt = np.log(t_arr[mask])
        ls = np.log(s_arr[mask])
        from numpy.polynomial.polynomial import polyfit
        c = polyfit(lt, ls, 1)  # c[0]=intercept, c[1]=slope
        sigma0 = np.exp(c[0])
        alpha = -c[1]  # σ(t) = σ₀ · t^(-α), so log(σ) = log(σ₀) - α·log(t)
        return float(sigma0), float(alpha)

    sigma0, alpha = _fit_power(t_mid, sigma_win)
    sigma0_up, alpha_up = _fit_power(t_mid, sigma_up_win)
    sigma0_dn, alpha_dn = _fit_power(t_mid, sigma_dn_win)

    return sigma0, alpha, sigma0_up, alpha_up, sigma0_dn, alpha_dn


def _eval_model(model_name, t, lp):
    """Evaluate a model's median prediction and return residuals."""
    if model_name == "pl":
        # Power law: log10(P) = a + b·log10(t)
        from scipy.stats import linregress
        lt = np.log10(t)
        slope, intercept, _, _, _ = linregress(lt, lp)
        return lp - (intercept + slope * lt)

    elif model_name == "exp":
        # Exponential: log10(P) = a + b·t
        from scipy.stats import linregress
        slope, intercept, _, _, _ = linregress(t, lp)
        return lp - (intercept + slope * t)

    elif model_name == "lppl":
        from btc_core import LPPLModel
        t_safe = np.maximum(t, 0.1)
        pred = LPPLModel._A + LPPLModel._B * np.log10(t_safe) + \
               LPPLModel._C * t_safe ** (-LPPLModel._D) * \
               np.cos(LPPLModel._W * np.log(t_safe) + LPPLModel._PHI)
        return lp - pred

    elif model_name == "hybppl":
        from btc_core import HybPPLModel
        m = HybPPLModel
        t_safe = np.maximum(t, 0.1)
        pred = m._A + m._B * np.log10(t_safe)
        pred += m._C * t_safe ** (-m._D) * np.cos(m._W * np.log(t_safe) + m._PHI)
        pred += m._C2 * np.cos(m._W2 * t_safe + m._PHI2)
        return lp - pred

    elif model_name == "hybppl_dd":
        from btc_core import HybPPLDDModel
        m = HybPPLDDModel
        t_safe = np.maximum(t, 0.1)
        pred = m._A + m._B * np.log10(t_safe)
        pred += m._C1 * t_safe ** (-m._D1) * np.cos(m._W_log * np.log(t_safe) + m._PHI1)
        pred += m._C2 * t_safe ** (-m._D2) * np.cos(m._W_cal * t_safe + m._PHI2)
        return lp - pred

    elif model_name == "eppl":
        from btc_core import EntropyPPLModel
        m = EntropyPPLModel
        t_safe = np.maximum(t, 0.1)
        pred = m._A + m._B * np.log10(t_safe)

        def _entropy(w, tt):
            x = w * tt
            raw = -x * np.log(np.maximum(x, 1e-30))
            return np.maximum(raw, 0.0) / (1.0 / np.e)

        pred += m._C1 * _entropy(m._w1, t_safe) * np.cos(m._W1 * np.log(t_safe) + m._P1)
        pred += m._C3 * _entropy(m._w2, t_safe) * np.cos(m._W2 * np.log(t_safe) + m._P3)
        pred += m._C2 * np.cos(m._Wc1 * t_safe + m._P2)
        pred += m._C4 * np.cos(m._Wc2 * t_safe + m._P4)
        return lp - pred

    elif model_name == "grdy":
        from btc_core import GreedyModel
        m = GreedyModel
        t_safe = np.maximum(t, 0.1)
        ln_t = np.log(t_safe)
        pred = m._alpha + m._beta * np.log10(t_safe)
        # v3: generic basis — iterate over stored _BASIS tuple
        for term in m._BASIS:
            space, damping, freq, phase, weight, d_param = term
            arg = freq * (ln_t if space == "log" else t_safe)
            osc = np.sin(arg) if phase == "sin" else np.cos(arg)
            if damping == "none":
                env = 1.0
            elif damping == "hybrid":
                env = t_safe ** (-d_param)
            else:
                env = m._entropy_env(t_safe, d_param)
            pred = pred + weight * env * osc
        return lp - pred

    elif model_name == "gomp":
        from btc_core import GompertzModel
        m = GompertzModel
        t_safe = np.maximum(t, 0.1)
        pred = m._K * np.exp(-np.exp(-m._r * (t_safe - m._t0)))
        return lp - pred

    elif model_name == "bpl":
        from btc_core import BrokenPowerLawModel
        m = BrokenPowerLawModel
        t_safe = np.maximum(t, 0.1)
        lt = np.log10(t_safe)
        a2 = m._a1 + (m._b1 - m._b2) * np.log10(m._t_break)
        pred = np.where(t_safe < m._t_break,
                        m._a1 + m._b1 * lt,
                        a2 + m._b2 * lt)
        return lp - pred

    elif model_name == "plo":
        from btc_core import OffsetPowerLawModel
        m = OffsetPowerLawModel
        t_safe = np.maximum(t + m._c, 1e-9)
        pred = m._A + m._m * np.log10(t_safe)
        return lp - pred

    elif model_name == "sexp":
        from btc_core import StretchedExponentialModel
        m = StretchedExponentialModel
        t_safe = np.maximum(t, 1e-9)
        pred = m._A + m._B * np.power(t_safe, m._beta)
        return lp - pred

    elif model_name == "logi":
        from btc_core import LogisticSCurveModel
        m = LogisticSCurveModel
        pred = m._K / (1.0 + np.exp(-m._r * (t - m._t0)))
        return lp - pred

    else:
        raise ValueError(f"Unknown model: {model_name}")


def _fit_one_model(args):
    """Worker function for parallel fitting."""
    model_name, t, lp = args
    try:
        residuals = _eval_model(model_name, t, lp)
        results = _fit_sigma_params(t, residuals)
        sigma_const = float(np.std(residuals))
        return model_name, sigma_const, results
    except Exception as e:
        return model_name, None, str(e)


def main():
    update = "--update" in sys.argv

    print("Loading price data ...")
    t, lp = _load_data()
    print(f"  {len(t)} observations, t = {t.min():.2f} – {t.max():.2f} yr")

    models = ["pl", "exp", "lppl", "hybppl", "hybppl_dd", "eppl", "grdy", "gomp", "bpl", "plo", "sexp", "logi"]

    print(f"\nFitting σ(t) = σ₀·t^(-α) for {len(models)} models (parallel) ...\n")

    # Parallel fitting
    results = {}
    with ProcessPoolExecutor(max_workers=min(len(models), 8)) as pool:
        futures = {pool.submit(_fit_one_model, (m, t, lp)): m for m in models}
        for future in as_completed(futures):
            model_name, sigma_const, fit_result = future.result()
            if sigma_const is None:
                print(f"  {model_name}: FAILED — {fit_result}")
                continue
            results[model_name] = (sigma_const, fit_result)

    # Display results
    print(f"{'Model':<12} {'σ_const':>8} {'σ₀':>8} {'α':>8} {'σ₀_up':>8} {'α_up':>8} {'σ₀_dn':>8} {'α_dn':>8}")
    print("─" * 80)
    for m in models:
        if m not in results:
            continue
        sigma_const, fit = results[m]
        if len(fit) == 4:
            s0, a, s0u, au = fit
            print(f"{m:<12} {sigma_const:>8.4f} {s0:>8.4f} {a:>8.4f} {s0u:>8.4f} {au:>8.4f}   —        —")
        else:
            s0, a, s0u, au, s0d, ad = fit
            print(f"{m:<12} {sigma_const:>8.4f} {s0:>8.4f} {a:>8.4f} {s0u:>8.4f} {au:>8.4f} {s0d:>8.4f} {ad:>8.4f}")

    # Show improvement: σ at t=2yr vs t=16yr
    print(f"\n{'Model':<12} {'σ(t=2yr)':>10} {'σ(t=16yr)':>10} {'ratio':>8} {'const σ':>10}")
    print("─" * 55)
    for m in models:
        if m not in results:
            continue
        sigma_const, fit = results[m]
        s0, a = fit[0], fit[1]
        s_early = s0 * 2.0 ** (-a)
        s_late = s0 * 16.0 ** (-a)
        print(f"{m:<12} {s_early:>10.4f} {s_late:>10.4f} {s_early/s_late:>8.2f}× {sigma_const:>10.4f}")

    if not update:
        print("\nDry run. Pass --update to write parameters to btc_core.py")
        return

    # ── Update btc_core/ submodules ────────────────────────────────────
    # After the btc_core.py → btc_core/ package split, each class lives in
    # a specific submodule; find-and-replace reads/writes each file separately.
    hard_coded = {
        "grdy": ("GreedyModel",      ROOT / "btc_core" / "_basis.py"),
        "eppl": ("EntropyPPLModel",  ROOT / "btc_core" / "_hybppl_eppl.py"),
    }

    for model_key, (class_name, core_path) in hard_coded.items():
        if model_key not in results:
            continue
        _, fit = results[model_key]
        s0, a, s0u, au, s0d, ad = fit

        print(f"\nUpdating {core_path.relative_to(ROOT)} ...")
        src = core_path.read_text()

        # Target the existing 4-line block inside the class body so each
        # refit REPLACES the fitted params in place. The prior pattern
        # targeted `_sigma = X` (a backward-compat scalar) and appended 4
        # new lines, which produced duplicate _sigma0_up/_alpha_*/_sigma0_down
        # definitions on every run (last-write-wins → old values silently
        # kept winning, new fitted params dropped).
        pattern = re.compile(
            rf"(class {class_name}[\s\S]*?)"
            rf"(    _sigma0_up\s*=\s*[\d.]+\s*\n"
            rf"\s*_alpha_up\s*=\s*[\d.]+\s*\n"
            rf"\s*_sigma0_down\s*=\s*[\d.]+\s*\n"
            rf"\s*_alpha_down\s*=\s*[\d.]+)"
        )
        match = pattern.search(src)
        if match:
            replacement = (
                f"    _sigma0_up   = {s0u:.6f}\n"
                f"    _alpha_up    = {au:.6f}\n"
                f"    _sigma0_down = {s0d:.6f}\n"
                f"    _alpha_down  = {ad:.6f}"
            )
            src = src[:match.start(2)] + replacement + src[match.end(2):]
            core_path.write_text(src)
            print(f"  Updated {class_name}")
        else:
            print(f"  WARNING: could not find 4-line sigma block for "
                  f"{class_name} in {core_path}; no replacement made.")

    print("Done.")


if __name__ == "__main__":
    main()
