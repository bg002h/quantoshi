#!/usr/bin/env python3
"""Fit all 36 HybPPL configuration variants via differential evolution.

Each config is: (n_log, n_cal, log_damps, cal_damps) where damps are
tuples of "d" (damped) or "u" (undamped) per frequency.

Model: log10(price) = A + B*log10(t) + sum(log_osc_i) + sum(cal_osc_i)

where:
  damped log:   C * t^(-D) * cos(W * ln(t) + PHI)
  undamped log: C * cos(W * ln(t) + PHI)
  damped cal:   C * t^(-D) * cos(W * t + PHI)
  undamped cal: C * cos(W * t + PHI)

Usage:
    btc_venv/bin/python3 tools/fit_all_hybppl_configs.py
    btc_venv/bin/python3 tools/fit_all_hybppl_configs.py --update  # write to btc_core.py
"""
import os
import sys
import json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from model_toolkit.data import load_prices
from scipy.optimize import differential_evolution

PI = np.pi


def config_key(n_log, n_cal, log_damps, cal_damps):
    """Build config key string, e.g. 'cfg_2dd_1u'."""
    def _spec(n, damps):
        if n == 0:
            return "0"
        return str(n) + "".join(damps)
    return f"cfg_{_spec(n_log, log_damps)}_{_spec(n_cal, cal_damps)}"


def build_model_fn(n_log, n_cal, log_damps, cal_damps):
    """Build a model evaluation function for this config.

    Returns (func, param_names, bounds).
    func(t, *params) -> log10(price)
    """
    param_names = ["A", "B"]
    bounds = [(-3, 1), (3, 7)]

    # Log-periodic terms
    for i in range(n_log):
        d = log_damps[i]
        suffix = str(i + 1) if n_log > 1 else ""
        param_names.append(f"C_log{suffix}")
        bounds.append((0.001, 3))
        param_names.append(f"W_log{suffix}")
        bounds.append((2, 80) if i > 0 else (2, 40))
        param_names.append(f"PHI_log{suffix}")
        bounds.append((-PI, PI))
        if d == "d":
            param_names.append(f"D_log{suffix}")
            bounds.append((0.01, 2))

    # Calendar terms
    for i in range(n_cal):
        d = cal_damps[i]
        suffix = str(i + 1) if n_cal > 1 else ""
        param_names.append(f"C_cal{suffix}")
        bounds.append((0.001, 2))
        param_names.append(f"W_cal{suffix}")
        bounds.append((0.5, 20) if i > 0 else (0.5, 10))
        param_names.append(f"PHI_cal{suffix}")
        bounds.append((-PI, PI))
        if d == "d":
            param_names.append(f"D_cal{suffix}")
            bounds.append((0, 2))

    def model_fn(t, *params):
        ts = np.maximum(t, 0.1)
        result = params[0] + params[1] * np.log10(ts)

        idx = 2
        # Log terms
        for i in range(n_log):
            C = params[idx]; idx += 1
            W = params[idx]; idx += 1
            PHI = params[idx]; idx += 1
            if log_damps[i] == "d":
                D = params[idx]; idx += 1
                result = result + C * ts**(-D) * np.cos(W * np.log(ts) + PHI)
            else:
                result = result + C * np.cos(W * np.log(ts) + PHI)

        # Cal terms
        for i in range(n_cal):
            C = params[idx]; idx += 1
            W = params[idx]; idx += 1
            PHI = params[idx]; idx += 1
            if cal_damps[i] == "d":
                D = params[idx]; idx += 1
                result = result + C * ts**(-D) * np.cos(W * ts + PHI)
            else:
                result = result + C * np.cos(W * ts + PHI)

        return result

    return model_fn, param_names, bounds


def all_configs():
    """Generate all 36 configurations."""
    configs = []
    for n_log in [0, 1, 2]:
        for n_cal in [0, 1, 2]:
            if n_log == 0:
                log_opts = [()]
            elif n_log == 1:
                log_opts = [("d",), ("u",)]
            else:
                log_opts = [("d", "d"), ("d", "u"), ("u", "u")]

            if n_cal == 0:
                cal_opts = [()]
            elif n_cal == 1:
                cal_opts = [("d",), ("u",)]
            else:
                cal_opts = [("d", "d"), ("d", "u"), ("u", "u")]

            for ld in log_opts:
                for cd in cal_opts:
                    configs.append((n_log, n_cal, list(ld), list(cd)))
    return configs


def main():
    update = "--update" in sys.argv

    print("Loading prices...")
    prices = load_prices("BitcoinPricesDaily.csv")
    t = prices.df_full["years"].values
    log_p = prices.df_full["log_price"].values
    mask = t >= 1.0
    t_fit = t[mask]
    lp_fit = log_p[mask]
    n = len(t_fit)
    ss_tot = np.sum((lp_fit - np.mean(lp_fit)) ** 2)
    print(f"  {n} data points\n")

    configs = all_configs()
    print(f"Fitting {len(configs)} configurations...\n")

    results = {}
    for n_log, n_cal, ld, cd in configs:
        key = config_key(n_log, n_cal, ld, cd)
        func, pnames, bounds = build_model_fn(n_log, n_cal, ld, cd)
        n_params = len(bounds)

        print(f"  {key} ({n_params}p)...", end=" ", flush=True)

        def obj(params, f=func):
            return np.sum((lp_fit - f(t_fit, *params)) ** 2)

        res = differential_evolution(
            obj, bounds, maxiter=5000, seed=42, tol=1e-12,
            polish=True, workers=1,
        )

        pred = func(t_fit, *res.x)
        ss_res = np.sum((lp_fit - pred) ** 2)
        r2 = 1 - ss_res / ss_tot
        sigma = float(np.std(lp_fit - pred))
        bic = n * np.log(ss_res / n) + n_params * np.log(n)

        results[key] = {
            "n_log": n_log, "n_cal": n_cal,
            "log_damps": ld, "cal_damps": cd,
            "params": {pn: float(pv) for pn, pv in zip(pnames, res.x)},
            "r2": r2, "sigma": sigma, "bic": bic, "n_params": n_params,
        }

        print(f"R²={r2:.6f}  σ={sigma:.6f}  BIC={bic:.1f}")

    # Summary table
    print(f"\n{'=' * 80}")
    print(f"{'Config':<20} {'Params':>6} {'R²':>10} {'σ':>10} {'BIC':>10}")
    print("-" * 80)
    for key in sorted(results, key=lambda k: -results[k]["r2"]):
        r = results[key]
        print(f"{key:<20} {r['n_params']:>6} {r['r2']:>10.6f} {r['sigma']:>10.6f} {r['bic']:>10.1f}")

    # Save to JSON for btc_core.py to load
    json_path = os.path.join(ROOT, "hybppl_config_params.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {json_path}")

    if update:
        print("\nUpdating btc_core.py with _HYBPPL_CONFIG_PARAMS...")
        core_path = os.path.join(ROOT, "btc_core.py")
        with open(core_path) as f:
            src = f.read()

        # Build the dict literal
        lines = ["_HYBPPL_CONFIG_PARAMS = {"]
        for key, r in sorted(results.items()):
            p = r["params"]
            param_str = ", ".join(f'"{k}": {v:.6f}' for k, v in p.items())
            lines.append(f'    "{key}": {{"n_log": {r["n_log"]}, "n_cal": {r["n_cal"]}, '
                        f'"log_damps": {r["log_damps"]}, "cal_damps": {r["cal_damps"]}, '
                        f'"params": {{{param_str}}}, '
                        f'"r2": {r["r2"]:.6f}, "sigma": {r["sigma"]:.6f}}},')
        lines.append("}")
        block = "\n".join(lines)

        marker = "# ── HybPPL config params (auto-generated) ──"
        if marker in src:
            # Replace existing block
            import re
            pattern = re.escape(marker) + r".*?(?=\n\nclass |\n# ── )"
            src = re.sub(pattern, marker + "\n" + block, src, flags=re.DOTALL)
        else:
            # Insert before first class that uses it
            insert_pos = src.find("class HybPPLConfigModel")
            if insert_pos == -1:
                insert_pos = src.find("class LogisticModel")
            if insert_pos == -1:
                print("WARNING: Could not find insertion point")
                return
            src = src[:insert_pos] + marker + "\n" + block + "\n\n\n" + src[insert_pos:]

        with open(core_path, "w") as f:
            f.write(src)
        print("btc_core.py updated.")


if __name__ == "__main__":
    main()
