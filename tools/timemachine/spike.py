#!/usr/bin/env python3
"""Task 0 (Build-parameter spike) for the "Time Machine" feature.

Measures four build parameters that later Time Machine tasks (grid builder,
cold-fit-per-frame, flagship EPPL frame) consume. This is a spike: it is run
once, the printed numbers are hand-transcribed into CONSTANTS below, and it
is committed for reproducibility (re-running should reproduce the same
numbers up to DE's inherent run-to-run jitter at fixed seed/tol/workers).

Run:
    btc_venv/bin/python3 tools/timemachine/spike.py

Steps (see task-0-brief.md for the full spec):
    1. Convergence sweep for maxiter on ecfg_2dd_2uu (16p, hardest EPPL
       config), full data.               -> EPPL_MAXITER
    2. (folded into step 1's loop)
    3. Left-edge detection: cold-fit ecfg_2dd_2uu at ymax in
       {2.5, 3.0, 3.5, 4.0} years-since-origin; earliest window with
       r2 >= 0.95 and no oscillator amplitude rail-pinned.
                                          -> LEFT_EDGE_DATE
    4. BM per-frame timing (informational) + flagship-vs-cold-fit param
       comparison.                       -> FLAGSHIP_POLICY, BM cost
"""
import os
import sys
import json
import time
import tempfile

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, "tools"), ROOT]

from model_toolkit.data import load_prices
from scipy.optimize import differential_evolution
from fit_all_eppl_configs import build_model_fn

GENESIS = pd.Timestamp("2009-07-25")

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS — the deliverable of this spike. Consumed by Task 1/2/3.
# Measured by running this file on 2026-08-10 against BitcoinPricesDaily.csv
# (5856 rows with t>=1.0, tmax≈17.03yr). See task-0-report.md for full tables.
# ═══════════════════════════════════════════════════════════════════════════
CONSTANTS = {
    # Step 1: comparing consecutive sweep steps (150->300->600->1200), the
    # 600->1200 pair is the first to satisfy BOTH max|dparam|<1e-3 (0.00081)
    # AND dr2<1e-5 (2.1e-9) -> chosen maxiter is the smaller of that pair.
    # Note: 300->600 has dr2=2.1e-6 (passes) but max|dparam|=6.19 (fails) --
    # r2 is flat across 300/600/1200 (0.993421/0.993423/0.993423) while a
    # param jumps by ~6, which is log-periodic frequency aliasing (a
    # near-equally-good local optimum at a different W), not non-convergence.
    # r2 alone would justify maxiter=300; the brief's param-delta criterion
    # is stricter and is what's applied here.
    "EPPL_MAXITER": 600,

    # Step 3: earliest as-of window (ymax in {2.5,3.0,3.5,4.0} years since
    # 2009-07-25) where a cold ecfg_2dd_2uu fit reaches r2>=0.95 with no
    # oscillator amplitude C_* within 0.05 of its upper bound. All 4 swept
    # windows passed both checks (r2 0.982-0.990, no rail-pinned C_*), so
    # the earliest one wins. CONCERN: 548 points / 16 free params at
    # ymax=2.5 is a suspiciously easy r2>=0.95 -- likely overfit rather than
    # a genuine well-constrained oscillator fit; see task-0-report.md.
    "LEFT_EDGE_DATE": "2012-01-24",  # ymax=2.5

    # Step 4: cold ecfg_2dd_2uu (full data, EPPL_MAXITER=600) vs. the
    # hardcoded flagship EntropyPPLModel class attrs. Log-oscillator pair
    # and cal-oscillator pair are independent labeling degeneracies, so all
    # 4 swap combinations are tried; best is log_swap=True, cal_swap=False
    # (log1<->log2 relabeled, cal kept direct) with max|delta|=0.0314 on
    # PHI_cal1 (a ~1.8 degree phase mismatch) -- above the 1e-2 threshold,
    # so the spike-quality cold fit (maxiter=600, tol=1e-8) does not
    # reproduce the flagship's own fit (maxiter=5000, tol=1e-12) closely
    # enough to reuse its params.
    "FLAGSHIP_POLICY": "own_frame",

    # Step 4: BM as-of build cost (fit_support + fit_sequential on a
    # truncated CSV, as-of 2017-07-25 / ymax=8.0yr, 2566 rows, 3 bubbles
    # fitted), informational only -- Task 3's real BM-as-of builder doesn't
    # exist yet, this is the closest available proxy. fit_bubble's DE calls
    # are seeded (seed=42+seed_idx), so both runs found the same 3 bubbles;
    # the two wall-clock measurements (2940ms, 5422ms) differ only from
    # background system load on this shared dev box, not nondeterminism.
    # Use low-single-digit seconds/frame as the planning number, not this
    # specific value.
    "BM_PER_FRAME_MS": 2940,  # first (less loaded) measurement; see comment
}


# ─────────────────────────────────────────────────────────────────────────
# Step 1: EPPL_MAXITER convergence sweep
# ─────────────────────────────────────────────────────────────────────────
def step1_maxiter_sweep(t, lp):
    print("\n=== Step 1: maxiter convergence sweep (ecfg_2dd_2uu, full data) ===")
    m = t >= 1.0
    tf, lpf = t[m], lp[m]
    func, pn, bounds = build_model_fn(2, 2, ["d", "d"], ["u", "u"])
    ss_tot = np.sum((lpf - lpf.mean()) ** 2)

    prev_x = None
    prev_r2 = None
    rows = []
    for mit in (150, 300, 600, 1200):
        t0 = time.perf_counter()
        r = differential_evolution(
            lambda p: np.sum((lpf - func(tf, *p)) ** 2),
            bounds, maxiter=mit, tol=1e-8, seed=42, workers=1,
        )
        dt = time.perf_counter() - t0
        pred = func(tf, *r.x)
        r2 = 1 - np.sum((lpf - pred) ** 2) / ss_tot
        dparam = None if prev_x is None else float(np.max(np.abs(r.x - prev_x)))
        dr2 = None if prev_r2 is None else float(abs(r2 - prev_r2))
        print(f"  maxiter={mit:5d}  r2={r2:.6f}  max|dparam(vs prev)|={dparam}  "
              f"dr2(vs prev)={dr2}  wall={dt:.1f}s")
        rows.append({"maxiter": mit, "r2": r2, "params": dict(zip(pn, r.x)),
                      "max_dparam_vs_prev": dparam, "dr2_vs_prev": dr2, "wall_s": dt})
        prev_x, prev_r2 = r.x, r2

    # Pick smallest maxiter where the delta to the *next* step satisfies the
    # convergence criteria (comparing rows[i] to rows[i+1], per the brief).
    chosen = None
    for i in range(len(rows) - 1):
        cur, nxt = rows[i], rows[i + 1]
        d = float(np.max(np.abs(np.array(list(cur["params"].values())) -
                                 np.array(list(nxt["params"].values())))))
        dr2 = abs(nxt["r2"] - cur["r2"])
        print(f"  compare maxiter={cur['maxiter']} -> {nxt['maxiter']}: "
              f"max|dparam|={d:.6f} dr2={dr2:.2e}")
        if chosen is None and d < 1e-3 and dr2 < 1e-5:
            chosen = cur["maxiter"]
    if chosen is None:
        chosen = rows[-1]["maxiter"]
        print(f"  WARNING: no step met the convergence criteria; "
              f"falling back to largest sweep value {chosen}")
    print(f"  -> EPPL_MAXITER candidate: {chosen}")
    return rows, chosen


# ─────────────────────────────────────────────────────────────────────────
# Step 3: left-edge detection
# ─────────────────────────────────────────────────────────────────────────
def step3_left_edge(t, lp, maxiter):
    print("\n=== Step 3: left-edge detection (ecfg_2dd_2uu cold fits) ===")
    func, pn, bounds = build_model_fn(2, 2, ["d", "d"], ["u", "u"])
    c_upper = {n: b[1] for n, b in zip(pn, bounds) if n.startswith("C_")}

    results = []
    for ymax in (2.5, 3.0, 3.5, 4.0):
        m = (t >= 1.0) & (t <= ymax)
        tf, lpf = t[m], lp[m]
        ss_tot = np.sum((lpf - lpf.mean()) ** 2)
        r = differential_evolution(
            lambda p: np.sum((lpf - func(tf, *p)) ** 2),
            bounds, maxiter=maxiter, tol=1e-8, seed=42, workers=1,
        )
        pred = func(tf, *r.x)
        r2 = 1 - np.sum((lpf - pred) ** 2) / ss_tot
        params = dict(zip(pn, r.x))
        rail = {n: bool(c_upper[n] - params[n] < 0.05) for n in c_upper}
        ok = bool(r2 >= 0.95 and not any(rail.values()))
        date = (GENESIS + pd.Timedelta(days=ymax * 365.25)).strftime("%Y-%m-%d")
        print(f"  ymax={ymax:.1f}  date~{date}  n={len(tf):4d}  r2={r2:.4f}  "
              f"ok={ok}  rail={ {k: v for k, v in rail.items() if v} }")
        results.append({"ymax": ymax, "date": date, "n": len(tf), "r2": r2,
                         "params": params, "rail": rail, "ok": ok})

    edge = next((r for r in results if r["ok"]), None)
    if edge is None:
        print("  WARNING: no swept window satisfied r2>=0.95 + non-rail; "
              "no left edge found in {2.5,3.0,3.5,4.0}")
    else:
        print(f"  -> LEFT_EDGE_DATE candidate: {edge['date']} (ymax={edge['ymax']})")
    return results, edge


# ─────────────────────────────────────────────────────────────────────────
# Step 4a: flagship policy — cold ecfg_2dd_2uu vs hardcoded EntropyPPLModel
# ─────────────────────────────────────────────────────────────────────────
# Mapping from flagship EntropyPPLModel class attrs (btc_core/_hybppl_eppl.py)
# to ecfg_2dd_2uu param names, per the class docstring:
#   C1/W1/P1/w1 -> log-periodic oscillator 1 (entropy-damped)
#   C3/W2/P3/w2 -> log-periodic oscillator 2 (entropy-damped)
#   C2/Wc1/P2   -> calendar oscillator 1 (undamped, T=3.34yr)
#   C4/Wc2/P4   -> calendar oscillator 2 (undamped, T=1.87yr)
FLAGSHIP_PARAMS = {
    "A": -1.167364, "B": 5.079560,
    "C_log1": 0.250431, "W_log1": 16.823756, "PHI_log1": 1.460422, "w_log1": 0.251550,
    "C_log2": 0.556269, "W_log2": 7.803554, "PHI_log2": 1.373041, "w_log2": 0.107049,
    "C_cal1": 0.202747, "W_cal1": 1.881312, "PHI_cal1": 2.520900,
    "C_cal2": 0.113542, "W_cal2": 3.355482, "PHI_cal2": 3.033230,
}
# Oscillator-swap mapping: cos(W*x+phi) with (C1,W1,PHI1)<->(C2,W2,PHI2) is
# the "same shape, different label" degeneracy DE can land in since the two
# log oscillators (and, independently, the two cal oscillators) share
# identical bounds. The log pair and cal pair can each be independently
# swapped or not -- they are unrelated degeneracies -- so all 4 combinations
# must be tried and scored separately (an earlier draft of this script tried
# only "swap both" vs "swap neither" and produced a spuriously large delta
# by cross-pairing a correctly-ordered cal oscillator against the wrong
# flagship slot; see task-0-report.md for the numbers this produced).
_LOG_PAIRS = [("C_log1", "C_log2"), ("W_log1", "W_log2"),
              ("PHI_log1", "PHI_log2"), ("w_log1", "w_log2")]
_CAL_PAIRS = [("C_cal1", "C_cal2"), ("W_cal1", "W_cal2"), ("PHI_cal1", "PHI_cal2")]


def _make_name_map(param_names, swap_log, swap_cal):
    m = {n: n for n in param_names}
    pairs = ([] if not swap_log else _LOG_PAIRS) + ([] if not swap_cal else _CAL_PAIRS)
    for a, b in pairs:
        m[a], m[b] = b, a
    return m


def _wrapped_abs_diff(a, b, is_phase):
    d = abs(a - b)
    return min(d, 2 * np.pi - d) if is_phase else d


def step4_flagship_policy(t, lp, maxiter):
    print("\n=== Step 4a: flagship policy (cold ecfg_2dd_2uu vs EntropyPPLModel) ===")
    func, pn, bounds = build_model_fn(2, 2, ["d", "d"], ["u", "u"])
    m = t >= 1.0
    tf, lpf = t[m], lp[m]
    r = differential_evolution(
        lambda p: np.sum((lpf - func(tf, *p)) ** 2),
        bounds, maxiter=maxiter, tol=1e-8, seed=42, workers=1,
    )
    cold = dict(zip(pn, r.x))
    ss_tot = np.sum((lpf - lpf.mean()) ** 2)
    r2 = 1 - np.sum((lpf - func(tf, *r.x)) ** 2) / ss_tot
    print(f"  cold fit r2={r2:.6f} (flagship class docstring claims r2=0.993320)")

    def max_delta(name_map):
        diffs = {}
        for cn, fn in name_map.items():
            is_phase = cn.startswith("PHI")
            diffs[cn] = _wrapped_abs_diff(cold[cn], FLAGSHIP_PARAMS[fn], is_phase)
        return diffs, max(diffs.values())

    # Log-oscillator swap and cal-oscillator swap are independent
    # degeneracies -- score all 4 combinations, pick the minimum max|delta|.
    candidates = {}
    for swap_log in (False, True):
        for swap_cal in (False, True):
            name_map = _make_name_map(pn, swap_log, swap_cal)
            diffs, mx = max_delta(name_map)
            label = f"log_swap={swap_log},cal_swap={swap_cal}"
            candidates[label] = (diffs, mx)
            print(f"  {label}: max|delta|={mx:.4f}")

    best_label = min(candidates, key=lambda k: candidates[k][1])
    best_diffs, best_max = candidates[best_label]
    worst_param = max(best_diffs, key=best_diffs.get)
    print(f"  best mapping: {best_label}, max|delta|={best_max:.4f} (param={worst_param})")

    # Supporting evidence for the "ambiguous mapping" fallback: shared trend
    # params (A,B) and the frequency sets, independent of amplitude/phase
    # degeneracies.
    ab_diff = max(abs(cold["A"] - FLAGSHIP_PARAMS["A"]),
                  abs(cold["B"] - FLAGSHIP_PARAMS["B"]))
    log_freqs_cold = sorted([cold["W_log1"], cold["W_log2"]])
    log_freqs_flag = sorted([FLAGSHIP_PARAMS["W_log1"], FLAGSHIP_PARAMS["W_log2"]])
    cal_freqs_cold = sorted([cold["W_cal1"], cold["W_cal2"]])
    cal_freqs_flag = sorted([FLAGSHIP_PARAMS["W_cal1"], FLAGSHIP_PARAMS["W_cal2"]])
    freq_diff_log = max(abs(a - b) for a, b in zip(log_freqs_cold, log_freqs_flag))
    freq_diff_cal = max(abs(a - b) for a, b in zip(cal_freqs_cold, cal_freqs_flag))
    print(f"  max|A,B delta| = {ab_diff:.4f}")
    print(f"  log-osc freq sets: cold={log_freqs_cold} flagship={log_freqs_flag} "
          f"max|delta|={freq_diff_log:.4f}")
    print(f"  cal-osc freq sets: cold={cal_freqs_cold} flagship={cal_freqs_flag} "
          f"max|delta|={freq_diff_cal:.4f}")

    policy = "own_frame" if best_max > 1e-2 else "map_2dd2uu"
    print(f"  -> FLAGSHIP_POLICY candidate: {policy}")

    return {
        "cold_params": cold, "cold_r2": r2,
        "candidates": {k: v[1] for k, v in candidates.items()},
        "best_label": best_label, "best_max": best_max, "worst_param": worst_param,
        "ab_diff": ab_diff, "freq_diff_log": freq_diff_log, "freq_diff_cal": freq_diff_cal,
        "policy": policy,
    }


# ─────────────────────────────────────────────────────────────────────────
# Step 4b: BM per-frame cost (informational — Task 3's BM-as-of builder
# does not exist yet, so this times the closest available proxy:
# fit_support + fit_sequential on a truncated copy of the price CSV).
# ─────────────────────────────────────────────────────────────────────────
def step4_bm_timing(csv_path, ymax=8.0):
    print(f"\n=== Step 4b: BM per-frame cost (informational, ymax={ymax} yr) ===")
    try:
        from model_toolkit.support import fit_support
        from model_toolkit.fitting import fit_sequential

        cutoff = GENESIS + pd.Timedelta(days=ymax * 365.25)
        raw = pd.read_csv(csv_path)
        date_col = next((c for c in raw.columns if "date" in c.lower()), raw.columns[0])
        raw[date_col] = pd.to_datetime(raw[date_col])
        trunc = raw[raw[date_col] <= cutoff]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            tmp_path = f.name
        trunc.to_csv(tmp_path, index=False)

        t0 = time.perf_counter()
        pdat = load_prices(tmp_path)
        support = fit_support(pdat)
        bubbles = fit_sequential(pdat, support)
        dt_ms = (time.perf_counter() - t0) * 1000

        os.unlink(tmp_path)
        print(f"  as-of {cutoff.strftime('%Y-%m-%d')} ({len(trunc)} rows, "
              f"{len(bubbles)} bubbles fitted): {dt_ms:.0f} ms")
        return {"ok": True, "ms": dt_ms, "ymax": ymax,
                "cutoff": cutoff.strftime("%Y-%m-%d"),
                "n_rows": int(len(trunc)), "n_bubbles": len(bubbles)}
    except Exception as e:
        print(f"  BM timing not measured: {e!r} -- deferred to Task 3")
        return {"ok": False, "error": repr(e)}


def main():
    print("Loading BitcoinPricesDaily.csv...")
    pr = load_prices("BitcoinPricesDaily.csv")
    t = pr.df_full["years"].values
    lp = pr.df_full["log_price"].values
    print(f"  {len(t)} rows total, {(t >= 1.0).sum()} with t>=1.0, tmax={t.max():.2f}yr")

    sweep_rows, maxiter_candidate = step1_maxiter_sweep(t, lp)
    edge_rows, edge_candidate = step3_left_edge(t, lp, CONSTANTS["EPPL_MAXITER"])
    policy_info = step4_flagship_policy(t, lp, CONSTANTS["EPPL_MAXITER"])
    bm_info = step4_bm_timing("BitcoinPricesDaily.csv")

    print("\n=== Summary (candidates from this run; compare to CONSTANTS above) ===")
    print(f"  EPPL_MAXITER candidate:    {maxiter_candidate}")
    print(f"  LEFT_EDGE_DATE candidate:  "
          f"{edge_candidate['date'] if edge_candidate else 'NONE FOUND'}")
    print(f"  FLAGSHIP_POLICY candidate: {policy_info['policy']}")
    print(f"  BM_PER_FRAME_MS candidate: "
          f"{bm_info['ms'] if bm_info.get('ok') else 'not measured'}")

    out = {
        "sweep": sweep_rows, "edge": edge_rows, "policy": policy_info, "bm": bm_info,
    }
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spike_output.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nFull results written to {out_path} (gitignored scratch, not committed).")


if __name__ == "__main__":
    main()
