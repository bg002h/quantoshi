#!/usr/bin/env python3
"""Monthly refit of ALL periodic power law models + related models.

Runs all DE-based fitting scripts with --update, then rebuilds the
HybPPL and EPPL config model parameter sets.

This replaces the old quantoshi-lppl-refit.service which only refitted
the 6 LPPL weighted/no-13 variants.

Expected runtime: ~60-90 minutes (all models sequentially).

Usage:
    btc_venv/bin/python3 tools/refit_all_ppl.py           # refit all
    btc_venv/bin/python3 tools/refit_all_ppl.py --dry-run  # list what would run
"""
import os
import sys
import subprocess
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
PYTHON = sys.executable

DRY_RUN = "--dry-run" in sys.argv

# ── All fitting scripts in order ─────────────────────────────────────────

SCRIPTS = [
    # LPPL family (base + multi-freq)
    ("LPPL",           "tools/fit_lppl.py"),
    ("LPPL\u2082",     "tools/fit_lppl2.py"),
    ("LPPL\u2083",     "tools/fit_lppl3.py"),
    ("LPPL\u2084",     "tools/fit_lppl4.py"),
    ("LPPL variants",  "tools/fit_lppl_variants.py"),  # weighted + no-13

    # LinPPL
    ("LinPPL",         "tools/fit_linppl.py"),

    # HybPPL family
    ("HybPPL",         "tools/fit_hybppl.py"),
    ("HybPPL DD",      "tools/fit_hybppl_dd.py"),
    ("Hyb2L",          "tools/fit_hyb2l.py"),
    ("Hyb2C",          "tools/fit_hyb2c.py"),
    ("Hyb2B",          "tools/fit_hyb2b.py"),
    ("Hyb4D",          "tools/fit_hyb4d.py"),

    # HybPPL config models (36 configs)
    ("HybPPL configs", "tools/fit_all_hybppl_configs.py"),

    # EPPL config models (36 configs)
    ("EPPL configs",   "tools/fit_all_eppl_configs.py"),

    # Other models
    ("Gompertz",       "tools/fit_gompertz.py"),
    ("Broken PL",      "tools/fit_bpl.py"),
    ("Offset PL",      "tools/fit_plo.py"),
    ("Stretched Exp",  "tools/fit_sexp.py"),
    ("Logistic",       "tools/fit_logistic.py"),
]

def main():
    print("=" * 60)
    print("Monthly PPL Model Refit")
    print("=" * 60)
    print(f"Scripts: {len(SCRIPTS)}")
    print(f"Mode: {'DRY RUN' if DRY_RUN else 'LIVE'}\n")

    if DRY_RUN:
        for name, script in SCRIPTS:
            exists = os.path.exists(os.path.join(ROOT, script))
            status = "\u2713" if exists else "\u2717 MISSING"
            print(f"  {status} {name:20s}  {script}")
        print("\nDry run complete. No changes made.")
        return

    t0_total = time.perf_counter()
    n_ok = 0
    n_fail = 0
    results = []

    for name, script in SCRIPTS:
        script_path = os.path.join(ROOT, script)
        if not os.path.exists(script_path):
            print(f"\n\u26a0 {name}: {script} NOT FOUND \u2014 skipping")
            results.append((name, "MISSING", 0))
            continue

        print(f"\n{'─' * 40}")
        print(f"Refitting {name} ({script})...")
        t0 = time.perf_counter()

        res = subprocess.run(
            [PYTHON, script_path, "--update"],
            capture_output=True, text=True,
            timeout=1800,  # 30 min max per script
        )

        elapsed = time.perf_counter() - t0

        if res.returncode != 0:
            print(f"  FAILED ({elapsed:.0f}s)")
            print(f"  stderr (last 1000 chars): {res.stderr[-1000:]}")
            results.append((name, "FAILED", elapsed))
            n_fail += 1
        else:
            last_line = res.stdout.strip().split("\n")[-1] if res.stdout.strip() else ""
            print(f"  OK ({elapsed:.0f}s) \u2014 {last_line}")
            results.append((name, "OK", elapsed))
            n_ok += 1

    total_elapsed = time.perf_counter() - t0_total

    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total time: {total_elapsed/60:.1f} minutes")
    print(f"  OK: {n_ok}  Failed: {n_fail}  Missing: {len(SCRIPTS)-n_ok-n_fail}")
    print()
    for name, status, elapsed in results:
        icon = "\u2713" if status == "OK" else ("\u2717" if status == "FAILED" else "\u26a0")
        print(f"  {icon} {name:20s}  {status:8s}  {elapsed:.0f}s")


if __name__ == "__main__":
    main()
