#!/usr/bin/env python3
"""Build model_data.pkl from BitcoinPricesDaily.csv -- standalone, no notebook.

Extracts computation code from sp_stripped.ipynb cells 0, 1, 2 and
runs them in sequence in a shared namespace.  Produces the same lean
pkl as running the stripped notebook via jupyter nbconvert.

NOTE: Uses exec() on trusted local notebook source to guarantee
identical computation.  This matches the build_ef_model.py pattern.

Usage:
    btc_venv/bin/python3 tools/build_bm_model.py
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def extract_cell(nb, idx):
    """Return source code of notebook cell as a single string."""
    s = nb["cells"][idx]["source"]
    return "".join(s) if isinstance(s, list) else s


def main():
    parser = argparse.ArgumentParser(description="Build model_data.pkl")
    parser.add_argument("--notebook",
                        default=os.path.join(ROOT, "sp_stripped.ipynb"),
                        help="Source notebook")
    args = parser.parse_args()

    with open(args.notebook) as f:
        nb = json.load(f)

    # Set non-interactive backend before Cell 0 imports matplotlib.
    # Required for headless servers (production VPS).
    import matplotlib
    matplotlib.use("Agg")

    os.chdir(ROOT)
    ns = {"__name__": "__main__"}

    for i, label in enumerate(["Bubble model", "QR/OLS fitting", "Export"]):
        print(f"Cell {i}: {label}...")
        code = compile(extract_cell(nb, i), f"{args.notebook}:cell{i}", "exec")
        # exec() on trusted local notebook cell source -- see module docstring
        exec(code, ns)  # noqa: S102

    print("Done.")


if __name__ == "__main__":
    main()
