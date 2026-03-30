#!/usr/bin/env python3
"""Compare two model_data.pkl files -- key-by-key values + sha256 hashes.

NOTE: Uses pickle.load for trusted, locally-generated model data files only.
"""
import hashlib
import struct
import sys
import pickle

import numpy as np

MODEL_KEYS = [
    "qr_fits", "QR_QUANTILES", "ols_intercept", "ols_slope", "GENESIS_DATE",
    "years_plot_bm", "support_plot_bm", "bm_comp_by_n", "bm_r2_comp",
    "bm_n_future_max", "bm_sigma0_up", "bm_sigma0_down", "bm_alpha_up",
    "bm_alpha_down", "price_dates", "price_years", "price_prices",
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def floats_identical(a, b):
    return struct.pack("d", a) == struct.pack("d", b)


def compare_values(a, b, path=""):
    """Recursively compare two values. Return list of mismatch descriptions."""
    mismatches = []
    if type(a) != type(b):
        mismatches.append(f"{path}: type {type(a).__name__} vs {type(b).__name__}")
        return mismatches
    if isinstance(a, dict):
        for k in sorted(set(list(a.keys()) + list(b.keys()))):
            if k not in a:
                mismatches.append(f"{path}[{k!r}]: missing in first")
            elif k not in b:
                mismatches.append(f"{path}[{k!r}]: missing in second")
            else:
                mismatches.extend(compare_values(a[k], b[k], f"{path}[{k!r}]"))
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            mismatches.append(f"{path}: length {len(a)} vs {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                mismatches.extend(compare_values(x, y, f"{path}[{i}]"))
    elif isinstance(a, float):
        if not floats_identical(a, b):
            mismatches.append(f"{path}: {a!r} vs {b!r}")
    elif isinstance(a, np.ndarray):
        if not np.array_equal(a, b):
            diff = np.where(a != b)
            mismatches.append(f"{path}: arrays differ at {len(diff[0])} positions")
    else:
        if a != b:
            mismatches.append(f"{path}: {a!r} vs {b!r}")
    return mismatches


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <reference.pkl> <candidate.pkl>")
        sys.exit(1)

    ref_path, cand_path = sys.argv[1], sys.argv[2]

    # SHA256 comparison
    ref_hash = sha256(ref_path)
    cand_hash = sha256(cand_path)
    print(f"SHA256 reference:  {ref_hash}")
    print(f"SHA256 candidate:  {cand_hash}")
    print(f"SHA256 match:      {'YES' if ref_hash == cand_hash else 'NO'}")
    print()

    # Key-by-key comparison
    ref = load_pkl(ref_path)
    cand = load_pkl(cand_path)

    all_ok = True
    for key in MODEL_KEYS:
        if key not in ref:
            print(f"  {key:25s}  SKIP (not in reference)")
            continue
        if key not in cand:
            print(f"  {key:25s}  MISSING in candidate")
            all_ok = False
            continue
        mismatches = compare_values(ref[key], cand[key], key)
        if mismatches:
            print(f"  {key:25s}  FAIL")
            for m in mismatches[:3]:
                print(f"    {m}")
            all_ok = False
        else:
            print(f"  {key:25s}  OK")

    extra = set(cand.keys()) - set(ref.keys()) - set(MODEL_KEYS)
    if extra:
        print(f"\nExtra keys in candidate: {sorted(extra)}")

    print(f"\nOverall: {'PASS' if all_ok else 'FAIL'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
