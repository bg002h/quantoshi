#!/usr/bin/env python3
"""Build model_data.pkl -- BM model via model_toolkit.

Usage:
    btc_venv/bin/python3 tools/build_bm_model.py
"""
import datetime as _dt
import json
import os
import sys
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, ROOT)  # make btc_core importable for the resqr fit phase
os.chdir(ROOT)


# Phase 2b.i: pre-parse --time-basis flag from argv so we can set
# QS_TIME_BASIS env var BEFORE any time_basis import. Without this,
# time_basis.py loads with TOML's calendar default and the build runs
# with calendar T_PER_YEAR=1.0 even when CLI says block.
def _early_time_basis_setup():
    """Look for --time-basis in argv, set QS_TIME_BASIS env var."""
    for i, arg in enumerate(sys.argv[1:], start=1):
        if arg == "--time-basis" and i + 1 < len(sys.argv):
            val = sys.argv[i + 1]
            if val in ("calendar", "block"):
                os.environ["QS_TIME_BASIS"] = val
            return
        if arg.startswith("--time-basis="):
            val = arg.split("=", 1)[1]
            if val in ("calendar", "block"):
                os.environ["QS_TIME_BASIS"] = val
            return

_early_time_basis_setup()


import numpy as np
from model_toolkit.data import load_prices
from model_toolkit.support import fit_support
from model_toolkit.fitting import fit_sequential, classify
from model_toolkit.prediction import predict_future
from model_toolkit.composite import build_composite, build_comp_by_n
from model_toolkit.bands import fit_qr_channels, fit_asymmetric_sigma, BM_QUANTILES
from model_toolkit.export import build_bm_pkl_dict, write_pkl

# 15 flagship in-scope models for residual QR sigma bands. LPPL family
# excluded (3.92yr halving cycle residual periodicity that PWL can't
# capture). HybPPL + Entropy PPL config variants (72) are appended at
# runtime inside _fit_all_resqr — they inherit the HybPPL/EPPL parent
# gating but the chart builder swaps in concrete cfg_* keys via
# _resolve_{hybppl,eppl}_master, so each variant needs its own fit.
RESQR_FLAGSHIP_MODELS = (
    "bub", "pl", "hybppl", "hybppl_dd",
    "hyb2l", "hyb2c", "hyb2b", "hyb4d",
    "eppl", "linppl", "exp", "pca",
    "grdy", "gomp", "bpl",
    # spl promoted 2026-08-08 (user). Tab 1 defaults to sigma_mode="resqr",
    # so a model without a bundle silently falls back to constant sigma and
    # draws visibly different band widths from bub/pl beside it. spl follows
    # BM. plo/sexp/logi are still unpromoted -- see F-7 in
    # docs/superpowers/followups.md.
    "spl",
)


def _build_model_instances(md_obj, price_years, price_prices, quantiles):
    """Instantiate every in-scope model from pkl data. Order matters: PCA
    consumes the already-built HybPPL family as source_models. HybPPL +
    Entropy PPL config variants (36 each) are appended at the end so
    their resqr bundles can be bound when the master-gate resolver
    swaps `"hybppl"`/`"eppl"` for a concrete `cfg_*`/`ecfg_*` key at
    chart build time."""
    import btc_core as bc  # lazy — avoid cycle with module-level imports

    instances = {}
    instances["bub"] = bc.BubbleModel(md_obj)
    instances["pl"] = bc.PowerLawModel(
        md_obj.ols_intercept, md_obj.ols_slope,
        md_obj.price_years, md_obj.price_prices,
        "2009-07-25", quantiles,
    )
    instances["hybppl"] = bc.HybPPLModel(price_years, price_prices, quantiles)
    instances["hybppl_dd"] = bc.HybPPLDDModel(price_years, price_prices, quantiles)
    instances["hyb2l"] = bc.Hyb2LModel(price_years, price_prices, quantiles)
    instances["hyb2c"] = bc.Hyb2CModel(price_years, price_prices, quantiles)
    instances["hyb2b"] = bc.Hyb2BModel(price_years, price_prices, quantiles)
    instances["hyb4d"] = bc.Hyb4DModel(price_years, price_prices, quantiles)
    instances["eppl"] = bc.EntropyPPLModel(price_years, price_prices, quantiles)
    instances["linppl"] = bc.LinPPLModel(price_years, price_prices, quantiles)
    instances["exp"] = bc.ExponentialModel(price_years, price_prices, quantiles)
    # PCA needs the HybPPL family available as source_models.
    instances["pca"] = bc.PCAModel(price_years, price_prices, quantiles,
                                    source_models=instances)
    instances["grdy"] = bc.GreedyModel(price_years, price_prices, quantiles)
    instances["gomp"] = bc.GompertzModel(price_years, price_prices, quantiles)
    instances["bpl"] = bc.BrokenPowerLawModel(price_years, price_prices, quantiles)
    instances["plo"] = bc.OffsetPowerLawModel(price_years, price_prices, quantiles)
    instances["sexp"] = bc.StretchedExponentialModel(price_years, price_prices, quantiles)
    instances["logi"] = bc.LogisticSCurveModel(price_years, price_prices, quantiles)
    instances["spl"] = bc.SaturatingPowerLawModel(price_years, price_prices, quantiles)
    # HybPPL config variants (36)
    for cfg_key in bc._HYBPPL_CONFIG_PARAMS:
        instances[cfg_key] = bc.HybPPLConfigModel(
            cfg_key, price_years, price_prices, quantiles)
    # Entropy PPL config variants (36)
    for ecfg_key in bc._EPPL_CONFIG_PARAMS:
        instances[ecfg_key] = bc.EPPLConfigModel(
            ecfg_key, price_years, price_prices, quantiles)
    return instances


def _fit_all_resqr(pkl_path):
    """Policy A (per-model skip on ValueError) / Policy B (global abort on
    ``bub`` RuntimeError, ``bub`` ValueError, or >50% skip rate).

    Returns ``(resqr_coefs, diagnostics)``. ``resqr_coefs`` is None on abort."""
    from model_toolkit.resqr_bands import (
        DEFAULT_KNOTS, DEFAULT_QUANTILES, fit_and_validate,
    )
    import btc_core as bc

    print()
    print("=" * 60)
    print("Residual QR sigma bands fit")
    print("=" * 60)

    md_obj = bc.load_model_data(pkl_path)
    price_years = np.asarray(md_obj.price_years, dtype=np.float64)
    price_prices = np.asarray(md_obj.price_prices, dtype=np.float64)
    log_price = np.log10(np.maximum(price_prices, 1e-10))
    quantiles = sorted(md_obj.qr_fits.keys())

    # Full fit scope = 15 flagships + 36 HybPPL cfg_* + 36 EPPL ecfg_*.
    model_keys = (
        list(RESQR_FLAGSHIP_MODELS)
        + list(bc._HYBPPL_CONFIG_PARAMS.keys())
        + list(bc._EPPL_CONFIG_PARAMS.keys())
    )
    print(f"Building {len(model_keys)} model instances "
          f"({len(RESQR_FLAGSHIP_MODELS)} flagships + "
          f"{len(bc._HYBPPL_CONFIG_PARAMS)} cfg_* + "
          f"{len(bc._EPPL_CONFIG_PARAMS)} ecfg_*)...")
    instances = _build_model_instances(
        md_obj, price_years, price_prices, quantiles,
    )

    resqr_coefs = {}
    diagnostics = {}
    skipped = []
    aborted = False
    abort_reason = None

    for key in model_keys:
        model = instances[key]
        try:
            log_median = np.asarray(model._model_log10(price_years), dtype=np.float64)
            residuals = log_price - log_median
            result = fit_and_validate(
                price_years, residuals, model_key=key,
                quantiles=DEFAULT_QUANTILES, knots=DEFAULT_KNOTS,
            )
        except ValueError as exc:
            print(f"  [SKIP] {key}: {exc}")
            skipped.append(key)
            diagnostics[key] = {"status": "skipped", "reason": str(exc)}
            if key == "bub":
                aborted = True
                abort_reason = f"bub failed Policy A: {exc}"
                break
            continue
        except RuntimeError as exc:
            print(f"  [ABORT] {key}: {exc}")
            aborted = True
            abort_reason = f"{key} Policy B (non-finite coefs): {exc}"
            diagnostics[key] = {"status": "aborted", "reason": str(exc)}
            break
        except Exception as exc:
            print(f"  [SKIP] {key}: unexpected {type(exc).__name__}: {exc}")
            traceback.print_exc()
            skipped.append(key)
            diagnostics[key] = {
                "status": "skipped",
                "reason": f"unexpected {type(exc).__name__}: {exc}",
            }
            if key == "bub":
                aborted = True
                abort_reason = f"bub failed unexpectedly: {exc}"
                break
            continue

        raw_cross_pct = 100.0 * result["raw_crossing_frac"]
        print(f"  [OK]   {key}: n={result['n_samples']}, "
              f"raw_cross={raw_cross_pct:.1f}%")
        resqr_coefs[key] = {
            "sorted_qs": result["sorted_qs"],
            "coef_matrix": result["coef_matrix"],
            "knots": DEFAULT_KNOTS,
        }
        diagnostics[key] = {
            "status": "ok",
            "n_samples": result["n_samples"],
            "raw_crossing_frac": result["raw_crossing_frac"],
            "in_sample_coverage": [float(x) for x in result["in_sample_coverage"]],
            "oos_coverage": [float(x) for x in result["oos_coverage"]],
        }

    skip_rate = len(skipped) / len(model_keys)
    if not aborted and skip_rate > 0.5:
        aborted = True
        abort_reason = f"{len(skipped)}/{len(model_keys)} models skipped (>50%)"

    if aborted:
        print()
        print(f"[ABORT] Residual QR fit aborted: {abort_reason}")
        print("Leaving pkl without resqr keys (runtime will fall back to constant σ)")
        return None, {
            "aborted": True,
            "reason": abort_reason,
            "per_model": diagnostics,
        }

    print()
    print(f"[DONE] {len(resqr_coefs)}/{len(model_keys)} models fit; "
          f"{len(skipped)} skipped: {skipped}")
    return resqr_coefs, {
        "aborted": False,
        "skipped": skipped,
        "per_model": diagnostics,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Build model_data.pkl via model_toolkit.")
    parser.add_argument(
        "--time-basis", choices=["calendar", "block"], default="calendar",
        help="Time axis for fits. 'calendar' (default) computes years since "
             "2009-07-25; 'block' computes blocks since T_ORIGIN_BLOCK "
             "(read from quantoshi.toml). Block mode requires "
             "BitcoinBlocksDaily.csv. Phase 2a: parameterized but block "
             "rebuild is Phase 2b's job.",
    )
    args = parser.parse_args()
    print(f"time_basis: {args.time_basis}")

    # Axis-aware artifact filenames. Calendar default keeps existing
    # filenames for back-compat; block writes to *_block.* siblings.
    if args.time_basis == "block":
        pkl_basename = "model_data_block.pkl"
        diag_basename = "model_data_block_resqr_diagnostics.json"
    else:
        pkl_basename = "model_data.pkl"
        diag_basename = "model_data_resqr_diagnostics.json"

    print("Loading prices...")
    prices = load_prices("BitcoinPricesDaily.csv", time_basis=args.time_basis)
    print(f"  {len(prices.df)} fitting points, {len(prices.df_full)} total")

    print("Fitting support line...")
    sup = fit_support(prices)
    print(f"  slope={sup.slope:.4f}, intercept={sup.intercept:.4f}")

    print("Fitting bubbles...")
    fitted = fit_sequential(prices, sup)
    major, minor = classify(fitted, n_major=5)
    print(f"  {len(major)} major + {len(minor)} minor")

    print("Predicting future bubbles...")
    f_maj, f_min = predict_future(major, minor, t_last_data=prices.years[-1],
                                   n_major=3, n_minor=1)
    all_future = sorted(f_maj + f_min, key=lambda b: b["t_rise"])
    print(f"  {len(f_maj)} major + {len(f_min)} minor predicted")

    print("Building composite...")
    comp = build_composite(sup, fitted, prices)
    cbn = build_comp_by_n(sup, fitted, all_future, comp.t_grid,
                           comp.hist_K_max, comp.total_plot)
    print(f"  R2={comp.r2:.4f}, comp_by_n: {len(cbn)} arrays")

    print("Fitting QR channels...")
    qr = fit_qr_channels(prices, BM_QUANTILES)
    print(f"  {len(qr.fits)} quantiles, OLS slope={qr.ols_slope:.4f}")

    print("Fitting sigma...")
    # Use comp_by_n[-1] (full composite incl. future bubbles) and df_full data
    # to match original fit_sigma.py behavior
    full_log_prices = np.log10(np.maximum(prices.df_full["price"].values, 1e-10))
    full_years = prices.df_full["years"].values
    sigma = fit_asymmetric_sigma(full_log_prices, np.array(cbn[-1]),
                                  comp.t_grid, full_years)
    print(f"  sigma0_up={sigma.sigma0_up:.4f}, alpha_up={sigma.alpha_up:.4f}")

    print("Writing pkl...")
    pkl_path = os.path.join(ROOT, pkl_basename)
    pkl_dict = build_bm_pkl_dict(prices, sup, comp, cbn, qr, sigma)
    write_pkl(pkl_dict, pkl_path)

    # ── Residual QR sigma bands (Task 3) ────────────────────────────────
    from model_toolkit.resqr_bands import DEFAULT_KNOTS, DEFAULT_QUANTILES
    resqr_coefs, diagnostics = _fit_all_resqr(pkl_path)
    if resqr_coefs is not None:
        pkl_dict["resqr_coefs"] = resqr_coefs
        pkl_dict["resqr_models"] = sorted(resqr_coefs.keys())
        pkl_dict["resqr_knots"] = list(DEFAULT_KNOTS)
        pkl_dict["resqr_quantiles"] = list(DEFAULT_QUANTILES)
        pkl_dict["resqr_build_ts"] = _dt.datetime.utcnow().isoformat() + "Z"
        write_pkl(pkl_dict, pkl_path)
    # Always write diagnostics (success or abort) alongside the pkl.
    diag_path = os.path.join(ROOT, diag_basename)
    with open(diag_path, "w") as f:
        json.dump(diagnostics, f, indent=2)
    print(f"Wrote diagnostics to {diag_path}")

    print("Done.")


if __name__ == "__main__":
    main()
