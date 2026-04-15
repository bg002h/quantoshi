"""Verification tests for the residual QR fit phase of build_bm_model.py.

These tests do NOT re-run the build — they inspect the on-disk
``model_data.pkl`` and ``model_data_resqr_diagnostics.json`` produced by
the most recent build. The build script is exercised end-to-end in CI by
running ``tools/build_bm_model.py`` before this test file.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "tools"))

import btc_core as bc  # noqa: E402

_PKL = _ROOT / "model_data.pkl"
_DIAG = _ROOT / "model_data_resqr_diagnostics.json"


# Expected flagship in-scope models (15). LPPL family excluded.
# HybPPL + EPPL config variants (72) are also fit and asserted separately.
EXPECTED_FLAGSHIP_MODELS = {
    "bub", "pl", "hybppl", "hybppl_dd",
    "hyb2l", "hyb2c", "hyb2b", "hyb4d",
    "eppl", "linppl", "exp", "pca",
    "grdy", "gomp", "bpl",
}
LPPL_KEYS = {"lppl", "lp2", "lp3", "lp4", "lppl_w", "lp2_w", "lp3_w", "lp4_w",
              "lp4_n13", "lp4_w_n13"}


@pytest.fixture(scope="module")
def diagnostics():
    if not _DIAG.exists():
        pytest.skip("model_data_resqr_diagnostics.json missing — rebuild first")
    with open(_DIAG) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def md():
    return bc.load_model_data(str(_PKL))


def test_diagnostics_not_aborted(diagnostics):
    assert diagnostics["aborted"] is False, diagnostics.get("reason")


def test_all_flagship_models_fit_ok(diagnostics):
    ok = {k for k, v in diagnostics["per_model"].items() if v["status"] == "ok"}
    missing = EXPECTED_FLAGSHIP_MODELS - ok
    assert not missing, f"flagship models missing: {missing}"


def test_config_variants_fit_ok(diagnostics):
    """All 36 HybPPL cfg_* + 36 Entropy PPL ecfg_* variants must fit so the
    chart builder's master-gate resolver can find resqr bundles for them."""
    ok = {k for k, v in diagnostics["per_model"].items() if v["status"] == "ok"}
    cfg_ok = {k for k in ok if k.startswith("cfg_")}
    ecfg_ok = {k for k in ok if k.startswith("ecfg_")}
    assert len(cfg_ok) == 36, f"expected 36 HybPPL cfg_* fits, got {len(cfg_ok)}"
    assert len(ecfg_ok) == 36, f"expected 36 EPPL ecfg_* fits, got {len(ecfg_ok)}"


def test_lppl_family_excluded(diagnostics):
    for key in LPPL_KEYS:
        assert key not in diagnostics["per_model"]


def test_interior_oos_coverage_within_tolerance(diagnostics):
    qs_arr = np.array([0.01, 0.05, 0.10, 0.25, 0.75, 0.90, 0.95, 0.99])
    interior = {0.05, 0.25, 0.75, 0.95}
    for key, entry in diagnostics["per_model"].items():
        if entry.get("status") != "ok":
            continue
        for q, cov in zip(qs_arr, entry["oos_coverage"]):
            if float(q) in interior:
                err = abs(cov - float(q))
                assert err <= 0.05, (
                    f"{key} q={q}: OOS cov {cov:.3f} deviates {err:.3f}"
                )


def test_raw_crossing_frac_below_warning_threshold(diagnostics):
    for key, entry in diagnostics["per_model"].items():
        if entry.get("status") != "ok":
            continue
        assert entry["raw_crossing_frac"] < 0.05, (
            f"{key} raw_crossing_frac={entry['raw_crossing_frac']:.3f}"
        )


def test_pkl_has_resqr_keys():
    m = bc.load_model_data(str(_PKL))
    assert m.resqr_coefs, "resqr_coefs should be populated after a rebuild"
    assert "bub" in m.resqr_coefs
    assert tuple(m.resqr_knots) == (3.0, 6.0, 9.0, 12.0)
    assert m.resqr_build_ts is not None


def test_diagnostics_n_samples_matches_price_history(diagnostics, md):
    """Every fitted model should have consumed the full price history."""
    expected_n = len(md.price_years)
    for key, entry in diagnostics["per_model"].items():
        if entry.get("status") != "ok":
            continue
        assert entry["n_samples"] == expected_n, (
            f"{key}: n_samples={entry['n_samples']} != {expected_n}"
        )
