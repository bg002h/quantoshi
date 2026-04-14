"""Unit tests for btc_web/engines/custom_fit.py and _custom_time_presets.py."""
from __future__ import annotations

import math
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "btc_web"))
sys.path.insert(0, str(_ROOT))

from engines import custom_fit as cf  # noqa: E402
from _custom_time_presets import (  # noqa: E402
    CAL_PRESETS, BLK_PRESETS, CAL_PRESET_BY_KEY, BLK_PRESET_BY_KEY,
)


# ──────────────────────────────────────────────────────────────────────────
# _compute_weights
# ──────────────────────────────────────────────────────────────────────────

def test_compute_weights_none_returns_ones():
    t = np.linspace(1.0, 10.0, 100)
    w, degraded = cf._compute_weights(t, "none")
    assert np.allclose(w, 1.0)
    assert degraded is False


def test_compute_weights_inv_t_monotone_and_mean_one():
    t = np.linspace(1.0, 10.0, 100)
    w, degraded = cf._compute_weights(t, "inv_t")
    assert w[0] > w[-1]  # early > late
    assert abs(w.mean() - 1.0) < 1e-9
    assert degraded is False


def test_compute_weights_inv_sqrt_t_mean_one():
    t = np.linspace(1.0, 10.0, 100)
    w, _ = cf._compute_weights(t, "inv_sqrt_t")
    assert abs(w.mean() - 1.0) < 1e-9


def test_compute_weights_log_density_mean_one():
    rng = np.random.default_rng(0)
    t = rng.uniform(1.0, 100.0, 500)
    w, degraded = cf._compute_weights(t, "log_density")
    assert abs(w.mean() - 1.0) < 1e-6
    assert degraded is False


def test_compute_weights_small_n_falls_back_to_uniform():
    t = np.linspace(1.0, 5.0, 20)
    w, degraded = cf._compute_weights(t, "log_density")
    assert np.allclose(w, 1.0)
    assert degraded is True


def test_compute_weights_unknown_mode_returns_uniform():
    t = np.linspace(1.0, 10.0, 100)
    w, _ = cf._compute_weights(t, "nonsense_mode")
    assert np.allclose(w, 1.0)


# ──────────────────────────────────────────────────────────────────────────
# fit_pl
# ──────────────────────────────────────────────────────────────────────────

def _synth_pl(slope=5.0, intercept=-1.5, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0.5, 16.0, n)
    log_p = slope * np.log10(t) + intercept + rng.normal(0, 0.05, n)
    return t, 10 ** log_p


def test_fit_pl_recovers_slope():
    t, p = _synth_pl(slope=5.73, intercept=-1.20)
    fi = cf.FitInput(t=t, price=p, weighting="none")
    r = cf.fit_pl(fi)
    assert r is not None
    assert abs(r.params["slope"] - 5.73) < 0.05
    assert abs(r.params["intercept"] - (-1.20)) < 0.1
    assert r.r2 > 0.99
    assert r.n_samples > 100


def test_fit_pl_returns_none_when_insufficient_samples():
    fi = cf.FitInput(
        t=np.array([1.0, 2.0]), price=np.array([10.0, 20.0]),
        weighting="none",
    )
    assert cf.fit_pl(fi) is None


def test_fit_pl_drops_t_le_zero():
    """Regression: PL must drop exactly the t ≤ 0 rows (not one fewer, not
    one more). A broken mask that kept a single negative-t row would still
    yield a valid fit but fail this equality check."""
    t = np.linspace(-5.0, 10.0, 100)  # half negative
    p = np.where(t > 0, 10 ** (3.0 * np.log10(np.abs(t) + 0.01)), 1.0)
    fi = cf.FitInput(t=t, price=p, weighting="none")
    r = cf.fit_pl(fi)
    assert r is not None
    expected_n = int((t > 0).sum())
    assert r.n_samples == expected_n, (
        f"fit_pl kept {r.n_samples} samples but mask says {expected_n}")


def test_fit_pl_weighted_differs_from_unweighted():
    t, p = _synth_pl()
    r_none = cf.fit_pl(cf.FitInput(t=t, price=p, weighting="none"))
    r_inv = cf.fit_pl(cf.FitInput(t=t, price=p, weighting="inv_t"))
    assert r_none.params["slope"] != r_inv.params["slope"]


# ──────────────────────────────────────────────────────────────────────────
# fit_exp
# ──────────────────────────────────────────────────────────────────────────

def test_fit_exp_recovers_and_keeps_negative_t():
    rng = np.random.default_rng(0)
    n = 500
    t = np.linspace(-5.0, 10.0, n)
    log_p = 0.35 * t + 2.0 + rng.normal(0, 0.05, n)
    fi = cf.FitInput(t=t, price=10 ** log_p, weighting="none")
    r = cf.fit_exp(fi)
    assert r is not None
    assert abs(r.params["slope"] - 0.35) < 0.02
    assert abs(r.params["intercept"] - 2.0) < 0.1
    assert r.n_samples == n  # no mask


def test_fit_exp_returns_none_when_insufficient_samples():
    fi = cf.FitInput(
        t=np.array([1.0, 2.0]), price=np.array([10.0, 20.0]),
        weighting="none",
    )
    assert cf.fit_exp(fi) is None


# ──────────────────────────────────────────────────────────────────────────
# fit_qr
# ──────────────────────────────────────────────────────────────────────────

def test_fit_qr_recovers_median_slope():
    t, p = _synth_pl(slope=5.5, n=500)
    fi = cf.FitInput(t=t, price=p, weighting="none")
    r = cf.fit_qr(fi)
    assert r is not None
    assert 0.50 in r.y_plot
    assert abs(r.params["slopes"][0.50] - 5.5) < 0.15


def test_fit_qr_reduced_quantiles_when_n_between_10_and_30():
    rng = np.random.default_rng(0)
    n = 20
    t = np.linspace(1.0, 15.0, n)
    p = 10 ** (3.0 * np.log10(t) + rng.normal(0, 0.05, n))
    fi = cf.FitInput(t=t, price=p, weighting="none")
    r = cf.fit_qr(fi)
    assert r is not None
    assert set(r.y_plot.keys()) == {0.25, 0.50, 0.75}


def test_fit_qr_returns_none_when_n_below_10():
    t = np.linspace(1.0, 5.0, 5)
    p = 10 ** (3.0 * np.log10(t))
    fi = cf.FitInput(t=t, price=p, weighting="none")
    assert cf.fit_qr(fi) is None


# ──────────────────────────────────────────────────────────────────────────
# fit_bm_floor
# ──────────────────────────────────────────────────────────────────────────

def test_fit_bm_floor_returns_valid_result():
    t = np.linspace(0.5, 15.0, 500)
    rng = np.random.default_rng(0)
    log_p = 5.5 * np.log10(t) + 0.5 + rng.uniform(-0.1, 0.6, 500)
    p = 10 ** log_p
    fi = cf.FitInput(t=t, price=p, weighting="none")
    r = cf.fit_bm_floor(fi)
    assert r is not None
    assert "slope" in r.params
    assert "intercept" in r.params
    assert r.n_samples > 100


def test_fit_bm_floor_skips_when_n_lt_50():
    t = np.linspace(1.0, 10.0, 30)
    p = 10 ** (3.0 * np.log10(t))
    fi = cf.FitInput(t=t, price=p, weighting="none")
    assert cf.fit_bm_floor(fi) is None


# ──────────────────────────────────────────────────────────────────────────
# build_fit_input / cached arrays
# ──────────────────────────────────────────────────────────────────────────

def test_build_fit_input_calendar_later_t0_produces_negative_t():
    """Calendar mode with t0=2015-01-01 should have early rows with negative t."""
    fi = cf.build_fit_input(
        scale="calendar", t0="2015-01-01", weighting="none")
    assert fi.t[0] < 0  # 2010-07-17 vs 2015-01-01 → negative
    assert fi.t[-1] > 10  # latest row well past 2015
    assert fi.weighting == "none"


def test_build_fit_input_block_mode_uses_raw_offset(monkeypatch):
    fake_blocks = np.arange(0, 5000, dtype=np.int64) * 2
    fake_prices = np.linspace(1.0, 1000.0, len(fake_blocks))
    fake_dates = pd.DatetimeIndex(
        pd.date_range("2010-07-17", periods=len(fake_blocks)))
    monkeypatch.setattr(cf, "_BLOCKS", fake_blocks)
    monkeypatch.setattr(cf, "_PRICES", fake_prices)
    monkeypatch.setattr(cf, "_DATES", fake_dates)
    fi = cf.build_fit_input(scale="block", t0=1000, weighting="inv_t")
    assert fi.t[0] == -1000  # first block (0) minus t0=1000
    assert fi.weighting == "inv_t"


# ──────────────────────────────────────────────────────────────────────────
# Preset drift guards
# ──────────────────────────────────────────────────────────────────────────

def test_cal_presets_all_before_2016():
    for key, d, _ in CAL_PRESETS:
        assert d < date(2016, 1, 1), f"preset {key} is on/after 2016"


def test_preset_counts_frozen():
    assert len(CAL_PRESETS) == 6
    assert len(BLK_PRESETS) == 5


def test_presets_are_tuples_not_lists():
    assert isinstance(CAL_PRESETS, tuple)
    assert isinstance(BLK_PRESETS, tuple)
    for entry in CAL_PRESETS:
        assert isinstance(entry, tuple)
    for entry in BLK_PRESETS:
        assert isinstance(entry, tuple)


def test_preset_lookup_dicts_populated():
    assert CAL_PRESET_BY_KEY["optimal"][0] == date(2009, 7, 25)
    assert BLK_PRESET_BY_KEY["block_0"][0] == 0


# ──────────────────────────────────────────────────────────────────────────
# Duplicate-t regression (block-mode forward-fill semantics)
# ──────────────────────────────────────────────────────────────────────────

def test_duplicate_t_values_fit_ok():
    """Block mode produces duplicate-t rows for days with no new blocks;
    all fits must handle them cleanly — AND must not silently dedup them.
    Asserting n_samples == len(t) catches a hypothetical future refactor
    that `pd.DataFrame.drop_duplicates('t')` or similar."""
    t = np.array([1.0] * 20 + list(np.linspace(2.0, 20.0, 60)))
    rng = np.random.default_rng(0)
    p = 10 ** (5.0 * np.log10(t) + rng.normal(0, 0.05, len(t)))
    fi = cf.FitInput(t=t, price=p, weighting="none")
    n_expected = len(t)  # all t > 0 so no mask drops

    r_pl = cf.fit_pl(fi)
    assert r_pl is not None and math.isfinite(r_pl.r2)
    assert r_pl.n_samples == n_expected, "fit_pl silently deduped"

    r_exp = cf.fit_exp(fi)
    assert r_exp is not None and math.isfinite(r_exp.r2)
    assert r_exp.n_samples == n_expected, "fit_exp silently deduped"

    r_qr = cf.fit_qr(fi)
    assert r_qr is not None
    assert r_qr.n_samples == n_expected, "fit_qr silently deduped"

    r_bm = cf.fit_bm_floor(fi)
    assert r_bm is not None
    assert r_bm.n_samples == n_expected, "fit_bm_floor silently deduped"
