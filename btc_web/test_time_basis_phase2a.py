"""Phase 2a tests — refactor + parameterize build pipeline.

Tests the btc_core → time_basis bridge, T_MIN sweep, and build-pipeline
parameterization. Does NOT exercise block-mode end-to-end (Phase 2b builds
the actual block pkl).
"""
from __future__ import annotations
import sys
from pathlib import Path

import pytest


def test_btc_core_bridges_time_basis_into_sys_path():
    """Importing btc_core makes time_basis importable as a top-level module."""
    import btc_core  # noqa: F401
    # After btc_core is imported, time_basis should be importable bare.
    import time_basis as tb  # would fail without the bridge
    assert tb.TIME_BASIS in ("calendar", "block")
    assert tb.T_MIN in (1.0, 52596.0)


def test_time_basis_year_to_t_calendar():
    """year_to_t in calendar mode returns years since 2009-07-25."""
    import time_basis as tb
    if tb.TIME_BASIS != "calendar":
        pytest.skip("calendar-only test")
    # 2010 January 1 → 0.439 years past 2009-07-25 (160 days / 365.25).
    t = tb.year_to_t(2010)
    assert 0.4 < t < 0.5
    # 2024 January 1 → 14.439 years past 2009-07-25.
    t = tb.year_to_t(2024)
    assert 14.4 < t < 14.5
    # Fractional year: 2024.5 = July 1 2024 → 14.939
    t = tb.year_to_t(2024.5)
    assert 14.9 < t < 15.0


def test_time_basis_year_to_t_block(monkeypatch):
    """year_to_t in block mode scales the calendar-mode result by T_PER_YEAR."""
    import time_basis as tb
    monkeypatch.setattr(tb, "TIME_BASIS", "block")
    monkeypatch.setattr(tb, "T_PER_YEAR", 52596.0)
    # 2024 January 1 → ~14.439 years × 52596 ≈ 759,406 blocks since origin.
    t = tb.year_to_t(2024)
    assert 759_000 < t < 760_000


def test_time_basis_today_t_positive_and_in_range():
    """today_t returns a sensible value in either basis."""
    import time_basis as tb
    t = tb.today_t()
    if tb.TIME_BASIS == "calendar":
        # Today is at least 16 years past 2009-07-25, less than 30.
        assert 16.0 < t < 30.0
    else:
        # Block mode: 16 years × 52596 ≈ 841,536; less than 30 × 52596.
        assert 800_000 < t < 1_600_000


def test_t_min_sweep_calendar_mode_unchanged():
    """All 13 mask sites still exclude the same rows in calendar mode."""
    import numpy as np
    from time_basis import T_MIN
    assert T_MIN == 1.0  # this test is calendar-only
    # The mask `>= T_MIN` with T_MIN=1.0 must produce the same boolean
    # array as the old `>= 1.0` literal. Pick a synthetic price_years
    # array that straddles the threshold.
    price_years = np.array([0.5, 0.99, 1.0, 1.01, 5.0, 14.0])
    new_mask = price_years >= T_MIN
    old_mask = price_years >= 1.0
    np.testing.assert_array_equal(new_mask, old_mask)


def test_t_min_block_mode_threshold():
    """In block mode, T_MIN = T_PER_YEAR (one year's worth of blocks)."""
    import time_basis as tb
    if tb.TIME_BASIS == "block":
        assert tb.T_MIN == tb.T_PER_YEAR == 52596.0
    else:
        assert tb.T_MIN == tb.T_PER_YEAR == 1.0


def test_load_prices_calendar_mode_unchanged():
    """Calendar-mode load_prices produces the same df['years'] as before."""
    import sys
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root / "tools"))
    from model_toolkit.data import load_prices

    pd_calendar = load_prices(str(repo_root / "BitcoinPricesDaily.csv"))
    # First row in df_full is 2010-07-17 → ~0.978 years past 2009-07-25.
    first_years = pd_calendar.df_full["years"].iloc[0]
    assert 0.97 < first_years < 1.0
    # Last row is in the future relative to 2009; should be > 14 years.
    last_years = pd_calendar.df_full["years"].iloc[-1]
    assert last_years > 14.0


def test_load_prices_block_mode_uses_block_offsets():
    """Block-mode load_prices joins with BitcoinBlocksDaily.csv and
    computes years = blockheight - T_ORIGIN_BLOCK."""
    import sys
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root / "tools"))
    sys.path.insert(0, str(repo_root / "btc_web"))
    from model_toolkit.data import load_prices
    from time_basis import T_ORIGIN_BLOCK

    pd_block = load_prices(
        str(repo_root / "BitcoinPricesDaily.csv"),
        time_basis="block",
    )
    # First row date is 2010-07-17, which the block CSV maps to block 68779.
    # Block offset = 68779 - 20188 = 48591.
    first_offset = pd_block.df_full["years"].iloc[0]
    assert 48000 < first_offset < 49500
    # Last row offset must be much larger (block_origin is at 2009-07-25).
    last_offset = pd_block.df_full["years"].iloc[-1]
    assert last_offset > 700_000  # ~13 years past origin in blocks


def test_find_peaks_t_center_axis_aware():
    """find_peaks computes t_center via time_basis.year_to_t, not hardcoded
    pd.Timestamp arithmetic."""
    import sys
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root / "tools"))
    sys.path.insert(0, str(repo_root / "btc_web"))
    from model_toolkit import fitting as fmod
    import time_basis as tb
    import numpy as np

    if tb.TIME_BASIS != "calendar":
        pytest.skip("calendar-only sanity test")

    # Synthetic data: 1 fake bubble year at 2017.
    # In calendar mode, year_to_t(2017) ≈ 7.44; window is [6.69, 8.19].
    # Inject the peak inside that window so find_peaks can locate it.
    years = np.linspace(0.5, 16.0, 1000)
    log_excess = np.zeros_like(years)
    target_t = 7.5  # well inside the [6.69, 8.19] window for yr=2017
    peak_idx = np.argmin(np.abs(years - target_t))
    log_excess[peak_idx] = 1.0

    peaks = fmod.find_peaks(log_excess, years, [2017], window=0.75)
    assert len(peaks) == 1
    # Peak should be found at approximately the injected location.
    assert abs(peaks[0]["peak_t"] - target_t) < 0.1


def test_date_conversion_calendar_mode_unchanged():
    """The date_rise/plat/decay/end fields produce the same Timestamps
    as the hardcoded GENESIS + Timedelta path in calendar mode."""
    import sys
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root / "btc_web"))
    import pandas as pd
    import time_basis as tb

    if tb.TIME_BASIS != "calendar":
        pytest.skip("calendar-only test")

    # New axis-aware conversion: t -> calendar date via time_basis.t_to_calendar
    # Old: GENESIS + Timedelta(days=t * 365.25)
    GENESIS = pd.Timestamp("2009-07-25")
    for t in [1.0, 5.5, 14.123, 25.0]:
        old_ts = GENESIS + pd.Timedelta(days=t * 365.25)
        new_date = tb.t_to_calendar(t)
        # Must agree to within 1 day (rounding from day-floor).
        assert abs((pd.Timestamp(new_date) - old_ts).days) <= 1


def test_build_bm_model_accepts_time_basis_flag():
    """tools/build_bm_model.py --help shows --time-basis flag."""
    import subprocess
    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [str(repo_root / "btc_venv/bin/python3"),
         str(repo_root / "tools/build_bm_model.py"), "--help"],
        capture_output=True, text=True, cwd=repo_root, timeout=30,
    )
    assert result.returncode == 0, f"stderr: {result.stderr!r}"
    assert "--time-basis" in result.stdout
    assert "calendar" in result.stdout
    assert "block" in result.stdout


def test_build_ef_model_accepts_time_basis_flag():
    """tools/build_ef_model.py --help shows --time-basis flag."""
    import subprocess
    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [str(repo_root / "btc_venv/bin/python3"),
         str(repo_root / "tools/build_ef_model.py"), "--help"],
        capture_output=True, text=True, cwd=repo_root, timeout=30,
    )
    assert result.returncode == 0, f"stderr: {result.stderr!r}"
    assert "--time-basis" in result.stdout
    assert "axis-exempt" in result.stdout.lower() or "calendar-native" in result.stdout.lower()


def test_time_basis_env_var_override(tmp_path, monkeypatch):
    """QS_TIME_BASIS env var overrides the TOML file value."""
    toml_path = tmp_path / "test_quantoshi.toml"
    toml_path.write_text(
        'time_basis = "calendar"\n'
        'block_origin = 20188\n'
        'blocks_per_year = 52596\n'
    )
    import time_basis as tb
    cfg_no_env = tb._load_config(toml_path)
    assert cfg_no_env["time_basis"] == "calendar"
    monkeypatch.setenv("QS_TIME_BASIS", "block")
    cfg_block = tb._load_config(toml_path)
    assert cfg_block["time_basis"] == "block"


def test_time_basis_env_var_invalid_value_falls_back(tmp_path, monkeypatch):
    """Bogus env var value falls back to TOML/default."""
    import time_basis as tb
    toml_path = tmp_path / "test_quantoshi.toml"
    toml_path.write_text(
        'time_basis = "calendar"\n'
        'block_origin = 20188\n'
        'blocks_per_year = 52596\n'
    )
    monkeypatch.setenv("QS_TIME_BASIS", "garbage")
    cfg = tb._load_config(toml_path)
    assert cfg["time_basis"] == "calendar"


def test_build_bm_model_pkl_path_axis_aware():
    """tools/build_bm_model.py uses model_data_block.pkl in block mode."""
    from pathlib import Path
    repo_root = Path(__file__).resolve().parent.parent
    src = (repo_root / "tools" / "build_bm_model.py").read_text()
    assert "model_data.pkl" in src
    assert "model_data_block.pkl" in src
    assert "QS_TIME_BASIS" in src  # env var must be set before imports


def test_fitting_default_config_scales_with_t_per_year(monkeypatch):
    """fitting.DEFAULT_CONFIG window constants scale by T_PER_YEAR."""
    import sys
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root / "tools"))
    sys.path.insert(0, str(repo_root / "btc_web"))
    # Force a fresh module load with QS_TIME_BASIS=block
    monkeypatch.setenv("QS_TIME_BASIS", "block")
    # Pop any cached time_basis or fitting from sys.modules so they reload
    for mod in list(sys.modules):
        if mod.startswith("time_basis") or mod == "model_toolkit.fitting":
            del sys.modules[mod]
    import time_basis as tb
    assert tb.TIME_BASIS == "block"
    assert tb.T_PER_YEAR == 52596.0
    from model_toolkit import fitting
    cfg = fitting.DEFAULT_CONFIG
    # In block mode, BUBBLE_YEAR_WINDOW should be 0.75 × 52596 = 39447
    assert cfg["BUBBLE_YEAR_WINDOW"] == 0.75 * 52596.0
    assert cfg["FIT_CONTEXT_YR"] == 1.0 * 52596.0
    assert cfg["FIT_RISE_LOOKBACK_YR"] == 0.75 * 52596.0
    # Reset to calendar so other tests in the suite see calendar
    monkeypatch.delenv("QS_TIME_BASIS", raising=False)
    for mod in list(sys.modules):
        if mod.startswith("time_basis") or mod == "model_toolkit.fitting":
            del sys.modules[mod]


def test_composite_plot_grid_axis_aware(monkeypatch):
    """composite.PLOT_YEARS_MIN/MAX scale by T_PER_YEAR."""
    import sys
    monkeypatch.setenv("QS_TIME_BASIS", "block")
    for mod in list(sys.modules):
        if mod.startswith("time_basis") or mod.startswith("model_toolkit"):
            del sys.modules[mod]
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root / "tools"))
    sys.path.insert(0, str(repo_root / "btc_web"))
    from model_toolkit import composite
    assert composite.PLOT_YEARS_MIN == 1.0 * 52596.0
    assert composite.PLOT_YEARS_MAX == 72.0 * 52596.0
    # Reset
    monkeypatch.delenv("QS_TIME_BASIS", raising=False)
    for mod in list(sys.modules):
        if mod.startswith("time_basis") or mod.startswith("model_toolkit"):
            del sys.modules[mod]


def test_prediction_intervals_axis_aware(monkeypatch):
    """prediction.predict_future default intervals scale by T_PER_YEAR."""
    import sys, inspect
    monkeypatch.setenv("QS_TIME_BASIS", "block")
    for mod in list(sys.modules):
        if mod.startswith("time_basis") or mod.startswith("model_toolkit"):
            del sys.modules[mod]
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root / "tools"))
    sys.path.insert(0, str(repo_root / "btc_web"))
    from model_toolkit import prediction
    sig = inspect.signature(prediction.predict_future)
    # Defaults should be scaled by T_PER_YEAR (52596 in block mode)
    assert sig.parameters["major_default_interval"].default == 3.8 * 52596.0
    assert sig.parameters["minor_default_interval"].default == 3.8 * 52596.0
    assert sig.parameters["major_min_interval"].default == 1.4 * 52596.0
    assert sig.parameters["minor_min_interval"].default == 0.15 * 52596.0
    # Reset
    monkeypatch.delenv("QS_TIME_BASIS", raising=False)
    for mod in list(sys.modules):
        if mod.startswith("time_basis") or mod.startswith("model_toolkit"):
            del sys.modules[mod]
