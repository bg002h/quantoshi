"""Raw MC, Task 1: history, cycle table, trend fits, price registry.

Numbers pinned here are the output of tools/raw_mc_cycle_stats.py on
BitcoinPricesDaily.csv truncated at FROZEN (spec "Measured" section).
"""
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import raw_mc
from raw_mc import (HALVINGS, ORIGIN, Trend, cycle_table, fit_set, fit_trend,
                    future_halvings, get_prices, load_months, months_for,
                    set_prices, standardise, t_of)
from raw_mc_testing import FROZEN, PRICES, raw_prices, registry_restored  # noqa: F401

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_CLI = _ROOT / "tools" / "raw_mc_cycle_stats.py"


def _months():
    return load_months(PRICES, asof=FROZEN)


def _fs(window):
    """(fit set, months, window) -- the tail of fit_trend's signature."""
    m = _months()
    fset, _ = fit_set(cycle_table(m, window), cycle_table(m, None))
    return fset, m, window


def _where(window):
    m = _months()
    return fit_set(cycle_table(m, window), cycle_table(m, None))[1]


# ── cycle table ─────────────────────────────────────────────────────────────

def test_default_cycle_table():
    tab = cycle_table(load_months(PRICES, asof=FROZEN), None)
    assert list(tab.columns) == ["cycle", "first", "last", "months", "full",
                                 "complete", "t", "g", "s"]
    assert list(tab.months) == [27, 44, 46, 47, 31]
    assert list(tab.full) == [True, True, True, True, False]
    assert list(tab.complete) == [False, True, True, True, False]
    assert list(tab["first"]) == ["2010-08", "2012-11", "2016-07", "2020-05", "2024-04"]
    assert list(tab["last"]) == ["2012-10", "2016-06", "2020-04", "2024-03", "2026-10"]


def test_cycle_table_annual_units():
    tab = cycle_table(_months(), None)
    # spec: g_c = 12 * mean monthly log return, s_c = sqrt(12) * std; the CLI
    # prints exp(g)-1 as 895.11 % and s as 160.10 % for cycle 1.
    assert np.expm1(tab.g[0]) * 100 == pytest.approx(895.11, abs=0.01)
    assert tab.s[0] * 100 == pytest.approx(160.10, abs=0.01)
    assert tab.t[0] == pytest.approx(2.14, abs=0.005)


def test_window_filters_months_by_year():
    tab = cycle_table(_months(), (2013, 2026))
    assert list(tab.cycle) == [2, 3, 4, 5]
    assert tab.months[0] == 42 and tab["first"][0] == "2013-01"


def test_window_with_no_months_gives_empty_table_with_columns():
    tab = cycle_table(_months(), (1990, 1991))
    assert len(tab) == 0
    assert list(tab.columns) == ["cycle", "first", "last", "months", "full",
                                 "complete", "t", "g", "s"]
    fset, where = fit_set(tab, cycle_table(_months(), None))
    assert where == "all history" and len(fset) == 4


# ── fit set ─────────────────────────────────────────────────────────────────

def test_window_2013_refits_on_three_cycles():
    assert _where((2013, 2026)) == "window"
    fset, m, w = _fs((2013, 2026))
    assert len(fset) == 3
    tr = fit_trend("power", fset, m, w)
    assert tr.params["n"] == pytest.approx(5.93, abs=0.005)
    assert tr.params["b"] == pytest.approx(0.72, abs=0.005)


def test_window_2015_falls_back_to_all_history():
    assert _where((2015, 2026)) == "all history"
    fset, m, w = _fs((2015, 2026))
    assert len(fset) == 4
    tr = fit_trend("power", fset, m, w)
    assert tr.params["n"] == pytest.approx(5.12, abs=0.005)


# ── trend fits ──────────────────────────────────────────────────────────────

def test_default_power_fit():
    tr = fit_trend("power", *_fs(None))
    assert tr.kind == "power"
    assert tr.params["n"] == pytest.approx(5.12, abs=0.005)
    assert tr.params["b"] == pytest.approx(0.49, abs=0.005)
    assert tr.params["a"] == pytest.approx(2.46, abs=0.005)


def test_exp_fit_values():
    tr = fit_trend("exp", *_fs(None))
    assert tr.kind == "exp"
    assert tr.params["alpha"] == pytest.approx(1.18, abs=0.005)
    assert tr.params["beta"] == pytest.approx(-0.179, abs=0.005)
    assert tr.params["gamma"] == pytest.approx(0.66, abs=0.005)
    assert tr.params["delta"] == pytest.approx(-0.087, abs=0.005)


def test_exp_fit_failure_returns_none(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("Optimal parameters not found")
    monkeypatch.setattr(raw_mc, "curve_fit", boom)
    assert fit_trend("exp", *_fs(None)) is None
    # the power law does not use curve_fit and is unaffected
    assert fit_trend("power", *_fs(None)) is not None


@pytest.mark.filterwarnings("ignore:Covariance of the parameters")
def test_exp_start_guess_never_takes_log_of_nonpositive_mean(monkeypatch):
    """Spec: 'the start guess must not take the log of a non-positive mean'.
    Force every full cycle's growth negative and capture curve_fit's p0."""
    fset, m, w = _fs(None)
    neg = fset.assign(g=-np.abs(fset.g))
    seen = []
    real = raw_mc.curve_fit

    def spy(f, x, y, p0=None, **k):
        seen.append(p0)
        return real(f, x, y, p0=p0, **k)
    monkeypatch.setattr(raw_mc, "curve_fit", spy)
    with np.errstate(divide="raise", invalid="raise"):   # log(<=0) would raise here
        fit_trend("exp", neg, m, w)                      # (fit may fail: only p0 matters)
    assert seen[0][0] == pytest.approx(np.log(1e-3))
    assert all(np.isfinite(p) for p0 in seen for p in p0)


def test_underdetermined_fit_set_returns_none():
    fset, m, w = _fs(None)
    for kind in ("power", "exp"):
        assert fit_trend(kind, fset.iloc[:1], m, w) is None


def test_trend_none_carries_window_pooled_moments():
    fset, m, w = _fs((2013, 2026))
    tr = fit_trend("none", fset, m, w)
    pooled = m[m.index.year >= 2013].r
    assert tr.kind == "none"
    assert tr.mu_m(5.0) == pytest.approx(pooled.mean())
    assert tr.sd_m(5.0) == pytest.approx(pooled.std())
    assert tr.g(5.0) == pytest.approx(12 * pooled.mean())
    assert tr.s(5.0) == pytest.approx(np.sqrt(12) * pooled.std())
    t = np.array([1.0, 7.0, 30.0])           # constant, shape-preserving
    assert tr.mu_m(t).shape == (3,) and np.ptp(tr.mu_m(t)) == 0.0


def test_unknown_trend_kind_raises():
    with pytest.raises(ValueError):
        fit_trend("linear", *_fs(None))


def test_trend_curves_and_monthly_scaling():
    power = Trend("power", dict(n=5.0, a=2.0, b=0.5))
    exp = Trend("exp", dict(alpha=1.0, beta=-0.1, gamma=0.5, delta=-0.05))
    t = np.array([4.0, 16.0])
    assert power.g(4.0) == pytest.approx(1.25)
    assert power.s(4.0) == pytest.approx(1.0)                # 2 * 4 ** -0.5
    np.testing.assert_allclose(power.g(t), [1.25, 0.3125])
    assert exp.g(10.0) == pytest.approx(np.exp(0.0))
    assert exp.s(10.0) == pytest.approx(np.exp(0.0))
    for tr in (power, exp):                                   # annual -> monthly
        assert tr.mu_m(7.0) == pytest.approx(tr.g(7.0) / 12)
        assert tr.sd_m(7.0) == pytest.approx(tr.s(7.0) / np.sqrt(12))


# ── months / phi / cycle assignment ─────────────────────────────────────────

def test_incomplete_trailing_month_dropped():
    m = load_months(PRICES, asof=pd.Timestamp("2026-11-08"))
    assert m.index[-1] == pd.Timestamp("2026-10-31")
    m = load_months(PRICES, asof=pd.Timestamp("2026-10-15"))
    assert m.index[-1] == pd.Timestamp("2026-09-30")
    # a month whose last day IS the asof date is complete
    m = load_months(PRICES, asof=pd.Timestamp("2026-10-31"))
    assert m.index[-1] == pd.Timestamp("2026-10-31")


def test_load_months_columns_and_index():
    m = _months()
    assert list(m.columns) == ["r", "t", "cycle", "phi"]
    assert m.index.is_month_end.all()
    assert m.index[0] == pd.Timestamp("2010-08-31")     # first return needs two closes


def test_phi_clipped():
    m = _months()
    assert m.loc["2012-11-30", "phi"] == 0.0      # midpoint precedes the 2012-11-28 halving
    assert m.loc["2012-11-30", "cycle"] == 2      # month-end falls after it
    assert (m.phi >= 0).all() and (m.phi < 1).all()
    assert m.loc["2012-10-31", "cycle"] == 1


def test_asof_none_uses_all_prices():
    full = load_months(PRICES)
    assert full.index[-1] <= PRICES.index[-1]
    assert load_months(PRICES, asof=PRICES.index[-1]).equals(full)


def test_t_of():
    assert t_of(ORIGIN) == 0.0
    assert isinstance(t_of(ORIGIN + pd.Timedelta(days=365)), float)
    assert t_of(ORIGIN + pd.Timedelta(days=365)) == pytest.approx(365 / 365.25)
    arr = t_of(pd.DatetimeIndex([ORIGIN, ORIGIN + pd.Timedelta(days=730)]))
    assert isinstance(arr, np.ndarray)
    np.testing.assert_allclose(arr, [0.0, 730 / 365.25])


# ── standardisation of thin partial cycles (F-22 N-4) ───────────────────────

def _thin_setup(asof):
    m = load_months(PRICES, asof=pd.Timestamp(asof))
    tab = cycle_table(m, None)
    fset, _ = fit_set(tab, tab)
    return m, fit_trend("power", fset, m, None)


def test_thin_partial_cycle_uses_trend_standardisation():
    m, tr = _thin_setup("2024-09-30")
    c5 = m[m.cycle == 5]
    assert len(c5) == 6                              # < 12 data months
    z = standardise(m, tr)
    assert np.isfinite(z[c5.index]).all()
    want = (c5.r - tr.mu_m(c5.t.values)) / tr.sd_m(c5.t.values)
    np.testing.assert_allclose(z[c5.index].to_numpy(), want.to_numpy())


def test_single_month_cycle_would_be_nan_on_own_sample():
    m, tr = _thin_setup("2024-04-30")
    c5 = m[m.cycle == 5]
    assert len(c5) == 1
    assert np.isnan(c5.r.std())                      # the bug N-4 guards against
    assert np.isfinite(standardise(m, tr)[c5.index]).all()


def test_cycles_with_enough_months_standardise_on_their_own_sample():
    m, tr = _thin_setup("2026-11-08")
    z = standardise(m, tr)
    for c in (1, 2, 3, 4, 5):                        # all have >= 12 months here
        zc = z[m.cycle == c]
        assert zc.mean() == pytest.approx(0.0, abs=1e-12)
        assert zc.std() == pytest.approx(1.0, abs=1e-12)


def test_trend_none_thin_cycle_uses_pooled_moments():
    m, _ = _thin_setup("2024-09-30")
    tr = fit_trend("none", m.iloc[:0], m, None)
    c5 = m[m.cycle == 5]
    z = standardise(m, tr)
    np.testing.assert_allclose(z[c5.index].to_numpy(),
                               ((c5.r - m.r.mean()) / m.r.std()).to_numpy())


# ── halvings ────────────────────────────────────────────────────────────────

def test_halvings_match_figures_common():
    from figures.common import _HALVING_EPOCHS, _HALVING_KNOWN_DATES
    assert HALVINGS[0] == pd.Timestamp("2009-01-03")
    assert list(HALVINGS[1:]) == [pd.Timestamp(d) for d in _HALVING_KNOWN_DATES]
    assert future_halvings(pd.Timestamp("2080-12-31")) == \
        [ts for ts, est in _HALVING_EPOCHS if est]


def test_future_halvings_bounds():
    assert future_halvings(pd.Timestamp("2024-04-20")) == []
    assert future_halvings(pd.Timestamp("2028-04-19")) == []
    assert future_halvings(pd.Timestamp("2028-04-20")) == [pd.Timestamp("2028-04-20")]
    assert future_halvings(pd.Timestamp("2036-01-01")) == [
        pd.Timestamp("2028-04-20"), pd.Timestamp("2032-04-20")]


# ── price registry ──────────────────────────────────────────────────────────

def test_price_registry_round_trip(raw_prices):  # noqa: F811
    s = pd.Series([1.0, 2.0], index=pd.to_datetime(["2020-01-01", "2020-01-02"]))
    set_prices(s, "fp1")
    got = get_prices()
    assert got[0] is s and got[1] == "fp1"
    raw_mc._set_registry(raw_mc._UNSET)
    with pytest.raises(RuntimeError, match="raw_mc prices not registered"):
        get_prices()


def test_fixture_registers_the_csv_series(raw_prices):  # noqa: F811
    assert get_prices()[0] is PRICES and get_prices()[1] == "csv"


def test_registry_restored_puts_back_the_previous_registration():
    outer = pd.Series([1.0], index=pd.to_datetime(["2020-01-01"]))
    inner = pd.Series([2.0], index=pd.to_datetime(["2020-01-02"]))
    with registry_restored():
        set_prices(outer, "outer")
        with registry_restored():
            set_prices(inner, "inner")
            assert get_prices()[1] == "inner"
        got = get_prices()
        assert got[0] is outer and got[1] == "outer"


def test_registry_restored_unregisters_when_nothing_was_registered():
    with registry_restored():
        raw_mc._set_registry(raw_mc._UNSET)
        with registry_restored():
            set_prices(PRICES, "x")
            assert get_prices()[1] == "x"
        assert raw_mc._registry is raw_mc._UNSET


def test_months_for_cache_keys_on_asof(raw_prices):  # noqa: F811
    a = months_for(None)
    b = months_for(pd.Timestamp("2024-04-20"))
    assert not a.equals(b) and len(b) < len(a)
    assert months_for(None) is a                       # cached
    assert months_for(pd.Timestamp("2024-04-20")) is b
    assert a.equals(load_months(PRICES))
    set_prices(PRICES, "csv")                          # re-register: cache cleared
    assert months_for(None) is not a
    assert months_for(None).equals(a)


def test_months_for_keys_on_data_fp(raw_prices):  # noqa: F811
    a = months_for(None)
    set_prices(PRICES.iloc[:-400], "csv2")
    b = months_for(None)
    assert len(b) < len(a)


def test_months_for_requires_registration():
    with registry_restored():
        raw_mc._set_registry(raw_mc._UNSET)
        with pytest.raises(RuntimeError, match="raw_mc prices not registered"):
            months_for(None)


def test_set_prices_calls_registered_clearers(raw_prices):  # noqa: F811
    calls = []
    fn = lambda: calls.append("cleared")                      # noqa: E731
    raw_mc._CACHE_CLEARERS.append(fn)
    try:
        set_prices(PRICES, "csv")
        assert calls == ["cleared"]
        raw_mc._set_registry(raw_mc._UNSET)                   # restore path clears too
        assert calls == ["cleared", "cleared"]
    finally:
        raw_mc._CACHE_CLEARERS.remove(fn)


# ── CLI and import hygiene ──────────────────────────────────────────────────

def test_cli_matches_module():
    out = subprocess.run([sys.executable, str(_CLI)], capture_output=True,
                         text=True, cwd=_ROOT, timeout=120)
    assert out.returncode == 0, out.stderr
    assert "fit on 4 full cycles: power n = 5.12" in out.stdout
    assert "fit set: window" in out.stdout
    out = subprocess.run([sys.executable, str(_CLI), "--from", "2015"],
                         capture_output=True, text=True, cwd=_ROOT, timeout=120)
    assert out.returncode == 0, out.stderr
    assert "fit set: all history" in out.stdout


def test_import_light():
    code = (
        "import sys; sys.path.insert(0, %r); import raw_mc; "
        "bad = [m for m in ('figures', '_app_ctx', 'mc_cache', 'btc_core', 'app') "
        "if m in sys.modules]; print(bad); sys.exit(1 if bad else 0)"
    ) % str(_ROOT / "btc_web")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, cwd=_ROOT, timeout=120)
    assert out.returncode == 0, out.stdout + out.stderr
