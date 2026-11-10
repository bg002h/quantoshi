"""Raw MC, Task 3: regime steering on the level-vs-trend state.

Spec "Regime filter (Raw) -- level relative to trend" + plan F-22 N-3 (D_0).
Frozen inputs (today = asof = the 2024-04-20 halving) except where a
mid-month today is the point of the test.
"""
import itertools
import re
import time

import numpy as np
import pandas as pd
import pytest

import raw_mc
from raw_mc import (RAW_SIMS_MAX, RAW_SRCS, RAW_TRENDS, bin_edges, bin_labels,
                    regime_options, seed_for, simulate, state_history, t_of)
from raw_mc_testing import PRICES, RAW_FP, SIM_TODAY, _req, raw_prices  # noqa: F401

SELECTIONS = [(i,) for i in range(5)] + list(itertools.combinations(range(5), 2))  # 15
MID = pd.Timestamp("2026-11-08")          # a mid-month "today"


def _closed_range(res, checked):
    return (min(res.edges[i] for i in checked), max(res.edges[i + 1] for i in checked))


def _median_state_at_40y(res):
    assert res.states.shape[1] == 481
    return float(np.median(res.states[:, -1]))


def _hist(trend_kind, window=(2010, 2024), asof=SIM_TODAY):
    months = raw_mc.months_for(asof)
    trend = simulate(_req(trend=trend_kind, sims=1, years=10, window=window,
                          asof=asof, today=asof, p0=float(PRICES[asof]))).trend
    return months, trend


# ── steering bounds (90 cases) ──────────────────────────────────────────────

@pytest.mark.parametrize("variant", RAW_SRCS)
@pytest.mark.parametrize("trend", RAW_TRENDS)
@pytest.mark.parametrize("checked", SELECTIONS)
def test_steering_targets(raw_prices, variant, trend, checked):  # noqa: F811
    res = simulate(_req(variant, trend, checked=checked, sims=1000))
    assert res.ok
    assert res.unchecked_share <= 0.05          # months 13.. only
    lo, hi = _closed_range(res, checked)
    assert lo - 0.5 <= _median_state_at_40y(res) <= hi + 0.5


# ── state definition ────────────────────────────────────────────────────────

@pytest.mark.parametrize("trend", RAW_TRENDS)
def test_state_history_definition(raw_prices, trend):  # noqa: F811
    months, tr = _hist(trend)
    states, S = state_history(months, tr, (2010, 2024))
    win = raw_mc._in_window(months, (2010, 2024))
    assert states.shape == (len(win),)
    # D_k = log P_k - sum mu - C, from the month-end closes themselves
    me = PRICES[PRICES.index <= SIM_TODAY].resample("ME").last()
    logp = np.log(me.loc[win.index].to_numpy())
    d = logp - np.cumsum(tr.mu_m(win.t.to_numpy()))
    d -= d.mean()
    assert S == pytest.approx(d.std(), rel=1e-9)
    np.testing.assert_allclose(states, d / S, atol=1e-9)
    assert abs(states.mean()) < 1e-9 and states.std() == pytest.approx(1.0)


@pytest.mark.parametrize("variant", RAW_SRCS)
@pytest.mark.parametrize("trend", RAW_TRENDS)
def test_simulated_state_recursion(raw_prices, variant, trend):  # noqa: F811
    """D_{k+1} = D_k + r_{k+1} - mu(t_{k+1}), with r read back from the paths."""
    res = simulate(_req(variant, trend, checked=(1, 2), sims=50, years=10))
    dates = [SIM_TODAY + pd.DateOffset(months=k) for k in range(121)]
    mu = res.trend.mu_m(t_of(pd.DatetimeIndex(dates[:-1]) + pd.Timedelta(days=15)))
    _, S = state_history(raw_mc.months_for(SIM_TODAY), res.trend, (2010, 2024))
    r = np.diff(np.log(res.paths), axis=1)
    want = res.states[:, :1] * S + np.concatenate(
        [np.zeros((50, 1)), np.cumsum(r - mu, axis=1)], axis=1)
    np.testing.assert_allclose(res.states * S, want, atol=1e-9)


def test_d0_includes_partial_month(raw_prices):  # noqa: F811
    """F-22 N-3: D_0 = D_lastME + log(P_today / P_lastME) - mu(t_today) * f."""
    window = (2010, 2026)
    p0 = float(PRICES[MID]) * 1.07               # live price != last close
    res = simulate(_req(trend="power", checked=(2,), sims=20, years=10,
                        window=window, today=MID, asof=MID, p0=p0))
    assert res.ok
    months = raw_mc.months_for(MID)
    states, S = state_history(months, res.trend, window)
    last_me = months.index[-1]
    assert last_me == pd.Timestamp("2026-10-31")
    f = 8 / 30
    d0 = (states[-1] * S + np.log(p0 / float(PRICES[last_me]))
          - res.trend.mu_m(t_of(MID)) * f)
    np.testing.assert_allclose(res.states[:, 0], d0 / S, rtol=0, atol=1e-12)
    no_f = (states[-1] * S + np.log(p0 / float(PRICES[last_me]))) / S
    assert abs(res.states[0, 0] - no_f) > 1e-4   # the partial-month term matters


# ── bins and labels ─────────────────────────────────────────────────────────

def test_outer_bins_closed(raw_prices):  # noqa: F811
    months, tr = _hist("power")
    states, _ = state_history(months, tr, (2010, 2024))
    edges = bin_edges(states, 5)
    assert len(edges) == 6
    assert edges[0] == states.min() and edges[-1] == states.max()
    assert np.all(np.diff(edges) > 0)
    counts = np.histogram(states, edges)[0]
    assert counts.max() - counts.min() <= 1          # equal-count
    res = simulate(_req(checked=(0,), sims=10))
    np.testing.assert_array_equal(res.edges, edges)


def test_labels_format(raw_prices):  # noqa: F811
    months, tr = _hist("power")
    labels = bin_labels(bin_edges(state_history(months, tr, (2010, 2024))[0], 5))
    assert len(labels) == 5
    pat = re.compile(r"^[−+]?\d\.\dσ to [−+]?\d\.\dσ vs trend$")
    for lab in labels:
        assert pat.match(lab), lab
        assert "-" not in lab                          # U+2212, never hyphen
    assert labels[0].startswith("−")
    assert bin_labels(np.array([-1.5, -0.8, -0.04, 0.8])) == [
        "−1.5σ to −0.8σ vs trend", "−0.8σ to 0.0σ vs trend", "0.0σ to +0.8σ vs trend"]


def test_regime_options_match_simulate(raw_prices):  # noqa: F811
    opts = regime_options(5, (2010, 2024), "power", asof=SIM_TODAY)
    assert [o["value"] for o in opts] == list(range(5))
    res = simulate(_req(checked=(0,), sims=10))
    assert [o["label"] for o in opts] == res.bin_labels
    opts[0]["label"] = "mutated"                       # caller copies are free
    assert regime_options(5, [2010, 2024], "power", asof=SIM_TODAY)[0]["label"] \
        == res.bin_labels[0]
    assert len(regime_options(3, (2010, 2024), "exp", asof=SIM_TODAY)) == 3


def test_regime_options_cleared_by_set_prices(raw_prices):  # noqa: F811
    """F-23 M-1: set_prices clears the regime-options cache even when the
    data_fp string is unchanged."""
    assert regime_options.cache_clear in raw_mc._CACHE_CLEARERS
    before = regime_options(5, (2010, 2026), "power")
    raw_mc.set_prices(PRICES[PRICES.index <= "2021-12-31"], RAW_FP)
    after = regime_options(5, (2010, 2026), "power")
    assert [o["label"] for o in before] != [o["label"] for o in after]


# ── no effective filter, status, cycle slots ────────────────────────────────

@pytest.mark.parametrize("variant", RAW_SRCS)
def test_no_effective_filter_variants(raw_prices, variant):  # noqa: F811
    base = simulate(_req(variant, checked=()))
    for checked in [(), (0, 1, 2, 3, 4), (7, 9)]:
        res = simulate(_req(variant, checked=checked))
        assert raw_mc.effective_checked(checked, 5) is None
        assert seed_for(_req(variant, checked=checked)) == seed_for(_req(variant))
        assert np.array_equal(res.paths, base.paths)
        assert res.states is None and res.unchecked_share == 0.0
        assert not any("steered" in s for s in res.status)


@pytest.mark.parametrize("variant", RAW_SRCS)
def test_note_reports_measured_share(raw_prices, variant):  # noqa: F811
    res = simulate(_req(variant, checked=(0,), sims=300))
    note = [s for s in res.status if "steered away from unchecked bins" in s]
    assert len(note) == 1 and "% of months" in note[0]
    assert note[0].startswith("Raw: paths are steered away from unchecked bins (")
    assert f"{res.unchecked_share * 100:.1f} % of months" in note[0]


@pytest.mark.parametrize("checked", [(0,), (2,), (1, 3)])
def test_cycle_block_redraw_keeps_slot(raw_prices, checked):  # noqa: F811
    """Every steered month copies pool[c + j] for a candidate c of ITS block's
    slot at ITS offset j, and each contiguous run of a block copied from one
    candidate lies in one complete cycle (block integrity per contiguous run:
    review I-1 fallback ruling).  The month's z is rebuilt from the paths, so
    a draw that disagrees with ``sources`` fails too (review I-2)."""
    res = simulate(_req(checked=checked, sims=300))
    assert res.ok and res.sources is not None
    win = raw_mc._in_window(raw_mc.months_for(SIM_TODAY), (2010, 2024))
    pool = raw_mc.standardise(win, res.trend).to_numpy()
    wcyc = win.cycle.to_numpy()
    dates = [SIM_TODAY + pd.DateOffset(months=k) for k in range(481)]
    mu = res.trend.mu_m(t_of(pd.DatetimeIndex(dates[:-1]) + pd.Timedelta(days=15)))
    z = (np.diff(np.log(res.paths), axis=1) - mu) / res.sigma_steps + res.z_shift
    np.testing.assert_allclose(z, pool[res.sources], atol=1e-8)
    redrawn = 0
    for b in res.schedule:
        k0, k1 = b["steps"]
        src = res.sources[:, k0:k1]
        start = src - np.arange(k1 - k0)                  # implied candidate per month
        assert np.isin(start, b["cand"]).all(), b["steps"]
        same_run = start[:, 1:] == start[:, :-1]           # month j continues j-1's copy
        assert (wcyc[src[:, 1:]][same_run] == wcyc[src[:, :-1]][same_run]).all()
        redrawn += int(np.sum(np.any(start != start[:, :1], axis=1)))
    assert redrawn > 0                                    # suffix redraws happened
    base = simulate(_req(sims=300))
    assert [b["steps"] for b in res.schedule] == [b["steps"] for b in base.schedule]
    assert base.sources is None


# ── performance ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("variant", RAW_SRCS)
def test_performance_filtered(raw_prices, variant):  # noqa: F811
    """Hard bound on CPU time (process_time), so xdist / host load cannot
    flake it; wall time is printed alongside (review I-3)."""
    for checked in [(0,), (4,), (2,), (1, 3)]:
        cpu = wall = np.inf
        for _ in range(3):
            c0, w0 = time.process_time(), time.perf_counter()
            res = simulate(_req(variant, "power", checked=checked,
                                sims=RAW_SIMS_MAX, years=40))
            cpu = min(cpu, time.process_time() - c0)
            wall = min(wall, time.perf_counter() - w0)
        assert res.ok and res.states is not None
        print(f"\n{variant} {checked}: best of 3 cpu {cpu * 1000:.0f} ms, wall "
              f"{wall * 1000:.0f} ms (target 200 ms, hard bound 600 ms cpu)")
        assert cpu < 0.6
