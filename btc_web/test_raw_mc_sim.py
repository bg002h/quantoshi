"""Raw MC, Task 2: bootstrap simulator, cycle grid, debias, honest replay.

Frozen inputs throughout (spec "Tests"): prices truncated at 2024-04-20, a
halving, which is also "today"; the start price is that day's close.  Only the
spec's bounds are pinned -- its prototype figures are illustrative (F-22 N-2).
"""
import time

import numpy as np
import pandas as pd
import pytest

import raw_mc
from raw_mc import (RAW_SIMS_MAX, RAW_SRCS, RawRequest, cycle_table,
                    effective_checked, future_halvings, seed_for, simulate,
                    standardise, t_of)
from raw_mc_testing import PRICES, raw_prices  # noqa: F401

TODAY = pd.Timestamp("2024-04-20")
P0 = float(PRICES[TODAY])
WINDOW = (2010, 2024)


def _req(**kw):
    base = dict(variant="raw_cycle", trend="power", window=WINDOW, bins=5,
                checked=(), sims=400, years=40, today=TODAY, p0=P0,
                data_fp="csv", asof=TODAY)
    base.update(kw)
    return RawRequest(**base)


def _whole_cycles(res):
    """{cycle: [blocks]} for cycles whose grid has exactly 48 steps."""
    by = {}
    for b in res.schedule:
        by.setdefault(b["cycle"], []).append(b)
    return {c: bs for c, bs in by.items()
            if sum(b["steps"][1] - b["steps"][0] for b in bs) == 48}


# ── determinism / seed ──────────────────────────────────────────────────────

def test_same_request_same_paths(raw_prices):  # noqa: F811
    for v in RAW_SRCS:
        a, b = simulate(_req(variant=v)), simulate(_req(variant=v))
        assert a.ok and np.array_equal(a.paths, b.paths)


def test_seed_ignores_start_price(raw_prices):  # noqa: F811
    for v in RAW_SRCS:
        a, b = simulate(_req(variant=v)), simulate(_req(variant=v, p0=2 * P0))
        assert seed_for(_req(variant=v)) == seed_for(_req(variant=v, p0=2 * P0))
        assert np.array_equal(b.paths, 2 * a.paths)


def test_seed_uses_effective_checked():
    s = {c: seed_for(_req(checked=c)) for c in
         [(2, 0), (0, 2), (0, 2, 2), (0, 1, 2, 3, 4), (), (0, 2, 9, -1), (9,)]}
    assert s[(2, 0)] == s[(0, 2)] == s[(0, 2, 2)] == s[(0, 2, 9, -1)]
    assert s[(0, 1, 2, 3, 4)] == s[()] == s[(9,)]
    assert s[(0, 2)] != s[()]
    assert effective_checked((2, 0, 2), 5) == (0, 2)
    assert effective_checked((0, 1, 2, 3, 4), 5) is None
    assert effective_checked((), 5) is None
    assert effective_checked((7, -1), 5) is None


def test_seed_changes_with_seeded_fields():
    base = seed_for(_req())
    for kw in [dict(variant="raw_iid"), dict(trend="exp"), dict(window=(2011, 2024)),
               dict(bins=4), dict(sims=401), dict(years=30),
               dict(today=TODAY + pd.Timedelta(days=1)), dict(data_fp="x"),
               dict(asof=None)]:
        assert seed_for(_req(**kw)) != base, kw


def test_seed_is_process_stable():
    # Literal computed once from this implementation and pasted (blake2b of
    # canonical JSON, so it never varies by process / PYTHONHASHSEED).
    assert seed_for(_req()) == SEED_LITERAL


SEED_LITERAL = 781618610631287260


# ── shapes, sigma, debias ───────────────────────────────────────────────────

def test_shapes_and_start(raw_prices):  # noqa: F811
    for v in RAW_SRCS:
        res = simulate(_req(variant=v, sims=123, years=10))
        assert res.paths.shape == (123, 121)
        assert np.all(res.paths[:, 0] == P0)
        assert res.t_axis.shape == (121,)
        assert res.t_axis[0] == t_of(TODAY)
        assert res.t_axis[12] == t_of(pd.Timestamp("2025-04-20"))
        assert res.sigma_steps.shape == (120,)
        assert res.states is None and res.edges is None


def _check_sigma(res, today, years):
    end = today + pd.DateOffset(years=years)
    fh = [h for h in future_halvings(end + pd.DateOffset(years=5)) if h > today]
    dates = [today + pd.DateOffset(months=k) for k in range(years * 12)]
    bounds = [today] + fh
    for i in range(len(bounds) - 1):
        lo, hi = bounds[i], bounds[i + 1]
        ks = [k for k, d in enumerate(dates) if lo <= d < hi]
        if not ks:
            continue
        sig = res.sigma_steps[ks]
        assert np.all(sig == sig[0])
        t_mid = (t_of(lo) + t_of(hi)) / 2          # current cycle: remaining span
        assert sig[0] == pytest.approx(res.trend.sd_m(t_mid), rel=1e-12)
    jumps = np.flatnonzero(np.diff(res.sigma_steps) != 0) + 1
    # each jump is the first step of a new projected cycle
    assert all(any(dates[k - 1] < h <= dates[k] for h in fh) for k in jumps)


def test_sigma_held_constant_per_cycle(raw_prices):  # noqa: F811
    for trend in ("power", "exp"):
        res = simulate(_req(trend=trend, sims=10))
        _check_sigma(res, TODAY, 40)
    mid = pd.Timestamp("2026-11-08")
    res = simulate(_req(variant="raw_iid", sims=10, today=mid, asof=mid,
                        window=(2010, 2026), p0=float(PRICES[mid])))
    assert res.ok
    _check_sigma(res, mid, 40)
    # the current (partial) cycle's sigma uses [today, 2028-04-20)'s midpoint,
    # not the whole 2024-2028 cycle's
    whole = (t_of(pd.Timestamp("2024-04-20")) + t_of(pd.Timestamp("2028-04-20"))) / 2
    assert res.sigma_steps[0] != pytest.approx(res.trend.sd_m(whole), rel=1e-6)


@pytest.mark.parametrize("trend", ["power", "exp"])
def test_cycle_debias_expected_zero(raw_prices, trend):  # noqa: F811
    res = simulate(_req(trend=trend, sims=10))
    assert res.ok and res.z_shift != 0
    months = raw_mc._in_window(raw_mc.months_for(TODAY), WINDOW)
    z = standardise(months, res.trend).to_numpy()
    whole = _whole_cycles(res)
    assert len(whole) == 10
    for c, blocks in whole.items():
        e = 0.0
        for b in blocks:
            L = b["steps"][1] - b["steps"][0]
            sums = [np.sum(z[s:s + L] - res.z_shift) for s in b["cand"]]
            e += np.mean(sums)
        assert abs(e) <= 1e-9, (c, e)


def test_independent_and_replay_do_not_debias(raw_prices):  # noqa: F811
    assert simulate(_req(variant="raw_iid")).z_shift == 0
    assert simulate(_req(trend="none")).z_shift == 0


@pytest.mark.parametrize("variant", RAW_SRCS)
@pytest.mark.parametrize("trend", ["power", "exp"])
def test_median_on_trend_at_halvings(raw_prices, variant, trend):  # noqa: F811
    res = simulate(_req(variant=variant, trend=trend, sims=RAW_SIMS_MAX))
    assert res.ok
    dates = [TODAY + pd.DateOffset(months=k) for k in range(481)]
    mid_t = t_of(pd.DatetimeIndex(dates[:-1]) + pd.Timedelta(days=15))
    trend_log = np.log(P0) + np.concatenate([[0], np.cumsum(res.trend.mu_m(mid_t))])
    halvings = future_halvings(dates[-1])
    assert len(halvings) == 10
    for h in halvings:
        k = dates.index(h)
        lp = np.log(res.paths[:, k])
        assert abs(np.median(lp) - trend_log[k]) <= 0.25 * np.std(lp), (h, k)


def test_blocks_never_cross_halvings(raw_prices):  # noqa: F811
    res = simulate(_req(sims=10))
    months = raw_mc._in_window(raw_mc.months_for(TODAY), WINDOW)
    tab = cycle_table(raw_mc.months_for(TODAY), WINDOW)
    complete = set(tab.cycle[tab.complete])
    cyc = months.cycle.to_numpy()
    dates = [TODAY + pd.DateOffset(months=k) for k in range(481)]
    bounds = [TODAY] + future_halvings(dates[-1] + pd.DateOffset(years=5))
    covered = []
    for b in res.schedule:
        k0, k1 = b["steps"]
        L = k1 - k0
        assert 1 <= L <= 3
        covered.extend(range(k0, k1))
        # future slice within one projected cycle
        i = max(j for j, h in enumerate(bounds) if dates[k0] >= h)
        assert dates[k1 - 1] < bounds[i + 1]
        for s in b["cand"]:
            span = cyc[s:s + L]
            assert len(span) == L and len(set(span)) == 1 and span[0] in complete
    assert covered == list(range(480))


# ── honest replay, refusals, status ─────────────────────────────────────────

def test_honest_replay_status_quotes_fan_pace(raw_prices):  # noqa: F811
    for v in RAW_SRCS:
        res = simulate(_req(variant=v, trend="none"))
        assert res.ok
        text = " ".join(res.status)
        assert "Fan median grows" in text and "history" in text
        pace = np.expm1(np.median(np.log(res.paths[:, -1] / P0)) / 40) * 100
        assert f"{pace:+.0f} %/yr" in text


def test_short_window_refuses(raw_prices):  # noqa: F811
    for v in RAW_SRCS:
        res = simulate(_req(variant=v, window=(2025, 2026)))
        assert not res.ok and "36" in res.reason
        assert res.reason in res.status


def test_cycle_mode_needs_two_complete_cycles(raw_prices):  # noqa: F811
    res = simulate(_req(window=(2017, 2026)))
    assert not res.ok and "complete cycles" in res.reason
    assert simulate(_req(variant="raw_iid", window=(2017, 2026))).ok


def test_exp_fit_failure_draws_nothing(raw_prices, monkeypatch):  # noqa: F811
    monkeypatch.setattr(raw_mc, "fit_trend", lambda *a, **k: None)
    res = simulate(_req(trend="exp"))
    assert not res.ok
    assert res.reason == "Exponential trend could not be fitted to these cycles"
    assert res.paths.size == 0


def test_data_fp_must_match_registry(raw_prices):  # noqa: F811
    with pytest.raises(AssertionError):
        simulate(_req(data_fp="stale"))


def test_status_line_contents(raw_prices):  # noqa: F811
    iid = " ".join(simulate(_req(variant="raw_iid")).status)
    cyc = " ".join(simulate(_req()).status)
    for text in (iid, cyc):
        assert "n = 5.12" in text and "vol ∝ t^−0.49" in text
        assert "fitted on 4 cycles" in text
        assert "Current cycle" in text
    assert "No memory: understates multi-month swings" in iid
    assert "No memory" not in cyc
    assert "2012–2016, 2016–2020, 2020–2024" in cyc
    assert "widened" in cyc
    assert "widened" not in iid
    rough = " ".join(simulate(_req(sims=50)).status)
    assert "rough below 100" in rough
    assert "rough below 100" not in cyc
    # mid-cycle today: current cycle pace against the curve
    mid = pd.Timestamp("2026-11-08")
    text = " ".join(simulate(_req(today=mid, asof=mid, window=(2010, 2026),
                                  p0=float(PRICES[mid]))).status)
    assert "Current cycle (31 months)" in text and "vs curve" in text
    exp = " ".join(simulate(_req(trend="exp")).status)
    assert "Exponential" in exp and "fitted on 4 cycles" in exp


def test_performance(raw_prices):  # noqa: F811
    for v in RAW_SRCS:
        best = np.inf
        for _ in range(3):
            t0 = time.perf_counter()
            res = simulate(_req(variant=v, sims=RAW_SIMS_MAX, years=40))
            best = min(best, time.perf_counter() - t0)
        assert res.ok
        print(f"\n{v}: best of 3 = {best * 1000:.0f} ms (target 200 ms)")
        assert best < 0.6
