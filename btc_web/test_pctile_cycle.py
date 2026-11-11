"""Percentile cycle fit (Tab 1 Percentile pill, 'Cycle fit').

One calendar-time sinusoid c + A·cos(2πf·t + φ) fitted to a model's
percentile series, extrapolated with a 1-yr block-bootstrap band.
Spec: docs/superpowers/specs/2026-11-10-percentile-cycle-fit-design.md
"""
import numpy as np
import pytest

import pctile_cycle as pc


def _synthetic(period=3.6, A=30.0, c=50.0, phi=1.0, noise=3.0, years=16.0,
               seed=1, ar=0.0):
    rng = np.random.default_rng(seed)
    t = np.arange(1.0, 1.0 + years, 1.0 / 365.25)
    e = rng.normal(0.0, noise, t.size)
    if ar:
        for i in range(1, t.size):
            e[i] = ar * e[i - 1] + e[i] * np.sqrt(1 - ar * ar)
    y = c + A * np.cos(2 * np.pi * t / period + phi) + e
    return t, y


def _wrap(d):
    return (d + np.pi) % (2 * np.pi) - np.pi


class TestMath:
    def test_recovers_known_sinusoid(self):
        t, y = _synthetic()
        f = pc.fit_cycle(t, y)
        assert abs(f.period - 3.6) < 0.02
        assert abs(f.A - 30.0) < 1.5
        assert abs(f.c - 50.0) < 1.0
        assert abs(_wrap(f.phi - 1.0)) < 0.05
        assert f.r2 > 0.95
        assert f.n == t.size

    def test_period_search_is_bounded(self):
        t = np.arange(1.0, 15.0, 1.0 / 365.25)
        y = 50 + 30 * np.cos(2 * np.pi * t / 0.5)
        f = pc.fit_cycle(t, y)
        assert pc.PMIN <= f.period <= pc.PMAX

    def test_wave_broadcasts(self):
        x = np.linspace(1, 20, 50)
        assert pc.wave(50.0, 30.0, 0.28, 1.0, x).shape == (50,)
        P = np.tile([50.0, 30.0, 0.28, 1.0], (5, 1))
        out = pc.wave(P[:, 0], P[:, 1], P[:, 2], P[:, 3], x)
        assert out.shape == (5, 50)
        np.testing.assert_allclose(out[2], pc.wave(50.0, 30.0, 0.28, 1.0, x))

    def test_bootstrap_deterministic_and_shaped(self):
        t, y = _synthetic(ar=0.99)
        f = pc.fit_cycle(t, y)
        a = pc.bootstrap_params(t, y, f)
        b = pc.bootstrap_params(t, y, f)
        assert a.shape == (pc.B_DEFAULT, 4)
        np.testing.assert_array_equal(a, b)

    def test_band_widens_into_the_future(self):
        t, y = _synthetic(noise=15.0, ar=0.995)
        f = pc.fit_cycle(t, y)
        P = pc.bootstrap_params(t, y, f)
        x = np.array([t[-1], t[-1] + 10.0])
        lo, hi = pc.band(P, x)
        assert np.all(lo <= hi)
        assert np.all((lo >= 0) & (hi <= 100))
        assert (hi[1] - lo[1]) > (hi[0] - lo[0])

    def test_turns_are_extrema_of_the_wave(self):
        f = pc.CycleFit(c=50.0, A=30.0, f=1 / 3.6, phi=1.0, r2=0.9, n=100)
        tt = pc.turns(f, 1.0, 30.0)
        assert len(tt) >= 15
        kinds = [k for _, _, k in tt]
        assert all(a != b for a, b in zip(kinds, kinds[1:]))
        assert [x for x, _, _ in tt] == sorted(x for x, _, _ in tt)
        for x, v, k in tt:
            assert 1.0 <= x <= 30.0
            near = pc.wave(f.c, f.A, f.f, f.phi, np.array([x - 0.01, x + 0.01]))
            if k == "peak":
                assert v == pytest.approx(80.0) and np.all(near <= v + 1e-9)
            else:
                assert v == pytest.approx(20.0) and np.all(near >= v - 1e-9)

    def test_turn_timing_brackets_central_turn(self):
        t, y = _synthetic(noise=10.0, ar=0.99)
        f = pc.fit_cycle(t, y)
        P = pc.bootstrap_params(t, y, f)
        for x, _, k in pc.turns(f, t[-1], t[-1] + 12):
            lo, hi = pc.turn_timing(P, x, k)
            assert lo <= x <= hi
            assert hi - lo < f.period          # one turn, not a neighbour

    def test_hindcast_pure_sinusoid_wins(self):
        t = np.arange(1.0, 17.0, 1.0 / 365.25)
        y = 50 + 30 * np.cos(2 * np.pi * t / 3.6 + 0.3)
        h = pc.hindcast(t, y)
        assert h["n"] >= 10
        assert h["win_rate"] == 1.0
        assert h["rmse_wave"] < 0.05 * h["rmse_flat"]

    def test_hindcast_too_short_returns_none(self):
        t = np.arange(1.0, 5.0, 1.0 / 365.25)
        y = 50 + 30 * np.cos(2 * np.pi * t / 3.6)
        h = pc.hindcast(t, y)
        assert h["n"] == 0
        assert h["rmse_wave"] is None and h["rmse_flat"] is None
        assert h["win_rate"] is None

    def test_fan_prices_monotone_rearranged_and_clipped(self):
        qs = np.array([0.1, 0.5, 0.9])
        # column 0 ordered; column 1 has Q50 above Q90 (crossed)
        logfan = np.array([[3.0, 3.0],
                           [4.0, 5.0],
                           [5.0, 4.0]])
        pct = np.array([70.0, 70.0])
        out = pc.fan_prices(logfan, qs, pct)
        # sorted column 1 == column 0, so both map to the same price
        assert out[0] == pytest.approx(10 ** 4.5)
        assert out[1] == pytest.approx(10 ** 4.5)
        lo = pc.fan_prices(logfan, qs, np.array([0.0, 0.0]))
        hi = pc.fan_prices(logfan, qs, np.array([100.0, 100.0]))
        np.testing.assert_allclose(lo, [1e3, 1e3])
        np.testing.assert_allclose(hi, [1e5, 1e5])
