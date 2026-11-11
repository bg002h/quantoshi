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


# ── Figure ────────────────────────────────────────────────────────────────────

import math                                                    # noqa: E402
import re                                                      # noqa: E402
import time                                                    # noqa: E402
from datetime import date                                      # noqa: E402

from btc_core import today_t, yr_to_t                          # noqa: E402
from conftest import M, _app_ctx                               # noqa: E402
from colors import (PCTILE_FIT_CONTEXT_OPACITY,                # noqa: E402
                    PCTILE_FORECAST_SHADE, PCTILE_FORECAST_SHADE_ALPHA,
                    _hex_alpha)
from figures import percentile as fp                           # noqa: E402

_PRICE_LABEL = re.compile(r"Q\d+ · \$[\d.,]+[kMB]?")


def _p(**kw):
    base = dict(xmin=2010, xmax=date.today().year + 1,
                active_models=["qr"], palette="default", xscale="lin",
                show_legend=True, show_today=True, minor_grid=False,
                chart_zoom=False, legend_pos="outside", user_model=None,
                sigma_mode="resqr", cycle_fit=True, cycle_censor=False,
                cycle_model="qr", cycle_horizon=20)
    base.update(kw)
    return base


def _names(fig):
    return [t.name for t in fig.data]


def _anns(fig):
    return list(fig.layout.annotations or [])


def _price_labels(fig):
    return [a for a in _anns(fig) if _PRICE_LABEL.search(a.text or "")]


_SHADE = _hex_alpha(PCTILE_FORECAST_SHADE, PCTILE_FORECAST_SHADE_ALPHA)


def _forecast_shades(fig):
    return [s for s in (fig.layout.shapes or []) if s.fillcolor == _SHADE]


def _forecast_turn_traces(fig):
    return [t for t in fig.data if (t.name or "").endswith("forecast turns")]


class TestFigure:
    def test_fit_off_matches_legacy_traces(self):
        legacy = {k: v for k, v in _p().items() if not k.startswith("cycle_")}
        a = fp.build_percentile_figure(M, legacy)
        b = fp.build_percentile_figure(M, _p(cycle_fit=False))
        assert _names(a) == _names(b)
        assert a.layout.xaxis.range == b.layout.xaxis.range

    def test_fit_on_adds_band_wave_actual_and_turns(self):
        fig = fp.build_percentile_figure(M, _p())
        names = _names(fig)
        assert "QR percentile (actual)" in names
        assert any("cycle fit" in (n or "") for n in names)
        assert "10–90% band (bootstrap)" in names
        fc = [t for t in fig.data if (t.name or "").startswith("QR cycle extrapolation")]
        assert len(fc) == 1 and fc[0].line.dash == "dash"
        assert _forecast_turn_traces(fig)

    def test_forecast_region_is_shaded_from_last_data_to_edge(self):
        fig = fp.build_percentile_figure(M, _p())
        (sh,) = _forecast_shades(fig)
        assert sh.x1 == pytest.approx(fig.layout.xaxis.range[1])
        assert sh.x0 == pytest.approx(float(M.price_years[-1]), abs=1e-6)

    @pytest.mark.parametrize("h", [10, 20, 30])
    def test_xrange_extends_to_horizon(self, h):
        fig = fp.build_percentile_figure(M, _p(cycle_horizon=h))
        assert fig.layout.xaxis.range[1] == pytest.approx(today_t(M.genesis) + h, abs=0.02)

    def test_zoomed_past_has_no_forecast_region(self):
        fig = fp.build_percentile_figure(M, _p(xmin=2014, xmax=2018))
        assert fig.layout.xaxis.range[1] <= yr_to_t(2018, M.genesis) + 1e-6
        assert not _forecast_shades(fig)
        assert not _price_labels(fig)
        assert not _forecast_turn_traces(fig)

    def test_forecast_turn_labels_have_prices(self):
        fig = fp.build_percentile_figure(M, _p())
        assert len(_price_labels(fig)) >= 4
        (tr,) = _forecast_turn_traces(fig)
        assert len(tr.x) == len(_price_labels(fig))
        assert all(v >= 0 for v in tr.error_x.array)
        assert all(v >= 0 for v in tr.error_x.arrayminus)

    def test_summary_line_reports_period_r2_and_hindcast(self):
        fig = fp.build_percentile_figure(M, _p())
        txt = [a.text for a in _anns(fig) if "period" in (a.text or "")]
        assert len(txt) == 1
        assert "R²" in txt[0] and "hindcast" in txt[0]

    def test_weak_fit_warns_and_drops_labels(self):
        fig = fp.build_percentile_figure(M, _p(cycle_model="bub"))
        assert any("weak cycle" in (a.text or "") for a in _anns(fig))
        assert not _price_labels(fig)

    def test_log_axis_annotations_use_log10(self):
        fig = fp.build_percentile_figure(M, _p(xscale="log"))
        (tr,) = _forecast_turn_traces(fig)
        labels = _price_labels(fig)
        assert labels
        xs = sorted(float(a.x) for a in labels)
        np.testing.assert_allclose(xs, sorted(math.log10(v) for v in tr.x), atol=1e-6)
        assert fig.layout.xaxis.type == "log"

    def test_wave_hover_carries_band_and_prices(self):
        fig = fp.build_percentile_figure(M, _p())
        w = [t for t in fig.data if (t.name or "").startswith("QR cycle fit")][0]
        assert len(w.customdata[0]) == 6          # date + 5 numbers
        assert "customdata[5]" in w.hovertemplate

    def test_fit_model_not_active_still_drawn(self):
        fig = fp.build_percentile_figure(M, _p(active_models=["bub"]))
        assert "QR percentile (actual)" in _names(fig)
        bm = [t for t in fig.data if t.name == "BM"]
        assert bm and bm[0].opacity == PCTILE_FIT_CONTEXT_OPACITY

    def test_censor_changes_fit_and_is_cached(self):
        fpk = str(_app_ctx._MODEL_FP)
        a = fp._cycle_result("qr", "resqr", False, fpk)
        b = fp._cycle_result("qr", "resqr", True, fpk)
        assert a["fit"].r2 != b["fit"].r2
        assert b["t"].size < a["t"].size
        assert fp._cycle_result("qr", "resqr", True, fpk) is b

    def test_unknown_model_falls_back_to_no_fit(self):
        fig = fp.build_percentile_figure(M, _p(cycle_model="nope"))
        assert not any("cycle fit" in (n or "") for n in _names(fig))

    def test_cycle_result_is_fast_enough(self):
        fp._cycle_result.cache_clear()
        t0 = time.perf_counter()
        fp._cycle_result("pl", "resqr", False, str(_app_ctx._MODEL_FP))
        assert time.perf_counter() - t0 < 3.0
