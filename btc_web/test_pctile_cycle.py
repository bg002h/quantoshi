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
        anchor = {k: max(x for x, _, kk in pc.turns(f, t[0], t[-1]) if kk == k)
                  for k in ("peak", "trough")}
        seen = {"peak": 0, "trough": 0}
        for x, _, k in pc.turns(f, t[-1], t[-1] + 12):
            seen[k] += 1
            lo, hi = pc.turn_timing(P, anchor[k], seen[k], k, x)
            assert lo <= x <= hi
            assert hi - lo < f.period          # one turn, not a neighbour

    def test_turn_just_after_the_last_close_is_not_matched_to_the_next_cycle(self):
        """Replicates split either side of the last close must still match
        the SAME turn (regression: a +1-period bar chained into a line)."""
        f = pc.CycleFit(c=50.0, A=30.0, f=1 / 3.6, phi=0.0, r2=0.9, n=100)
        P = np.array([[50.0, 30.0, 1 / 3.6, d] for d in np.linspace(-0.2, 0.2, 41)])
        anchor = 3.6 * 3                       # an in-sample peak (phase 0)
        t_last = 3.6 * 4 - 0.01                # just before the next peak
        lo, hi = pc.turn_timing(P, anchor, 1, "peak", 3.6 * 4)
        assert hi - lo < 0.5 and lo < t_last < hi

    def test_turn_timing_matches_by_cycle_index(self):
        """Every replicate equal to the central fit -> zero-width bar at the
        j-th turn, for j beyond the first."""
        f = pc.CycleFit(c=50.0, A=30.0, f=1 / 3.6, phi=1.0, r2=0.9, n=100)
        P = np.tile([f.c, f.A, f.f, f.phi], (20, 1))
        peaks = [x for x, _, k in pc.turns(f, 10.0, 40.0) if k == "peak"]
        for j, x in enumerate(peaks[1:], start=1):
            lo, hi = pc.turn_timing(P, peaks[0], j, "peak", x)
            assert lo == pytest.approx(x) and hi == pytest.approx(x)

    def test_at_limit_flags_a_boundary_period(self):
        t = np.arange(1.0, 17.0, 1.0 / 365.25)
        f = pc.fit_cycle(t, 50 + 30 * np.cos(2 * np.pi * t / 12.0))
        assert f.at_limit and f.period == pytest.approx(pc.PMAX, abs=0.05)
        assert not pc.fit_cycle(*_synthetic()).at_limit

    def test_likely_band_covers_about_80pct_in_sample(self):
        t, y = _synthetic(noise=15.0, ar=0.995)
        f = pc.fit_cycle(t, y)
        P = pc.bootstrap_params(t, y, f)
        lo, hi = pc.likely_band(P, t, y, f, t[::7])
        inside = ((y[::7] >= lo) & (y[::7] <= hi)).mean()
        assert 0.65 <= inside <= 0.95
        flo, fhi = pc.band(P, t[::7])
        assert np.mean(hi - lo) > np.mean(fhi - flo)       # wider than the fit band
        lo2, hi2 = pc.likely_band(P, t, y, f, t[::7])
        np.testing.assert_array_equal(lo, lo2)             # deterministic

    def test_hindcast_y_at_replaces_the_series_per_cutoff(self):
        t = np.arange(1.0, 17.0, 1.0 / 365.25)
        y = 50 + 30 * np.cos(2 * np.pi * t / 3.6 + 0.3)
        calls = []

        def y_at(cut):
            calls.append(cut)
            return np.full_like(t, 50.0) if cut < 10 else y
        h_leaky = pc.hindcast(t, y)
        h = pc.hindcast(t, y, y_at=y_at)
        assert calls and h["n"] == h_leaky["n"]
        assert h["win_rate"] < h_leaky["win_rate"]
        assert pc.hindcast(t, y, y_at=lambda c: None)["n"] == 0

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


class TestBases:
    def test_wide_calendar_reaches_a_one_year_wave(self):
        t = np.arange(1.0, 15.0, 1.0 / 365.25)
        y = 50 + 30 * np.cos(2 * np.pi * t / 1.0)
        assert pc.fit_cycle(t, y, basis="calw").period == pytest.approx(1.0, abs=0.01)
        assert pc.fit_cycle(t, y).period >= pc.PMIN

    def test_log_basis_recovers_log_periodic(self):
        rng = np.random.default_rng(3)
        t = np.arange(1.0, 17.0, 1.0 / 365.25)
        y = 50 + 30 * np.cos(7.2 * np.log(t) + 0.5) + rng.normal(0, 3, t.size)
        f = pc.fit_cycle(t, y, basis="log")
        assert f.space == "log" and f.basis == "log"
        assert 2 * np.pi * f.f == pytest.approx(7.2, abs=0.1)
        assert f.period == pytest.approx(np.exp(2 * np.pi / 7.2), rel=0.02)
        assert f.r2 > 0.9

    def test_log_turns_are_extrema_and_alternate(self):
        f = pc.CycleFit(c=50.0, A=30.0, f=7.2 / (2 * np.pi), phi=0.5, r2=0.9,
                        n=100, basis="log")
        tt = pc.turns(f, 1.0, 60.0)
        assert len(tt) >= 4
        kinds = [k for _, _, k in tt]
        assert all(a != b for a, b in zip(kinds, kinds[1:]))
        for x, v, k in tt:
            near = pc.wave(f.c, f.A, f.f, f.phi, np.array([x * 0.999, x * 1.001]), "log")
            assert (np.all(near <= v + 1e-9) if k == "peak" else np.all(near >= v - 1e-9))
        peaks = [x for x, _, k in tt if k == "peak"]
        np.testing.assert_allclose(np.diff(np.log(peaks)), 1.0 / f.f, rtol=1e-9)

    def test_log_bootstrap_and_timing(self):
        rng = np.random.default_rng(4)
        t = np.arange(1.0, 17.0, 1.0 / 365.25)
        y = 50 + 30 * np.cos(7.2 * np.log(t) + 0.5) + rng.normal(0, 8, t.size)
        f = pc.fit_cycle(t, y, basis="log")
        P = pc.bootstrap_params(t, y, f, B=40)
        lo, hi = pc.band(P, np.array([t[-1], t[-1] + 10]), space="log")
        assert np.all(lo <= hi)
        for x, _, k in pc.turns(f, t[-1], t[-1] + 30):
            a, b = pc.turn_timing(P, x, 0, k, x, space="log")
            assert a <= x <= b

    def test_hindcast_honours_basis(self):
        t = np.arange(1.0, 17.0, 1.0 / 365.25)
        y = 50 + 30 * np.cos(7.2 * np.log(t) + 0.5)
        assert pc.hindcast(t, y, basis="log")["win_rate"] == 1.0

    def test_is_weak_is_in_sample_only(self):
        """The hindcast is reported, never a gate (review r2 I-1)."""
        assert not pc.is_weak(pc.CycleFit(50, 30, 0.28, 0, 0.5, 100))
        assert pc.is_weak(pc.CycleFit(50, 30, 0.28, 0, 0.1, 100))
        assert pc.is_weak(pc.CycleFit(50, 30, 0.125, 0, 0.5, 100, at_limit=True))

    def test_beats_flat(self):
        assert pc.beats_flat(dict(n=10, rmse_wave=20.0, rmse_flat=27.0)) is True
        assert pc.beats_flat(dict(n=10, rmse_wave=28.0, rmse_flat=25.0)) is False
        assert pc.beats_flat(dict(n=0, rmse_wave=None, rmse_flat=None)) is None

    def test_likely_band_is_smooth_week_to_week(self):
        """r2 I-4: one synchronised residual path per refit stepped 12-15 pp."""
        t, y = _synthetic(noise=15.0, ar=0.995)
        f = pc.fit_cycle(t, y)
        P = pc.bootstrap_params(t, y, f)
        grid = np.arange(t[-1], t[-1] + 20, 7 / 365.25)
        lo, hi = pc.likely_band(P, t, y, f, grid)
        assert np.max(np.abs(np.diff(hi))) < 6.0
        assert np.max(np.abs(np.diff(lo))) < 6.0

    def test_censor_mask_fits_without_but_scores_with(self):
        """r2 I-3: days withheld from the fit are still scored."""
        t = np.arange(1.0, 17.0, 7 / 365.25)
        y = 50 + 30 * np.cos(2 * np.pi * t / 3.6)
        bad = (t > 9.0) & (t < 10.0)
        y_bad = y.copy()
        y_bad[bad] = 100.0                         # a window the wave misfits
        mask = ~bad
        scored = pc.hindcast(t, y_bad, step=1, fit_mask=mask)
        dropped = pc.hindcast(t[mask], y_bad[mask], step=1)
        assert scored["rmse_wave"] > dropped["rmse_wave"]   # misfit still scored

    def test_every_basis_key_has_a_label_and_valid_space(self):
        for k, (space, fmin, fmax, label) in pc.BASES.items():
            assert space in ("cal", "log") and 0 < fmin < fmax and label
        assert pc.DEFAULT_BASIS in pc.BASES


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
        assert not (b.layout.meta or {})

    def test_fit_on_adds_bands_wave_actual_and_turns(self):
        fig = fp.build_percentile_figure(M, _p())
        names = _names(fig)
        assert "QR percentile (actual)" in names
        assert any("cycle fit" in (n or "") for n in names)
        assert "fit uncertainty (10–90% of refits)" in names
        assert "likely range (10–90%)" in names
        fc = [t for t in fig.data if (t.name or "").startswith("QR cycle extrapolation")]
        assert len(fc) == 1 and fc[0].line.dash == "dash"
        assert _forecast_turn_traces(fig)

    @pytest.mark.parametrize("trim_days", [0, 45, 90, 135, 165, 180])
    def test_default_qr_stays_a_forecast_as_data_moves(self, trim_days):
        """r2 I-1: the default view must not lose its forecast after a
        routine price update. Weakness is in-sample only."""
        mdl = _app_ctx.PRICE_MODELS["qr"]
        t = np.asarray(M.price_years, float)
        px = np.asarray(M.price_prices, float)
        keep = t >= 1.0
        t, px = t[keep], px[keep]
        if trim_days:
            t, px = t[:-trim_days], px[:-trim_days]
        y = np.asarray(fp._percentile_series(mdl, t, px, sigma_mode="resqr"))
        assert not pc.is_weak(pc.fit_cycle(t, y))

    @pytest.mark.parametrize("h", [10, 20, 30])
    def test_xrange_extends_to_horizon(self, h):
        fig = fp.build_percentile_figure(M, _p(cycle_horizon=h))
        assert fig.layout.xaxis.range[1] == pytest.approx(today_t(M.genesis) + h, abs=0.02)

    def test_bad_horizon_falls_back_to_20(self):
        fig = fp.build_percentile_figure(M, _p(cycle_horizon=7))
        assert fig.layout.xaxis.range[1] == pytest.approx(today_t(M.genesis) + 20, abs=0.02)

    def test_forecast_region_is_shaded_from_last_data_to_edge(self):
        fig = fp.build_percentile_figure(M, _p())
        (sh,) = _forecast_shades(fig)
        assert sh.x1 == pytest.approx(fig.layout.xaxis.range[1])
        assert sh.x0 == pytest.approx(float(M.price_years[-1]), abs=1e-6)

    def test_handle_at_this_year_is_not_a_forecast(self):
        """xmax = this year means 1 Jan, before today: no forecast."""
        fig = fp.build_percentile_figure(M, _p(xmax=date.today().year))
        assert not _forecast_shades(fig) and not _forecast_turn_traces(fig)

    def test_zoomed_past_has_no_forecast_region(self):
        fig = fp.build_percentile_figure(M, _p(xmin=2014, xmax=2018))
        assert fig.layout.xaxis.range[1] <= yr_to_t(2018, M.genesis) + 1e-6
        assert not _forecast_shades(fig)
        assert not _price_labels(fig)
        assert not _forecast_turn_traces(fig)

    def test_timing_bars_stay_within_a_period(self):
        """Regression: the first trough sits days after the last close; its
        bar must not jump a whole cycle (bars chained into a flat line)."""
        fig = fp.build_percentile_figure(M, _p())
        (tr,) = _forecast_turn_traces(fig)
        per = fp._cycle_result("qr", "resqr", False, "cal",
                               str(_app_ctx._MODEL_FP))["fit"].period
        widths = np.add(tr.error_x.array, tr.error_x.arrayminus)
        assert widths[0] < per / 2
        assert np.all(widths < per)

    @pytest.mark.parametrize("h,n", [(10, 6), (20, 4), (30, 2)])
    def test_linear_axis_labels_the_next_turns_by_horizon(self, h, n):
        fig = fp.build_percentile_figure(M, _p(cycle_horizon=h))
        (tr,) = _forecast_turn_traces(fig)
        labels = _price_labels(fig)
        assert len(labels) == n <= len(tr.x)
        assert all(v >= 0 for v in tr.error_x.array)
        assert all(v >= 0 for v in tr.error_x.arrayminus)
        # the labelled turns are exactly the EARLIEST n (kinds alternate)
        np.testing.assert_allclose(sorted(float(a.x) for a in labels),
                                   sorted(tr.x)[:n])

    def test_log_axis_labels_next_peak_and_trough_at_log10(self):
        fig = fp.build_percentile_figure(M, _p(xscale="log"))
        (tr,) = _forecast_turn_traces(fig)
        labels = _price_labels(fig)
        assert len(labels) == 2
        first_two = sorted(tr.x)[:2]
        np.testing.assert_allclose(sorted(float(a.x) for a in labels),
                                   [math.log10(v) for v in first_two], atol=1e-6)
        assert fig.layout.xaxis.type == "log"

    def test_info_lines_report_element_period_r2_hindcast_and_coverage(self):
        fig = fp.build_percentile_figure(M, _p())
        info = fig.layout.meta["cycle_info"]
        head = info[0]
        assert "QR" in head and "calendar 2–8 yr" in head and "period" in head
        assert "R²" in head
        assert any("Out-of-sample (as-of fans, no look-ahead)" in ln for ln in info)
        assert any("sat inside them" in ln for ln in info)
        assert info[-1] == "Descriptive fit, not a forecast."

    def test_non_qr_is_not_checked_out_of_sample_and_shows_no_win_share(self):
        """r2 I-2: an optimistic hindcast is not a win rate beside QR's."""
        fig = fp.build_percentile_figure(M, _p(cycle_model="pl"))
        info = fig.layout.meta["cycle_info"]
        hl = [ln for ln in info if "out-of-sample" in ln]
        assert hl and hl[0].startswith("Not checked out-of-sample")
        assert not re.search(r"in \d+ of \d+", " ".join(info))
        assert any(n == "PL cycle extrapolation (not checked out-of-sample)"
                   for n in _names(fig))

    def test_weak_fit_draws_no_forecast(self):
        fig = fp.build_percentile_figure(M, _p(cycle_model="bub"))
        assert any("weak cycle" in (a.text or "") for a in _anns(fig))
        assert fig.layout.meta["cycle_info"][0].startswith("⚠ Weak cycle in BM")
        assert not _price_labels(fig)
        assert not _forecast_turn_traces(fig)
        assert not any((n or "").endswith("cycle extrapolation") for n in _names(fig))
        assert not _forecast_shades(fig)
        assert fig.layout.xaxis.range[1] <= today_t(M.genesis) + 1e-6
        assert "likely range (10–90%)" not in _names(fig)

    def test_hindcast_failure_is_reported_not_suppressing(self, monkeypatch):
        fp._cycle_result.cache_clear()
        monkeypatch.setattr(fp._pc, "beats_flat", lambda hind: False)
        try:
            fig = fp.build_percentile_figure(M, _p())
            assert any("no out-of-sample edge" in (a.text or "") for a in _anns(fig))
            assert any("No out-of-sample edge" in ln
                       for ln in fig.layout.meta["cycle_info"])
            assert _forecast_turn_traces(fig)            # forecast still drawn
        finally:
            fp._cycle_result.cache_clear()

    def test_wave_hover_carries_bands_and_prices(self):
        fig = fp.build_percentile_figure(M, _p())
        w = [t for t in fig.data if (t.name or "").startswith("QR cycle fit")][0]
        assert len(w.customdata[0]) == 8          # date + 7 numbers
        assert "customdata[7]" in w.hovertemplate

    def test_fit_model_not_active_still_drawn(self):
        fig = fp.build_percentile_figure(M, _p(active_models=["bub"]))
        assert "QR percentile (actual)" in _names(fig)
        bm = [t for t in fig.data if t.name == "BM"]
        assert bm and bm[0].opacity == PCTILE_FIT_CONTEXT_OPACITY

    def test_picked_master_is_the_same_series_as_its_line(self):
        """I-3: picking EPPL fits the eppl master the model list draws."""
        fig = fp.build_percentile_figure(M, _p(active_models=["eppl"], cycle_model="eppl"))
        names = _names(fig)
        assert "EPPL percentile (actual)" in names
        assert "EPPL" not in names                 # not drawn twice under one name

    @pytest.mark.parametrize("key", list(pc.CYCLE_MODEL_KEYS))
    @pytest.mark.parametrize("basis", list(pc.BASES))
    def test_every_picker_model_and_element_builds(self, key, basis):
        if key not in _app_ctx.PRICE_MODELS:
            pytest.skip(f"{key} not registered here")
        fig = fp.build_percentile_figure(M, _p(cycle_model=key, cycle_basis=basis))
        assert fig.layout.meta["cycle_info"]

    def test_censor_changes_fit_and_is_cached(self):
        fpk = str(_app_ctx._MODEL_FP)
        a = fp._cycle_result("qr", "resqr", False, "cal", fpk)
        b = fp._cycle_result("qr", "resqr", True, "cal", fpk)
        assert a["fit"].r2 != b["fit"].r2
        assert b["fit"].n < a["fit"].n                 # withheld from the fit
        assert b["t"].size == a["t"].size              # ...but kept for scoring
        assert b["honest"] and b["hind"]["n"] == a["hind"]["n"]
        assert fp._cycle_result("qr", "resqr", True, "cal", fpk) is b
        info = fp.build_percentile_figure(M, _p(cycle_censor=True)).layout.meta["cycle_info"]
        assert "R² (2019–20 excluded from the fit)" in info[0]

    def test_qr_hindcast_is_honest_and_coverage_is_sane(self):
        r = fp._cycle_result("qr", "resqr", False, "cal", str(_app_ctx._MODEL_FP))
        assert r["honest"] and r["hind"]["n"] >= 10
        assert 0.6 <= r["cov_lik"] <= 0.97
        assert r["cov_fit"] < r["cov_lik"]

    def test_unknown_model_or_basis_falls_back_to_no_fit(self):
        for kw in (dict(cycle_model="nope"), dict(cycle_basis="nope"),
                   dict(cycle_model="u1")):
            fig = fp.build_percentile_figure(M, _p(**kw))
            assert not any("cycle fit" in (n or "") for n in _names(fig))

    def test_log_basis_figure(self):
        fig = fp.build_percentile_figure(M, _p(cycle_basis="log"))
        assert "log-time element" in fig.layout.title.text
        assert "log-time element" in fig.layout.meta["cycle_info"][-4] or any(
            "log-time element" in ln for ln in fig.layout.meta["cycle_info"])

    def test_cycle_result_is_fast_enough(self):
        fp._cycle_result.cache_clear()
        t0 = time.perf_counter()
        fp._cycle_result("qr", "resqr", False, "cal", str(_app_ctx._MODEL_FP))
        assert time.perf_counter() - t0 < 4.0


# ── Wiring + share links ─────────────────────────────────────────────────────

import base64                                                  # noqa: E402
import gzip                                                    # noqa: E402
import json                                                    # noqa: E402
from unittest.mock import patch                                # noqa: E402

from bub_views import STYLE_OUTPUT_IDS, VIEW_MODES, mode_styles  # noqa: E402
from snapshot import (_SNAPSHOT_CONTROLS, _CHECKLIST_OPTIONS,  # noqa: E402
                      _decode_snapshot, _decode_snapshot_v4, _encode_snapshot,
                      _encode_snapshot_v4)
from snapshot_defaults import (SNAPSHOT_DEFAULTS,              # noqa: E402
                               _compute_snapshot_defaults_fingerprint)

_FIT = "bub-pctile-fit:value"
_NEW = ["bub-pctile-fit", "bub-pctile-fit-model", "bub-pctile-fit-basis",
        "bub-pctile-fit-horizon"]
_PRE_FIT_FP = "0eac47fc"      # last defaults fingerprint before the fields


def _idx(cid):
    return [c for c, _ in _SNAPSHOT_CONTROLS].index(cid)


def _v4(fp, diffs):
    j = json.dumps([fp, diffs, None], separators=(",", ":"))
    return f"{fp}:" + base64.urlsafe_b64encode(gzip.compress(j.encode())).decode()


def _walk_ids(node, out):
    pid = getattr(node, "id", None)
    if pid is not None:
        out[pid] = node
    ch = getattr(node, "children", None)
    for c in (ch if isinstance(ch, (list, tuple)) else [ch]):
        if c is not None and hasattr(c, "to_plotly_json"):
            _walk_ids(c, out)
    return out


class TestWiring:
    def test_percentile_mode_shows_its_ctl_span_only(self):
        i = STYLE_OUTPUT_IDS.index("bub-pctile-ctl-wrap")
        for mode in VIEW_MODES:
            want = {"display": "inline"} if mode == "percentile" else {"display": "none"}
            assert mode_styles(mode)[i] == want, mode

    def test_controls_exist_with_snapshot_defaults(self):
        from layout.common import _pctile_cycle_controls
        ids = _walk_ids(_pctile_cycle_controls(), {})
        assert ids["bub-pctile-fit"].value == ["fit"]
        assert [o["value"] for o in ids["bub-pctile-fit"].options] == ["fit", "censor"]
        assert ids["bub-pctile-fit-model"].value == "qr"
        assert ids["bub-pctile-fit-basis"].value == "cal"
        assert ids["bub-pctile-fit-horizon"].value == 20
        models = [o["value"] for o in ids["bub-pctile-fit-model"].options]
        assert models[0] == "qr" and set(models) <= set(pc.CYCLE_MODEL_KEYS)
        assert [o["value"] for o in ids["bub-pctile-fit-basis"].options] == list(pc.BASES)
        assert [o["value"] for o in ids["bub-pctile-fit-horizon"].options] == list(pc.HORIZONS)

    def test_update_callback_passes_coerced_cycle_params(self):
        from callbacks.charts import update_bub_pctile
        seen = {}

        def grab(p):
            seen.update(p)
            return fp.build_percentile_figure(M, p)
        args = ("percentile", [2010, 2027], [], "lin", ["bub"], "outside",
                "default", "resqr")
        with patch("utils._get_pctile_fig", side_effect=grab):
            update_bub_pctile(*args, ["fit", "censor"], "pl", "log", 30, None, False)
            assert (seen["cycle_fit"], seen["cycle_censor"], seen["cycle_model"],
                    seen["cycle_basis"], seen["cycle_horizon"]) == (True, True, "pl", "log", 30)
            update_bub_pctile(*args, None, "u1", "zzz", 7, None, False)
            assert (seen["cycle_fit"], seen["cycle_censor"], seen["cycle_model"],
                    seen["cycle_basis"], seen["cycle_horizon"]) == (False, False, "qr", "cal", 20)

    def test_info_div_is_filled_clientside_from_layout_meta(self):
        scripts = [s for s in _app_ctx.app._inline_scripts if "cycle_info" in s]
        assert len(scripts) == 1
        cm = _app_ctx.app.callback_map
        keys = [k for k in cm if "bub-pctile-info.children" in k]
        assert len(keys) == 1
        assert [i["id"] for i in cm[keys[0]]["inputs"]] == ["bub-pctile-graph"]


class TestSnapshot:
    def test_new_fields_appended_at_the_absolute_tail(self):
        assert [c for c, _ in _SNAPSHOT_CONTROLS[348:352]] == _NEW
        assert all(p == "value" for _, p in _SNAPSHOT_CONTROLS[348:352])
        assert len(_SNAPSHOT_CONTROLS) == 352
        assert _SNAPSHOT_CONTROLS[347] == ("bub-mc-extremes", "value")

    def test_defaults_options_and_tab_controls(self):
        from callbacks.routing import _TAB_CONTROLS
        assert SNAPSHOT_DEFAULTS[_FIT] == ["fit"]
        assert SNAPSHOT_DEFAULTS["bub-pctile-fit-model:value"] == "qr"
        assert SNAPSHOT_DEFAULTS["bub-pctile-fit-basis:value"] == "cal"
        assert SNAPSHOT_DEFAULTS["bub-pctile-fit-horizon:value"] == 20
        assert _CHECKLIST_OPTIONS["bub-pctile-fit"] == ["fit", "censor"]
        assert set(_NEW) <= _TAB_CONTROLS["bubble"]

    def test_round_trip(self):
        st = {"bub-view-mode:data": "percentile", _FIT: ["fit", "censor"],
              "bub-pctile-fit-model:value": "pl",
              "bub-pctile-fit-basis:value": "log",
              "bub-pctile-fit-horizon:value": 30}
        dec = _decode_snapshot_v4(_encode_snapshot_v4(st))
        for k, v in st.items():
            assert dec[k] == v, k

    def test_fit_off_survives_a_new_link(self):
        dec = _decode_snapshot_v4(_encode_snapshot_v4(
            {"bub-view-mode:data": "percentile", _FIT: []}))
        assert dec[_FIT] == []

    def test_current_fingerprint_is_registered_and_old_one_kept(self):
        from snapshot import _registry_lookup
        fp_now = _compute_snapshot_defaults_fingerprint()
        assert _registry_lookup(fp_now) is not None
        old = _registry_lookup(_PRE_FIT_FP)
        assert old is not None and _FIT not in old

    def test_old_v4_percentile_link_restores_with_the_fit_off(self):
        state = _decode_snapshot_v4(
            _v4(_PRE_FIT_FP, {str(_idx("bub-view-mode")): "percentile"}))
        assert state["bub-view-mode:data"] == "percentile"
        assert state[_FIT] == []

    def test_old_positional_link_restores_with_the_fit_off(self):
        values = [None] * _idx("bub-pctile-fit")
        values[_idx("bub-view-mode")] = "percentile"
        j = json.dumps([values, None], separators=(",", ":"))
        state = _decode_snapshot(
            base64.urlsafe_b64encode(gzip.compress(j.encode())).decode())
        assert state[_FIT] == []

    def test_new_v4_link_of_another_tab_keeps_the_default_on(self):
        """The rule keys on the field being ABSENT; a new link always carries
        it (registered default), even one scoped to another tab."""
        fp_now = _compute_snapshot_defaults_fingerprint()
        state = _decode_snapshot_v4(_v4(fp_now, {}))
        assert state[_FIT] == ["fit"]
