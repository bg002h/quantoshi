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

_PRICE_LABEL = re.compile(r"(Peak|Trough) \w{3} \d{4} · Q\d+ · \$[\d.,]+[kMB]?")


def _p(**kw):
    base = dict(xmin=2010, xmax=date.today().year + 21,
                active_models=["qr"], palette="default", xscale="lin",
                show_legend=True, show_today=True, minor_grid=False,
                chart_zoom=False, legend_pos="outside", user_model=None,
                sigma_mode="resqr", cycle_fit=True, cycle_censor=False,
                cycle_model="qr", cycle_labels=True)
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

    @pytest.mark.parametrize("xmax", [2033, 2050, 2080])
    def test_slider_end_drives_the_forecast(self, xmax):
        """Note D1: the year slider's right end is the forecast's end."""
        fig = fp.build_percentile_figure(M, _p(xmax=xmax))
        edge = yr_to_t(xmax, M.genesis)
        assert fig.layout.xaxis.range[1] == pytest.approx(edge, abs=1e-9)
        (fc,) = [t for t in fig.data
                 if (t.name or "").startswith("QR cycle extrapolation")]
        assert fc.x[-1] == pytest.approx(edge, abs=1e-9)
        assert max(fig.layout.xaxis.tickvals) <= edge + 1e-9
        if xmax % 5 == 0:                         # ticks every 5 yr: '50, '80
            assert f"'{xmax % 100:02d}" in list(fig.layout.xaxis.ticktext)

    def test_slider_end_on_log_axis(self):
        fig = fp.build_percentile_figure(M, _p(xmax=2080, xscale="log"))
        assert fig.layout.xaxis.range[1] == pytest.approx(
            math.log10(yr_to_t(2080, M.genesis)), abs=1e-9)

    def test_ignores_a_stray_horizon_param(self):
        a = fp.build_percentile_figure(M, _p(xmax=2040))
        b = fp.build_percentile_figure(M, _p(xmax=2040, cycle_horizon=30))
        assert a.layout.xaxis.range == b.layout.xaxis.range

    def test_band_grid_reaches_the_slider_domain(self):
        """Note D2: bands must not clamp flat before the slider's max."""
        from layout.bubble import XRANGE_MAX
        assert fp._GRID_END_YEAR >= XRANGE_MAX + 1
        r = fp._cycle_result("qr", "resqr", False, "cal", str(_app_ctx._MODEL_FP))
        assert r["grid"][-1] >= yr_to_t(XRANGE_MAX, M.genesis)
        fig = fp.build_percentile_figure(M, _p(xmax=XRANGE_MAX))
        up = [t for t in fig.data if t.name == "likely range (10–90%) (upper)"][0]
        tail = np.asarray(up.y[-60:], float)
        assert np.ptp(tail) > 0.5                  # still varying at 2080

    def test_fit_off_and_weak_stay_capped_at_today(self, monkeypatch):
        fig = fp.build_percentile_figure(M, _p(xmax=2080, cycle_fit=False))
        assert fig.layout.xaxis.range[1] <= today_t(M.genesis) + 1e-6
        fp._cycle_result.cache_clear()
        monkeypatch.setattr(fp._pc, "is_weak", lambda fit: True)   # review N-2
        try:
            fig = fp.build_percentile_figure(M, _p(xmax=2080))
            assert fig.layout.xaxis.range[1] <= today_t(M.genesis) + 1e-6
        finally:
            fp._cycle_result.cache_clear()

    def test_turn_timing_does_not_depend_on_the_left_handle(self):
        """Whole-diff review I-1: with the left handle past today the cycle
        index must still count from the last close."""
        def rows(xmin):
            fig = fp.build_percentile_figure(M, _p(xmin=xmin, xmax=2060))
            (tr,) = _forecast_turn_traces(fig)
            return {round(float(x), 6): list(cd) for x, cd in zip(tr.x, tr.customdata)}
        full, late = rows(2010), rows(2030)
        assert late and set(late) <= set(full)
        for x, cd in late.items():
            assert cd == full[x], x

    def test_left_handle_past_data_drops_the_in_sample_trace(self):
        """Whole-diff review M-3."""
        fig = fp.build_percentile_figure(M, _p(xmin=2030, xmax=2060))
        assert not any("cycle fit (" in (n or "") for n in _names(fig))
        assert any("cycle extrapolation" in (n or "") for n in _names(fig))
        assert len(_price_labels(fig)) == 2        # first visible of each kind

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

    @pytest.mark.parametrize("xmax", [2040, 2080])
    def test_linear_axis_labels_only_the_next_peak_and_trough(self, xmax):
        """Note D5: one label per kind whatever the span."""
        fig = fp.build_percentile_figure(M, _p(xmax=xmax))
        (tr,) = _forecast_turn_traces(fig)
        labels = _price_labels(fig)
        assert len(labels) == 2 < len(tr.x)
        assert {a.text.split(" ")[0] for a in labels} == {"Peak", "Trough"}
        assert all(v >= 0 for v in tr.error_x.array)
        assert all(v >= 0 for v in tr.error_x.arrayminus)
        np.testing.assert_allclose(sorted(float(a.x) for a in labels),
                                   sorted(tr.x)[:2])

    def test_labels_switch_off_keeps_markers_and_bars(self):
        on = fp.build_percentile_figure(M, _p())
        off = fp.build_percentile_figure(M, _p(cycle_labels=False))
        assert _price_labels(on) and not _price_labels(off)
        (tr,) = _forecast_turn_traces(off)
        assert tr.error_x.array is not None and len(tr.x) >= 2
        assert _names(on) == _names(off)
        fig = fp.build_percentile_figure(M, _p(cycle_model="pl", cycle_censor=True,
                                               cycle_labels=False))
        assert any(a.text == "withheld<br>from fit" for a in _anns(fig))

    def test_label_text_has_range_and_end_prices(self):
        """Note D5: kind, date, Q, price / date range / price at each end."""
        fig = fp.build_percentile_figure(M, _p())
        (tr,) = _forecast_turn_traces(fig)
        mdl = _app_ctx.PRICE_MODELS["qr"]
        for lab in _price_labels(fig):
            i = int(np.argmin(np.abs(np.asarray(tr.x) - float(lab.x))))
            x, v = float(tr.x[i]), float(tr.y[i])
            a = x - float(tr.error_x.arrayminus[i])
            b = x + float(tr.error_x.array[i])
            want = []
            for t_ in (a, b):
                qs, lf = fp._logfan(mdl, np.array([t_]), "resqr")
                want.append(fp._money(float(pc.fan_prices(lf, qs, np.array([v]))[0])))
            line1, line2, line3, line4 = lab.text.split("<br>")
            assert line2 == (f"range {fp._mon_yr(a, M.genesis)} – "
                             f"{fp._mon_yr(b, M.genesis)}")
            assert line3 == f"{want[0]} – {want[1]}"
            cd = tr.customdata[i]
            assert list(cd[5:7]) == want           # hover carries them too
            r = fp._cycle_result("qr", "resqr", False, "cal", str(_app_ctx._MODEL_FP))
            qs, lf = fp._logfan(mdl, np.array([x]), "resqr")
            bands = [fp._money(float(pc.fan_prices(lf, qs, np.array(
                [np.interp(x, r["grid"], r[k])]))[0]))
                for k in ("fit_lo", "fit_hi", "lik_lo", "lik_hi")]
            assert line4 == (f"wave {bands[0]} – {bands[1]} · "
                             f"likely {bands[2]} – {bands[3]}")
            assert list(cd[7:11]) == bands
            assert lab.yanchor == ("bottom" if line1.startswith("Peak") else "top")

    def test_capped_timing_end_reads_or_later_and_dash(self, monkeypatch):
        real = fp._pc.turn_timing

        def wide(params, t_a, j, kind, t_e, **kw):
            a, _ = real(params, t_a, j, kind, t_e, **kw)
            return a, t_e + 500.0
        monkeypatch.setattr(fp._pc, "turn_timing", wide)
        fig = fp.build_percentile_figure(M, _p())
        for lab in _price_labels(fig):
            l2, l3 = lab.text.split("<br>")[1:3]
            assert l2.endswith(" or later") and l3.endswith(" – —"), lab.text

    def test_label_at_right_edge_anchors_right(self):
        """Review M-4: a turn in the last 15% of the axis hangs inward. The
        slider end is set from the fit (the year after the first forecast
        turn), so the case exists whatever today's date (whole-diff M-1)."""
        r = fp._cycle_result("qr", "resqr", False, "cal", str(_app_ctx._MODEL_FP))
        t_last = float(r["t"][-1])
        first = min(x for x, _, _ in pc.turns(r["fit"], t_last, t_last + 10)
                    if x > t_last)
        xmax = int(fp._mon_yr(first, M.genesis).split()[1]) + 1
        fig = fp.build_percentile_figure(M, _p(xmax=xmax))
        lo, hi = fig.layout.xaxis.range
        (lab,) = [a for a in _price_labels(fig) if abs(float(a.x) - first) < 1e-9]
        assert (float(lab.x) - lo) / (hi - lo) > 0.85
        assert lab.xanchor == "right"
        mid = fp.build_percentile_figure(M, _p(xmin=2010, xmax=2080))
        assert {a.xanchor for a in _price_labels(mid)} == {"center"}

    def test_label_headroom(self):
        """Review M-3: room for a 3-line box over a Q95 peak."""
        fig = fp.build_percentile_figure(M, _p())
        lo, hi = fig.layout.yaxis.range
        assert lo <= -26 and hi >= 126

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
        assert "\u00b7 log-time \u00b7" in fig.layout.title.text
        assert any("log-time element" in ln for ln in fig.layout.meta["cycle_info"])

    def test_cycle_result_is_not_pathologically_slow(self):
        """~1.0 s cold when run alone on dev; the suite runs -n logical (24
        workers), which took it to 5 s. This ceiling only catches an order-of-
        magnitude regression (e.g. a daily grid or a lost cache)."""
        fp._cycle_result.cache_clear()
        t0 = time.perf_counter()
        fp._cycle_result("qr", "resqr", False, "cal", str(_app_ctx._MODEL_FP))
        assert time.perf_counter() - t0 < 20.0


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
_LAB = "bub-pctile-fit-labels:value"
_PRE_LABELS_FP = "757574d3"   # last fingerprint before the labels switch
# The operator's own share link (2026-11-14): Percentile, linear, slider
# 2010-2027, fit on QR, horizon +20 -> prod drew the forecast to ~Nov 2046.
_OPERATOR_LINK = ("757574d3:H4sIAEWp-GoC_w2MMQrEMBAD_6J6bXZtbwz-SkgR7lwEjAkhqcL9_"
                  "YQKMQzSiupM-WbIi4yGccy-XxA42prUVJKmugkWcrDo4tGJRgbVEsyCFQ5MK5o"
                  "KkrNx9uvT532MTpUz36zYT-YzxvYHsN3TJXYAAAA=")


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
        assert ids["bub-pctile-fit-horizon"].style.get("display") == "none"
        assert ids["bub-pctile-fit-labels"].value == ["labels"]
        assert [o["value"] for o in ids["bub-pctile-fit-labels"].options] == ["labels"]
        models = [o["value"] for o in ids["bub-pctile-fit-model"].options]
        assert models[0] == "qr" and set(models) <= set(pc.CYCLE_MODEL_KEYS)
        assert [o["value"] for o in ids["bub-pctile-fit-basis"].options] == list(pc.BASES)
        assert [o["value"] for o in ids["bub-pctile-fit-horizon"].options] == [10, 20, 30]
        assert not hasattr(pc, "HORIZONS") and not hasattr(fp, "_LINEAR_LABELS")

    def test_horizon_is_not_an_input_labels_is(self):
        from test_no_orphan_callbacks import _all_callbacks
        cm = _all_callbacks()
        (ids,) = [[i["id"] for i in v["inputs"]] for k, v in cm.items()
                  if "bub-pctile-graph.figure" in k
                  and any(i["id"] == "bub-pctile-fit" for i in v["inputs"])]
        assert "bub-pctile-fit-horizon" not in ids
        assert "bub-pctile-fit-labels" in ids

    def test_update_callback_passes_coerced_cycle_params(self):
        from callbacks.charts import update_bub_pctile
        seen = {}

        def grab(p):
            seen.update(p)
            return fp.build_percentile_figure(M, p)
        args = ("percentile", [2010, 2027], [], "lin", ["bub"], "outside",
                "default", "resqr")
        with patch("utils._get_pctile_fig", side_effect=grab):
            update_bub_pctile(*args, ["fit", "censor"], "pl", "log", [], None, False)
            assert (seen["cycle_fit"], seen["cycle_censor"], seen["cycle_model"],
                    seen["cycle_basis"], seen["cycle_labels"]) == (True, True, "pl", "log", False)
            assert "cycle_horizon" not in seen
            update_bub_pctile(*args, ["fit"], "qr", "cal", ["labels"], None, False)
            assert seen["cycle_labels"] is True
            update_bub_pctile(*args, ["fit"], "qr", "cal", None, None, False)
            assert seen["cycle_labels"] is False
            update_bub_pctile(*args, None, "u1", "zzz", 7, None, False)
            assert (seen["cycle_fit"], seen["cycle_censor"], seen["cycle_model"],
                    seen["cycle_basis"], seen["cycle_labels"]) == (False, False, "qr", "cal", True)

    def test_info_div_is_filled_clientside_from_layout_meta(self):
        scripts = [s for s in _app_ctx.app._inline_scripts if "cycle_info" in s]
        assert len(scripts) == 1
        cm = _app_ctx.app.callback_map
        keys = [k for k in cm if "bub-pctile-info.children" in k]
        assert len(keys) == 1
        assert [i["id"] for i in cm[keys[0]]["inputs"]] == ["bub-pctile-graph"]

    def test_info_div_js_parses(self, tmp_path):
        """Regression: a Python-level "\\n" became a raw newline inside a JS
        string literal, so the browser rejected the function and the summary
        never showed. Pin the escape, and syntax-check with node when present."""
        import shutil
        import subprocess
        (js,) = [s for s in _app_ctx.app._inline_scripts if "cycle_info" in s]
        assert 'join("\\n")' in js
        node = shutil.which("node")
        if node is None:
            pytest.skip("node not installed")
        start = js.index("function(fig)")
        depth, end = 0, None
        for i, ch in enumerate(js[start:], start):
            depth += ch == "{"
            depth -= ch == "}"
            if ch == "}" and depth == 0:
                end = i + 1
                break
        f = tmp_path / "fn.js"
        f.write_text("var f = " + js[start:end] + ";\n")
        r = subprocess.run([node, "--check", str(f)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr


class TestSnapshot:
    def test_new_fields_appended_at_the_absolute_tail(self):
        assert [c for c, _ in _SNAPSHOT_CONTROLS[348:352]] == _NEW
        assert all(p == "value" for _, p in _SNAPSHOT_CONTROLS[348:352])
        assert _SNAPSHOT_CONTROLS[352] == ("bub-pctile-fit-labels", "value")
        assert len(_SNAPSHOT_CONTROLS) == 353
        assert _SNAPSHOT_CONTROLS[347] == ("bub-mc-extremes", "value")

    def test_defaults_options_and_tab_controls(self):
        from callbacks.routing import _TAB_CONTROLS
        assert SNAPSHOT_DEFAULTS[_FIT] == ["fit"]
        assert SNAPSHOT_DEFAULTS["bub-pctile-fit-model:value"] == "qr"
        assert SNAPSHOT_DEFAULTS["bub-pctile-fit-basis:value"] == "cal"
        assert SNAPSHOT_DEFAULTS["bub-pctile-fit-horizon:value"] == 20
        assert _CHECKLIST_OPTIONS["bub-pctile-fit"] == ["fit", "censor"]
        assert set(_NEW) <= _TAB_CONTROLS["bubble"]
        assert SNAPSHOT_DEFAULTS[_LAB] == ["labels"]
        assert _CHECKLIST_OPTIONS["bub-pctile-fit-labels"] == ["labels"]
        assert "bub-pctile-fit-labels" in _TAB_CONTROLS["bubble"]

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


class _Trig:
    def __init__(self, tid):
        self.triggered_id = tid
        self.triggered_prop_ids = {f"{tid}.value": tid}


class TestDashFold:
    def test_unregistered_fingerprint_link_restores_with_the_fit_off(self):
        """Dash review M1: an evicted / unknown fp falls back to today's
        defaults, which must not switch the fit on for an old link."""
        state = _decode_snapshot_v4(
            _v4("deadbeef", {str(_idx("bub-view-mode")): "percentile"}))
        assert state["bub-view-mode:data"] == "percentile"
        assert state[_FIT] == []

    def test_fit_off_pins_sub_settings_to_one_cache_key(self):
        from callbacks.charts import _cycle_params
        a = _cycle_params([], "pl", "log", [])
        b = _cycle_params(None, "bub", "calw", ["labels"])
        assert a == b == dict(cycle_fit=False, cycle_censor=False,
                              cycle_model="qr", cycle_basis="cal",
                              cycle_labels=True)

    def test_sub_dropdown_with_fit_off_does_not_rebuild(self):
        import dash
        from callbacks.charts import update_bub_pctile
        args = ("percentile", [2010, 2027], [], "lin", ["bub"], "outside",
                "default", "resqr")
        with patch("callbacks.charts.ctx", _Trig("bub-pctile-fit-basis")):
            assert update_bub_pctile(*args, [], "qr", "log", ["labels"], None, False) \
                is dash.no_update
        with patch("callbacks.charts.ctx", _Trig("bub-pctile-fit-labels")):
            assert update_bub_pctile(*args, [], "qr", "cal", [], None, False) \
                is dash.no_update
        with patch("callbacks.charts.ctx", _Trig("bub-pctile-fit")), \
                patch("utils._get_pctile_fig", return_value=fp.go.Figure()):
            assert update_bub_pctile(*args, [], "qr", "cal", ["labels"], None, False) \
                is not dash.no_update               # unticking redraws

    def test_one_availability_filter(self):
        from callbacks.charts import _cycle_params
        avail = pc.available_models(_app_ctx.PRICE_MODELS)
        assert avail[0] == "qr" and set(avail) <= set(pc.CYCLE_MODEL_KEYS)
        fake = {k: v for k, v in _app_ctx.PRICE_MODELS.items() if k != "ef"}
        assert "ef" not in pc.available_models(fake)
        with patch.dict(_app_ctx.PRICE_MODELS, {}, clear=False):
            _app_ctx.PRICE_MODELS.pop("ef", None)
            assert _cycle_params(["fit"], "ef", "cal", ["labels"])["cycle_model"] == "qr"


class TestWholeDiffFold:
    _PRICE = re.compile(r"^\$[\d.,]+[kMBT]?$")

    def test_hover_prices_are_preformatted_not_si(self):
        """I-1: d3 '.3s' rendered $0.36 as '$362m' (milli)."""
        fig = fp.build_percentile_figure(M, _p())
        for tr in fig.data:
            assert ".3s" not in (tr.hovertemplate or ""), tr.name
        w = [t for t in fig.data if (t.name or "").startswith("QR cycle fit")][0]
        early = w.customdata[0]                     # 2010: sub-dollar fan
        for v in early[5:8]:
            assert self._PRICE.match(v), v
        assert early[5].startswith("$0.")
        turns = [t for t in fig.data if (t.name or "").endswith("fitted turns")][0]
        assert all(self._PRICE.match(r[2]) for r in turns.customdata)

    def test_crafted_basis_type_does_not_raise(self):
        """M-1: a list/dict from a hand-made link must coerce, not 500."""
        from callbacks.charts import _cycle_params
        for bad in (["cal"], {"a": 1}, 3):
            assert _cycle_params(["fit"], "qr", bad, 20)["cycle_basis"] == "cal"

    def test_far_timing_bounds_are_clamped_not_crashing(self, monkeypatch):
        """M-2: a bound centuries out used to overflow pandas (year 2262)."""
        monkeypatch.setattr(fp._pc, "turn_timing",
                            lambda params, t_a, j, k, t_e, space="cal": (t_e - 900.0, t_e + 900.0))
        fig = fp.build_percentile_figure(M, _p())
        (tr,) = _forecast_turn_traces(fig)
        assert all("or later" in r[4] for r in tr.customdata)
        assert max(np.add(tr.x, tr.error_x.array)) <= float(M.price_years[-1]) + fp._TIMING_CAP + 1e-6

    def test_optimistic_no_edge_box_says_even_against_todays_fan(self, monkeypatch):
        """M-3: a non-QR box must not claim an out-of-sample check."""
        fp._cycle_result.cache_clear()
        monkeypatch.setattr(fp._pc, "beats_flat", lambda hind: False)
        try:
            fig = fp.build_percentile_figure(M, _p(cycle_model="pl"))
            txt = " ".join(a.text or "" for a in _anns(fig))
            assert "even against today's fan" in txt and "out-of-sample edge" not in txt
            info = " ".join(fig.layout.meta["cycle_info"])
            assert "optimistic check" in info and "loses even so" in info
        finally:
            fp._cycle_result.cache_clear()

    def test_at_limit_legend_says_so(self):
        """M-4: HybPPL's calendar fit sits on the 2-yr grid edge."""
        r = fp._cycle_result("hybppl", "resqr", False, "cal", str(_app_ctx._MODEL_FP))
        if not r["fit"].at_limit:
            pytest.skip("HybPPL no longer at the search limit with today's data")
        fig = fp.build_percentile_figure(M, _p(cycle_model="hybppl"))
        assert "HybPPL cycle fit (period at search limit)" in _names(fig)

    def test_censored_hindcast_line_says_the_fits_withheld_a_window(self):
        """M-5."""
        info = fp.build_percentile_figure(M, _p(cycle_censor=True)).layout.meta["cycle_info"]
        assert any("Each fit withheld 2019" in ln for ln in info)

    def test_chunked_grid_scan_matches_one_pass(self):
        """M-6: chunking is a memory change only."""
        t, y = _synthetic(noise=10.0, ar=0.99)
        fs = np.linspace(1 / 8, 1 / 2, 400)
        assert pc._grid_best(t, y, fs, chunk=50) == pc._grid_best(t, y, fs, chunk=10_000)


class TestSliderLabelsSnapshot:
    """Note 2026-11-14 D4 + D7: the labels switch and old links' forecast."""

    def _st(self, **kw):
        st = {"bub-view-mode:data": "percentile", _FIT: ["fit"],
              "bub-pctile-fit-horizon:value": 20, "bub-xrange:value": [2010, 2027]}
        st.update(kw)
        return st

    def test_operator_link_keeps_its_forecast_length(self):
        st = _decode_snapshot_v4(_OPERATOR_LINK)
        y = date.today().year
        assert _LAB not in st
        assert st["bub-xscale:value"] == "linear"
        assert st["bub-xrange:value"] == [2010, min(y + 21, 2080)]

    def test_old_link_on_another_pill_keeps_its_range(self):
        from snapshot import _migrate_pctile_horizon
        st = _migrate_pctile_horizon(self._st(**{"bub-view-mode:data": "price"}))
        assert st["bub-xrange:value"] == [2010, 2027]

    def test_old_link_ending_before_today_untouched(self):
        from snapshot import _migrate_pctile_horizon
        y = date.today().year
        st = _migrate_pctile_horizon(self._st(**{"bub-xrange:value": [2010, y]}))
        assert st["bub-xrange:value"] == [2010, y]

    def test_new_link_untouched(self):
        from snapshot import _migrate_pctile_horizon
        st = _migrate_pctile_horizon(self._st(**{_LAB: ["labels"]}))
        assert st["bub-xrange:value"] == [2010, 2027]

    def test_exact_end_not_max_and_clamped(self):
        from snapshot import _migrate_pctile_horizon, _XRANGE_MAX
        from layout.bubble import XRANGE_MAX
        assert _XRANGE_MAX == XRANGE_MAX
        y = date.today().year
        st = _migrate_pctile_horizon(self._st(**{
            "bub-pctile-fit-horizon:value": 10, "bub-xrange:value": [2010, 2060]}))
        assert st["bub-xrange:value"] == [2010, min(y + 11, 2080)]
        with patch("snapshot._XRANGE_MAX", y + 5):
            st = _migrate_pctile_horizon(self._st())
        assert st["bub-xrange:value"] == [2010, y + 5]

    def test_fit_off_bad_horizon_or_bad_range_untouched(self):
        from snapshot import _migrate_pctile_horizon
        for kw in ({_FIT: []}, {"bub-pctile-fit-horizon:value": 7},
                   {"bub-xrange:value": None}, {"bub-xrange:value": "x"}):
            st = self._st(**kw)
            before = dict(st)
            assert _migrate_pctile_horizon(st) == before, kw
        assert _migrate_pctile_horizon({"bub-view-mode:data": "percentile",
                                        _FIT: ["fit"]}) is not None

    def test_old_links_lack_the_labels_key_new_ones_carry_it(self):
        old = _decode_snapshot_v4(_v4(_PRE_LABELS_FP, {}))
        assert _LAB not in old
        values = [None] * _idx("bub-pctile-fit-labels")
        j = json.dumps([values, None], separators=(",", ":"))
        pos = _decode_snapshot(base64.urlsafe_b64encode(gzip.compress(j.encode())).decode())
        assert _LAB not in pos
        new = _decode_snapshot_v4(_v4(_compute_snapshot_defaults_fingerprint(), {}))
        assert new[_LAB] == ["labels"]
        off = _decode_snapshot_v4(_encode_snapshot_v4({_LAB: []}))
        assert off[_LAB] == []

    def test_evicted_fingerprint_still_migrates(self):
        """Review I-1: an unregistered fp falls back to today's defaults,
        which must not hand an old link the labels key."""
        st = _decode_snapshot_v4(_v4("deadbeef", {
            str(_idx("bub-view-mode")): "percentile",
            str(_idx("bub-pctile-fit")): 1,
            str(_idx("bub-pctile-fit-horizon")): 20,
            str(_idx("bub-xrange")): [2010, 2027]}))
        assert _LAB not in st
        assert st["bub-xrange:value"] == [2010, min(date.today().year + 21, 2080)]

    def test_current_fp_registered_and_prod_era_fp_kept(self):
        from snapshot import _registry_lookup
        assert _registry_lookup(_compute_snapshot_defaults_fingerprint()) is not None
        assert _registry_lookup(_PRE_LABELS_FP) is not None


class TestSliderLabelsWholeDiffFold:
    def test_evicting_the_pre_labels_fp_still_migrates(self, monkeypatch):
        """Whole-diff review I-2: once 757574d3 itself is evicted every
        retained entry carries the labels key; the link must still migrate."""
        import snapshot
        reg = dict(snapshot._load_registry())
        reg.pop(_PRE_LABELS_FP)
        fake = {_compute_snapshot_defaults_fingerprint(): reg[
            _compute_snapshot_defaults_fingerprint()]}
        monkeypatch.setattr(snapshot, "_registry_cache", fake)
        assert snapshot._registry_lookup(_PRE_LABELS_FP) is None
        st = _decode_snapshot_v4(_OPERATOR_LINK)
        assert _LAB not in st
        assert st["bub-xrange:value"] == [2010, min(date.today().year + 21, 2080)]

    def test_returning_from_a_capped_view_restores_the_default_end(self):
        import dash
        from callbacks.charts import toggle_bub_view
        cap = date.today().year + 1
        with patch("callbacks.charts.ctx", _Trig("bub-view-pctile")):
            out = toggle_bub_view(0, 0, 0, 1, 0, [2010, cap])
            assert out[-1] == [2010, 2033]
            out = toggle_bub_view(0, 0, 0, 1, 0, [2015, 2040])
            assert out[-1] is dash.no_update
        with patch("callbacks.charts.ctx", _Trig("bub-view-resid")):
            assert toggle_bub_view(1, 0, 0, 0, 0, [2010, cap])[-1] is dash.no_update
