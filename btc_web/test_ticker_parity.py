"""The navbar percentile and the Percentile view must be the same number.

Reported by the operator 2026-09-06: the navbar showed BM ~99 % while
`/1.4` showed ~79 % for the same coin at the same moment. Diagnosed in
`docs/superpowers/agent-reports/2026-09-06-ticker-vs-percentile-mismatch.md`.

Cause: `find_percentile(t, price)` takes `sigma_mode="constant"` from its
signature default (`btc_core/_base.py`), and the ticker never passed one,
while every chart runs at the app default `"resqr"`
(`tab_defaults.py`). Measured at the last CSV close, t=17.117:

    constant -> 99.020 %      <- navbar
    resqr    -> 79.652 %      <- /1.4
                19.37 pp apart, of which the live price and the deliberate
                data lag explain only ~0.4 pp.

Because the ticker cycles ten models, this is not one wrong number but
seven (`qr`, `lp3` and `ef` have no resqr bundle, so σ-mode is a no-op for
them).
"""
import os

import pytest

os.environ.setdefault("TESTING", "1")


@pytest.fixture(scope="module")
def ctx():
    import app  # noqa: F401
    import _app_ctx
    from btc_core._helpers import today_t
    import pandas as pd
    import pathlib

    csv = pathlib.Path(__file__).resolve().parent.parent / "BitcoinPricesDaily.csv"
    price = float(pd.read_csv(csv).iloc[-1, -1])
    return _app_ctx, today_t(_app_ctx.M.genesis), price


def test_ticker_uses_the_same_sigma_mode_the_charts_render_with(ctx):
    """The knob itself, so a future edit to either default is caught even if
    both happen to agree numerically on the day the test runs."""
    from callbacks.ticker import _ticker_sigma_mode
    from tab_defaults import bubble_defaults

    assert _ticker_sigma_mode() == bubble_defaults()["sigma_mode"]


def test_navbar_percentile_equals_percentile_view_for_every_cycled_model(ctx):
    """The number a user reads in the navbar must be the number the chart
    would put them at, for every model the ticker can cycle to."""
    _app_ctx, t, price = ctx
    from callbacks.ticker import _MODEL_CYCLE, _ticker_sigma_mode
    from tab_defaults import bubble_defaults

    view_mode = bubble_defaults()["sigma_mode"]
    disagree = []
    for key in _MODEL_CYCLE:
        mdl = _app_ctx.PRICE_MODELS.get(key)
        if mdl is None:
            continue
        ticker_p = mdl.find_percentile(t, price, sigma_mode=_ticker_sigma_mode())
        view_p = mdl.find_percentile(t, price, sigma_mode=view_mode)
        if ticker_p is None or view_p is None:
            continue
        if abs(ticker_p - view_p) > 1e-12:
            disagree.append(f"{key}: navbar {ticker_p*100:.3f}% vs "
                            f"view {view_p*100:.3f}% "
                            f"({abs(ticker_p-view_p)*100:.2f} pp)")
    assert not disagree, (
        "navbar percentile disagrees with the Percentile view:\n  "
        + "\n  ".join(disagree))


def test_bubble_model_specifically_agrees(ctx):
    """BM is the default model and the one the operator reported. Its
    constant-σ fan has collapsed to 1.69x Q1->Q99 at t=17.1 against 3.29x
    for resqr, which is why it is the worst-affected of the ten."""
    _app_ctx, t, price = ctx
    from callbacks.ticker import _ticker_sigma_mode
    from tab_defaults import bubble_defaults

    m = _app_ctx.PRICE_MODELS["bub"]
    navbar = m.find_percentile(t, price, sigma_mode=_ticker_sigma_mode())
    view = m.find_percentile(t, price, sigma_mode=bubble_defaults()["sigma_mode"])
    assert navbar == pytest.approx(view, abs=1e-12), (
        f"navbar {navbar*100:.3f}% vs view {view*100:.3f}%")


def test_markov_keeps_the_constant_default(ctx):
    """NOT everything should move to resqr: the 1.2 GB MC cache is binned
    against constant-σ percentiles, so changing markov.py's call would
    invalidate every cached scenario. Guard against a well-meaning sweep.
    """
    import pathlib
    src = (pathlib.Path(__file__).resolve().parent / "markov.py").read_text()
    idx = src.find("find_percentile")
    assert idx > 0, "markov.py no longer calls find_percentile — re-check this guard"
    call = src[idx:idx + 200]
    assert "sigma_mode" not in call.split(")")[0], (
        "markov.py must keep the constant-σ default — the MC cache is binned "
        "against it (see the 2026-09-06 ticker mismatch report)")
