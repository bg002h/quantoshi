"""Setting a periodic-power-law component count to 0 must drop that oscillator.

Reported by the operator 2026-09-07: "setting low period components to 0 in
config panel doesn't change the displayed trace".

Cause: the HybPPL/EPPL master resolvers built their config key with

    _build_hybppl_config_key(cfg_a_nlog or 1, cfg_a_ncal or 1, ...)

and ``0 or 1`` is ``1``. A user asking for zero log-periodic oscillators got
the one-oscillator model, silently — n_log=0 and n_log=1 both resolved to
``cfg_1d_1u``.

This is the falsy-zero footgun CLAUDE.md already warns about for callback
inputs ("use ``float(x) if x is not None else default``"), reappearing in the
resolvers. The zero-component models are registered and fine — ``cfg_0_0``,
``cfg_0_1d``, ``cfg_0_1u`` and friends all exist — so nothing was missing
except the resolver's willingness to ask for them.

Model B already used ``or 0`` and so was never affected.
"""
import os

import pytest

os.environ.setdefault("TESTING", "1")

_A_DIRS = ("d", "d", "u", "u")          # log1d, log2d, cal1d, cal2d
_B_OFF = ([], 0, 0, "d", "d", "u", "u")  # model B disabled


def _hyb(nlog, ncal):
    import app  # noqa: F401
    from callbacks.charts._resolvers import _resolve_hybppl_master
    return _resolve_hybppl_master(["hybppl"], nlog, ncal, *_A_DIRS, *_B_OFF)


def _ep(nlog, ncal):
    import app  # noqa: F401
    from callbacks.charts._resolvers import _resolve_eppl_master
    return _resolve_eppl_master(["eppl"], nlog, ncal, *_A_DIRS, *_B_OFF)


# ── the zero-component models exist ─────────────────────────────────────────

@pytest.mark.parametrize("key", ["cfg_0_0", "cfg_0_1d", "cfg_0_1u",
                                 "ecfg_0_0", "ecfg_0_1d", "ecfg_0_1u"])
def test_zero_component_variants_are_registered(key):
    import app  # noqa: F401
    import _app_ctx
    assert key in _app_ctx.PRICE_MODELS, (
        f"{key} is not registered — the resolver cannot select it")


# ── zero must mean zero ─────────────────────────────────────────────────────

def test_hybppl_zero_log_oscillators_selects_a_zero_log_model():
    assert _hyb(0, 1) == ["cfg_0_1u"]


def test_hybppl_zero_cal_oscillators_selects_a_zero_cal_model():
    assert _hyb(1, 0) == ["cfg_1d_0"]


def test_hybppl_zero_of_both_selects_the_plain_model():
    assert _hyb(0, 0) == ["cfg_0_0"]


def test_eppl_zero_log_oscillators_selects_a_zero_log_model():
    assert _ep(0, 1) == ["ecfg_0_1u"]


def test_eppl_zero_cal_oscillators_selects_a_zero_cal_model():
    assert _ep(1, 0) == ["ecfg_1d_0"]


@pytest.mark.parametrize("resolver,prefix", [(_hyb, "cfg"), (_ep, "ecfg")])
def test_zero_is_distinguishable_from_one(resolver, prefix):
    """The bug's signature: 0 and 1 produced the SAME key, so the chart never
    changed. They must differ on both axes."""
    assert resolver(0, 1) != resolver(1, 1), f"{prefix}: n_log 0 == n_log 1"
    assert resolver(1, 0) != resolver(1, 1), f"{prefix}: n_cal 0 == n_cal 1"


# ── None still falls back, and the other counts still work ──────────────────

def test_none_still_defaults_to_one():
    """Only 0 changed meaning. An unset value keeps the previous default, so a
    share link or a fresh layout that omits the field is unaffected."""
    assert _hyb(None, None) == ["cfg_1d_1u"]
    assert _ep(None, None) == ["ecfg_1d_1u"]


@pytest.mark.parametrize("nlog,ncal,want", [
    (1, 1, "cfg_1d_1u"),
    (2, 1, "cfg_2dd_1u"),
    (1, 2, "cfg_1d_2uu"),
    (2, 2, "cfg_2dd_2uu"),
])
def test_nonzero_counts_unchanged(nlog, ncal, want):
    assert _hyb(nlog, ncal) == [want]


def test_no_master_is_left_alone():
    import app  # noqa: F401
    from callbacks.charts._resolvers import _resolve_hybppl_master
    assert _resolve_hybppl_master(["bub", "pl"], 0, 0, *_A_DIRS,
                                  *_B_OFF) == ["bub", "pl"]
