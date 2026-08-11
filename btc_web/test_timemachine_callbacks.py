"""Task 10 — Time Machine callback tests.

Covers the pure gate helper ``_asof_frame`` that translates the Time Machine
toggle + as-of slider value into the ``asof_frame`` index threaded into
``update_bubble``'s figure params. The reveal / play / exclusion callbacks are
clientside (JS) and covered by the ``import app`` boot check + manual DEV
verification rather than unit tests.
"""
from callbacks.timemachine import _asof_frame


def test_asof_frame_gate():
    # Toggle off → no as-of frame regardless of slider value.
    assert _asof_frame([], 5) is None
    # Toggle on + a slider value → that value is the frame index.
    assert _asof_frame(["on"], 5) == 5
    # Toggle on but no slider value yet → None (nothing to render as-of).
    assert _asof_frame(["on"], None) is None


def test_asof_frame_none_toggle():
    # A None toggle (never initialised) behaves like "off".
    assert _asof_frame(None, 3) is None


def test_asof_frame_zero_index():
    # Frame index 0 is the FIRST frame and is valid — the gate must use an
    # explicit ``is not None`` check, never a falsy test that swallows 0.
    assert _asof_frame(["on"], 0) == 0


def test_tm_eligible_models():
    # Time Machine single-model constraint admits exactly BM, the EPPL master,
    # QR, spl, and logi — the five keys _asof_resolve has an as-of path for.
    from callbacks.timemachine import _TM_ELIGIBLE
    assert set(_TM_ELIGIBLE) == {"bub", "eppl", "qr", "spl", "logi"}


def test_update_bub_swatches_preserves_tm_restriction():
    """Regression: a tab re-visit bumps bubble-first-render, which fires
    callbacks.charts._update_bub_swatches — the callback that owns
    bub-model-show.options. It MUST keep the Time Machine restriction (only
    eligible models) while TM is on, or the ineligible models silently reappear
    on the next re-render (_tm_single_model is toggle-driven and does not
    re-fire on a re-render)."""
    from callbacks.charts import _update_bub_swatches
    from callbacks.timemachine import _TM_ELIGIBLE
    off = {o["value"] for o in _update_bub_swatches(1, "default", {}, [])}
    on = {o["value"] for o in _update_bub_swatches(1, "default", {}, ["on"])}
    assert len(off) > len(_TM_ELIGIBLE)            # TM off → full model list
    assert on <= set(_TM_ELIGIBLE)                 # TM on → only eligible
    assert "bub" in on
    assert not (on & {"lppl", "hybppl", "pca"})    # ineligible models hidden
