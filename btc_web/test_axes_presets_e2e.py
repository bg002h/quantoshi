"""End-to-end Playwright tests for Tab 1's one-tap axes presets.

REQUIRED, not optional: the Dash 4.0 multi-Input bug's failure mode is a
silent no-op, so every preset button must actually be clicked in a browser.

Requires: pip install playwright && python -m playwright install firefox
Run:      cd /scratch/code/bitcoinprojections && btc_venv/bin/python3 \
              -m pytest btc_web/test_axes_presets_e2e.py -v --timeout=90
          (dev server must be running on :8050)
"""
import datetime
import time

import pytest

try:
    from playwright.sync_api import sync_playwright
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False

pytestmark = pytest.mark.skipif(not HAS_PLAYWRIGHT, reason="playwright not installed")

BASE_URL = "http://localhost:8050"


# Dash 4 renders its own slider (NOT rc-slider). Each thumb is a
# span.dash-slider-thumb carrying aria-valuenow. Verified in the live DOM.
def _slider(page, eid):
    return page.evaluate(
        f'[...document.querySelectorAll("#{eid} [aria-valuenow]")]'
        f'.map(n => Number(n.getAttribute("aria-valuenow")))')


def _radio(page, eid):
    return page.evaluate(
        f'(() => {{ const r = [...document.querySelectorAll('
        f'"#{eid} input[type=radio]")].find(x => x.checked);'
        f' return r ? r.value : null; }})()')


def _n_checked(page, eid):
    return page.evaluate(
        f'[...document.querySelectorAll("#{eid} input[type=checkbox]")]'
        f'.filter(c => c.checked).length')


def _wait_until(predicate, timeout=15.0):
    """Poll until predicate() is truthy. Returns the final value."""
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = predicate()
        if last:
            return last
        time.sleep(0.25)
    return last


@pytest.fixture(scope="module")
def page():
    with sync_playwright() as p:
        browser = p.firefox.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1280, "height": 900})
        pg = ctx.new_page()
        pg.goto(f"{BASE_URL}/1", wait_until="networkidle", timeout=30000)
        pg.wait_for_selector("#bub-axes-presets", state="attached", timeout=15000)
        time.sleep(1.5)  # let the restore/first-render cascade settle
        yield pg
        browser.close()


def test_current_year_sets_xrange(page):
    yr = datetime.date.today().year
    page.click("#bub-axes-preset-cur_year")
    got = _wait_until(lambda: _slider(page, "bub-xrange") == [yr, yr + 1])
    assert _slider(page, "bub-xrange") == [yr, yr + 1], (
        f"expected [{yr}, {yr + 1}], got {_slider(page, 'bub-xrange')}. "
        "A silent no-op here means the Dash multi-Input bug, a missing "
        "callbacks/__init__.py import, or a JS error -- check the console.")


def test_current_year_leaves_scales_alone(page):
    """Per-preset field ownership (spec D1): cur_year writes X range only."""
    before = (_radio(page, "bub-xscale"), _radio(page, "bub-yscale"))
    page.click("#bub-axes-preset-cur_year")
    time.sleep(1.0)
    assert (_radio(page, "bub-xscale"), _radio(page, "bub-yscale")) == before


LINK_XRANGE = [2015, 2040]  # deliberately not the system default


@pytest.fixture(scope="module")
def share_hash():
    """A share link whose bub-xrange differs from the system default.

    Uses the v4 encoder/prefix -- production always emits q4 (see
    callbacks/snapshot_cb.py's share-link builder), never the legacy q3
    tested elsewhere in this suite. This matters here specifically: q4
    decode back-fills EVERY control from the historical-defaults registry
    (snapshot.py::_decode_snapshot_v4), so snap["bub-xrange:value"] is
    present in the restored state whether or not the link itself supplied
    it. That's exactly the condition layout/bubble.py::_JS_DEFAULT's
    equal-to-default narrowing exists to detect -- a q3-encoded fixture
    would never exercise it, because q3 omits fields the link didn't set.

    Also encodes bub-auto-y:value=[] (Auto OFF) to exercise the
    empty-list-is-not-"absent" hazard: a regression to `snap[k] || fallback`
    style presence testing must not silently re-enable auto-Y when the link
    asked for it off.

    Proves it decodes before any browser sees it, so a prefix/encoder mismatch
    fails here with a clear message instead of as a mystery browser failure.
    """
    # btc_web/ is already on sys.path via btc_web/conftest.py, which pytest
    # auto-loads for this directory. No path manipulation needed.
    from snapshot import _encode_snapshot_v4, _decode_snapshot_v4, _SNAP_PREFIX_V4

    # _decode_snapshot_v4 takes the fp-prefixed blob WITHOUT the "q4:" prefix
    # -- the real dispatcher (callbacks/snapshot_cb.py::
    # _decode_snapshot_by_prefix) strips "q4:" before calling it, and
    # _encode_snapshot_v4 already returns "<fp>:<blob>" (the fp is part of
    # its return value, not the prefix). Decode that directly; prepend only
    # "q4:" for the URL-facing return value.
    encoded = _encode_snapshot_v4({
        'bub-xrange:value': LINK_XRANGE,
        'bub-auto-y:value': [],
    })
    decoded = _decode_snapshot_v4(encoded)
    assert decoded is not None, (
        f"{_SNAP_PREFIX_V4!r} does not pair with _encode_snapshot_v4 -- use "
        "the encoder matching the current prefix.")
    assert decoded.get("bub-xrange:value") == LINK_XRANGE
    assert decoded.get("bub-auto-y:value") == []
    return f"{_SNAP_PREFIX_V4}{encoded}"


def test_default_restores_system_defaults(page):
    """No share link: Default returns all five controls to factory values."""
    # The module-scoped `page` carries state from the two `cur_year` tests
    # that ran before this one (bub-xrange moved to [cur_yr, cur_yr+1]).
    # Capturing baseline_y without reloading grabs auto-Y's fit for THAT
    # narrow window, not the page's true on-load fit -- reload first so the
    # baseline actually matches what "load" means. This only shows up under
    # serial execution (`-p no:randomly -n0`); pytest.ini's `-n auto` masks
    # it because each test lands on its own xdist worker with a fresh page.
    page.goto(f"{BASE_URL}/1", wait_until="networkidle", timeout=30000)
    page.wait_for_selector("#bub-axes-presets", state="attached", timeout=15000)
    time.sleep(1.5)
    baseline_y = _slider(page, "bub-yrange")

    # Disturb the axes using the feature itself plus two direct clicks.
    page.click("#bub-axes-preset-cur_year")
    time.sleep(0.8)
    page.click("#bub-xscale input[value='linear']")
    page.click("#bub-auto-y input[type=checkbox]")
    time.sleep(1.0)
    assert _n_checked(page, "bub-auto-y") == 0

    page.click("#bub-axes-preset-default")
    _wait_until(lambda: _slider(page, "bub-xrange") == [2010, 2033])

    assert _slider(page, "bub-xrange") == [2010, 2033]
    assert _radio(page, "bub-xscale") == "log"
    assert _radio(page, "bub-yscale") == "log"
    assert _n_checked(page, "bub-auto-y") == 1
    # Y is asserted against the baseline captured on load, NOT against
    # SNAPSHOT_DEFAULTS: with auto-Y on, auto_bubble_yrange recomputes and
    # rounds, so a freshly loaded page reads [-1.5, 6.0], not [-1.5, 6.05]
    # (spec section 7.2).
    _wait_until(lambda: _slider(page, "bub-yrange") == baseline_y)
    assert _slider(page, "bub-yrange") == baseline_y


def test_default_restores_share_link_xrange(page, share_hash):
    """ACCEPTANCE TEST for the URL requirement.

    On a page loaded from a share link, Default returns to THAT LINK's axes,
    not the system defaults.
    """
    # "#" is the hash separator (CLAUDE.md: "URL format: host/N#q3:...",
    # same rule for q4 -- share_hash already carries the "q4:" prefix). An
    # earlier draft glued the blob onto the path instead
    # (f"{BASE_URL}/1{share_hash}"), which the router doesn't recognize.
    page.goto(f"{BASE_URL}/1#{share_hash}", wait_until="networkidle", timeout=30000)
    page.wait_for_selector("#bub-axes-presets", state="attached", timeout=15000)
    _wait_until(lambda: _slider(page, "bub-xrange") == LINK_XRANGE)
    assert _slider(page, "bub-xrange") == LINK_XRANGE, "share link did not restore"

    # Move away from the link's range, then ask for it back.
    page.click("#bub-axes-preset-cur_year")
    _wait_until(lambda: _slider(page, "bub-xrange") != LINK_XRANGE)
    assert _slider(page, "bub-xrange") != LINK_XRANGE

    page.click("#bub-axes-preset-default")
    _wait_until(lambda: _slider(page, "bub-xrange") == LINK_XRANGE)
    assert _slider(page, "bub-xrange") == LINK_XRANGE, (
        "Default fell back to the system default instead of the link's value")
    # []-falsy hazard: the link's bub-auto-y was [] (Auto OFF). A regression
    # to `snap[k] || fallback`-style presence testing in _JS_DEFAULT would
    # treat [] as absent and silently re-enable auto-Y here instead of
    # honoring the link's value.
    assert _n_checked(page, "bub-auto-y") == 0, (
        "Default re-enabled auto-Y instead of honoring the link's [] value "
        "-- check _JS_DEFAULT's presence test for a `||`/falsy regression")

    page.goto(f"{BASE_URL}/1", wait_until="networkidle", timeout=30000)
    page.wait_for_selector("#bub-axes-presets", state="attached", timeout=15000)
    time.sleep(1.0)


def test_default_is_view_aware_in_cagr(page):
    """No share link + CAGR view: the fallback is the CAGR range."""
    page.click("#bub-view-cagr")
    _wait_until(lambda: _slider(page, "bub-xrange") == [2025, 2050])

    page.click("#bub-axes-preset-cur_year")
    _wait_until(lambda: _slider(page, "bub-xrange") != [2025, 2050])

    page.click("#bub-axes-preset-default")
    _wait_until(lambda: _slider(page, "bub-xrange") == [2025, 2050])
    assert _slider(page, "bub-xrange") == [2025, 2050]

    page.click("#bub-view-price")  # restore for any later test
    time.sleep(0.8)
