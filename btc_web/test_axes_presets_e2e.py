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

    Proves it decodes before any browser sees it, so a prefix/encoder mismatch
    fails here with a clear message instead of as a mystery browser failure.
    """
    # btc_web/ is already on sys.path via btc_web/conftest.py, which pytest
    # auto-loads for this directory. No path manipulation needed.
    from snapshot import _encode_snapshot, _decode_snapshot, _SNAP_PREFIX

    # _decode_snapshot takes the PREFIX-STRIPPED blob -- the real dispatcher
    # (callbacks/snapshot_cb.py::_decode_snapshot_by_prefix) strips "q3:"
    # before calling it, and every other test in test_snapshot.py follows the
    # same convention. Decode the unprefixed string here; prepend the prefix
    # only for the URL-facing return value.
    encoded = _encode_snapshot({'bub-xrange:value': LINK_XRANGE})
    decoded = _decode_snapshot(encoded)
    assert decoded is not None, (
        f"{_SNAP_PREFIX!r} does not pair with _encode_snapshot -- use the "
        "encoder matching the current prefix.")
    assert decoded.get("bub-xrange:value") == LINK_XRANGE
    return f"{_SNAP_PREFIX}{encoded}"


def test_default_restores_system_defaults(page):
    """No share link: Default returns all five controls to factory values."""
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
    # "#" is the hash separator (CLAUDE.md: "URL format: host/N#q3:..."); the
    # brief's literal f"{BASE_URL}/1{share_hash}" glues the blob onto the
    # path instead, which the router doesn't recognize.
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
