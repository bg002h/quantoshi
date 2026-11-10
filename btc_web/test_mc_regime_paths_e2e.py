"""End-to-end: a real click that applies a price-regime filter ticks "Show
individual price paths"; a programmatic write of the regime value does not.

The unit half (test_mc_regime_paths.py) runs the JS under node. This is the
browser wiring: does a real click reach the clientside callback via the
capture-phase stamp, and does set_props (what a share-link restore amounts to)
leave the paths box alone?

Requires: pip install playwright && python -m playwright install firefox
Run:      cd /scratch/code/bitcoinprojections && btc_venv/bin/python3 \
              -m pytest btc_web/test_mc_regime_paths_e2e.py -v -n0 -p no:randomly
          (dev server must be running on :8050)
"""
import time

import pytest

try:
    from playwright.sync_api import sync_playwright
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False

pytestmark = pytest.mark.skipif(not HAS_PLAYWRIGHT, reason="playwright not installed")

BASE_URL = "http://localhost:8050"
_MC = "#bub-mc-enable input[type=checkbox]"
_ADV = "#bub-mc-advanced input[type=checkbox]"
_PATHS = "#bub-mc-paths input[type=checkbox]"
_REGIME = "#bub-mc-regime input[type=checkbox]"


def _wait_until(predicate, timeout=15.0):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = predicate()
        if last:
            return last
        time.sleep(0.25)
    return last


@pytest.fixture()
def page():
    """Fresh page, MC on and the advanced panel open, paths box UNticked."""
    p = sync_playwright().start()
    browser = p.firefox.launch(headless=True)
    pg = browser.new_context(viewport={"width": 1280, "height": 900}).new_page()
    pg.goto(f"{BASE_URL}/1", wait_until="networkidle", timeout=30000)
    pg.wait_for_selector(_MC, state="attached", timeout=15000)
    time.sleep(1.5)
    for sel in (_MC, _ADV):
        box = pg.locator(sel).first
        box.scroll_into_view_if_needed()
        box.click()
    pg.wait_for_selector(_REGIME, state="visible", timeout=10000)
    assert not pg.locator(_PATHS).first.is_checked(), \
        "fixture: paths box must start unticked (default off)"
    yield pg
    browser.close()
    p.stop()


def _paths_on(page):
    return page.locator(_PATHS).first.is_checked()


def test_a_real_click_that_applies_the_filter_ticks_paths(page):
    page.locator(_REGIME).nth(4).click()          # untick Bubble
    assert _wait_until(lambda: _paths_on(page)), \
        "paths box did not tick after a real regime click"


def test_a_manual_untick_survives_the_next_regime_click(page):
    page.locator(_REGIME).nth(4).click()
    assert _wait_until(lambda: _paths_on(page))
    page.locator(_PATHS).first.click()            # user turns paths back off
    assert _wait_until(lambda: not _paths_on(page))
    page.locator(_REGIME).nth(3).click()          # narrow the filter further
    time.sleep(3.0)
    assert not _paths_on(page), "a later regime click overrode the user's untick"


def test_a_programmatic_regime_write_leaves_paths_alone(page):
    page.evaluate('window.dash_clientside.set_props("bub-mc-regime", {value: [0, 1]})')
    assert _wait_until(lambda: not page.locator(_REGIME).nth(4).is_checked()), \
        "set_props did not change the regime — the test would pass vacuously"
    time.sleep(3.0)
    assert not _paths_on(page)


def test_clearing_the_filter_does_not_untick_paths(page):
    page.locator(_REGIME).nth(4).click()
    assert _wait_until(lambda: _paths_on(page))
    page.locator(_REGIME).nth(4).click()          # all five checked again
    time.sleep(3.0)
    assert _paths_on(page)
