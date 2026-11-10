"""End-to-end: a real tick of Tab 1's MC "Activate" widens the X range; a
programmatic enable does not.

The unit half (test_mc_autowiden.py) runs the JS under node. This is the
browser wiring: does a real click reach the clientside callback, and does a
write of bub-mc-enable that is NOT a click (share-link restore, Time Machine
force-off) leave the range alone?

Requires: pip install playwright && python -m playwright install firefox
Run:      cd /scratch/code/bitcoinprojections && btc_venv/bin/python3 \
              -m pytest btc_web/test_mc_autowiden_e2e.py -v -n0 -p no:randomly
          (dev server must be running on :8050)
"""
import time

import pytest

from layout.bubble import MC_AUTOWIDEN_YEARS

try:
    from playwright.sync_api import sync_playwright
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False

pytestmark = pytest.mark.skipif(not HAS_PLAYWRIGHT, reason="playwright not installed")

BASE_URL = "http://localhost:8050"
_MC_BOX = "#bub-mc-enable input[type=checkbox]"


def _slider(page, eid):
    return page.evaluate(
        f'[...document.querySelectorAll("#{eid} [aria-valuenow]")]'
        f'.map(n => Number(n.getAttribute("aria-valuenow")))')


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
    """A fresh page per test: both tests start from MC off, default range."""
    with sync_playwright() as p:
        browser = p.firefox.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1280, "height": 900})
        pg = ctx.new_page()
        pg.goto(f"{BASE_URL}/1", wait_until="networkidle", timeout=30000)
        pg.wait_for_selector(_MC_BOX, state="attached", timeout=15000)
        time.sleep(1.5)  # let the restore/first-render cascade settle
        yield pg
        browser.close()


def test_a_real_tick_widens_the_right_edge_to_start_plus_ten(page):
    lo, hi = _slider(page, "bub-xrange")
    box = page.locator(_MC_BOX).first
    box.scroll_into_view_if_needed()
    box.click()
    got = _wait_until(lambda: (lambda r: r if r[1] > hi else None)(
        _slider(page, "bub-xrange")))
    assert got, f"X range never widened from [{lo}, {hi}]"
    # Cached start years are 2028 / 2031 / 2035; the default is the first.
    assert got[0] == lo
    assert got[1] in {y + MC_AUTOWIDEN_YEARS for y in (2028, 2031, 2035)}, got


def test_a_programmatic_enable_leaves_the_range_alone(page):
    before = _slider(page, "bub-xrange")
    page.evaluate(
        'window.dash_clientside.set_props("bub-mc-enable", {value: ["yes"]})')
    assert _wait_until(lambda: page.locator(_MC_BOX).first.is_checked()), \
        "set_props did not enable MC — the test would pass vacuously"
    time.sleep(3.0)  # the widen (if wrongly fired) lands well inside this
    assert _slider(page, "bub-xrange") == before
