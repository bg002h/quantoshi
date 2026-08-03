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
