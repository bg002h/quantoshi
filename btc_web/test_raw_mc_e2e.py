"""End-to-end: Tab 1's model-free "Raw" Monte Carlo, driven in a real browser.

The unit halves (test_raw_mc_*.py) cover the simulation, the figure and the
callback in-process, and run the clientside JS under node. This is the browser
wiring: the Dash 4 dropdown actually switching mode, the CSS that hides the
paid / premium chrome (a rule that only a computed style can prove), the real
click stamps behind the "regime filter ticks paths" and "tick MC widens the X
range" callbacks, and update_bubble being called once per regime click.

Every "is not shown" assertion is paired with the same element shown in model
mode first (_model_chrome_is_shown), so a wrong selector cannot pass vacuously.

Requires: pip install playwright && python -m playwright install firefox
Run:      cd /scratch/code/bitcoinprojections-raw-mc && btc_venv/bin/python3 \
              -m pytest btc_web/test_raw_mc_e2e.py -v -n0 -p no:randomly -o addopts=""
          (dev server must be running on :8050, started from THIS checkout)
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
_MC_BOX = "#bub-mc-enable input[type=checkbox]"
_PATHS = "#bub-mc-paths input[type=checkbox]"
_EXTREMES = "#bub-mc-extremes input[type=checkbox]"
_RAW_REGIME = "#bub-mc-raw-regime input[type=checkbox]"
_PANEL = "#bub-mc-panel"
_WRAP = "#bubble-graph-chart-wrap"
_RAW_MODE_CLASS = "qs-mc-raw-mode"
_PREMIUM_CLASS = "mc-premium-chart"
_RAW_CYCLE = "Raw · halving cycle"
_RAW_IID = "Raw · independent"
_BUBBLE = "Bubble Model"
_DEFAULT_LINES = 20


def _wait_until(predicate, timeout=20.0):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = predicate()
        if last:
            return last
        time.sleep(0.25)
    return last


class _Env:
    """A page plus every update_bubble POST it has sent since the last clear.

    update_bubble is the only callback that writes bub-mc-raw-status.children,
    so that output key in the POST body identifies it (its first Output,
    bubble-graph.figure, is shared with allow_duplicate Patch callbacks).
    """

    def __init__(self, page):
        self.page = page
        self.bubble_posts = []
        page.on("request", self._on_request)

    def _on_request(self, req):
        if req.method != "POST" or "_dash-update-component" not in req.url:
            return
        try:
            out = req.post_data_json.get("output", "")
        except Exception:                       # not JSON: not ours
            return
        if "bubble-graph.figure" in out and "bub-mc-raw-status.children" in out:
            self.bubble_posts.append(out)


def _open():
    p = sync_playwright().start()
    browser = p.firefox.launch(headless=True)
    pg = browser.new_context(viewport={"width": 1280, "height": 900}).new_page()
    pg.goto(f"{BASE_URL}/1", wait_until="networkidle", timeout=30000)
    pg.wait_for_selector(_MC_BOX, state="attached", timeout=15000)
    time.sleep(1.5)                  # let the restore / first-render cascade settle
    return p, browser, _Env(pg)


def _tick_mc(page):
    box = page.locator(_MC_BOX).first
    box.scroll_into_view_if_needed()
    box.click()                      # a real click: the autowiden stamp needs one
    assert _wait_until(lambda: page.locator(_MC_BOX).first.is_checked()), \
        "the click did not tick MC"


def _names(page):
    return page.evaluate(
        """() => { const g = document.querySelector('#bubble-graph .js-plotly-plot');
                   return g && g.data ? g.data.map(d => d.name || null) : []; }""")


def _has_trace(page, text):
    return any(n and text in n for n in _names(page))


def _line_count(page, group):
    return page.evaluate(
        """(group) => { const g = document.querySelector('#bubble-graph .js-plotly-plot');
                        return g && g.data ? g.data.filter(
                            d => d.legendgroup === group && d.mode === 'lines').length : 0; }""",
        group)


def _slider(page, eid):
    return page.evaluate(
        f'[...document.querySelectorAll("#{eid} [aria-valuenow]")]'
        f'.map(n => Number(n.getAttribute("aria-valuenow")))')


def _choose_src(page, label):
    """Open the Dash 4 model-source dropdown and click the option's text.

    The option list lives in a popup whose id is the trigger's aria-controls;
    scoping to it keeps "Bubble Model" from matching the model-show checklist
    and the other tabs' source dropdowns.
    """
    trigger = page.locator("#bub-mc-model-src")
    trigger.scroll_into_view_if_needed()
    trigger.click()
    popup = page.locator(f'[id="{trigger.get_attribute("aria-controls")}"]')
    popup.get_by_role("option", name=label, exact=True).click()


def _panel_in_raw_mode(page):
    return _RAW_MODE_CLASS in (page.get_attribute(_PANEL, "class") or "").split()


def _wrap_is_premium(page):
    return _PREMIUM_CLASS in (page.get_attribute(_WRAP, "class") or "").split()


def _model_chrome_is_shown(page):
    return {
        "run button": any(b.is_visible() for b in
                          page.locator(f"{_PANEL} [id*='mc-run-btn']").all()),
        "paid badge": any(b.is_visible() for b in
                          page.locator(f"{_PANEL} .model-panel-paid-badge").all()),
        "saved simulation card": any(b.is_visible() for b in
                                     page.locator(_PANEL).get_by_text("Saved Simulation").all()),
        "cost line": page.locator("#bub-mc-cost").is_visible(),
    }


@pytest.fixture()
def mc_on():
    """Fresh /1, MC ticked by a real click, model fan drawn (model mode)."""
    p, browser, env = _open()
    _tick_mc(env.page)
    assert _wait_until(lambda: _has_trace(env.page, "MC median")), \
        "model-mode MC fan never drew"
    env.page.wait_for_selector("#bub-mc-model-src", state="visible", timeout=10000)
    yield env
    browser.close()
    p.stop()


@pytest.fixture()
def raw_on(mc_on):
    """mc_on, then 'Raw · halving cycle' chosen and its fan drawn."""
    _choose_src(mc_on.page, _RAW_CYCLE)
    assert _wait_until(lambda: _has_trace(mc_on.page, "MC median · raw cycle")), \
        f"Raw fan never drew; traces: {_names(mc_on.page)}"
    time.sleep(1.0)                  # let any in-flight duplicate settle
    return mc_on


@pytest.fixture()
def fresh():
    """Fresh /1 with MC OFF and the default X range."""
    p, browser, env = _open()
    yield env
    browser.close()
    p.stop()


# ── the Raw fan and the chrome it removes ────────────────────────────────────

def test_raw_cycle_draws_its_fan_and_drops_the_paid_chrome(mc_on):
    page = mc_on.page
    # Control: model mode shows every piece of chrome we are about to expect gone.
    assert _model_chrome_is_shown(page) == {
        "run button": True, "paid badge": True,
        "saved simulation card": True, "cost line": True}
    assert _wait_until(lambda: _wrap_is_premium(page)), \
        "control: model-mode MC should carry the premium chart class"
    assert not _panel_in_raw_mode(page)

    _choose_src(page, _RAW_CYCLE)
    assert _wait_until(lambda: _has_trace(page, "MC median · raw cycle")), \
        f"no 'MC median · raw cycle' legend entry; traces: {_names(page)}"
    assert _panel_in_raw_mode(page)

    status = page.locator("#bub-mc-raw-status")
    assert status.is_visible(), "Raw status line is not visible"
    assert _wait_until(lambda: status.inner_text().strip()), "Raw status line is empty"

    shown = [k for k, v in _model_chrome_is_shown(page).items() if v]
    assert not shown, f"model/paid chrome still visible in Raw: {shown}"
    assert _wait_until(lambda: not _wrap_is_premium(page)), \
        "bubble-graph-chart-wrap still has mc-premium-chart in Raw"
    assert page.locator("#bub-mc-model-src").is_visible()
    # Raw-only controls are shown, model-only ones are not.
    for sel in ("#bub-mc-raw-regime", "#bub-mc-lines", "#bub-mc-extremes"):
        assert page.locator(sel).is_visible(), f"{sel} should be visible in Raw"
    assert not page.locator("#bub-mc-start-yr").is_visible()


def test_raw_independent_names_its_legend_entry(mc_on):
    _choose_src(mc_on.page, _RAW_IID)
    assert _wait_until(lambda: _has_trace(mc_on.page, "MC median · raw independent")), \
        f"traces: {_names(mc_on.page)}"
    assert _panel_in_raw_mode(mc_on.page)


# ── example lines and extremes ───────────────────────────────────────────────

def test_ticking_paths_draws_twenty_lines_and_extremes_add_two(raw_on):
    page = raw_on.page
    assert _line_count(page, "mc-spaghetti") == 0, "control: no lines before the tick"
    assert not _has_trace(page, "Highest end")

    page.locator(_PATHS).first.click()
    assert _wait_until(lambda: _line_count(page, "mc-spaghetti") == _DEFAULT_LINES), \
        f"expected {_DEFAULT_LINES} line traces, got {_line_count(page, 'mc-spaghetti')}"
    assert not _has_trace(page, "Highest end"), "extremes drew before their box was ticked"

    page.locator(_EXTREMES).first.click()
    assert _wait_until(lambda: _has_trace(page, "Highest end")
                       and _has_trace(page, "Lowest end")), \
        f"extremes missing; traces: {_names(page)}"
    assert _line_count(page, "mc-spaghetti") == _DEFAULT_LINES


# ── the regime box: a real click ticks paths, in one redraw ──────────────────

def test_a_real_raw_regime_click_ticks_paths(raw_on):
    page = raw_on.page
    assert not page.locator(_PATHS).first.is_checked(), "fixture: paths must start off"
    boxes = page.locator(_RAW_REGIME)
    assert boxes.count() == 5
    boxes.nth(4).scroll_into_view_if_needed()
    boxes.nth(4).click()
    assert not boxes.nth(4).is_checked(), "the click did not untick the regime box"
    assert _wait_until(lambda: page.locator(_PATHS).first.is_checked()), \
        "paths box did not tick after a real Raw regime click"
    assert _wait_until(lambda: _line_count(page, "mc-spaghetti") == _DEFAULT_LINES), \
        "the ticked paths box produced no example lines"


def test_a_raw_regime_click_costs_exactly_one_update_bubble_post(raw_on):
    page = raw_on.page
    boxes = page.locator(_RAW_REGIME)
    boxes.nth(4).scroll_into_view_if_needed()
    raw_on.bubble_posts.clear()
    boxes.nth(4).click()
    assert _wait_until(lambda: page.locator(_PATHS).first.is_checked()), \
        "paths box did not tick, so the click path under test never ran"
    assert _wait_until(lambda: _line_count(page, "mc-spaghetti") == _DEFAULT_LINES)
    time.sleep(4.0)                  # a duplicate POST (review I-2) lands well inside this
    assert len(raw_on.bubble_posts) == 1, \
        f"{len(raw_on.bubble_posts)} update_bubble POSTs for one regime click"


def test_a_programmatic_raw_regime_write_leaves_paths_alone(raw_on):
    page = raw_on.page
    page.evaluate(
        'window.dash_clientside.set_props("bub-mc-raw-regime", {value: [0, 1, 2, 3]})')
    assert _wait_until(lambda: not page.locator(_RAW_REGIME).nth(4).is_checked()), \
        "set_props did not change the regime: the test would pass vacuously"
    time.sleep(3.0)
    assert not page.locator(_PATHS).first.is_checked()


# ── auto-widen: Raw starts today, not at bub-mc-start-yr ─────────────────────

def test_a_real_mc_tick_in_raw_widens_to_this_year_plus_ten(fresh):
    page = fresh.page
    this_year = page.evaluate("new Date().getFullYear()")
    before = _slider(page, "bub-xrange")
    # MC is off, so the model-source box is folded away: set it as a share-link
    # restore would, with a start year whose model-mode answer is far from Raw's.
    page.evaluate('window.dash_clientside.set_props("bub-mc-model-src", {value: "raw_cycle"})')
    page.evaluate('window.dash_clientside.set_props("bub-mc-start-yr", {value: 2035})')
    assert _wait_until(lambda: _panel_in_raw_mode(page)), "set_props did not select Raw"
    assert _wait_until(lambda: "2035" in (page.locator("#bub-mc-start-yr").text_content() or "")), \
        "set_props did not set the start year: the test would pass vacuously"
    assert not page.locator(_MC_BOX).first.is_checked()
    assert _slider(page, "bub-xrange") == before

    raw_hi = max(before[1], this_year + 10)
    model_hi = max(before[1], 2035 + 10)
    assert raw_hi != model_hi, "this year is too close to 2035 to tell Raw from model mode"

    _tick_mc(page)
    got = _wait_until(lambda: (lambda r: r if r != before else None)(_slider(page, "bub-xrange")))
    assert got, f"X range never widened from {before}"
    assert got == [this_year, raw_hi], \
        f"got {got}; Raw wants [{this_year}, {raw_hi}], model mode would give hi={model_hi}"
    assert _wait_until(lambda: _has_trace(page, "MC median · raw cycle")), \
        f"the tick did not draw the Raw fan; traces: {_names(page)}"


# ── back to the model ────────────────────────────────────────────────────────

def test_switching_back_to_bubble_restores_model_chrome_and_fan(raw_on):
    page = raw_on.page
    assert _panel_in_raw_mode(page)
    _choose_src(page, _BUBBLE)
    assert _wait_until(lambda: not _panel_in_raw_mode(page)), "Raw mode class stuck"

    # The model fan returns and the Raw one is gone.
    assert _wait_until(lambda: _has_trace(page, "MC median")
                       and not _has_trace(page, "raw")), \
        f"model fan did not replace the Raw fan; traces: {_names(page)}"
    assert _wait_until(lambda: all(_model_chrome_is_shown(page).values())), \
        f"model chrome not restored: {_model_chrome_is_shown(page)}"
    assert _wait_until(lambda: _wrap_is_premium(page)), \
        "model mode lost the premium chart class"
    for sel in ("#bub-mc-raw-status", "#bub-mc-raw-regime", "#bub-mc-lines",
                "#bub-mc-extremes"):
        assert not page.locator(sel).is_visible(), f"{sel} still visible in model mode"
    assert page.locator("#bub-mc-start-yr").is_visible()
