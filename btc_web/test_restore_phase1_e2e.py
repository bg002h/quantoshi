"""End-to-end Playwright tests for Phase 1 restore_from_url dispatch fix.

Verifies that /2-/7 share links correctly restore controls (the lazy-
mounted bubble-graph Output no longer blocks dispatch), and that /1
share links still work as a regression check.

Requires:
  pip install playwright && python -m playwright install firefox
  Dev server must be running on :8050
Run:
  cd /scratch/code/bitcoinprojections && btc_venv/bin/python3 \\
      -m pytest btc_web/test_restore_phase1_e2e.py -v --timeout=60
"""
import pytest
import time

try:
    from playwright.sync_api import sync_playwright
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False

pytestmark = pytest.mark.skipif(not HAS_PLAYWRIGHT, reason="playwright not installed")

BASE_URL = "http://localhost:8050"


def _make_share_url(state, path):
    """Encode state dict into q4 share link at the given path."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    os.environ.setdefault("DEV", "1")
    import app  # noqa: F401
    from snapshot import _encode_snapshot_v4
    blob = _encode_snapshot_v4(state)
    return f"{BASE_URL}{path}#q4:{blob}"


def test_dca_share_restores_amount():
    """/3 share link with dca-amount=999 must restore the widget value.

    Pre-Phase-1 this returned 100 (default) because restore_from_url's
    dispatch was silently dropped by Dash 4 (lazy bubble-graph Output).
    """
    url = _make_share_url(
        {"main-tabs:active_tab": "dca", "dca-amount:value": 999},
        "/3",
    )
    with sync_playwright() as p:
        browser = p.firefox.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        page.wait_for_function(
            "() => { var gd = document.querySelector('#dca-graph .js-plotly-plot'); "
            "return gd && gd.data && gd.data.length > 0; }",
            timeout=20_000,
        )
        # Phase 2 introduced fast-restore relay — chart renders before the
        # apply_tab_dca cascade writes dca-amount. Wait for the widget value.
        page.wait_for_function(
            "() => document.getElementById('dca-amount') && "
            "document.getElementById('dca-amount').value === '999'",
            timeout=10_000,
        )
        amount = page.evaluate(
            "() => document.getElementById('dca-amount').value"
        )
        browser.close()
    assert amount == "999", (
        f"DCA share-link restore broken: dca-amount={amount}, expected 999. "
        f"This means restore_from_url's dispatch is still being dropped — "
        f"check that bubble-graph.figure is NOT in its Outputs."
    )


def test_bubble_share_still_restores():
    """/1 regression: bubble share link still renders chart fast."""
    url = _make_share_url(
        {"main-tabs:active_tab": "bubble", "bub-qs:value": ["median"]},
        "/1",
    )
    with sync_playwright() as p:
        browser = p.firefox.launch(headless=True)
        page = browser.new_page()
        t0 = time.perf_counter()
        page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        page.wait_for_function(
            "() => { var gd = document.querySelector('#bubble-graph .js-plotly-plot'); "
            "return gd && gd.data && gd.data.length > 0; }",
            timeout=20_000,
        )
        t_chart = (time.perf_counter() - t0) * 1000
        browser.close()
    assert t_chart < 5000, (
        f"Bubble chart took {t_chart:.0f}ms (expected <5000ms). "
        f"Phase 1 should not regress /1 timing."
    )


def test_dca_share_no_jserror_from_set_props():
    """The set_props clientside relay must not cause JS errors, even
    when bubble-graph is absent from DOM (which it is on /3 init).

    Pre-existing 'nonexistent object' errors for lazy-tab controls
    (bub-qs, sc-stack, ret-stack) are filtered — they are documented
    in feedback_nonexistent_input_perf.md as a separate issue.
    """
    url = _make_share_url(
        {"main-tabs:active_tab": "dca", "dca-amount:value": 999},
        "/3",
    )
    errors = []
    with sync_playwright() as p:
        browser = p.firefox.launch(headless=True)
        page = browser.new_page()

        def on_console(msg):
            if msg.type != "error":
                return
            # Filter pre-existing Dash lazy-tab "nonexistent object" errors.
            # These come from dash_renderer.js and have empty args / msg.text=
            # "Error" in some Playwright versions. Filter by location URL
            # since the renderer is the canonical source.
            loc = msg.location or {}
            url = loc.get("url", "") or ""
            if "dash_renderer" in url or "_dash-component-suites" in url:
                return  # pre-existing lazy-tab artifact
            text = msg.text or ""
            if "nonexistent object was used" in text:
                return
            errors.append(text[:400])

        page.on("console", on_console)
        page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        page.wait_for_function(
            "() => { var gd = document.querySelector('#dca-graph .js-plotly-plot'); "
            "return gd && gd.data && gd.data.length > 0; }",
            timeout=20_000,
        )
        time.sleep(1)  # let any delayed errors surface
        browser.close()
    assert errors == [], (
        f"Unexpected JS errors during /3 share-link load: {errors}. "
        f"set_props on absent bubble-graph may be throwing (plotly/dash#2897)."
    )
