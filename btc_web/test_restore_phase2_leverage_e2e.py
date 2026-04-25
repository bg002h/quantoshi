"""Phase 2 ship 4 end-to-end Playwright tests for /7 (leverage) fast modal close.

Leverage is simpler than DCA/retire/SC: 3 Outputs (figure, readout, table),
no MC, no SC. Only the figure is delivered via the relay; readout + table
come from the cascade. No post-restore guard on update_leverage.

Requires:
  pip install playwright && python -m playwright install firefox
  Dev server must be running on :8050
Run:
  cd /scratch/code/bitcoinprojections && btc_venv/bin/python3 \\
      -m pytest btc_web/test_restore_phase2_leverage_e2e.py -v --timeout=60 -n 0
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


def test_lev_share_fast_modal_close():
    """/7 share with non-default lev-horizon=8 renders chart fast.

    lev-horizon is a Slider component without a DOM .value attribute, so
    we verify restore worked by checking the readout text contains the
    encoded horizon (cascade-rendered after widget value applied).
    """
    url = _make_share_url(
        {
            "main-tabs:active_tab": "leverage",
            "lev-horizon:value": 8.0,
            "lev-cagr:value": 30.0,
            "lev-price:value": 100000.0,
        },
        "/7",
    )
    with sync_playwright() as p:
        browser = p.firefox.launch(headless=True)
        page = browser.new_page()
        t0 = time.perf_counter()
        page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        page.wait_for_function(
            "() => { var gd = document.querySelector('#lev-graph .js-plotly-plot'); "
            "return gd && gd.data && gd.data.length > 0; }",
            timeout=20_000,
        )
        t_chart = (time.perf_counter() - t0) * 1000
        # Cascade fills readout with "Horizon H = 8.00 yr" after apply_tab_leverage
        page.wait_for_function(
            "() => { var el = document.getElementById('lev-readout'); "
            "return el && el.textContent && el.textContent.includes('8.00 yr'); }",
            timeout=15_000,
        )
        readout_text = page.evaluate(
            "() => document.getElementById('lev-readout').textContent"
        )
        browser.close()
    assert "8.00 yr" in readout_text, (
        f"lev-horizon=8 not reflected in readout: {readout_text[:200]}"
    )
    assert t_chart < 5000, (
        f"Leverage chart took {t_chart:.0f}ms (expected <5000ms with fast path)"
    )


def test_lev_readout_fills_via_cascade():
    """/7 share — readout should fill via cascade (lev-readout is NOT
    delivered by the relay; only the figure is)."""
    url = _make_share_url(
        {
            "main-tabs:active_tab": "leverage",
            "lev-horizon:value": 5.0,
        },
        "/7",
    )
    with sync_playwright() as p:
        browser = p.firefox.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        page.wait_for_function(
            "() => { var gd = document.querySelector('#lev-graph .js-plotly-plot'); "
            "return gd && gd.data && gd.data.length > 0; }",
            timeout=20_000,
        )
        # Cascade rebuilds figure + readout + table; wait for readout text.
        page.wait_for_function(
            "() => { var el = document.getElementById('lev-readout'); "
            "return el && el.textContent && el.textContent.length > 50; }",
            timeout=15_000,
        )
        readout_text = page.evaluate(
            "() => document.getElementById('lev-readout').textContent"
        )
        browser.close()
    # readout includes "Buy:" + "Sell:" + "Horizon" lines (per _readout helper)
    assert "Buy" in readout_text and "Sell" in readout_text, (
        f"readout missing expected content: {readout_text[:200]}"
    )


def test_supercharge_share_still_restores():
    """Phase 2 ship 3 regression: /5 supercharge share still works fast."""
    url = _make_share_url(
        {
            "main-tabs:active_tab": "supercharge",
            "sc-wd:value": 12000,
            "sc-stack:value": 1.0,
            "sc-model-show:value": ["bub"],
        },
        "/5",
    )
    with sync_playwright() as p:
        browser = p.firefox.launch(headless=True)
        page = browser.new_page()
        t0 = time.perf_counter()
        page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        page.wait_for_function(
            "() => { var gd = document.querySelector('#supercharge-graph .js-plotly-plot'); "
            "return gd && gd.data && gd.data.length > 0; }",
            timeout=20_000,
        )
        t_chart = (time.perf_counter() - t0) * 1000
        browser.close()
    assert t_chart < 5000, f"/5 supercharge took {t_chart:.0f}ms (regression)"
