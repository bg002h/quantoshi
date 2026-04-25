"""Phase 2 ship 5 end-to-end Playwright tests for /2 (heatmap) fast modal close.

Heatmap is structurally different — active model comes from `hm-active-model`
Store (set by pill bar), not from `hm-model-show` checklist.

Requires:
  pip install playwright && python -m playwright install firefox
  Dev server must be running on :8050
Run:
  cd /scratch/code/bitcoinprojections && btc_venv/bin/python3 \\
      -m pytest btc_web/test_restore_phase2_heatmap_e2e.py -v --timeout=60 -n 0
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


def test_hm_share_fast_modal_close():
    """/2 share with non-default entry-yr renders chart fast."""
    url = _make_share_url(
        {
            "main-tabs:active_tab": "heatmap",
            "hm-active-model:data": "bub",
            "hm-entry-yr:value": 2030,
        },
        "/2",
    )
    with sync_playwright() as p:
        browser = p.firefox.launch(headless=True)
        page = browser.new_page()
        t0 = time.perf_counter()
        page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        page.wait_for_function(
            "() => { var gd = document.querySelector('#heatmap-graph .js-plotly-plot'); "
            "return gd && gd.data && gd.data.length > 0; }",
            timeout=20_000,
        )
        t_chart = (time.perf_counter() - t0) * 1000
        browser.close()
    assert t_chart < 5000, (
        f"Heatmap chart took {t_chart:.0f}ms (expected <5000ms with fast path)"
    )


def test_hm_mc_share_falls_back():
    """MC-enabled /2 share — modal closes via cascade (MC pipeline complex)."""
    url = _make_share_url(
        {
            "main-tabs:active_tab": "heatmap",
            "hm-mc-enable:value": ["yes"],
            "hm-active-model:data": "bub",
        },
        "/2",
    )
    with sync_playwright() as p:
        browser = p.firefox.launch(headless=True)
        page = browser.new_page()
        t0 = time.perf_counter()
        page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        page.wait_for_function(
            "() => { var gd = document.querySelector('#heatmap-graph .js-plotly-plot'); "
            "return gd && gd.data && gd.data.length > 0; }",
            timeout=25_000,
        )
        t_chart = (time.perf_counter() - t0) * 1000
        browser.close()
    assert t_chart < 25000, (
        f"MC fallback heatmap took {t_chart:.0f}ms (>25s — fallback broken)"
    )


def test_hm_no_phantom_rebuild():
    """Post-restore guard suppresses cascade rebuilds."""
    url = _make_share_url(
        {
            "main-tabs:active_tab": "heatmap",
            "hm-active-model:data": "bub",
            "hm-entry-yr:value": 2030,
            "hm-entry-q:value": 25,
        },
        "/2",
    )
    with sync_playwright() as p:
        browser = p.firefox.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        page.wait_for_function(
            "() => { var gd = document.querySelector('#heatmap-graph .js-plotly-plot'); "
            "return gd && gd.data && gd.data.length > 0; }",
            timeout=20_000,
        )
        time.sleep(1.5)
        count = page.evaluate("() => window.__heatmapBuildCount || 0")
        browser.close()
    assert count == 1, (
        f"heatmap-build-count={count} after restore (expected 1). "
        f">=2 means the post-restore guard failed and the cascade "
        f"rebuilt the figure unnecessarily."
    )


def test_leverage_share_still_restores():
    """Phase 2 ship 4 regression: /7 leverage share still works fast."""
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
        t0 = time.perf_counter()
        page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        page.wait_for_function(
            "() => { var gd = document.querySelector('#lev-graph .js-plotly-plot'); "
            "return gd && gd.data && gd.data.length > 0; }",
            timeout=20_000,
        )
        t_chart = (time.perf_counter() - t0) * 1000
        browser.close()
    assert t_chart < 5000, f"/7 leverage took {t_chart:.0f}ms (regression)"
