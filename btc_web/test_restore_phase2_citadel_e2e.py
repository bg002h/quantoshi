"""Phase 2 ship 6 end-to-end Playwright tests for /6 (citadel) fast modal close.

Citadel is unique — its update_citadel uses background=True (Celery), and
the simulation only runs on user button click. On citadel-first-render it
loads a CACHED DEFAULT chart (Q25% bub). Phase 2 ship 6 doesn't build a
figure server-side; it just writes active-chart-committed=hash_str so the
modal closes fast via the active-chart-committed listener. The cached
default chart loads instantly via the existing cascade.

Requires:
  pip install playwright && python -m playwright install firefox
  Dev server must be running on :8050
Run:
  cd /scratch/code/bitcoinprojections && btc_venv/bin/python3 \\
      -m pytest btc_web/test_restore_phase2_citadel_e2e.py -v --timeout=60 -n 0
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


def test_cp_share_loads_with_cached_default():
    """/6 share link loads cached default citadel chart fast.

    The cached default chart (Q25% bub) loads via the citadel-first-render
    path, NOT from the share-link state. The user has to click "Run
    Simulation" to compute their actual share-link scenario. Phase 2 ship
    6 only enables fast modal close — it doesn't build the figure.
    """
    url = _make_share_url(
        {
            "main-tabs:active_tab": "citadel",
            "cp-spend:value": 8000,
            "cp-stack:value": 5.0,
        },
        "/6",
    )
    with sync_playwright() as p:
        browser = p.firefox.launch(headless=True)
        page = browser.new_page()
        t0 = time.perf_counter()
        page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        page.wait_for_function(
            "() => { var gd = document.querySelector('#citadel-graph .js-plotly-plot'); "
            "return gd && gd.data && gd.data.length > 0; }",
            timeout=20_000,
        )
        t_chart = (time.perf_counter() - t0) * 1000
        # Verify the share-link controls were applied (cascade path).
        page.wait_for_function(
            "() => document.getElementById('cp-spend') && "
            "document.getElementById('cp-spend').value === '8000'",
            timeout=10_000,
        )
        spend = page.evaluate("() => document.getElementById('cp-spend').value")
        browser.close()
    assert spend == "8000", f"cp-spend={spend}, expected 8000"
    assert t_chart < 5000, (
        f"Citadel chart took {t_chart:.0f}ms (expected <5000ms with cached default)"
    )


def test_heatmap_share_still_restores():
    """Phase 2 ship 5 regression: /2 heatmap share still works fast."""
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
    assert t_chart < 5000, f"/2 heatmap took {t_chart:.0f}ms (regression)"
