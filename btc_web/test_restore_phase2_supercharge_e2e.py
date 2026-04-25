"""Phase 2 ship 3 end-to-end Playwright tests for /5 (supercharge) fast modal close.

Mirrors test_restore_phase2_dca_e2e.py / test_restore_phase2_retire_e2e.py.

Requires:
  pip install playwright && python -m playwright install firefox
  Dev server must be running on :8050
Run:
  cd /scratch/code/bitcoinprojections && btc_venv/bin/python3 \\
      -m pytest btc_web/test_restore_phase2_supercharge_e2e.py -v --timeout=90 -n 0
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


def test_sc_share_fast_modal_close():
    """/5 share with sc-wd=12000 renders chart fast and restores widget.
    Explicitly disables MC for fast-path assertion (default flipped to on
    in commit 093c665)."""
    url = _make_share_url(
        {
            "main-tabs:active_tab": "supercharge",
            "sc-wd:value": 12000,
            "sc-stack:value": 1.0,
            "sc-model-show:value": ["bub"],
            "sc-mc-enable:value": [],
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
        page.wait_for_function(
            "() => document.getElementById('sc-wd') && "
            "document.getElementById('sc-wd').value === '12000'",
            timeout=10_000,
        )
        wd = page.evaluate("() => document.getElementById('sc-wd').value")
        browser.close()
    assert wd == "12000", f"sc-wd={wd}, expected 12000"
    assert t_chart < 5000, (
        f"Supercharge chart took {t_chart:.0f}ms (expected <5000ms with fast path)"
    )


def test_sc_mc_share_falls_back():
    """MC-enabled /5 share — modal eventually closes via cascade path."""
    url = _make_share_url(
        {
            "main-tabs:active_tab": "supercharge",
            "sc-mc-enable:value": ["yes"],
            "sc-wd:value": 5000,
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
            timeout=25_000,
        )
        t_chart = (time.perf_counter() - t0) * 1000
        browser.close()
    assert t_chart < 25000, (
        f"MC fallback supercharge took {t_chart:.0f}ms (>25s — fallback broken)"
    )


def test_sc_no_phantom_rebuild():
    """Post-restore guard suppresses cascade rebuilds. MC off so the
    fast-path builder runs and the post-restore guard is exercised."""
    url = _make_share_url(
        {
            "main-tabs:active_tab": "supercharge",
            "sc-wd:value": 12000,
            "sc-stack:value": 1.0,
            "sc-model-show:value": ["bub"],
            "sc-mc-enable:value": [],
        },
        "/5",
    )
    with sync_playwright() as p:
        browser = p.firefox.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        page.wait_for_function(
            "() => { var gd = document.querySelector('#supercharge-graph .js-plotly-plot'); "
            "return gd && gd.data && gd.data.length > 0; }",
            timeout=20_000,
        )
        time.sleep(1.5)
        count = page.evaluate("() => window.__superchargeBuildCount || 0")
        browser.close()
    assert count == 1, (
        f"supercharge-build-count={count} after restore (expected 1). "
        f">=2 means the post-restore guard failed and the cascade "
        f"rebuilt the figure unnecessarily."
    )


def test_retire_share_still_restores():
    """Phase 2 ship 2 regression: /4 retire share still works fast."""
    url = _make_share_url(
        {
            "main-tabs:active_tab": "retire",
            "ret-wd:value": 8000,
        },
        "/4",
    )
    with sync_playwright() as p:
        browser = p.firefox.launch(headless=True)
        page = browser.new_page()
        t0 = time.perf_counter()
        page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        page.wait_for_function(
            "() => { var gd = document.querySelector('#retire-graph .js-plotly-plot'); "
            "return gd && gd.data && gd.data.length > 0; }",
            timeout=20_000,
        )
        t_chart = (time.perf_counter() - t0) * 1000
        browser.close()
    assert t_chart < 5000, f"/4 retire took {t_chart:.0f}ms (regression)"
