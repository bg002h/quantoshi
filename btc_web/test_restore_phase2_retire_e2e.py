"""Phase 2 ship 2 end-to-end Playwright tests for /4 (retire) fast modal close.

Mirrors test_restore_phase2_dca_e2e.py with retire-specific paths.

Requires:
  pip install playwright && python -m playwright install firefox
  Dev server must be running on :8050
Run:
  cd /scratch/code/bitcoinprojections && btc_venv/bin/python3 \\
      -m pytest btc_web/test_restore_phase2_retire_e2e.py -v --timeout=90 -n 0
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


def test_retire_share_fast_modal_close():
    """/4 share with ret-wd=8000 renders chart fast and restores widget."""
    url = _make_share_url(
        {
            "main-tabs:active_tab": "retire",
            "ret-wd:value": 8000,
            "ret-model-show:value": ["bub"],
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
        page.wait_for_function(
            "() => document.getElementById('ret-wd') && "
            "document.getElementById('ret-wd').value === '8000'",
            timeout=10_000,
        )
        wd = page.evaluate("() => document.getElementById('ret-wd').value")
        browser.close()
    assert wd == "8000", f"ret-wd={wd}, expected 8000"
    assert t_chart < 5000, (
        f"Retire chart took {t_chart:.0f}ms (expected <5000ms with fast path)"
    )


def test_retire_yr_range_restored():
    """/4 share with non-default yr-range — chart x-axis reflects slider."""
    url = _make_share_url(
        {
            "main-tabs:active_tab": "retire",
            "ret-yr-range:value": [2035, 2060],
            "ret-wd:value": 5000,
            "ret-model-show:value": ["bub"],
        },
        "/4",
    )
    with sync_playwright() as p:
        browser = p.firefox.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        page.wait_for_function(
            "() => { var gd = document.querySelector('#retire-graph .js-plotly-plot'); "
            "return gd && gd.data && gd.data.length > 0; }",
            timeout=20_000,
        )
        time.sleep(1)
        x_range = page.evaluate("""() => {
            var gd = document.querySelector('#retire-graph .js-plotly-plot');
            if (!gd || !gd.layout || !gd.layout.xaxis) return null;
            return gd.layout.xaxis.range;
        }""")
        browser.close()
    assert x_range is not None
    T_ORIGIN_YEAR = 2009.56
    def _to_calendar_year(v):
        if isinstance(v, (int, float)):
            return v + T_ORIGIN_YEAR
        return float(str(v)[:4])
    yr_lo = _to_calendar_year(x_range[0])
    yr_hi = _to_calendar_year(x_range[1])
    # Plotly auto-pads xaxis beyond data, so end-year may extend.
    # Start year is sharp; end year just needs to be >= snapshot's 2060.
    assert 2034.5 <= yr_lo <= 2035.5 and yr_hi >= 2059.5, (
        f"chart x-range = [{x_range[0]}, {x_range[1]}] (calendar years "
        f"{yr_lo:.1f}-{yr_hi:.1f}); expected calendar start ~2035, end ≥2060"
    )


def test_retire_mc_share_falls_back():
    """MC-enabled /4 share — modal eventually closes (cascade path)."""
    url = _make_share_url(
        {
            "main-tabs:active_tab": "retire",
            "ret-mc-enable:value": ["yes"],
            "ret-wd:value": 5000,
            "ret-model-show:value": ["bub"],
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
            timeout=25_000,
        )
        t_chart = (time.perf_counter() - t0) * 1000
        browser.close()
    assert t_chart < 25000, (
        f"MC fallback retire took {t_chart:.0f}ms (>25s — fallback broken)"
    )


def test_retire_no_phantom_rebuild():
    """Post-restore guard suppresses cascade rebuilds.

    retire-build-count Store + clientside increment on retire-graph.figure
    mutation. Single delivery via the relay = count==1. Cascade rebuild =
    count>=2 (guard failed).
    """
    url = _make_share_url(
        {
            "main-tabs:active_tab": "retire",
            "ret-wd:value": 8000,
            "ret-yr-range:value": [2035, 2060],
            "ret-model-show:value": ["bub"],
        },
        "/4",
    )
    with sync_playwright() as p:
        browser = p.firefox.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        page.wait_for_function(
            "() => { var gd = document.querySelector('#retire-graph .js-plotly-plot'); "
            "return gd && gd.data && gd.data.length > 0; }",
            timeout=20_000,
        )
        time.sleep(1.5)
        count = page.evaluate("() => window.__retireBuildCount || 0")
        browser.close()
    assert count == 1, (
        f"retire-build-count={count} after restore (expected 1). "
        f">=2 means the post-restore guard failed and the cascade "
        f"rebuilt the figure unnecessarily."
    )


def test_dca_share_still_restores():
    """Phase 2 ship 1 regression: /3 DCA share still works fast."""
    url = _make_share_url(
        {
            "main-tabs:active_tab": "dca",
            "dca-amount:value": 999,
            "dca-model-show:value": ["bub"],
        },
        "/3",
    )
    with sync_playwright() as p:
        browser = p.firefox.launch(headless=True)
        page = browser.new_page()
        t0 = time.perf_counter()
        page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        page.wait_for_function(
            "() => { var gd = document.querySelector('#dca-graph .js-plotly-plot'); "
            "return gd && gd.data && gd.data.length > 0; }",
            timeout=20_000,
        )
        t_chart = (time.perf_counter() - t0) * 1000
        browser.close()
    assert t_chart < 5000, f"/3 DCA took {t_chart:.0f}ms (regression)"
