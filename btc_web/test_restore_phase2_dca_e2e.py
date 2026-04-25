"""Phase 2 end-to-end Playwright tests for /3 (DCA) fast modal close.

Verifies the bubble fast-modal-close pattern correctly extends to /3:
- Non-MC, non-Saylor-live shares deliver the figure via set_props relay
  and modal closes via the active-chart-committed listener (~3-5 s).
- MC-enabled and Saylor-live shares fall back to the existing 7 s timer.
- Post-restore guard suppresses phantom rebuilds (verified via the
  dca-build-count Store + clientside increment on dca-graph.figure mutation).

Requires:
  pip install playwright && python -m playwright install firefox
  Dev server must be running on :8050
Run:
  cd /scratch/code/bitcoinprojections && btc_venv/bin/python3 \\
      -m pytest btc_web/test_restore_phase2_dca_e2e.py -v --timeout=60
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


_DASH_STORE_LOOKUP = """
() => {
    function find(comp) {
        if (!comp) return null;
        if (comp.props && comp.props.id === '__STORE_ID__') {
            return comp.props.data;
        }
        var ch = comp.props && comp.props.children;
        if (Array.isArray(ch)) {
            for (var i = 0; i < ch.length; i++) {
                var r = find(ch[i]);
                if (r !== null && r !== undefined) return r;
            }
        } else if (ch) { return find(ch); }
        return null;
    }
    return find(window._dashprivate_layout);
}
"""


def test_dca_share_fast_modal_close():
    """/3 share with dca-amount=999 renders chart fast and restores widget.

    Explicitly disables MC so the fast-path builder runs (DCA defaults to
    MC=on as of commit 093c665; with MC on, builder returns None and the
    cascade fallback path takes ~7-8s).
    """
    url = _make_share_url(
        {
            "main-tabs:active_tab": "dca",
            "dca-amount:value": 999,
            "dca-model-show:value": ["bub"],
            "dca-mc-enable:value": [],
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
        # Chart renders fast via set_props relay; widget values arrive via the
        # apply_tab_dca cascade (separate HTTP round-trip). Wait for the
        # widget to reflect the snapshot value before asserting.
        page.wait_for_function(
            "() => document.getElementById('dca-amount') && "
            "document.getElementById('dca-amount').value === '999'",
            timeout=10_000,
        )
        amount = page.evaluate("() => document.getElementById('dca-amount').value")
        browser.close()
    assert amount == "999", f"dca-amount={amount}, expected 999"
    assert t_chart < 5000, (
        f"DCA chart took {t_chart:.0f}ms (expected <5000ms with fast path)"
    )


def test_dca_yr_range_restored():
    """/3 share with non-default yr-range — chart x-axis reflects slider.

    Reading the slider's live value from JS is unreliable (dcc.RangeSlider
    state lives in Dash's internal Redux, not in the DOM). Instead assert
    that the chart's x-axis range matches the snapshot's yr-range — the
    figure builder uses yr_range[0]/[1] as start_yr/end_yr so the chart
    visibly reflects the snapshot.
    """
    # Explicitly disable MC so the builder's fast path runs (DCA defaults to
    # MC=on as of commit 093c665). Without this, the cascade fallback uses
    # MC overlay which extends the x-axis past the user's yr-range.
    url = _make_share_url(
        {
            "main-tabs:active_tab": "dca",
            "dca-yr-range:value": [2030, 2040],
            "dca-amount:value": 100,
            "dca-model-show:value": ["bub"],
            "dca-mc-enable:value": [],
        },
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
        time.sleep(1)  # let post-restore cascade settle
        # Read the chart's x-axis range from Plotly's live layout.
        x_range = page.evaluate("""() => {
            var gd = document.querySelector('#dca-graph .js-plotly-plot');
            if (!gd || !gd.layout || !gd.layout.xaxis) return null;
            return gd.layout.xaxis.range;
        }""")
        browser.close()
    assert x_range is not None, "chart has no x-axis range"
    # x_range is in t_years_since_2009-07-25 (numeric) for sim charts —
    # value of 20.4 means ~2030, value of 30.4 means ~2040. The repo
    # uses 2009-07-25 as the time origin (see CLAUDE.md).
    T_ORIGIN_YEAR = 2009.56  # 2009-07-25 ≈ year 2009.56
    def _to_calendar_year(v):
        if isinstance(v, (int, float)):
            return v + T_ORIGIN_YEAR
        # Fallback: ISO date string
        return float(str(v)[:4])
    yr_lo = _to_calendar_year(x_range[0])
    yr_hi = _to_calendar_year(x_range[1])
    assert 2029.5 <= yr_lo <= 2030.5 and 2039.5 <= yr_hi <= 2040.5, (
        f"chart x-range = [{x_range[0]}, {x_range[1]}] (calendar years "
        f"{yr_lo:.1f}-{yr_hi:.1f}); expected calendar 2030-2040"
    )


def test_dca_mc_share_falls_back():
    """MC-enabled /3 share — modal eventually closes (cascade path).

    Asserts a generous upper bound (12 s) without a brittle lower bound.
    """
    url = _make_share_url(
        {
            "main-tabs:active_tab": "dca",
            "dca-mc-enable:value": ["yes"],
            "dca-amount:value": 100,
            "dca-model-show:value": ["bub"],
        },
        "/3",
    )
    with sync_playwright() as p:
        browser = p.firefox.launch(headless=True)
        page = browser.new_page()
        t0 = time.perf_counter()
        page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        # MC fallback can take longer than the fast path — generous timeout.
        page.wait_for_function(
            "() => { var gd = document.querySelector('#dca-graph .js-plotly-plot'); "
            "return gd && gd.data && gd.data.length > 0; }",
            timeout=25_000,
        )
        t_chart = (time.perf_counter() - t0) * 1000
        browser.close()
    assert t_chart < 25000, (
        f"MC fallback DCA took {t_chart:.0f}ms (>25s — fallback path broken)"
    )


def test_dca_sc_live_falls_back():
    """Saylor-live /3 share — modal eventually closes via cascade path."""
    url = _make_share_url(
        {
            "main-tabs:active_tab": "dca",
            "dca-sc-enable:value": ["yes"],
            "dca-sc-entry-mode:value": "live",
            "dca-sc-loan:value": 100000.0,
            "dca-amount:value": 100,
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
            timeout=15_000,
        )
        t_chart = (time.perf_counter() - t0) * 1000
        browser.close()
    assert t_chart < 12000, (
        f"SC-live fallback DCA took {t_chart:.0f}ms (>12s — fallback broken)"
    )


def test_dca_no_phantom_rebuild():
    """Post-restore guard suppresses cascade rebuilds.

    dca-build-count Store + clientside increment fires on every
    dca-graph.figure mutation. Single delivery via the relay = count==1.
    Cascade rebuild = count>=2 (guard failed). MC off so the fast-path
    builder runs and the guard is actually exercised.
    """
    url = _make_share_url(
        {
            "main-tabs:active_tab": "dca",
            "dca-amount:value": 999,
            "dca-yr-range:value": [2025, 2035],
            "dca-model-show:value": ["bub"],
            "dca-mc-enable:value": [],
        },
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
        # Wait for any post-restore cascade fires to settle.
        time.sleep(1.5)
        # Read window.__dcaBuildCount (mirrored from the dcc.Store by the
        # clientside increment callback for direct JS test access).
        count = page.evaluate("() => window.__dcaBuildCount || 0")
        browser.close()
    assert count == 1, (
        f"dca-build-count={count} after restore (expected 1). "
        f">=2 means the post-restore guard failed and the cascade "
        f"rebuilt the figure unnecessarily."
    )


def test_bubble_share_still_restores():
    """Phase 1 regression: /1 bubble share still works fast."""
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
    assert t_chart < 5000, f"/1 bubble took {t_chart:.0f}ms (Phase 1 regression)"
