"""End-to-end Playwright smoke test for the Plot Appearance control plane.

Covers the critical symptoms that motivated the JS/DOM redesign:
  - Reset button repopulates control panel values on every tab
  - Slider changes propagate across tabs
  - Changes survive page reload (localStorage)
  - No callback overload cycle

Requires: pip install playwright && python -m playwright install firefox
Run:      cd /scratch/code/bitcoinprojections && btc_venv/bin/python3 \
              -m pytest btc_web/test_plot_appearance_e2e.py -v --timeout=60
          (dev server must be running on :8050)
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


@pytest.fixture(scope="module")
def page():
    """Launch Firefox, navigate to /1 (bubble), wait for layout to render."""
    with sync_playwright() as p:
        browser = p.firefox.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1280, "height": 900})
        pg = ctx.new_page()
        pg.goto(f"{BASE_URL}/1", wait_until="networkidle", timeout=30000)
        pg.wait_for_selector("#bub-plot-trace-width", state="attached", timeout=15000)
        # Give the JS IIFE one setInterval tick to bootstrap + one to settle.
        time.sleep(1.2)
        yield pg
        browser.close()


def _value(page, eid: str):
    return page.evaluate(f'(document.getElementById("{eid}") || {{}}).value')


def _set_localstorage(page, key: str, obj):
    import json as _json
    page.evaluate(f'localStorage.setItem("{key}", {_json.dumps(_json.dumps(obj))})')


def _get_localstorage(page, key: str):
    return page.evaluate(f'localStorage.getItem("{key}")')


def _click(page, eid: str):
    page.evaluate(f'document.getElementById("{eid}").click()')


def _dispatch_input(page, eid: str, value):
    """Set an input value and fire both 'input' and 'change' so listeners run."""
    page.evaluate(f"""
        (() => {{
            var el = document.getElementById("{eid}");
            if (!el) return;
            el.value = {value!r};
            el.dispatchEvent(new Event('input', {{bubbles: true}}));
            el.dispatchEvent(new Event('change', {{bubbles: true}}));
        }})()
    """)
    time.sleep(0.7)  # let the 500ms poll tick apply across peer tabs


def _switch_tab(page, tab_path: str):
    """Navigate by URL instead of clicking (Dash routes via url.pathname)."""
    page.evaluate(f'window.history.pushState(null, "", "{tab_path}")')
    # Force Dash router by dispatching popstate
    page.evaluate('window.dispatchEvent(new Event("popstate"))')
    time.sleep(0.8)


# ---------------------------------------------------------------------------
# Tests run as a sequence — localStorage state persists across tests in module.
# ---------------------------------------------------------------------------


class TestPlotAppearancePanel:

    def test_01_bootstrap_paints_defaults(self, page):
        """JS IIFE should read localStorage (or DEFAULTS) and paint all 4
        always-rendered tabs' controls on first tick."""
        # Clear any stale localStorage from prior runs.
        page.evaluate('localStorage.removeItem("plot-appearance")')
        page.reload(wait_until="networkidle")
        page.wait_for_selector("#bub-plot-trace-width", state="attached", timeout=15000)
        time.sleep(1.0)  # bootstrap + one tick
        assert _value(page, "bub-plot-trace-width") == "2.5"
        assert _value(page, "dca-plot-trace-width") == "2.5"
        assert _value(page, "ret-plot-trace-width") == "2.5"
        assert _value(page, "sc-plot-trace-width") == "2.5"

    def test_02_change_on_bub_propagates_to_peers(self, page):
        _dispatch_input(page, "bub-plot-trace-width", "5")
        assert _value(page, "bub-plot-trace-width") == "5"
        assert _value(page, "dca-plot-trace-width") == "5"
        assert _value(page, "ret-plot-trace-width") == "5"
        assert _value(page, "sc-plot-trace-width") == "5"

    def test_03_localstorage_updated(self, page):
        import json as _json
        raw = _get_localstorage(page, "plot-appearance")
        assert raw is not None
        s = _json.loads(raw)
        assert s["trace_width"] == 5

    def test_04_change_on_dca_propagates_back_to_bub(self, page):
        _dispatch_input(page, "dca-plot-grid-major-width", "2")
        assert _value(page, "dca-plot-grid-major-width") == "2"
        assert _value(page, "bub-plot-grid-major-width") == "2"

    def test_05_citadel_lazy_load_picks_up_state(self, page):
        """Navigate to /6 for the first time; after ≤500ms the cp-* controls
        should exist and show the current non-default values."""
        _switch_tab(page, "/6")
        page.wait_for_selector("#cp-plot-trace-width", state="attached", timeout=15000)
        # Lazy-mount + JS poll tick (500ms) + applyStateToDOM. 1.5s
        # gives enough headroom under pytest's slower scheduling.
        time.sleep(1.5)
        assert _value(page, "cp-plot-trace-width") == "5"
        assert _value(page, "cp-plot-grid-major-width") == "2"

    def test_06_reset_on_citadel_resets_all_tabs(self, page):
        # Navigate to /1 and type pt_size/pt_alpha via real keyboard so
        # Dash's React state for bub-ptsize/bub-ptalpha actually updates.
        _switch_tab(page, "/1")
        page.wait_for_selector("#bub-ptsize", state="attached")
        time.sleep(0.5)
        page.focus("#bub-ptsize")
        page.evaluate('document.getElementById("bub-ptsize").select()')
        page.keyboard.type("17")
        page.keyboard.press("Tab")
        page.focus("#bub-ptalpha")
        page.evaluate('document.getElementById("bub-ptalpha").select()')
        page.keyboard.type("0.9")
        page.keyboard.press("Tab")
        time.sleep(0.5)
        assert _value(page, "bub-ptsize") == "17"
        assert _value(page, "bub-ptalpha") == "0.9"

        # Navigate to Citadel and wait for the cp reset button to mount
        # in the lazy-loaded citadel subtree.
        _switch_tab(page, "/6")
        page.wait_for_selector("#cp-plot-appearance-reset", state="attached")
        page.evaluate('document.querySelectorAll(".drawer-collapsed").forEach(el => el.classList.remove("drawer-collapsed"));')
        time.sleep(0.4)
        # page.dispatch_event bypasses visibility/actionability checks and
        # dispatches a real native event that Dash recognizes (unlike
        # document.getElementById(...).click() which Dash ignores).
        page.dispatch_event("#cp-plot-appearance-reset", "click")
        time.sleep(1.5)  # allow Dash round-trip for bub-ptsize/bub-ptalpha
        # All 5 tabs' 6 JS-managed fields should show defaults.
        for prefix in ("bub", "dca", "ret", "sc", "cp"):
            assert _value(page, f"{prefix}-plot-trace-width") == "2.5", prefix
            assert _value(page, f"{prefix}-plot-grid-major-width") == "1", prefix
            # Colors are lowercased on write; defaults match.
            assert _value(page, f"{prefix}-plot-grid-major-color").lower() == "#888888"
            assert _value(page, f"{prefix}-plot-grid-minor-color").lower() == "#b0b0b0"
            assert _value(page, f"{prefix}-plot-pt-color").lower() == "#2c3e50"

        # The kept Dash callback must have fired too — pt_size and pt_alpha
        # are server-rendered bubble controls, not JS-managed. This catches
        # the cloneNode-broke-React-fiber regression.
        assert _value(page, "bub-ptsize") == "10", "Dash reset callback did not fire for pt_size"
        assert _value(page, "bub-ptalpha") == "0.5", "Dash reset callback did not fire for pt_alpha"

    def test_07_localstorage_reset_to_defaults(self, page):
        import json as _json
        raw = _get_localstorage(page, "plot-appearance")
        s = _json.loads(raw)
        assert s["trace_width"] == 2.5
        assert s["grid_major_width"] == 1.0
        assert s["grid_major_color"] == "#888888"
        assert s["grid_minor_color"] == "#b0b0b0"
        assert s["pt_color"] == "#2c3e50"

    def test_08_change_persists_across_reload(self, page):
        _dispatch_input(page, "bub-plot-trace-width", "4")
        # Verify localStorage actually updated before reloading.
        import json as _json
        raw = _get_localstorage(page, "plot-appearance")
        assert raw is not None, "localStorage was not written by _dispatch_input"
        assert _json.loads(raw)["trace_width"] == 4
        page.reload(wait_until="networkidle")
        page.wait_for_selector("#bub-plot-trace-width", state="attached", timeout=15000)
        # JS IIFE bootstraps synchronously, then setInterval 500ms polls.
        # Allow a generous window for React hydration + first apply.
        time.sleep(2.0)
        assert _value(page, "bub-plot-trace-width") == "4"

    def test_09_cleared_number_input_falls_back_to_default(self, page):
        _dispatch_input(page, "dca-plot-trace-width", "")
        # JS handler should parseFloat("") → NaN → DEFAULTS.trace_width = 2.5
        assert _value(page, "dca-plot-trace-width") == "2.5"

    def test_10_reset_button_clicks_do_not_spam_callbacks(self, page):
        """Click reset 5 times in rapid succession; JS should remain responsive
        and localStorage should equal defaults exactly once at the end."""
        for _ in range(5):
            _click(page, "bub-plot-appearance-reset")
            time.sleep(0.08)
        time.sleep(0.8)
        import json as _json
        raw = _get_localstorage(page, "plot-appearance")
        s = _json.loads(raw)
        assert s["trace_width"] == 2.5
        # Panel still responsive.
        _dispatch_input(page, "ret-plot-trace-width", "3")
        assert _value(page, "ret-plot-trace-width") == "3"
        assert _value(page, "bub-plot-trace-width") == "3"
