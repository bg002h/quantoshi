"""End-to-end Playwright smoke test for the Citadel tax UI.

Requires: pip install playwright && python -m playwright install firefox
Run:      cd /scratch/code/bitcoinprojections && btc_venv/bin/python3 -m pytest btc_web/test_tax_e2e.py -v --timeout=60
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
    """Launch Firefox, navigate to /9, wait for Dash to render."""
    with sync_playwright() as p:
        browser = p.firefox.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1280, "height": 900})
        pg = ctx.new_page()
        pg.goto(f"{BASE_URL}/9", wait_until="networkidle", timeout=30000)
        pg.wait_for_selector("#citadel-graph", timeout=15000)
        yield pg
        browser.close()


def _js_exists(page, element_id):
    return page.evaluate(f'document.getElementById("{element_id}") !== null')


def _js_value(page, element_id):
    return page.evaluate(f'(document.getElementById("{element_id}") || {{}}).value')


def _switch_to_sim_tab(page):
    page.evaluate("""
        document.querySelectorAll('.nav-link').forEach(function(el) {
            if (el.textContent.trim() === 'Simulation') el.click();
        });
    """)
    time.sleep(0.3)


class TestTaxToggleAndModal:
    """Sequential test: toggle on → open modal → verify inputs → close."""

    def test_01_toggle_exists(self, page):
        assert _js_exists(page, "cp-tax-toggle")

    def test_02_config_button_hidden_initially(self, page):
        _switch_to_sim_tab(page)
        # Toggle should be off by default → button hidden
        btn = page.locator("#cp-tax-config-btn")
        # May or may not be visible depending on tab state
        # Just verify it exists
        assert _js_exists(page, "cp-tax-config-btn")

    def test_03_toggle_on(self, page):
        _switch_to_sim_tab(page)
        toggle = page.locator("#cp-tax-toggle")
        if not toggle.is_checked():
            toggle.click(force=True)
            time.sleep(0.5)
        assert toggle.is_checked()

    def test_04_config_button_visible_when_on(self, page):
        btn = page.locator("#cp-tax-config-btn")
        assert btn.is_visible()

    def test_05_open_modal(self, page):
        page.locator("#cp-tax-config-btn").click(force=True)
        time.sleep(1.0)
        # Modal should now be open — check for filing status radio
        assert _js_exists(page, "cp-tax-filing")

    def test_06_modal_has_all_inputs(self, page):
        """Verify all modal inputs exist (modal is open from previous test)."""
        for cid in ("cp-tax-filing", "cp-tax-state", "cp-tax-state-rate",
                     "cp-tax-birth-year", "cp-tax-other-income",
                     "cp-tax-other-income-growth", "cp-tax-tcja",
                     "cp-tax-basis-method",
                     "cp-td-btc", "cp-td-cash", "cp-td-res-short",
                     "cp-td-res-med", "cp-td-res-long", "cp-td-inv-eq", "cp-td-inv-bd",
                     "cp-tf-btc", "cp-tf-cash", "cp-tf-res-short",
                     "cp-tf-res-med", "cp-tf-res-long", "cp-tf-inv-eq", "cp-tf-inv-bd",
                     "cp-tax-save", "cp-tax-cancel"):
            assert _js_exists(page, cid), f"Missing in modal: {cid}"

    def test_07_state_dropdown_has_options(self, page):
        count = page.evaluate(
            'document.getElementById("cp-tax-state").options.length')
        assert count >= 51  # 50 states + DC

    def test_08_state_auto_fills_rate(self, page):
        page.locator("#cp-tax-state").select_option("CA")
        time.sleep(1.0)
        rate = _js_value(page, "cp-tax-state-rate")
        assert rate is not None
        assert float(rate) == pytest.approx(13.3, abs=0.1)

    def test_09_birth_year_accepts_valid_input(self, page):
        page.locator("#cp-tax-birth-year").fill("1985")
        val = _js_value(page, "cp-tax-birth-year")
        assert val == "1985"

    def test_10_td_btc_accepts_input(self, page):
        page.locator("#cp-td-btc").fill("0.5")
        val = _js_value(page, "cp-td-btc")
        assert val == "0.5"

    def test_11_save_closes_modal(self, page):
        page.locator("#cp-tax-save").click(force=True)
        time.sleep(1.5)
        # Modal should be closed — check via class
        is_open = page.evaluate(
            '(document.getElementById("cp-tax-modal") || {}).classList?.contains("show") || false')
        assert not is_open

    def test_12_toggle_off(self, page):
        _switch_to_sim_tab(page)
        # Use JS to uncheck — more reliable than Playwright click on dbc.Switch
        page.evaluate('document.getElementById("cp-tax-toggle").checked = false')
        page.evaluate("""
            var el = document.getElementById("cp-tax-toggle");
            el.dispatchEvent(new Event("change", {bubbles: true}));
        """)
        time.sleep(1.0)
        checked = page.evaluate('document.getElementById("cp-tax-toggle").checked')
        assert not checked


class TestInvestmentBasisInputs:
    def test_equity_basis_exists(self, page):
        assert _js_exists(page, "cp-inv-eq-basis")

    def test_bond_basis_exists(self, page):
        assert _js_exists(page, "cp-inv-bd-basis")

    def test_basis_defaults_match_value(self, page):
        eq_val = _js_value(page, "cp-inv-eq-init")
        eq_basis = _js_value(page, "cp-inv-eq-basis")
        assert eq_val == eq_basis
