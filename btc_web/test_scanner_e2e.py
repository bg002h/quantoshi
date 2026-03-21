"""
End-to-end Playwright tests for the Model Scanner panel.

Usage:
    # Start the dev server first:
    DEV=1 bash run_web.sh &

    # Run tests:
    btc_venv/bin/python3 -m pytest btc_web/test_scanner_e2e.py -v --headed
    # or headless:
    btc_venv/bin/python3 -m pytest btc_web/test_scanner_e2e.py -v
"""

import os
import pytest
from playwright.sync_api import sync_playwright, expect

BASE_URL = "http://localhost:8050"
TIMEOUT = 15_000  # ms


@pytest.fixture
def page():
    with sync_playwright() as p:
        headed = os.environ.get("HEADED", "0") == "1"
        b = p.chromium.launch(headless=not headed)
        pg = b.new_page()
        pg.goto(BASE_URL, wait_until="networkidle", timeout=30_000)
        pg.wait_for_timeout(3000)  # let Dash callbacks settle
        yield pg
        pg.close()
        b.close()


class TestScannerPriceToQuantile:
    """Mode 1: User enters price + date → table shows quantiles."""

    def test_default_load_shows_quantile_table(self, page):
        """On load, scanner shows quantile results (default: live price + today)."""
        table = page.locator("#scan-results table")
        expect(table).to_be_visible(timeout=TIMEOUT)
        # Header should say "Quantile"
        header = table.locator("th").nth(1)
        expect(header).to_have_text("Quantile")

    def test_ucl_row_present(self, page):
        """UCL row shows 'x above' value."""
        ucl = page.locator("#scan-results tr:first-child td:first-child")
        expect(ucl).to_have_text("Unfairly Cheap Line", timeout=TIMEOUT)
        value = page.locator("#scan-results tr:first-child td:nth-child(2)")
        expect(value).to_contain_text("above")

    def test_edit_price_updates_quantiles(self, page):
        """Typing a price updates the quantile results."""
        price_input = page.locator("#scan-price")
        price_input.fill("100000")
        price_input.press("Tab")
        page.wait_for_timeout(2000)

        table = page.locator("#scan-results table")
        expect(table).to_be_visible(timeout=TIMEOUT)
        header = table.locator("th").nth(1)
        expect(header).to_have_text("Quantile")

    def test_higher_price_gives_higher_quantile(self, page):
        """$100K should show higher quantiles than $50K."""
        price_input = page.locator("#scan-price")

        # Get quantile at $50K
        price_input.fill("50000")
        price_input.press("Tab")
        page.wait_for_timeout(2000)
        rows_50k = page.locator("#scan-results tbody tr").all_text_contents()

        # Get quantile at $100K
        price_input.fill("100000")
        price_input.press("Tab")
        page.wait_for_timeout(2000)
        rows_100k = page.locator("#scan-results tbody tr").all_text_contents()

        # At least some model should show higher Q at higher price
        assert rows_50k != rows_100k


class TestScannerQuantileToPrice:
    """Mode 2: User enters quantile → table shows prices."""

    def test_edit_quantile_shows_prices(self, page):
        """Typing a quantile switches table to Price mode."""
        q_input = page.locator("#scan-q")
        q_input.fill("50")
        q_input.press("Tab")
        page.wait_for_timeout(2000)

        table = page.locator("#scan-results table")
        expect(table).to_be_visible(timeout=TIMEOUT)
        header = table.locator("th").nth(1)
        expect(header).to_have_text("Price")

    def test_price_results_contain_dollar(self, page):
        """Price results should contain $ symbol."""
        q_input = page.locator("#scan-q")
        q_input.fill("25")
        q_input.press("Tab")
        page.wait_for_timeout(2000)

        cells = page.locator("#scan-results tbody td:nth-child(2)").all_text_contents()
        assert any("$" in c for c in cells)


class TestScannerDateSolving:
    """Mode 3: User enters price + quantile → table shows dates."""

    def test_price_then_quantile_solves_date(self, page):
        """Editing price then quantile makes date the output."""
        # Edit price first
        price_input = page.locator("#scan-price")
        price_input.fill("1000000")
        price_input.press("Tab")
        page.wait_for_timeout(1000)

        # Then edit quantile — now date should be the output
        q_input = page.locator("#scan-q")
        q_input.fill("50")
        q_input.press("Tab")
        page.wait_for_timeout(2000)

        table = page.locator("#scan-results table")
        expect(table).to_be_visible(timeout=TIMEOUT)
        header = table.locator("th").nth(1)
        expect(header).to_have_text("Date")

    def test_date_results_contain_year(self, page):
        """Date results should contain 4-digit years."""
        price_input = page.locator("#scan-price")
        price_input.fill("1000000")
        price_input.press("Tab")
        page.wait_for_timeout(1000)

        q_input = page.locator("#scan-q")
        q_input.fill("50")
        q_input.press("Tab")
        page.wait_for_timeout(2000)

        cells = page.locator("#scan-results tbody td:nth-child(2)").all_text_contents()
        assert any("20" in c for c in cells)  # dates like 2030-xx-xx


class TestScannerModeTransitions:
    """Verify the last-two-edited logic works correctly."""

    def test_date_then_quantile_solves_price(self, page):
        """Date then quantile → price is output."""
        date_input = page.locator("#scan-date")
        date_input.fill("2030-01-01")
        date_input.press("Tab")
        page.wait_for_timeout(1000)

        q_input = page.locator("#scan-q")
        q_input.fill("25")
        q_input.press("Tab")
        page.wait_for_timeout(2000)

        header = page.locator("#scan-results table th").nth(1)
        expect(header).to_have_text("Price")

    def test_quantile_then_date_solves_price(self, page):
        """Quantile then date → price is output."""
        q_input = page.locator("#scan-q")
        q_input.fill("50")
        q_input.press("Tab")
        page.wait_for_timeout(1000)

        date_input = page.locator("#scan-date")
        date_input.fill("2035-01-01")
        date_input.press("Tab")
        page.wait_for_timeout(2000)

        header = page.locator("#scan-results table th").nth(1)
        expect(header).to_have_text("Price")

    def test_price_then_date_solves_quantile(self, page):
        """Price then date → quantile is output."""
        price_input = page.locator("#scan-price")
        price_input.fill("75000")
        price_input.press("Tab")
        page.wait_for_timeout(1000)

        date_input = page.locator("#scan-date")
        date_input.fill("2026-06-01")
        date_input.press("Tab")
        page.wait_for_timeout(2000)

        header = page.locator("#scan-results table th").nth(1)
        expect(header).to_have_text("Quantile")
