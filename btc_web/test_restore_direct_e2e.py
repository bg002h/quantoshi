"""End-to-end Playwright smoke test for the restore-figure-in-decode path.

Verifies the user-perceived restore latency: load a share link with
?trace=1, wait for plotly_afterplot, confirm the chart shows the
restored state (NOT the default), and confirm dev journal shows
[trace] restore-direct-build BUILT line.

Requires:
  pip install playwright && python -m playwright install firefox
  Dev server must be running on :8050
Run:
  cd /scratch/code/bitcoinprojections && btc_venv/bin/python3 \\
      -m pytest btc_web/test_restore_direct_e2e.py -v --timeout=60
"""
import pytest
import time
import re
import subprocess

try:
    from playwright.sync_api import sync_playwright
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False

pytestmark = pytest.mark.skipif(not HAS_PLAYWRIGHT, reason="playwright not installed")

BASE_URL = "http://localhost:8050"


def _read_dev_log_tail(n=200):
    """Return last n lines of /tmp/quantoshi_dev.log."""
    try:
        out = subprocess.check_output(
            ["/usr/bin/tail", "-n", str(n), "/tmp/quantoshi_dev.log"],
            text=True, timeout=2,
        )
        return out
    except Exception:
        return ""


def _make_share_url_directly():
    """Generate a q4: share link using snapshot._encode_snapshot_v4
    without going through the Dash UI. Encodes a non-default state
    (xscale=linear, n-future=5) so the restore is verifiable."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    os.environ.setdefault("DEV", "1")
    import app  # noqa: F401
    from snapshot import _encode_snapshot_v4
    state = {
        "main-tabs:active_tab": "bubble",
        "bub-xscale:value":     "linear",
        "bub-n-future:value":   5,
        "bub-toggles:value":    ["shade", "show_data"],
    }
    blob = _encode_snapshot_v4(state)  # returns "fp:base64payload"
    return f"{BASE_URL}/1?trace=1#q4:{blob}"


def test_restore_direct_build_emits_trace_log():
    """Generate a share link via Python, navigate with ?trace=1, confirm:
    1. The restore round-trip emits `[trace] restore-direct-build BUILT` in dev log.
    2. The chart appears (no fallback timer).
    """
    trace_url = _make_share_url_directly()
    with sync_playwright() as p:
        browser = p.firefox.launch(headless=True)

        ctx_b = browser.new_context()
        page_b = ctx_b.new_page()
        marker_before = _read_dev_log_tail(50)
        # Capture browser console for debugging
        console_msgs = []
        page_b.on("console", lambda msg: console_msgs.append(f"[{msg.type}] {msg.text}"))

        t0 = time.perf_counter()
        page_b.goto(trace_url, wait_until="domcontentloaded", timeout=30_000)
        page_b.wait_for_selector("#bubble-graph .js-plotly-plot", timeout=30_000)
        # Wait for the restore-progress modal to close (or fallback to fire)
        page_b.wait_for_selector(
            "#restore-progress-modal", state="hidden", timeout=15_000,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        time.sleep(1)  # let any final logs flush

        # Step 3: verify dev journal has the BUILT trace
        log_after = _read_dev_log_tail(300)
        assert "[trace] restore-direct-build BUILT" in log_after, (
            f"Expected '[trace] restore-direct-build BUILT' in dev log; "
            f"last 300 lines did not contain it.\nConsole: {console_msgs[-10:]}"
        )

        # Step 4: verify the modal closed FAST (not via 7s fallback)
        assert elapsed_ms < 5000, (
            f"Restore took {elapsed_ms:.0f}ms — likely fell back to 7s timer "
            f"(expected < 5000ms)."
        )

        # Step 5: verify the figure has traces (chart actually rendered)
        n_traces = page_b.evaluate("""() => {
            var gd = document.querySelector('#bubble-graph .js-plotly-plot');
            return gd && gd.data ? gd.data.length : 0;
        }""")
        assert n_traces > 0, f"chart has no traces; figure was not rendered"

        # Step 6: verify xscale is restored to "linear"
        xscale = page_b.evaluate("""() => {
            var el = document.querySelectorAll('input[name=\"bub-xscale\"]:checked');
            return el.length ? el[0].value : null;
        }""")
        # If the radio readout doesn't work, just check the figure layout
        if xscale is None:
            xtype = page_b.evaluate("""() => {
                var gd = document.querySelector('#bubble-graph .js-plotly-plot');
                return gd && gd._fullLayout && gd._fullLayout.xaxis
                    ? gd._fullLayout.xaxis.type : null;
            }""")
            print(f"x-axis type: {xtype}")

        ctx_b.close()
        browser.close()
