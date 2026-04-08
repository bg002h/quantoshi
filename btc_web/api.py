"""Flask API routes for BTCPay payment-gated MC simulations and documentation.

Routes:
    POST /api/mc/invoice          — create a BTCPay invoice
    GET  /api/mc/invoice/<id>     — check invoice status
    POST /api/mc/verify           — verify a payment token
    GET  /docs/architecture       — architecture guide (HTML)
    GET  /docs/user-manual        — user manual (HTML)

Registered on the Flask server via register_routes(server).
"""

import os
import re
import time
import logging
import markdown as _md
from collections import defaultdict
from flask import jsonify, request

import btcpay

_INVOICE_ID_RE = re.compile(r'^[a-zA-Z0-9_-]{1,64}$')

log = logging.getLogger(__name__)

# ── Rate limiting ────────────────────────────────────────────────────────────
# Per-IP counters with hourly reset.
# Two limits: 20 outstanding unpaid invoices/hr, 100 paid invoices/hr.

_WINDOW = 3600  # 1 hour
_MAX_UNPAID = 20                # max outstanding unpaid invoices per IP per window
_MAX_PAID_PER_HR = 100          # max paid invoices per IP per window
_ALLOWED_TABS = ("dca", "ret", "hm", "sc")
_ALLOWED_MC_YEARS = (10, 20, 30, 40)

# {ip: [(timestamp, paid_bool), ...]}
_invoice_log: dict[str, list] = defaultdict(list)


def _prune(ip: str) -> None:
    """Remove entries older than the rate-limit window."""
    cutoff = time.time() - _WINDOW
    _invoice_log[ip] = [(t, p) for t, p in _invoice_log[ip] if t > cutoff]


def _check_rate_limit(ip: str) -> str | None:
    """Return an error message if rate-limited, else None."""
    _prune(ip)
    entries = _invoice_log[ip]
    paid   = sum(1 for _, p in entries if p)
    unpaid = sum(1 for _, p in entries if not p)
    if unpaid >= _MAX_UNPAID:
        return "Too many unpaid invoices. Please pay existing invoices first."
    if paid >= _MAX_PAID_PER_HR:
        return "Hourly invoice limit reached. Please try again later."
    return None


def _record_invoice(ip: str) -> None:
    """Record a new invoice creation (initially unpaid)."""
    _invoice_log[ip].append((time.time(), False))


def _mark_paid(ip: str, invoice_id: str) -> None:
    """Mark the most recent unpaid entry as paid (approximate — just flips one)."""
    entries = _invoice_log[ip]
    for i in range(len(entries) - 1, -1, -1):
        if not entries[i][1]:
            entries[i] = (entries[i][0], True)
            break


# ── Route registration ───────────────────────────────────────────────────────

def _client_ip() -> str:
    """Real client IP — trusts nginx X-Real-IP, falls back to remote_addr."""
    return request.headers.get("X-Real-IP", request.remote_addr) or "unknown"


def register_routes(server) -> None:
    """Register MC payment API routes on the Flask server."""

    # ── Documentation routes ─────────────────────────────────────────────
    _DOC_DIR = os.path.join(os.path.dirname(__file__), "..", "docs")

    def _render_doc(filename, title):
        doc_path = os.path.join(_DOC_DIR, filename)
        try:
            with open(doc_path) as f:
                md_content = f.read()
        except FileNotFoundError:
            return "Document not found", 404
        html_body = _md.markdown(md_content, extensions=["tables", "fenced_code"])
        return f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} — Quantoshi</title>
<style>
body {{ background:#1a1a2e; color:#ddd; font-family:system-ui,sans-serif;
       max-width:900px; margin:0 auto; padding:24px 16px; line-height:1.6; }}
a {{ color:#00d4ff; }}
h1,h2,h3 {{ color:#00d4ff; }}
h4,h5,h6 {{ color:#8ecae6; }}
table {{ border-collapse:collapse; width:100%; margin:16px 0; }}
th,td {{ border:1px solid #444; padding:6px 10px; text-align:left; }}
th {{ background:#2a3a5e; color:#00d4ff; }}
code {{ background:#16213e; padding:2px 6px; border-radius:3px; font-size:0.9em; }}
pre {{ background:#16213e; padding:12px; border-radius:6px; overflow-x:auto; }}
pre code {{ background:none; padding:0; }}
.back {{ display:inline-block; margin-bottom:16px; color:#888; text-decoration:none; }}
.back:hover {{ color:#00d4ff; }}
</style>
</head><body>
<a class="back" href="/">\u2190 Back to Quantoshi</a>
<article>{html_body}</article>
</body></html>"""

    @server.route("/B")
    def _sensitivity_sweep():
        svg_path = os.path.join(os.path.dirname(__file__), "..", "sensitivity_sweep.svg")
        try:
            with open(svg_path) as f:
                return f.read(), 200, {"Content-Type": "image/svg+xml"}
        except FileNotFoundError:
            return "Not generated yet", 404

    @server.route("/BB")
    def _sensitivity_sweep_ef():
        svg_path = os.path.join(os.path.dirname(__file__), "..", "sensitivity_sweep_ef.svg")
        try:
            with open(svg_path) as f:
                return f.read(), 200, {"Content-Type": "image/svg+xml"}
        except FileNotFoundError:
            return "Not generated yet", 404

    @server.route("/D")
    def _residual_fft():
        svg_path = os.path.join(os.path.dirname(__file__), "..", "residual_fft.svg")
        try:
            with open(svg_path) as f:
                svg = f.read()
        except FileNotFoundError:
            return "Not generated yet", 404
        html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Residual FFT \u2014 Quantoshi</title>
<style>
body { background:#1a1a2e; color:#cccccc; font-family:system-ui,sans-serif;
       max-width:1300px; margin:0 auto; padding:24px 16px; line-height:1.5; }
h1, h2 { color:#00d4ff; }
h1 { font-size:22px; }
h2 { font-size:16px; margin-top:28px; }
a { color:#FF9F40; text-decoration:none; }
a:hover { color:#FFD080; text-decoration:underline; }
.muted { color:#888; font-size:12px; }
.desc { background:#101a2e; padding:10px 16px; border-radius:6px;
        border-left:3px solid #00d4ff; margin:8px 0 12px 0;
        font-size:13px; color:#b8ccd8; line-height:1.55; }
.desc strong { color:#00d4ff; }
table { border-collapse:collapse; margin:8px 0 12px 0; font-size:12px; }
th, td { border:1px solid #444; padding:4px 10px; text-align:right; }
th { background:#16213e; color:#00d4ff; text-align:center; }
td:first-child, td:nth-child(2) { font-weight:600; color:#FFD080; }
.formula { background:#0e1624; padding:10px 14px; border-radius:6px;
           border-left:3px solid #FF9F40; font-size:13px; margin:8px 0 12px 0;
           font-family: ui-monospace, monospace; }
img, svg { max-width:100%; height:auto; display:block; border-radius:6px; }
.back-link { display:inline-block; margin-top:24px; color:#888; }
</style>
</head><body>
<h1>Residual FFT power spectra</h1>
<p class="muted">
FFT of log-space residuals (log\u2081\u2080 price \u2212 model fit) across
BM floor, BM composite, LPPL\u2081, LPPL\u2082, LPPL\u2084. Sampled on a
uniform ln(t) grid, which puts angular frequency \u03c9 in log-time
units. Hann-windowed. Cap at \u03c9=100.
</p>
<h2>\u03c9 \u2192 calendar-year cycle conversion</h2>
<p class="desc">
\u03c9 is angular frequency in <strong>log-time</strong>, not calendar
time. The log-period T<sub>ln t</sub>=2\u03c0/\u03c9 is constant; the
corresponding calendar gap <em>stretches</em> as t grows. Each
successive cycle is longer than the previous by a fixed ratio
r = exp(T<sub>ln t</sub>) = exp(2\u03c0/\u03c9).
</p>
<div class="formula">
T<sub>ln t</sub> = 2\u03c0 / \u03c9 &nbsp;\u2192&nbsp;
ratio r = e<sup>T<sub>ln t</sub></sup>
</div>
<table>
<thead><tr>
<th>\u03c9</th><th>T<sub>ln t</sub></th><th>ratio r</th>
<th>gap at t=5yr</th><th>gap at t=10yr</th><th>gap at t=16yr</th><th>gap at t=30yr</th>
</tr></thead>
<tbody>
<tr><td>4.5</td><td>1.396</td><td>4.04\u00d7</td><td>15.2 yr</td><td>30.4 yr</td><td>48.6 yr</td><td>91.2 yr</td></tr>
<tr><td>6.7</td><td>0.938</td><td>2.55\u00d7</td><td>7.8 yr</td><td>15.6 yr</td><td>25.0 yr</td><td>46.9 yr</td></tr>
<tr><td>7.3</td><td>0.861</td><td>2.37\u00d7</td><td>6.8 yr</td><td>13.7 yr</td><td>21.9 yr</td><td>41.0 yr</td></tr>
<tr><td>8.9</td><td>0.706</td><td>2.03\u00d7</td><td>5.1 yr</td><td>10.3 yr</td><td>16.4 yr</td><td>30.7 yr</td></tr>
<tr><td>13.4</td><td>0.469</td><td>1.60\u00d7</td><td>3.0 yr</td><td>5.98 yr</td><td>9.57 yr</td><td>17.9 yr</td></tr>
<tr><td>17.9</td><td>0.351</td><td>1.42\u00d7</td><td>2.1 yr</td><td>4.20 yr</td><td>6.72 yr</td><td>12.6 yr</td></tr>
<tr><td>20.1</td><td>0.313</td><td>1.37\u00d7</td><td>1.8 yr</td><td>3.67 yr</td><td>5.87 yr</td><td>11.0 yr</td></tr>
<tr><td>26.8</td><td>0.234</td><td>1.26\u00d7</td><td>1.3 yr</td><td>2.64 yr</td><td>4.23 yr</td><td>7.92 yr</td></tr>
<tr><td>31.3</td><td>0.201</td><td>1.22\u00d7</td><td>1.1 yr</td><td>2.23 yr</td><td>3.57 yr</td><td>6.69 yr</td></tr>
<tr><td>37.9</td><td>0.166</td><td>1.18\u00d7</td><td>0.9 yr</td><td>1.81 yr</td><td>2.90 yr</td><td>5.43 yr</td></tr>
<tr><td>44.7</td><td>0.141</td><td>1.15\u00d7</td><td>0.76 yr</td><td>1.51 yr</td><td>2.42 yr</td><td>4.54 yr</td></tr>
<tr><td>67.0</td><td>0.094</td><td>1.098\u00d7</td><td>0.49 yr</td><td>0.98 yr</td><td>1.57 yr</td><td>2.94 yr</td></tr>
</tbody>
</table>
<p class="muted">
Calendar-year gap = t\u00b7(r\u22121). At t=10 yr, \u03c9=7.3 \u2192 13.7 yr
to next peak; at t=30 yr the same \u03c9 predicts 41 yr. Cycles stretch
with time \u2014 classic log-periodicity.
</p>
""" + svg + """
<a href="/" class="back-link">\u2190 Back to Quantoshi</a>
</body></html>"""
        return html, 200, {"Content-Type": "text/html"}

    @server.route("/G")
    def _wave_basis():
        svg_path = os.path.join(os.path.dirname(__file__), "..", "wave_basis_comparison.svg")
        try:
            with open(svg_path) as f:
                return f.read(), 200, {"Content-Type": "image/svg+xml"}
        except FileNotFoundError:
            return "Not generated yet — run tools/pca_basis_search.py", 404

    @server.route("/F")
    def _excess_fits():
        """Detrended LinPPL/HybPPL fits — oscillator-only on BM-excess."""
        html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LPPL family on excess \u2014 Quantoshi</title>
<style>
body { background:#1a1a2e; color:#cccccc; font-family:system-ui,sans-serif;
       max-width:1300px; margin:0 auto; padding:24px 16px; line-height:1.5; }
h1 { color:#00d4ff; font-size:22px; }
h2 { color:#00d4ff; margin-top:24px; }
a { color:#FF9F40; text-decoration:none; }
a:hover { color:#FFD080; text-decoration:underline; }
img { max-width:100%; height:auto; display:block; border-radius:6px;
       margin-top:12px; }
.muted { color:#888; font-size:12px; }
.desc { background:#101a2e; padding:10px 16px; border-radius:6px;
        border-left:3px solid #00d4ff; margin:8px 0 10px 0;
        font-size:13px; color:#b8ccd8; line-height:1.55; }
.desc strong { color:#00d4ff; }
hr { border:none; border-top:1px solid #444; margin:32px 0; }
.back-link { display:inline-block; margin-top:24px; color:#888; }
table { border-collapse:collapse; margin:8px 0 12px 0; font-size:12px; }
th, td { border:1px solid #444; padding:4px 10px; text-align:right; }
th { background:#16213e; color:#00d4ff; text-align:center; }
</style>
</head><body>
<h1>LPPL family oscillators fit to BM-excess</h1>
<p class="muted">
Instead of fitting log\u2081\u2080(price) with a joint trend+oscillation
model, we fix the trend to the BM support line
(A<sub>sup</sub>=-1.559, B<sub>sup</sub>=5.125) and fit only the
oscillation against <code>log_excess = log_price \u2212 BM_support</code>.
Cleaner decomposition: floor is known; cycles are free.
Generated by <code>tools/fit_linppl_hybppl_excess.py</code>.
</p>
<p class="muted">
Note: R\u00b2 here measures fit quality <em>against the excess signal
itself</em> \u2014 it's naturally lower than joint fits which include
the trend variance in ss_tot. Compare \u03c3 to see the actual fit
quality.
</p>
<h2>Summary</h2>
<table>
<thead><tr><th>Model</th><th>params</th><th>R\u00b2 on excess</th><th>\u03c3</th><th>Notes</th></tr></thead>
<tbody>
<tr><td>LPPL on excess</td><td>5</td><td>0.412</td><td>0.227</td><td>W=7.59 (log-time primary)</td></tr>
<tr><td>LPPL\u2082 on excess</td><td>8</td><td>0.476</td><td>0.214</td><td>W\u2081=20.9, W\u2082=7.24 (both log-time)</td></tr>
<tr><td>LinPPL on excess</td><td>5</td><td>0.437</td><td>0.222</td><td>W_cal=1.77 \u2192 T=3.56yr (halving)</td></tr>
<tr><td>HybPPL on excess</td><td>8</td><td><strong>0.699</strong></td><td><strong>0.162</strong></td><td>W_log=7.48 + W_cal=1.75 (T=3.59yr)</td></tr>
</tbody>
</table>
<p class="muted">
HybPPL_excess at 8 params dominates LPPL\u2082_excess at 8 params (R\u00b2
0.699 vs 0.476, \u03c3 0.162 vs 0.214) \u2014 two log-time frequencies
overlap, but log-time + calendar-time are genuinely independent.
</p>
<hr>
<section>
<h2>LPPL on excess (5 params)</h2>
<p class="desc">
<strong>Classic LPPL oscillator only</strong> \u2014
excess = a\u2080 + C\u00b7t\u207b\u1d30\u00b7cos(\u03c9\u00b7ln t + \u03c6).
W converges to 7.59 (close to LPPL\u2081's joint-fit value of 7.56) with
damping D=0.59. Fit quality comparable to LinPPL_excess at the same
parameter count.
</p>
<img src="/excess_fits/fit_lppl_excess.svg" alt="LPPL on excess">
</section>
<hr>
<section>
<h2>LPPL\u2082 on excess (8 params)</h2>
<p class="desc">
<strong>LPPL\u2082 oscillators only</strong> \u2014 two log-time
frequencies. The optimizer parks the "damped primary" at W\u2081=20.9
with D=0.01 (lower bound \u2014 essentially undamped) and the
"undamped secondary" at W\u2082=7.24. Both hit near the canonical LPPL
frequencies (7 and 21), but combining them gains only \u223C6% R\u00b2
over single LPPL\u2081 (0.476 vs 0.412) at a cost of 3 more parameters.
Two log-time frequencies carry overlapping information.
</p>
<img src="/excess_fits/fit_lppl2_excess.svg" alt="LPPL2 on excess">
</section>
<hr>
<section>
<h2>LinPPL on excess (5 params)</h2>
<p class="desc">
<strong>LinPPL oscillator only</strong> \u2014
excess = a\u2080 + C\u00b7t\u207b\u1d30\u00b7cos(\u03c9_cal\u00b7t + \u03c6).
W_cal converges to 1.77 rad/yr (T=3.56yr, Bitcoin's halving cycle).
D=0.01 (at lower bound) confirms no damping \u2014 the halving cycle
persists undamped.
</p>
<img src="/excess_fits/fit_linppl_excess.svg" alt="LinPPL on excess">
</section>
<hr>
<section>
<h2>HybPPL on excess (8 params)</h2>
<p class="desc">
<strong>HybPPL oscillators only</strong> \u2014
excess = a\u2080 + C\u2081\u00b7t\u207b\u1d30\u00b7cos(\u03c9_log\u00b7ln t + \u03c6\u2081)
                  + C\u2082\u00b7cos(\u03c9_cal\u00b7t + \u03c6\u2082).
The log-periodic primary pulls W_log=7.48 (classic LPPL\u2081
frequency); the calendar metronome locks onto T=3.59yr (halving cycle)
with undamped amplitude 0.23. Primary damping D=0.66 means early-cycle
bubbles fade strongly \u2014 so future behavior is dominated by the
halving metronome.
</p>
<img src="/excess_fits/fit_hybppl_excess.svg" alt="HybPPL on excess">
</section>
<a href="/" class="back-link">\u2190 Back to Quantoshi</a>
</body></html>"""
        return html, 200, {"Content-Type": "text/html"}

    @server.route("/excess_fits/<path:filename>")
    def _excess_fits_asset(filename):
        allowed = {"fit_lppl_excess.svg", "fit_lppl2_excess.svg",
                   "fit_linppl_excess.svg", "fit_hybppl_excess.svg",
                   "fit_lppl_excess.csv", "fit_lppl2_excess.csv",
                   "fit_linppl_excess.csv", "fit_hybppl_excess.csv"}
        if filename not in allowed:
            return "Not found", 404
        ctype = "image/svg+xml" if filename.endswith(".svg") else "text/csv"
        asset_path = os.path.join(os.path.dirname(__file__), "..", filename)
        try:
            with open(asset_path) as f:
                return f.read(), 200, {"Content-Type": ctype}
        except FileNotFoundError:
            return "Not generated yet", 404

    @server.route("/E")
    def _regime_shift():
        # Prefer the new multi-model HTML page; fall back to legacy SVG
        html_path = os.path.join(os.path.dirname(__file__), "..", "regime_shift_all.html")
        svg_path = os.path.join(os.path.dirname(__file__), "..", "regime_shift.svg")
        try:
            with open(html_path) as f:
                return f.read(), 200, {"Content-Type": "text/html"}
        except FileNotFoundError:
            pass
        try:
            with open(svg_path) as f:
                return f.read(), 200, {"Content-Type": "image/svg+xml"}
        except FileNotFoundError:
            return "Not generated yet", 404

    @server.route("/regime_shift/<path:filename>")
    def _regime_shift_asset(filename):
        # Serve individual SVGs + CSVs referenced by regime_shift_all.html
        # Restrict to expected filenames to avoid path traversal
        allowed_svg = {
            "regime_shift_pl_2yr.svg", "regime_shift_pl_5yr.svg",
            "regime_shift_pl_7yr.svg",
            "regime_shift_pl_9yr.svg", "regime_shift_linppl_5yr.svg",
            "regime_shift_hybppl_5yr.svg",
            "regime_shift_lp1_5yr.svg", "regime_shift_lp2_5yr.svg",
            "regime_shift_lp3_7yr.svg", "regime_shift_lp3_9yr.svg",
            "regime_shift_lp4_7yr.svg", "regime_shift_lp4_9yr.svg",
            "regime_shift_bm_7yr.svg", "regime_shift_bm_9yr.svg",
            "regime_shift_pl_2yr_clipped.svg", "regime_shift_pl_5yr_clipped.svg",
            "regime_shift_pl_7yr_clipped.svg", "regime_shift_pl_9yr_clipped.svg",
            "regime_shift_pl_2yr_timing_unbounded.svg",
            "regime_shift_pl_2yr_timing_clipped.svg",
            "regime_shift_pl_1yr.svg",
            "regime_shift_pl_1yr_timing_unbounded.svg",
            "regime_shift_pl_6mo.svg",
            "regime_shift_pl_6mo_timing_unbounded.svg",
        }
        allowed_csv = {
            "regime_shift_pl_2yr.csv", "regime_shift_pl_5yr.csv",
            "regime_shift_pl_7yr.csv",
            "regime_shift_pl_9yr.csv", "regime_shift_linppl_5yr.csv",
            "regime_shift_hybppl_5yr.csv",
            "regime_shift_lp1_5yr.csv", "regime_shift_lp2_5yr.csv",
            "regime_shift_lp3_7yr.csv", "regime_shift_lp3_9yr.csv",
            "regime_shift_lp4_7yr.csv", "regime_shift_lp4_9yr.csv",
            "regime_shift_bm_7yr.csv", "regime_shift_bm_9yr.csv",
            "regime_shift_pl_2yr_clipped.csv", "regime_shift_pl_5yr_clipped.csv",
            "regime_shift_pl_7yr_clipped.csv", "regime_shift_pl_9yr_clipped.csv",
            "regime_shift_pl_2yr_timing_unbounded.csv",
            "regime_shift_pl_2yr_timing_clipped.csv",
            "regime_shift_pl_2yr_timing_unbounded_xcorr.csv",
            "regime_shift_pl_2yr_timing_clipped_xcorr.csv",
            "regime_shift_pl_1yr.csv",
            "regime_shift_pl_1yr_timing_unbounded.csv",
            "regime_shift_pl_1yr_timing_unbounded_xcorr.csv",
            "regime_shift_pl_6mo.csv",
            "regime_shift_pl_6mo_timing_unbounded.csv",
            "regime_shift_pl_6mo_timing_unbounded_xcorr.csv",
        }
        if filename in allowed_svg:
            ctype = "image/svg+xml"
        elif filename in allowed_csv:
            ctype = "text/csv"
        else:
            return "Not found", 404
        asset_path = os.path.join(os.path.dirname(__file__), "..", filename)
        try:
            with open(asset_path) as f:
                return f.read(), 200, {"Content-Type": ctype}
        except FileNotFoundError:
            return "Not generated yet", 404

    @server.route("/C")
    def _sensitivity_pq():
        svg_path = os.path.join(os.path.dirname(__file__), "..", "sensitivity_pq.svg")
        try:
            with open(svg_path) as f:
                return f.read(), 200, {"Content-Type": "image/svg+xml"}
        except FileNotFoundError:
            return "Not generated yet", 404

    @server.route("/docs/architecture")
    def _docs_architecture():
        return _render_doc("architecture.md", "Architecture Guide")

    @server.route("/docs/user-manual")
    def _docs_user_manual():
        return _render_doc("user_manual.md", "User Manual")

    # ── Cache stats endpoint ────────────────────────────────────────────
    @server.route("/api/cache-stats")
    def _cache_stats():
        from utils import _ALL_CACHES
        from cache import redis_available
        stats = {}
        for name, cache in _ALL_CACHES.items():
            info = cache.cache_info()
            total = info.hits + info.misses
            stats[name] = {
                "hits": info.hits,
                "misses": info.misses,
                "size": info.currsize,
                "maxsize": info.maxsize,
                "rate": f"{info.hits/total:.1%}" if total else "n/a",
            }
        return jsonify({
            "worker_pid": os.getpid(),
            "redis": redis_available(),
            "caches": stats,
        })

    if not btcpay._HAS_BTCPAY:
        # No BTCPay configured — register stub routes that always return "free"
        @server.route("/api/mc/invoice", methods=["POST"])
        def _mc_invoice_stub():
            return jsonify({"free": True, "message": "BTCPay not configured — MC is free"}), 200

        @server.route("/api/mc/invoice/<invoice_id>", methods=["GET"])
        def _mc_status_stub(invoice_id):
            return jsonify({"status": "Settled", "paid": True}), 200

        @server.route("/api/mc/invoice/<invoice_id>/payment", methods=["GET"])
        def _mc_payment_stub(invoice_id):
            return jsonify({"methods": []}), 200

        @server.route("/api/mc/verify", methods=["POST"])
        def _mc_verify_stub():
            return jsonify({"valid": True}), 200

        log.info("MC payment routes registered (stub — BTCPay not configured)")
        return

    # ── POST /api/mc/invoice ─────────────────────────────────────────────────

    @server.route("/api/mc/invoice", methods=["POST"])
    def _mc_create_invoice():
        ip = _client_ip()

        # Rate limit
        err = _check_rate_limit(ip)
        if err:
            return jsonify({"error": err}), 429

        data = request.get_json(silent=True) or {}
        tab      = data.get("tab", "dca")
        mc_years = int(data.get("mc_years", 10))
        start_yr = int(data.get("start_yr", 2026))

        # Validate
        if tab not in _ALLOWED_TABS:
            return jsonify({"error": "Invalid tab"}), 400
        if mc_years not in _ALLOWED_MC_YEARS:
            return jsonify({"error": "Invalid mc_years"}), 400

        # Free tier check
        entry_q = float(data.get("entry_q", btcpay.MC_DEFAULT_ENTRY_Q))
        model_key = data.get("model_key", "bub")
        if btcpay.is_free_tier(model_key, mc_years, start_yr, entry_q):
            return jsonify({"free": True, "message": "Free tier — no payment needed"}), 200

        try:
            result = btcpay.create_invoice(tab, mc_years)
        except Exception as e:
            log.error("BTCPay create_invoice failed: %s", e)
            return jsonify({"error": "Payment service unavailable"}), 503

        _record_invoice(ip)
        log.info("Invoice created: %s (%s sats, %s %dyr, live)",
                 result["invoice_id"], result["amount_sats"], tab, mc_years)

        return jsonify(result), 201

    # ── GET /api/mc/invoice/<id> ─────────────────────────────────────────────

    @server.route("/api/mc/invoice/<invoice_id>", methods=["GET"])
    def _mc_check_invoice(invoice_id):
        if not _INVOICE_ID_RE.match(invoice_id):
            return jsonify({"error": "Invalid invoice ID"}), 400

        tab      = request.args.get("tab", "dca")
        mc_years = int(request.args.get("mc_years", 10))

        try:
            result = btcpay.check_invoice(invoice_id)
        except Exception as e:
            log.error("BTCPay check_invoice failed: %s", e)
            return jsonify({"error": "Payment service unavailable"}), 503

        # If paid, generate a payment token and mark rate-limit entry
        if result["paid"]:
            token = btcpay.generate_payment_token(invoice_id, tab, mc_years)
            result["payment_token"] = token
            ip = _client_ip()
            _mark_paid(ip, invoice_id)

        return jsonify(result), 200

    # ── POST /api/mc/verify ──────────────────────────────────────────────────

    @server.route("/api/mc/verify", methods=["POST"])
    def _mc_verify_token():
        data = request.get_json(silent=True) or {}
        token      = data.get("payment_token", "")
        invoice_id = data.get("invoice_id", "")
        tab        = data.get("tab", "")
        mc_years   = int(data.get("mc_years", 0))

        if not all([token, invoice_id, tab, mc_years]):
            return jsonify({"valid": False, "error": "Missing fields"}), 400

        valid = btcpay.verify_payment_token(token, invoice_id, tab, mc_years)
        return jsonify({"valid": valid}), 200

    # ── GET /api/mc/invoice/<id>/payment ─────────────────────────────────────

    @server.route("/api/mc/invoice/<invoice_id>/payment", methods=["GET"])
    def _mc_payment_methods(invoice_id):
        if not _INVOICE_ID_RE.match(invoice_id):
            return jsonify({"error": "Invalid invoice ID"}), 400

        try:
            methods = btcpay.get_payment_methods(invoice_id)
        except Exception as e:
            log.error("BTCPay get_payment_methods failed: %s", e)
            return jsonify({"error": "Could not fetch payment methods"}), 503

        # Generate QR codes as SVG data URIs
        for m in methods:
            link = m.get("payment_link") or m.get("destination", "")
            m["qr_svg"] = btcpay.generate_qr_svg(link)

        return jsonify({"methods": methods}), 200

    log.info("MC payment routes registered (BTCPay active)")
