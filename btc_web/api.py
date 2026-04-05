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
                return f.read(), 200, {"Content-Type": "image/svg+xml"}
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
            "regime_shift_pl_5yr.svg", "regime_shift_pl_7yr.svg",
            "regime_shift_pl_9yr.svg", "regime_shift_linppl_5yr.svg",
            "regime_shift_hybppl_5yr.svg",
            "regime_shift_lp1_5yr.svg", "regime_shift_lp2_5yr.svg",
            "regime_shift_lp3_7yr.svg", "regime_shift_lp3_9yr.svg",
            "regime_shift_lp4_7yr.svg", "regime_shift_lp4_9yr.svg",
            "regime_shift_bm_7yr.svg", "regime_shift_bm_9yr.svg",
        }
        allowed_csv = {
            "regime_shift_pl_5yr.csv", "regime_shift_pl_7yr.csv",
            "regime_shift_pl_9yr.csv", "regime_shift_linppl_5yr.csv",
            "regime_shift_hybppl_5yr.csv",
            "regime_shift_lp1_5yr.csv", "regime_shift_lp2_5yr.csv",
            "regime_shift_lp3_7yr.csv", "regime_shift_lp3_9yr.csv",
            "regime_shift_lp4_7yr.csv", "regime_shift_lp4_9yr.csv",
            "regime_shift_bm_7yr.csv", "regime_shift_bm_9yr.csv",
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
