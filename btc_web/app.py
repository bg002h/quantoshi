"""app.py — Quantoshi web app (Plotly Dash).

Run locally:
    cd /scratch/code/bitcoinprojections
    bash run_web.sh
    → http://localhost:8050  (or http://<your-ip>:8050 from phone/tablet)

Multi-user server deploy (via btc-web.service):
    gunicorn 'btc_web.app:server' -b 0.0.0.0:8050 -w 4

Privacy model:
    - Lot data lives exclusively in each user's browser localStorage.
    - Nothing is persisted server-side; lots only transit server memory
      transiently during Dash callbacks (never written to disk).
"""

import sys
import time
from pathlib import Path

# ── make btc_app/ importable ──────────────────────────────────────────────────
_HERE    = Path(__file__).parent
_ROOT    = _HERE.parent
_BTC_APP = _ROOT / "btc_app"
for _p in (_ROOT, _BTC_APP):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import json
import pandas as pd
from functools import lru_cache

import dash
import dash_bootstrap_components as dbc
from flask import request as flask_request

from btc_core import load_model_data
from figures import FREQ_PPY
from mc_overlay import save_trans_cache_to_disk, _get_transition_matrix
import atexit
atexit.register(save_trans_cache_to_disk)
try:
    from markov import build_transition_matrix
    _HAS_MARKOV = True
except ImportError:
    _HAS_MARKOV = False

# ── load model (once at startup) ──────────────────────────────────────────────
M = load_model_data()

# ── create Dash app ───────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=["/assets/bootstrap_flatly.min.css"],
    suppress_callback_exceptions=True,
    meta_tags=[
        {"name": "color-scheme", "content": "only light"},
        {"name": "referrer", "content": "no-referrer"},
    ],
)
app.title = "Quantoshi"
server = app.server  # for gunicorn

@server.route("/health")
def _health():
    from flask import jsonify as _jsonify
    from utils import _price_cache
    from mc_cache import _CACHE
    price_age = time.time() - _price_cache["ts"] if _price_cache["price"] else -1
    bp = btcpay.check_health() if flask_request.args.get("btcpay") else None
    result = {
        "status": "ok",
        "model": M is not None,
        "price_age_s": round(price_age),
        "mc_cache": bool(_CACHE),
        "markov": _HAS_MARKOV,
        "btcpay": btcpay._HAS_BTCPAY,
    }
    if bp is not None:
        result["btcpay_health"] = bp
    return _jsonify(result), 200

@server.after_request
def _cache_headers(response):
    path = flask_request.path
    if path in ('/_dash-layout', '/_dash-dependencies'):
        response.headers.update({
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma':        'no-cache',
            'Expires':       '0',
        })
    elif path.startswith('/_dash-component-suites/'):
        response.headers['Cache-Control'] = 'public, max-age=31536000, immutable'

    # ── Security headers ─────────────────────────────────────────────────
    response.headers['Referrer-Policy'] = 'no-referrer'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['Permissions-Policy'] = (
        'camera=(), microphone=(), geolocation=(), '
        'interest-cohort=(), usb=()'
    )
    response.headers['X-DNS-Prefetch-Control'] = 'off'

    # ── Onion-Location (clearnet only — shows ".onion available" in Tor) ─
    _is_onion = flask_request.host.endswith('.onion')
    if not _is_onion:
        response.headers['Onion-Location'] = (
            'http://u5dprelc4ti7xoczb5sbtye6qidlji2l6psmkx35anvxgjyqrkmu32ad.onion'
            + flask_request.path
        )

    # ── Content-Security-Policy (tighter on .onion — no clearnet leaks) ──
    _frame_src = btcpay.BTCPAY_URL.rstrip('/') if btcpay._HAS_BTCPAY else "'none'"
    if _is_onion:
        _connect = ("'self'"
                    " http://jxnpv6ef3yo2kqpeu6u3nmv343k7vpyn7katlfdoc3n7hgvz7l5woqid.onion"
                    " ws://jxnpv6ef3yo2kqpeu6u3nmv343k7vpyn7katlfdoc3n7hgvz7l5woqid.onion"
                    " http://explorerzydxu5ecjrkwceayqybizmpjjznk5izmitf2modhcusuqlid.onion")
    else:
        _connect = ("'self' https://mempool.space wss://mempool.space"
                    " https://blockstream.info")
    response.headers['Content-Security-Policy'] = "; ".join([
        "default-src 'self'",
        "script-src 'self' 'unsafe-inline'",
        "style-src 'self' 'unsafe-inline'",
        f"frame-src {_frame_src}",
        "frame-ancestors 'none'",
        "base-uri 'self'",
        "img-src 'self' data: blob:",
        f"connect-src {_connect}",
        "font-src 'self'",
    ])
    return response

# ── populate shared context ──────────────────────────────────────────────────
import _app_ctx
_app_ctx.M = M
_app_ctx.app = app
_app_ctx.server = server
_app_ctx._HAS_MARKOV = _HAS_MARKOV

import btcpay
_app_ctx._HAS_BTCPAY = btcpay._HAS_BTCPAY

import api
api.register_routes(server)
_app_ctx._ALL_QS = [q for q in M.QR_QUANTILES if 0.001 <= q <= 0.999]
_app_ctx._DEF_QS = [q for q in [0.001, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
                    if q in M.qr_fits]

from utils import _startup_heatmap_defaults, _nearest_quantile, _get_bubble_fig, \
    _get_dca_fig, _get_retire_fig, _get_supercharge_fig, _get_heatmap_fig
_app_ctx._HM_ENTRY_Q_DEFAULT = _startup_heatmap_defaults()

# ── import layout (sets app.layout) and callbacks (registers @callbacks) ─────
import snapshot   # noqa: F401 — defines _CHECKLIST_OPTIONS using _app_ctx
import layout     # noqa: F401 — sets app.layout
import callbacks  # noqa: F401 — registers all callbacks

# ══════════════════════════════════════════════════════════════════════════════
# Pre-warm LRU caches on worker startup
# ══════════════════════════════════════════════════════════════════════════════
def _prewarm_caches():
    yr_now = pd.Timestamp.today().year

    # Bubble (default: no quantiles, log-log, 3 future bubbles)
    _get_bubble_fig(dict(
        selected_qs = [],
        shade=True, show_ols=False, show_data=True, show_today=True,
        show_legend=False, minor_grid=False,
        show_comp=True, show_sup=False,
        xscale="log", yscale="log",
        xmin=2012, xmax=yr_now + 4,
        ymin=0.01, ymax=1e7,
        n_future=3, pt_size=3, pt_alpha=0.3,
        stack=0, show_stack=False, use_lots=False, lots=[],
        comp_color="#FFD700", comp_lw=2.0,
        sup_color="#888888", sup_lw=1.5,
    ))

    # DCA (default: $100/mo, Q50%, current_yr–current_yr+10)
    _get_dca_fig(dict(
        start_stack=0, use_lots=False,
        amount=100.0, freq="Monthly",
        start_yr=yr_now, end_yr=yr_now + 10,
        disp_mode="btc", log_y=False, show_today=False,
        show_legend=False, minor_grid=False,
        selected_qs=[0.50], lots=[],
        sc_enabled=False, sc_loan_amount=0,
        sc_rate=_app_ctx.SC_DEFAULT_RATE,
        sc_loan_type="interest_only", sc_term_months=48.0,
        sc_repeats=0, sc_rollover=False,
        sc_entry_mode="live",
        sc_custom_price=float(_app_ctx.SC_DEFAULT_PRICE),
        sc_tax_rate=0.33, sc_live_price=None,
    ))

    # Retire (default: $5000/mo, Q1%+Q10%+Q25%, 2031–2075, 4% inflation)
    _get_retire_fig(dict(
        start_stack=1.0, use_lots=False,
        wd_amount=5000.0, freq="Monthly",
        start_yr=2031, end_yr=2075,
        inflation=4.0, disp_mode="btc",
        log_y=True, annotate=True,
        show_legend=False, minor_grid=True,
        selected_qs=[0.01, 0.10, 0.25],
        lots=[],
    ))

    # Supercharge (default: Mode A, 1 BTC, annually, Q0.1%+Q10%)
    _get_supercharge_fig(dict(
        mode         = "a",
        start_stack  = 1.0,
        start_yr     = 2033,
        delays       = [0.0, 0.0, 0.0, 1.0, 2.0],
        freq         = "Annually",
        inflation    = 4.0,
        selected_qs  = [q for q in [0.001, 0.10] if q in M.qr_fits],
        chart_layout = 2,
        display_q    = _nearest_quantile(0.05, _app_ctx._ALL_QS),
        wd_amount    = 100000,
        end_yr       = 2075,
        disp_mode    = "usd",
        log_y        = True,
        annotate     = True,
        show_legend  = False,
        minor_grid   = True,
        target_yr    = 2060,
        lots         = [],
        use_lots     = False,
    ))

_prewarm_caches()

# Pre-warm default transition matrix
if _HAS_MARKOV:
    _get_transition_matrix(M, 5, 30, [2010, pd.Timestamp.today().year])

# ── Background MC figure prewarm (runs in each worker's first request) ───────
def _prewarm_mc_caches():
    """Pre-warm LRU caches for free-tier MC figures (DCA, Retire, SC)."""
    import logging as _log
    _log.getLogger(__name__).info("MC prewarm: starting background warm")
    from mc_cache import MC_FREE_SIMS, MC_FREE_START_YRS, MC_FREE_ENTRY_Q, MC_FREE_YEARS
    yr_now = pd.Timestamp.today().year

    def _mc_overrides(s_yr, mc_yrs):
        return dict(
            mc_enabled=True, mc_amount=100, mc_infl=4.0,
            mc_bins=5, mc_sims=MC_FREE_SIMS, mc_years=mc_yrs,
            mc_freq="Monthly", mc_window=[2010, yr_now],
            mc_start_yr=s_yr, mc_entry_q=MC_FREE_ENTRY_Q,
            mc_live_price=0, mc_blocked_bins=(),
            mc_free_tier=True,
        )

    def _try(fig_fn, params, label, s_yr, mc_yrs):
        try:
            fig_fn(params)
        except Exception as e:
            _log.getLogger(__name__).warning("MC prewarm %s %d/%d failed: %s",
                                             label, s_yr, mc_yrs, e)

    for s_yr in MC_FREE_START_YRS:
        for mc_yrs in MC_FREE_YEARS:
            mc = _mc_overrides(s_yr, mc_yrs)
            _try(_get_dca_fig, dict(
                start_stack=0, use_lots=False,
                amount=100.0, freq="Monthly",
                start_yr=yr_now, end_yr=yr_now + mc_yrs,
                disp_mode="btc", log_y=False, show_today=False,
                show_legend=False, legend_pos="outside", minor_grid=False,
                selected_qs=[0.50], lots=[],
                sc_enabled=False, sc_loan_amount=0,
                sc_rate=_app_ctx.SC_DEFAULT_RATE,
                sc_loan_type="interest_only", sc_term_months=48.0,
                sc_repeats=0, sc_rollover=False,
                sc_entry_mode="live",
                sc_custom_price=float(_app_ctx.SC_DEFAULT_PRICE),
                sc_tax_rate=0.33, sc_live_price=None,
                **mc,
            ), "DCA", s_yr, mc_yrs)
            _try(_get_retire_fig, dict(
                start_stack=1.0, use_lots=False,
                wd_amount=5000.0, freq="Monthly",
                start_yr=2031, end_yr=2075,
                inflation=4.0, disp_mode="btc",
                log_y=True, annotate=True,
                show_legend=False, legend_pos="outside", minor_grid=True,
                selected_qs=[0.01, 0.10, 0.25], lots=[],
                mc_start_stack=1.0,
                **{k: v for k, v in mc.items() if k != "mc_amount"},
                mc_amount=5000,
            ), "Ret", s_yr, mc_yrs)
            _try(_get_supercharge_fig, dict(
                mode="a", start_stack=1.0, start_yr=2033,
                delays=[0.0, 0.0, 0.0, 1.0, 2.0],
                freq="Annually", inflation=4.0,
                selected_qs=[q for q in [0.001, 0.10] if q in M.qr_fits],
                chart_layout=2,
                display_q=_nearest_quantile(0.05, _app_ctx._ALL_QS),
                wd_amount=100000, end_yr=2075, disp_mode="usd",
                log_y=True, annotate=True,
                show_legend=False, legend_pos="outside", minor_grid=True,
                target_yr=2060, lots=[], use_lots=False,
                mc_start_stack=1.0,
                **{k: v for k, v in mc.items() if k != "mc_amount"},
                mc_amount=5000,
            ), "SC", s_yr, mc_yrs)
    _log.getLogger(__name__).info("MC prewarm: done")

_mc_prewarm_triggered = False

@server.before_request
def _trigger_mc_prewarm():
    global _mc_prewarm_triggered
    if not _mc_prewarm_triggered and _HAS_MARKOV:
        _mc_prewarm_triggered = True
        import threading
        threading.Thread(target=_prewarm_mc_caches, daemon=True,
                         name="mc-prewarm").start()

# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os, socket
    port = int(os.environ.get("PORT", 8050))
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except Exception:
        local_ip = "127.0.0.1"
    print(f"\n  Quantoshi Web App")
    print(f"  Local:   http://localhost:{port}")
    print(f"  Network: http://{local_ip}:{port}\n")
    dev = os.environ.get("DEV", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=dev, use_reloader=False)
