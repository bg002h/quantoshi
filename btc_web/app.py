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

import logging
import sys
import time
from pathlib import Path

# ── make btc_app/ importable ──────────────────────────────────────────────────
_HERE    = Path(__file__).parent
_ROOT    = _HERE.parent
_BTC_APP = _ROOT / "archive" / "btc_app"
for _p in (_ROOT, _BTC_APP):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from datetime import date

import dash
import dash_bootstrap_components as dbc
from flask import request as flask_request

from btc_core import load_model_data, BubbleModel, PowerLawModel, LPPLModel, ExponentialModel, S2FModel, EmpiricalFloorModel, QuantileRegressionModel
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
    if path in ('/', '/_dash-layout', '/_dash-dependencies') or path.startswith('/1') or path.startswith('/2') or path.startswith('/3') or path.startswith('/4') or path.startswith('/5') or path.startswith('/6') or path.startswith('/7') or path.startswith('/8') or path.startswith('/9'):
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

# ── register price models ──────────────────────────────────────────────────
_app_ctx.PRICE_MODELS["bub"] = BubbleModel(M)
_app_ctx.PRICE_MODELS["qr"] = QuantileRegressionModel(M)
_app_ctx.PRICE_MODELS["pl"]  = PowerLawModel(
    M.ols_intercept, M.ols_slope, M.price_years, M.price_prices,
    M.genesis, M.QR_QUANTILES)
_app_ctx.PRICE_MODELS["lppl"] = LPPLModel(M.price_years, M.price_prices, M.QR_QUANTILES)
_app_ctx.PRICE_MODELS["exp"] = ExponentialModel(M.price_years, M.price_prices, M.QR_QUANTILES)
_app_ctx.PRICE_MODELS["s2f"] = S2FModel(M.price_years, M.price_prices, M.genesis)
# ── Empirical Floor (conditional — only if pkl exists) ────────────────
_ef_pkl = Path(__file__).parent.parent / "btc_app" / "model_data_ef.pkl"
if _ef_pkl.exists():
    _app_ctx.PRICE_MODELS["ef"] = EmpiricalFloorModel(str(_ef_pkl))
_app_ctx.DEFAULT_MODEL = _app_ctx.PRICE_MODELS["bub"]

# ── compute per-quantile R² for all models ───────────────────────────────
import numpy as np
from btc_core import compute_model_r2, _compute_log_r2
for _mdl in _app_ctx.PRICE_MODELS.values():
    compute_model_r2(_mdl, M.price_years, M.price_prices)

# OLS R²
_ols_pred = 10 ** (M.ols_intercept + M.ols_slope * np.log10(
    np.maximum(M.price_years[M.price_years >= 1.0], 0.1)))
M.ols_r2 = _compute_log_r2(M.price_prices[M.price_years >= 1.0], _ols_pred)

# ── Apply thermal color palette to quantile traces ────────────────────────
from figures import _build_thermal_colors
_thermal = _build_thermal_colors(M.QR_QUANTILES)
_app_ctx.DEFAULT_MODEL.colors.update(_thermal)
M.qr_colors.update(_thermal)  # propagate to layout.py quantile panel dots

import btcpay
_app_ctx._HAS_BTCPAY = btcpay._HAS_BTCPAY

import api
api.register_routes(server)
_app_ctx._ALL_QS = [q for q in M.QR_QUANTILES if 0.001 <= q <= 0.999]
_app_ctx._DEF_QS = [q for q in [0.001, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
                    if q in _app_ctx.DEFAULT_MODEL.fits]

from utils import _startup_heatmap_defaults, _nearest_quantile, _get_bubble_fig, \
    _get_dca_fig, _get_retire_fig, _get_supercharge_fig, _get_heatmap_fig, \
    _get_citadel_fig
_app_ctx._HM_ENTRY_Q_DEFAULT = _startup_heatmap_defaults()

# ── import layout (sets app.layout) and callbacks (registers @callbacks) ─────
import snapshot   # noqa: F401 — defines _CHECKLIST_OPTIONS using _app_ctx
import layout     # noqa: F401 — sets app.layout
import callbacks  # noqa: F401 — registers all callbacks

# ══════════════════════════════════════════════════════════════════════════════
# Pre-warm LRU caches on worker startup
# ══════════════════════════════════════════════════════════════════════════════
def _prewarm_caches():
    """Pre-warm L0 caches. Tries Redis first; falls back to full compute."""
    import time as _time
    import hashlib as _hl
    from tab_defaults import (bubble_defaults, heatmap_defaults, dca_defaults,
                              retire_defaults, supercharge_defaults, citadel_defaults)
    from utils import (_compute_cache_key, _serialize_result, _deserialize_result,
                       _L0_BUILDERS)
    import utils as _utils

    t0 = _time.time()

    # Build the list of (prefix, params_dict) to prewarm
    _entries = []

    _entries.append(("bub", bubble_defaults()))

    bub_pl = bubble_defaults()
    bub_pl["active_models"] = ["bub", "pl"]
    _entries.append(("bub", bub_pl))

    _entries.append(("dca", dca_defaults()))
    _entries.append(("ret", retire_defaults()))

    sc = supercharge_defaults()
    sc["selected_qs"] = [q for q in [0.001, 0.10] if q in _app_ctx.DEFAULT_MODEL.fits]
    _entries.append(("sc", sc))

    cp = citadel_defaults()
    cp["selected_qs"] = [0.01, 0.10, 0.25]
    _entries.append(("cp", cp))

    hm = heatmap_defaults()
    hm["entry_q"] = _app_ctx._HM_ENTRY_Q_DEFAULT
    _entries.append(("hm", hm))

    # Compute cache keys for each entry
    keyed = [(pfx, p, _compute_cache_key(pfx, dict(p))) for pfx, p in _entries]

    # Try loading from Redis L0
    try:
        from cache import get_l0, set_l0, redis_available
        if redis_available():
            hits = []
            for pfx, p, json_key in keyed:
                params_hash = _hl.sha256(json_key.encode()).hexdigest()[:16]
                raw = get_l0(pfx, params_hash)
                if raw:
                    hits.append((pfx, json_key, raw))
                else:
                    break  # any miss → full recompute

            if len(hits) == len(keyed):
                # All found in Redis — pin in L0 and skip computation
                for pfx, json_key, raw in hits:
                    cache_fn = _L0_BUILDERS[pfx][0]
                    result = _deserialize_result(raw)
                    cache_fn.pin(json_key, result)
                logging.getLogger(__name__).info("L0 warm from Redis: %d entries in %.1fs",
                            len(hits), _time.time() - t0)
                return
    except Exception as e:
        logging.getLogger(__name__).debug("L0 Redis check failed: %s", e)

    # Redis miss or unavailable — full compute
    logging.getLogger(__name__).info("L0 computing %d entries...", len(keyed))
    redis_ok = True
    for pfx, p, json_key in keyed:
        cache_fn = _L0_BUILDERS[pfx][0]
        get_fn = _L0_BUILDERS[pfx][1]
        result = get_fn(dict(p))  # compute figure (also populates L1)
        cache_fn.pin(json_key, result)  # pin in L0

        # Store in Redis L0
        try:
            from cache import set_l0, redis_available
            if redis_ok and redis_available():
                params_hash = _hl.sha256(json_key.encode()).hexdigest()[:16]
                set_l0(pfx, params_hash, _serialize_result(result))
            else:
                redis_ok = False
        except Exception:
            redis_ok = False

    if not redis_ok:
        _utils._l0_needs_flush = True
        logging.getLogger(__name__).info("L0 compute done (Redis unavailable — deferred flush enabled)")
    else:
        logging.getLogger(__name__).info("L0 compute + Redis store: %d entries in %.1fs",
                    len(keyed), _time.time() - t0)

_prewarm_caches()
from utils import _log_cache_stats
_log_cache_stats()

# Pre-warm default transition matrix
if _HAS_MARKOV:
    _get_transition_matrix(M, 5, 30, [2010, date.today().year])

# ── Background MC figure prewarm (runs in each worker's first request) ───────
def _prewarm_mc_caches():
    """Pre-warm LRU caches for free-tier MC figures (DCA, Retire, SC)."""
    import logging as _log
    _log.getLogger(__name__).info("MC prewarm: starting background warm")
    from mc_cache import MC_FREE_SIMS, CACHED_START_YRS, ENTRY_PCT_BINS, MC_YEARS_OPTIONS
    yr_now = date.today().year

    def _mc_overrides(s_yr, mc_yrs, entry_q=10):
        return dict(
            mc_enabled=True, mc_amount=100, mc_infl=4.0,
            mc_bins=5, mc_sims=MC_FREE_SIMS, mc_years=mc_yrs,
            mc_freq="Monthly", mc_window=[2010, yr_now],
            mc_start_yr=s_yr, mc_entry_q=entry_q,
            mc_live_price=0, mc_blocked_bins=(),
            mc_free_tier=True,
        )

    def _try(fig_fn, params, label, s_yr, mc_yrs):
        try:
            fig_fn(params)
        except Exception as e:
            _log.getLogger(__name__).warning("MC prewarm %s %d/%d failed: %s",
                                             label, s_yr, mc_yrs, e)

    from tab_defaults import dca_defaults, retire_defaults, supercharge_defaults

    # Prewarm one representative combo per start year (bub model, Q10%, 40yr)
    for s_yr in CACHED_START_YRS:
        for mc_yrs in MC_YEARS_OPTIONS:
            mc = _mc_overrides(s_yr, mc_yrs)
            _dca = dca_defaults()
            _dca["end_yr"] = yr_now + mc_yrs
            _dca.update(mc)
            _try(_get_dca_fig, _dca, "DCA", s_yr, mc_yrs)

            _ret = retire_defaults()
            _ret["mc_start_stack"] = 1.0
            _ret.update({k: v for k, v in mc.items() if k != "mc_amount"})
            _ret["mc_amount"] = 5000
            _try(_get_retire_fig, _ret, "Ret", s_yr, mc_yrs)

            _sc = supercharge_defaults()
            _sc["selected_qs"] = [q for q in [0.001, 0.10] if q in _app_ctx.DEFAULT_MODEL.fits]
            _sc["display_q"] = _nearest_quantile(0.05, _app_ctx._ALL_QS)
            _sc["mc_start_stack"] = 1.0
            _sc.update({k: v for k, v in mc.items() if k != "mc_amount"})
            _sc["mc_amount"] = 5000
            _try(_get_supercharge_fig, _sc, "SC", s_yr, mc_yrs)
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
