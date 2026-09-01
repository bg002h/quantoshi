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
import os
import sys
import time
from pathlib import Path
import _app_ctx

_BOOT_T0 = time.monotonic()
def _boot_mark(label: str) -> None:
    """Boot-time phase marker. Writes to stderr so systemd/gunicorn capture it.
    Set QS_BOOT_TRACE=1 to enable (defaults off — noisy on every restart)."""
    if os.environ.get("QS_BOOT_TRACE") == "1":
        print(f"[boot +{time.monotonic() - _BOOT_T0:5.2f}s] {label}",
              file=sys.stderr, flush=True)
_boot_mark("enter app.py")

# ── make project root importable (btc_core/ package lives there) ─────────────
_HERE    = Path(__file__).parent
_ROOT    = _HERE.parent
for _p in (_ROOT,):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from datetime import date

import dash
import dash_bootstrap_components as dbc
from flask import request as flask_request

from btc_core import (load_model_data, BubbleModel, PowerLawModel,
                       LPPLModel, LPPL2Model, LPPL3Model, LPPL4Model,
                       LPPLModelW, LPPL2ModelW, LPPL3ModelW, LPPL4ModelW,
                       LPPL4ModelN13, LPPL4ModelWN13, LinPPLModel, HybPPLModel,
                       HybPPLDDModel,
                       Hyb2LModel, Hyb2CModel, Hyb2BModel, Hyb4DModel, PCAModel,
                       GreedyModel, EntropyPPLModel,
                       ExponentialModel, GompertzModel, LogisticSCurveModel,
                       OffsetPowerLawModel, StretchedExponentialModel,
                       SaturatingPowerLawModel,
                       BrokenPowerLawModel,
                       S2FModel, EmpiricalFloorModel, QuantileRegressionModel,
                       HybPPLConfigModel, _HYBPPL_CONFIG_PARAMS,
                       EPPLConfigModel, _EPPL_CONFIG_PARAMS)
from figures import FREQ_PPY
from mc_overlay import save_trans_cache_to_disk, _get_transition_matrix
import atexit
atexit.register(save_trans_cache_to_disk)
_HAS_MARKOV = _app_ctx._HAS_MARKOV
_boot_mark("imports + MC load done")

# ── load model (once at startup) ──────────────────────────────────────────────
M = load_model_data()
_boot_mark("load_model_data done")

# ── Background callback manager (diskcache) ─────────────────────────────────
try:
    from dash import DiskcacheManager
    import diskcache
    _bg_cache = diskcache.Cache("/tmp/quantoshi-dash-cache")
    _bg_manager = DiskcacheManager(_bg_cache, expire=600)
except ImportError:
    _bg_manager = None

# ── Color artifact regeneration (DEV mode only) ──────────────────────────────
# In dev mode, regenerate _colors_generated.css/js from colors.py on
# every startup so edits to colors.py propagate without manual steps.
# In production (gunicorn), the checked-in artifacts are used as-is to
# avoid race conditions between workers writing the same files.
if os.environ.get("DEV"):
    try:
        import sys as _sys
        from pathlib import Path as _Path
        _proj_root = _Path(__file__).resolve().parent.parent
        _sys.path.insert(0, str(_proj_root))
        from tools.generate_color_artifacts import write_artifacts as _write_color_artifacts
        _write_color_artifacts()
    except Exception as _e:
        # Non-fatal — never block startup. Manual generator run is the fallback.
        print(f"[colors] DEV-mode color artifact regen skipped: {_e}")

# ── create Dash app ───────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=["/assets/bootstrap_flatly.min.css"],
    suppress_callback_exceptions=True,
    background_callback_manager=_bg_manager,
    meta_tags=[
        {"name": "color-scheme", "content": "only light"},
        {"name": "referrer", "content": "same-origin"},
    ],
)
app.title = "Quantoshi"
server = app.server  # for gunicorn


# ── Per-callback timing log (opt-in via QS_TRACE_CALLBACKS=1 env var) ──────
# Hooks Dash's /_dash-update-component endpoint. Logs every callback's
# Output target + duration to journal under [trace-cb]. Enabled via env
# var so prod traffic isn't always logged. Set in systemd unit when
# debugging restore performance.
import os as _qs_os, time as _qs_time, json as _qs_json
if _qs_os.environ.get("QS_TRACE_CALLBACKS") == "1":
    from flask import request as _qs_req, g as _qs_g

    @server.before_request
    def _qs_cb_before():
        if _qs_req.path == "/_dash-update-component":
            _qs_g._qs_t0 = _qs_time.perf_counter()

    @server.after_request
    def _qs_cb_after(response):
        if _qs_req.path == "/_dash-update-component" and hasattr(_qs_g, "_qs_t0"):
            dt_ms = (_qs_time.perf_counter() - _qs_g._qs_t0) * 1000
            outputs = "?"
            try:
                body = _qs_req.get_json(silent=True) or {}
                outputs = body.get("output") or "?"
                if isinstance(outputs, str) and len(outputs) > 200:
                    outputs = outputs[:200] + "…"
            except Exception:
                pass
            print(f"[trace-cb] {dt_ms:6.1f}ms output={outputs}", flush=True)
        return response

@server.route("/A")
def _palette_picker():
    from flask import send_from_directory
    return send_from_directory("assets", "palette_picker.html")



@server.route("/_trace", methods=["POST"])
def _trace_post():
    """Browser-console trace sink. Mobile devs can't open DevTools, so the
    qs_trace.js client POSTs its trace log here when ?trace=1 is set;
    we mirror to the gunicorn journal under [trace-client] so it appears
    in `journalctl -u quantoshi`. Bounded payload, fire-and-forget."""
    from flask import request as _req
    raw = _req.get_data(cache=False, as_text=True)
    if not raw or len(raw) > 16 * 1024:
        return ("", 204)
    label = (_req.args.get("label") or "restore")[:32]
    ua = (_req.headers.get("User-Agent") or "")[:60]
    # Print directly so it lands in the gunicorn-captured stream regardless
    # of logger configuration.
    print(f"[trace-client] label={label} ua={ua} payload={raw}", flush=True)
    return ("", 204)


@server.route("/health")
def _health():
    from flask import jsonify as _jsonify
    from utils import _price_cache
    from mc_cache import _CACHE
    import pathlib
    price_age = time.time() - _price_cache["ts"] if _price_cache["price"] else -1
    # Cache file age (oldest .npz in mc_cache/ or citadel_band_cache/)
    cache_dirs = [
        pathlib.Path(__file__).parent / "mc_cache",
        pathlib.Path(__file__).parent / "citadel_band_cache",
    ]
    mtimes = []
    for d in cache_dirs:
        if d.exists():
            mtimes.extend(f.stat().st_mtime for f in d.glob("*.npz"))
    cache_age_days = (time.time() - min(mtimes)) / 86400 if mtimes else -1
    bp = btcpay.check_health() if flask_request.args.get("btcpay") else None
    # Custom Time Axis block map load state (section 5 case E detection)
    try:
        from engines.custom_fit import _BLOCK_MAP_LOADED
        block_map_loaded = bool(_BLOCK_MAP_LOADED)
    except Exception:
        block_map_loaded = False
    # model_data.pkl age (daily build canary)
    _pkl_path = pathlib.Path(__file__).parent.parent / "model_data.pkl"
    if _pkl_path.exists():
        model_build_age_hours = round((time.time() - _pkl_path.stat().st_mtime) / 3600, 1)
    else:
        model_build_age_hours = -1
    result = {
        "status": "ok",
        "model": M is not None,
        "price_age_s": round(price_age),
        "cache_age_days": round(cache_age_days, 1),
        "cache_warn_45d": cache_age_days > 45,
        "cache_stale_90d": cache_age_days > 90,
        "mc_cache": bool(_CACHE),
        "markov": _HAS_MARKOV,
        "btcpay": btcpay._HAS_BTCPAY,
        "block_map_loaded": block_map_loaded,
        "model_build_age_hours": model_build_age_hours,
        "model_build_stale_72h": model_build_age_hours > 72,
        "resqr_bands": {
            "available": _HAS_RESQR,
            "model_count": len(getattr(M, "resqr_coefs", {}) or {}),
            "build_ts": getattr(M, "resqr_build_ts", None),
        },
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
        # ETag based on callback-graph hash — helps revalidation on iOS Safari
        if path == '/_dash-dependencies':
            response.headers['ETag'] = f'"{_DEPLOY_FP}"'
    elif path.startswith('/_dash-component-suites/'):
        response.headers['Cache-Control'] = 'public, max-age=31536000, immutable'

    # ── Security headers ─────────────────────────────────────────────────
    response.headers['Referrer-Policy'] = 'same-origin'
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
        # Onion stays strict — no Google Fonts (fingerprint surface). Tor
        # users see the OS fallback stack defined in style.css.
        _style_src = "'self' 'unsafe-inline'"
        _font_src = "'self'"
    else:
        # https://dash-version.plotly.com: Plotly.js 6.x fires a background
        # version-check on every page load. Without this whitelist, Chrome
        # logs a CSPViolation + a cascading JS error on every visit. Purely
        # cosmetic — the block would be fine — but the errors crowd out
        # real errors in DevTools. Onion CSP does NOT get this exception:
        # clearnet version-check is a fingerprinting surface we'd rather not
        # expose to Tor users.
        _connect = ("'self' https://mempool.space wss://mempool.space"
                    " https://blockstream.info"
                    " https://dash-version.plotly.com:8080")
        # Google Fonts (DM Serif Display + Inter) loaded from layout head.
        _style_src = "'self' 'unsafe-inline' https://fonts.googleapis.com"
        _font_src = "'self' https://fonts.gstatic.com"
    # Static pages set their own CSP (allows MathJax CDN) — don't overwrite
    if 'Content-Security-Policy' not in response.headers:
        response.headers['Content-Security-Policy'] = "; ".join([
            "default-src 'self'",
            "script-src 'self' 'unsafe-inline'",
            f"style-src {_style_src}",
            f"frame-src {_frame_src}",
            "frame-ancestors 'none'",
            "base-uri 'self'",
            "img-src 'self' data: blob:",
            f"connect-src {_connect}",
            f"font-src {_font_src}",
        ])
    return response

# ── populate shared context ──────────────────────────────────────────────────
import _app_ctx
_app_ctx.M = M
_boot_mark("_app_ctx.M set (post init_models)")
_app_ctx.app = app
_app_ctx.server = server

# ── register price models ──────────────────────────────────────────────────
_app_ctx.PRICE_MODELS["bub"] = BubbleModel(M)
_app_ctx.PRICE_MODELS["qr"] = QuantileRegressionModel(M)
_app_ctx.PRICE_MODELS["pl"]  = PowerLawModel(
    M.ols_intercept, M.ols_slope, M.price_years, M.price_prices,
    M.genesis, M.QR_QUANTILES)
_app_ctx.PRICE_MODELS["lppl"] = LPPLModel(M.price_years, M.price_prices, M.QR_QUANTILES)
_app_ctx.PRICE_MODELS["lp2"] = LPPL2Model(M.price_years, M.price_prices, M.QR_QUANTILES)
_app_ctx.PRICE_MODELS["lp3"] = LPPL3Model(M.price_years, M.price_prices, M.QR_QUANTILES)
_app_ctx.PRICE_MODELS["lp4"] = LPPL4Model(M.price_years, M.price_prices, M.QR_QUANTILES)
_app_ctx.PRICE_MODELS["lppl_w"] = LPPLModelW(M.price_years, M.price_prices, M.QR_QUANTILES)
_app_ctx.PRICE_MODELS["lp2_w"] = LPPL2ModelW(M.price_years, M.price_prices, M.QR_QUANTILES)
_app_ctx.PRICE_MODELS["lp3_w"] = LPPL3ModelW(M.price_years, M.price_prices, M.QR_QUANTILES)
_app_ctx.PRICE_MODELS["lp4_w"] = LPPL4ModelW(M.price_years, M.price_prices, M.QR_QUANTILES)
_app_ctx.PRICE_MODELS["lp4_n13"] = LPPL4ModelN13(M.price_years, M.price_prices, M.QR_QUANTILES)
_app_ctx.PRICE_MODELS["lp4_w_n13"] = LPPL4ModelWN13(M.price_years, M.price_prices, M.QR_QUANTILES)
_app_ctx.PRICE_MODELS["linppl"] = LinPPLModel(M.price_years, M.price_prices, M.QR_QUANTILES)
_app_ctx.PRICE_MODELS["hybppl"] = HybPPLModel(M.price_years, M.price_prices, M.QR_QUANTILES)
_app_ctx.PRICE_MODELS["hybppl_dd"] = HybPPLDDModel(M.price_years, M.price_prices, M.QR_QUANTILES)
_app_ctx.PRICE_MODELS["hyb2l"] = Hyb2LModel(M.price_years, M.price_prices, M.QR_QUANTILES)
_app_ctx.PRICE_MODELS["hyb2c"] = Hyb2CModel(M.price_years, M.price_prices, M.QR_QUANTILES)
_app_ctx.PRICE_MODELS["hyb2b"] = Hyb2BModel(M.price_years, M.price_prices, M.QR_QUANTILES)
_app_ctx.PRICE_MODELS["hyb4d"] = Hyb4DModel(M.price_years, M.price_prices, M.QR_QUANTILES)
# ── HybPPL config family (36 pre-fitted variants) ───────────────────────
for _cfg_key in _HYBPPL_CONFIG_PARAMS:
    _app_ctx.PRICE_MODELS[_cfg_key] = HybPPLConfigModel(
        _cfg_key, M.price_years, M.price_prices, M.QR_QUANTILES)
# ── EPPL config family (36 pre-fitted variants) ──────────────────────────
for _ecfg_key in _EPPL_CONFIG_PARAMS:
    _app_ctx.PRICE_MODELS[_ecfg_key] = EPPLConfigModel(
        _ecfg_key, M.price_years, M.price_prices, M.QR_QUANTILES)
_app_ctx.PRICE_MODELS["pca"] = PCAModel(
    M.price_years, M.price_prices, M.QR_QUANTILES,
    source_models=_app_ctx.PRICE_MODELS)  # must come after HybPPL family
_app_ctx.PRICE_MODELS["grdy"] = GreedyModel(M.price_years, M.price_prices, M.QR_QUANTILES)
_app_ctx.PRICE_MODELS["eppl"] = EntropyPPLModel(M.price_years, M.price_prices, M.QR_QUANTILES)
_app_ctx.PRICE_MODELS["exp"] = ExponentialModel(M.price_years, M.price_prices, M.QR_QUANTILES)
_app_ctx.PRICE_MODELS["gomp"] = GompertzModel(M.price_years, M.price_prices, M.QR_QUANTILES)
_app_ctx.PRICE_MODELS["bpl"] = BrokenPowerLawModel(M.price_years, M.price_prices, M.QR_QUANTILES)
_app_ctx.PRICE_MODELS["plo"] = OffsetPowerLawModel(M.price_years, M.price_prices, M.QR_QUANTILES)
_app_ctx.PRICE_MODELS["sexp"] = StretchedExponentialModel(M.price_years, M.price_prices, M.QR_QUANTILES)
_app_ctx.PRICE_MODELS["logi"] = LogisticSCurveModel(M.price_years, M.price_prices, M.QR_QUANTILES)
_app_ctx.PRICE_MODELS["spl"] = SaturatingPowerLawModel(M.price_years, M.price_prices, M.QR_QUANTILES)
# Actual block heights so S2F stock/flow + halving timing track the real chain
# (BitcoinBlocksDaily.csv, kept equal-length with prices by the daily update).
# Future dates extrapolate at 144 blocks/day inside the model. Falls back to the
# idealized schedule if the CSV is missing/unreadable.
_s2f_block_years = _s2f_block_heights = None
_block_csv = Path(__file__).parent.parent / "BitcoinBlocksDaily.csv"
if _block_csv.exists():
    try:
        import pandas as _pd
        _bdf = _pd.read_csv(_block_csv)
        _s2f_block_years = (
            (_pd.to_datetime(_bdf["date"], format="mixed") - M.genesis)
            .dt.days.to_numpy(dtype=float) / 365.25)
        _s2f_block_heights = _bdf["blockheight"].to_numpy(dtype=float)
    except Exception:
        _s2f_block_years = _s2f_block_heights = None

_app_ctx.PRICE_MODELS["s2f"] = S2FModel(
    M.price_years, M.price_prices, M.genesis,
    block_years=_s2f_block_years, block_heights=_s2f_block_heights)
_app_ctx.PRICE_MODELS["s2f_inst"] = S2FModel(
    M.price_years, M.price_prices, M.genesis,
    block_years=_s2f_block_years, block_heights=_s2f_block_heights,
    flow_mode="instantaneous")
# ── Empirical Floor (conditional — only if pkl exists) ────────────────
_ef_pkl = Path(__file__).parent.parent / "model_data_ef.pkl"
if _ef_pkl.exists():
    _app_ctx.PRICE_MODELS["ef"] = EmpiricalFloorModel(str(_ef_pkl))
_app_ctx.DEFAULT_MODEL = _app_ctx.PRICE_MODELS["bub"]

# ── bind resqr coefficient bundles onto their models (Task 4) ─────────────
# Models without a bundle silently fall back to constant σ at runtime.
_resqr_bound = 0
for _rq_key, _rq_bundle in (M.resqr_coefs or {}).items():
    _rq_model = _app_ctx.PRICE_MODELS.get(_rq_key)
    if _rq_model is None:
        continue
    _rq_model._resqr = {
        "sorted_qs": _rq_bundle["sorted_qs"],
        "coef_matrix": _rq_bundle["coef_matrix"],
        "knots": tuple(_rq_bundle.get("knots", M.resqr_knots or (3.0, 6.0, 9.0, 12.0))),
    }
    _resqr_bound += 1
_HAS_RESQR = _resqr_bound > 0
_app_ctx._HAS_RESQR = _HAS_RESQR
print(f"[resqr] bound {_resqr_bound} model bundles  _HAS_RESQR={_HAS_RESQR}")
_boot_mark("model registration + resqr bind done")

# ── compute per-quantile R² for all models ───────────────────────────────
# Used only by the Model Info accordion (/9.N). Costs ~2.3s at boot because
# there are ~80 registered models (incl. 72 config-family variants) × ~7
# quantiles × ~3700 price observations each. Move to a daemon thread so
# the module finishes sooner; the first /8 visitor might see missing R²
# values during the first ~2-3s post-restart, but the numbers are a
# diagnostic aid, not critical.
import numpy as np
from btc_core import compute_model_r2, _compute_log_r2
def _background_r2_compute() -> None:
    try:
        for _mdl in _app_ctx.PRICE_MODELS.values():
            compute_model_r2(_mdl, M.price_years, M.price_prices)
    except Exception as e:
        logging.getLogger(__name__).warning("background R² compute failed: %s", e)
# Tests inspect r2_per_quantile synchronously at import time; DEV runs as
# single-process where async complexity isn't worth it. Background only
# in the gunicorn prod path.
if os.environ.get("DEV") != "1" and os.environ.get("TESTING") != "1":
    import threading as _threading
    _threading.Thread(target=_background_r2_compute,
                      daemon=True, name="qs-bg-r2").start()
else:
    _background_r2_compute()
_boot_mark("R² compute backgrounded")

# OLS R²
_ols_pred = 10 ** (M.ols_intercept + M.ols_slope * np.log10(
    np.maximum(M.price_years[M.price_years >= 1.0], 0.1)))
M.ols_r2 = _compute_log_r2(M.price_prices[M.price_years >= 1.0], _ols_pred)

# ── Thermal color pipeline removed (2026-04-12). Quantile traces and bands
# now use model-specific colors (via _get_model_color + quantile_opacity).
# The quantile panel sidebar dots use DIM_TEXT with opacity-fading.
# The _thermal_color / _build_thermal_colors machinery in figures/common.py
# is retained as dead code in case the heatmap or other future features need it.

# ── Pre-compute auto-Y envelope grids for clientside auto_bubble_yrange ──────
# For each model: log10(price) at Q_min and Q_max on a 100-point t-grid.
# Stored in _app_ctx.AUTO_Y_GRID, injected into a dcc.Store at layout time.
_auto_y_t = np.linspace(0.5, 72.0, 100)
_auto_y_grid = {"t": [round(float(v), 4) for v in _auto_y_t],
                "genesis_epoch_days": int((M.genesis - __import__('pandas').Timestamp("1970-01-01")).days),
                "models": {}}
for _key, _mdl in _app_ctx.PRICE_MODELS.items():
    if not _mdl.quantized:
        # Non-quantized (S2F): store median only
        try:
            _p50 = np.log10(np.maximum(
                np.asarray(_mdl.price_at(0.5, _auto_y_t), float), 1e-10))
            _auto_y_grid["models"][_key] = {
                "lo": [round(float(v), 4) for v in _p50],
                "hi": [round(float(v), 4) for v in _p50],
            }
        except Exception:
            pass
        continue
    _qs = sorted(_mdl.fits.keys())
    _q_lo, _q_hi = _qs[0], _qs[-1]
    try:
        _lo = np.log10(np.maximum(
            np.asarray(_mdl.price_at(_q_lo, _auto_y_t), float), 1e-10))
        _hi = np.log10(np.maximum(
            np.asarray(_mdl.price_at(_q_hi, _auto_y_t), float), 1e-10))
        _auto_y_grid["models"][_key] = {
            "lo": [round(float(v), 4) for v in _lo],
            "hi": [round(float(v), 4) for v in _hi],
        }
    except Exception:
        pass
_app_ctx.AUTO_Y_GRID = _auto_y_grid
_boot_mark("auto_y grid built")

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
# Heatmap entry-percentile default depends on a live BTC price fetch
# (Binance → CoinGecko fallbacks, 5s timeout each). That fetch historically
# cost ~3-4s of boot time. Start with the safe fallback (50%) and background
# the fetch so the module completes sooner. The update_price_ticker callback
# refreshes _HM_ENTRY_Q_DEFAULT every 20 min at runtime regardless, so the
# brief "stale default" window affects only the first ~5-10s after restart.
_app_ctx._HM_ENTRY_Q_DEFAULT = 50.0
def _background_startup_ticker_fetch() -> None:
    try:
        _app_ctx._HM_ENTRY_Q_DEFAULT = _startup_heatmap_defaults()
    except Exception as e:
        logging.getLogger(__name__).warning("startup price fetch failed: %s", e)
if os.environ.get("DEV") != "1":
    import threading as _threading
    _threading.Thread(target=_background_startup_ticker_fetch,
                      daemon=True, name="qs-bg-ticker").start()
else:
    # DEV: run inline (single process, no async complexity worth)
    _background_startup_ticker_fetch()
_boot_mark("backgrounded startup ticker fetch")

# ── import layout (sets app.layout) and callbacks (registers @callbacks) ─────
import snapshot   # noqa: F401 — defines _CHECKLIST_OPTIONS using _app_ctx
import layout     # noqa: F401 — sets app.layout
import callbacks  # noqa: F401 — registers all callbacks

# ── Deploy fingerprint (callback-graph hash for cache busting) ────────────────
# Compute a short hash of the callback list at startup. Used as ETag on
# /_dash-dependencies so stale cached responses are invalidated, even on
# iOS Safari which sometimes ignores Cache-Control: no-store.
import hashlib as _hl_deploy
_DEPLOY_FP = _hl_deploy.md5(repr(app._callback_list).encode()).hexdigest()[:12]
del _hl_deploy

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
    # Mobile variant: SC callback injects is_mobile from viewport-width;
    # prewarming both so mobile users also hit L1 on first visit.
    sc_mobile = dict(sc)
    sc_mobile["is_mobile"] = True
    _entries.append(("sc", sc_mobile))

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

import os as _os

# ── Prewarm split: critical vs. background ──────────────────────────────
# `_prewarm_caches` stays on the critical path because it populates the
# per-worker `_pinned` dict that `_serve_layout` reads to pre-inject
# figures into the initial HTML (the zero-callback tab-switch feature).
# If Redis L0 has entries from a prior run, this is ~0.5s; only the
# post-FLUSHDB case takes the full ~4s.
if _os.environ.get("DEV") != "1":
    _boot_mark("_prewarm_caches start")
    _prewarm_caches()
    _boot_mark("_prewarm_caches end")
    from utils import _log_cache_stats
    _log_cache_stats()
else:
    logging.getLogger(__name__).info("DEV mode — skipping prewarm (first request per tab will be slower)")

# Non-critical background warm: static analysis pages + default transition
# matrix. Neither blocks a user's first tab render; the static pages are
# served from their own Flask route once rendered to disk, and the default
# transition matrix is only consumed by the MC overlay (per-worker cache,
# lazy-built). We kick them off on a daemon thread so the gunicorn master
# finishes module-level work and forks workers sooner.
def _background_warm_noncritical() -> None:
    try:
        from static_pages import render_static_pages
        render_static_pages()
        if _HAS_MARKOV:
            _get_transition_matrix(M, 5, 30, [2010, date.today().year])
        logging.getLogger(__name__).info("Background warm (static pages + MC matrix) done")
    except Exception as e:
        logging.getLogger(__name__).warning("Background warm failed: %s", e)

if _os.environ.get("DEV") != "1":
    import threading as _threading
    _threading.Thread(target=_background_warm_noncritical,
                      daemon=True, name="qs-bg-warm").start()
else:
    # DEV: single-process, no harm in running inline.
    _background_warm_noncritical()

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
def _skip_source_maps():
    # Browsers with DevTools open fetch the adjacent .js.map source map for each
    # Dash component bundle. Dash doesn't register those paths, so
    # serve_component_suites raises DependencyException, which Flask logs as an
    # ERROR-level Traceback. The prod health check greps the error log for
    # 'Traceback' and then fires a daily false-alarm popup. Source maps aren't
    # served in prod anyway — return a clean 404 before dispatch so no exception
    # is raised or logged.
    from flask import request, abort
    if request.path.endswith(".js.map"):
        abort(404)


@server.before_request
def _trigger_mc_prewarm():
    global _mc_prewarm_triggered
    if not _mc_prewarm_triggered and _HAS_MARKOV:
        _mc_prewarm_triggered = True
        import threading
        threading.Thread(target=_prewarm_mc_caches, daemon=True,
                         name="mc-prewarm").start()

_boot_mark("module load complete (gunicorn will now fork workers)")

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
