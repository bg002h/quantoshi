"""Shared application context — populated by app.py before other modules load.

This module avoids circular imports: app.py creates the Dash app and model,
stores them here, then imports layout/callbacks/etc. which read from here.

Static constants (FREQ_PPY, FREQ_STEP_DAYS) are defined here so both
figures.py and mc_overlay.py can import them without circular dependencies.
"""

import math

from colors import (
    BTC_ORANGE,
    MODEL_TRACE_COLORS,
    PALETTES,
)

# ── Float quantization (shared by utils.py and figures/common.py) ────────────
def _q3(x):
    """Round a number to 3 significant figures."""
    if x is None or x == 0:
        return x
    exp = math.floor(math.log10(abs(x)))
    factor = 10 ** (exp - 2)
    return round(x / factor) * factor

# ── Static constants (no population needed) ──────────────────────────────────
FREQ_PPY = {"Daily": 365, "Weekly": 52, "Monthly": 12, "Quarterly": 4, "Annually": 1}
FREQ_LABEL = {"Daily": "/day", "Weekly": "/wk", "Monthly": "/mo", "Quarterly": "/qtr", "Annually": "/yr"}
FREQ_STEP_DAYS = {"Daily": 1, "Weekly": 7, "Monthly": 30, "Quarterly": 91, "Annually": 365}
ANNOT_STAGGER_Y = [-20, -33, -46, -59, -72]  # annotation y-offsets for staggering (~1 font-height apart)
FONT_LEGEND = 10              # legend / small info text
# Fallback opacity for the ghost Q50% trace shown when no quantile bands
# are selected in the Projection Quantiles panel. Referenced by figures/
# bubble.py (rendering) and layout/bubble.py (help-text hint).
FALLBACK_Q50_OPACITY = 0.25
MODEL_SENTINELS = frozenset({"mc", "bub"})  # reserved keys in model-show checklists

# LPPL family variants managed by dedicated LPPL config panel (bubble tab)
# Hidden from the standard Display Models checklist on tab 1 only;
# still available for programmatic access + other tabs.
LPPL_FAMILY_HIDDEN_FROM_BUBBLE = frozenset({
    "lppl_w", "lp2_w", "lp3_w", "lp4_w",
    "lp4_n13", "lp4_w_n13",
})

# HybPPL named variants hidden from bubble Display Models —
# managed by the HybPPL family config panel instead.
HYBPPL_FAMILY_HIDDEN = frozenset({
    "hybppl", "hybppl_dd", "hyb2l", "hyb2c", "hyb2b", "hyb4d",
    "linppl",
})

# Per-model trace colors — high-contrast, colorblind-safe, one color per model.
# Used when shade bands are active so traces stand out against any band color.
# Designed for luminance variation (readable without color vision).
# Component decomposition — family dropdown options and trace palette.
# The "lppl" family is resolved at render time via the LPPL config panel.
DECOMP_FAMILIES = {
    "bub":       "BM",
    "ef":        "EF",
    "lppl":      "LPPL (family)",
    "linppl":    "LinPPL",
    "hybppl":    "HybPPL",
    "hybppl_dd":    "HybPPL (DD)",
    "hyb2l":        "HybPPL +2L",
    "hyb2c":        "HybPPL +2C",
    "hyb2b":        "HybPPL +2B",
    "pca":          "PCA",
    "grdy":         "Greedy Select",
    "eppl":         "Entropy PPL",
    "hyb4d":        "HybPPL 4D",
    "hybppl_cfg_a": "HybPPL (A)",
    "hybppl_cfg_b": "HybPPL (B)",
    "eppl_cfg_a":   "Entropy PPL (A)",
    "eppl_cfg_b":   "Entropy PPL (B)",
}

# DECOMP_COLORS — derived view from per-palette decomp_colors keys
DECOMP_COLORS = {pkey: PALETTES[pkey]["decomp_colors"] for pkey in PALETTES}

# DECOMP_SUM_COLOR — derived view from per-palette decomp_sum_color keys
DECOMP_SUM_COLOR = {pkey: PALETTES[pkey]["decomp_sum_color"] for pkey in PALETTES}

PALETTE_LABELS = {
    "default": "Default",
    "cb-brian": "Deuteranomaly",
    "cb-rg": "Colorblind (R-G)",
    "cb-full": "Colorblind (Full)",
}

# ── Dollar / loan defaults ───────────────────────────────────────────────────
MAX_USD = 4_294_967_295        # uint32 max — clamp for dollar amount inputs
LOT_DEFAULT_PRICE = 69_420     # Stack Tracker default lot price ($)

# ── Unfairly Cheap Line — unique two-point power law floor ───────────────────
# Pinned by Sept 21 2015 ($229) and Jan 1 2023 ($16,905) — only 2 breaches in
# 16 years.  Feasible slope region is 0.0026 wide — effectively a unique line.
UCL_SLOPE     = 5.510508
UCL_INTERCEPT = -1.989444

# ── Shared financial math ────────────────────────────────────────────────────

def _compute_sc_loan(principal, amount, r, term_periods, loan_type):
    """Cap principal so payment ≤ DCA amount, compute loan payment.

    Returns (principal, pmt, capped).
    """
    capped = False
    if r > 0:
        if loan_type == "amortizing":
            # PV of annuity formula: max loan where periodic payment = DCA amount
            max_principal = amount * (1 - (1 + r) ** (-term_periods)) / r
        else:
            # Interest-only: max loan where interest payment = DCA amount
            max_principal = amount / r
        if principal > max_principal:
            principal = max_principal
            capped = True
    # Standard amortizing payment formula (PMT = PV * r / (1 - (1+r)^-n))
    if loan_type == "amortizing":
        pmt = principal * r / (1 - (1 + r) ** (-term_periods)) if r > 0 else principal / term_periods
    else:
        pmt = principal * r
    return principal, pmt, capped


# ── Singleton capability flags (evaluated ONCE at import time) ────────────────
# Every module reads these from _app_ctx instead of doing its own try/except.

try:
    from markov import build_transition_matrix  # noqa: F401
    _HAS_MARKOV = True
except ImportError:
    _HAS_MARKOV = False

try:
    from celery import Celery  # noqa: F401
    _HAS_CELERY = True
except ImportError:
    _HAS_CELERY = False

try:
    import socket as _socket
    # Fast pre-check: can we even connect to Redis port? (avoids slow library timeout)
    _sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    _sock.settimeout(0.2)
    _sock.connect(('localhost', 6379))
    _sock.close()
    # Port is open — now create the Redis client
    import redis as _redis_mod
    _REDIS = _redis_mod.Redis(host='localhost', port=6379, db=0,
                               socket_timeout=1, socket_connect_timeout=1)
    _REDIS.ping()
    _HAS_REDIS = True
except Exception:
    _REDIS = None
    _HAS_REDIS = False


def redis_available() -> bool:
    """Check if Redis is connected."""
    return _HAS_REDIS


def redis_client():
    """Return the shared Redis client (or None)."""
    return _REDIS


# Model fingerprint: changes when model_data.pkl is regenerated
import hashlib as _hashlib
import os as _os

def _compute_model_fingerprint() -> str:
    for path in ("model_data.pkl", "btc_app/model_data.pkl", "archive/btc_app/model_data.pkl"):
        if _os.path.exists(path):
            st = _os.stat(path)
            return _hashlib.md5(f"{st.st_mtime}:{st.st_size}".encode()).hexdigest()[:8]
    return "unknown"

_MODEL_FP = _compute_model_fingerprint()

# BTCPay (evaluated after env vars are loaded)
_HAS_BTCPAY = False  # overwritten by app.py from btcpay._HAS_BTCPAY

# ── Dynamic state (populated by app.py at startup) ──────────────────────────
M = None                   # ModelData instance
PRICE_MODELS = {}          # {"bub": BubbleModel, "pl": PowerLawModel, ...}
DEFAULT_MODEL = None       # set to PRICE_MODELS["bub"] by app.py
app = None                 # dash.Dash instance
server = None              # Flask server (= app.server)
_ALL_QS = []               # filtered QR quantiles (0.001–0.999)
_DEF_QS = []               # default quantile subset
_HM_ENTRY_Q_DEFAULT = 50.0 # live heatmap entry percentile
