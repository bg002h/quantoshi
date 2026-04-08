"""Shared application context — populated by app.py before other modules load.

This module avoids circular imports: app.py creates the Dash app and model,
stores them here, then imports layout/callbacks/etc. which read from here.

Static constants (FREQ_PPY, FREQ_STEP_DAYS) are defined here so both
figures.py and mc_overlay.py can import them without circular dependencies.
"""

import math

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
BTC_ORANGE = "#f7931a"
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
    "hyb4d":        "HybPPL 4D",
}

# 7-color decomposition palette per color scheme (cycles if model has >7 comps)
DECOMP_COLORS = {
    "default":  ["#E64A19", "#1976D2", "#388E3C", "#7B1FA2",
                 "#F57C00", "#00796B", "#5D4037"],
    "cb-brian": ["#D81B60", "#1E88E5", "#004D40", "#F4511E",
                 "#6A1B9A", "#00695C", "#3E2723"],
    "cb-rg":    ["#E69F00", "#56B4E9", "#009E73", "#F0E442",
                 "#0072B2", "#D55E00", "#CC79A7"],
    "cb-full":  ["#000000", "#505050", "#808080", "#A0A0A0",
                 "#C0C0C0", "#6A6A6A", "#303030"],
}

# Dedicated sum-trace color per palette (distinct from individual components)
DECOMP_SUM_COLOR = {
    "default":  "#000000",
    "cb-brian": "#000000",
    "cb-rg":    "#000000",
    "cb-full":  "#F5793A",
}

MODEL_TRACE_COLORS = {
    "bub": "#DAA520",   # goldenrod — matches bubble composite
    "qr":  "#B0BEC5",   # blue-grey — muted
    "pl":  "#00E5FF",   # electric cyan — cool, very high luminance
    "lppl":"#FF6D00",   # deep orange — warm, medium-high luminance
    "lp2": "#FF9F40",   # lighter orange — LPPL family variant
    "lp3": "#FFD080",   # even lighter orange — LPPL family variant 3
    "lp4": "#FFE0A0",   # palest orange — LPPL family variant 4
    "linppl": "#00B8A0", # teal — LinPPL (linear-periodic, distinct family)
    "hybppl": "#7B68EE", # medium slate blue — HybPPL (hybrid log+linear)
    "hybppl_dd": "#B39DDB", # lavender — HybPPL (DD)
    "exp": "#CE93D8",   # muted lavender — low-priority model
    "ef":  "#E8C860",   # lighter goldenrod — BM family variant
    "s2f": "#FFD700",   # gold — warm, high luminance
    "hyb2l": "#6A5ACD",  # slate blue — HybPPL +2nd log
    "hyb2c": "#20B2AA",  # light sea green — HybPPL +2nd cal
    "hyb2b": "#DB7093",  # pale violet red — HybPPL +both
    "hyb4d": "#8B6914",  # dark goldenrod — HybPPL 4D
    "pca":  "#4B0082",   # indigo — PCA model
    "gomp": "#4682B4",   # steel blue — logistic saturation
    "bpl": "#CD853F",   # peru/tan — broken power law
}

# ── Color palettes (default + colorblind-safe alternatives) ──────────────
PALETTES = {
    "default": {
        "model_colors": {
            "bub": "#FFD700", "qr": "#0055FF", "pl": "#00BB00",
            "lppl": "#EE0000", "lp2": "#FF6666", "lp3": "#FFAAAA", "lp4": "#FFCCCC", "linppl": "#00D4AA", "hybppl": "#9370DB", "hybppl_dd": "#B39DDB", "ef": "#FFE066", "exp": "#9933FF", "s2f": "#FF7700",
            "u1": "#333333",
            "hyb2l": "#6A5ACD", "hyb2c": "#20B2AA", "hyb2b": "#DB7093",
            "hyb4d": "#8B6914", "pca": "#4B0082",
            "gomp": "#4682B4", "bpl": "#CD853F",
        },
        "thermal_stops": [
            (0.001, "#0d47a1"), (0.01, "#1565c0"), (0.015, "#1976d2"),
            (0.05, "#42a5f5"), (0.10, "#80deea"), (0.25, "#b2dfdb"),
            (0.50, "#bdbdbd"), (0.75, "#ffcc80"), (0.90, "#f7931a"),
            (0.95, "#e65100"), (0.99, "#c62828"), (0.999, "#7f0000"),
        ],
        "non_quantized_model": "#8B4513",
        "delay_colors": ["#00c853", "#fdd835", "#ff9100", "#ff5252", "#b71c1c"],
        "annot_colors": ["#00a844", "#d4b12e", "#e07d00", "#d44040", "#8f1616"],
        "today_line": "#FF6600",
        "hm_c_lo": "#2166AC", "hm_c_mid1": "#F7F7F7",
        "hm_c_mid2": "#FF8C00", "hm_c_hi": "#CC1100",
        "hm_loss_text": "#ff8a80", "hm_exceptional_text": "#ffd700",
    },
    "cb-brian": {
        "model_colors": {
            "bub": "#FFD54F", "qr": "#556B2F", "pl": "#C635F5",
            "lppl": "#AD1457", "lp2": "#D81B60", "lp3": "#F06292", "lp4": "#F8BBD0", "linppl": "#006064", "hybppl": "#4527A0", "hybppl_dd": "#8E24AA", "ef": "#FFE082", "exp": "#E0E0E0", "s2f": "#777777",
            "u1": "#333333",
            "hyb2l": "#5B4AB0", "hyb2c": "#1A9A8F", "hyb2b": "#C4607A",
            "hyb4d": "#7A5B10", "pca": "#3A006F",
            "gomp": "#3B6FA0", "bpl": "#B87333",
        },
        "thermal_stops": [
            (0.001, "#0d47a1"), (0.01, "#1565c0"), (0.015, "#1976d2"),
            (0.05, "#42a5f5"), (0.10, "#80deea"), (0.25, "#b2dfdb"),
            (0.50, "#bdbdbd"), (0.75, "#ffcc80"), (0.90, "#f7931a"),
            (0.95, "#e65100"), (0.99, "#c62828"), (0.999, "#7f0000"),
        ],
        "non_quantized_model": "#8B4513",
        "delay_colors": ["#00c853", "#fdd835", "#ff9100", "#ff5252", "#b71c1c"],
        "annot_colors": ["#00a844", "#d4b12e", "#e07d00", "#d44040", "#8f1616"],
        "today_line": "#FF6600",
        "hm_c_lo": "#2166AC", "hm_c_mid1": "#F7F7F7",
        "hm_c_mid2": "#FF8C00", "hm_c_hi": "#CC1100",
        "hm_loss_text": "#ff8a80", "hm_exceptional_text": "#ffd700",
    },
    "cb-rg": {
        "model_colors": {
            "bub": "#F5793A", "qr": "#A8A8A8", "pl": "#0F2080",
            "lppl": "#85C0F9", "lp2": "#B0D8FF", "lp3": "#D4E9FF", "lp4": "#EAF4FF", "linppl": "#FFB000", "hybppl": "#D4A017", "hybppl_dd": "#ECC060", "ef": "#F5A060", "exp": "#BBBBBB", "s2f": "#F5C242",
            "hyb2l": "#7B68EE", "hyb2c": "#2E8B57", "hyb2b": "#CC6699",
            "hyb4d": "#8B7500", "pca": "#551A8B",
            "gomp": "#4169E1", "bpl": "#CC7722",
            "u1": "#333333",
        },
        "thermal_stops": [
            (0.001, "#0d47a1"), (0.01, "#1565c0"), (0.015, "#1976d2"),
            (0.05, "#56B4E9"), (0.10, "#88CCEE"), (0.25, "#AACCBB"),
            (0.50, "#BBBBBB"), (0.75, "#E69F00"), (0.90, "#D55E00"),
            (0.95, "#CC6633"), (0.99, "#882255"), (0.999, "#661155"),
        ],
        "non_quantized_model": "#CC79A7",
        "delay_colors": ["#0072B2", "#E69F00", "#CC79A7", "#AA4499", "#332288"],
        "annot_colors": ["#005B8E", "#B87E00", "#AA6088", "#883377", "#221166"],
        "today_line": "#D55E00",
        "hm_c_lo": "#2166AC", "hm_c_mid1": "#F7F7F7",
        "hm_c_mid2": "#E69F00", "hm_c_hi": "#882255",
        "hm_loss_text": "#CC79A7", "hm_exceptional_text": "#E69F00",
    },
    "cb-full": {
        "model_colors": {
            "bub": "#F0C040", "qr": "#606060", "pl": "#B0E0E6",
            "lppl": "#1A1A1A", "lp2": "#444444", "lp3": "#707070", "lp4": "#A0A0A0", "linppl": "#2A2A2A", "hybppl": "#505050", "hybppl_dd": "#989898", "ef": "#F0D870", "exp": "#909090", "s2f": "#FFE066",
            "u1": "#333333",
            "hyb2l": "#6060A0", "hyb2c": "#4A8A7A", "hyb2b": "#A06080",
            "hyb4d": "#7A7A50", "pca": "#4A4A70",
            "gomp": "#5B7FAA", "bpl": "#AA8844",
        },
        "thermal_stops": [
            (0.001, "#1a1a2e"), (0.01, "#3d1f56"), (0.015, "#6B3074"),
            (0.05, "#995588"), (0.10, "#BB7799"), (0.25, "#CCAAAA"),
            (0.50, "#BBBBBB"), (0.75, "#88BBAA"), (0.90, "#558899"),
            (0.95, "#336677"), (0.99, "#224466"), (0.999, "#112244"),
        ],
        "non_quantized_model": "#DDCC77",
        "delay_colors": ["#882255", "#CC6677", "#DDCC77", "#117733", "#332288"],
        "annot_colors": ["#661144", "#AA4455", "#BBAA55", "#0D5C28", "#221166"],
        "today_line": "#CC79A7",
        "hm_c_lo": "#882255", "hm_c_mid1": "#F7F7F7",
        "hm_c_mid2": "#44AA99", "hm_c_hi": "#004488",
        "hm_loss_text": "#CC6677", "hm_exceptional_text": "#DDCC77",
    },
}
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
