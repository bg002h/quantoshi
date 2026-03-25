"""Shared application context — populated by app.py before other modules load.

This module avoids circular imports: app.py creates the Dash app and model,
stores them here, then imports layout/callbacks/etc. which read from here.

Static constants (FREQ_PPY, FREQ_STEP_DAYS) are defined here so both
figures.py and mc_overlay.py can import them without circular dependencies.
"""

# ── Static constants (no population needed) ──────────────────────────────────
FREQ_PPY = {"Daily": 365, "Weekly": 52, "Monthly": 12, "Quarterly": 4, "Annually": 1}
FREQ_STEP_DAYS = {"Daily": 1, "Weekly": 7, "Monthly": 30, "Quarterly": 91, "Annually": 365}
ANNOT_STAGGER_Y = [-20, -33, -46, -59, -72]  # annotation y-offsets for staggering (~1 font-height apart)
BTC_ORANGE = "#f7931a"
FONT_LEGEND = 10              # legend / small info text
MODEL_SENTINELS = frozenset({"mc", "bub"})  # reserved keys in model-show checklists

# Per-model trace colors — high-contrast, colorblind-safe, one color per model.
# Used when shade bands are active so traces stand out against any band color.
# Designed for luminance variation (readable without color vision).
MODEL_TRACE_COLORS = {
    "bub": "#000000",   # black — primary, maximum contrast on light bg
    "qr":  "#FFD700",   # gold — warm, high luminance
    "pl":  "#00E5FF",   # electric cyan — cool, very high luminance
    "lppl":"#FF6D00",   # deep orange — warm, medium-high luminance
    "exp": "#82B1FF",   # soft blue — cool, medium luminance
    "ef":  "#FF80AB",   # pink — warm, medium luminance
    "s2f": "#B0BEC5",   # blue-grey — neutral
}

# ── Color palettes (default + colorblind-safe alternatives) ──────────────
PALETTES = {
    "default": {
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
    "cb-rg": "Colorblind (R-G)",
    "cb-full": "Colorblind (Full)",
}

# ── Dollar / loan defaults (shared across layout, callbacks, app prewarm) ────
MAX_USD = 4_294_967_295        # uint32 max — clamp for dollar amount inputs
SC_DEFAULT_RATE = 13.0         # Stack-celerator default annual interest rate (%)
SC_DEFAULT_PRICE = 80_000      # Stack-celerator default custom entry price ($)
SC_DEFAULT_TAX = 33            # capital gains tax rate (%)
SC_DEFAULT_TERM = 12           # loan term (months)
SC_DEFAULT_START_YR = 2033     # Supercharger default withdrawal start year
SC_DEFAULT_WD = 5000           # Supercharger default withdrawal ($/period)
SC_DEFAULT_END_YR = 2075       # Supercharger default end year
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


# ── Dynamic state (populated by app.py at startup) ──────────────────────────
M = None                   # ModelData instance
PRICE_MODELS = {}          # {"bub": BubbleModel, "pl": PowerLawModel, ...}
DEFAULT_MODEL = None       # set to PRICE_MODELS["bub"] by app.py
app = None                 # dash.Dash instance
server = None              # Flask server (= app.server)
_HAS_MARKOV = False
_HAS_BTCPAY = False        # set True by app.py if BTCPay env vars present
_ALL_QS = []               # filtered QR quantiles (0.001–0.999)
_DEF_QS = []               # default quantile subset
_HM_ENTRY_Q_DEFAULT = 50.0 # live heatmap entry percentile
