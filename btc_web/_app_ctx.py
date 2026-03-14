"""Shared application context — populated by app.py before other modules load.

This module avoids circular imports: app.py creates the Dash app and model,
stores them here, then imports layout/callbacks/etc. which read from here.

Static constants (FREQ_PPY, FREQ_STEP_DAYS) are defined here so both
figures.py and mc_overlay.py can import them without circular dependencies.
"""

# ── Static constants (no population needed) ──────────────────────────────────
FREQ_PPY = {"Daily": 365, "Weekly": 52, "Monthly": 12, "Quarterly": 4, "Annually": 1}
FREQ_STEP_DAYS = {"Daily": 1, "Weekly": 7, "Monthly": 30, "Quarterly": 91, "Annually": 365}
ANNOT_STAGGER_Y = [-20, -33, -46]  # annotation y-offsets for staggering (~1 font-height apart)
BTC_ORANGE = "#f7931a"
FONT_LEGEND = 10              # legend / small info text

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
M = None                   # ModelData instance (kept for transition)
model = None               # PriceModel (QRBubbleModel wrapping M)
history = None             # PriceHistory (genesis, price arrays)
theme = None               # ThemeConfig (colors, sizes, CAGR constants)
models = {}                # registry: short_name → PriceModel
app = None                 # dash.Dash instance
server = None              # Flask server (= app.server)
_HAS_MARKOV = False
_HAS_BTCPAY = False        # set True by app.py if BTCPay env vars present
_ALL_QS = []               # filtered QR quantiles (0.001–0.999)
_DEF_QS = []               # default quantile subset
_HM_ENTRY_Q_DEFAULT = 50.0 # live heatmap entry percentile
