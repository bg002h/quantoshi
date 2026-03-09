"""BTCPay Server Greenfield API client for Quantoshi MC payment gating.

Handles invoice creation, status checking, payment token generation,
and pricing logic.  All BTCPay communication is isolated to this module.

Configuration via environment variables:
    BTCPAY_URL          — BTCPay server URL (typically a .onion address)
    BTCPAY_API_KEY      — Greenfield API key (create+view invoice permissions)
    BTCPAY_STORE_ID     — BTCPay store identifier
    BTCPAY_SOCKS_PROXY  — SOCKS5 proxy for Tor (default: socks5h://127.0.0.1:9050)
    BTCPAY_HMAC_SECRET  — HMAC key for payment tokens (defaults to API key)
"""

import os
import hmac
import hashlib
import base64
import logging
from datetime import date

import requests

log = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────

BTCPAY_URL      = os.environ.get("BTCPAY_URL", "")
BTCPAY_API_KEY  = os.environ.get("BTCPAY_API_KEY", "")
BTCPAY_STORE_ID = os.environ.get("BTCPAY_STORE_ID", "")
SOCKS_PROXY     = os.environ.get("BTCPAY_SOCKS_PROXY", "socks5h://127.0.0.1:9050")
HMAC_SECRET     = os.environ.get("BTCPAY_HMAC_SECRET", BTCPAY_API_KEY)

_HAS_BTCPAY = bool(BTCPAY_URL and BTCPAY_API_KEY and BTCPAY_STORE_ID)

if _HAS_BTCPAY:
    log.info("BTCPay configured: %s (store %s…)", BTCPAY_URL[:40], BTCPAY_STORE_ID[:8])
else:
    log.info("BTCPay not configured — MC simulations are free (dev mode)")

# ── Pricing ──────────────────────────────────────────────────────────────────

# {horizon_years: (cached_sats, live_sats)}
_PRICE_BASE = {10: (100, 500), 20: (200, 1000), 30: (300, 1500), 40: (400, 2000)}
_HM_DISCOUNT = 0.5  # heatmap pays half

# Free tier: 10yr horizon + tab-specific default (start_yr, entry_q)
from mc_cache import MC_DEFAULT_YEARS, MC_DEFAULT_ENTRY_Q, MC_DEFAULT_START_YR, \
    CACHED_START_YRS, MC_BINS, MC_SIMS, MC_FREQ

# Each tuple = (years, start_yr, entry_q) that qualifies for free tier
_FREE_TIER_DEFAULTS = {
    (10, 2026, 10),   # heatmap
    (10, 2026, 50),   # DCA
    (10, 2031, 50),   # retire
}


def compute_price(tab: str, mc_years: int, is_cached: bool) -> int:
    """Return price in satoshis for a given MC simulation request."""
    cached_sats, live_sats = _PRICE_BASE.get(mc_years, _PRICE_BASE[10])
    price = cached_sats if is_cached else live_sats
    if tab == "hm":
        price = int(price * _HM_DISCOUNT)
    return price


def is_free_tier(mc_years: int, start_yr: int, entry_q: float = 0,
                 mc_bins: int = MC_BINS, mc_sims: int = MC_SIMS,
                 mc_freq: str = MC_FREQ) -> bool:
    """Check if the requested params match the free tier (no payment needed).

    Free tier requires default simulator settings (bins, sims, freq)
    in addition to matching a (years, start_yr, entry_q) tuple.
    """
    if int(mc_bins) != MC_BINS or int(mc_sims) > MC_SIMS or (mc_freq or MC_FREQ) != MC_FREQ:
        return False
    return (int(mc_years), int(start_yr), int(entry_q)) in _FREE_TIER_DEFAULTS


def is_cached_request(start_yr: int) -> bool:
    """Check if start_yr has pre-computed cache files."""
    return start_yr in CACHED_START_YRS


# ── HMAC Payment Tokens ─────────────────────────────────────────────────────

def _hmac_sign(message: str) -> str:
    """Compute HMAC-SHA256 of message, return URL-safe base64."""
    sig = hmac.new(HMAC_SECRET.encode(), message.encode(), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(sig).decode().rstrip("=")


def generate_payment_token(invoice_id: str, tab: str, mc_years: int) -> str:
    """Create a signed token proving payment was verified server-side.

    Token is scoped to tab + horizon and expires daily.
    """
    msg = f"{invoice_id}:{tab}:{mc_years}:{date.today().isoformat()}"
    return _hmac_sign(msg)


def verify_payment_token(token: str, invoice_id: str, tab: str, mc_years: int) -> bool:
    """Verify that a payment token is valid for the given params."""
    msg = f"{invoice_id}:{tab}:{mc_years}:{date.today().isoformat()}"
    expected = _hmac_sign(msg)
    return hmac.compare_digest(token, expected)


# ── BTCPay Greenfield API ───────────────────────────────────────────────────

def _session() -> requests.Session:
    """Create a requests session with auth headers and optional Tor proxy."""
    s = requests.Session()
    s.headers["Authorization"] = f"token {BTCPAY_API_KEY}"
    s.headers["Content-Type"] = "application/json"
    if ".onion" in BTCPAY_URL:
        s.proxies = {"http": SOCKS_PROXY, "https": SOCKS_PROXY}
    return s


def _api_url(path: str) -> str:
    return f"{BTCPAY_URL.rstrip('/')}/api/v1/stores/{BTCPAY_STORE_ID}{path}"


def create_invoice(tab: str, mc_years: int, is_cached: bool,
                   description: str = "") -> dict:
    """Create a BTCPay invoice via the Greenfield API.

    Returns:
        {invoice_id, checkout_url, amount_sats, status, expires_at}

    Raises:
        requests.RequestException on network/API errors.
    """
    amount_sats = compute_price(tab, mc_years, is_cached)
    if not description:
        tier = "cached" if is_cached else "live"
        description = f"MC Simulation ({mc_years}yr, {tier})"

    payload = {
        "amount": str(amount_sats),
        "currency": "SATS",
        "checkout": {
            "speedPolicy": "MediumSpeed",
            "expirationMinutes": 15,
        },
        "metadata": {
            "itemDesc": description,
            "tab": tab,
            "mc_years": mc_years,
            "is_cached": is_cached,
        },
    }

    s = _session()
    resp = s.post(_api_url("/invoices"), json=payload, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    return {
        "invoice_id": data["id"],
        "checkout_url": data.get("checkoutLink", ""),
        "amount_sats": amount_sats,
        "status": data.get("status", "New"),
        "expires_at": data.get("expirationTime", ""),
    }


def check_invoice(invoice_id: str) -> dict:
    """Check status of a BTCPay invoice.

    Returns:
        {status, paid}

    BTCPay statuses: New, Processing, Settled, Expired, Invalid.
    We consider 'Settled' and 'Processing' as paid (Processing = seen but
    not fully confirmed; MediumSpeed policy means 1-conf for on-chain).
    """
    s = _session()
    resp = s.get(_api_url(f"/invoices/{invoice_id}"), timeout=15)
    resp.raise_for_status()
    data = resp.json()
    status = data.get("status", "New")
    paid = status in ("Settled", "Processing")
    return {"status": status, "paid": paid}


def get_payment_methods(invoice_id: str) -> list[dict]:
    """Fetch payment methods for an invoice from BTCPay Greenfield API.

    Returns list of dicts, each with:
        method, destination, amount, payment_link
    """
    s = _session()
    resp = s.get(_api_url(f"/invoices/{invoice_id}/payment-methods"), timeout=15)
    resp.raise_for_status()
    result = []
    for m in resp.json():
        if not m.get("activated"):
            continue
        result.append({
            "method": m.get("paymentMethodId", ""),
            "destination": m.get("destination", ""),
            "amount": m.get("amount", "0"),
            "payment_link": m.get("paymentLink", ""),
        })
    return result


# ── QR code generation ─────────────────────────────────────────────────────

try:
    import qrcode
    import qrcode.image.svg
    _HAS_QR = True
except ImportError:
    _HAS_QR = False

def generate_qr_svg(data: str) -> str:
    """Generate a QR code as a base64 SVG data URI.

    BOLT11 invoices are uppercased for smaller QR codes (alphanumeric mode).
    Returns empty string if qrcode library is unavailable.
    """
    if not _HAS_QR or not data:
        return ""
    import io as _io
    # Uppercase BOLT11 for more compact QR (case-insensitive per spec)
    qr_data = data.upper() if data.lower().startswith("ln") else data
    qr = qrcode.QRCode(box_size=8, border=2)
    qr.add_data(qr_data)
    qr.make(fit=True)
    img = qr.make_image(image_factory=qrcode.image.svg.SvgPathImage)
    buf = _io.BytesIO()
    img.save(buf)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/svg+xml;base64,{b64}"
