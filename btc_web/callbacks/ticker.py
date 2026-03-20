"""Live BTC price ticker callback."""

from dash import html, Input, Output, callback, no_update

import _app_ctx
from btc_core import fmt_price, today_t, _find_lot_percentile
from utils import _fetch_btc_price


# ══════════════════════════════════════════════════════════════════════════════
# Callback — live BTC price ticker (Binance, refreshes every 5 min)
# ══════════════════════════════════════════════════════════════════════════════

@callback(
    Output("price-ticker",        "children"),
    Output("price-ticker-mobile", "children"),
    Output("btc-price-store",     "data"),
    Output("hm-entry-q",          "value", allow_duplicate=True),
    Output("hm-mc-entry-q",       "value", allow_duplicate=True),
    Output("dca-mc-entry-q",      "value", allow_duplicate=True),
    Output("price-sparkline",     "children"),
    Input("price-interval", "n_intervals"),
    Input("ticker-mode-store",    "data"),
    prevent_initial_call="initial_duplicate",
)
def update_price_ticker(_, mode):
    price = _fetch_btc_price()
    if price is None:
        return "\u20bf \u2014", "\u20bf \u2014", no_update, no_update, no_update, no_update, ""
    pct = _find_lot_percentile(today_t(_app_ctx.M.genesis), price, _app_ctx.M.qr_fits)
    pct_str = f"Q{pct*100:.1f}%" if pct is not None else "\u2014"
    pct_val = round(pct * 100, 1) if pct is not None else no_update
    # Snap to nearest 10% for cache-aligned dropdowns (hm-mc, dca-mc)
    snapped_pct = max(10, min(90, round(pct * 10) * 10)) if pct is not None else no_update
    # Ticker mode: sats/$ or USD (both show percentile)
    if mode == "sats":
        sats = round(1e8 / price) if price > 0 else 0
        txt = f"{sats:,} sats/$  \u00b7  {pct_str}"
        txt_m = f"{sats:,}s/$\u00b7{pct_str}"
    else:
        txt = f"\u20bf {fmt_price(price)}  \u00b7  {pct_str}"
        txt_m = f"\u20bf{fmt_price(price)}\u00b7{pct_str}"
    # Sparkline SVG (24h) — data URI image
    from utils import _fetch_sparkline_svg
    spark_uri = _fetch_sparkline_svg()
    spark = html.Img(src=spark_uri, height="18",
                     style={"verticalAlign": "middle"}) if spark_uri else ""
    return txt, txt_m, price, pct_val, snapped_pct, snapped_pct, spark
