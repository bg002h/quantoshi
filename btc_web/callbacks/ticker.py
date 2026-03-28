"""Live BTC price ticker callback with multi-model percentile cycling."""

from dash import html, Input, Output, State, callback, clientside_callback, no_update

import _app_ctx
from btc_core import fmt_price, today_t, UserModel
from utils import _fetch_btc_price


# ── Model display order and colors ──────────────────────────────────────────
# Cycle order: QR → BM → PL → LPPL → Exp → EF (skip S2F — non-quantized)
_MODEL_CYCLE = ["qr", "bub", "pl", "lppl", "exp", "ef"]
_MODEL_COLORS = {
    "qr":   "#5dade2",   # sky blue
    "bub":  "#f39c12",   # amber/gold
    "pl":   "#2ecc71",   # green
    "lppl": "#e74c3c",   # red
    "exp":  "#9b59b6",   # purple
    "ef":   "#1abc9c",   # teal
}


# ══════════════════════════════════════════════════════════════════════════════
# Callback — live BTC price ticker (refreshes every 20 min)
# ══════════════════════════════════════════════════════════════════════════════

@callback(
    Output("price-ticker",        "children"),
    Output("price-ticker-mobile", "children"),
    Output("btc-price-store",     "data"),
    Output("hm-entry-q",          "value", allow_duplicate=True),
    Output("hm-mc-entry-q",       "value", allow_duplicate=True),
    Output("dca-mc-entry-q",      "value", allow_duplicate=True),
    Output("price-sparkline",     "children"),
    Output("model-percentiles-store", "data"),
    Output("ticker-pct",          "children", allow_duplicate=True),
    Output("ticker-pct",          "style",    allow_duplicate=True),
    Output("ticker-pct-mobile",   "children", allow_duplicate=True),
    Output("ticker-pct-mobile",   "style",    allow_duplicate=True),
    Output("ticker-model-idx",    "data",     allow_duplicate=True),
    Input("price-interval", "n_intervals"),
    Input("ticker-mode-store",    "data"),
    Input("user-model-store", "data"),
    prevent_initial_call="initial_duplicate",
)
def update_price_ticker(_, mode, user_model_data):
    price = _fetch_btc_price()
    if price is None:
        return ("\u20bf \u2014", "\u20bf \u2014", no_update,
                no_update, no_update, no_update, "",
                no_update, no_update, no_update, no_update, no_update, no_update)

    t = today_t(_app_ctx.M.genesis)

    # Compute percentile for every model in the cycle
    pcts = {}
    for key in _MODEL_CYCLE:
        mdl = _app_ctx.PRICE_MODELS.get(key)
        if mdl is None:
            continue
        try:
            p = mdl.find_percentile(t, price)
            pcts[key] = round(p * 100) if p is not None else None
        except Exception:
            pcts[key] = None

    # QR percentile for heatmap entry sync (always QR)
    qr_pct = pcts.get("qr")
    pct_val = round(qr_pct, 1) if qr_pct is not None else no_update
    snapped_pct = max(10, min(90, round(qr_pct / 10) * 10)) if qr_pct is not None else no_update

    # Price text (without percentile — percentile is in separate span)
    if mode == "sats":
        sats = round(1e8 / price) if price > 0 else 0
        txt = f"{sats:,} sats/$"
        txt_m = f"{sats:,}s/$"
    else:
        txt = f"\u20bf {fmt_price(price)}"
        txt_m = f"\u20bf{fmt_price(price)}"

    # Sparkline SVG
    from utils import _fetch_sparkline_svg
    spark_uri = _fetch_sparkline_svg()
    spark = html.Img(src=spark_uri, height="18",
                     style={"verticalAlign": "middle"}) if spark_uri else ""

    # Build percentiles data for clientside cycling
    model_data = []
    for key in _MODEL_CYCLE:
        mdl = _app_ctx.PRICE_MODELS.get(key)
        if mdl is None:
            continue
        p = pcts.get(key)
        label = mdl.legend_name if hasattr(mdl, 'legend_name') else key.upper()
        color = _MODEL_COLORS.get(key, "#ffffff")
        if p is not None:
            model_data.append({"key": key, "label": label, "pct": p, "color": color})

    # Append user model if defined
    if user_model_data:
        try:
            um = UserModel.from_store_dict(user_model_data)
            if um:
                um_pct = um.find_percentile(t, price)
                um_pct_int = round(um_pct * 100) if um_pct is not None else None
                if um_pct_int is not None:
                    model_data.append({"key": "u1", "label": "U1",
                                       "pct": um_pct_int, "color": "#e67e22"})
        except Exception:
            pass

    # Default display: QR (index 0)
    default = model_data[0] if model_data else {"label": "QR", "pct": 0, "color": "#5dade2"}
    pct_text = f"{default['label']}{default['pct']}%"
    pct_color = default["color"]
    pct_style_d = {"color": pct_color}
    pct_style_m = {"color": pct_color}

    return (txt, txt_m, price, pct_val, snapped_pct, snapped_pct, spark,
            model_data, pct_text, pct_style_d, pct_text, pct_style_m, 0)


# ══════════════════════════════════════════════════════════════════════════════
# Clientside callback — cycle model on click (instant, no server round-trip)
# ══════════════════════════════════════════════════════════════════════════════

clientside_callback(
    """
    function(n_clicks, n_clicks_m, current_idx, model_data) {
        if (!model_data || model_data.length === 0) {
            return [window.dash_clientside.no_update,
                    window.dash_clientside.no_update,
                    window.dash_clientside.no_update,
                    window.dash_clientside.no_update,
                    window.dash_clientside.no_update];
        }
        var idx = ((current_idx || 0) + 1) % model_data.length;
        var m = model_data[idx];
        var text = m.label + m.pct + "%";
        return [idx, text, {"color": m.color}, text, {"color": m.color}];
    }
    """,
    Output("ticker-model-idx", "data"),
    Output("ticker-pct", "children"),
    Output("ticker-pct", "style"),
    Output("ticker-pct-mobile", "children"),
    Output("ticker-pct-mobile", "style"),
    Input("ticker-pct", "n_clicks"),
    Input("ticker-pct-mobile", "n_clicks"),
    State("ticker-model-idx", "data"),
    State("model-percentiles-store", "data"),
    prevent_initial_call=True,
)
