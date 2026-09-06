"""Live BTC price ticker callback with multi-model percentile cycling."""

from dash import html, Input, Output, State, callback, clientside_callback, no_update

import _app_ctx
from btc_core import fmt_price, today_t, UserModel
from utils import _fetch_btc_price


# ── Model display order and colors ──────────────────────────────────────────
# Use default config-panel-resolved keys so ticker matches chart display.
# LPPL default: lp3 (3 freqs). HybPPL default: cfg_1d_1u. EPPL default: ecfg_1d_1u.
_MODEL_CYCLE = ["qr", "bub", "pl", "spl", "lp3", "cfg_1d_1u", "ecfg_1d_1u", "pca", "grdy", "ef"]
from colors import TICKER_MODEL_COLORS as _MODEL_COLORS, WHITE, USER_MODEL_TICKER_ORANGE
# Label overrides for config keys (otherwise shows the raw key)
_MODEL_LABELS = {
    "lp3":        "LPPL\u2083",
    "cfg_1d_1u":  "HybPPL",
    "ecfg_1d_1u": "EPPL",
}


# ══════════════════════════════════════════════════════════════════════════════
# Callback — live BTC price ticker (refreshes every 20 min)
# ══════════════════════════════════════════════════════════════════════════════

@callback(
    Output("price-ticker",        "children"),
    Output("price-ticker-mobile", "children"),
    Output("btc-price-store",     "data"),
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
def _ticker_sigma_mode():
    """The σ-mode the navbar percentile is computed at.

    Must match what the charts render with (`tab_defaults.bubble_defaults()
    ["sigma_mode"]`, i.e. `"resqr"`), or the navbar tells the user a
    different story from the Percentile view they click through to.

    It did not, until 2026-09-06: `find_percentile`'s signature default is
    `"constant"` (`btc_core/_base.py`) and this callback passed no
    `sigma_mode`, so the navbar read BM 99.020 % while `/1.4` read 79.652 %
    for the same coin — 19.37 pp apart, of which the live price and the
    deliberate data lag explained only ~0.4 pp. Five of the ten cycled
    models disagreed.

    Falls back to constant σ when no resqr bundle is loaded, which is also
    what `qr`, `lp3` and `ef` do per-model regardless of this setting.

    NOT a `State("bub-sigma-mode", "value")`: that control lives inside the
    lazy-loaded Tab 1, so referencing it here would raise Dash's
    "nonexistent object" error on any load that starts elsewhere — a
    measured prod cost, see memory `feedback_nonexistent_input_perf.md`.
    A user who flips the σ toggle changes their chart, not the navbar.
    """
    return "resqr" if getattr(_app_ctx, "_HAS_RESQR", False) else "constant"


def update_price_ticker(_, mode, user_model_data):
    # NOTE: this callback used to also write hm-entry-q, hm-mc-entry-q,
    # dca-mc-entry-q. Those cross-tab Outputs errored on page load
    # because the heatmap/dca tabs weren't yet mounted. Initial values
    # are now seeded once at worker startup via _startup_heatmap_defaults
    # and from layout defaults; they do not auto-refresh during a session.
    price = _fetch_btc_price()
    if price is None:
        return ("\u20bf \u2014", "\u20bf \u2014", no_update,
                "",
                no_update, no_update, no_update, no_update, no_update, no_update)

    t = today_t(_app_ctx.M.genesis)

    # Compute percentile for every model in the cycle
    pcts = {}
    for key in _MODEL_CYCLE:
        mdl = _app_ctx.PRICE_MODELS.get(key)
        if mdl is None:
            continue
        try:
            p = mdl.find_percentile(t, price, sigma_mode=_ticker_sigma_mode())
            pcts[key] = round(p * 100) if p is not None else None
        except Exception:
            pcts[key] = None

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
        label = _MODEL_LABELS.get(key, mdl.legend_name if hasattr(mdl, 'legend_name') else key.upper())
        color = _MODEL_COLORS.get(key, WHITE)
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
                    model_data.append({"key": "u1", "label": "U\u2081",
                                       "pct": um_pct_int, "color": USER_MODEL_TICKER_ORANGE})
        except Exception:
            pass

    # Default display: QR (index 0)
    default = model_data[0] if model_data else {"label": "QR", "pct": 0, "color": _MODEL_COLORS["qr"]}
    pct_text = f"{default['label']}{default['pct']}%"
    pct_color = default["color"]
    pct_style_d = {"color": pct_color}
    pct_style_m = {"color": pct_color}

    return (txt, txt_m, price, spark,
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
