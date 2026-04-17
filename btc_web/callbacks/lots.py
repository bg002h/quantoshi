"""Stack Tracker lot management callbacks — CRUD, summary, sync."""

import json
import base64
import logging

import dash
from dash import html, Input, Output, State, ctx, callback
import pandas as pd

import _app_ctx
from btc_core import fmt_price, _find_lot_percentile
from callbacks.coerce import _format_lots_for_table

logger = logging.getLogger(__name__)


@callback(
    Output("lot-pct-preview", "children"),
    Input("lot-date",  "value"),
    Input("lot-price", "value"),
)
def preview_percentile(date_str, price):
    if not date_str or not price or float(price) <= 0:
        return ""
    try:
        t   = (pd.Timestamp(date_str) - _app_ctx.M.genesis).days / 365.25
        pct = _find_lot_percentile(t, float(price), _app_ctx.M.qr_fits)
        return f"Q{pct*100:.2f}%"
    except Exception:
        return ""


@callback(
    Output("lots-store",        "data"),
    Output("lots-table",        "data"),
    Output("lots-table",        "selected_rows"),
    Output("lots-summary",      "children"),
    Output("lots-import-status","children"),
    Input("lot-add-btn",        "n_clicks"),
    Input("lot-del-btn",        "n_clicks"),
    Input("lot-clear-btn",      "n_clicks"),
    Input("lots-import-upload", "contents"),
    State("lot-date",           "value"),
    State("lot-btc",            "value"),
    State("lot-price",          "value"),
    State("lot-notes",          "value"),
    State("lots-table",         "selected_rows"),
    State("lots-store",         "data"),
    prevent_initial_call=True,
)
def manage_lots(add_n, del_n, clear_n, import_contents,
                date_str, btc_amt, price_val, notes,
                selected_rows, lots_data):
    """CRUD for Stack Tracker lots (add/delete/clear/import JSON)."""
    triggered = ctx.triggered_id
    lots = list(lots_data or [])
    import_status = dash.no_update

    if triggered == "lot-add-btn":
        if not date_str or not btc_amt or not price_val:
            raise dash.exceptions.PreventUpdate
        try:
            btc   = float(btc_amt)
            price = float(price_val)
            if btc <= 0 or price <= 0:
                raise ValueError
            t     = (pd.Timestamp(date_str) - _app_ctx.M.genesis).days / 365.25
            pct_q = _app_ctx.DEFAULT_MODEL.find_percentile(t, price)
            lots.append({
                "date":  date_str,
                "btc":   round(btc, 8),
                "price": round(price, 2),
                "pct_q": round(pct_q, 6),
                "notes": (notes or "").strip(),
            })
            lots.sort(key=lambda l: l["date"])
        except Exception:
            raise dash.exceptions.PreventUpdate

    elif triggered == "lot-del-btn":
        if selected_rows:
            lots = [l for i, l in enumerate(lots) if i not in selected_rows]

    elif triggered == "lot-clear-btn":
        lots = []

    elif triggered == "lots-import-upload" and import_contents:
        try:
            _hdr, b64 = import_contents.split(",", 1)
            # Cap pre-decode: 5000 lots at ~200 bytes each is ~1 MB of JSON;
            # 2 MB base64-encoded is a comfortable ceiling that stops OOM
            # uploads before base64.b64decode allocates.
            if len(b64) > 2_000_000:
                raise ValueError("file too large (max ~1.5 MB)")
            raw  = base64.b64decode(b64).decode("utf-8")
            data = json.loads(raw)
            if not isinstance(data, list):
                raise ValueError("expected a JSON array")
            if len(data) > 5000:
                raise ValueError(f"too many lots ({len(data)}), max 5000")
            # recompute pct_q in case file came from a different model version
            parsed = []
            for row in data:
                if not isinstance(row, dict):
                    raise ValueError("each lot must be a JSON object")
                if not isinstance(row.get("date"), str):
                    raise ValueError("lot missing or invalid 'date'")
                if not isinstance(row.get("btc"), (int, float)):
                    raise ValueError("lot missing or invalid 'btc'")
                if not isinstance(row.get("price"), (int, float)):
                    raise ValueError("lot missing or invalid 'price'")
                t     = (pd.Timestamp(row["date"]) - _app_ctx.M.genesis).days / 365.25
                pct_q = _app_ctx.DEFAULT_MODEL.find_percentile(t, float(row["price"]))
                parsed.append({
                    "date":  row["date"],
                    "btc":   round(float(row["btc"]), 8),
                    "price": round(float(row["price"]), 2),
                    "pct_q": round(pct_q, 6),
                    "notes": str(row.get("notes", "")).strip()[:200],
                })
            parsed.sort(key=lambda l: l["date"])
            lots = parsed
            import_status = f"Imported {len(lots)} lot(s) \u2713"
            logger.info("Lot import: %d lots", len(lots))
        except Exception as e:
            import_status = f"Import failed: {e}"
            logger.warning("Lot import failed: %s", e)
            raise dash.exceptions.PreventUpdate

    table_data = _format_lots_for_table(lots)
    return lots, table_data, [], _lots_summary(lots), import_status


def sync_table_on_load(lots_data):
    """Plain helper kept for tests + __init__ re-export. The live callback is
    a clientside port (below) that mirrors this logic in JS."""
    lots = lots_data or []
    return _format_lots_for_table(lots), _lots_summary(lots)


# Clientside port of sync_table_on_load. Mirrors:
#   - btc_core.fmt_price (USD with $/comma/suffix formatting)
#   - coerce._format_lots_for_table (adds total_paid + formatted pct_q)
#   - lots._lots_summary (count / total BTC / avg cost / total paid / avg pct)
# Keep this in sync with the Python helpers if they ever change.
_app_ctx.app.clientside_callback(
    """
    function(lots_data) {
        var lots = lots_data || [];
        function fmtPrice(p) {
            if (p == null || isNaN(p)) return '$0';
            function withCommas(s) {
                return s.replace(/\\B(?=(\\d{3})+(?!\\d))/g, ',');
            }
            if (p >= 1e18) return '$' + withCommas((p / 1e18).toFixed(1)) + 'Qi';
            if (p >= 1e15) return '$' + withCommas((p / 1e15).toFixed(1)) + 'Q';
            if (p >= 1e12) return '$' + withCommas((p / 1e12).toFixed(1)) + 'T';
            if (p >= 1e9)  return '$' + withCommas((p / 1e9).toFixed(1))  + 'B';
            if (p >= 1)    return '$' + withCommas(Math.round(p).toString());
            return '$' + p.toFixed(2);
        }
        var table = lots.map(function(l) {
            var row = {};
            for (var k in l) { if (Object.prototype.hasOwnProperty.call(l, k)) row[k] = l[k]; }
            row.total_paid = fmtPrice(l.btc * l.price);
            row.pct_q = 'Q' + (l.pct_q * 100).toFixed(2) + '%';
            return row;
        });
        var summary;
        if (!lots.length) {
            summary = 'No lots.';
        } else {
            var totalBtc = 0, totalPaid = 0, weightedPct = 0;
            for (var i = 0; i < lots.length; i++) {
                totalBtc    += lots[i].btc;
                totalPaid   += lots[i].btc * lots[i].price;
                weightedPct += lots[i].pct_q * lots[i].btc;
            }
            var avgPrice = totalBtc ? totalPaid / totalBtc : 0;
            var avgPct   = totalBtc ? weightedPct / totalBtc : 0;
            // %.8g — trim trailing zeros after up to 8 significant digits
            var btcStr = parseFloat(totalBtc.toPrecision(8)).toString();
            summary = lots.length + ' lot(s)  |  ' + btcStr + ' BTC  |  '
                    + 'Avg ' + fmtPrice(avgPrice) + '/BTC  |  '
                    + 'Total paid ' + fmtPrice(totalPaid) + '  |  '
                    + 'Avg Q' + (avgPct * 100).toFixed(2) + '%';
        }
        return [table, summary];
    }
    """,
    Output("lots-table",   "data",    allow_duplicate=True),
    Output("lots-summary", "children", allow_duplicate=True),
    Input("lots-store",    "data"),
    prevent_initial_call=True,
)


def _lots_summary(lots):
    """Generate summary text for lot display (total BTC, avg cost, P&L)."""
    if not lots:
        return "No lots."
    total_btc  = sum(l["btc"] for l in lots)
    total_paid = sum(l["btc"] * l["price"] for l in lots)
    avg_price  = total_paid / total_btc if total_btc else 0
    avg_pct    = sum(l["pct_q"] * l["btc"] for l in lots) / total_btc if total_btc else 0
    return (f"{len(lots)} lot(s)  |  {total_btc:.8g} BTC  |  "
            f"Avg {fmt_price(avg_price)}/BTC  |  "
            f"Total paid {fmt_price(total_paid)}  |  "
            f"Avg Q{avg_pct*100:.2f}%")


# ── Stack Tracker: clientside JSON export (data never leaves the browser) ─────
_app_ctx.app.clientside_callback(
    """
    function(n_clicks, lots_data) {
        if (!n_clicks) return window.dash_clientside.no_update;
        var data = lots_data || [];
        var json = JSON.stringify(data, null, 2);
        var blob = new Blob([json], {type: 'application/json'});
        var url  = URL.createObjectURL(blob);
        var a    = document.createElement('a');
        a.href     = url;
        a.download = 'btc_lots.json';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        return window.dash_clientside.no_update;
    }
    """,
    Output("lots-export-dummy", "data"),
    Input("lots-export-btn",    "n_clicks"),
    State("lots-store",         "data"),
    prevent_initial_call=True,
)
