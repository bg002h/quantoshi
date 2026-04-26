"""MC payment callbacks — BTCPay integration, invoice creation, polling."""

import os
import logging

import dash
from dash import html, Input, Output, State, ctx, callback, no_update, ALL

import _app_ctx
import btcpay
from colors import MC_FREE_GREEN, ERROR_RED_DARK
from callbacks.coerce import _ci, _cf
from callbacks.charts._resolvers import _resolve_mc_model_src
from mc_cache import (MC_DEFAULT_YEARS, MC_DEFAULT_START_YR, MC_DEFAULT_ENTRY_Q)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# MC payment callbacks (BTCPay integration)
# ══════════════════════════════════════════════════════════════════════════════

# Ordered tab list — pattern-matched mc-run-btn Inputs return a list in this
# order (dict ALL collapses to a single list arg keyed by pattern match order,
# which follows layout registration; we sort explicitly below to be safe).
_MC_TABS = ("dca", "ret", "hm", "sc", "cp", "bub")

_MC_QUANT_THRESHOLD = 50_000  # sats — trigger quant warning modal

@callback(
    Output("mc-pay-modal",   "is_open", allow_duplicate=True),
    Output("mc-pay-invoice", "data",    allow_duplicate=True),
    Output("mc-pay-info",    "children"),
    Output("mc-pay-status",  "children",   allow_duplicate=True),
    Output("mc-pay-poll",    "disabled",   allow_duplicate=True),
    Output("mc-pay-poll",    "n_intervals", allow_duplicate=True),
    Output("mc-pay-trigger", "data",    allow_duplicate=True),
    Output("mc-quant-modal", "is_open",    allow_duplicate=True),
    Output("mc-quant-cost-info", "children", allow_duplicate=True),
    Output("mc-quant-onchain-note", "children", allow_duplicate=True),
    Output({"type": "mc-run-status", "tab": ALL}, "children", allow_duplicate=True),
    # Pattern-matched Input: binds to {type:"mc-run-btn",tab:ALL} components
    # as they mount (lazy tabs). Avoids "nonexistent object" errors at load.
    Input({"type": "mc-run-btn", "tab": ALL}, "n_clicks"),
    State("dca-mc-years", "value"), State("dca-mc-start-yr", "value"),
    State("dca-mc-entry-q", "value"),
    State("ret-mc-years", "value"), State("ret-mc-start-yr", "value"),
    State("ret-mc-entry-q", "value"),
    State("hm-mc-years", "value"), State("hm-mc-start-yr", "value"),
    State("hm-mc-entry-q", "value"),
    State("sc-mc-years", "value"), State("sc-mc-start-yr", "value"),
    State("sc-mc-entry-q", "value"),
    State("cp-mc-years", "value"), State("cp-mc-start-yr", "value"),
    State("cp-mc-entry-q", "value"),
    State("bub-mc-years", "value"), State("bub-mc-start-yr", "value"),
    State("bub-mc-entry-q", "value"),
    State("mc-pay-trigger", "data"),
    *(State(f"{pfx}-mc-model-src", "value") for pfx in _MC_TABS),
    *(State(f"{pfx}-mc-price-val", "data") for pfx in _MC_TABS),
    # LPPL/HybPPL/EPPL config-modal States (global) — feed _resolve_mc_model_src
    # so the free-tier check uses the same resolved variant the chart callback
    # builds with. Without these, "lppl"/"hybppl"/"eppl" master selection from
    # the MC source dropdown lands in inconsistent free-vs-paid decisions
    # between this callback and update_*. See e634b54 + the bugfix in this commit.
    State("lppl-n-freqs", "value"),
    State("lppl-weighted", "value"),
    State("lppl-no-13", "value"),
    State("hybppl-cfg-a-nlog",  "value"),
    State("hybppl-cfg-a-ncal",  "value"),
    State("hybppl-cfg-a-log1d", "value"),
    State("hybppl-cfg-a-log2d", "value"),
    State("hybppl-cfg-a-cal1d", "value"),
    State("hybppl-cfg-a-cal2d", "value"),
    State("eppl-cfg-a-nlog",  "value"),
    State("eppl-cfg-a-ncal",  "value"),
    State("eppl-cfg-a-log1d", "value"),
    State("eppl-cfg-a-log2d", "value"),
    State("eppl-cfg-a-cal1d", "value"),
    State("eppl-cfg-a-cal2d", "value"),
    prevent_initial_call=True,
)
def _mc_payment_initiate(*args):
    """Handle Run Simulation button clicks — check free tier or create invoice."""
    # Pattern-matched Input collapses to a single list-arg at args[0].
    # ctx.triggered_id is a dict like {"type":"mc-run-btn","tab":"dca"}.
    triggered = ctx.triggered_id
    if not isinstance(triggered, dict) or triggered.get("type") != "mc-run-btn":
        raise dash.exceptions.PreventUpdate
    tab = triggered.get("tab")
    if tab not in _MC_TABS:
        raise dash.exceptions.PreventUpdate

    # Extract the relevant tab's MC params from states
    tab_idx = _MC_TABS.index(tab)
    n_tabs = len(_MC_TABS)  # 6 tabs (was 5; bub appended in this commit)
    # Layout: 1 list-Input (args[0]), then (years, start_yr, entry_q) × n_tabs, trigger, n_tabs model_srcs, n_tabs prices
    state_base = 1  # skip the single pattern-match Input list
    mc_years  = _ci(args[state_base + tab_idx * 3],     MC_DEFAULT_YEARS)
    start_yr  = _ci(args[state_base + tab_idx * 3 + 1], MC_DEFAULT_START_YR)
    entry_q   = _cf(args[state_base + tab_idx * 3 + 2], MC_DEFAULT_ENTRY_Q)
    cur_trigger = args[state_base + n_tabs * 3] or 0  # after n_tabs×3 tab states
    # Model source: n_tabs values after trigger
    model_srcs = args[state_base + n_tabs * 3 + 1 : state_base + n_tabs * 3 + 1 + n_tabs]
    mc_model_src = model_srcs[tab_idx] or "bub"
    # Price stores: n_tabs values after model sources
    price_start = state_base + n_tabs * 3 + 1 + n_tabs
    price_vals = args[price_start : price_start + n_tabs]
    tab_price = _ci(price_vals[tab_idx], 0)
    # LPPL/HybPPL/EPPL config-modal States: 15 values after price stores.
    # Resolve mc_model_src master ("hybppl" / "eppl" / "lppl") to its concrete
    # variant the chart callback will use — keeps free-tier decision
    # symmetric with update_dca/retire/sc/heatmap.
    cfg_start = price_start + n_tabs
    (lppl_n_freqs, lppl_weighted, lppl_no_13,
     hyb_a_nlog, hyb_a_ncal, hyb_a_log1d, hyb_a_log2d,
     hyb_a_cal1d, hyb_a_cal2d,
     ep_a_nlog, ep_a_ncal, ep_a_log1d, ep_a_log2d,
     ep_a_cal1d, ep_a_cal2d) = args[cfg_start : cfg_start + 15]
    mc_model_src = _resolve_mc_model_src(
        mc_model_src,
        lppl_n_freqs, lppl_weighted, lppl_no_13,
        hyb_a_nlog, hyb_a_ncal, hyb_a_log1d, hyb_a_log2d,
        hyb_a_cal1d, hyb_a_cal2d,
        ep_a_nlog, ep_a_ncal, ep_a_log1d, ep_a_log2d,
        ep_a_cal1d, ep_a_cal2d)

    # Default outputs: modal closed, no change to per-tab status
    no_tab_status = [dash.no_update] * n_tabs
    # Helper: base return with quant modal closed
    def _ret(*vals):
        """Insert quant-modal defaults (closed, no update) into return tuple.
        Layout: 7 payment outputs + 3 quant outputs + 1 pattern-match list
        output (mc-run-status for all tabs)."""
        return vals[:7] + (False, dash.no_update, "") + vals[7:]

    # Quant-tier warning (>50k sats) — fires before payment/free checks
    if tab_price > _MC_QUANT_THRESHOLD:
        onchain_note = ("(Bitcoin on-chain payment may be necessary)"
                        if tab_price > 250_000 else "")
        return (dash.no_update, dash.no_update, dash.no_update,
                dash.no_update, dash.no_update, dash.no_update, dash.no_update,
                True, f"Estimated cost: {tab_price:,} sats", onchain_note,
                no_tab_status)

    # If BTCPay not configured or DEV mode, just increment trigger (free mode)
    if not _app_ctx._HAS_BTCPAY or os.environ.get("DEV") == "1":
        return _ret(False, dash.no_update, dash.no_update,
                    dash.no_update, True, 0, cur_trigger + 1,
                    no_tab_status)

    # Free tier check (bins/sims/freq not available here; uses defaults)
    if btcpay.is_free_tier(mc_model_src, mc_years, start_yr, entry_q):
        tab_statuses = list(no_tab_status)
        tab_statuses[tab_idx] = html.Span("Free tier", style={"color": MC_FREE_GREEN})
        return _ret(False, dash.no_update, dash.no_update,
                    dash.no_update, True, 0, cur_trigger + 1,
                    tab_statuses)

    # Create invoice for live simulation
    try:
        result = btcpay.create_invoice(tab, mc_years)
    except Exception:
        tab_statuses = list(no_tab_status)
        tab_statuses[tab_idx] = html.Span(
            "Payment service unavailable", style={"color": ERROR_RED_DARK})
        return _ret(False, dash.no_update, "",
                    "", True, 0, dash.no_update, tab_statuses)

    # Open payment modal — clientside callback handles iframe vs QR display
    invoice_data = {
        "invoice_id": result["invoice_id"],
        "tab": tab,
        "mc_years": mc_years,
        "checkout_url": result.get("checkout_url", ""),
        "amount_sats": result.get("amount_sats", 0),
    }
    info = f"Cost: {invoice_data['amount_sats']} sats \u2022 {mc_years}yr MC simulation"

    return _ret(True, invoice_data, info,
                "Waiting for payment...", False, 0,
                dash.no_update, no_tab_status)


# ── Quant warning modal — proceed / cancel ────────────────────────────────────

_app_ctx.app.clientside_callback(
    "function(n, t) { return [false, (t || 0) + 1]; }",
    Output("mc-quant-modal", "is_open", allow_duplicate=True),
    Output("mc-pay-trigger", "data", allow_duplicate=True),
    Input("mc-quant-proceed", "n_clicks"),
    State("mc-pay-trigger", "data"),
    prevent_initial_call=True,
)

_app_ctx.app.clientside_callback(
    "function(n) { return false; }",
    Output("mc-quant-modal", "is_open", allow_duplicate=True),
    Input("mc-quant-cancel", "n_clicks"),
    prevent_initial_call=True,
)


# ── Clientside polling: check invoice status every 3s ─────────────────────────
_app_ctx.app.clientside_callback(
    """
    function(n, invoice_data, cur_trigger) {
        if (!invoice_data || !invoice_data.invoice_id) {
            return [window.dash_clientside.no_update, true,
                    window.dash_clientside.no_update,
                    window.dash_clientside.no_update,
                    window.dash_clientside.no_update];
        }
        var inv = invoice_data;
        var url = "/api/mc/invoice/" + inv.invoice_id
                  + "?tab=" + inv.tab + "&mc_years=" + inv.mc_years;

        return fetch(url)
            .then(function(r) { return r.json(); })
            .then(function(d) {
                if (d.paid) {
                    // Payment confirmed — store token, close modal, trigger MC
                    var token = {
                        payment_token: d.payment_token,
                        invoice_id: inv.invoice_id,
                        tab: inv.tab,
                        mc_years: inv.mc_years
                    };
                    return [token, true, false,
                            "Payment confirmed!",
                            (cur_trigger || 0) + 1];
                }
                if (d.status === "Expired" || d.status === "Invalid") {
                    return [window.dash_clientside.no_update, true, false,
                            "Invoice " + d.status.toLowerCase() + ". Please try again.",
                            window.dash_clientside.no_update];
                }
                // Still waiting
                return [window.dash_clientside.no_update,
                        window.dash_clientside.no_update,
                        window.dash_clientside.no_update,
                        "Waiting for payment...",
                        window.dash_clientside.no_update];
            })
            .catch(function(e) {
                return [window.dash_clientside.no_update,
                        window.dash_clientside.no_update,
                        window.dash_clientside.no_update,
                        "Checking payment...",
                        window.dash_clientside.no_update];
            });
    }
    """,
    Output("mc-pay-token",   "data"),
    Output("mc-pay-poll",    "disabled"),
    Output("mc-pay-modal",   "is_open"),
    Output("mc-pay-status",  "children"),
    Output("mc-pay-trigger", "data"),
    Input("mc-pay-poll",     "n_intervals"),
    State("mc-pay-invoice",  "data"),
    State("mc-pay-trigger",  "data"),
    prevent_initial_call=True,
)


# ── Cancel payment modal ─────────────────────────────────────────────────────
_app_ctx.app.clientside_callback(
    "function(n) { return [false, true, '']; }",
    Output("mc-pay-modal", "is_open",  allow_duplicate=True),
    Output("mc-pay-poll",  "disabled", allow_duplicate=True),
    Output("mc-pay-status","children", allow_duplicate=True),
    Input("mc-pay-cancel", "n_clicks"),
    prevent_initial_call=True,
)


# ── Clientside: detect onion vs clearnet and set up payment display ──────────
_app_ctx.app.clientside_callback(
    """
    function(invoice_data) {
        var nu = window.dash_clientside.no_update;
        if (!invoice_data || !invoice_data.invoice_id) {
            return [{"display":"none"}, "about:blank", {"display":"none"}, null];
        }
        var isOnion = window.location.hostname.endsWith('.onion');
        if (isOnion) {
            return [{}, invoice_data.checkout_url, {"display":"none"}, null];
        }
        // Clearnet: hide iframe, fetch payment methods, show QR + text
        return fetch("/api/mc/invoice/" + invoice_data.invoice_id + "/payment")
            .then(function(r) { return r.json(); })
            .then(function(d) {
                return [{"display":"none"}, "about:blank", {}, d.methods || []];
            })
            .catch(function() {
                return [{"display":"none"}, "about:blank", {},
                        [{method:"error", destination:"Could not load payment methods",
                          amount:"", qr_svg:""}]];
            });
    }
    """,
    Output("mc-pay-iframe-wrap", "style"),
    Output("mc-pay-iframe",      "src"),
    Output("mc-pay-details",     "style"),
    Output("mc-pay-methods",     "data"),
    Input("mc-pay-invoice",      "data"),
    prevent_initial_call=True,
)


# ── Clientside: render active payment method (QR + dest + amount) ────────────
_app_ctx.app.clientside_callback(
    """
    function(methods, ln_clicks, chain_clicks) {
        var nu = window.dash_clientside.no_update;
        if (!methods || !methods.length) return ["", "", ""];
        var ctx = window.dash_clientside.callback_context;
        var triggered = (ctx && ctx.triggered && ctx.triggered.length)
                        ? ctx.triggered[0].prop_id : "";
        var wantLn = triggered.indexOf("chain-btn") === -1;
        var m;
        if (wantLn) {
            m = methods.find(function(x) { return x.method && x.method.indexOf("LN") !== -1; });
        } else {
            m = methods.find(function(x) { return x.method && x.method.indexOf("LN") === -1; });
        }
        if (!m) m = methods[0];
        var qr = m.qr_svg || "";
        var dest = m.destination || "";
        var amt = parseFloat(m.amount || "0");
        var info;
        if (wantLn) {
            var sats = Math.round(amt * 1e8);
            info = sats.toLocaleString() + " sats";
        } else {
            info = amt + " BTC";
        }
        return [qr, dest, info];
    }
    """,
    Output("mc-pay-qr",          "src"),
    Output("mc-pay-dest",        "children"),
    Output("mc-pay-amount-info", "children"),
    Input("mc-pay-methods",      "data"),
    Input("mc-pay-ln-btn",       "n_clicks"),
    Input("mc-pay-chain-btn",    "n_clicks"),
    prevent_initial_call=True,
)


# ── Clientside: toggle LN/on-chain button active state ──────────────────────
_app_ctx.app.clientside_callback(
    """
    function(ln_clicks, chain_clicks) {
        var ctx = window.dash_clientside.callback_context;
        var triggered = (ctx && ctx.triggered && ctx.triggered.length)
                        ? ctx.triggered[0].prop_id : "";
        var isChain = triggered.indexOf("chain-btn") !== -1;
        return [!isChain, !isChain, isChain, isChain];
    }
    """,
    Output("mc-pay-ln-btn",    "active"),
    Output("mc-pay-chain-btn", "outline"),
    Output("mc-pay-chain-btn", "active"),
    Output("mc-pay-ln-btn",    "outline"),
    Input("mc-pay-ln-btn",     "n_clicks"),
    Input("mc-pay-chain-btn",  "n_clicks"),
    prevent_initial_call=True,
)


# ── Clientside: copy payment destination to clipboard ────────────────────────
_app_ctx.app.clientside_callback(
    """
    function(n) {
        var el = document.getElementById("mc-pay-dest");
        if (el && el.textContent) {
            navigator.clipboard.writeText(el.textContent);
        }
        return "";
    }
    """,
    Output("mc-pay-status", "children", allow_duplicate=True),
    Input("mc-pay-copy-btn", "n_clicks"),
    prevent_initial_call=True,
)
