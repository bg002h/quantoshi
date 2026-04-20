"""Snapshot / Share callbacks — encode/decode, history, QR, palette presets."""

import json
import logging

import dash
from dash import html, Input, Output, State, ctx, callback, no_update
import dash_bootstrap_components as dbc
import pandas as pd
from flask import request as flask_request

import _app_ctx
from colors import (FALLBACK_MODEL_GRAY, HM_PRESET_PALETTES, PALETTE_DEFAULT_HM_PRESET,
                    UI_FONT_SM, UI_FONT_MD)
from snapshot import (_SNAPSHOT_CONTROLS, _CHECKLIST_OPTIONS,
                      _SNAP_PREFIX, _SNAP_PREFIX_V1, _SNAP_PREFIX_V2,
                      _encode_snapshot, _decode_snapshot, _decode_snapshot_v1,
                      _list_to_mask, _mask_to_list)
from callbacks.routing import _TAB_CONTROLS, _TAB_TO_PATH

logger = logging.getLogger(__name__)


def _decode_snapshot_by_prefix(h):
    """Decode a snapshot hash, auto-detecting the version prefix.

    Returns (state_dict, prefix, encoded_payload) or (None, None, None).
    """
    if h.startswith(_SNAP_PREFIX):
        return _decode_snapshot(h[len(_SNAP_PREFIX):]), _SNAP_PREFIX, h[len(_SNAP_PREFIX):]
    if h.startswith(_SNAP_PREFIX_V2):
        return _decode_snapshot(h[len(_SNAP_PREFIX_V2):]), _SNAP_PREFIX_V2, h[len(_SNAP_PREFIX_V2):]
    if h.startswith(_SNAP_PREFIX_V1):
        return _decode_snapshot_v1(h[len(_SNAP_PREFIX_V1):]), _SNAP_PREFIX_V1, h[len(_SNAP_PREFIX_V1):]
    return None, None, None


@callback(
    Output("snapshot-state-store", "data"),
    Output("loaded-hash-store",    "data"),
    Input("url", "hash"),
    prevent_initial_call=False,
)
def restore_from_url(hash_str):
    """Decode snapshot hash from URL → intermediate store for apply_snapshot."""
    if not hash_str:
        return no_update, no_update
    h = hash_str.lstrip("#")
    state, prefix, encoded = _decode_snapshot_by_prefix(h)
    if not state:
        logger.warning("Snapshot decode failed for hash: %s\u2026", hash_str[:20])
        return no_update, no_update
    # Legacy-link coercion: if this deployment has no resqr bundles, drop
    # "resqr" sigma_mode back to "constant" so the radio + chart stay in sync.
    # Done upstream (before apply_snapshot) so all consumers see the coerced
    # value, including the lazy bubble relay.
    if not getattr(_app_ctx, "_HAS_RESQR", False):
        if state.get("bub-sigma-mode:value") == "resqr":
            state = dict(state)
            state["bub-sigma-mode:value"] = "constant"
    logger.info("Snapshot restored: %d controls, lots=%s",
                sum(1 for k in state if k != "_lots"), "yes" if "_lots" in state else "no")
    return state, hash_str


# Split _SNAPSHOT_CONTROLS into eager (mounted at page load) vs lazy-bubble
# (live inside bubble-lazy, only mounted when user visits /1 or the bubble
# tab is the initial_tab). A share link to /2-/9#q3:... would otherwise try
# to write bub-*, scan-*, cta-* components that don't exist in the DOM.
_BUBBLE_LAZY_PREFIXES = ("bub-", "scan-", "cta-")
_BUBBLE_LAZY_CONTROLS = [(cid, prop) for cid, prop in _SNAPSHOT_CONTROLS
                         if cid.startswith(_BUBBLE_LAZY_PREFIXES)]
_EAGER_CONTROLS = [(cid, prop) for cid, prop in _SNAPSHOT_CONTROLS
                   if not cid.startswith(_BUBBLE_LAZY_PREFIXES)]


@callback(
    *[Output(cid, prop, allow_duplicate=True) for cid, prop in _EAGER_CONTROLS],
    Output("snapshot-lots", "data", allow_duplicate=True),
    Output("snapshot-apply-bubble", "data", allow_duplicate=True),
    Input("snapshot-state-store", "data"),
    prevent_initial_call=True,
)
def apply_snapshot(state):
    """Apply decoded snapshot state to eager (non-bubble) controls; route
    bubble-lazy controls through snapshot-apply-bubble relay store so they
    land only when the bubble tab is mounted."""
    n_outs = len(_EAGER_CONTROLS) + 2
    if not state:
        return [no_update] * n_outs
    results = [state.get(f"{cid}:{prop}", no_update)
               for cid, prop in _EAGER_CONTROLS]
    results.append(state.get("_lots", None))  # snapshot-lots
    # Bubble relay payload: dict of {cid:prop: value} for ids the stage-2
    # callback will write. Pass the full state dict so the relay can use
    # the same "cid:prop" lookup pattern.
    results.append(state)
    return results


@callback(
    *[Output(cid, prop, allow_duplicate=True) for cid, prop in _BUBBLE_LAZY_CONTROLS],
    Input("bubble-first-render", "data"),
    State("snapshot-apply-bubble", "data"),
    prevent_initial_call=True,
)
def apply_snapshot_bubble(_trigger, state):
    """Stage-2: apply bubble-lazy snapshot values once bubble tab is mounted."""
    n_outs = len(_BUBBLE_LAZY_CONTROLS)
    if not state:
        return [no_update] * n_outs
    return [state.get(f"{cid}:{prop}", no_update)
            for cid, prop in _BUBBLE_LAZY_CONTROLS]


@callback(
    Output("share-url-display", "value"),
    Output("link-history",      "data"),
    Input("share-copy-btn",    "n_clicks"),
    Input("loaded-hash-store", "data"),
    State("share-scope",        "value"),
    State("share-include-lots", "value"),
    State("lots-store",         "data"),
    *[State(cid, prop) for cid, prop in _SNAPSHOT_CONTROLS],
    State("link-history",       "data"),
    prevent_initial_call=True,
)
def manage_snapshot(n_btn, loaded_hash, share_scope, include_lots, lots_data, *rest):
    *ctrl_vals, history = rest
    history  = list(history or [])
    existing = {h["hash"] for h in history}
    triggered = ctx.triggered_id

    if triggered == "share-copy-btn":
        state = {f"{cid}:{prop}": val
                 for (cid, prop), val in zip(_SNAPSHOT_CONTROLS, ctrl_vals)}
        if include_lots and lots_data:
            state["_lots"] = lots_data
        active_tab = state.get("main-tabs:active_tab") or "bubble"
        tab_path   = _TAB_TO_PATH.get(active_tab, "/1")
        scope      = share_scope or "all"
        tab_filter = _TAB_CONTROLS.get(active_tab) if scope == "tab" else None
        encoded    = _encode_snapshot(state, tab_filter=tab_filter)
        base_url   = flask_request.host_url.rstrip("/")
        full_url   = f"{base_url}{tab_path}#{_SNAP_PREFIX}{encoded}"
        _add_snapshot_entry(history, existing, encoded, full_url,
                            bool(include_lots and lots_data), scope, active_tab)
        return full_url, history

    if triggered == "loaded-hash-store" and loaded_hash:
        h = loaded_hash.lstrip("#")
        state, prefix, encoded = _decode_snapshot_by_prefix(h)
        if not state:
            return no_update, no_update
        active_tab = state.get("main-tabs:active_tab") or "bubble"
        tab_path   = _TAB_TO_PATH.get(active_tab, "/1")
        base_url   = flask_request.host_url.rstrip("/")
        full_url   = f"{base_url}{tab_path}#{prefix}{encoded}"
        if _add_snapshot_entry(history, existing, encoded, full_url,
                               "_lots" in state, "unknown", active_tab):
            return no_update, history

    return no_update, no_update


def _add_snapshot_entry(history, existing, encoded, full_url,
                        includes_lots, scope, tab):
    """Append a snapshot entry to history if not already present.

    Mutates history in-place and returns True if an entry was added.
    """
    if encoded in existing:
        return False
    history.insert(0, {
        "hash": encoded, "url": full_url,
        "ts": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "includes_lots": includes_lots,
        "scope": scope,
        "tab": tab,
    })
    history[:] = history[:50]
    return True


def update_effective_lots(local_lots, snapshot_lots):
    """Plain helper kept for tests + __init__ re-export. The live callback is
    a clientside port (below)."""
    return snapshot_lots if snapshot_lots is not None else (local_lots or [])


_app_ctx.app.clientside_callback(
    """
    function(local_lots, snapshot_lots) {
        return (snapshot_lots != null) ? snapshot_lots : (local_lots || []);
    }
    """,
    Output("effective-lots", "data"),
    Input("lots-store",    "data"),
    Input("snapshot-lots", "data"),
    prevent_initial_call=True,
)


def update_snapshot_banner(snapshot_lots):
    """Plain helper kept for tests + __init__ re-export. The live callback is
    a clientside port (below)."""
    if not snapshot_lots:
        return []
    n = len(snapshot_lots)
    return dbc.Alert([
        html.Span(f"Showing {n} lot(s) from a shared link.  "),
        dbc.Button("Restore my lots", id="restore-lots-btn",
                   color="link", size="sm", className="p-0 ms-1 align-baseline"),
    ], color="info", className="py-1 px-3 mb-2 d-flex align-items-center")


# Build the snapshot banner clientside. The Alert + Button component tree is
# static apart from the lot count, so we emit Dash component descriptors
# (namespace/type/props) directly. The Restore button keeps its id so the
# clientside restore_my_lots callback continues to fire on click.
_app_ctx.app.clientside_callback(
    """
    function(snapshot_lots) {
        if (!snapshot_lots || !snapshot_lots.length) return [];
        var n = snapshot_lots.length;
        return [{
            namespace: 'dash_bootstrap_components',
            type: 'Alert',
            props: {
                color: 'info',
                className: 'py-1 px-3 mb-2 d-flex align-items-center',
                children: [
                    {
                        namespace: 'dash_html_components',
                        type: 'Span',
                        props: {children: 'Showing ' + n + ' lot(s) from a shared link.  '}
                    },
                    {
                        namespace: 'dash_bootstrap_components',
                        type: 'Button',
                        props: {
                            id: 'restore-lots-btn',
                            children: 'Restore my lots',
                            color: 'link',
                            size: 'sm',
                            className: 'p-0 ms-1 align-baseline'
                        }
                    }
                ]
            }
        }];
    }
    """,
    Output("snapshot-lots-banner", "children"),
    Input("snapshot-lots", "data"),
)


_app_ctx.app.clientside_callback(
    "function(n) { return null; }",
    Output("snapshot-lots", "data", allow_duplicate=True),
    Input("restore-lots-btn", "n_clicks"),
    prevent_initial_call=True,
)


@callback(
    Output("link-history-display", "children"),
    Input("link-history", "data"),
)
def render_link_history(history):
    if not history:
        return html.Small("No links yet.", className="text-muted")
    items = []
    for entry in history:
        badge = (dbc.Badge("lots", color="info", pill=True, className="me-1")
                 if entry.get("includes_lots") else None)
        items.append(dbc.ListGroupItem([
            html.Div([
                html.Small(entry.get("ts", ""), className="text-muted me-2"),
                badge,
            ], className="mb-1"),
            dbc.InputGroup([
                dbc.Input(value=entry.get("url", ""), readonly=True, size="sm",
                          style={"fontFamily":"monospace","fontSize": UI_FONT_MD}),
            ], size="sm"),
            html.Div(
                html.A("\u21a9 Restore this configuration",
                       href=("#" + entry["url"].split("#", 1)[1])
                            if "#" in entry.get("url", "")
                            else f"#{_SNAP_PREFIX}{entry['hash']}",
                       className="small"),
                className="mt-1",
            ),
        ], className="py-2"))
    return dbc.ListGroup(items, flush=True,
                         style={"maxHeight":"300px","overflowY":"auto"})


_app_ctx.app.clientside_callback(
    "function(n) { return []; }",
    Output("link-history", "data", allow_duplicate=True),
    Input("clear-history-btn", "n_clicks"),
    prevent_initial_call=True,
)


# ── Ticker mode toggle (USD ↔ sats/$) ────────────────────────────────────────
_app_ctx.app.clientside_callback(
    """
    function(n, current) {
        if (!n) return window.dash_clientside.no_update;
        return current === 'sats' ? 'usd' : 'sats';
    }
    """,
    Output("ticker-mode-store", "data"),
    Input("ticker-mode-toggle", "n_clicks"),
    State("ticker-mode-store", "data"),
    prevent_initial_call=True,
)

# ── Ticker mode toggle label ────────────────────────────────────────────────
_app_ctx.app.clientside_callback(
    """
    function(mode) {
        return mode === 'sats' ? '$/\\u20bf' : 'sats/$';
    }
    """,
    Output("ticker-mode-toggle", "children"),
    Input("ticker-mode-store", "data"),
)

# ── Share modal: copy feedback toast ──────────────────────────────────────────
_app_ctx.app.clientside_callback(
    """
    function(n) {
        if (!n) return '';
        var toast = document.createElement('div');
        toast.className = 'copy-toast';
        toast.textContent = 'Copied!';
        document.body.appendChild(toast);
        setTimeout(function() { toast.remove(); }, 2000);
        return '';
    }
    """,
    Output("copy-toast-container", "children"),
    Input("share-clipboard", "n_clicks"),
    prevent_initial_call=True,
)

# ── Share modal: capture chart preview thumbnail ─────────────────────────────
_TAB_GRAPH_MAP = {
    "bubble": "bubble-graph",
    "heatmap": "heatmap-graph",
    "dca": "dca-graph",
    "retire": "retire-graph",
    "supercharge": "supercharge-graph",
}
_app_ctx.app.clientside_callback(
    """
    function(is_open, active_tab) {
        if (!is_open) return {'display': 'none'};
        var graphMap = %s;
        var graphId = graphMap[active_tab];
        if (!graphId) return {'display': 'none'};
        var el = document.getElementById(graphId);
        if (!el) return {'display': 'none'};
        var plotDiv = el.querySelector('.js-plotly-plot');
        if (!plotDiv || !plotDiv.data) return {'display': 'none'};
        /* Use Plotly.toImage for a small preview */
        Plotly.toImage(plotDiv, {format: 'png', width: 400, height: 240, scale: 1})
            .then(function(dataUrl) {
                var img = document.getElementById('share-preview-thumb');
                if (img) {
                    img.src = dataUrl;
                    img.style.display = 'block';
                }
            });
        return {'display': 'none'};  /* initial hide; JS sets it directly */
    }
    """ % str({k: v for k, v in _TAB_GRAPH_MAP.items()}).replace("'", '"'),
    Output("share-preview-thumb", "style"),
    Input("share-modal", "is_open"),
    State("main-tabs", "active_tab"),
    prevent_initial_call=True,
)

# ── Stack Tracker lot badge — orange dot on tab when lots exist ───────────────
_app_ctx.app.clientside_callback(
    """
    function(lots) {
        var tabs = document.querySelectorAll('.nav-tabs .nav-link');
        for (var i = 0; i < tabs.length; i++) {
            var text = tabs[i].textContent || '';
            if (text.indexOf('Stack Tracker') !== -1) {
                var existing = tabs[i].querySelector('.tab-lot-badge');
                if (lots && lots.length > 0) {
                    if (!existing) {
                        var dot = document.createElement('span');
                        dot.className = 'tab-lot-badge';
                        tabs[i].appendChild(dot);
                    }
                } else if (existing) {
                    existing.remove();
                }
                break;
            }
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("lots-store", "data", allow_duplicate=True),
    Input("lots-store", "data"),
    prevent_initial_call="initial_duplicate",
)

# ── Heatmap color palette presets ─────────────────────────────────────────────
# HM_PRESET_PALETTES is a flat 4-preset dict: rwg / rbg / bwo / mono.
# Presets are site-palette-invariant. The site-wide palette determines which
# preset is auto-selected as the default (see _auto_select_hm_preset below).
import json as _json
_HM_PALETTES_JS = _json.dumps({k: list(v) for k, v in HM_PRESET_PALETTES.items()})

_app_ctx.app.clientside_callback(
    f"""function(preset_name) {{
        var palettes = {_HM_PALETTES_JS};
        if (!preset_name || !palettes[preset_name]) {{
            return [window.dash_clientside.no_update,
                    window.dash_clientside.no_update,
                    window.dash_clientside.no_update,
                    window.dash_clientside.no_update];
        }}
        return palettes[preset_name];
    }}""",
    Output("hm-c-lo", "value", allow_duplicate=True),
    Output("hm-c-mid1", "value", allow_duplicate=True),
    Output("hm-c-mid2", "value", allow_duplicate=True),
    Output("hm-c-hi", "value", allow_duplicate=True),
    Input("hm-palette", "value"),
    prevent_initial_call=True,
)


def apply_hm_palette(preset_name):
    """Plain helper kept for tests + __init__ re-export."""
    if not preset_name or preset_name not in HM_PRESET_PALETTES:
        return no_update, no_update, no_update, no_update
    return HM_PRESET_PALETTES[preset_name]


# When the site-wide palette changes, auto-switch the heatmap preset
# dropdown to the palette's recommended default (red/green for the
# default palette; blue/orange for all colorblind palettes).
#
# NB: this is a CLIENTSIDE callback to avoid a known Dash dcc.Dropdown
# stale-label bug where a Python-callback-driven value change doesn't
# re-render the visible label. The clientside version touches the React
# state directly and the label refreshes correctly.
_app_ctx.app.clientside_callback(
    """
    function(palette_key) {
        var map = {"default": "rwg", "cb-brian": "bwo", "cb-rg": "bwo", "cb-full": "bwo"};
        return map[palette_key || "default"] || "rwg";
    }
    """,
    Output("hm-palette", "value", allow_duplicate=True),
    Input("palette-store", "data"),
    prevent_initial_call=True,
)


# ── Share modal: QR code for generated link ──────────────────────────────────
@callback(
    Output("share-qr-img", "src"),
    Output("share-qr-img", "style"),
    Output("share-qr-label", "style"),
    Input("share-url-display", "value"),
    prevent_initial_call=True,
)
def generate_share_qr(url):
    _hidden = {"display": "none"}
    if not url or url.startswith("Click"):
        return "", _hidden, _hidden
    try:
        import qrcode
        import qrcode.image.svg
        import io
        import base64
        qr = qrcode.QRCode(version=None, error_correction=qrcode.constants.ERROR_CORRECT_L,
                            box_size=8, border=2)
        qr.add_data(url)
        qr.make(fit=True)
        img = qr.make_image(image_factory=qrcode.image.svg.SvgPathImage)
        buf = io.BytesIO()
        img.save(buf)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return (f"data:image/svg+xml;base64,{b64}",
                {"display": "block", "margin": "10px auto", "maxWidth": "160px"},
                {"display": "block", "fontSize": UI_FONT_SM, "color": FALLBACK_MODEL_GRAY,
                 "textAlign": "center", "marginBottom": "8px"})
    except Exception:
        return "", _hidden, _hidden
