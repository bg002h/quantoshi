"""Navbar drawers, palette sync, image export, share modal, price pulse."""

from dash import Input, Output, State, callback, no_update

import _app_ctx
from colors import LIGHTBOX_BG


# ══════════════════════════════════════════════════════════════════════════════
# Viewport width -- fires on page load, provides mobile detection for charts
# ══════════════════════════════════════════════════════════════════════════════

_app_ctx.app.clientside_callback(
    "function() { return window.innerWidth; }",
    Output("viewport-width", "data"),
    Input("url", "pathname"),
)


# ══════════════════════════════════════════════════════════════════════════════
# Mobile nav drawer: auto-collapse after 3s, toggle on tap
# ══════════════════════════════════════════════════════════════════════════════

_app_ctx.app.clientside_callback(
    """
    function(n) {
        var drawer = document.getElementById("mobile-nav-drawer");
        var toggle = document.getElementById("mobile-nav-toggle");
        if (!drawer || !toggle) return window.dash_clientside.no_update;

        if (!window._navDrawerInit) {
            window._navDrawerInit = true;
            // Auto-collapse after 3 seconds
            setTimeout(function() {
                if (!window._navDrawerManual) {
                    drawer.classList.add("collapsed");
                    toggle.classList.add("visible");
                }
            }, 3000);
        }
        // On tap: toggle open/closed
        if (n) {
            window._navDrawerManual = true;
            var isCollapsed = drawer.classList.contains("collapsed");
            if (isCollapsed) {
                drawer.classList.remove("collapsed");
                toggle.classList.remove("visible");
                // Re-collapse after 4s
                setTimeout(function() {
                    drawer.classList.add("collapsed");
                    toggle.classList.add("visible");
                }, 4000);
            } else {
                drawer.classList.add("collapsed");
                toggle.classList.add("visible");
            }
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("mobile-nav-toggle", "className"),
    Input("mobile-nav-toggle", "n_clicks"),
    prevent_initial_call=False,
)

# ── Desktop nav drawer: auto-collapse after 4s, toggle on tap ─────────────────
_app_ctx.app.clientside_callback(
    """
    function(n) {
        var drawer = document.getElementById("desktop-nav-drawer");
        var toggle = document.getElementById("desktop-nav-toggle");
        if (!drawer || !toggle) return window.dash_clientside.no_update;

        if (!window._deskDrawerInit) {
            window._deskDrawerInit = true;
            setTimeout(function() {
                if (!window._deskDrawerManual) {
                    drawer.classList.add("collapsed");
                    toggle.classList.add("visible");
                }
            }, 4000);
        }
        if (n) {
            window._deskDrawerManual = true;
            var isCollapsed = drawer.classList.contains("collapsed");
            if (isCollapsed) {
                drawer.classList.remove("collapsed");
                toggle.classList.remove("visible");
                setTimeout(function() {
                    drawer.classList.add("collapsed");
                    toggle.classList.add("visible");
                }, 5000);
            } else {
                drawer.classList.add("collapsed");
                toggle.classList.add("visible");
            }
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("desktop-nav-toggle", "className"),
    Input("desktop-nav-toggle", "n_clicks"),
    prevent_initial_call=False,
)

# ── Price ticker pulse + green/red flash ──────────────────────────────────────
_app_ctx.app.clientside_callback(
    """
    function(children) {
        ["price-ticker", "price-ticker-mobile"].forEach(function(id) {
            var el = document.getElementById(id);
            if (!el) return;
            /* Parse current price from text */
            var m = (el.textContent || "").match(/\\$([\\d.]+)(K|M)?/);
            var newPrice = null;
            if (m) {
                newPrice = parseFloat(m[1]);
                if (m[2] === "K") newPrice *= 1000;
                else if (m[2] === "M") newPrice *= 1000000;
            }
            /* Compare with stored previous price */
            var prev = parseFloat(el.getAttribute("data-prev-price"));
            if (newPrice) el.setAttribute("data-prev-price", newPrice);

            el.classList.remove("price-pulse", "price-flash-green", "price-flash-red");
            void el.offsetWidth;
            el.classList.add("price-pulse");
            if (newPrice && prev && newPrice !== prev) {
                el.classList.add(newPrice > prev ? "price-flash-green" : "price-flash-red");
            }
        });
        return window.dash_clientside.no_update;
    }
    """,
    Output("price-ticker", "className", allow_duplicate=True),
    Input("price-ticker", "children"),
    prevent_initial_call="initial_duplicate",
)


# ══════════════════════════════════════════════════════════════════════════════
# Share modal
# ══════════════════════════════════════════════════════════════════════════════

@callback(
    Output("share-modal", "is_open"),
    Input("share-btn",          "n_clicks"),
    Input("share-btn-mobile",   "n_clicks"),
    Input("share-modal-close",  "n_clicks"),
    State("share-modal",        "is_open"),
    prevent_initial_call=True,
)
def toggle_share_modal(n1, n1m, n2, is_open):
    return not is_open


# ══════════════════════════════════════════════════════════════════════════════
# Callbacks -- image export (client-side, no kaleido/Chrome needed)
# Uses Plotly.downloadImage() which renders in the browser.
# ══════════════════════════════════════════════════════════════════════════════

# Lazy-load hi-res watermark base64 on first export click
_EXPORT_TAB_IDS = ["bubble", "heatmap", "dca", "retire", "supercharge", "citadel"]

@callback(
    Output("wm-b64-store", "data"),
    [Input(f"{t}-export-btn", "n_clicks") for t in _EXPORT_TAB_IDS],
    State("wm-b64-store", "data"),
    prevent_initial_call=True,
)
def _lazy_load_watermarks(*args):
    current = args[-1]
    if current is not None:
        return no_update
    from figures.common import _LOGO_B64_ALL
    return _LOGO_B64_ALL or no_update


_EXPORT_TABS = [
    ("bubble",  "bubble-graph"),
    ("heatmap", "heatmap-graph"),
    ("dca",     "dca-graph"),
    ("retire",  "retire-graph"),
    ("supercharge", "supercharge-graph"),
    ("citadel", "citadel-graph"),
]

for _tab_id, _graph_id in _EXPORT_TABS:
    _app_ctx.app.clientside_callback(
        f"""
        function(n_clicks, fmt, fname, scale, figure, wmStore) {{
            if (!n_clicks) return window.dash_clientside.no_update;
            if (!figure)   return window.dash_clientside.no_update;
            var s = scale || 2;
            var fig = JSON.parse(JSON.stringify(figure));
            /* Swap watermark for hi-res version if available */
            if (wmStore && fig.layout && fig.layout.images) {{
                var wmB64 = wmStore[String(s)];
                if (wmB64) {{
                    for (var i = 0; i < fig.layout.images.length; i++) {{
                        if (fig.layout.images[i].source &&
                            fig.layout.images[i].source.indexOf('data:image/png;base64,') === 0) {{
                            fig.layout.images[i].source = wmB64;
                            break;
                        }}
                    }}
                }}
            }}
            if ((fmt || 'png') === 'html') {{
                var fn = (fname || '{_tab_id}') + '.html';
                var html = '<!DOCTYPE html>\\n<html><head>'
                    + '<meta charset="utf-8">'
                    + '<title>' + fn + ' — Quantoshi</title>'
                    + '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"><\\/script>'
                    + '<style>body{{margin:0;background:{LIGHTBOX_BG}}}'
                    + '#chart{{width:100vw;height:100vh}}</style>'
                    + '</head><body>'
                    + '<div id="chart"></div><script>'
                    + 'Plotly.newPlot("chart",'
                    + JSON.stringify(fig.data) + ','
                    + JSON.stringify(fig.layout) + ','
                    + '{{responsive:true}});'
                    + '<\\/script></body></html>';
                var blob = new Blob([html], {{type: 'text/html'}});
                var a = document.createElement('a');
                a.href = URL.createObjectURL(blob);
                a.download = fn;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(a.href);
                return window.dash_clientside.no_update;
            }}
            Plotly.downloadImage(fig, {{
                format:   fmt   || 'png',
                width:    1920,
                height:   1080,
                scale:    s,
                filename: fname || '{_tab_id}'
            }});
            return window.dash_clientside.no_update;
        }}
        """,
        Output(f"{_tab_id}-dl-dummy", "data"),
        Input(f"{_tab_id}-export-btn", "n_clicks"),
        State(f"{_tab_id}-fmt",        "value"),
        State(f"{_tab_id}-fname",      "value"),
        State(f"{_tab_id}-scale",      "value"),
        State(f"{_tab_id}-graph",      "figure"),
        State("wm-b64-store",          "data"),
        prevent_initial_call=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Palette selector <-> palette-store sync
# ══════════════════════════════════════════════════════════════════════════════

# Select writes to store (clientside -- no server round-trip)
_app_ctx.app.clientside_callback(
    "function(val) { return val || 'default'; }",
    Output("palette-store", "data", allow_duplicate=True),
    Input("palette-select", "value"),
    prevent_initial_call=True,
)

# Store restores select on page load / snapshot restore
_app_ctx.app.clientside_callback(
    "function(data) { return data || 'default'; }",
    Output("palette-select", "value"),
    Input("palette-store", "data"),
    prevent_initial_call=True,
)

# Per-tab palette selectors → store. Each chart tab renders its own
# `palette-select-{tab_key}` dbc.Select via layout.common._palette_selector.
# Forward: selector change writes palette-store. One callback per tab to
# avoid the Dash 4 bug where a single multi-Input callback errors out if
# ANY Input id is missing from the initial layout (e.g. lazy-loaded Citadel
# tab's palette-select-cp). Each callback is simple: val → store.
_TAB_PALETTE_KEYS = ("bub", "hm", "dca", "ret", "sc", "cp")

for _k in _TAB_PALETTE_KEYS:
    _app_ctx.app.clientside_callback(
        "function(val) { return val || 'default'; }",
        Output("palette-store", "data", allow_duplicate=True),
        Input(f"palette-select-{_k}", "value"),
        prevent_initial_call=True,
    )

# Reverse: store change → each per-tab selector (keep them in sync).
# One callback per tab so lazy-loaded tabs don't block others.
for _k in _TAB_PALETTE_KEYS:
    _app_ctx.app.clientside_callback(
        "function(data) { return data || 'default'; }",
        Output(f"palette-select-{_k}", "value", allow_duplicate=True),
        Input("palette-store", "data"),
        prevent_initial_call=True,
    )


# ── data-palette attribute on <html>: keeps CSS selector :root[data-palette]
#    in sync with palette-store after post-load palette switches ──────────────
_app_ctx.app.clientside_callback(
    """
    function(palette_data) {
        if (palette_data && typeof palette_data === "string") {
            document.documentElement.dataset.palette = palette_data;
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("palette-store", "data", allow_duplicate=True),
    Input("palette-store", "data"),
    prevent_initial_call=True,
)


# Note: heatmap color inputs (hm-c-lo/mid1/mid2/hi) are now driven solely
# by callbacks/snapshot_cb.py::apply_hm_palette, which reads both the
# heatmap's own preset dropdown AND the site-wide palette-store. This
# replaces the former _update_hm_colors_on_palette callback here which
# wrote palette-level hm_c_* defaults — those defaults (still present on
# each palette) are now only consulted as fallbacks when no preset is
# active (i.e. user-picked "custom").
