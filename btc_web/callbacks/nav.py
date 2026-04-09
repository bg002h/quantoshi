"""Navbar drawers, palette sync, image export, share modal, price pulse."""

from dash import Input, Output, State, callback

import _app_ctx


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
                    + '<style>body{{margin:0;background:#1a1a2e}}'
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

# Tab palette selector → store
_app_ctx.app.clientside_callback(
    "function(val) { return val; }",
    Output("palette-store", "data", allow_duplicate=True),
    Input("palette-select-tab", "value"),
    prevent_initial_call=True,
)

# Store → tab palette selector (sync)
_app_ctx.app.clientside_callback(
    "function(data) { return data || 'default'; }",
    Output("palette-select-tab", "value"),
    Input("palette-store", "data"),
    prevent_initial_call=True,
)


# ── Heatmap colors: update 4 color inputs when palette changes ───────────────
@callback(
    Output("hm-c-lo",   "value", allow_duplicate=True),
    Output("hm-c-mid1", "value", allow_duplicate=True),
    Output("hm-c-mid2", "value", allow_duplicate=True),
    Output("hm-c-hi",   "value", allow_duplicate=True),
    Input("palette-store", "data"),
    prevent_initial_call=True,
)
def _update_hm_colors_on_palette(pal_key):
    pal = _app_ctx.PALETTES.get(pal_key or "default", _app_ctx.PALETTES["default"])
    return pal["hm_c_lo"], pal["hm_c_mid1"], pal["hm_c_mid2"], pal["hm_c_hi"]
