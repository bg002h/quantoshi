"""Tab routing, navigation, splash, easter eggs, image export, drawers."""

import dash
from dash import Input, Output, State, callback, ctx, no_update, ALL

import _app_ctx
from layout.splash import _SPLASH_QUOTES_JS
from layout.faq import _FAQ


# ══════════════════════════════════════════════════════════════════════════════
# Viewport width — fires on page load, provides mobile detection for charts
# ══════════════════════════════════════════════════════════════════════════════

_app_ctx.app.clientside_callback(
    "function() { return window.innerWidth; }",
    Output("viewport-width", "data"),
    Input("url", "pathname"),
)


# ══════════════════════════════════════════════════════════════════════════════
# SC mode / display-q toggles
# ══════════════════════════════════════════════════════════════════════════════

@callback(
    Output("sc-mode-a-collapse", "is_open"),
    Output("sc-mode-b-collapse", "is_open"),
    Output("sc-depl-note-collapse", "is_open"),
    Input("sc-mode", "value"),
)
def toggle_sc_mode(mode):
    return mode == "a", mode == "b", mode == "a"


@callback(
    Output("sc-display-q-collapse", "is_open"),
    Input("sc-chart-layout", "value"),
)
def toggle_sc_display_q(layout):
    return "shade" not in (layout or [])


# ══════════════════════════════════════════════════════════════════════════════
# Callback — pathname-based tab routing (/1 … /6)
# ══════════════════════════════════════════════════════════════════════════════

_PATH_TO_TAB = {
    "/1": "bubble", "/2": "heatmap", "/3": "dca",
    "/4": "retire",  "/5": "supercharge", "/6": "stack",
    "/7": "model_info", "/8": "faq",
}
_TAB_TO_PATH = {v: k for k, v in _PATH_TO_TAB.items()}

# Component IDs that belong to each tab (for single-tab share links)
_TAB_CONTROLS = {
    "bubble":      {"bub-qs","bub-xscale","bub-yscale","bub-xrange","bub-yrange",
                    "bub-toggles","bub-bubble-toggles","bub-n-future","bub-ptsize",
                    "bub-ptalpha","bub-stack","bub-show-stack","bub-use-lots","bub-auto-y",
                    "bub-legend-pos","bub-model-show",
                    "scan-price","scan-date","scan-q"},
    "heatmap":     {"hm-entry-yr","hm-entry-q","hm-exit-range","hm-exit-qs","hm-mode",
                    "hm-b1","hm-b2","hm-c-lo","hm-c-mid1","hm-c-mid2","hm-c-hi",
                    "hm-grad","hm-vfmt","hm-cell-fs","hm-toggles","hm-stack","hm-use-lots",
                    "hm-model-show","hm-mc-model-src","hm-active-model"},
    "dca":         {"dca-stack","dca-use-lots","dca-amount","dca-freq","dca-freq-unlock",
                    "dca-infl","dca-yr-range",
                    "dca-disp","dca-toggles","dca-qs",
                    "dca-sc-enable","dca-sc-loan","dca-sc-rate","dca-sc-term",
                    "dca-sc-type","dca-sc-repeats",
                    "dca-sc-entry-mode","dca-sc-custom-price","dca-sc-tax",
                    "dca-sc-rollover","dca-legend-pos","dca-model-show","dca-mc-model-src"},
    "retire":      {"ret-stack","ret-use-lots","ret-wd","ret-freq","ret-freq-unlock",
                    "ret-yr-range",
                    "ret-infl","ret-disp","ret-toggles","ret-legend-pos","ret-qs",
                    "ret-model-show","ret-mc-model-src"},
    "supercharge": {"sc-stack","sc-use-lots","sc-start-yr","sc-d0","sc-d1","sc-d2",
                    "sc-d3","sc-d4","sc-freq","sc-freq-unlock","sc-infl","sc-qs",
                    "sc-mode","sc-wd",
                    "sc-end-yr","sc-target-yr","sc-disp","sc-toggles","sc-legend-pos",
                    "sc-chart-layout","sc-display-q","sc-model-show","sc-mc-model-src"},
    "stack":       set(),
    "model_info":  set(),
    "faq":         set(),
}
# Palette is global — add to every tab so single-tab share links include it
for _tab_set in _TAB_CONTROLS.values():
    _tab_set.add("palette-store")

# MC controls per tab (for single-tab share links)
_TAB_CONTROLS["dca"].update({
    "dca-mc-enable", "dca-mc-start-yr", "dca-mc-entry-q", "dca-mc-years",
    "dca-mc-bins", "dca-mc-regime", "dca-mc-sims", "dca-mc-window", "dca-mc-advanced",
})
_TAB_CONTROLS["retire"].update({
    "ret-mc-enable", "ret-mc-start-yr", "ret-mc-entry-q", "ret-mc-years",
    "ret-mc-bins", "ret-mc-regime", "ret-mc-sims", "ret-mc-window", "ret-mc-advanced",
})
_TAB_CONTROLS["heatmap"].update({
    "hm-mc-enable", "hm-mc-start-yr", "hm-mc-entry-q", "hm-mc-years",
    "hm-mc-bins", "hm-mc-regime", "hm-mc-sims", "hm-mc-window", "hm-mc-advanced",
    "hm-palette",
})
_TAB_CONTROLS["supercharge"].update({
    "sc-mc-enable", "sc-mc-start-yr", "sc-mc-entry-q", "sc-mc-years",
    "sc-mc-bins", "sc-mc-regime", "sc-mc-sims", "sc-mc-window", "sc-mc-advanced",
})

_app_ctx.app.clientside_callback(
    """
    function(pathname, splashOpen) {
        var NU = window.dash_clientside.no_update;
        var map = {"/1":"bubble","/2":"heatmap","/3":"dca",
                   "/4":"retire","/5":"supercharge","/6":"stack",
                   "/7":"model_info","/8":"faq"};
        /* While splash modal is open, defer the tab switch so chart
           callbacks don't fire into a container hidden behind the modal. */
        if (splashOpen) {
            window._pendingTabPath = pathname;
            return NU;
        }
        var p = window._pendingTabPath || pathname;
        if (window._pendingTabPath) {
            /* Splash just closed — force Plotly resize after chart renders
               to handle rapid-dismiss edge case where layout hasn't settled. */
            setTimeout(function() {
                window.dispatchEvent(new Event("resize"));
            }, 1200);
        }
        window._pendingTabPath = null;
        if (p && /^\\/7\\.\\d+$/.test(p)) { return "model_info"; }
        if (p && /^\\/8\\.\\d+$/.test(p)) { return "faq"; }
        var tab = map[p];
        return tab ? tab : NU;
    }
    """,
    Output("main-tabs", "active_tab", allow_duplicate=True),
    Input("url", "pathname"),
    Input("splash-modal", "is_open"),
    prevent_initial_call="initial_duplicate",
)

# ── Journey tracker: update milestones in localStorage on every page load ─────
_app_ctx.app.clientside_callback(
    """
    function(pathname, journey, price) {
        var NU = window.dash_clientside.no_update;
        var now = Date.now();
        var THIRTY_SIX_H = 36 * 3600 * 1000;
        var ONE_DAY = 24 * 3600 * 1000;
        if (!journey || !journey.first_ts) {
            /* First ever visit */
            return {
                first_ts: now, first_price: price || null,
                visits: 1, tabs_seen: [],
                streak_days: 1, last_visit_ts: now, streak_unlocked: false
            };
        }
        /* Returning visitor — update visits + streak */
        journey.visits = (journey.visits || 0) + 1;
        if (!journey.first_price && price) { journey.first_price = price; }
        var gap = now - (journey.last_visit_ts || 0);
        if (gap >= ONE_DAY / 2 && gap <= THIRTY_SIX_H) {
            journey.streak_days = (journey.streak_days || 1) + 1;
            journey.last_visit_ts = now;
            if (journey.streak_days >= 7) { journey.streak_unlocked = true; }
        } else if (gap > THIRTY_SIX_H) {
            journey.streak_days = 1;
            journey.last_visit_ts = now;
        }
        return journey;
    }
    """,
    Output("journey-store", "data"),
    Input("url", "pathname"),
    State("journey-store", "data"),
    State("btc-price-store", "data"),
    prevent_initial_call=False,
)

# ── Journey: track tab visits ────────────────────────────────────────────────
_app_ctx.app.clientside_callback(
    """
    function(tab, journey) {
        if (!tab || !journey) return window.dash_clientside.no_update;
        var seen = journey.tabs_seen || [];
        if (seen.indexOf(tab) === -1) {
            seen.push(tab);
            journey.tabs_seen = seen;
            return journey;
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("journey-store", "data", allow_duplicate=True),
    Input("main-tabs", "active_tab"),
    State("journey-store", "data"),
    prevent_initial_call="initial_duplicate",
)

# ── Easter egg: 6 clicks on logo → genesis quote in splash modal ─────────────
_JOURNEY_BODY = """
        var journey = null;
        try { journey = JSON.parse(localStorage.getItem("journey-store")); } catch(e) {}
        var jText = "";
        var jStyle = {"display":"none"};
        var _jnow = Date.now();
        if (journey && journey.first_ts) {
            var days = Math.floor((_jnow - journey.first_ts) / 86400000);
            var parts = [];
            if (days >= 1) {
                parts.push("\\u2615 Day " + days + " of your Bitcoin journey");
            } else {
                parts.push("\\u2615 Welcome to the rabbit hole");
            }
            if (journey.first_price) {
                var cp = null;
                try {
                    var el = document.getElementById("price-ticker");
                    if (el) {
                        var m = (el.textContent || "").match(/\\$([\\.\\d]+)(K|M)?/);
                        if (m) {
                            cp = parseFloat(m[1]);
                            if (m[2] === "K") cp *= 1000;
                            else if (m[2] === "M") cp *= 1000000;
                        }
                    }
                } catch(e) {}
                if (cp && cp > 0) {
                    var pct = ((cp - journey.first_price) / journey.first_price * 100);
                    var sign = pct >= 0 ? "+" : "";
                    parts.push("\\u20bf was $" + Math.round(journey.first_price).toLocaleString()
                               + " when you first visited (" + sign + pct.toFixed(1) + "%)");
                } else {
                    parts.push("\\u20bf was $" + Math.round(journey.first_price).toLocaleString()
                               + " when you first visited");
                }
            }
            if (journey.visits && journey.visits > 1) {
                parts.push("Visit #" + journey.visits);
            }
            var tabCount = (journey.tabs_seen || []).length;
            if (tabCount > 0 && tabCount < 8) {
                parts.push(tabCount + "/8 tabs explored");
            } else if (tabCount >= 8) {
                parts.push("\\u2b50 All 8 tabs explored!");
            }
            /* Prepend noble title if knighted */
            try {
                var wf = JSON.parse(localStorage.getItem("wizard-flags")) || {};
                if (wf.noble_title) {
                    parts.unshift("\\u2694\\ufe0f " + wf.noble_title);
                }
            } catch(e) {}
            if (parts.length > 0) {
                jText = parts.join("  \\u00b7  ");
                jStyle = {"display":"block", "textAlign":"center", "fontSize":"12px",
                          "color":"#888", "marginTop":"16px", "lineHeight":"1.7"};
            }
        }
"""

# ── Splash quote: show if 6+ hours since last visit (regular quotes only) ─────
_app_ctx.app.clientside_callback(
    """
    function(ts_store) {
        var SIX_HOURS = 6 * 3600 * 1000;
        var now = Date.now();
        var last = ts_store ? parseInt(ts_store) : 0;
        var isDev = (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1");
        if (!isDev && now - last >= SIX_HOURS) {
            var quotes = """ + _SPLASH_QUOTES_JS + """;
            /* Deterministic pseudo-random shuffle using epoch as seed */
            var seed = Math.floor(now / (6 * 3600 * 1000));
            // Mulberry32: fast deterministic PRNG (no crypto needed, just shuffling quotes)
            function mulberry32(a) { return function() {
                a |= 0; a = a + 0x6D2B79F5 | 0;
                var t = Math.imul(a ^ a >>> 15, 1 | a);
                t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
                return ((t ^ t >>> 14) >>> 0) / 4294967296;
            }}
            var rng = mulberry32(seed);
            for (var i = quotes.length - 1; i > 0; i--) {
                var j = Math.floor(rng() * (i + 1));
                var tmp = quotes[i]; quotes[i] = quotes[j]; quotes[j] = tmp;
            }
            var idx = 0;
            var q = quotes[idx];
            """ + _JOURNEY_BODY + """
            /* Hide onion knight button during regular splash */
            var _kw2 = document.getElementById("onion-knight-wrap");
            if (_kw2) _kw2.style.display = "none";
            return [true, now.toString(), '"' + q[0] + '"', "\\u2014 " + q[1],
                    {"display":"none"}, jText, jStyle];
        }
        return [false, window.dash_clientside.no_update,
                window.dash_clientside.no_update, window.dash_clientside.no_update,
                window.dash_clientside.no_update,
                window.dash_clientside.no_update, window.dash_clientside.no_update];
    }
    """,
    Output("splash-modal", "is_open"),
    Output("splash-ts-store", "data"),
    Output("splash-quote-text", "children"),
    Output("splash-quote-attr", "children"),
    Output("splash-next-wrap", "style"),
    Output("journey-stats", "children"),
    Output("journey-stats", "style"),
    Input("splash-ts-store", "data"),
    prevent_initial_call=False,
)

# Dismiss splash
_app_ctx.app.clientside_callback(
    """
    function(n) {
        if (n) { return false; }
        return window.dash_clientside.no_update;
    }
    """,
    Output("splash-modal", "is_open", allow_duplicate=True),
    Input("splash-dismiss", "n_clicks"),
    prevent_initial_call="initial_duplicate",
)


_EGG_JS = """
    function(n) {
        var NU = window.dash_clientside.no_update;
        if (!n) return [NU, NU, NU, NU, NU, NU];
        window._eggClicks = (window._eggClicks || 0) + 1;
        clearTimeout(window._eggTimer);
        if (window._eggClicks >= 6) {
            window._eggClicks = 0;
            window._splashIdx = 0;
            """ + _JOURNEY_BODY + """
            /* Show Accept Knighthood button on .onion (or dev) if not already knighted.
               setTimeout: React re-renders modal children when is_open flips,
               so DOM manipulation must happen after React settles. */
            setTimeout(function() {
                var _wfE = {};
                try { _wfE = JSON.parse(localStorage.getItem("wizard-flags")) || {}; } catch(e) {}
                var _kw = document.getElementById("onion-knight-wrap");
                if (_kw) {
                    var _isOnion = location.hostname.endsWith(".onion");
                    var _isDevE = !_isOnion && location.hostname !== "quantoshi.xyz";
                    _kw.style.display = ((_isOnion || _isDevE) && !_wfE.knighted) ? "block" : "none";
                }
                var _rk = document.getElementById("replay-knight-wrap");
                if (_rk) _rk.style.display = _wfE.knighted ? "block" : "none";
            }, 200);
            return [true,
                    "\\u201cThe Times 03/Jan/2009 Chancellor on brink of second bailout for banks.\\u201d",
                    "\\u2014 Bitcoin Genesis Block",
                    {"display":"inline"}, jText, jStyle];
        }
        window._eggTimer = setTimeout(function() { window._eggClicks = 0; }, 3000);
        return [NU, NU, NU, NU, NU, NU];
    }
"""
_app_ctx.app.clientside_callback(
    _EGG_JS,
    Output("splash-modal", "is_open", allow_duplicate=True),
    Output("splash-quote-text", "children", allow_duplicate=True),
    Output("splash-quote-attr", "children", allow_duplicate=True),
    Output("splash-next-wrap", "style", allow_duplicate=True),
    Output("journey-stats", "children", allow_duplicate=True),
    Output("journey-stats", "style", allow_duplicate=True),
    Input("logo-easter-egg", "n_clicks"),
    prevent_initial_call=True,
)
_app_ctx.app.clientside_callback(
    _EGG_JS,
    Output("splash-modal", "is_open", allow_duplicate=True),
    Output("splash-quote-text", "children", allow_duplicate=True),
    Output("splash-quote-attr", "children", allow_duplicate=True),
    Output("splash-next-wrap", "style", allow_duplicate=True),
    Output("journey-stats", "children", allow_duplicate=True),
    Output("journey-stats", "style", allow_duplicate=True),
    Input("logo-easter-egg-mobile", "n_clicks"),
    prevent_initial_call=True,
)

# Next quote button — cycle through all quotes (genesis first, then regular)
_app_ctx.app.clientside_callback(
    """
    function(n) {
        if (!n) return [window.dash_clientside.no_update, window.dash_clientside.no_update];
        var quotes = """ + _SPLASH_QUOTES_JS + """;
        window._splashIdx = ((window._splashIdx || 0) + 1) % quotes.length;
        var q = quotes[window._splashIdx];
        return ['"' + q[0] + '"', "\\u2014 " + q[1]];
    }
    """,
    Output("splash-quote-text", "children", allow_duplicate=True),
    Output("splash-quote-attr", "children", allow_duplicate=True),
    Input("splash-next", "n_clicks"),
    prevent_initial_call="initial_duplicate",
)

# ── Onion knighting: Accept Knighthood button → close splash + play ceremony ──
_app_ctx.app.clientside_callback(
    """
    function(n) {
        if (!n) return window.dash_clientside.no_update;
        /* Hide the button immediately */
        var kw = document.getElementById("onion-knight-wrap");
        if (kw) kw.style.display = "none";
        /* Play the onion ceremony after modal closes */
        setTimeout(function() {
            if (window._playOnionKnighting) window._playOnionKnighting();
        }, 400);
        return false;
    }
    """,
    Output("splash-modal", "is_open", allow_duplicate=True),
    Input("onion-knight-btn", "n_clicks"),
    prevent_initial_call="initial_duplicate",
)


# ── Replay knighting from easter egg panel ────────────────────────────────────
_app_ctx.app.clientside_callback(
    """
    function(n) {
        if (!n) return window.dash_clientside.no_update;
        if (window._replayKnighting) {
            setTimeout(function() { window._replayKnighting(); }, 400);
        }
        return false;
    }
    """,
    Output("splash-modal", "is_open", allow_duplicate=True),
    Input("replay-knight-link", "n_clicks"),
    prevent_initial_call=True,
)

# ── LT-8c: Welcome message for returning knights ─────────────────────────────
_app_ctx.app.clientside_callback(
    """
    function(ts) {
        try {
            var flags = JSON.parse(localStorage.getItem("wizard-flags") || "{}");
            if (flags.noble_title) {
                return "Welcome back, " + flags.noble_title + ". The quest continues.";
            }
        } catch(e) {}
        return "";
    }
    """,
    Output("knight-welcome", "children"),
    Input("splash-ts-store", "data"),
)


# ── Mobile nav drawer: auto-collapse after 3s, toggle on tap ──────────────────
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


_MODEL_INFO_ITEMS = ["mi-qr", "mi-pl", "mi-lppl", "mi-exp", "mi-s2f", "mi-mc", "mi-compare"]


@callback(
    Output("model-info-accordion", "active_item"),
    Input("url", "pathname"),
    prevent_initial_call=False,
)
def open_model_info_item(pathname):
    """Open a specific Model Info accordion item when pathname is /7.N (1-indexed)."""
    if not pathname or not pathname.startswith("/7."):
        return no_update
    try:
        n = int(pathname[3:])
        if 1 <= n <= len(_MODEL_INFO_ITEMS):
            return _MODEL_INFO_ITEMS[n - 1]
    except (ValueError, IndexError):
        pass
    return no_update


@callback(
    Output("mi-lightbox", "is_open"),
    Output("mi-lightbox-img", "src"),
    Input({"type": "mi-img", "src": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def open_model_info_lightbox(n_clicks_list):
    """Open lightbox modal when a Model Info image is clicked."""
    if not any(n_clicks_list):
        return False, ""
    triggered = ctx.triggered_id
    if triggered and isinstance(triggered, dict):
        return True, triggered["src"]
    return False, ""


@callback(
    Output("faq-accordion", "active_item"),
    Input("url", "pathname"),
    prevent_initial_call=False,
)
def open_faq_item(pathname):
    """Open a specific FAQ accordion item when pathname is /8.N (1-indexed)."""
    if not pathname or not pathname.startswith("/8."):
        return no_update
    try:
        n = int(pathname[3:])
        if 1 <= n <= len(_FAQ):
            return f"faq-{n - 1}"
    except (ValueError, IndexError):
        pass
    return no_update


# Scroll opened FAQ item to top of screen
_app_ctx.app.clientside_callback(
    """
    function(activeItem) {
        if (!activeItem) return window.dash_clientside.no_update;
        setTimeout(function() {
            var acc = document.getElementById('faq-accordion');
            if (!acc) return;
            var items = acc.querySelectorAll('.accordion-item');
            var idx = parseInt((activeItem || '').replace('faq-', ''), 10);
            if (!isNaN(idx) && items[idx]) {
                var y = items[idx].getBoundingClientRect().top + window.pageYOffset - 60;
                window.scrollTo({top: y, behavior: 'smooth'});
            }
        }, 300);
        return window.dash_clientside.no_update;
    }
    """,
    Output("faq-accordion", "className", allow_duplicate=True),
    Input("faq-accordion", "active_item"),
    prevent_initial_call='initial_duplicate',
)


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
# Callbacks — image export (client-side, no kaleido/Chrome needed)
# Uses Plotly.downloadImage() which renders in the browser.
# ══════════════════════════════════════════════════════════════════════════════

_EXPORT_TABS = [
    ("bubble",  "bubble-graph"),
    ("heatmap", "heatmap-graph"),
    ("dca",     "dca-graph"),
    ("retire",  "retire-graph"),
    ("supercharge", "supercharge-graph"),
]

for _tab_id, _graph_id in _EXPORT_TABS:
    _app_ctx.app.clientside_callback(
        f"""
        function(n_clicks, fmt, fname, scale, figure, wmStore) {{
            if (!n_clicks) return window.dash_clientside.no_update;
            if (!figure)   return window.dash_clientside.no_update;
            var s = scale || 2;
            var fig = JSON.parse(JSON.stringify(figure));
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
# Palette selector ↔ palette-store sync
# ══════════════════════════════════════════════════════════════════════════════

# Select writes to store (clientside — no server round-trip)
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


# ══════════════════════════════════════════════════════════════════════════════
# Heatmap model pill bar — click to select active model
# ══════════════════════════════════════════════════════════════════════════════

_HM_PILL_IDS = ["hm-pill-bub"] + [f"hm-pill-{k}" for k in _app_ctx.PRICE_MODELS if k != "bub"]
if _app_ctx._HAS_MARKOV:
    _HM_PILL_IDS.append("hm-pill-mc")


@callback(
    Output("hm-active-model", "data", allow_duplicate=True),
    *[Output(pid, "outline") for pid in _HM_PILL_IDS],
    *[Input(pid, "n_clicks") for pid in _HM_PILL_IDS],
    prevent_initial_call=True,
)
def _hm_pill_click(*args):
    trigger = ctx.triggered_id
    if not trigger:
        raise dash.exceptions.PreventUpdate
    model_key = trigger.replace("hm-pill-", "")
    outlines = [pid != trigger for pid in _HM_PILL_IDS]
    return (model_key, *outlines)


# Sync pill styles on snapshot restore / page load (hm-active-model store update)
@callback(
    *[Output(pid, "outline", allow_duplicate=True) for pid in _HM_PILL_IDS],
    Input("hm-active-model", "data"),
    prevent_initial_call=True,
)
def _hm_pill_sync(model_key):
    model_key = model_key or "bub"
    active_id = f"hm-pill-{model_key}"
    return tuple(pid != active_id for pid in _HM_PILL_IDS)
