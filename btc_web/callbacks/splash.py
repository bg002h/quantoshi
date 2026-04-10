"""Splash modal, easter eggs, journey tracking, knighting ceremony."""

from dash import Input, Output, State

import _app_ctx
from layout.splash import _SPLASH_QUOTES_JS


# JS helper: parse markdown [text](url) → array of React descriptors
# (strings and html.A components). Returned as children of html.P so
# that Dash's React renderer displays clickable links for quotes that
# contain markdown syntax. Must be inlined in each clientside callback.
_PARSE_MD_JS = r"""
function parseMd(s) {
    var out = [];
    var re = /\[([^\]]+)\]\(([^)]+)\)/g;
    var last = 0, m;
    while ((m = RegExp.prototype.exec.call(re, s)) !== null) {
        if (m.index > last) out.push(s.slice(last, m.index));
        out.push({
            namespace: "dash_html_components",
            type: "A",
            props: {children: m[1], href: m[2], target: "_blank", rel: "noopener"}
        });
        last = re.lastIndex;
    }
    if (last < s.length) out.push(s.slice(last));
    return out;
}
function quoteChildren(text) { return ['"'].concat(parseMd(text)).concat(['"']); }
"""


# ══════════════════════════════════════════════════════════════════════════════
# Journey tracker: update milestones in localStorage on every page load
# ══════════════════════════════════════════════════════════════════════════════

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
        /* Returning visitor -- update visits + streak */
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


# ══════════════════════════════════════════════════════════════════════════════
# Journey stats JS snippet (reused by splash quote + easter egg callbacks)
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# Splash quote: show if 6+ hours since last visit (regular quotes only)
# ══════════════════════════════════════════════════════════════════════════════

_app_ctx.app.clientside_callback(
    """
    function(ts_store) {
        """ + _PARSE_MD_JS + """
        var SIX_HOURS = 6 * 3600 * 1000;
        var now = Date.now();
        var last = ts_store ? parseInt(ts_store) : 0;
        var isDev = (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1");
        /* Skip splash when loading a share link — let snapshot restore complete
           uninterrupted. Dismissing the modal during restore causes race conditions. */
        var _h = window.location.hash || "";
        var hasShareHash = _h.indexOf("q3:") !== -1 || _h.indexOf("q2:") !== -1 || _h.indexOf("q1:") !== -1;
        /* Skip splash on deep links — user navigated to a specific tab/item */
        var _p = window.location.pathname.replace(/\/+$/, "") || "/";
        var isDeepLink = _p !== "/" && _p !== "/1";
        if (!isDev && !hasShareHash && !isDeepLink && now - last >= SIX_HOURS) {
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
            return [true, now.toString(), quoteChildren(q[0]), "\\u2014 " + q[1],
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


# ══════════════════════════════════════════════════════════════════════════════
# Easter egg: 6 clicks on logo -> genesis quote in splash modal
# ══════════════════════════════════════════════════════════════════════════════

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

# Next quote button -- cycle through all quotes (genesis first, then regular)
_app_ctx.app.clientside_callback(
    """
    function(n) {
        """ + _PARSE_MD_JS + """
        if (!n) return [window.dash_clientside.no_update, window.dash_clientside.no_update];
        var quotes = """ + _SPLASH_QUOTES_JS + """;
        window._splashIdx = ((window._splashIdx || 0) + 1) % quotes.length;
        var q = quotes[window._splashIdx];
        return [quoteChildren(q[0]), "\\u2014 " + q[1]];
    }
    """,
    Output("splash-quote-text", "children", allow_duplicate=True),
    Output("splash-quote-attr", "children", allow_duplicate=True),
    Input("splash-next", "n_clicks"),
    prevent_initial_call="initial_duplicate",
)


# ══════════════════════════════════════════════════════════════════════════════
# Onion knighting: Accept Knighthood button -> close splash + play ceremony
# ══════════════════════════════════════════════════════════════════════════════

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
