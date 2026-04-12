/**
 * Model Scanner — chart overlay markers.
 *
 * - Expanding ring at current price/date ("you are here")
 * - Radar beacon on each clicked model row in the scanner results
 */
(function() {
    "use strict";

    // Default fallback radar color (cyan) when palette lookup fails.
    var DEFAULT_COLOR = "0, 212, 255";

    /**
     * Convert a hex color like "#ff0044" to an "r, g, b" triple string
     * suitable for CSS rgba() / --radar-color-rgb variable.
     * Returns null if the input isn't a valid 6-digit hex.
     */
    function hexToRgb(hex) {
        if (!hex || typeof hex !== "string") return null;
        var m = hex.replace("#", "").match(/^[0-9a-fA-F]{6}$/);
        if (!m) return null;
        var n = parseInt(m[0], 16);
        return ((n >> 16) & 255) + ", " + ((n >> 8) & 255) + ", " + (n & 255);
    }

    /**
     * Look up a model's color in the current site-wide palette and return
     * it as an "r, g, b" string. Handles family variants (LPPL/HybPPL/EPPL
     * config flavors) by falling back to their master's color. Reads the
     * active palette from document.documentElement.dataset.palette which
     * is kept in sync with palette-store by nav.py's clientside callback.
     */
    function modelColorRgb(modelKey) {
        if (!modelKey) return DEFAULT_COLOR;
        var palKey = document.documentElement.dataset.palette || "default";
        var palettes = window.QS_PALETTES || {};
        var pal = palettes[palKey] || palettes["default"] || null;
        if (!pal || !pal.model_colors) return DEFAULT_COLOR;
        var mc = pal.model_colors;
        var hex = mc[modelKey];
        // LPPL family variants → lppl master
        if (!hex && /^lp\d/.test(modelKey)) hex = mc["lppl"];
        // HybPPL family extras + linppl → hybppl master
        if (!hex && (modelKey === "linppl" || /^hyb/.test(modelKey))) hex = mc["hybppl"];
        // HybPPL config slots (cfg_*) → hybppl master
        if (!hex && modelKey.indexOf("cfg_") === 0) hex = mc["hybppl"];
        // EPPL config slots (ecfg_*) → eppl master
        if (!hex && modelKey.indexOf("ecfg_") === 0) hex = mc["eppl"];
        if (!hex) return DEFAULT_COLOR;
        return hexToRgb(hex) || DEFAULT_COLOR;
    }

    // Track which model keys have active radar beacons
    var _activeModels = {};

    function getPlotContext() {
        var graph = document.getElementById("bubble-graph");
        if (!graph) return null;
        var plot = graph.querySelector(".js-plotly-plot");
        if (!plot || !plot._fullLayout) return null;
        var xa = plot._fullLayout.xaxis;
        var ya = plot._fullLayout.yaxis;
        if (!xa || !ya || xa._offset === undefined) return null;
        return {plot: plot, xa: xa, ya: ya, size: plot._fullLayout._size};
    }

    function toPixel(ctx, t, price) {
        var xVal = ctx.xa.type === "log" ? Math.log10(t) : t;
        var yVal = ctx.ya.type === "log" ? Math.log10(price) : price;
        var xPx = ctx.xa.l2p(xVal) + ctx.xa._offset;
        var yPx = ctx.ya.l2p(yVal) + ctx.ya._offset;
        if (isNaN(xPx) || isNaN(yPx)) return null;
        // Bounds check
        if (xPx < ctx.xa._offset || xPx > ctx.xa._offset + ctx.size.w) return null;
        if (yPx < ctx.ya._offset || yPx > ctx.ya._offset + ctx.size.h) return null;
        return {x: xPx, y: yPx};
    }

    function getLivePrice() {
        var ticker = document.getElementById("price-ticker");
        if (!ticker) return null;
        var txt = ticker.textContent || "";
        var m = txt.match(/\$([\d,.]+)(K|M)?/);
        if (m) {
            var val = parseFloat(m[1].replace(/,/g, ""));
            if (m[2] === "K") val *= 1e3;
            else if (m[2] === "M") val *= 1e6;
            return val;
        }
        // sats/$ mode
        var sm = txt.match(/([\d,]+)\s*sats\/\$/);
        if (sm) {
            var sats = parseFloat(sm[1].replace(/,/g, ""));
            if (sats > 0) return 1e8 / sats;
        }
        return null;
    }

    function getScannerInputs() {
        var priceEl = document.getElementById("scan-price");
        var dateEl = document.getElementById("scan-date");
        if (!priceEl || !dateEl) return null;

        var priceVal = priceEl.value;
        var dateStr = dateEl.value;

        // Fall back to live price
        if (!priceVal) {
            var live = getLivePrice();
            if (live) priceVal = String(live);
        }

        var price = parseFloat(priceVal);
        if (!price || !dateStr) return null;

        var genesis = new Date("2009-07-25T00:00:00");
        var date = new Date(dateStr + "T00:00:00");
        var t = (date - genesis) / (365.25 * 86400000);
        if (t <= 0) return null;

        return {t: t, price: price};
    }

    // ── Expanding ring at current price ──────────────────────────────────────

    function placeRing(container, xPx, yPx) {
        var ring = document.createElement("div");
        ring.className = "price-ring-marker";
        ring.style.left = xPx + "px";
        ring.style.top = yPx + "px";
        ring.innerHTML =
            '<div class="price-ring-pulse"></div>' +
            '<div class="price-ring-dot"></div>';
        container.appendChild(ring);
    }

    // ── Radar beacon on clicked model rows ───────────────────────────────────

    function placeRadar(container, xPx, yPx, colorRgb) {
        var marker = document.createElement("div");
        marker.className = "radar-marker";
        marker.style.left = xPx + "px";
        marker.style.top = yPx + "px";
        marker.style.setProperty("--radar-color-rgb", colorRgb);
        marker.innerHTML =
            '<div class="radar-ring"></div>' +
            '<div class="radar-sweep"></div>' +
            '<div class="radar-dot"></div>';
        container.appendChild(marker);
    }

    // ── Main update ──────────────────────────────────────────────────────────

    function updateMarkers() {
        document.querySelectorAll(".price-ring-marker, .radar-marker").forEach(function(el) {
            el.remove();
        });

        var ctx = getPlotContext();
        if (!ctx) return;

        ctx.plot.style.position = "relative";

        // 1. Expanding ring at today's live price ("you are here")
        var livePrice = getLivePrice();
        if (livePrice) {
            var genesis = new Date("2009-07-25T00:00:00");
            var now = new Date();
            var tNow = (now - genesis) / (365.25 * 86400000);
            var pos = toPixel(ctx, tNow, livePrice);
            if (pos) placeRing(ctx.plot, pos.x, pos.y);
        }

        // 2. Radar beacons for active model rows
        var rows = document.querySelectorAll("#scan-results tr[data-model]");
        rows.forEach(function(row) {
            var model = row.getAttribute("data-model");
            if (!_activeModels[model]) return;

            var rt = parseFloat(row.getAttribute("data-t"));
            var rp = parseFloat(row.getAttribute("data-price"));
            if (!rt || !rp) return;

            var rPos = toPixel(ctx, rt, rp);
            if (rPos) {
                placeRadar(ctx.plot, rPos.x, rPos.y, modelColorRgb(model));
            }
        });
    }

    // ── Row click handling (event delegation) ────────────────────────────────

    function _rowHighlight(row, model, on) {
        var rgb = modelColorRgb(model);
        if (on) {
            row.style.background = "rgba(" + rgb + ", 0.18)";
            row.style.outline = "1px solid rgba(" + rgb + ", 0.4)";
        } else {
            row.style.background = "";
            row.style.outline = "";
        }
    }

    function handleRowClick(e) {
        var row = e.target.closest("tr[data-model]");
        if (!row) return;
        var model = row.getAttribute("data-model");
        if (_activeModels[model]) {
            delete _activeModels[model];
            _rowHighlight(row, model, false);
        } else {
            _activeModels[model] = true;
            _rowHighlight(row, model, true);
        }
        updateMarkers();
    }

    function reapplyActiveClasses() {
        var rows = document.querySelectorAll("#scan-results tr[data-model]");
        rows.forEach(function(row) {
            var model = row.getAttribute("data-model");
            if (_activeModels[model]) {
                _rowHighlight(row, model, true);
            }
        });
    }

    // ── Init ─────────────────────────────────────────────────────────────────

    var _inputsBound = false;

    function bindInputListeners() {
        if (_inputsBound) return;
        var allFound = true;
        ["scan-price", "scan-date"].forEach(function(id) {
            var el = document.getElementById(id);
            if (el) {
                el.addEventListener("change", function() {
                    setTimeout(updateMarkers, 300);
                });
                el.addEventListener("input", function() {
                    setTimeout(updateMarkers, 300);
                });
            } else {
                allFound = false;
            }
        });
        if (allFound) _inputsBound = true;
    }

    function init() {
        // Watch scan-results for Dash re-renders → re-apply active classes
        var resultsEl = document.getElementById("scan-results");
        if (resultsEl) {
            resultsEl.addEventListener("click", handleRowClick);
            new MutationObserver(function() {
                reapplyActiveClasses();
                setTimeout(updateMarkers, 100);
            }).observe(resultsEl, {childList: true, subtree: true});
        }

        var checkPlot = setInterval(function() {
            var graph = document.getElementById("bubble-graph");
            if (!graph) return;

            var plot = graph.querySelector(".js-plotly-plot");
            if (plot && plot.on) {
                plot.on("plotly_afterplot", function() {
                    setTimeout(updateMarkers, 150);
                });
                plot.on("plotly_relayout", function() {
                    setTimeout(updateMarkers, 150);
                });
                clearInterval(checkPlot);
                bindInputListeners();
                setTimeout(updateMarkers, 2000);
            }
            bindInputListeners();

            // Also try to bind scan-results listener if it appeared late
            if (!resultsEl) {
                resultsEl = document.getElementById("scan-results");
                if (resultsEl) {
                    resultsEl.addEventListener("click", handleRowClick);
                    new MutationObserver(function() {
                        reapplyActiveClasses();
                        setTimeout(updateMarkers, 100);
                    }).observe(resultsEl, {childList: true, subtree: true});
                }
            }
        }, 500);
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        setTimeout(init, 1000);
    }
})();
