/**
 * Model Scanner — radar beacon marker overlay.
 * Positions animated radar markers over the Plotly bubble chart.
 * Default beacon at live price on initial load.
 */
(function() {
    "use strict";

    function hexToRgb(hex) {
        hex = hex.replace("#", "");
        if (hex.length === 3) hex = hex[0]+hex[0]+hex[1]+hex[1]+hex[2]+hex[2];
        var r = parseInt(hex.substring(0,2), 16);
        var g = parseInt(hex.substring(2,4), 16);
        var b = parseInt(hex.substring(4,6), 16);
        return r + ", " + g + ", " + b;
    }

    function updateMarkers() {
        document.querySelectorAll(".radar-marker").forEach(function(el) {
            el.remove();
        });

        var graph = document.getElementById("bub-graph");
        if (!graph) return;

        var plot = graph.querySelector(".js-plotly-plot");
        if (!plot || !plot._fullLayout) return;

        var xa = plot._fullLayout.xaxis;
        var ya = plot._fullLayout.yaxis;
        if (!xa || !ya || !xa._offset) return;

        var priceEl = document.getElementById("scan-price");
        var dateEl = document.getElementById("scan-date");
        if (!priceEl || !dateEl) return;

        var price = parseFloat(priceEl.value);
        var dateStr = dateEl.value;
        if (!price || !dateStr) return;

        var genesis = new Date("2009-07-25T00:00:00");
        var date = new Date(dateStr + "T00:00:00");
        var t = (date - genesis) / (365.25 * 86400000);
        if (t <= 0) return;

        var xVal = xa.type === "log" ? Math.log10(t) : t;
        var yVal = ya.type === "log" ? Math.log10(price) : price;

        var xPx = xa.l2p(xVal) + xa._offset;
        var yPx = ya.l2p(yVal) + ya._offset;

        if (isNaN(xPx) || isNaN(yPx)) return;

        var container = graph.querySelector(".plot-container") || graph;
        container.style.position = "relative";

        // Get active rows
        var store = document.getElementById("scan-active-rows");
        var active = [];
        if (store) {
            try {
                var raw = store.textContent || store.innerText || "[]";
                // Dash stores data as JSON in a specific way
                var parsed = JSON.parse(raw);
                if (Array.isArray(parsed)) active = parsed;
            } catch(e) {
                // Try getting from Dash store props
            }
        }

        // If no active rows, show default beacon at live price
        if (active.length === 0) {
            placeMarker(container, xPx, yPx, "0, 212, 255", 0);
        } else {
            active.forEach(function(modelKey, idx) {
                var color = "0, 212, 255";  // default accent
                // Try to get model color from the scanner table
                var row = document.querySelector(
                    '[id*=\'"model": "' + modelKey + '"\']');
                if (row) {
                    var style = window.getComputedStyle(row);
                    // Could extract color, but simpler to use known palettes
                }
                // Model color mapping
                var colorMap = {
                    "qr": "247, 147, 26",    // BTC orange
                    "bub": "0, 212, 255",     // cyan
                    "pl": "100, 130, 200",    // blue/purple
                    "lppl": "60, 200, 160",   // green/teal
                    "exp": "255, 100, 100",   // red
                    "s2f": "180, 120, 200",   // purple
                    "ef": "210, 160, 40",     // amber
                };
                if (colorMap[modelKey]) color = colorMap[modelKey];

                placeMarker(container, xPx, yPx, color, idx);
            });
        }
    }

    function placeMarker(container, xPx, yPx, colorRgb, idx) {
        var marker = document.createElement("div");
        marker.className = "radar-marker";
        marker.style.left = xPx + "px";
        marker.style.top = yPx + "px";
        marker.style.setProperty("--radar-color-rgb", colorRgb);
        // Scale slightly for multiple markers
        var scale = 1 + idx * 0.3;
        marker.style.width = (40 * scale) + "px";
        marker.style.height = (40 * scale) + "px";

        marker.innerHTML =
            '<div class="radar-ring"></div>' +
            '<div class="radar-sweep"></div>' +
            '<div class="radar-dot"></div>';

        var sweep = marker.querySelector(".radar-sweep");
        sweep.style.animationDelay = (idx * 0.7) + "s";
        var dot = marker.querySelector(".radar-dot");
        dot.style.animationDelay = (idx * 0.7) + "s";

        container.appendChild(marker);
    }

    // Observe scan-active-rows store for changes
    var storeObserver = new MutationObserver(function() {
        setTimeout(updateMarkers, 200);
    });

    // Observe scan-price and scan-date for changes
    var inputObserver = new MutationObserver(function() {
        setTimeout(updateMarkers, 200);
    });

    function init() {
        var store = document.getElementById("scan-active-rows");
        if (store) {
            storeObserver.observe(store, {
                attributes: true, childList: true,
                characterData: true, subtree: true
            });
        }
        ["scan-price", "scan-date"].forEach(function(id) {
            var el = document.getElementById(id);
            if (el) {
                inputObserver.observe(el, {
                    attributes: true, attributeFilter: ["value"]
                });
            }
        });

        // Listen for plotly relayout (zoom/pan)
        var graph = document.getElementById("bub-graph");
        if (graph) {
            var checkPlot = setInterval(function() {
                var plot = graph.querySelector(".js-plotly-plot");
                if (plot && plot.on) {
                    plot.on("plotly_relayout", function() {
                        setTimeout(updateMarkers, 100);
                    });
                    plot.on("plotly_afterplot", function() {
                        setTimeout(updateMarkers, 100);
                    });
                    clearInterval(checkPlot);
                }
            }, 500);
        }

        // Initial marker
        setTimeout(updateMarkers, 2000);
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
