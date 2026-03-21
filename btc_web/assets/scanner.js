/**
 * Model Scanner — radar beacon marker overlay.
 *
 * Uses Plotly's afterplot event to position radar markers. The marker data
 * comes from data attributes set by a Dash clientside callback.
 */
(function() {
    "use strict";

    var COLOR_MAP = {
        "qr":  "247, 147, 26",
        "bub": "0, 212, 255",
        "pl":  "100, 130, 200",
        "lppl":"60, 200, 160",
        "exp": "255, 100, 100",
        "s2f": "180, 120, 200",
        "ef":  "210, 160, 40",
    };

    function updateMarkers() {
        document.querySelectorAll(".radar-marker").forEach(function(el) {
            el.remove();
        });

        var graph = document.getElementById("bubble-graph");
        if (!graph) return;

        var plot = graph.querySelector(".js-plotly-plot");
        if (!plot || !plot._fullLayout) return;

        var xa = plot._fullLayout.xaxis;
        var ya = plot._fullLayout.yaxis;
        if (!xa || !ya || xa._offset === undefined) return;

        // Read scanner inputs
        var priceEl = document.getElementById("scan-price");
        var dateEl = document.getElementById("scan-date");
        if (!priceEl || !dateEl) return;

        var priceVal = priceEl.value;
        var dateStr = dateEl.value;

        // Fall back to live price if empty
        if (!priceVal) {
            var store = document.getElementById("btc-price-store");
            if (store) {
                // Dash stores render data as a hidden element
                // Try multiple approaches
                priceVal = store.getAttribute("data-dash-is-loading") !== null ?
                    null : null;
            }
        }

        var price = parseFloat(priceVal);
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

        // Check bounds
        var plotArea = plot._fullLayout._size;
        if (xPx < xa._offset || xPx > xa._offset + plotArea.w) return;
        if (yPx < ya._offset || yPx > ya._offset + plotArea.h) return;

        // The chart-wrap div has position:relative already
        var container = document.getElementById("bubble-graph-chart-wrap") ||
                        graph.closest("[style*='position']") ||
                        graph;
        container.style.position = "relative";

        // Always show default beacon at current position
        placeMarker(container, xPx, yPx, "0, 212, 255", 0);
    }

    function placeMarker(container, xPx, yPx, colorRgb, idx) {
        var marker = document.createElement("div");
        marker.className = "radar-marker";
        marker.style.left = xPx + "px";
        marker.style.top = yPx + "px";
        marker.style.setProperty("--radar-color-rgb", colorRgb);

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

    // Re-render markers after every chart update
    function init() {
        var graph = document.getElementById("bubble-graph");
        if (!graph) return;

        var checkPlot = setInterval(function() {
            var plot = graph.querySelector(".js-plotly-plot");
            if (plot && plot.on) {
                plot.on("plotly_afterplot", function() {
                    setTimeout(updateMarkers, 150);
                });
                plot.on("plotly_relayout", function() {
                    setTimeout(updateMarkers, 150);
                });
                clearInterval(checkPlot);
                // Initial marker
                setTimeout(updateMarkers, 2000);
            }
        }, 500);

        // Also update when scanner inputs change
        ["scan-price", "scan-date"].forEach(function(id) {
            var el = document.getElementById(id);
            if (el) {
                el.addEventListener("change", function() {
                    setTimeout(updateMarkers, 300);
                });
            }
        });
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        setTimeout(init, 1000);
    }
})();
