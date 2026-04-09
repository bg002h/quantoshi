/* Responsive chart scaling — thicken lines and enlarge markers on desktop.
 *
 * Plotly figures are built server-side with fixed pixel values tuned for
 * mobile. On desktop (>768px), traces look too thin. This script hooks
 * into Plotly's afterplot event and calls Plotly.restyle() to scale up
 * line widths, marker sizes, and marker opacity.
 *
 * Runs once per chart render. A data attribute prevents re-application
 * on subsequent afterplot events (e.g., hover, zoom).
 */
(function() {
    var DESKTOP_MIN = 768;
    var LINE_SCALE  = 1.5;
    var MARKER_SCALE = 1.3;
    var OPACITY_SCALE = 1.4;

    var GRAPH_IDS = [
        'bubble-graph', 'heatmap-graph', 'dca-graph',
        'retire-graph', 'supercharge-graph', 'citadel-graph'
    ];

    function applyDesktopScaling(gd) {
        if (!gd || !gd.data || window.innerWidth <= DESKTOP_MIN) return;
        if (gd.getAttribute('data-responsive-scaled')) return;

        var lineIndices = [];
        var lineWidths = [];
        var markerIndices = [];
        var markerSizes = [];
        var opacityIndices = [];
        var opacities = [];

        gd.data.forEach(function(tr, i) {
            // Scale line widths
            if (tr.line && tr.line.width != null) {
                lineIndices.push(i);
                lineWidths.push(tr.line.width * LINE_SCALE);
            }
            // Scale marker sizes
            if (tr.marker && tr.marker.size != null && typeof tr.marker.size === 'number') {
                markerIndices.push(i);
                markerSizes.push(tr.marker.size * MARKER_SCALE);
            }
            // Scale marker opacity
            if (tr.marker && tr.marker.opacity != null && typeof tr.marker.opacity === 'number') {
                opacityIndices.push(i);
                opacities.push(Math.min(1.0, tr.marker.opacity * OPACITY_SCALE));
            }
        });

        if (lineIndices.length > 0) {
            Plotly.restyle(gd, {'line.width': lineWidths}, lineIndices);
        }
        if (markerIndices.length > 0) {
            Plotly.restyle(gd, {'marker.size': markerSizes}, markerIndices);
        }
        if (opacityIndices.length > 0) {
            Plotly.restyle(gd, {'marker.opacity': opacities}, opacityIndices);
        }

        gd.setAttribute('data-responsive-scaled', '1');
    }

    function hookGraph(id) {
        var gd = document.getElementById(id);
        if (!gd) return;
        gd.on('plotly_afterplot', function() {
            applyDesktopScaling(gd);
        });
        // Also apply immediately if already rendered
        if (gd.data && gd.data.length > 0) {
            applyDesktopScaling(gd);
        }
    }

    // Hook existing graphs and watch for new ones (lazy-loaded tabs)
    function hookAll() {
        GRAPH_IDS.forEach(hookGraph);
    }

    // Initial hook after DOM settles
    if (document.readyState === 'complete') {
        hookAll();
    } else {
        window.addEventListener('load', hookAll);
    }

    // Re-hook when lazy-loaded tabs inject new graph elements
    var observer = new MutationObserver(function() {
        GRAPH_IDS.forEach(function(id) {
            var gd = document.getElementById(id);
            if (gd && !gd._hasResponsiveHook) {
                gd._hasResponsiveHook = true;
                gd.on('plotly_afterplot', function() {
                    applyDesktopScaling(gd);
                });
            }
        });
    });
    observer.observe(document.body, {childList: true, subtree: true});

    // Clear the flag when figure data changes (callback re-renders chart)
    // so scaling re-applies to the fresh traces.
    var origRestyle = window._plotlyRestyleOrig;
    GRAPH_IDS.forEach(function(id) {
        var check = setInterval(function() {
            var gd = document.getElementById(id);
            if (gd) {
                clearInterval(check);
                var origOn = gd.on;
                gd.on('plotly_afterplot', function() {
                    // If the flag was cleared by a new render, re-apply
                    if (!gd.getAttribute('data-responsive-scaled') && window.innerWidth > DESKTOP_MIN) {
                        applyDesktopScaling(gd);
                    }
                });
                // Watch for figure updates that clear our scaling
                gd.on('plotly_react', function() {
                    gd.removeAttribute('data-responsive-scaled');
                });
            }
        }, 500);
        // Stop checking after 30s
        setTimeout(function() { clearInterval(check); }, 30000);
    });
})();
