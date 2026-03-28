/**
 * Raw chart click capture for draw-mode point placement.
 *
 * Converts pixel click coordinates to data coordinates using Plotly's
 * axis transforms (p2d). Bypasses Plotly's trace-snapping behavior
 * so clicks register at the exact tap position, not the nearest data point.
 */
(function() {
    "use strict";

    var _listening = false;

    function attachClickHandler() {
        var graphDiv = document.getElementById("bubble-graph");
        if (!graphDiv || _listening) return;

        // Use the actual plot area (inside axes), not the whole graph div
        graphDiv.addEventListener("click", function(event) {
            // Only capture during draw mode (check the store)
            var storeEl = document.getElementById("draw-mode-store");
            if (!storeEl) return;
            var storeData;
            try {
                // Dash stores data as a JSON string in a hidden div
                // Access via the React fiber/props
                var reactKey = Object.keys(storeEl).find(function(k) {
                    return k.startsWith("__reactFiber") || k.startsWith("__reactInternalInstance");
                });
                if (reactKey) {
                    var fiber = storeEl[reactKey];
                    // Walk up to find memoizedProps with data
                    var node = fiber;
                    for (var i = 0; i < 10 && node; i++) {
                        if (node.memoizedProps && node.memoizedProps.data !== undefined) {
                            storeData = node.memoizedProps.data;
                            break;
                        }
                        node = node.return;
                    }
                }
            } catch(e) {
                return;  // can't read store, skip
            }

            if (!storeData || (storeData.phase !== "placing_p1" && storeData.phase !== "placing_p2")) {
                return;  // not in draw mode
            }

            // Get Plotly layout for axis transforms
            var layout = graphDiv._fullLayout;
            if (!layout || !layout.xaxis || !layout.yaxis) return;

            var xaxis = layout.xaxis;
            var yaxis = layout.yaxis;

            // Convert pixel position to data coordinates
            // event.offsetX/Y is relative to the graph div
            // Subtract the plot area margins to get position relative to axes
            var plotLeft = layout._size.l;
            var plotTop = layout._size.t;
            var px = event.offsetX || event.layerX || 0;
            var py = event.offsetY || event.layerY || 0;

            // p2d converts pixel → data coordinate (handles log axes)
            var dataX = xaxis.p2d(px - plotLeft);
            var dataY = yaxis.p2d(py - plotTop);

            // Sanity check — point must be within plot area
            if (dataX === undefined || dataY === undefined ||
                isNaN(dataX) || isNaN(dataY) ||
                dataX <= 0 || dataY <= 0) {
                return;
            }

            // Write to the raw-chart-click store via Dash's setProps
            var clickStore = document.getElementById("raw-chart-click");
            if (clickStore) {
                // Dash uses setProps on the component — find it via React internals
                var ck = Object.keys(clickStore).find(function(k) {
                    return k.startsWith("__reactFiber") || k.startsWith("__reactInternalInstance");
                });
                if (ck) {
                    var cf = clickStore[ck];
                    var cn = cf;
                    for (var j = 0; j < 10 && cn; j++) {
                        if (cn.memoizedProps && typeof cn.memoizedProps.setProps === "function") {
                            cn.memoizedProps.setProps({
                                data: {t: dataX, price: dataY, ts: Date.now()}
                            });
                            break;
                        }
                        cn = cn.return;
                    }
                }
            }
        }, true);  // capture phase

        _listening = true;
    }

    // Attach after DOM is ready
    if (document.readyState === "complete") {
        setTimeout(attachClickHandler, 500);
    } else {
        window.addEventListener("load", function() {
            setTimeout(attachClickHandler, 500);
        });
    }

    // Also re-attach on Dash page updates (tab switches can remount)
    var observer = new MutationObserver(function() {
        if (!_listening && document.getElementById("bubble-graph")) {
            attachClickHandler();
        }
    });
    observer.observe(document.body, {childList: true, subtree: true});
})();
