/**
 * Raw chart click capture for draw-mode point placement.
 *
 * Converts pixel click coordinates to data coordinates using Plotly's
 * axis transforms (p2d). Sets coordinates on a global variable and
 * clicks a hidden trigger button so Dash picks up the event.
 */
(function() {
    "use strict";

    var _listening = false;

    function attachClickHandler() {
        var graphDiv = document.getElementById("bubble-graph");
        if (!graphDiv || _listening) return;

        graphDiv.addEventListener("click", function(event) {
            // Check draw mode via a data attribute set by the toast visibility
            // (Simpler than reading React state — the FAB has class "draw-active" during draw mode)
            var fab = document.getElementById("user-model-fab");
            if (!fab || !fab.classList.contains("draw-active")) return;

            var layout = graphDiv._fullLayout;
            if (!layout || !layout.xaxis || !layout.yaxis) return;

            var xaxis = layout.xaxis;
            var yaxis = layout.yaxis;
            var plotLeft = layout._size.l;
            var plotTop = layout._size.t;

            // Get click position relative to the graph div
            var rect = graphDiv.getBoundingClientRect();
            var px = event.clientX - rect.left;
            var py = event.clientY - rect.top;

            // p2d converts pixel → data coordinate (handles log axes)
            var dataX = xaxis.p2d(px - plotLeft);
            var dataY = yaxis.p2d(py - plotTop);

            if (dataX === undefined || dataY === undefined ||
                isNaN(dataX) || isNaN(dataY) ||
                dataX <= 0 || dataY <= 0) {
                return;
            }

            // Store coordinates globally and click the hidden trigger button
            window._rawChartClick = {t: dataX, price: dataY, ts: Date.now()};
            var trigger = document.getElementById("raw-click-trigger");
            if (trigger) trigger.click();
        }, true);

        _listening = true;
    }

    if (document.readyState === "complete") {
        setTimeout(attachClickHandler, 500);
    } else {
        window.addEventListener("load", function() {
            setTimeout(attachClickHandler, 500);
        });
    }

    // Re-attach on DOM changes (tab switches can remount)
    new MutationObserver(function() {
        if (!_listening && document.getElementById("bubble-graph")) {
            attachClickHandler();
        }
    }).observe(document.body, {childList: true, subtree: true});
})();
