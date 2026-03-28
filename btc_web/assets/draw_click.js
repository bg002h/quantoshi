/**
 * Raw chart click capture for draw-mode point placement.
 *
 * Converts pixel click coordinates to data coordinates using Plotly's
 * axis transforms (p2d). Sets coordinates on a global variable and
 * clicks a hidden trigger button so Dash picks up the event.
 *
 * Uses event delegation on document.body because React/Dash replaces
 * the #bubble-graph DOM node when the chart callback fires, which would
 * orphan any listeners attached directly to that element.
 */
(function() {
    "use strict";

    document.body.addEventListener("click", function(event) {
        // Only act when the click is inside #bubble-graph
        var wrapperDiv = document.getElementById("bubble-graph");
        if (!wrapperDiv || !wrapperDiv.contains(event.target)) return;

        // Check draw mode via FAB class
        var fab = document.getElementById("user-model-fab");
        if (!fab || !fab.classList.contains("draw-active")) return;

        // In Dash 4, _fullLayout lives on the inner .js-plotly-plot div,
        // not on the outer #bubble-graph wrapper.
        var graphDiv = wrapperDiv.querySelector(".js-plotly-plot") || wrapperDiv;
        var layout = graphDiv._fullLayout;
        if (!layout || !layout.xaxis || !layout.yaxis) return;

        var xaxis = layout.xaxis;
        var yaxis = layout.yaxis;
        var plotLeft = layout._size.l;
        var plotTop = layout._size.t;

        // Get click position relative to the Plotly plot div
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
})();
