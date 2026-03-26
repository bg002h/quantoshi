/**
 * sc_legend.js — Toggle depletion annotations when legend entries are clicked.
 *
 * Depletion arrows are layout.annotations with `name` set to the legendgroup
 * of their owning model. When a legend entry is clicked, this handler:
 * 1. Looks up the clicked trace's legendgroup via gd.data[curveNumber]
 * 2. Toggles ALL traces in that legendgroup (visible <-> "legendonly")
 * 3. Toggles ALL annotations whose name matches the legendgroup
 * 4. Returns false to prevent Plotly's default toggle (we handle everything)
 */
(function() {
    "use strict";

    function _bind(graphId) {
        var wrapper = document.getElementById(graphId);
        if (!wrapper) return;
        var gd = wrapper.querySelector(".js-plotly-plot") || wrapper;
        if (gd._scLegendBound || typeof gd.on !== "function") return;
        gd._scLegendBound = true;

        gd.on("plotly_legendclick", function(eventData) {
            var clickedTrace = gd.data[eventData.curveNumber];
            if (!clickedTrace || !clickedTrace.legendgroup) return;

            var lg = clickedTrace.legendgroup;
            var wasVisible = clickedTrace.visible !== "legendonly" && clickedTrace.visible !== false;
            var newVis = wasVisible ? "legendonly" : true;
            var newAnnotVis = !wasVisible;

            // Build per-trace visibility update (only traces in this legendgroup)
            var visArray = gd.data.map(function(t) {
                if (t.legendgroup === lg) return newVis;
                return t.visible === undefined ? true : t.visible;
            });

            // Build updated annotations (toggle matching name)
            var newAnnots = (gd.layout.annotations || []).map(function(a) {
                if (a.name === lg) {
                    return Object.assign({}, a, {visible: newAnnotVis});
                }
                return a;
            });

            Plotly.restyle(gd, {visible: visArray});
            Plotly.relayout(gd, {annotations: newAnnots});
            return false;
        });
    }

    // Observe DOM for the supercharge graph (re-bind after Dash re-renders)
    var _timer = null;
    var _observer = new MutationObserver(function() {
        clearTimeout(_timer);
        _timer = setTimeout(function() { _bind("supercharge-graph"); }, 200);
    });
    _observer.observe(document.body, {childList: true, subtree: true});

    document.addEventListener("DOMContentLoaded", function() {
        _bind("supercharge-graph");
    });
})();
