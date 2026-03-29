/**
 * Force Plotly charts to resize when a tab becomes visible.
 * Without this, charts injected at layout time render at a small
 * default size and only expand after a visible reflow.
 */
(function() {
    "use strict";
    var _tabs = document.getElementById("main-tabs");
    if (!_tabs) {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
    function init() {
        var observer = new MutationObserver(function() {
            // Find all visible Plotly charts and trigger a resize
            var plots = document.querySelectorAll(".tab-pane.active .js-plotly-plot");
            plots.forEach(function(plot) {
                if (plot && typeof Plotly !== "undefined") {
                    Plotly.Plots.resize(plot);
                }
            });
        });
        var tabs = document.getElementById("main-tabs");
        if (tabs) {
            observer.observe(tabs, {attributes: true, subtree: true, attributeFilter: ["class"]});
        }
    }
})();
