/* ── Tor SVG override: switch chart download from PNG to SVG on .onion ────────
   Tor Browser's canvas fingerprinting protection corrupts PNG exports.
   SVG bypasses canvas entirely. Only activates on .onion hostnames. */
(function() {
    "use strict";
    if (!location.hostname.endsWith(".onion")) return;

    function patchPlots() {
        var plots = document.querySelectorAll(".js-plotly-plot");
        for (var i = 0; i < plots.length; i++) {
            var ctx = plots[i]._context;
            if (ctx && ctx.toImageButtonOptions) {
                ctx.toImageButtonOptions.format = "svg";
                delete ctx.toImageButtonOptions.scale;
            }
        }
    }

    /* Patch after Dash renders and on DOM changes (tab switches) */
    var observer = new MutationObserver(patchPlots);
    observer.observe(document.body, {childList: true, subtree: true});
    setTimeout(patchPlots, 2000);
})();
