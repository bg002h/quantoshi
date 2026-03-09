/* ── Tor SVG override: switch chart download from PNG to SVG on .onion ────────
   Tor Browser's canvas fingerprinting protection corrupts PNG exports.
   SVG bypasses canvas entirely. Only activates on .onion hostnames. */
(function() {
    "use strict";
    if (!location.hostname.endsWith(".onion")) return;

    function patchPlots() {
        var patched = 0;
        var plots = document.querySelectorAll(".js-plotly-plot");
        for (var i = 0; i < plots.length; i++) {
            var ctx = plots[i]._context;
            if (ctx && ctx.toImageButtonOptions && ctx.toImageButtonOptions.format !== "svg") {
                ctx.toImageButtonOptions.format = "svg";
                delete ctx.toImageButtonOptions.scale;
                patched++;
            }
        }
        return patched;
    }

    /* Plotly renders async — poll until contexts are available */
    var attempts = 0;
    var poller = setInterval(function() {
        patchPlots();
        attempts++;
        if (attempts > 30) clearInterval(poller);
    }, 500);

    /* Also patch on DOM changes (tab switches create new graphs) */
    var observer = new MutationObserver(function() {
        setTimeout(patchPlots, 300);
    });
    observer.observe(document.body, {childList: true, subtree: true});
})();
