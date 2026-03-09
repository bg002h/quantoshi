/* ── Tor SVG override: switch chart download from PNG to SVG on .onion ────────
   Tor Browser's canvas fingerprinting protection corrupts PNG exports.
   SVG bypasses canvas entirely. Only activates on .onion hostnames.
   Intercepts Plotly.downloadImage to force SVG regardless of UI selection. */
(function() {
    "use strict";
    if (!location.hostname.endsWith(".onion")) return;

    function hookPlotly() {
        if (!window.Plotly || !Plotly.downloadImage) return false;
        if (Plotly._svgPatched) return true;
        var orig = Plotly.downloadImage;
        Plotly.downloadImage = function(gd, opts) {
            opts = Object.assign({}, opts || {}, {format: "svg"});
            delete opts.scale;
            return orig.call(this, gd, opts);
        };
        Plotly._svgPatched = true;
        return true;
    }

    /* Plotly loads async — poll until available */
    if (!hookPlotly()) {
        var attempts = 0;
        var poller = setInterval(function() {
            if (hookPlotly() || ++attempts > 30) clearInterval(poller);
        }, 500);
    }
})();
