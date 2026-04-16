/**
 * Force cache: "no-store" on Dash internal API requests.
 *
 * iOS Safari aggressively caches /_dash-dependencies and /_dash-layout
 * responses even with Cache-Control: no-cache, no-store headers.  When
 * the callback graph changes between deploys, a stale cached response
 * causes IndexError on every callback request (the browser sends the
 * wrong number of Input/State values).
 *
 * This monkey-patch intercepts fetch() calls to Dash internal endpoints
 * and forces cache:"no-store", which bypasses all browser caches
 * including iOS Safari's BFCache.
 */
(function () {
    var _origFetch = window.fetch;
    var _dashPaths = ["/_dash-dependencies", "/_dash-layout"];

    window.fetch = function (input, init) {
        var url = typeof input === "string" ? input : (input && input.url) || "";
        var isDash = _dashPaths.some(function (p) {
            return url.indexOf(p) !== -1;
        });
        if (isDash) {
            init = Object.assign({}, init || {}, { cache: "no-store" });
        }
        return _origFetch.call(this, input, init);
    };
})();
