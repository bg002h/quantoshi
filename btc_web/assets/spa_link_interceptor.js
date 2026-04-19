/* ── SPA-link delegated click interceptor ───────────────────────────────
   Intercepts clicks on any <a href="/mi.N">, <a href="/faq.N">, or
   <a href="/leverage">, <a href="/N"> — and converts them to SPA
   navigation (pushState + _dashprivate_pushstate) instead of full browser
   navigation, which would hit the Flask static routes and break out of
   the Dash app.

   Required because dcc.Link(refresh=False) inside dbc.Checklist option
   labels does not receive React event binding (the checklist rewrites
   the label subtree and strips event handlers). This is a belt-and-
   suspenders layer that works regardless of how the <a> was authored.

   Installs a SINGLE document-level capture-phase listener at load time.
*/
(function() {
    "use strict";
    if (window.__qs_spa_interceptor_installed) return;
    window.__qs_spa_interceptor_installed = true;

    function isInAppPath(href) {
        if (!href) return false;
        // Only intercept relative paths; leave external links alone.
        if (href.indexOf("://") !== -1) return false;
        if (href[0] !== "/") return false;
        // Named deep-link routes
        if (/^\/(?:mi|faq)(?:\.\d+)?$/.test(href)) return true;
        // Named app routes
        if (href === "/leverage") return true;
        // Numeric app routes like /1, /8, /1.2.5
        if (/^\/\d+(?:\.\d+)*$/.test(href)) return true;
        return false;
    }

    document.addEventListener("click", function(e) {
        // Skip modifier-key clicks (open in new tab, etc.)
        if (e.metaKey || e.shiftKey || e.altKey || e.ctrlKey) return;
        if (e.button !== 0) return;

        // Walk up to find the anchor
        var el = e.target;
        while (el && el !== document && el.tagName !== "A") el = el.parentElement;
        if (!el || el.tagName !== "A") return;

        // target="_blank" or other non-self targets: let browser handle
        var t = el.getAttribute("target");
        if (t && t !== "_self") return;

        var href = el.getAttribute("href");
        if (!isInAppPath(href)) return;

        e.preventDefault();
        if (location.pathname !== href) {
            window.history.pushState({}, "", href);
            window.dispatchEvent(new Event("_dashprivate_pushstate"));
        }
        window.scrollTo(0, 0);
    }, /* capture */ true);
})();
