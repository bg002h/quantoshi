/* MC bins dropdowns (all tabs, model + Raw): when the USER raises the bin
   count and every old bin was ticked, tick the new bins too (follow-up F-25).
   Without this the regime value stays at the old bins and the new top bins
   are silently filtered out.

   A Dash callback cannot tell a user edit from a share-link restore (which
   writes bins and regime together), so a real click / Enter inside the bins
   control is stamped here and consumed by window.__qsBinsTick, which the
   clientside callbacks (layout/mc_controls.py::mc_bins_tick_js) call. Same
   pattern as assets/mc_autowiden.js. Dash 4 renders the option list in a
   portal whose id is the trigger's aria-controls, so a click inside that
   popup counts as a click on its trigger.

   Capture phase: the stamp is set before React handles the selection. */
(function () {
    'use strict';
    var BINS_RE = /^(dca|ret|hm|sc|cp|bub)-mc-bins$|^bub-mc-raw-bins$/;
    var FRESH_MS = 3000;

    function binsIdFor(node) {
        for (var n = node; n; n = n.parentElement) {
            var id = n.id;
            if (!id) continue;
            if (BINS_RE.test(id)) return id;
            var trig = document.querySelector('[aria-controls="' + id + '"]');
            if (trig && BINS_RE.test(trig.id || '')) return trig.id;
        }
        return null;
    }

    function stamp(e) {
        var id = binsIdFor(e.target);
        if (!id) return;
        window.__qsBinsUserAt = window.__qsBinsUserAt || {};
        window.__qsBinsUserAt[id] = Date.now();
    }

    document.addEventListener('click', stamp, true);
    document.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' || e.key === ' ') stamp(e);
    }, true);

    /* bins: the new bin count; regime: the regime checklist value; id: the
       bins control id; dflt: its layout default (the previous count before
       the first change). Returns the new regime value or no_update. */
    window.__qsBinsTick = function (bins, regime, id, dflt) {
        var noUpdate = window.dash_clientside.no_update;
        var prevs = window.__qsBinsPrev = window.__qsBinsPrev || {};
        var at = (window.__qsBinsUserAt || {})[id];
        if (window.__qsBinsUserAt) delete window.__qsBinsUserAt[id];
        var prev = prevs[id] !== undefined ? prevs[id] : dflt;
        var n = parseInt(bins, 10);
        if (isNaN(n)) return noUpdate;
        prevs[id] = n;
        if (at === undefined || Date.now() - at > FRESH_MS) return noUpdate;
        if (n <= prev) return noUpdate;
        var have = {};
        (regime || []).forEach(function (v) { have[parseInt(v, 10)] = true; });
        for (var i = 0; i < prev; i++) {
            if (!have[i]) return noUpdate;
        }
        var out = [];
        for (var j = 0; j < n; j++) out.push(j);
        return out;
    };
})();
