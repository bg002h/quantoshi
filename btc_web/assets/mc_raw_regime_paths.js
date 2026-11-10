/* Tab 1 Raw MC: remember when the user last clicked a trend-deviation regime box.

   Same role as assets/mc_regime_paths.js, for the Raw checklist
   #bub-mc-raw-regime: the clientside callback layout/mc_controls.py
   MC_REGIME_PATHS_JS ticks "Show individual price paths" when the user's
   click applies a Raw regime filter, and consumes this stamp. A Dash callback
   cannot tell a user edit from a programmatic write (a share-link restore
   writes the same prop), so the click is recorded here.

   Capture phase: runs before React's handler, so the stamp is already set
   when the value change reaches the callback. Keyboard toggles (Space on the
   focused box) dispatch a click too. */
(function () {
    'use strict';
    document.addEventListener('click', function (e) {
        var t = e.target;
        if (t && t.closest && t.closest('#bub-mc-raw-regime')) {
            window.__qsMcRawRegimeClickAt = Date.now();
        }
    }, true);
})();
