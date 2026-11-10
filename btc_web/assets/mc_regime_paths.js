/* Tab 1: remember when the user last clicked a price-regime box.

   The clientside callback layout/mc_controls.py MC_REGIME_PATHS_JS ticks "Show
   individual price paths" when the user's click applies a regime filter. A
   Dash callback cannot tell a user edit from a programmatic write -- a
   share-link restore writes the same prop -- so the click is recorded here
   and consumed there (same pattern as assets/mc_autowiden.js).

   Capture phase: runs before React's handler, so the stamp is already set
   when the value change reaches the callback. Keyboard toggles (Space on the
   focused box) dispatch a click too. */
(function () {
    'use strict';
    document.addEventListener('click', function (e) {
        var t = e.target;
        if (t && t.closest && t.closest('#bub-mc-regime')) {
            window.__qsMcRegimeClickAt = Date.now();
        }
    }, true);
})();
