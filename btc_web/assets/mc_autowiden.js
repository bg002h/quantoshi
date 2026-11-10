/* Tab 1: remember when the user last clicked the MC "Activate" checklist.

   The MC auto-widen clientside callback (layout/bubble.py MC_AUTOWIDEN_JS)
   widens the X range only when bub-mc-enable was changed by a real click.
   A Dash callback cannot tell a user edit from a programmatic write -- a
   share-link restore writes the same prop -- so the click is recorded here
   and consumed there.

   Capture phase: runs before React's handler, so the stamp is already set
   when the value change reaches the callback. Keyboard toggles (Space on the
   focused box) dispatch a click too. */
(function () {
    'use strict';
    document.addEventListener('click', function (e) {
        var t = e.target;
        if (t && t.closest && t.closest('#bub-mc-enable')) {
            window.__qsMcEnableClickAt = Date.now();
        }
    }, true);
})();
