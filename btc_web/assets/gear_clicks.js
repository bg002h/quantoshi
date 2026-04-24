/* Gear icons (⚙️) inside Display Models checklist labels open config modals.
   Because the span sits inside a <label>, the browser's default click-on-label
   behavior toggles the associated checkbox — causing "configure BM/LPPL/HybPPL
   /EPPL" to also uncheck (or check) the model. Suppress the label default on
   any click that originates inside a `.qs-gear` span. Dash's React-attached
   n_clicks handler still fires (React binds directly to the span), so modal
   opening works, but the checkbox toggle is prevented.

   CRITICAL: preventDefault() only — do NOT stopPropagation(). Propagation is
   how React's synthetic-event dispatch reaches the n_clicks listener.
   Original working fix: 9de1b15. */
(function () {
    'use strict';
    document.addEventListener('click', function (e) {
        var t = e.target;
        while (t && t !== document.body) {
            if (t.classList && t.classList.contains('qs-gear')) {
                e.preventDefault();
                return;
            }
            t = t.parentNode;
        }
    }, true);  /* capture phase — run before the label's default action */
})();
