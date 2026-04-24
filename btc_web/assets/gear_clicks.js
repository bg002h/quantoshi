/* Gear icons open BM/LPPL/HybPPL/EPPL config modals without routing through
   Dash callbacks. Why: per-tab gears live in lazy-loaded tabs, so naming
   them as clientside-callback Inputs produces "nonexistent object" errors
   that block the callback from firing even for the one gear that exists.

   Instead: a single capture-phase document click listener reads the
   clicked gear's data-family attribute and opens the matching modal via
   window.dash_clientside.set_props. Also preventDefault()s the click to
   suppress the <label>'s implicit checkbox-toggle and block any
   href-based navigation/scroll. */
(function () {
    'use strict';
    var FAMILY_TO_MODAL = {
        bm:     'bm-config-modal',
        lppl:   'lppl-config-modal',
        hybppl: 'hybppl-config-modal',
        eppl:   'eppl-config-modal',
    };
    document.addEventListener('click', function (e) {
        var el = e.target;
        while (el && el !== document.body) {
            if (el.classList && el.classList.contains('qs-gear')) {
                e.preventDefault();
                var fam = el.getAttribute('data-family');
                var modalId = FAMILY_TO_MODAL[fam];
                if (modalId && window.dash_clientside && window.dash_clientside.set_props) {
                    window.dash_clientside.set_props(modalId, { is_open: true });
                }
                return;
            }
            el = el.parentNode;
        }
    }, true);  /* capture phase — run before the label's default action */
})();
