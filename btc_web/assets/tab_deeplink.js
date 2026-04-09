/* Remove deeplink-pending class after Dash hydrates — allows tab panes to render normally.
   The CSS rule body.deeplink-pending hides non-active panes to prevent the flash of tab 1. */
(function() {
    var observer = new MutationObserver(function() {
        if (document.querySelector('#main-tabs .tab-pane.active')) {
            document.body.classList.remove('deeplink-pending');
            observer.disconnect();
        }
    });
    if (document.body.classList.contains('deeplink-pending')) {
        observer.observe(document.body, {childList: true, subtree: true});
        /* Safety fallback — remove after 2s regardless */
        setTimeout(function() {
            document.body.classList.remove('deeplink-pending');
            observer.disconnect();
        }, 2000);
    }
})();
