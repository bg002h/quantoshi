/* Remove the white cover overlay after Dash hydrates the correct tab.
   The inline <script> in index_string creates #deeplink-cover as a
   full-screen white div that hides the tab-1 flash during hydration. */
(function() {
    function removeCover() {
        var c = document.getElementById('deeplink-cover');
        if (c) c.remove();
    }

    var cover = document.getElementById('deeplink-cover');
    if (!cover) return;

    var observer = new MutationObserver(function() {
        if (document.querySelector('#main-tabs .tab-pane.active.show')) {
            removeCover();
            observer.disconnect();
        }
    });
    observer.observe(document.body, {childList: true, subtree: true, attributes: true});

    /* Safety fallback */
    setTimeout(function() {
        removeCover();
        observer.disconnect();
    }, 4000);
})();
