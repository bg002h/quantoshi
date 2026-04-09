/* Remove deeplink-pending class after Dash hydrates the correct tab.
   The inline <style> + <script> in index_string hide #main-tabs until
   this observer detects React has rendered the active tab pane. */
(function() {
    var html = document.documentElement;
    if (!html.classList.contains('deeplink-pending')) return;

    var observer = new MutationObserver(function() {
        if (document.querySelector('#main-tabs .tab-pane.active')) {
            html.classList.remove('deeplink-pending');
            observer.disconnect();
        }
    });
    observer.observe(document.body, {childList: true, subtree: true});

    /* Safety fallback */
    setTimeout(function() {
        html.classList.remove('deeplink-pending');
        observer.disconnect();
    }, 3000);
})();
