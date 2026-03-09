/* Re-render chart when user taps the already-active tab.
   dbc.Tabs doesn't re-fire active_tab on same-tab clicks,
   so Plotly charts that rendered incorrectly stay broken.
   A window resize event triggers Plotly's responsive handler. */
(function() {
    "use strict";
    function setup() {
        var tabs = document.getElementById("main-tabs");
        if (!tabs) { setTimeout(setup, 500); return; }
        tabs.addEventListener("click", function(e) {
            var link = e.target.closest(".nav-link");
            if (link && link.classList.contains("active")) {
                setTimeout(function() {
                    window.dispatchEvent(new Event("resize"));
                }, 200);
            }
        });
    }
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", setup);
    } else {
        setTimeout(setup, 500);
    }
})();
