/* Gear icons are <a href="#"> so they count as interactive content
   (label skips its checkbox-toggle activation behavior). Cancel the
   default navigation to "#" — but DO NOT stop propagation: Dash's
   n_clicks listener must still fire to open the config modal. */
(function () {
    "use strict";
    document.addEventListener("click", function (e) {
        var t = e.target;
        if (t && t.classList && t.classList.contains("qs-gear")
            && t.tagName === "A") {
            e.preventDefault();
        }
    });
})();
