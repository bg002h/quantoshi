/* Heatmap swipe container — click labels to scroll, highlight active panel. */
(function() {
    "use strict";

    function setup() {
        var wrap = document.getElementById("hm-swipe-wrap");
        var qrLbl = document.getElementById("hm-sw-qr-lbl");
        var mcLbl = document.getElementById("hm-sw-mc-lbl");
        if (!wrap || !qrLbl || !mcLbl) {
            setTimeout(setup, 500);
            return;
        }

        function updateLabels() {
            var atMC = wrap.scrollLeft > wrap.offsetWidth * 0.4;
            qrLbl.style.opacity = atMC ? "0.5" : "1";
            qrLbl.style.fontWeight = atMC ? "normal" : "bold";
            mcLbl.style.opacity = atMC ? "1" : "0.5";
            mcLbl.style.fontWeight = atMC ? "bold" : "normal";
        }

        wrap.addEventListener("scroll", updateLabels);

        qrLbl.addEventListener("click", function() {
            wrap.scrollTo({ left: 0, behavior: "smooth" });
        });
        mcLbl.addEventListener("click", function() {
            wrap.scrollTo({ left: wrap.scrollWidth, behavior: "smooth" });
        });

        /* Expose a scroll-to-MC function for Dash clientside callbacks */
        wrap._scrollToMC = function() {
            setTimeout(function() {
                wrap.scrollTo({ left: wrap.scrollWidth, behavior: "smooth" });
                updateLabels();
            }, 150);
        };
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", setup);
    } else {
        setTimeout(setup, 500);
    }
})();
