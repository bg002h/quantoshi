/* ── Streak reward: orange Q after 7-day visit streak ───────────────────────
   Reads journey-store from localStorage on page load. If streak_unlocked is
   true, adds .streak-unlocked to all .brand-q elements permanently. */
(function() {
    "use strict";
    var _isDev = (location.hostname !== "quantoshi.xyz" &&
                  !location.hostname.endsWith(".onion"));

    function applyStreak() {
        /* On dev, skip auto-apply — knighting.js handles the transition */
        if (_isDev) return;
        try {
            var j = JSON.parse(localStorage.getItem("journey-store"));
            if (j && j.streak_unlocked) {
                var els = document.querySelectorAll(".brand-q");
                for (var i = 0; i < els.length; i++) {
                    els[i].classList.add("streak-unlocked");
                }
                var brands = document.querySelectorAll(".brand-uantoshi");
                for (var b = 0; b < brands.length; b++) {
                    brands[b].classList.add("streak-shimmer");
                }
            }
        } catch(e) {}
    }
    /* Run once DOM is ready, and again after 2s in case Dash renders late */
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", applyStreak);
    } else {
        applyStreak();
    }
    setTimeout(applyStreak, 2000);

    /* Also listen for storage changes (journey-store updated by Dash callback) */
    var orig = Storage.prototype.setItem;
    Storage.prototype.setItem = function(key, val) {
        orig.call(this, key, val);
        if (key === "journey-store" && !_isDev) {
            try {
                var j = JSON.parse(val);
                if (j && j.streak_unlocked) {
                    var els = document.querySelectorAll(".brand-q");
                    for (var i = 0; i < els.length; i++) {
                        els[i].classList.add("streak-unlocked");
                    }
                    var brands = document.querySelectorAll(".brand-uantoshi");
                    for (var b = 0; b < brands.length; b++) {
                        brands[b].classList.add("streak-shimmer");
                    }
                }
            } catch(e) {}
        }
    };
})();
