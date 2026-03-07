/* Controls drawer — auto-collapse after delay on desktop, hover/tap to expand.
   Jousting lance pin keeps drawer open when pinned. */
(function() {
    "use strict";

    var COLLAPSE_DELAY = 5000;   /* ms before auto-collapse */
    var REARM_DELAY    = 3000;   /* ms after mouse leaves before re-collapse */
    var timers = [];             /* per-column collapse timers */
    var pinned = [];             /* per-column pin state */

    /* Sword-in-stone pin — Unicode dagger + rock */

    function isDesktop() {
        return window.matchMedia("(min-width: 768px)").matches;
    }

    function createPinButton(col, i) {
        var btn = document.createElement("button");
        btn.className = "drawer-pin";
        btn.title = "Pin panel open";
        btn.innerHTML =
            '<span class="pin-sword">\uD83D\uDDE1\uFE0F</span>' +
            '<span class="pin-stone">\uD83E\uDEA8</span>' +
            '<span class="pin-label">pinned</span>';
        btn.addEventListener("click", function(e) {
            e.stopPropagation();
            pinned[i] = !pinned[i];
            btn.classList.toggle("pinned", pinned[i]);
            btn.title = pinned[i] ? "Unpin panel" : "Pin panel open";
            if (pinned[i]) {
                clearTimeout(timers[i]);
                col.classList.remove("drawer-collapsed");
            } else {
                armCollapse(i, col, REARM_DELAY);
            }
        });
        /* Insert as first child so it sits above the scrolling content */
        col.insertBefore(btn, col.firstChild);
        return btn;
    }

    function armCollapse(idx, col, delay) {
        clearTimeout(timers[idx]);
        if (!isDesktop() || pinned[idx]) return;
        timers[idx] = setTimeout(function() {
            col.classList.add("drawer-collapsed");
        }, delay);
    }

    function setup() {
        var cols = document.querySelectorAll(".controls-col");
        if (!cols.length) { setTimeout(setup, 500); return; }

        cols.forEach(function(col, i) {
            timers[i] = null;
            pinned[i] = false;

            createPinButton(col, i);

            /* Initial arm */
            armCollapse(i, col, COLLAPSE_DELAY);

            /* Hover: expand immediately, rearm on leave */
            col.addEventListener("mouseenter", function() {
                col.classList.remove("drawer-collapsed");
                clearTimeout(timers[i]);
            });
            col.addEventListener("mouseleave", function() {
                armCollapse(i, col, REARM_DELAY);
            });

            /* Touch / click on collapsed drawer: toggle */
            col.addEventListener("click", function(e) {
                if (col.classList.contains("drawer-collapsed")) {
                    e.stopPropagation();
                    col.classList.remove("drawer-collapsed");
                    clearTimeout(timers[i]);
                }
            });

            /* Responsive: remove collapsed state if window shrinks to mobile */
            window.matchMedia("(min-width: 768px)").addEventListener("change", function(mq) {
                if (!mq.matches) {
                    col.classList.remove("drawer-collapsed");
                    clearTimeout(timers[i]);
                } else {
                    armCollapse(i, col, COLLAPSE_DELAY);
                }
            });
        });

        /* Re-arm on tab switch (but respect pins) */
        var tabsEl = document.getElementById("main-tabs");
        if (tabsEl) {
            tabsEl.addEventListener("click", function() {
                cols.forEach(function(col, i) {
                    col.classList.remove("drawer-collapsed");
                    clearTimeout(timers[i]);
                    armCollapse(i, col, COLLAPSE_DELAY);
                });
            });
        }
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", setup);
    } else {
        setTimeout(setup, 800);
    }
})();
