/* After Dash renders the Display Models checklist, each .qs-gear lives
   inside a <label>. Any click inside a <label> toggles its <input>, so
   the modal-open callback is fighting the checkbox toggle. Move every
   gear OUT of its label to be a sibling right after it. Wrap the
   (label, gear) pair in a flex row so they stay on the same line. */
(function () {
    "use strict";

    function hoistGears(root) {
        if (!root || !root.querySelectorAll) return;
        var gears = root.querySelectorAll(".qs-gear");
        for (var i = 0; i < gears.length; i++) {
            var g = gears[i];
            if (g.dataset.hoisted === "1") continue;
            var label = g.closest("label");
            if (!label || !label.parentNode) continue;
            var parent = label.parentNode;

            // Wrap (label, gear) in an inline-flex row keeping them on one line.
            var row = document.createElement("span");
            row.className = "qs-gear-row";
            row.style.display = "inline-flex";
            row.style.alignItems = "center";
            row.style.width = "100%";
            parent.insertBefore(row, label);
            row.appendChild(label);
            row.appendChild(g);
            g.dataset.hoisted = "1";

            // Cancel default # jump (gear is <a href="#">). Don't stop
            // propagation — Dash's n_clicks listener must still fire.
            if (!g.dataset.bound) {
                g.addEventListener("click", function (e) {
                    e.preventDefault();
                });
                g.dataset.bound = "1";
            }
        }
    }

    function run() { hoistGears(document); }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", run);
    } else {
        run();
    }
    new MutationObserver(function (muts) {
        for (var i = 0; i < muts.length; i++) {
            if (muts[i].addedNodes && muts[i].addedNodes.length) {
                run();
                return;
            }
        }
    }).observe(document.body, { childList: true, subtree: true });
})();
