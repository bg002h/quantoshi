/* ── #10: Navbar parallax depth effect ───────────────────────────────────────
   Subtle 3D tilt on the navbar responding to mouse position (desktop only).
   Max tilt: ±1.5deg rotateX, ±2deg rotateY — enough to feel alive, not dizzy. */
(function() {
    "use strict";
    if (window.matchMedia("(max-width: 767px)").matches) return;

    var navbar = null;
    var ticking = false;

    function onMove(e) {
        if (ticking) return;
        ticking = true;
        requestAnimationFrame(function() {
            if (!navbar) navbar = document.getElementById("main-navbar");
            if (!navbar) { ticking = false; return; }
            var rect = navbar.getBoundingClientRect();
            /* Normalise cursor position relative to navbar center: -1..+1 */
            var cx = (e.clientX - rect.left) / rect.width  * 2 - 1;
            var cy = (e.clientY - rect.top)  / rect.height * 2 - 1;
            /* Clamp when cursor is outside navbar (still moves, just saturates) */
            cx = Math.max(-1, Math.min(1, cx));
            cy = Math.max(-1, Math.min(1, cy));
            var rotX = -cy * 1.5;   /* tilt forward/back */
            var rotY =  cx * 2.0;   /* tilt left/right   */
            navbar.style.transform =
                "perspective(800px) rotateX(" + rotX + "deg) rotateY(" + rotY + "deg)";
            ticking = false;
        });
    }

    function onLeave() {
        if (!navbar) navbar = document.getElementById("main-navbar");
        if (navbar) navbar.style.transform = "";
    }

    document.addEventListener("mousemove", onMove, {passive: true});
    document.addEventListener("mouseleave", onLeave);
})();
