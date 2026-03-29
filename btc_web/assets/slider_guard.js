/**
 * slider_guard.js — Prevent accidental slider activation on mobile.
 *
 * Dash 4's slider (Radix UI) binds onPointerDown on the slider root and
 * calls setPointerCapture + preventDefault, which hijacks scroll gestures
 * when a finger touches anywhere in the slider area.
 *
 * Fix: capture-phase pointerdown listener that stops propagation when the
 * touch lands inside a slider but NOT on a thumb element.  React never sees
 * the event, so the slider stays inert.  We do NOT call preventDefault(),
 * so the browser still handles vertical scrolling natively.
 */
(function () {
    document.addEventListener("pointerdown", function (e) {
        if (e.pointerType === "mouse") return;  // desktop clicks unaffected

        var el = e.target;
        var insideSlider = false;
        var onThumb = false;

        while (el && el !== document.body) {
            if (el.classList.contains("dash-slider-thumb")) onThumb = true;
            if (el.classList.contains("dash-slider-root") ||
                el.classList.contains("rc-slider")) {
                insideSlider = true;
                break;
            }
            el = el.parentElement;
        }

        if (insideSlider && !onThumb) {
            e.stopImmediatePropagation();
        }
    }, true);  // capture phase — fires before React's root handler
})();
