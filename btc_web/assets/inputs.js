/**
 * inputs.js — Enter-key dismissal, visual feedback, and mobile scroll guard.
 *
 * On Enter key press in a number/text input:
 *   1. Blur the input (dismisses mobile keyboard)
 *   2. Flash a brief green border to confirm the value was accepted
 *
 * Scroll guard: prevents accidental checkbox toggles on mobile when user is
 * scrolling vertically through quantile/toggle panels. If finger moves >10px
 * vertically between touchstart and click, the click is suppressed.
 */
document.addEventListener("keydown", function (e) {
    if (e.key !== "Enter") return;
    var el = e.target;
    if (!el || el.tagName !== "INPUT") return;
    var t = (el.type || "").toLowerCase();
    if (t !== "number" && t !== "text") return;

    el.blur();
    el.classList.remove("input-accepted");
    // Force reflow so re-adding the class restarts the animation
    void el.offsetWidth;
    el.classList.add("input-accepted");
});

/* ── Mobile scroll guard for checkboxes ─────────────────────────────────────
   Track touch movement; if vertical displacement exceeds threshold,
   mark the touch as a scroll and suppress the subsequent click event.
*/
(function () {
    var SCROLL_THRESHOLD = 10; // px
    var touchStartY = null;
    var wasScroll = false;

    document.addEventListener("touchstart", function (e) {
        touchStartY = e.touches[0].clientY;
        wasScroll = false;
    }, { passive: true });

    document.addEventListener("touchmove", function (e) {
        if (touchStartY !== null &&
            Math.abs(e.touches[0].clientY - touchStartY) > SCROLL_THRESHOLD) {
            wasScroll = true;
        }
    }, { passive: true });

    document.addEventListener("click", function (e) {
        if (!wasScroll) return;
        var el = e.target;
        // Only guard checkbox inputs and their labels inside checklist grids
        if (el.closest && el.closest(".q-panel-grid, .form-check")) {
            e.preventDefault();
            e.stopPropagation();
        }
        wasScroll = false;
    }, true); // capture phase — fire before React handlers
})();

/* ── Slider touch delay ─────────────────────────────────────────────────────
   On mobile, sliders capture vertical scroll gestures before the browser can
   distinguish scroll from drag.  Fix: on touchstart inside an .rc-slider,
   immediately disable pointer-events (lets scroll pass through).  After a
   short delay, if no significant vertical movement occurred, re-enable the
   slider so horizontal drags work normally.

   Auto-relock: if the slider is armed but no handle drag occurs within
   IDLE_MS (400ms) and no touch is active on the handle, re-lock it so the
   next scroll gesture isn't accidentally captured.
*/
(function () {
    var DELAY_MS       = 60;   // ms before arming slider
    var SCROLL_PX      = 8;    // vertical movement that confirms scroll intent
    var IDLE_MS        = 400;  // ms of no drag activity before re-locking
    var activeSlider   = null;
    var startY         = null;
    var scrollDetected = false;
    var timer          = null;
    var idleTimer      = null;
    var handleTouched  = false; // true while finger is on .rc-slider-handle

    function lockSlider(slider) {
        slider.style.pointerEvents = "none";
    }

    function unlockSlider(slider) {
        slider.style.pointerEvents = "";
    }

    function cleanup() {
        clearTimeout(timer);
        clearTimeout(idleTimer);
        if (activeSlider) {
            unlockSlider(activeSlider);
            activeSlider = null;
        }
        startY = null;
        scrollDetected = false;
        handleTouched = false;
    }

    document.addEventListener("touchstart", function (e) {
        var slider = e.target.closest && e.target.closest(".rc-slider");
        if (!slider) return;

        // Track whether the touch landed on the handle itself
        handleTouched = !!(e.target.closest &&
                           e.target.closest(".rc-slider-handle"));

        // Immediately block the slider so the browser handles the touch
        clearTimeout(idleTimer);
        activeSlider = slider;
        startY = e.touches[0].clientY;
        scrollDetected = false;
        lockSlider(slider);

        // After delay, if finger hasn't scrolled, arm slider for drag
        timer = setTimeout(function () {
            if (!scrollDetected && activeSlider) {
                unlockSlider(activeSlider);
                // Start idle countdown — re-lock if no drag happens
                idleTimer = setTimeout(function () {
                    if (activeSlider && !handleTouched) {
                        lockSlider(activeSlider);
                    }
                }, IDLE_MS);
            }
        }, DELAY_MS);
    }, { passive: true });

    document.addEventListener("touchmove", function (e) {
        if (!activeSlider || startY === null) return;
        if (Math.abs(e.touches[0].clientY - startY) > SCROLL_PX) {
            scrollDetected = true;
        }
        // If the handle is being dragged, cancel the idle re-lock
        if (handleTouched) {
            clearTimeout(idleTimer);
        }
    }, { passive: true });

    document.addEventListener("touchend", function () { cleanup(); },
                              { passive: true });
    document.addEventListener("touchcancel", function () { cleanup(); },
                              { passive: true });
})();
