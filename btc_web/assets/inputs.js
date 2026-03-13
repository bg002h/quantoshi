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
/* ── Slider scroll guard ───────────────────────────────────────────────────
   CSS sets .rc-slider { touch-action: pan-y } which tells the browser to
   handle vertical scrolling natively through slider areas.  Only
   deliberately horizontal movements activate the slider.  No JS delay
   guard is needed — the browser's built-in gesture disambiguation
   handles the scroll-vs-drag distinction reliably.
*/
