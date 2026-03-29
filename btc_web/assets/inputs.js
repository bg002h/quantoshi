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

/* ── Slider scroll guard ───────────────────────────────────────────────────
   Pure CSS solution in style.css — see ".dash-slider-root" rules.
   touch-action: pan-y + ::before overlay on touch devices.
*/
