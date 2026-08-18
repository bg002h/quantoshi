/* Slider touch scroll-guard (coarse-pointer / touch devices only).
 *
 * Why JS on top of the CSS touch-action guard:
 *   `touch-action` is evaluated by the browser at *touchstart*, on the element
 *   under the finger at that instant. That covers "grab the handle" but NOT a
 *   drag that STARTS on the handle and then drifts vertically onto the value
 *   label (or any margin the pseudo-element misses) — mid-gesture the browser
 *   can still commit to a page scroll. CSS alone cannot fix the whole gesture.
 *
 * What this does:
 *   If a touch STARTS on a slider (thumb / enlarged ::after grab zone / track /
 *   value label — anything inside `.dash-slider-root`, or the tooltip), we
 *   preventDefault() every touchmove of that gesture, which blocks page
 *   scrolling for its full duration. Radix (dcc.Slider) drives the drag with
 *   Pointer Events + setPointerCapture, which are NOT cancelled by touchmove
 *   preventDefault, so the slider keeps tracking the finger.
 *
 * Two deliberate properties (fixed 2026-08-18):
 *   1. Detection is by the actual touch TARGET (`e.target.closest(...)`), which
 *      is z-order aware: a tap on a modal/button that merely happens to sit
 *      above a slider is NOT guarded (an earlier proximity check reached through
 *      modals and could eat clicks near sliders).
 *   2. The non-passive touchmove listener is attached ONLY for the duration of a
 *      real slider gesture, not page-wide — so ordinary taps/scrolls elsewhere
 *      keep the browser's fast passive path.
 *
 * Scoped to pointer:coarse so desktop mouse is completely untouched.
 */
(function () {
  if (!window.matchMedia || !window.matchMedia('(pointer: coarse)').matches) return;

  function onSlider(el) {
    return !!(el && el.closest && el.closest('.dash-slider-root, .dash-slider-tooltip'));
  }
  function block(e) { if (e.cancelable) e.preventDefault(); }
  function endGesture() {
    document.removeEventListener('touchmove', block, { passive: false, capture: true });
  }

  document.addEventListener('touchstart', function (e) {
    endGesture(); // clear any stale listener from an interrupted gesture
    if (onSlider(e.target)) {
      document.addEventListener('touchmove', block, { passive: false, capture: true });
    }
  }, { passive: true, capture: true });

  document.addEventListener('touchend', endGesture, { passive: true, capture: true });
  document.addEventListener('touchcancel', endGesture, { passive: true, capture: true });
})();
