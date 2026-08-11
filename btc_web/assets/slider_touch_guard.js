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
 *   If a touch STARTS on (or within GRAB_RADIUS of) a slider thumb or its value
 *   label, we preventDefault() every touchmove of that gesture, which blocks
 *   page scrolling for its full duration. Radix (dcc.Slider) drives the drag
 *   with Pointer Events + setPointerCapture, which are NOT cancelled by
 *   touchmove preventDefault, so the slider keeps tracking the finger. This is
 *   the same scroll suppression the thumb's own `touch-action:none` already
 *   applies (and coexists with a working drag) — just extended to the whole
 *   grab region and the whole gesture.
 *
 * Scoped to pointer:coarse so desktop mouse is completely untouched.
 */
(function () {
  if (!window.matchMedia || !window.matchMedia('(pointer: coarse)').matches) return;

  var GRAB_RADIUS = 26; // px beyond a thumb/label that still counts as "the handle"
  var guarding = false;

  function nearGrab(x, y) {
    var els = document.querySelectorAll('.dash-slider-thumb, .dash-slider-tooltip');
    for (var i = 0; i < els.length; i++) {
      var r = els[i].getBoundingClientRect();
      if (r.width === 0 && r.height === 0) continue; // hidden (label not shown yet)
      var dx = x < r.left ? r.left - x : (x > r.right ? x - r.right : 0);
      var dy = y < r.top ? r.top - y : (y > r.bottom ? y - r.bottom : 0);
      if (dx * dx + dy * dy <= GRAB_RADIUS * GRAB_RADIUS) return true;
    }
    return false;
  }

  document.addEventListener('touchstart', function (e) {
    var t = e.touches && e.touches[0];
    guarding = !!t && nearGrab(t.clientX, t.clientY);
  }, { passive: true, capture: true });

  document.addEventListener('touchmove', function (e) {
    if (guarding && e.cancelable) e.preventDefault();
  }, { passive: false, capture: true });

  function release() { guarding = false; }
  document.addEventListener('touchend', release, { passive: true, capture: true });
  document.addEventListener('touchcancel', release, { passive: true, capture: true });
})();
