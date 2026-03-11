/**
 * inputs.js — Enter-key dismissal + visual feedback for numeric inputs.
 *
 * On Enter key press in a number/text input:
 *   1. Blur the input (dismisses mobile keyboard)
 *   2. Flash a brief green border to confirm the value was accepted
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
