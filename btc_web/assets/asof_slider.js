// Time Machine as-of slider: tooltip transform.
// Dash dcc.Slider tooltip {"transform": "asofDate"} calls this with the slider
// VALUE (a frame index) and shows whatever string it returns on the handle.
// We render the frame's calendar DATE instead of the raw index. The frames
// list is kept on a window global, synced from the bub-asof-frames Store by a
// clientside callback in callbacks/timemachine.py (#4b). Falls back to the raw
// value until the sync has run.
window.dccFunctions = window.dccFunctions || {};
window.dccFunctions.asofDate = function (value) {
    var f = window.__QS_ASOF_FRAMES;
    if (f && f[value] != null) { return f[value]; }
    return value;
};
