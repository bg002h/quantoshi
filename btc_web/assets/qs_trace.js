// Snapshot-restore trace instrumentation. Disabled unless URL contains
// ?trace=1 or #trace=1. Stages push timestamps to window.__qsTraceLog;
// dumped via console.table after plotly_afterplot or modal close.
//
// Usage:
//   https://quantoshi.xyz/1?trace=1#q4:...
// Then watch the JS console: a single console.table appears at modal
// close summarizing every stage's relative timing.
(function () {
    var enabled = (window.location.search.indexOf('trace=1') >= 0)
               || (window.location.hash.indexOf('trace=1') >= 0);
    window.__qsTraceEnabled = enabled;
    window.__qsTraceLog = [];

    window.__qsTrace = function (stage, extra) {
        if (!window.__qsTraceEnabled) return;
        var entry = { stage: stage, t: performance.now() };
        if (extra) entry.extra = extra;
        window.__qsTraceLog.push(entry);
    };

    window.__qsDumpTrace = function (label) {
        if (!window.__qsTraceEnabled) return;
        if (!window.__qsTraceLog.length) return;
        var t0 = window.__qsTraceLog[0].t;
        var rows = window.__qsTraceLog.map(function (r, i) {
            var prev = i > 0 ? window.__qsTraceLog[i - 1].t : r.t;
            return {
                stage: r.stage,
                from_start_ms: Math.round(r.t - t0),
                delta_ms: Math.round(r.t - prev),
                extra: r.extra || ''
            };
        });
        console.log('=== Quantoshi trace: ' + (label || 'restore') + ' ===');
        console.table(rows);
        window.__qsTraceLog = [];
    };

    // First trace point: page load. Lets us see how long until hash detected.
    if (enabled) {
        window.__qsTrace('page-load');
    }
})();
