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
        try { console.table(rows); } catch (e) { console.log(rows); }

        // Mirror to server log so mobile users can read the trace via
        // journalctl. fetch keepalive=true survives page navigation; sendBeacon
        // is a fallback for older browsers.
        try {
            var body = JSON.stringify(rows);
            var url = '/_trace?label=' + encodeURIComponent(label || 'restore');
            if (navigator.sendBeacon) {
                navigator.sendBeacon(url, body);
            } else {
                fetch(url, { method: 'POST', body: body, keepalive: true,
                             headers: { 'Content-Type': 'application/json' } });
            }
        } catch (e) { /* best-effort */ }

        window.__qsTraceLog = [];
    };

    // First trace point: page load. Lets us see how long until hash detected.
    if (enabled) {
        window.__qsTrace('page-load');
    }
})();
