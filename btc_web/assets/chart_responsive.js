/* Chart appearance — applies user-customized trace/grid settings.
 *
 * Reads from the "plot-appearance" localStorage entry (written by
 * Dash dcc.Store(storage_type="local")). Applies via Plotly.restyle
 * (traces) and Plotly.relayout (grids/axes).
 *
 * Critical: uses ABSOLUTE values to avoid the "grows unboundedly on
 * every poll" bug that came from multiplying current values.
 */
(function() {
    'use strict';
    var IS_DESKTOP = window.innerWidth > 768;

    var DEFAULTS = {
        trace_width: 2.5,
        grid_major_width: 1.0,
        grid_major_color: "#888888",
        grid_minor_width: 0.8,
        grid_minor_color: "#B0B0B0",
        pt_color: "#2C3E50",
    };

    /* Desktop multipliers applied to USER-SUPPLIED absolute values only
       (never to current Plotly state — that caused unbounded growth). */
    var DESKTOP = {
        trace_mult: 1.5,
        grid_mult: 1.5,
    };

    var IDS = ['bubble-graph','heatmap-graph','dca-graph',
               'retire-graph','supercharge-graph','citadel-graph'];

    /* Cache of last applied values per chart — to detect figure replacement
       and avoid redundant restyle/relayout calls. */
    var _applied = {};

    function gd(id) {
        var w = document.getElementById(id);
        if (!w) return null;
        return w.querySelector('.js-plotly-plot') || w;
    }

    function getSettings() {
        try {
            var raw = localStorage.getItem("plot-appearance");
            if (raw) return JSON.parse(raw);
        } catch(e) {}
        return DEFAULTS;
    }

    function fingerprint(s) {
        return [s.trace_width, s.grid_major_width, s.grid_major_color,
                s.grid_minor_width, s.grid_minor_color, s.pt_color].join("|");
    }

    function needsApply(g, id, fp) {
        if (!g || !g.data || g.data.length === 0) return false;
        var last = _applied[id];
        if (!last) return true;
        if (last.fp !== fp) return true;
        /* If Dash replaced the figure, line widths will be back to server-side
           values and won't match our target. Re-apply. */
        for (var i = 0; i < g.data.length; i++) {
            if (g.data[i].line && g.data[i].line.width != null) {
                if (Math.abs(g.data[i].line.width - last.targetTraceWidth) > 0.1) return true;
                return false;
            }
        }
        return false;
    }

    function applySettings(g, id, s) {
        /* Target absolute values — computed once, used everywhere */
        var targetTraceWidth = s.trace_width * (IS_DESKTOP ? DESKTOP.trace_mult : 1.0);
        var targetGridMajor  = s.grid_major_width * (IS_DESKTOP ? DESKTOP.grid_mult : 1.0);
        var targetGridMinor  = s.grid_minor_width * (IS_DESKTOP ? DESKTOP.grid_mult : 1.0);

        /* ── Trace restyling ──────────────────────────────────────────── */
        var li=[], lw=[], ptIdx=[], ptColors=[];
        g.data.forEach(function(t, i) {
            if (t.line && t.line.width != null) {
                li.push(i);
                lw.push(targetTraceWidth);
            }
            /* Price data scatter — recolor to user's pt_color */
            if (t.mode === "markers" && t.name === "Price data") {
                ptIdx.push(i);
                ptColors.push(s.pt_color);
            }
        });
        try {
            if (li.length) Plotly.restyle(g, {'line.width': lw}, li);
            if (ptIdx.length) Plotly.restyle(g, {'marker.color': ptColors}, ptIdx);
        } catch(e) { return; }

        /* ── Layout relayout: grids, axis colors/widths ────────────────── */
        var layout = g.layout || {};
        var u = {};
        Object.keys(layout).forEach(function(k) {
            if (!/^[xy]axis\d*$/.test(k)) return;
            var userAx = layout[k] || {};
            u[k + '.gridwidth'] = targetGridMajor;
            u[k + '.gridcolor'] = s.grid_major_color;
            /* Minor grid — only style if user explicitly enabled it */
            if (userAx.minor && userAx.minor.showgrid) {
                u[k + '.minor.gridwidth'] = targetGridMinor;
                u[k + '.minor.gridcolor'] = s.grid_minor_color;
            }
        });
        try {
            if (Object.keys(u).length) Plotly.relayout(g, u);
        } catch(e) {}

        _applied[id] = {fp: fingerprint(s), targetTraceWidth: targetTraceWidth};
    }

    /* Hide static preview images once Plotly has rendered the chart */
    function hidePreviews() {
        IDS.forEach(function(gid) {
            var g = gd(gid);
            if (!g || !g.data || g.data.length === 0) return;
            var name = gid.replace('-graph', '');
            var img = document.getElementById(name + '-preview-img');
            if (img && img.style.display !== 'none') {
                img.style.display = 'none';
            }
        });
    }

    setInterval(function() {
        var s = getSettings();
        var fp = fingerprint(s);
        IDS.forEach(function(id) {
            var g = gd(id);
            if (needsApply(g, id, fp)) applySettings(g, id, s);
        });
        hidePreviews();
    }, 500);
})();
