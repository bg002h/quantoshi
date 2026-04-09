/* Responsive chart scaling — thicken everything on desktop.
 *
 * Calls Plotly.restyle + Plotly.relayout to scale up traces, grids,
 * fonts on viewports >768px. Re-applies after every figure update.
 */
(function() {
    'use strict';
    var DESKTOP_MIN = 768;
    if (window.innerWidth <= DESKTOP_MIN) return;

    var LINE_SCALE    = 3.0;
    var MARKER_SCALE  = 2.0;
    var OPACITY_SCALE = 2.0;
    var GRID_SCALE    = 3.0;
    var FONT_SCALE    = 1.8;
    var AXIS_SCALE    = 2.0;

    var GRAPH_IDS = [
        'bubble-graph', 'heatmap-graph', 'dca-graph',
        'retire-graph', 'supercharge-graph', 'citadel-graph'
    ];

    function getPlotlyDiv(wrapperId) {
        var wrapper = document.getElementById(wrapperId);
        if (!wrapper) return null;
        return wrapper.querySelector('.js-plotly-plot') || wrapper;
    }

    /* Track when WE are restyling to ignore our own plotly_react events */
    var _scaling = false;

    function scheduleScale(gd, id) {
        if (_scaling) return;  /* ignore events from our own restyle/relayout */
        gd._responsiveScaled = false;
        /* Small delay lets Dash finish its update before we restyle */
        setTimeout(function() { scaleChart(gd); }, 150);
    }

    function scaleChart(gd) {
        if (!gd || !gd.data) return false;
        if (gd._responsiveScaled) return true;

        var lay = gd._fullLayout || gd.layout || {};

        /* ── Traces ──────────────────────────────────────────────── */
        var lineIdx = [], lineW = [];
        var mkIdx = [], mkSz = [];
        var opIdx = [], opVals = [];

        gd.data.forEach(function(tr, i) {
            if (tr.line && tr.line.width != null) {
                lineIdx.push(i);
                lineW.push(tr.line.width * LINE_SCALE);
            }
            if (tr.marker && tr.marker.size != null && typeof tr.marker.size === 'number') {
                mkIdx.push(i);
                mkSz.push(tr.marker.size * MARKER_SCALE);
            }
            if (tr.marker && tr.marker.opacity != null && typeof tr.marker.opacity === 'number') {
                opIdx.push(i);
                opVals.push(Math.min(1.0, tr.marker.opacity * OPACITY_SCALE));
            }
        });

        try {
            _scaling = true;
            if (lineIdx.length) Plotly.restyle(gd, {'line.width': lineW}, lineIdx);
            if (mkIdx.length)   Plotly.restyle(gd, {'marker.size': mkSz}, mkIdx);
            if (opIdx.length)   Plotly.restyle(gd, {'marker.opacity': opVals}, opIdx);
        } catch(e) { _scaling = false; return false; }

        /* ── Layout (grid, axes, fonts) ──────────────────────────── */
        var upd = {};

        Object.keys(lay).forEach(function(k) {
            if (!/^[xy]axis\d*$/.test(k)) return;
            var ax = lay[k] || {};
            upd[k + '.gridwidth'] = (ax.gridwidth || 1) * GRID_SCALE;
            upd[k + '.linewidth'] = (ax.linewidth || 1) * AXIS_SCALE;
            if (ax.tickfont && ax.tickfont.size)
                upd[k + '.tickfont.size'] = Math.round(ax.tickfont.size * FONT_SCALE);
            if (ax.title && ax.title.font && ax.title.font.size)
                upd[k + '.title.font.size'] = Math.round(ax.title.font.size * FONT_SCALE);
            if (ax.minor && ax.minor.gridwidth != null)
                upd[k + '.minor.gridwidth'] = ax.minor.gridwidth * GRID_SCALE;
        });

        if (lay.title && lay.title.font && lay.title.font.size)
            upd['title.font.size'] = Math.round(lay.title.font.size * FONT_SCALE);
        if (lay.legend && lay.legend.font && lay.legend.font.size)
            upd['legend.font.size'] = Math.round(lay.legend.font.size * FONT_SCALE);
        if (lay.annotations) {
            lay.annotations.forEach(function(ann, i) {
                if (ann.font && ann.font.size)
                    upd['annotations[' + i + '].font.size'] = Math.round(ann.font.size * FONT_SCALE);
            });
        }

        try {
            if (Object.keys(upd).length) Plotly.relayout(gd, upd);
        } catch(e) { /* relayout failed */ }
        _scaling = false;

        gd._responsiveScaled = true;
        return true;
    }

    /* Hook a graph div to re-scale after every Dash figure update */
    function hookGraph(id) {
        var gd = getPlotlyDiv(id);
        if (!gd || gd._responsiveHooked) return;
        gd._responsiveHooked = true;

        /* plotly_afterplot fires after Dash replaces the figure */
        gd.on('plotly_afterplot', function() {
            if (!gd._responsiveScaled) {
                scaleChart(gd);
            }
        });

        /* plotly_react fires when Dash calls Plotly.react (new figure data) */
        gd.on('plotly_react', function() {
            scheduleScale(gd, id);
        });

        /* Initial scale */
        if (gd.data && gd.data.length > 0) {
            scaleChart(gd);
        }
    }

    /* Poll to find and hook graphs (handles initial load + lazy tabs) */
    var interval = setInterval(function() {
        GRAPH_IDS.forEach(function(id) {
            hookGraph(id);
        });
    }, 500);
    setTimeout(function() { clearInterval(interval); }, 60000);

    /* MutationObserver for lazy-loaded tabs */
    var observer = new MutationObserver(function() {
        GRAPH_IDS.forEach(function(id) {
            hookGraph(id);
        });
    });
    observer.observe(document.body, {childList: true, subtree: true});
})();
