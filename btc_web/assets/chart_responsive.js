/* Responsive chart scaling — thicken everything on desktop. */
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

    function scaleChart(gd) {
        if (!gd || !gd.data) return false;
        if (gd._responsiveScaled) return true;

        /* Use _fullLayout for relayout — it has the computed axis properties */
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
            if (lineIdx.length) Plotly.restyle(gd, {'line.width': lineW}, lineIdx);
            if (mkIdx.length)   Plotly.restyle(gd, {'marker.size': mkSz}, mkIdx);
            if (opIdx.length)   Plotly.restyle(gd, {'marker.opacity': opVals}, opIdx);
        } catch(e) { return false; }

        /* ── Layout (grid, axes, fonts) ──────────────────────────── */
        var upd = {};

        Object.keys(lay).forEach(function(k) {
            if (!/^[xy]axis\d*$/.test(k)) return;
            var ax = lay[k] || {};
            if (ax.gridwidth != null)
                upd[k + '.gridwidth'] = ax.gridwidth * GRID_SCALE;
            if (ax._gridWidthInit || ax.gridwidth != null)
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
        } catch(e) { /* relayout failed — traces still scaled */ }

        gd._responsiveScaled = true;
        /* DEBUG — remove after confirming */
        document.title = 'SCALED ' + gd.id + ' t=' + lineIdx.length + ' lay=' + Object.keys(upd).length;
        return true;
    }

    /* Poll until at least one chart is scaled, then slow down */
    var scaled = {};
    var interval = setInterval(function() {
        GRAPH_IDS.forEach(function(id) {
            if (scaled[id]) return;
            var gd = getPlotlyDiv(id);
            if (gd && gd.data && gd.data.length > 0) {
                if (scaleChart(gd)) scaled[id] = true;
            }
        });
    }, 500);
    setTimeout(function() { clearInterval(interval); }, 60000);

    /* MutationObserver for lazy-loaded tabs */
    var observer = new MutationObserver(function() {
        GRAPH_IDS.forEach(function(id) {
            if (scaled[id]) return;
            var gd = getPlotlyDiv(id);
            if (gd && gd.data && gd.data.length > 0) {
                if (scaleChart(gd)) scaled[id] = true;
            }
        });
    });
    observer.observe(document.body, {childList: true, subtree: true});
})();
