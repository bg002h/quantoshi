/* Responsive chart scaling — thicken everything on desktop.
 *
 * Uses MutationObserver to detect when Plotly renders SVG, then
 * calls Plotly.restyle + Plotly.relayout to scale up visual weight.
 */
(function() {
    'use strict';
    var DESKTOP_MIN = 768;
    document.title = 'RESPONSIVE JS LOADED w=' + window.innerWidth;
    if (window.innerWidth <= DESKTOP_MIN) return;  /* mobile — skip entirely */

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

    function scaleChart(gd) {
        if (!gd || !gd.data || !gd.layout) return;
        if (gd._responsiveScaled) return;

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

        if (lineIdx.length) Plotly.restyle(gd, {'line.width': lineW}, lineIdx);
        if (mkIdx.length)   Plotly.restyle(gd, {'marker.size': mkSz}, mkIdx);
        if (opIdx.length)   Plotly.restyle(gd, {'marker.opacity': opVals}, opIdx);

        /* ── Layout (grid, axes, fonts) ──────────────────────────── */
        var lay = gd.layout;
        var upd = {};

        Object.keys(lay).forEach(function(k) {
            if (!/^[xy]axis\d*$/.test(k)) return;
            var ax = lay[k] || {};
            if (ax.gridwidth != null)
                upd[k + '.gridwidth'] = ax.gridwidth * GRID_SCALE;
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

        if (Object.keys(upd).length) Plotly.relayout(gd, upd);

        gd._responsiveScaled = true;
    }

    /* Poll for graphs — simpler and more reliable than event hooks */
    function checkAll() {
        GRAPH_IDS.forEach(function(id) {
            var gd = document.getElementById(id);
            if (gd && gd.data && gd.data.length > 0 && !gd._responsiveScaled) {
                scaleChart(gd);
            }
        });
    }

    /* Check periodically for the first 30 seconds */
    var interval = setInterval(checkAll, 500);
    setTimeout(function() { clearInterval(interval); }, 30000);

    /* Also check on any DOM mutation (catches lazy-loaded tabs + callback re-renders) */
    var observer = new MutationObserver(function() {
        GRAPH_IDS.forEach(function(id) {
            var gd = document.getElementById(id);
            if (gd && gd.data && gd.data.length > 0 && !gd._responsiveScaled) {
                scaleChart(gd);
            }
        });
    });
    observer.observe(document.body, {childList: true, subtree: true});

    /* Reset flag when Dash re-renders a figure (plotly_react fires after newPlot) */
    setInterval(function() {
        GRAPH_IDS.forEach(function(id) {
            var gd = document.getElementById(id);
            if (gd && gd._responsiveScaled && gd._fullLayout && gd._fullLayout._replotting) {
                gd._responsiveScaled = false;
            }
        });
    }, 1000);
})();
