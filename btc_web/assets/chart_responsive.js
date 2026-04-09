/* Responsive chart scaling — thicken everything on desktop.
 *
 * Plotly figures are built server-side with fixed pixel values tuned for
 * mobile. On desktop (>768px), lines, markers, grids, and fonts look too
 * thin/small. This script hooks into plotly_afterplot and calls
 * Plotly.restyle() + Plotly.relayout() to scale up visual weight.
 */
(function() {
    var DESKTOP_MIN = 768;

    /* ── Scale factors ─────────────────────────────────────────────── */
    var LINE_SCALE    = 2.0;   /* trace line widths */
    var MARKER_SCALE  = 1.5;   /* marker diameters */
    var OPACITY_SCALE = 1.5;   /* marker opacity (capped at 1.0) */
    var GRID_SCALE    = 2.0;   /* grid line widths */
    var FONT_SCALE    = 1.4;   /* axis tick labels, title, legend */
    var AXIS_SCALE    = 1.5;   /* axis line widths */

    var GRAPH_IDS = [
        'bubble-graph', 'heatmap-graph', 'dca-graph',
        'retire-graph', 'supercharge-graph', 'citadel-graph'
    ];

    function scaleFont(obj, key) {
        /* Return scaled font size if present, else undefined */
        if (obj && obj[key] && obj[key].size) {
            return Math.round(obj[key].size * FONT_SCALE);
        }
        return undefined;
    }

    function applyDesktopScaling(gd) {
        if (!gd || !gd.data || window.innerWidth <= DESKTOP_MIN) return;
        if (gd.getAttribute('data-responsive-scaled')) return;

        /* ── Trace scaling (restyle) ───────────────────────────────── */
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

        /* ── Layout scaling (relayout) ─────────────────────────────── */
        var layout = gd.layout || {};
        var updates = {};

        /* Grid + axis lines for xaxis, yaxis, and any secondary axes */
        var axisKeys = Object.keys(layout).filter(function(k) {
            return /^[xy]axis\d*$/.test(k);
        });
        axisKeys.forEach(function(ak) {
            var ax = layout[ak] || {};
            if (ax.gridwidth != null) {
                updates[ak + '.gridwidth'] = ax.gridwidth * GRID_SCALE;
            }
            if (ax.linewidth != null) {
                updates[ak + '.linewidth'] = ax.linewidth * AXIS_SCALE;
            } else {
                updates[ak + '.linewidth'] = 1.5;  /* default is ~1, bump to 1.5 */
            }
            if (ax.tickfont && ax.tickfont.size) {
                updates[ak + '.tickfont.size'] = Math.round(ax.tickfont.size * FONT_SCALE);
            }
            if (ax.title && ax.title.font && ax.title.font.size) {
                updates[ak + '.title.font.size'] = Math.round(ax.title.font.size * FONT_SCALE);
            }
            /* Minor grid */
            if (ax.minor && ax.minor.gridwidth != null) {
                updates[ak + '.minor.gridwidth'] = ax.minor.gridwidth * GRID_SCALE;
            }
        });

        /* Title font */
        if (layout.title && layout.title.font && layout.title.font.size) {
            updates['title.font.size'] = Math.round(layout.title.font.size * FONT_SCALE);
        }

        /* Legend font */
        if (layout.legend && layout.legend.font && layout.legend.font.size) {
            updates['legend.font.size'] = Math.round(layout.legend.font.size * FONT_SCALE);
        }

        /* Annotation fonts */
        if (layout.annotations && layout.annotations.length) {
            layout.annotations.forEach(function(ann, i) {
                if (ann.font && ann.font.size) {
                    updates['annotations[' + i + '].font.size'] = Math.round(ann.font.size * FONT_SCALE);
                }
            });
        }

        if (Object.keys(updates).length) {
            Plotly.relayout(gd, updates);
        }

        gd.setAttribute('data-responsive-scaled', '1');
    }

    /* ── Hooking ───────────────────────────────────────────────────── */

    function hookGraph(id) {
        var gd = document.getElementById(id);
        if (!gd) return;
        gd._hasResponsiveHook = true;
        gd.on('plotly_afterplot', function() {
            applyDesktopScaling(gd);
        });
        gd.on('plotly_react', function() {
            gd.removeAttribute('data-responsive-scaled');
        });
        if (gd.data && gd.data.length > 0) {
            applyDesktopScaling(gd);
        }
    }

    function hookAll() {
        GRAPH_IDS.forEach(hookGraph);
    }

    if (document.readyState === 'complete') {
        hookAll();
    } else {
        window.addEventListener('load', hookAll);
    }

    /* Watch for lazy-loaded tabs injecting new graph elements */
    var observer = new MutationObserver(function() {
        GRAPH_IDS.forEach(function(id) {
            var gd = document.getElementById(id);
            if (gd && !gd._hasResponsiveHook) {
                hookGraph(id);
            }
        });
    });
    observer.observe(document.body, {childList: true, subtree: true});
})();
