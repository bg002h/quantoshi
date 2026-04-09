/* Responsive chart scaling — thicken everything on desktop.
 *
 * Polls for charts every 500ms. Compares current trace widths against
 * a snapshot taken before scaling. Only re-applies when Dash has
 * replaced the figure with fresh (unscaled) data.
 */
(function() {
    'use strict';
    if (window.innerWidth <= 768) return;

    var S = {line: 3.0, marker: 2.0, opacity: 2.0, grid: 3.0, font: 1.8, axis: 2.0};

    var IDS = ['bubble-graph','heatmap-graph','dca-graph',
               'retire-graph','supercharge-graph','citadel-graph'];

    /* Store the scaled line width of trace 0 to detect if figure was replaced */
    var _scaledWidths = {};

    function gd(id) {
        var w = document.getElementById(id);
        if (!w) return null;
        return w.querySelector('.js-plotly-plot') || w;
    }

    function needsScaling(g, id) {
        if (!g || !g.data || g.data.length === 0) return false;
        /* Find first trace with a line width */
        for (var i = 0; i < g.data.length; i++) {
            if (g.data[i].line && g.data[i].line.width != null) {
                var cur = g.data[i].line.width;
                var prev = _scaledWidths[id];
                /* If we've scaled before and current width matches, skip */
                if (prev != null && Math.abs(cur - prev) < 0.1) return false;
                return true;
            }
        }
        return false;
    }

    function scale(g, id) {
        /* Traces */
        var li=[], lw=[], mi=[], ms=[], oi=[], ov=[];
        g.data.forEach(function(t,i) {
            if (t.line && t.line.width != null) { li.push(i); lw.push(t.line.width * S.line); }
            if (t.marker && typeof t.marker.size === 'number') { mi.push(i); ms.push(t.marker.size * S.marker); }
            if (t.marker && typeof t.marker.opacity === 'number') { oi.push(i); ov.push(Math.min(1, t.marker.opacity * S.opacity)); }
        });
        try {
            if (li.length) Plotly.restyle(g, {'line.width': lw}, li);
            if (mi.length) Plotly.restyle(g, {'marker.size': ms}, mi);
            if (oi.length) Plotly.restyle(g, {'marker.opacity': ov}, oi);
        } catch(e) { return; }

        /* Remember the scaled width of first line trace to detect future resets */
        if (lw.length > 0) _scaledWidths[id] = lw[0];

        /* Layout */
        var lay = g._fullLayout || g.layout || {}, u = {};
        Object.keys(lay).forEach(function(k) {
            if (!/^[xy]axis\d*$/.test(k)) return;
            var a = lay[k] || {};
            u[k+'.gridwidth'] = (a.gridwidth||1) * S.grid;
            u[k+'.linewidth'] = (a.linewidth||1) * S.axis;
            if (a.tickfont && a.tickfont.size) u[k+'.tickfont.size'] = Math.round(a.tickfont.size * S.font);
            if (a.title && a.title.font && a.title.font.size) u[k+'.title.font.size'] = Math.round(a.title.font.size * S.font);
            if (a.minor && a.minor.gridwidth != null) u[k+'.minor.gridwidth'] = a.minor.gridwidth * S.grid;
        });
        if (lay.title && lay.title.font && lay.title.font.size) u['title.font.size'] = Math.round(lay.title.font.size * S.font);
        if (lay.legend && lay.legend.font && lay.legend.font.size) u['legend.font.size'] = Math.round(lay.legend.font.size * S.font);
        if (lay.annotations) lay.annotations.forEach(function(a,i) {
            if (a.font && a.font.size) u['annotations['+i+'].font.size'] = Math.round(a.font.size * S.font);
        });
        try { if (Object.keys(u).length) Plotly.relayout(g, u); } catch(e) {}
    }

    setInterval(function() {
        IDS.forEach(function(id) {
            var g = gd(id);
            if (needsScaling(g, id)) scale(g, id);
        });
    }, 500);
})();
