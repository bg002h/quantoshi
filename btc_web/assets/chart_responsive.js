/* Chart appearance — applies user-customized trace/grid settings to all charts.
 *
 * Reads from the "plot-appearance" dcc.Store (localStorage key
 * "_dash_persistence" under the hood). Applies via Plotly.restyle
 * (traces) and Plotly.relayout (grids/axes/fonts).
 *
 * Desktop gets additional scaling on top of user settings (×2 for
 * marker size/opacity, ×1.5 font sizes, etc.) since mobile defaults
 * are tuned for small screens.
 */
(function() {
    'use strict';
    var IS_DESKTOP = window.innerWidth > 768;

    /* Default values — match btc_web/callbacks/plot_appearance.py _DEFAULTS */
    var DEFAULTS = {
        trace_width: 2.5,
        grid_major_width: 1.0,
        grid_major_color: "#888888",
        grid_minor_width: 0.8,
        grid_minor_color: "#B0B0B0",
        bm_color: "#C8960C",
    };

    /* Desktop multipliers applied on top of user values */
    var DESKTOP = {
        marker: 2.0, opacity: 2.0, font: 1.8, axis: 2.0,
        trace_mult: 1.5, grid_mult: 1.5,  /* additional boost over user value */
    };

    var IDS = ['bubble-graph','heatmap-graph','dca-graph',
               'retire-graph','supercharge-graph','citadel-graph'];

    /* Track what we last applied per chart to detect Dash-driven re-renders */
    var _lastApplied = {};

    function gd(id) {
        var w = document.getElementById(id);
        if (!w) return null;
        return w.querySelector('.js-plotly-plot') || w;
    }

    /* Read user settings from localStorage (dcc.Store persistence key) */
    function getUserSettings() {
        try {
            /* dcc.Store with storage_type="local" stores under
               "_dash_persistence" keyed by component id */
            var raw = localStorage.getItem("plot-appearance");
            if (raw) return JSON.parse(raw);
        } catch(e) {}
        return DEFAULTS;
    }

    function settingsFingerprint(s) {
        return [s.trace_width, s.grid_major_width, s.grid_major_color,
                s.grid_minor_width, s.grid_minor_color, s.bm_color].join("|");
    }

    function needsApply(g, id, fp) {
        if (!g || !g.data || g.data.length === 0) return false;
        var last = _lastApplied[id];
        if (!last) return true;
        if (last.fp !== fp) return true;  /* user changed settings */
        /* Check if Dash reset the trace widths */
        for (var i = 0; i < g.data.length; i++) {
            if (g.data[i].line && g.data[i].line.width != null) {
                if (Math.abs(g.data[i].line.width - last.traceWidth) > 0.1) return true;
                return false;
            }
        }
        return false;
    }

    function applySettings(g, id, s) {
        var traceWidth = s.trace_width * (IS_DESKTOP ? DESKTOP.trace_mult : 1.0);
        var gridMajor  = s.grid_major_width * (IS_DESKTOP ? DESKTOP.grid_mult : 1.0);
        var gridMinor  = s.grid_minor_width * (IS_DESKTOP ? DESKTOP.grid_mult : 1.0);

        /* ── Traces: set absolute line width + BM color ──────────────── */
        var li=[], lw=[], mi=[], ms=[], oi=[], ov=[];
        var bmIdx=[], bmColors=[];
        g.data.forEach(function(t,i) {
            if (t.line && t.line.width != null) {
                li.push(i);
                lw.push(traceWidth);
            }
            if (IS_DESKTOP && t.marker && typeof t.marker.size === 'number') {
                mi.push(i);
                ms.push(t.marker.size * DESKTOP.marker);
            }
            if (IS_DESKTOP && t.marker && typeof t.marker.opacity === 'number') {
                oi.push(i);
                ov.push(Math.min(1, t.marker.opacity * DESKTOP.opacity));
            }
            /* Recolor BM traces to user's chosen color. BM trace names
               start with "BM" (quantile bands) or "Bubble" (support/composite). */
            var name = t.name || "";
            if ((name.indexOf("BM") === 0 || name.indexOf("Bubble") === 0)
                && t.line && t.line.color) {
                bmIdx.push(i);
                bmColors.push(s.bm_color);
            }
        });
        try {
            if (li.length) Plotly.restyle(g, {'line.width': lw}, li);
            if (mi.length) Plotly.restyle(g, {'marker.size': ms}, mi);
            if (oi.length) Plotly.restyle(g, {'marker.opacity': ov}, oi);
            if (bmIdx.length) Plotly.restyle(g, {'line.color': bmColors}, bmIdx);
        } catch(e) { return; }

        /* ── Layout: grids, colors, fonts, axes ──────────────────────── */
        /* Use the ORIGINAL user-supplied layout (g.layout) not _fullLayout
           to check minor grid state — _fullLayout contains Plotly defaults
           that make it look enabled when it isn't. */
        var fullLay = g._fullLayout || g.layout || {};
        var userLay = g.layout || {}, u = {};
        Object.keys(fullLay).forEach(function(k) {
            if (!/^[xy]axis\d*$/.test(k)) return;
            var a = fullLay[k] || {};
            var userAx = userLay[k] || {};
            u[k+'.gridwidth'] = gridMajor;
            u[k+'.gridcolor'] = s.grid_major_color;
            if (IS_DESKTOP) {
                u[k+'.linewidth'] = (a.linewidth||1) * DESKTOP.axis;
                if (a.tickfont && a.tickfont.size)
                    u[k+'.tickfont.size'] = Math.round(a.tickfont.size * DESKTOP.font);
                if (a.title && a.title.font && a.title.font.size)
                    u[k+'.title.font.size'] = Math.round(a.title.font.size * DESKTOP.font);
            }
            /* Minor grid — only style if the USER explicitly enabled it.
               userAx.minor is only set by figure builders when minor_grid
               checkbox is checked. */
            if (userAx.minor && userAx.minor.showgrid) {
                u[k+'.minor.gridwidth'] = gridMinor;
                u[k+'.minor.gridcolor'] = s.grid_minor_color;
            }
        });

        if (IS_DESKTOP) {
            if (lay.title && lay.title.font && lay.title.font.size)
                u['title.font.size'] = Math.round(lay.title.font.size * DESKTOP.font);
            if (lay.legend && lay.legend.font && lay.legend.font.size)
                u['legend.font.size'] = Math.round(lay.legend.font.size * DESKTOP.font);
            if (lay.annotations) lay.annotations.forEach(function(a,i) {
                if (a.font && a.font.size)
                    u['annotations['+i+'].font.size'] = Math.round(a.font.size * DESKTOP.font);
            });
        }

        try { if (Object.keys(u).length) Plotly.relayout(g, u); } catch(e) {}

        _lastApplied[id] = {fp: settingsFingerprint(s), traceWidth: traceWidth};
    }

    /* Hide static preview images once Plotly has rendered each chart */
    function hidePreviews() {
        IDS.forEach(function(gid) {
            var g = gd(gid);
            if (!g || !g.data || g.data.length === 0) return;
            var name = gid.replace('-graph', '');
            var img = document.getElementById(name + '-preview-img');
            if (img) img.style.display = 'none';
        });
    }

    setInterval(function() {
        var s = getUserSettings();
        var fp = settingsFingerprint(s);
        IDS.forEach(function(id) {
            var g = gd(id);
            if (needsApply(g, id, fp)) applySettings(g, id, s);
        });
        hidePreviews();
    }, 500);
})();
