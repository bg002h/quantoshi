/* Chart appearance — applies user-customized trace/grid settings.
 *
 * Reads from the "plot-appearance" localStorage entry (written by
 * Dash dcc.Store(storage_type="local")). Applies via Plotly.restyle
 * (traces) and Plotly.relayout (grids/axes).
 *
 * Critical invariants (learned from a long, painful growth-bug saga):
 *
 *  1. NEVER read current Plotly state and multiply it. Always compute
 *     ABSOLUTE target values from user settings. Any "read * constant"
 *     pattern compounds every poll → unbounded growth.
 *
 *  2. NEVER touch marker.size. Server-side figures set it from pt_size;
 *     the JS layer has no business modifying it. Any size manipulation
 *     here will eventually leak into a growth loop.
 *
 *  3. Be idempotent across hot-reloads. Hot-reload re-evaluates the IIFE
 *     without clearing the previous setInterval. We stash the handle on
 *     window.__chartResponsiveInterval and clear it on each re-entry so
 *     we never have two loops fighting (one of which may be the old
 *     buggy version still running from a stale module load).
 *
 *  4. Hide the static preview <img> as soon as Plotly has data. Use BOTH
 *     the poll (belt) and Plotly's plotly_afterplot event (suspenders).
 */
(function() {
    'use strict';

    /* ── 1. Idempotent init: clear any previous interval/handlers ─────── */
    if (window.__chartResponsiveInterval) {
        try { clearInterval(window.__chartResponsiveInterval); } catch(e) {}
        window.__chartResponsiveInterval = null;
    }

    var IS_DESKTOP = window.innerWidth > 768;

    var DEFAULTS = {
        trace_width: 2.5,
        grid_major_width: 1.0,
        grid_major_color: "#888888",
        grid_minor_width: 0.8,
        grid_minor_color: "#B0B0B0",
        pt_color: "#2C3E50",
    };

    /* Desktop multipliers applied ONLY to user-supplied absolute settings,
       NEVER to current Plotly state. */
    var DESKTOP = {
        trace_mult: 1.5,
        grid_mult: 1.5,
    };

    var IDS = ['bubble-graph','heatmap-graph','dca-graph',
               'retire-graph','supercharge-graph','citadel-graph'];

    /* Per-chart cache: last applied fingerprint + target trace width.
       Fresh on every module load (that's fine — first poll re-applies). */
    var _applied = {};
    /* Per-chart flag: has plotly_afterplot been bound? */
    var _bound = {};
    /* Per-chart flag: has preview been hidden? (avoid DOM writes). */
    var _previewHidden = {};

    function gd(id) {
        var w = document.getElementById(id);
        if (!w) return null;
        return w.querySelector('.js-plotly-plot') || w;
    }

    function getSettings() {
        try {
            var raw = localStorage.getItem("plot-appearance");
            if (raw) {
                var parsed = JSON.parse(raw);
                if (parsed && typeof parsed === "object") return parsed;
            }
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
        /* If Dash replaced the figure, line widths drop back to server-side
           values and won't match our target. Re-apply once. */
        for (var i = 0; i < g.data.length; i++) {
            if (g.data[i].line && g.data[i].line.width != null) {
                if (Math.abs(g.data[i].line.width - last.targetTraceWidth) > 0.1) return true;
                return false;
            }
        }
        return false;
    }

    function applySettings(g, id, s) {
        /* Absolute target values — computed from user settings only. */
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
            /* Price data scatter — recolor to user's pt_color.
               NOTE: we deliberately do NOT touch marker.size here.
               See invariant #2 at the top of the file. */
            if (t.mode === "markers" && t.name === "Price data") {
                ptIdx.push(i);
                ptColors.push(s.pt_color);
            }
        });
        try {
            if (li.length) Plotly.restyle(g, {'line.width': lw}, li);
            if (ptIdx.length) Plotly.restyle(g, {'marker.color': ptColors}, ptIdx);
        } catch(e) { return; }

        /* ── Layout relayout: grids + axis colors/widths ──────────────── */
        var layout = g.layout || {};
        var u = {};
        Object.keys(layout).forEach(function(k) {
            if (!/^[xy]axis\d*$/.test(k)) return;
            var userAx = layout[k] || {};
            u[k + '.gridwidth'] = targetGridMajor;
            u[k + '.gridcolor'] = s.grid_major_color;
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

    /* ── Preview-image hide ──────────────────────────────────────────────
       Called from the poll loop AND from plotly_afterplot. Idempotent. */
    function hidePreviewFor(gid) {
        if (_previewHidden[gid]) return;
        var g = gd(gid);
        if (!g || !g.data || g.data.length === 0) return;
        var name = gid.replace('-graph', '');
        var img = document.getElementById(name + '-preview-img');
        if (img) {
            img.style.display = 'none';
            img.style.visibility = 'hidden';
            img.style.opacity = '0';
            img.style.pointerEvents = 'none';
            _previewHidden[gid] = true;
        }
    }

    function hidePreviews() {
        IDS.forEach(hidePreviewFor);
    }

    /* Bind a one-time plotly_afterplot handler that hides this chart's
       preview the moment Plotly finishes its first render. This is the
       reliable path; the poll is just a fallback. */
    function bindAfterplot(gid) {
        if (_bound[gid]) return;
        var g = gd(gid);
        if (!g || typeof g.on !== 'function') return;
        _bound[gid] = true;
        g.on('plotly_afterplot', function() {
            hidePreviewFor(gid);
        });
        /* Fire once in case the first render already happened. */
        hidePreviewFor(gid);
    }

    /* ── Poll loop — belt, with plotly_afterplot as suspenders ────────── */
    window.__chartResponsiveInterval = setInterval(function() {
        var s = getSettings();
        var fp = fingerprint(s);
        IDS.forEach(function(id) {
            var g = gd(id);
            bindAfterplot(id);
            if (needsApply(g, id, fp)) applySettings(g, id, s);
        });
        hidePreviews();
    }, 500);
})();
