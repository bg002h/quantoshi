/* Plot Appearance panel controller — pure JS/DOM control plane.
 *
 * Owns every read and write of localStorage["plot-appearance"] and every
 * DOM update to the 30 Plot Appearance inputs (5 tabs × 6 fields) and the
 * 5 reset buttons. No Dash callbacks involved. chart_responsive.js reads
 * localStorage independently every 500ms.
 *
 * Design doc: docs/superpowers/specs/2026-04-10-plot-appearance-control-plane-design.md
 *
 * Invariants:
 *  1. localStorage["plot-appearance"] is the single source of truth.
 *  2. Every state change writes localStorage + applies to all 30 inputs.
 *  3. No Dash callback ever writes to {prefix}-plot-* values; React has no
 *     reason to re-render them, so JS-set DOM values are stable.
 *  4. Idempotent across hot-reload via window.__paCleanup.
 *  5. cloneNode(true) + replaceChild is used to drop stale listeners
 *     atomically — do not track listener references manually.
 */
(function() {
    'use strict';

    /* ── Idempotent init: clear any prior interval / wired-set ─────────── */
    if (window.__paCleanup) {
        try { window.__paCleanup(); } catch(e) {}
    }

    var DEFAULTS = {
        trace_width: 2.5,
        grid_major_width: 1.0,
        grid_major_color: "#888888",
        grid_minor_width: 0.8,
        grid_minor_color: "#b0b0b0",
        pt_color: "#2c3e50"
    };

    var PREFIXES = ['bub', 'dca', 'ret', 'sc', 'cp'];

    /* [id-kebab-suffix, state-key, type] */
    var FIELDS = [
        ['trace-width',      'trace_width',      'number'],
        ['grid-major-width', 'grid_major_width', 'number'],
        ['grid-major-color', 'grid_major_color', 'color'],
        ['grid-minor-width', 'grid_minor_width', 'number'],
        ['grid-minor-color', 'grid_minor_color', 'color'],
        ['pt-color',         'pt_color',         'color']
    ];

    var _wired = new Set();  // element identities already wired
    var _interval = null;
    var _lastFp = null;

    function fingerprint(s) {
        return [s.trace_width, s.grid_major_width, s.grid_major_color,
                s.grid_minor_width, s.grid_minor_color, s.pt_color].join('|');
    }

    function ctrlId(prefix, kebab) { return prefix + '-plot-' + kebab; }
    function btnId(prefix)         { return prefix + '-plot-appearance-reset'; }

    function readState() {
        try {
            var raw = localStorage.getItem("plot-appearance");
            if (raw) {
                var s = JSON.parse(raw);
                if (s && typeof s === "object") {
                    for (var k in DEFAULTS) {
                        if (s[k] == null) s[k] = DEFAULTS[k];
                    }
                    return s;
                }
            }
        } catch(e) {}
        return Object.assign({}, DEFAULTS);
    }

    function writeState(s) {
        try {
            localStorage.setItem("plot-appearance", JSON.stringify(s));
        } catch(e) {}
        try {
            window.dispatchEvent(new CustomEvent("plot-appearance-changed", {detail: s}));
        } catch(e) {}
    }

    function applyStateToDOM(s) {
        var fp = fingerprint(s);
        if (fp === _lastFp) return;
        _lastFp = fp;
        PREFIXES.forEach(function(prefix) {
            FIELDS.forEach(function(f) {
                var el = document.getElementById(ctrlId(prefix, f[0]));
                if (!el) return;
                var v = s[f[1]];
                if (v == null) return;
                if (f[2] === 'number') {
                    if (String(el.value) !== String(v)) {
                        el.value = String(v);
                    }
                } else {
                    /* color: compare lowercased hex */
                    var hex = String(v).toLowerCase();
                    if (String(el.value).toLowerCase() !== hex) {
                        el.value = hex;
                    }
                }
            });
        });
    }

    function makeInputHandler(field) {
        return function(ev) {
            var s = readState();
            var raw = ev.target.value;
            if (field[2] === 'number') {
                var n = parseFloat(raw);
                if (isNaN(n)) n = DEFAULTS[field[1]];
                s[field[1]] = n;
            } else {
                s[field[1]] = String(raw).toLowerCase();
            }
            writeState(s);
            applyStateToDOM(s);
        };
    }

    function makeResetHandler() {
        return function() {
            var s = Object.assign({}, DEFAULTS);
            writeState(s);
            applyStateToDOM(s);
            /* DO NOT call preventDefault or stopPropagation — the click
               must still reach Dash so the kept pt_size/pt_alpha reset
               clientside callback fires and updates bub-ptsize/bub-ptalpha. */
        };
    }

    function wireElement(el, listener, eventName) {
        if (!el.parentNode) return el;  // detached; skip
        /* Drop any stale listeners atomically via cloneNode, then bind. */
        var clone = el.cloneNode(true);
        el.parentNode.replaceChild(clone, el);
        clone.addEventListener(eventName, listener);
        _wired.add(clone);
        return clone;
    }

    function rewireNewControls() {
        var didWire = false;
        PREFIXES.forEach(function(prefix) {
            FIELDS.forEach(function(f) {
                var el = document.getElementById(ctrlId(prefix, f[0]));
                if (!el || _wired.has(el)) return;
                /* For number inputs, listen on 'input' (live). For color
                   inputs, listen on 'change' to avoid writing on every
                   picker pixel-drag (Chrome fires 'input' continuously). */
                var eventName = (f[2] === 'number') ? 'input' : 'change';
                wireElement(el, makeInputHandler(f), eventName);
                didWire = true;
            });
            var btn = document.getElementById(btnId(prefix));
            if (btn && !_wired.has(btn)) {
                wireElement(btn, makeResetHandler(), 'click');
                didWire = true;
            }
        });
        if (didWire) _lastFp = null;
    }

    function tick() {
        rewireNewControls();
        applyStateToDOM(readState());
    }

    /* ── Synchronous bootstrap — paint state before first interval tick to
         avoid the up-to-500ms cold-start flash on color inputs. ────────── */
    tick();

    /* ── Main loop — belt-and-suspenders; picks up lazy-loaded Citadel
         controls and any React re-renders. ──────────────────────────────── */
    _interval = setInterval(tick, 500);

    window.__paCleanup = function() {
        if (_interval) {
            try { clearInterval(_interval); } catch(e) {}
            _interval = null;
        }
        _wired = new Set();
        _lastFp = null;
    };
})();
