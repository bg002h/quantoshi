/* ── Mobile bottom sheet — vanilla JS ───────────────────────────────────────
   Drag targets:
   - .sheet-handle (touch-action:none) — swipe up to open, down to close
   - .sheet-pullzone (touch-action:none) — swipe down to close (expanded only)
   Both have touch-action:none so the browser cannot intercept for scrolling.
*/
(function() {
    "use strict";
    var HANDLE_H = 90;

    function isMobile() { return window.innerWidth <= 767; }
    function getScrim() { return document.getElementById('sheet-scrim'); }

    function collapseAll() {
        document.querySelectorAll('.controls-col.sheet-expanded').forEach(function(el) {
            el.classList.remove('sheet-expanded');
            el.style.transform = '';
        });
        var scrim = getScrim();
        if (scrim) scrim.classList.remove('active');
    }

    function expandCol(col) {
        collapseAll();
        if (col) col.classList.add('sheet-expanded');
        var scrim = getScrim();
        if (scrim) scrim.classList.add('active');
    }

    // ── Scrim dismiss ───────────────────────────────────────────────────────
    document.addEventListener('click', function(e) {
        if (!isMobile()) return;
        if (e.target.id === 'sheet-scrim') collapseAll();
    });

    // ── Handle tap to toggle ────────────────────────────────────────────────
    document.addEventListener('click', function(e) {
        if (!isMobile()) return;
        var handle = e.target.closest('.sheet-handle');
        if (!handle) return;
        var col = handle.closest('.controls-col');
        if (!col) return;
        col.classList.contains('sheet-expanded') ? collapseAll() : expandCol(col);
    });

    // ── Touch drag ──────────────────────────────────────────────────────────
    var _startY = 0, _startTx = 0, _dragging = false, _dragCol = null;

    document.addEventListener('touchstart', function(e) {
        if (!isMobile()) return;
        _dragging = false;
        _dragCol = null;

        /* Match either .sheet-handle or .sheet-pullzone — both have touch-action:none */
        var target = e.target.closest('.sheet-handle') || e.target.closest('.sheet-pullzone');
        if (!target) return;

        _dragCol = target.closest('.controls-col');
        if (!_dragCol) return;
        _dragging = true;
        _startY = e.touches[0].clientY;
        var expanded = _dragCol.classList.contains('sheet-expanded');
        _startTx = expanded ? 0 : _dragCol.offsetHeight - HANDLE_H;
        _dragCol.style.transition = 'none';
    }, {passive: true});

    document.addEventListener('touchmove', function(e) {
        if (!_dragging || !_dragCol) return;
        var dy = e.touches[0].clientY - _startY;
        /* When expanded, only allow downward drag */
        if (_startTx === 0 && dy < 0) return;
        try { e.preventDefault(); } catch(_) {}
        var newY = Math.max(0, _startTx + dy);
        newY = Math.min(newY, _dragCol.offsetHeight - HANDLE_H);
        _dragCol.style.transform = 'translateY(' + newY + 'px)';
    }, {passive: false});

    document.addEventListener('touchend', function() {
        if (!_dragging || !_dragCol) return;
        _dragging = false;
        _dragCol.style.transition = '';
        var rect = _dragCol.getBoundingClientRect();
        var visibleH = window.innerHeight - rect.top;
        var scrim = getScrim();
        if (visibleH > _dragCol.offsetHeight * 0.3) {
            _dragCol.classList.add('sheet-expanded');
            if (scrim) scrim.classList.add('active');
        } else {
            _dragCol.classList.remove('sheet-expanded');
            _dragCol.style.transform = '';
            if (scrim) scrim.classList.remove('active');
        }
        _dragCol = null;
    }, {passive: true});

    // ── Close on tab switch ─────────────────────────────────────────────────
    var obs = new MutationObserver(function(muts) {
        if (!isMobile()) return;
        for (var i = 0; i < muts.length; i++) {
            if (muts[i].target.classList && muts[i].target.classList.contains('tab-pane')) {
                collapseAll();
                return;
            }
        }
    });
    (function watch() {
        var tc = document.querySelector('.tab-content');
        if (tc) obs.observe(tc, {childList: true, subtree: true, attributes: true});
        else setTimeout(watch, 500);
    })();
})();
