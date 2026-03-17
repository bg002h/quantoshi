/* ── Mobile bottom sheet ────────────────────────────────────────────────────
   Open:  tap handle OR swipe up on handle
   Close: tap ✕ button, tap scrim, OR swipe down on handle
   The handle has touch-action:none in CSS — browser cannot intercept.
   Content scrolls freely with no gesture conflicts.
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
        var s = getScrim();
        if (s) s.classList.remove('active');
    }

    function expandCol(col) {
        collapseAll();
        if (col) col.classList.add('sheet-expanded');
        var s = getScrim();
        if (s) s.classList.add('active');
    }

    // ── Close button ────────────────────────────────────────────────────────
    document.addEventListener('click', function(e) {
        if (!isMobile()) return;
        if (e.target.closest('.sheet-close-btn')) { collapseAll(); return; }
        if (e.target.id === 'sheet-scrim') { collapseAll(); return; }
    });

    // ── Handle tap to toggle ────────────────────────────────────────────────
    document.addEventListener('click', function(e) {
        if (!isMobile()) return;
        var handle = e.target.closest('.sheet-handle');
        if (!handle || e.target.closest('.sheet-close-btn')) return;
        var col = handle.closest('.controls-col');
        if (!col) return;
        col.classList.contains('sheet-expanded') ? collapseAll() : expandCol(col);
    });

    // ── Handle drag (touch-action:none guarantees no browser interference) ──
    var _startY = 0, _startTx = 0, _dragging = false, _dragCol = null;

    document.addEventListener('touchstart', function(e) {
        if (!isMobile()) return;
        var handle = e.target.closest('.sheet-handle');
        if (!handle || e.target.closest('.sheet-close-btn')) return;
        _dragCol = handle.closest('.controls-col');
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
        if (_startTx === 0 && dy < 0) return; /* expanded: only allow down */
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
        if (visibleH > _dragCol.offsetHeight * 0.3) {
            _dragCol.classList.add('sheet-expanded');
            var s = getScrim(); if (s) s.classList.add('active');
        } else {
            _dragCol.classList.remove('sheet-expanded');
            _dragCol.style.transform = '';
            var s = getScrim(); if (s) s.classList.remove('active');
        }
        _dragCol = null;
    }, {passive: true});

    // ── Close on tab switch ─────────────────────────────────────────────────
    var obs = new MutationObserver(function(muts) {
        if (!isMobile()) return;
        for (var i = 0; i < muts.length; i++) {
            if (muts[i].target.classList && muts[i].target.classList.contains('tab-pane')) {
                collapseAll(); return;
            }
        }
    });
    (function watch() {
        var tc = document.querySelector('.tab-content');
        if (tc) obs.observe(tc, {childList: true, subtree: true, attributes: true});
        else setTimeout(watch, 500);
    })();
})();
