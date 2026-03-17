/* ── Mobile bottom sheet — vanilla JS (no Dash callback dependency) ─────────
   Handles: handle tap/drag, body swipe-down-to-close, scrim dismiss,
   tab-switch close. Only active on viewports ≤ 767px.
*/
(function() {
    "use strict";
    var HANDLE_H = 90;   /* visible handle height when collapsed (matches CSS) */
    var DIR_THRESHOLD = 8; /* px of movement before committing to direction */

    function isMobile() { return window.innerWidth <= 767; }

    function getScrim() { return document.getElementById('sheet-scrim'); }

    function collapseAll() {
        document.querySelectorAll('.controls-col.sheet-expanded').forEach(function(el) {
            el.classList.remove('sheet-expanded');
            el.style.transform = '';
            el.style.overflowY = '';
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
        if (col.classList.contains('sheet-expanded')) {
            collapseAll();
        } else {
            expandCol(col);
        }
    });

    // ── Touch drag state ────────────────────────────────────────────────────
    var _startY = 0, _startTx = 0;
    var _dragging = false;    /* committed to dragging */
    var _pending = false;     /* touch started, waiting for direction */
    var _dragCol = null;
    var _fromHandle = false;

    document.addEventListener('touchstart', function(e) {
        if (!isMobile()) return;
        _dragging = false;
        _pending = false;
        _dragCol = null;
        _fromHandle = false;

        var handle = e.target.closest('.sheet-handle');
        if (handle) {
            _dragCol = handle.closest('.controls-col');
            if (!_dragCol) return;
            _fromHandle = true;
            _dragging = true;  /* handle drag commits immediately (touch-action:none) */
            _startY = e.touches[0].clientY;
            var expanded = _dragCol.classList.contains('sheet-expanded');
            _startTx = expanded ? 0 : _dragCol.offsetHeight - HANDLE_H;
            _dragCol.style.transition = 'none';
            return;
        }

        /* Body touch — only track if expanded and near top of scroll */
        var col = e.target.closest('.controls-col');
        if (col && col.classList.contains('sheet-expanded') && col.scrollTop <= 5) {
            _dragCol = col;
            _pending = true;  /* don't commit yet — wait for direction */
            _startY = e.touches[0].clientY;
        }
    }, {passive: true});

    document.addEventListener('touchmove', function(e) {
        if (!isMobile() || !_dragCol) return;

        /* Direction detection: first DIR_THRESHOLD px decide scroll vs drag */
        if (_pending) {
            var dy0 = e.touches[0].clientY - _startY;
            if (Math.abs(dy0) < DIR_THRESHOLD) return; /* not enough movement */
            if (dy0 > 0 && _dragCol.scrollTop <= 5) {
                /* Swiping DOWN from top → commit to drag-to-close */
                _pending = false;
                _dragging = true;
                _startTx = 0;
                _dragCol.style.overflowY = 'hidden';
                _dragCol.style.transition = 'none';
            } else {
                /* Swiping UP or not at top → let browser scroll */
                _pending = false;
                _dragCol = null;
                return;
            }
        }

        if (!_dragging) return;

        var dy = e.touches[0].clientY - _startY;
        /* When expanded (_startTx===0), only allow downward drag */
        if (_startTx === 0 && dy < 0) return;
        try { e.preventDefault(); } catch(_) {}
        var newY = Math.max(0, _startTx + dy);
        var maxY = _dragCol.offsetHeight - HANDLE_H;
        newY = Math.min(newY, maxY);
        _dragCol.style.transform = 'translateY(' + newY + 'px)';
    }, {passive: false});

    document.addEventListener('touchend', function() {
        if (!isMobile()) return;
        _pending = false;
        if (!_dragging || !_dragCol) {
            _dragCol = null;
            return;
        }
        _dragging = false;
        _dragCol.style.transition = '';
        _dragCol.style.overflowY = '';
        var rect = _dragCol.getBoundingClientRect();
        var visibleH = window.innerHeight - rect.top;
        if (visibleH > _dragCol.offsetHeight * 0.3) {
            _dragCol.classList.add('sheet-expanded');
            var scrim = getScrim();
            if (scrim) scrim.classList.add('active');
        } else {
            _dragCol.classList.remove('sheet-expanded');
            _dragCol.style.transform = '';
            var scrim = getScrim();
            if (scrim) scrim.classList.remove('active');
        }
        _dragCol = null;
    }, {passive: true});

    // ── Close sheet on tab switch ───────────────────────────────────────────
    var observer = new MutationObserver(function(mutations) {
        if (!isMobile()) return;
        for (var i = 0; i < mutations.length; i++) {
            if (mutations[i].target.classList &&
                mutations[i].target.classList.contains('tab-pane')) {
                collapseAll();
                return;
            }
        }
    });
    function startObserving() {
        var tc = document.querySelector('.tab-content');
        if (tc) observer.observe(tc, {childList: true, subtree: true, attributes: true});
        else setTimeout(startObserving, 500);
    }
    startObserving();
})();
