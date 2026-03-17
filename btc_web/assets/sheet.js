/* ── Mobile bottom sheet — vanilla JS (no Dash callback dependency) ─────────
   Handles: FAB toggle, handle tap, handle drag, scrim dismiss, tab-switch close.
   Only active on viewports ≤ 767px (matches CSS media query).
*/
(function() {
    "use strict";

    function isMobile() {
        return window.innerWidth <= 767;
    }

    function getSheet() {
        /* Find the controls-col inside the currently active tab pane */
        var active = document.querySelector('.tab-pane.active .controls-col');
        return active || document.querySelector('.controls-col');
    }

    function getScrim() {
        return document.getElementById('sheet-scrim');
    }

    function collapseAll() {
        /* Remove sheet-expanded from ALL controls-cols (prevents stale state across tabs) */
        document.querySelectorAll('.controls-col.sheet-expanded').forEach(function(el) {
            el.classList.remove('sheet-expanded');
            el.style.transform = '';
        });
        var scrim = getScrim();
        if (scrim) scrim.classList.remove('active');
    }

    function expand() {
        collapseAll();
        var sheet = getSheet(), scrim = getScrim();
        if (sheet) sheet.classList.add('sheet-expanded');
        if (scrim) scrim.classList.add('active');
    }

    function collapse() {
        collapseAll();
    }

    function toggle() {
        var sheet = getSheet();
        if (sheet && sheet.classList.contains('sheet-expanded')) {
            collapse();
        } else {
            expand();
        }
    }

    // ── FAB button ──────────────────────────────────────────────────────────
    document.addEventListener('click', function(e) {
        if (!isMobile()) return;
        var fab = e.target.closest('#mobile-settings-fab');
        if (fab) { e.preventDefault(); toggle(); }
    });

    // ── Scrim dismiss ───────────────────────────────────────────────────────
    document.addEventListener('click', function(e) {
        if (!isMobile()) return;
        if (e.target.id === 'sheet-scrim') collapse();
    });

    // ── Handle tap ──────────────────────────────────────────────────────────
    document.addEventListener('click', function(e) {
        if (!isMobile()) return;
        var handle = e.target.closest('.sheet-handle');
        if (!handle) return;
        var col = handle.closest('.controls-col');
        if (!col) return;
        var scrim = getScrim();
        if (col.classList.contains('sheet-expanded')) {
            collapseAll();
        } else {
            collapseAll();
            col.classList.add('sheet-expanded');
            if (scrim) scrim.classList.add('active');
        }
    });

    // ── Drag (handle to open, anywhere in sheet to close when scrolled to top) ─
    var _startY = 0, _startTx = 0, _dragging = false, _dragCol = null;

    document.addEventListener('touchstart', function(e) {
        if (!isMobile()) return;
        // Start drag from handle (open or close)
        var handle = e.target.closest('.sheet-handle');
        if (handle) {
            _dragCol = handle.closest('.controls-col');
            if (!_dragCol) return;
            _dragging = true;
            _startY = e.touches[0].clientY;
            var expanded = _dragCol.classList.contains('sheet-expanded');
            _startTx = expanded ? 0 : _dragCol.offsetHeight - 90;
            _dragCol.style.transition = 'none';
            return;
        }
        // Start drag from expanded sheet body (swipe down to close)
        // Only when scrolled to top so it doesn't fight content scrolling
        var col = e.target.closest('.controls-col');
        if (col && col.classList.contains('sheet-expanded') && col.scrollTop <= 0) {
            _dragCol = col;
            _dragging = true;
            _startY = e.touches[0].clientY;
            _startTx = 0;
            _dragCol.style.transition = 'none';
        }
    }, {passive: true});

    document.addEventListener('touchmove', function(e) {
        if (!_dragging || !isMobile() || !_dragCol) return;
        var dy = e.touches[0].clientY - _startY;
        // If dragging from expanded body, only allow downward movement
        if (_startTx === 0 && dy < 0) return;
        var newY = Math.max(0, _startTx + dy);
        var maxY = _dragCol.offsetHeight - 90;
        newY = Math.min(newY, maxY);
        _dragCol.style.transform = 'translateY(' + newY + 'px)';
    }, {passive: true});

    document.addEventListener('touchend', function() {
        if (!_dragging || !isMobile() || !_dragCol) return;
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

    // ── Close sheet on tab switch ───────────────────────────────────────────
    var observer = new MutationObserver(function(mutations) {
        if (!isMobile()) return;
        for (var i = 0; i < mutations.length; i++) {
            if (mutations[i].target.classList &&
                mutations[i].target.classList.contains('tab-pane')) {
                collapse();
                return;
            }
        }
    });

    // Observe tab content changes
    function startObserving() {
        var tabContent = document.querySelector('.tab-content');
        if (tabContent) {
            observer.observe(tabContent, {childList: true, subtree: true, attributes: true});
        } else {
            setTimeout(startObserving, 500);
        }
    }
    startObserving();
})();
