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
        return document.querySelector('.controls-col');
    }

    function getScrim() {
        return document.getElementById('sheet-scrim');
    }

    function expand() {
        var sheet = getSheet(), scrim = getScrim();
        if (sheet) sheet.classList.add('sheet-expanded');
        if (scrim) scrim.classList.add('active');
    }

    function collapse() {
        var sheet = getSheet(), scrim = getScrim();
        if (sheet) { sheet.classList.remove('sheet-expanded'); sheet.style.transform = ''; }
        if (scrim) scrim.classList.remove('active');
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
        if (handle) toggle();
    });

    // ── Handle drag ─────────────────────────────────────────────────────────
    var _startY = 0, _startTx = 0, _dragging = false;

    document.addEventListener('touchstart', function(e) {
        if (!isMobile()) return;
        var handle = e.target.closest('.sheet-handle');
        if (!handle) return;
        _dragging = true;
        _startY = e.touches[0].clientY;
        var sheet = getSheet();
        if (!sheet) return;
        var expanded = sheet.classList.contains('sheet-expanded');
        _startTx = expanded ? 0 : sheet.offsetHeight - 90;
        sheet.style.transition = 'none';
    }, {passive: true});

    document.addEventListener('touchmove', function(e) {
        if (!_dragging || !isMobile()) return;
        var sheet = getSheet();
        if (!sheet) return;
        var dy = e.touches[0].clientY - _startY;
        var newY = Math.max(0, _startTx + dy);
        var maxY = sheet.offsetHeight - 90;
        newY = Math.min(newY, maxY);
        sheet.style.transform = 'translateY(' + newY + 'px)';
    }, {passive: true});

    document.addEventListener('touchend', function() {
        if (!_dragging || !isMobile()) return;
        _dragging = false;
        var sheet = getSheet();
        if (!sheet) return;
        sheet.style.transition = '';
        var rect = sheet.getBoundingClientRect();
        var visibleH = window.innerHeight - rect.top;
        if (visibleH > sheet.offsetHeight * 0.3) {
            expand();
        } else {
            collapse();
        }
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
