/**
 * Double-click on a tab header forces a figure reload.
 *
 * Increments the matching {tab}-first-render store, which triggers
 * the chart callback to re-run (cache hit = instant refresh).
 */
(function() {
    "use strict";
    var _lastClick = {tab: null, ts: 0};
    var _DBLCLICK_MS = 400;

    document.addEventListener("click", function(e) {
        // Find the closest tab link (.nav-link inside #main-tabs)
        var link = e.target.closest("#main-tabs .nav-link");
        if (!link) return;

        // Extract tab_id from the aria-controls or data attribute
        var tabId = link.getAttribute("aria-controls") ||
                    link.getAttribute("data-bs-target");
        if (!tabId) return;
        // dbc.Tabs uses tab_id as the panel id prefix
        tabId = tabId.replace(/^#?tab-/, "").replace(/-panel$/, "");

        var now = Date.now();
        if (_lastClick.tab === tabId && (now - _lastClick.ts) < _DBLCLICK_MS) {
            // Double-click detected — increment the first-render store
            var storeId = tabId + "-first-render";
            var storeEl = document.getElementById(storeId);
            if (storeEl) {
                // Read current value from React props and increment
                var rk = Object.keys(storeEl).find(function(k) {
                    return k.startsWith("__reactFiber") || k.startsWith("__reactInternalInstance");
                });
                if (rk) {
                    var fiber = storeEl[rk];
                    var node = fiber;
                    for (var i = 0; i < 15 && node; i++) {
                        if (node.memoizedProps && typeof node.memoizedProps.setProps === "function") {
                            var cur = node.memoizedProps.data || 0;
                            node.memoizedProps.setProps({data: cur + 1});
                            break;
                        }
                        node = node.return;
                    }
                }
            }
            _lastClick = {tab: null, ts: 0};
        } else {
            _lastClick = {tab: tabId, ts: now};
        }
    }, true);
})();
