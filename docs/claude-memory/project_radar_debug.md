---
name: Radar beacon not showing on bubble chart
description: scanner.js radar marker overlay doesn't display — needs debugging with browser dev tools
type: project
---

The animated radar beacon (scanner.js) doesn't appear on the bubble chart despite the CSS animations and JS being deployed.

**Symptoms:** No dot/animation visible even after entering a price in the scanner. No JS errors in console.

**Likely causes to investigate:**
1. `document.getElementById("bubble-graph")` may not find the element (Dash wraps graph in loading container)
2. The Plotly `_fullLayout.xaxis.l2p()` coordinate conversion may return values outside the visible area
3. The `plotly_afterplot` event may not fire on the bubble tab (check if it fires on tab switch)
4. The `.radar-marker` div may be created but hidden behind other elements (z-index issue)
5. Safari `conic-gradient` support may be incomplete (test in Chrome first)

**Debug steps for next session:**
1. Add `console.log` statements to scanner.js at key points (init, updateMarkers, coordinate calc)
2. Test in Chrome (more reliable Plotly/CSS support)
3. Check if the marker div exists in DOM inspector after entering a price
4. Verify xPx/yPx values are within the chart bounds

**Also fix:** FAQ `<table>` inside `<p>` DOM nesting warning (not related but should clean up).

**How to apply:** Debug scanner.js in browser, add console.log, test in Chrome.
