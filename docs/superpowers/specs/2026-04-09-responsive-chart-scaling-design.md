# Responsive Chart Scaling — Client-Side Restyle

**Date:** 2026-04-09
**Status:** Approved

## Problem

Plotly chart traces (line widths, marker sizes, grid lines) are set server-side as fixed pixel values. On mobile with warm screens, they look fine. On desktop with cool white monitors, lines appear thin and low-contrast — especially the BM gold lines on white backgrounds.

## Solution

A single clientside callback watches `viewport-width` and calls `Plotly.restyle()` on each chart's `dcc.Graph` after render, scaling up line widths, marker sizes, and marker opacity on desktop (viewport > 768px).

## Design

### Scale factors

| Property | Mobile (≤768px) | Desktop (>768px) | Scale |
|----------|----------------|-------------------|-------|
| `line.width` | 1.0× (as-is) | 1.5× | Multiply existing width by 1.5 |
| `marker.size` | 1.0× (as-is) | 1.3× | Multiply existing size by 1.3 |
| `marker.opacity` | 1.0× (as-is) | min(1.0, existing × 1.4) | Boost, cap at 1.0 |

### Implementation

One JS asset file (`btc_web/assets/chart_responsive.js`) that:

1. Listens for Plotly's `plotly_afterplot` event on each graph div
2. Checks `window.innerWidth > 768`
3. If desktop, iterates traces, reads their current `line.width` / `marker.size` / `marker.opacity`, multiplies by scale factor, and calls `Plotly.restyle()`
4. Stores a flag on the element to avoid re-applying on subsequent afterplot events

### Trace filtering

- Only restyle traces where `trace.line && trace.line.width != null` (skip fill-only traces)
- For markers: only where `trace.marker && trace.marker.size != null`
- Skip `heatmap` trace types entirely

### Charts affected

All 6: `bubble-graph`, `heatmap-graph` (annotation lines only), `dca-graph`, `retire-graph`, `supercharge-graph`, `citadel-graph`.

### No server changes

The figure JSON is untouched. No cache key changes. No callback registration. Pure client-side JS.

## File

| File | Action |
|------|--------|
| `btc_web/assets/chart_responsive.js` | Create |
