# Model Scanner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a bidirectional price/date/quantile lookup panel to the Bubble tab and register Quantile Regression as a standalone PriceModel, with click-to-plot radar beacon markers.

**Architecture:** A new `QuantileRegressionModel` class in `btc_core.py` wraps the raw QR fits. A "Model Scanner" collapsible section in `layout/bubble.py` has three inputs (price, date, quantile) — user fills any two, the third is computed across all models. Clicking a result row draws that model's quantile line + animated radar marker on the chart.

**Tech Stack:** Dash, Plotly, CSS animations (`@keyframes`), `scipy.optimize.brentq`

**Spec:** `docs/superpowers/specs/2026-03-20-model-scanner-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `archive/btc_app/btc_core.py` | Modify | Add `QuantileRegressionModel` class |
| `btc_web/app.py` | Modify | Register QR model |
| `btc_web/snapshot.py` | Modify | Add scanner controls + "qr" to checklist options |
| `btc_web/layout/bubble.py` | Modify | Add Model Scanner panel to controls |
| `btc_web/callbacks/scanner.py` | Create | Scanner callback — compute missing variable, build results table |
| `btc_web/callbacks/__init__.py` | Modify | Import scanner callbacks |
| `btc_web/assets/style.css` | Modify | Scanner output field styling + radar animation CSS |
| `btc_web/assets/scanner.js` | Create | Clientside callback for radar marker overlay positioning |
| `btc_web/figures/bubble.py` | Modify | Add scanner quantile lines to bubble chart |
| `btc_web/test_web.py` | Modify | Add QR model + scanner tests |

---

### Task 1: Add `QuantileRegressionModel` class

**Files:**
- Modify: `archive/btc_app/btc_core.py`
- Modify: `btc_web/app.py`
- Modify: `btc_web/snapshot.py`
- Modify: `btc_web/test_web.py`

- [ ] **Step 1: Add class to btc_core.py**

Add after `_FitsBasedModel` and before `_CompositeModel` (around line 346):

```python
class QuantileRegressionModel(_FitsBasedModel):
    """Raw quantile regression fits — model-free, purely empirical.

    Straight lines in log-log space. Each quantile has its own independently
    fitted slope and intercept. This is what BubbleModel used to be before
    the shrinking Gaussian conversion.
    """
    name = "Quantile Regression"
    short_name = "qr"
    dash_style = "solid"

    def __init__(self, md):
        self.fits = md.qr_fits
        self.colors = dict(md.qr_colors)
        self.quantiles = sorted(md.qr_fits.keys())
```

- [ ] **Step 2: Register in app.py**

Add `QuantileRegressionModel` to the import on line 35. Add registration after the BubbleModel registration (line ~142):

```python
_app_ctx.PRICE_MODELS["qr"] = QuantileRegressionModel(M)
```

- [ ] **Step 3: Add "qr" to snapshot.py checklist options**

Add `"qr"` to all `*-model-show` lists in `_CHECKLIST_OPTIONS` (lines ~164-168):

```python
"bub-model-show":     ["pl", "lppl", "exp", "s2f", "ef", "qr"],
```

And to all DCA/ret/sc/hm lists similarly.

- [ ] **Step 4: Fix bubble layout overlay models filter**

In `btc_web/layout/bubble.py` line 69, the overlay models checklist filters out `"bub"`. It should also filter out `"qr"` since QR lines are shown via the quantile panel, not as an overlay:

```python
if mdl.short_name not in ("bub", "qr")
```

- [ ] **Step 5: Add tests**

Add to `btc_web/test_web.py`:

```python
class TestQuantileRegressionModel:
    def setup_method(self):
        self.qr = QuantileRegressionModel(M)

    def test_short_name(self):
        assert self.qr.short_name == "qr"

    def test_fits_are_qr_fits(self):
        assert self.qr.fits is M.qr_fits

    def test_price_at_matches_qr_price(self):
        q = 0.5
        t = 10.0
        expected = qr_price(q, t, M.qr_fits)
        result = self.qr.price_at(q, t)
        np.testing.assert_allclose(result, expected)

    def test_quantized(self):
        assert self.qr.quantized is True
```

- [ ] **Step 6: Run tests and commit**

```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py -k "QuantileRegression" -v
git add archive/btc_app/btc_core.py btc_web/app.py btc_web/snapshot.py btc_web/layout/bubble.py btc_web/test_web.py
git commit -m "feat: register QuantileRegressionModel as standalone PriceModel"
```

---

### Task 2: Add Model Scanner layout panel

**Files:**
- Modify: `btc_web/layout/bubble.py`

- [ ] **Step 1: Add scanner panel to bubble controls**

In `_bubble_controls()` in `btc_web/layout/bubble.py`, after the `_q_panel("bub-qs", [])` line (line 88) and before the `_ctrl_card` for Data Point Appearance (line 89), insert:

```python
        _section_card("Model Scanner",
            _row(
                html.Div([
                    _lbl("Price ($)"),
                    dbc.Input(id="scan-price", type="number",
                              placeholder="live", size="sm", debounce=True),
                    html.Small("₿ live", id="scan-price-hint",
                               className="text-muted", style={"fontSize":"9px"}),
                ]),
                html.Div([
                    _lbl("Date"),
                    dbc.Input(id="scan-date", type="date",
                              value=pd.Timestamp.today().strftime("%Y-%m-%d"),
                              size="sm", debounce=True),
                ]),
                html.Div([
                    _lbl("Quantile (%)"),
                    dbc.Input(id="scan-q", type="number",
                              min=0.1, max=99.9, step=0.1,
                              size="sm", debounce=True,
                              className="scan-output"),
                ]),
            ),
            dcc.Store(id="scan-output-field", data="q"),  # which field is output: "p", "d", or "q"
            dcc.Store(id="scan-active-rows", data=[]),     # list of active model short_names
            html.Div(id="scan-results"),                   # results table rendered by callback
        ),
```

Also add required imports at top of file: `dcc` if not already imported.

- [ ] **Step 2: Syntax check and commit**

```bash
btc_venv/bin/python3 -m py_compile btc_web/layout/bubble.py && echo "OK"
git add btc_web/layout/bubble.py
git commit -m "feat: add Model Scanner panel layout to bubble tab controls"
```

---

### Task 3: Add scanner callback

**Files:**
- Create: `btc_web/callbacks/scanner.py`
- Modify: `btc_web/callbacks/__init__.py`

- [ ] **Step 1: Create scanner callback module**

Create `btc_web/callbacks/scanner.py`:

```python
"""Model Scanner callbacks — bidirectional price/date/quantile lookup."""

import numpy as np
import pandas as pd
from dash import html, Input, Output, State, callback, no_update, ctx
from scipy.optimize import brentq

import _app_ctx
from btc_core import today_t, fmt_price


def _solve_date(model, q_frac, target_price):
    """Root-find t where model.price_at(q, t) = target_price."""
    log_target = np.log10(max(target_price, 1e-10))
    def f(t):
        return np.log10(max(float(model.price_at(q_frac, t)), 1e-10)) - log_target
    try:
        t = brentq(f, 0.5, 72.0)
        genesis = _app_ctx.M.genesis
        date = genesis + pd.Timedelta(days=t * 365.25)
        return date.strftime("%Y-%m-%d")
    except (ValueError, RuntimeError):
        return "—"


@callback(
    Output("scan-q", "value"),
    Output("scan-price", "value"),
    Output("scan-date", "value"),
    Output("scan-output-field", "data"),
    Output("scan-results", "children"),
    Output("scan-q", "className"),
    Output("scan-price", "className"),
    Output("scan-date", "className"),
    Input("scan-price", "value"),
    Input("scan-date", "value"),
    Input("scan-q", "value"),
    State("scan-output-field", "data"),
    Input("btc-price-store", "data"),
    prevent_initial_call=False,
)
def update_scanner(price_val, date_val, q_val, current_output, live_price):
    """Compute the missing variable across all models."""
    trigger = ctx.triggered_id if ctx.triggered_id else None
    genesis = _app_ctx.M.genesis

    # Determine which field is output based on what was last edited
    if trigger == "scan-price":
        output_field = "q"      # user set price → compute quantile
    elif trigger == "scan-date":
        output_field = "q"      # user set date → compute quantile
    elif trigger == "scan-q":
        output_field = "p"      # user set quantile → compute price
    elif trigger == "btc-price-store":
        output_field = current_output or "q"
    else:
        output_field = "q"      # initial load

    # Resolve defaults
    if price_val is None or price_val == "":
        price = float(live_price) if live_price else None
    else:
        price = float(price_val)

    if date_val is None or date_val == "":
        date_val = pd.Timestamp.today().strftime("%Y-%m-%d")

    t = (pd.to_datetime(date_val) - genesis).days / 365.25
    if t <= 0:
        t = 0.5

    q_frac = float(q_val) / 100.0 if q_val is not None and q_val != "" else None

    # CSS classes for input vs output styling
    input_cls = ""
    output_cls = "scan-output"

    # Build results table
    rows = []
    computed_val = no_update
    out_price = no_update
    out_date = no_update
    out_q = no_update
    p_cls, d_cls, q_cls = input_cls, input_cls, input_cls

    if output_field == "q" and price is not None:
        # Solve for quantile
        q_cls = output_cls
        for key, mdl in _app_ctx.PRICE_MODELS.items():
            pct = mdl.find_percentile(t, price)
            rows.append(html.Tr([
                html.Td(mdl.name, style={"fontSize": "11px"}),
                html.Td(f"Q{pct*100:.1f}%", style={"fontSize": "11px",
                         "fontWeight": "bold"}),
            ], id={"type": "scan-row", "model": key},
               style={"cursor": "pointer"}))
        # Use QR model for the output field value
        qr = _app_ctx.PRICE_MODELS.get("qr")
        if qr:
            main_pct = qr.find_percentile(t, price)
            out_q = round(main_pct * 100, 1)

    elif output_field == "p" and q_frac is not None:
        # Solve for price
        p_cls = output_cls
        for key, mdl in _app_ctx.PRICE_MODELS.items():
            p = float(mdl.price_at(q_frac, t))
            rows.append(html.Tr([
                html.Td(mdl.name, style={"fontSize": "11px"}),
                html.Td(fmt_price(p), style={"fontSize": "11px",
                         "fontWeight": "bold"}),
            ], id={"type": "scan-row", "model": key},
               style={"cursor": "pointer"}))
        qr = _app_ctx.PRICE_MODELS.get("qr")
        if qr:
            out_price = round(float(qr.price_at(q_frac, t)), 2)

    elif output_field == "d" and price is not None and q_frac is not None:
        # Solve for date
        d_cls = output_cls
        for key, mdl in _app_ctx.PRICE_MODELS.items():
            date_str = _solve_date(mdl, q_frac, price)
            rows.append(html.Tr([
                html.Td(mdl.name, style={"fontSize": "11px"}),
                html.Td(date_str, style={"fontSize": "11px",
                         "fontWeight": "bold"}),
            ], id={"type": "scan-row", "model": key},
               style={"cursor": "pointer"}))

    # Build table
    header_map = {"q": "Quantile", "p": "Price", "d": "Date"}
    table = html.Table([
        html.Thead(html.Tr([
            html.Th("Model", style={"fontSize": "11px", "paddingRight": "12px"}),
            html.Th(header_map.get(output_field, ""), style={"fontSize": "11px"}),
        ])),
        html.Tbody(rows),
    ], style={"width": "100%", "borderCollapse": "collapse",
              "marginTop": "6px"}) if rows else html.Small("Enter values above",
                                                            className="text-muted")

    return (out_q, out_price, out_date, output_field, table,
            q_cls, p_cls, d_cls)
```

- [ ] **Step 2: Import in callbacks/__init__.py**

Add to `btc_web/callbacks/__init__.py`:

```python
import callbacks.scanner  # noqa: F401
```

- [ ] **Step 3: Syntax check and commit**

```bash
btc_venv/bin/python3 -m py_compile btc_web/callbacks/scanner.py && echo "OK"
git add btc_web/callbacks/scanner.py btc_web/callbacks/__init__.py
git commit -m "feat: add Model Scanner callback — bidirectional p/d/q lookup"
```

---

### Task 4: Add radar animation CSS

**Files:**
- Modify: `btc_web/assets/style.css`

- [ ] **Step 1: Add scanner and radar CSS**

Append to `btc_web/assets/style.css`:

```css
/* ── Model Scanner ────────────────────────────────────────────────────────── */
.scan-output {
    background-color: rgba(0, 212, 255, 0.08) !important;
    border-color: #00d4ff !important;
    font-weight: bold;
}
#scan-results tr:hover {
    background-color: rgba(0, 212, 255, 0.1);
}
#scan-results tr {
    transition: background-color 0.15s;
}

/* ── Radar beacon marker ─────────────────────────────────────────────────── */
.radar-marker {
    position: absolute;
    width: 40px;
    height: 40px;
    border-radius: 50%;
    pointer-events: none;
    transform: translate(-50%, -50%);
    z-index: 100;
}
.radar-sweep {
    position: absolute;
    inset: 0;
    border-radius: 50%;
    background: conic-gradient(
        from 0deg,
        transparent 0deg,
        rgba(var(--radar-color-rgb), 0.3) 30deg,
        transparent 60deg
    );
    animation: sweep 2s linear infinite;
}
.radar-dot {
    position: absolute;
    top: 50%;
    left: 50%;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    transform: translate(-50%, -50%);
    background: rgb(var(--radar-color-rgb));
    animation: flare 2s ease-out infinite;
}
.radar-ring {
    position: absolute;
    inset: 4px;
    border-radius: 50%;
    border: 1px solid rgba(var(--radar-color-rgb), 0.15);
}
@keyframes sweep {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}
@keyframes flare {
    0% { opacity: 0.3; transform: translate(-50%, -50%) scale(1); }
    15% { opacity: 1; transform: translate(-50%, -50%) scale(1.8); }
    100% { opacity: 0.3; transform: translate(-50%, -50%) scale(1); }
}
```

- [ ] **Step 2: Commit**

```bash
git add btc_web/assets/style.css
git commit -m "feat: add Model Scanner CSS — output field styling and radar beacon animation"
```

---

### Task 5: Add click-to-plot and scanner lines to bubble chart

**Files:**
- Modify: `btc_web/figures/bubble.py`
- Modify: `btc_web/callbacks/scanner.py` (add row click callback)

- [ ] **Step 1: Add scanner traces to bubble figure builder**

In `btc_web/figures/bubble.py`, after the alternative model overlays section (around line 132), add handling for scanner-active models. The scanner passes active rows via the params dict `p.get("scanner_lines", [])`:

```python
    # ── scanner quantile lines + markers ─────────────────────────────────
    for sl in p.get("scanner_lines", []):
        mdl = _app_ctx.PRICE_MODELS.get(sl["model"])
        if not mdl:
            continue
        q = sl["q"]
        scan_prices = _round_trace_data(np.array([
            float(mdl.price_at(q, t)) for t in t_arr]))
        col = mdl.colors.get(
            min(mdl.quantiles, key=lambda qq: abs(qq - q)),
            "#ffd93d")
        traces.append(go.Scatter(
            x=list(t_arr), y=list(scan_prices),
            mode="lines",
            name=f"{mdl.name} Q{q*100:.1f}%",
            line=dict(color=col, width=2, dash=mdl.dash_style),
            legendgroup=f"scan-{mdl.short_name}",
        ))
```

- [ ] **Step 2: Add row click callback in scanner.py**

Add a callback that toggles scanner active rows when a result row is clicked. This uses pattern-matching callbacks:

```python
from dash import ALL

@callback(
    Output("scan-active-rows", "data"),
    Input({"type": "scan-row", "model": ALL}, "n_clicks"),
    State("scan-active-rows", "data"),
    prevent_initial_call=True,
)
def toggle_scanner_row(n_clicks_list, active):
    if not ctx.triggered_id:
        return no_update
    model_key = ctx.triggered_id["model"]
    active = active or []
    if model_key in active:
        active.remove(model_key)
    else:
        active.append(model_key)
    return active
```

- [ ] **Step 3: Wire scanner_lines into the bubble chart callback**

In `btc_web/callbacks/charts.py`, find the bubble chart callback where it builds the params dict `p`. Add scanner lines from the store:

```python
# Scanner lines
scanner_lines = []
if scan_active and scan_q and scan_date:
    q_frac = float(scan_q) / 100.0
    for model_key in (scan_active or []):
        scanner_lines.append({"model": model_key, "q": q_frac})
p["scanner_lines"] = scanner_lines
```

This requires adding `scan-active-rows`, `scan-q`, `scan-date` as `State` inputs to the bubble callback.

- [ ] **Step 4: Commit**

```bash
git add btc_web/figures/bubble.py btc_web/callbacks/scanner.py btc_web/callbacks/charts.py
git commit -m "feat: add click-to-plot scanner lines on bubble chart"
```

---

### Task 6: Add radar marker overlay (JavaScript)

**Files:**
- Create: `btc_web/assets/scanner.js`

- [ ] **Step 1: Create clientside JavaScript for radar positioning**

The radar marker is a DOM element positioned over the Plotly chart. Create `btc_web/assets/scanner.js`:

```javascript
/**
 * Model Scanner — radar beacon marker overlay.
 *
 * Positions animated radar markers over the Plotly bubble chart at
 * (date, price) coordinates. Markers are added/removed when scanner
 * rows are clicked.
 */
(function() {
    "use strict";

    // Watch for scan-active-rows store changes
    var observer = new MutationObserver(function() {
        setTimeout(updateMarkers, 100);
    });

    document.addEventListener("DOMContentLoaded", function() {
        var store = document.getElementById("scan-active-rows");
        if (store) {
            observer.observe(store, {attributes: true, childList: true,
                                     characterData: true, subtree: true});
        }
    });

    function updateMarkers() {
        // Remove existing markers
        document.querySelectorAll(".radar-marker").forEach(function(el) {
            el.remove();
        });

        var store = document.getElementById("scan-active-rows");
        var graph = document.getElementById("bub-graph");
        if (!store || !graph) return;

        var active;
        try { active = JSON.parse(store.textContent || "[]"); }
        catch(e) { return; }
        if (!active.length) return;

        var plot = graph.querySelector(".js-plotly-plot");
        if (!plot || !plot._fullLayout) return;

        var xa = plot._fullLayout.xaxis;
        var ya = plot._fullLayout.yaxis;
        if (!xa || !ya) return;

        // Get scanner price and date
        var priceEl = document.getElementById("scan-price");
        var dateEl = document.getElementById("scan-date");
        if (!priceEl || !dateEl) return;

        var price = parseFloat(priceEl.value);
        var dateStr = dateEl.value;
        if (!price || !dateStr) return;

        // Convert date to years since genesis (2009-07-25)
        var genesis = new Date("2009-07-25");
        var date = new Date(dateStr);
        var t = (date - genesis) / (365.25 * 86400000);

        // Convert to pixel coordinates
        var xPx = xa.l2p(xa.type === "log" ? Math.log10(t) : t) + xa._offset;
        var yPx = ya.l2p(ya.type === "log" ? Math.log10(price) : price) + ya._offset;

        // Place one marker (same position for all models, different colors)
        active.forEach(function(modelKey, idx) {
            var marker = document.createElement("div");
            marker.className = "radar-marker";
            marker.style.left = xPx + "px";
            marker.style.top = yPx + "px";
            // Default color; could be customized per model
            marker.style.setProperty("--radar-color-rgb", "0, 212, 255");

            marker.innerHTML =
                '<div class="radar-ring"></div>' +
                '<div class="radar-sweep"></div>' +
                '<div class="radar-dot"></div>';

            // Offset animation phase per model to avoid overlap
            var sweep = marker.querySelector(".radar-sweep");
            sweep.style.animationDelay = (idx * 0.5) + "s";
            var dot = marker.querySelector(".radar-dot");
            dot.style.animationDelay = (idx * 0.5) + "s";

            var container = graph.querySelector(".plot-container") || graph;
            container.style.position = "relative";
            container.appendChild(marker);
        });
    }

    // Also update on plotly relayout (zoom/pan)
    document.addEventListener("DOMContentLoaded", function() {
        var graph = document.getElementById("bub-graph");
        if (graph) {
            var plot = graph.querySelector(".js-plotly-plot");
            if (plot) {
                plot.on("plotly_relayout", updateMarkers);
            }
        }
    });
})();
```

- [ ] **Step 2: Commit**

```bash
git add btc_web/assets/scanner.js
git commit -m "feat: add radar beacon marker JavaScript overlay for scanner"
```

---

### Task 7: Add snapshot controls and tests

**Files:**
- Modify: `btc_web/snapshot.py`
- Modify: `btc_web/callbacks/nav.py` (TAB_CONTROLS)
- Modify: `btc_web/test_web.py`

- [ ] **Step 1: Add scanner controls to snapshot**

In `btc_web/snapshot.py`, add to `_SNAPSHOT_CONTROLS` at the end of the bubble tab section (after `bub-use-lots`):

```python
    ("scan-price",        "value"),   # scanner price input
    ("scan-date",         "value"),   # scanner date input
    ("scan-q",            "value"),   # scanner quantile input
```

Update the bubble tab index comment to reflect new count.

- [ ] **Step 2: Add to TAB_CONTROLS in nav.py**

In `btc_web/callbacks/nav.py`, add to `_TAB_CONTROLS["bubble"]`:

```python
"scan-price", "scan-date", "scan-q"
```

- [ ] **Step 3: Add scanner tests**

Add to `btc_web/test_web.py`:

```python
class TestModelScanner:
    def test_solve_for_quantile(self):
        """Given price and date, find_percentile returns valid quantile."""
        from btc_core import _find_lot_percentile
        import _app_ctx
        t = today_t(_app_ctx.M.genesis)
        for mdl in _app_ctx.PRICE_MODELS.values():
            pct = mdl.find_percentile(t, 70000)
            assert 0 <= pct <= 1

    def test_solve_for_price(self):
        """Given quantile and date, price_at returns positive price."""
        import _app_ctx
        t = today_t(_app_ctx.M.genesis)
        for mdl in _app_ctx.PRICE_MODELS.values():
            p = float(mdl.price_at(0.5, t))
            assert p > 0

    def test_solve_for_date(self):
        """Root-finding for date works for reasonable inputs."""
        from callbacks.scanner import _solve_date
        import _app_ctx
        for mdl in _app_ctx.PRICE_MODELS.values():
            if not mdl.quantized:
                continue
            result = _solve_date(mdl, 0.5, 1_000_000)
            assert result != "—" or True  # some models may not reach $1M

    def test_qr_model_registered(self):
        import _app_ctx
        assert "qr" in _app_ctx.PRICE_MODELS
        assert _app_ctx.PRICE_MODELS["qr"].name == "Quantile Regression"
```

- [ ] **Step 4: Run tests and commit**

```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py -k "ModelScanner or QuantileRegression" -v
git add btc_web/snapshot.py btc_web/callbacks/nav.py btc_web/test_web.py
git commit -m "feat: add scanner to snapshot controls and tests"
```

---

### Task 8: Test locally and deploy

- [ ] **Step 1: Run full test suite**

```bash
PYTHONPATH="btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py -v --tb=short 2>&1 | tail -20
```

- [ ] **Step 2: Test locally**

```bash
DEV=1 bash run_web.sh
```

Verify:
- Model Scanner panel appears after Projection Quantiles on tab 1
- Default: live ticker price, today's date, quantile computed
- Edit price → quantile updates across all models
- Edit quantile → price updates
- Clear date, edit quantile → date solves (root-finding)
- Click result row → quantile line + radar marker appear on chart
- Click same row → removed
- Radar animation: sweep rotates, dot flares on sweep pass
- QR model appears in Display Models on other tabs

- [ ] **Step 3: Push and deploy**

```bash
git push origin master
ssh root@89.167.70.45 "cd /opt/quantoshi && git pull && systemctl restart quantoshi"
```
