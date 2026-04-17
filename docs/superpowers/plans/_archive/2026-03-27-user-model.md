# User-Defined Model — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users click two points on the bubble chart to define a custom power law model, fully quantized via empirical residual distribution, registered across all tabs with orange color coding and ticker cycling.

**Architecture:** A `UserModel` class (extends `_FitsBasedModel`) is constructed from two user-clicked points on the log-log bubble chart. Residuals against historical data produce asymmetric quantile shifts (parallel lines, same slope). The model lives in a `dcc.Store("user-model-store")` (session-only) and is reconstructed per-callback. A state machine in `draw-mode-store` manages the tap-zoom-tap refinement flow. An invisible background scatter trace enables `clickData` on empty chart space during draw mode.

**Tech Stack:** Python 3, Plotly Dash 4.0.0, NumPy, `_FitsBasedModel` inheritance

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `archive/btc_app/btc_core.py` | Modify | Add `UserModel` class extending `_FitsBasedModel` |
| `btc_web/callbacks/user_model.py` | **Create** | All draw-mode callbacks: FAB, click capture, accept/adjust/cancel, model construction |
| `btc_web/callbacks/__init__.py` | Modify | Register `user_model` callback module |
| `btc_web/layout/__init__.py` | Modify | Add `user-model-store`, `draw-mode-store` |
| `btc_web/layout/common.py` | Modify | Add confirmation menu + toast to FAB layout |
| `btc_web/figures/bubble.py` | Modify | Invisible background trace in draw mode; 3px width for user's own quantile |
| `btc_web/callbacks/charts.py` | Modify | Add `clickData`, `draw-mode-store`, `user-model-store` to bubble callback |
| `btc_web/callbacks/ticker.py` | Modify | Add U1 to cycling when store has data |
| `btc_web/assets/style.css` | Modify | Menu overlay, toast, chart glow styles |
| `btc_web/test_defaults.py` | Modify | UserModel unit tests |

---

## Task 1: UserModel Class

**Files:**
- Modify: `archive/btc_app/btc_core.py`
- Test: `btc_web/test_defaults.py`

- [ ] **Step 1: Write the failing tests**

Append to `btc_web/test_defaults.py`:

```python
# ── UserModel tests ──────────────────────────────────────────────────────────

def test_user_model_from_two_points():
    """UserModel constructs from two points and historical data."""
    from btc_web.app import app  # noqa: F401
    import _app_ctx
    from btc_core import UserModel
    M = _app_ctx.M
    model = UserModel.from_points(
        t1=5.0, p1=1000.0, t2=15.0, p2=100000.0,
        price_years=M.price_years, price_prices=M.price_prices,
        quantiles=M.QR_QUANTILES)
    assert 0 < model.own_quantile < 1
    assert len(model.fits) == len(M.QR_QUANTILES)
    assert all("intercept" in f and "slope" in f for f in model.fits.values())
    assert model.quantized is True
    assert model.short_name == "u1"


def test_user_model_parallel_lines():
    """All quantile lines share the same slope."""
    from btc_web.app import app  # noqa: F401
    import _app_ctx
    from btc_core import UserModel
    M = _app_ctx.M
    model = UserModel.from_points(
        t1=5.0, p1=1000.0, t2=15.0, p2=100000.0,
        price_years=M.price_years, price_prices=M.price_prices,
        quantiles=M.QR_QUANTILES)
    slopes = [f["slope"] for f in model.fits.values()]
    assert len(set(round(s, 10) for s in slopes)) == 1


def test_user_model_colors_are_orange():
    """All quantile colors should be orange."""
    from btc_web.app import app  # noqa: F401
    import _app_ctx
    from btc_core import UserModel
    M = _app_ctx.M
    model = UserModel.from_points(
        t1=5.0, p1=1000.0, t2=15.0, p2=100000.0,
        price_years=M.price_years, price_prices=M.price_prices,
        quantiles=M.QR_QUANTILES)
    assert all(c == "#e67e22" for c in model.colors.values())


def test_user_model_r2_reasonable():
    """R² values should be in valid range."""
    from btc_web.app import app  # noqa: F401
    import _app_ctx
    from btc_core import UserModel
    M = _app_ctx.M
    model = UserModel.from_points(
        t1=5.0, p1=1000.0, t2=15.0, p2=100000.0,
        price_years=M.price_years, price_prices=M.price_prices,
        quantiles=M.QR_QUANTILES)
    for q, r2 in model.r2_per_quantile.items():
        assert r2 is None or -5 < r2 < 1


def test_user_model_store_roundtrip():
    """Model survives serialize → deserialize via store dict."""
    from btc_web.app import app  # noqa: F401
    import _app_ctx
    from btc_core import UserModel
    M = _app_ctx.M
    model = UserModel.from_points(
        t1=5.0, p1=1000.0, t2=15.0, p2=100000.0,
        price_years=M.price_years, price_prices=M.price_prices,
        quantiles=M.QR_QUANTILES)
    store = model.to_store_dict()
    restored = UserModel.from_store_dict(store)
    assert restored.own_quantile == model.own_quantile
    assert len(restored.fits) == len(model.fits)
    for q in model.fits:
        assert abs(restored.fits[q]["slope"] - model.fits[q]["slope"]) < 1e-10
        assert abs(restored.fits[q]["intercept"] - model.fits[q]["intercept"]) < 1e-10
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /scratch/code/bitcoinprojections && PYTHONPATH=".:btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_defaults.py::test_user_model_from_two_points -v --timeout=30`
Expected: ImportError — `UserModel` does not exist.

- [ ] **Step 3: Implement `UserModel` class in `btc_core.py`**

Add after the `S2FModel` class (end of file, before any `if __name__` block):

```python
class UserModel(_FitsBasedModel):
    """User-defined power law model from two clicked points on log-log chart.

    Fully quantized: parallel lines (same slope, shifted intercepts) derived
    from the empirical residual distribution against historical prices.
    """
    name = "User Model"
    short_name = "u1"
    legend_name = "U1"
    dash_style = "solid"
    quantized = True

    def __init__(self, slope, intercept, shifts, quantiles, r2_per_quantile, own_quantile):
        self.fits = {q: {"intercept": intercept + shifts[q], "slope": slope}
                     for q in quantiles}
        self.quantiles = sorted(quantiles)
        self.r2_per_quantile = r2_per_quantile or {}
        self.own_quantile = own_quantile
        self.colors = {q: "#e67e22" for q in self.quantiles}

    @classmethod
    def from_points(cls, t1, p1, t2, p2, price_years, price_prices, quantiles):
        """Factory: two chart points + historical data → fully quantized model."""
        log_t1, log_p1 = np.log10(max(t1, 0.01)), np.log10(max(p1, 1e-10))
        log_t2, log_p2 = np.log10(max(t2, 0.01)), np.log10(max(p2, 1e-10))
        denom = log_t2 - log_t1
        if abs(denom) < 1e-12:
            denom = 1e-12  # avoid division by zero for near-vertical lines
        slope = (log_p2 - log_p1) / denom
        intercept = log_p1 - slope * log_t1

        # Residuals against historical data
        mask = price_years >= 0.5
        t_hist = np.asarray(price_years[mask], float)
        p_hist = np.asarray(price_prices[mask], float)
        predicted = intercept + slope * np.log10(np.maximum(t_hist, 0.01))
        residuals = np.log10(np.maximum(p_hist, 1e-10)) - predicted

        own_quantile = float(np.mean(residuals <= 0))
        shifts = {q: float(np.percentile(residuals, q * 100)) for q in quantiles}

        # R² per quantile
        r2 = {}
        for q in quantiles:
            pred_q = 10.0 ** (intercept + shifts[q] + slope * np.log10(np.maximum(t_hist, 0.01)))
            r2_val = _compute_log_r2(p_hist, pred_q)
            if r2_val is not None:
                r2[q] = r2_val

        return cls(slope, intercept, shifts, quantiles, r2, own_quantile)

    def to_store_dict(self):
        """Serialize to JSON-safe dict for dcc.Store."""
        slope = self.fits[self.quantiles[0]]["slope"]
        return {
            "slope": slope,
            "intercepts": {str(q): self.fits[q]["intercept"] for q in self.quantiles},
            "r2": {str(q): v for q, v in self.r2_per_quantile.items()},
            "own_quantile": self.own_quantile,
            "quantiles": [float(q) for q in self.quantiles],
        }

    @classmethod
    def from_store_dict(cls, d):
        """Reconstruct from dcc.Store dict."""
        if not d:
            return None
        quantiles = [float(q) for q in d["quantiles"]]
        slope = d["slope"]
        intercepts = {float(q): v for q, v in d["intercepts"].items()}
        r2 = {float(q): v for q, v in d["r2"].items()} if d.get("r2") else {}
        model = cls.__new__(cls)
        model.fits = {q: {"intercept": intercepts[q], "slope": slope} for q in quantiles}
        model.quantiles = sorted(quantiles)
        model.r2_per_quantile = r2
        model.own_quantile = d["own_quantile"]
        model.colors = {q: "#e67e22" for q in quantiles}
        return model
```

- [ ] **Step 4: Run tests**

Run: `cd /scratch/code/bitcoinprojections && PYTHONPATH=".:btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_defaults.py -k "user_model" -v --timeout=60`
Expected: All 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add archive/btc_app/btc_core.py btc_web/test_defaults.py
git commit -m "feat: add UserModel class — fully quantized from two points + empirical residuals"
```

---

## Task 2: Layout — Stores, Confirmation Menu, Toast

**Files:**
- Modify: `btc_web/layout/__init__.py`
- Modify: `btc_web/layout/common.py`
- Modify: `btc_web/assets/style.css`

- [ ] **Step 1: Add stores to layout/__init__.py**

Add after the existing `ticker-model-idx` store (~line 76):

```python
    dcc.Store(id="user-model-store", storage_type="memory", data=None),
    dcc.Store(id="draw-mode-store", storage_type="memory",
              data={"phase": "idle", "point1": None, "point2": None, "pre_draw_zoom": None}),
```

- [ ] **Step 2: Add confirmation menu + toast to `_chart_tab_layout_with_fab` in common.py**

Update the existing `_chart_tab_layout_with_fab` to include the confirmation menu overlay and toast text. The menu is hidden by default and shown via callback when in `confirming_p1` or `confirming_p2` phase.

```python
def _chart_tab_layout_with_fab(controls_fn, graph_id, filename):
    """Chart tab with FAB for user model drawing + confirmation menu + toast."""
    import dash_bootstrap_components as dbc

    fab = html.Button(
        "\u270e",  # ✎ pencil
        id="user-model-fab",
        n_clicks=0,
        style={
            "position": "absolute", "bottom": "14px", "right": "14px",
            "zIndex": 10, "width": "42px", "height": "42px",
            "borderRadius": "50%", "border": "2px solid rgba(255,255,255,0.3)",
            "backgroundColor": "rgba(30,30,40,0.85)",
            "color": "#e67e22", "fontSize": "20px",
            "cursor": "pointer", "display": "flex",
            "alignItems": "center", "justifyContent": "center",
            "boxShadow": "0 2px 8px rgba(0,0,0,0.4)",
            "transition": "all 0.2s ease",
            "lineHeight": "1",
        },
        title="Draw a custom model (click 2 points)",
    )

    confirm_menu = html.Div(
        id="draw-confirm-menu",
        style={"display": "none", "position": "absolute", "bottom": "60px",
               "left": "50%", "transform": "translateX(-50%)", "zIndex": 15,
               "backgroundColor": "rgba(30,30,40,0.95)", "borderRadius": "8px",
               "padding": "8px 12px", "boxShadow": "0 4px 16px rgba(0,0,0,0.5)",
               "whiteSpace": "nowrap"},
        children=[
            dbc.Button("\u2713 Accept", id="draw-accept-btn", color="success",
                       size="sm", className="me-2"),
            dbc.Button("\u21bb Adjust", id="draw-adjust-btn", color="warning",
                       size="sm", className="me-2"),
            dbc.Button("\u2715 Cancel", id="draw-cancel-btn", color="secondary",
                       size="sm"),
        ],
    )

    toast = html.Div(
        id="draw-toast",
        style={"display": "none", "position": "absolute", "top": "10px",
               "left": "50%", "transform": "translateX(-50%)", "zIndex": 15,
               "backgroundColor": "rgba(230,126,34,0.9)", "color": "#fff",
               "borderRadius": "6px", "padding": "6px 14px", "fontSize": "13px",
               "fontWeight": "600", "whiteSpace": "nowrap",
               "pointerEvents": "none"},
        children="Tap two points to define your model",
    )

    # Redraw/Delete menu (shown when model exists and FAB is tapped)
    model_menu = html.Div(
        id="draw-model-menu",
        style={"display": "none", "position": "absolute", "bottom": "60px",
               "left": "50%", "transform": "translateX(-50%)", "zIndex": 15,
               "backgroundColor": "rgba(30,30,40,0.95)", "borderRadius": "8px",
               "padding": "8px 12px", "boxShadow": "0 4px 16px rgba(0,0,0,0.5)",
               "whiteSpace": "nowrap"},
        children=[
            dbc.Button("\u270e Redraw", id="draw-redraw-btn", color="warning",
                       size="sm", className="me-2"),
            dbc.Button("\u2715 Delete", id="draw-delete-btn", color="danger",
                       size="sm", className="me-2"),
            dbc.Button("Cancel", id="draw-dismiss-btn", color="secondary",
                       size="sm"),
        ],
    )

    return dbc.Row([
        dbc.Col([
            controls_fn(),
        ], width=3, className="controls-col overflow-auto",
                style={"maxHeight": "85vh"}),
        dbc.Col([
            html.Div(id=f"{graph_id}-chart-wrap",
                     style={"position": "relative"}, children=[
                dcc.Loading(
                    dcc.Graph(id=graph_id, style=_STYLE_GRAPH_H,
                              config={"scrollZoom": False,
                                      "displayModeBar": "hover",
                                      "toImageButtonOptions": {"format": "png", "scale": 2,
                                                               "filename": filename}}),
                    type="default", color=_BTC_ORANGE,
                ),
                fab,
                confirm_menu,
                model_menu,
                toast,
            ]),
            _export_row(graph_id.replace("-graph", "")),
        ], width=9),
    ], className="g-0")
```

- [ ] **Step 3: Add CSS for menus, toast, and draw-mode chart glow**

Append to `btc_web/assets/style.css`:

```css
/* ── Draw mode: chart border glow ────────────────────────────────────────── */
.draw-mode-active {
    box-shadow: inset 0 0 0 2px rgba(230, 126, 34, 0.5) !important;
    border-radius: 8px;
}

/* ── Draw confirm/model menus: mobile responsive ─────────────────────────── */
@media (max-width: 767px) {
    #draw-confirm-menu, #draw-model-menu {
        bottom: 50px !important;
        padding: 6px 8px !important;
    }
    #draw-confirm-menu .btn, #draw-model-menu .btn {
        font-size: 12px !important;
        padding: 4px 8px !important;
    }
}

/* ── Draw toast: fade out after 3s ───────────────────────────────────────── */
#draw-toast.visible {
    display: block !important;
    animation: toastFadeOut 3s ease-in-out forwards;
}
@keyframes toastFadeOut {
    0%, 70% { opacity: 1; }
    100% { opacity: 0; }
}
/* pointer-events: none already set inline on #draw-toast */
```

- [ ] **Step 4: Verify app loads**

Run: `cd /scratch/code/bitcoinprojections && PYTHONPATH=".:btc_web:archive/btc_app" TESTING=1 btc_venv/bin/python3 -c "from btc_web.app import app; print('OK')"`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add btc_web/layout/__init__.py btc_web/layout/common.py btc_web/assets/style.css
git commit -m "feat: add draw-mode stores, confirmation menu, toast, chart glow CSS"
```

---

## Task 3: Draw-Mode Callbacks (State Machine)

**Files:**
- Create: `btc_web/callbacks/user_model.py`
- Modify: `btc_web/callbacks/__init__.py`

This is the largest task. It implements the full state machine for point placement.

- [ ] **Step 1: Create `btc_web/callbacks/user_model.py`**

```python
"""User-defined model: draw-mode state machine + model construction."""

from dash import Input, Output, State, callback, clientside_callback, no_update, ctx
import _app_ctx
from btc_core import UserModel


# ══════════════════════════════════════════════════════════════════════════════
# FAB click → toggle draw mode or show model menu
# ══════════════════════════════════════════════════════════════════════════════

@callback(
    Output("draw-mode-store", "data", allow_duplicate=True),
    Output("draw-confirm-menu", "style", allow_duplicate=True),
    Output("draw-model-menu", "style", allow_duplicate=True),
    Output("draw-toast", "className", allow_duplicate=True),
    Output("user-model-fab", "className", allow_duplicate=True),
    Input("user-model-fab", "n_clicks"),
    State("draw-mode-store", "data"),
    State("user-model-store", "data"),
    prevent_initial_call=True,
)
def on_fab_click(n_clicks, draw_state, model_data):
    if not n_clicks:
        return no_update, no_update, no_update, no_update, no_update

    phase = draw_state.get("phase", "idle") if draw_state else "idle"
    _HIDDEN = {"display": "none"}

    # If in any draw phase → abort, return to idle
    if phase in ("placing_p1", "confirming_p1", "placing_p2", "confirming_p2"):
        return ({"phase": "idle", "point1": None, "point2": None, "pre_draw_zoom": None},
                _HIDDEN, _HIDDEN, "", "")

    # If showing_menu → dismiss
    if phase == "showing_menu":
        return ({"phase": "idle", "point1": draw_state.get("point1"),
                 "point2": draw_state.get("point2"),
                 "pre_draw_zoom": draw_state.get("pre_draw_zoom")},
                _HIDDEN, _HIDDEN, "", "")

    # idle → check if model exists
    if model_data:
        # Show redraw/delete menu
        menu_style = {"display": "flex", "position": "absolute", "bottom": "60px",
                      "left": "50%", "transform": "translateX(-50%)", "zIndex": 15,
                      "backgroundColor": "rgba(30,30,40,0.95)", "borderRadius": "8px",
                      "padding": "8px 12px", "boxShadow": "0 4px 16px rgba(0,0,0,0.5)",
                      "whiteSpace": "nowrap"}
        new_state = dict(draw_state or {})
        new_state["phase"] = "showing_menu"
        return new_state, _HIDDEN, menu_style, "", ""

    # No model → enter draw mode
    new_state = {"phase": "placing_p1", "point1": None, "point2": None, "pre_draw_zoom": None}
    return new_state, _HIDDEN, _HIDDEN, "visible", "draw-active"


# ══════════════════════════════════════════════════════════════════════════════
# Redraw / Delete / Dismiss buttons
# ══════════════════════════════════════════════════════════════════════════════

@callback(
    Output("draw-mode-store", "data", allow_duplicate=True),
    Output("user-model-store", "data", allow_duplicate=True),
    Output("draw-model-menu", "style", allow_duplicate=True),
    Output("draw-toast", "className", allow_duplicate=True),
    Output("user-model-fab", "className", allow_duplicate=True),
    Input("draw-redraw-btn", "n_clicks"),
    Input("draw-delete-btn", "n_clicks"),
    Input("draw-dismiss-btn", "n_clicks"),
    State("draw-mode-store", "data"),
    prevent_initial_call=True,
)
def on_model_menu(redraw_clicks, delete_clicks, dismiss_clicks, draw_state):
    triggered = ctx.triggered_id
    _HIDDEN = {"display": "none"}

    if triggered == "draw-redraw-btn":
        # Clear model, enter draw mode
        new_state = {"phase": "placing_p1", "point1": None, "point2": None, "pre_draw_zoom": None}
        return new_state, None, _HIDDEN, "visible", "draw-active"

    if triggered == "draw-delete-btn":
        # Clear model, return to idle
        new_state = {"phase": "idle", "point1": None, "point2": None, "pre_draw_zoom": None}
        return new_state, None, _HIDDEN, "", ""

    # dismiss
    new_state = dict(draw_state or {})
    new_state["phase"] = "idle"
    return new_state, no_update, _HIDDEN, "", ""


# ══════════════════════════════════════════════════════════════════════════════
# Accept / Adjust / Cancel buttons
# ══════════════════════════════════════════════════════════════════════════════

@callback(
    Output("draw-mode-store", "data", allow_duplicate=True),
    Output("draw-confirm-menu", "style", allow_duplicate=True),
    Output("user-model-store", "data", allow_duplicate=True),
    Output("user-model-fab", "className", allow_duplicate=True),
    Output("draw-toast", "className", allow_duplicate=True),
    Input("draw-accept-btn", "n_clicks"),
    Input("draw-adjust-btn", "n_clicks"),
    Input("draw-cancel-btn", "n_clicks"),
    State("draw-mode-store", "data"),
    prevent_initial_call=True,
)
def on_confirm_action(accept, adjust, cancel, draw_state):
    triggered = ctx.triggered_id
    _HIDDEN = {"display": "none"}
    phase = draw_state.get("phase", "idle") if draw_state else "idle"

    if triggered == "draw-cancel-btn":
        # Remove current point, stay in placing phase
        if phase == "confirming_p1":
            new_state = dict(draw_state)
            new_state["phase"] = "placing_p1"
            new_state["point1"] = None
            return new_state, _HIDDEN, no_update, "draw-active", ""
        elif phase == "confirming_p2":
            new_state = dict(draw_state)
            new_state["phase"] = "placing_p2"
            new_state["point2"] = None
            return new_state, _HIDDEN, no_update, "draw-active", ""
        return draw_state, _HIDDEN, no_update, "draw-active", ""

    if triggered == "draw-adjust-btn":
        # Stay in same placing phase (the bubble callback will zoom 2x)
        if phase == "confirming_p1":
            new_state = dict(draw_state)
            new_state["phase"] = "placing_p1"
            new_state["_adjust_zoom"] = True  # signal to bubble callback to zoom
            return new_state, _HIDDEN, no_update, "draw-active", ""
        elif phase == "confirming_p2":
            new_state = dict(draw_state)
            new_state["phase"] = "placing_p2"
            new_state["_adjust_zoom"] = True
            return new_state, _HIDDEN, no_update, "draw-active", ""
        return draw_state, _HIDDEN, no_update, "draw-active", ""

    if triggered == "draw-accept-btn":
        if phase == "confirming_p1":
            new_state = dict(draw_state)
            new_state["phase"] = "placing_p2"
            return new_state, _HIDDEN, no_update, "draw-active", ""
        elif phase == "confirming_p2":
            # Both points accepted → construct model
            p1 = draw_state["point1"]
            p2 = draw_state["point2"]
            M = _app_ctx.M
            model = UserModel.from_points(
                t1=p1["t"], p1=p1["price"],
                t2=p2["t"], p2=p2["price"],
                price_years=M.price_years,
                price_prices=M.price_prices,
                quantiles=list(M.QR_QUANTILES),
            )
            store_data = model.to_store_dict()
            new_state = {"phase": "idle", "point1": None, "point2": None, "pre_draw_zoom": None}
            return new_state, _HIDDEN, store_data, "", ""

    return draw_state, _HIDDEN, no_update, "", ""


# ══════════════════════════════════════════════════════════════════════════════
# Tab switch → auto-cancel draw mode
# ══════════════════════════════════════════════════════════════════════════════

@callback(
    Output("draw-mode-store", "data", allow_duplicate=True),
    Output("draw-confirm-menu", "style", allow_duplicate=True),
    Output("draw-model-menu", "style", allow_duplicate=True),
    Output("user-model-fab", "className", allow_duplicate=True),
    Input("main-tabs", "active_tab"),
    State("draw-mode-store", "data"),
    prevent_initial_call=True,
)
def on_tab_switch(active_tab, draw_state):
    _HIDDEN = {"display": "none"}
    phase = draw_state.get("phase", "idle") if draw_state else "idle"
    if phase != "idle":
        new_state = {"phase": "idle", "point1": None, "point2": None, "pre_draw_zoom": None}
        return new_state, _HIDDEN, _HIDDEN, ""
    return no_update, no_update, no_update, no_update
```

- [ ] **Step 2: Register callback module in `callbacks/__init__.py`**

Add this line alongside the other `import callbacks.*` statements:

```python
import callbacks.user_model  # noqa: F401
```

- [ ] **Step 3: Verify app loads**

Run: `cd /scratch/code/bitcoinprojections && PYTHONPATH=".:btc_web:archive/btc_app" TESTING=1 btc_venv/bin/python3 -c "from btc_web.app import app; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add btc_web/callbacks/user_model.py btc_web/callbacks/__init__.py
git commit -m "feat: draw-mode state machine — FAB, accept/adjust/cancel, tab-switch abort"
```

---

## Task 4: Wire Bubble Callback — Click Capture + Figure Integration

**Files:**
- Modify: `btc_web/callbacks/charts.py` (bubble callback only)
- Modify: `btc_web/figures/bubble.py`
- Modify: `btc_web/callbacks/user_model.py`

- [ ] **Step 1: Add click-capture callback to `user_model.py`**

**IMPORTANT:** `clickData` must NOT be an `Input` on the bubble callback — that would cause a full figure rebuild on every chart click. Instead, add a **separate lightweight callback** in `user_model.py` that captures clicks during draw mode:

```python
@callback(
    Output("draw-mode-store", "data", allow_duplicate=True),
    Output("draw-confirm-menu", "style", allow_duplicate=True),
    Input("bubble-graph", "clickData"),
    State("draw-mode-store", "data"),
    prevent_initial_call=True,
)
def on_chart_click(click_data, draw_state):
    """Capture click coordinates during draw mode."""
    phase = (draw_state or {}).get("phase", "idle")
    _HIDDEN = {"display": "none"}

    if phase not in ("placing_p1", "placing_p2"):
        return no_update, no_update

    if not click_data or not click_data.get("points"):
        return no_update, no_update

    pt = click_data["points"][0]
    clicked = {"t": pt["x"], "price": pt["y"]}
    new_draw = dict(draw_state)

    if phase == "placing_p1":
        new_draw["point1"] = clicked
        new_draw["phase"] = "confirming_p1"
    else:
        new_draw["point2"] = clicked
        new_draw["phase"] = "confirming_p2"

    menu_style = {"display": "flex", "position": "absolute", "bottom": "60px",
                  "left": "50%", "transform": "translateX(-50%)", "zIndex": 15,
                  "backgroundColor": "rgba(30,30,40,0.95)", "borderRadius": "8px",
                  "padding": "8px 12px", "boxShadow": "0 4px 16px rgba(0,0,0,0.5)",
                  "whiteSpace": "nowrap"}
    return new_draw, menu_style
```

- [ ] **Step 2: Add draw-mode and user-model state to bubble callback (as State, not Input)**

In `btc_web/callbacks/charts.py`, modify the bubble callback:

Add to **States** (NOT Inputs):
```python
    State("draw-mode-store", "data"),
    State("user-model-store", "data"),
```

Also add `draw-mode-store` as an **Input** so the figure rebuilds when draw state changes (e.g., point placed, mode entered/exited):
```python
    Input("draw-mode-store", "data"),
```

Change `Output("bubble-graph", "figure")` to `Output("bubble-graph", "figure", allow_duplicate=True)` and add `prevent_initial_call="initial_duplicate"`.

In the params dict assembly, add:
```python
    draw_mode = draw_phase if draw_phase != "idle" else None,
    draw_point1 = (draw_state or {}).get("point1"),
    draw_point2 = (draw_state or {}).get("point2"),
    user_model = user_model_store,
```

Where `draw_phase = (draw_state or {}).get("phase", "idle")` and `draw_state` / `user_model_store` come from the State inputs.

- [ ] **Step 2: Modify `figures/bubble.py` — invisible background trace + user model rendering**

In `build_bubble_figure`, add:

**Invisible background trace (when draw mode is active):**
```python
if p.get("draw_mode"):
    # Dense invisible grid so clickData fires on empty chart space
    import numpy as np
    bg_t = np.logspace(np.log10(max(t_lo, 0.1)), np.log10(t_hi), 50)
    bg_p = np.logspace(np.log10(max(y_lo, 0.01)), np.log10(y_hi), 50)
    bg_tt, bg_pp = np.meshgrid(bg_t, bg_p)
    traces.insert(0, go.Scatter(
        x=bg_tt.ravel().tolist(), y=bg_pp.ravel().tolist(),
        mode="markers", marker=dict(size=20, opacity=0.001),
        hoverinfo="skip", showlegend=False,
        name="_bg_click_target",
    ))
```

**Draw-mode point markers:**
```python
if p.get("draw_point1"):
    pt = p["draw_point1"]
    traces.append(go.Scatter(
        x=[pt["t"]], y=[pt["price"]],
        mode="markers", marker=dict(size=12, color="#e67e22", symbol="circle",
                                     line=dict(color="white", width=2)),
        showlegend=False, hoverinfo="skip", name="_draw_p1",
    ))
if p.get("draw_point2"):
    pt = p["draw_point2"]
    traces.append(go.Scatter(
        x=[pt["t"]], y=[pt["price"]],
        mode="markers", marker=dict(size=12, color="#e67e22", symbol="circle",
                                     line=dict(color="white", width=2)),
        showlegend=False, hoverinfo="skip", name="_draw_p2",
    ))
```

**User model overlay (3px for own quantile):**

In the overlay model loop, handle `"u1"` specially:
```python
if model_key == "u1":
    um_data = p.get("user_model")
    if not um_data:
        continue
    mdl = UserModel.from_store_dict(um_data)
    if not mdl:
        continue
else:
    mdl = _app_ctx.PRICE_MODELS.get(model_key)
    if not mdl:
        continue
```

And when rendering quantile lines for U1, use 3px for the `own_quantile`:
```python
lw = 3.0 if (model_key == "u1" and abs(q - mdl.own_quantile) < 0.001) else _OVERLAY_LINE_WIDTH
```

**Set `dragmode=False` during draw mode:**
```python
if p.get("draw_mode"):
    layout["dragmode"] = False
```

- [ ] **Step 3: Run tests**

Run: `cd /scratch/code/bitcoinprojections && PYTHONPATH=".:btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_web.py -k "bubble" -v --timeout=60`
Expected: All bubble tests pass (pre-existing failures excluded).

- [ ] **Step 4: Commit**

```bash
git add btc_web/callbacks/charts.py btc_web/figures/bubble.py
git commit -m "feat: bubble callback click capture + invisible background trace + user model overlay"
```

---

## Task 5: Ticker Integration + Display Models Checklists

**Files:**
- Modify: `btc_web/callbacks/ticker.py`
- Modify: `btc_web/callbacks/user_model.py` (add checklist injection callback)

- [ ] **Step 1: Add U1 to ticker cycling**

In `btc_web/callbacks/ticker.py`:

Add `State("user-model-store", "data")` to the callback inputs.

After the main `_MODEL_CYCLE` loop that builds `model_data`, add:

```python
    # Append user model if defined
    if user_model_data:
        from btc_core import UserModel
        um = UserModel.from_store_dict(user_model_data)
        if um:
            t = today_t(_app_ctx.M.genesis)
            try:
                um_pct = um.find_percentile(t, price)
                um_pct_int = round(um_pct * 100) if um_pct is not None else None
                if um_pct_int is not None:
                    model_data.append({"key": "u1", "label": "U1",
                                       "pct": um_pct_int, "color": "#e67e22"})
            except Exception:
                pass
```

- [ ] **Step 2: Add dynamic checklist option injection to `user_model.py`**

Add a callback that watches `user-model-store` and injects/removes the "U1" option from Display Models checklists on all tabs:

```python
# Prefixes for all tabs with model-show checklists
_MODEL_SHOW_PREFIXES = ["bub", "dca", "ret", "sc"]  # heatmap uses pill bar, not checklist

@callback(
    [Output(f"{p}-model-show", "options", allow_duplicate=True) for p in _MODEL_SHOW_PREFIXES],
    Input("user-model-store", "data"),
    [State(f"{p}-model-show", "options") for p in _MODEL_SHOW_PREFIXES],
    prevent_initial_call=True,
)
def inject_user_model_option(user_data, *current_options_list):
    results = []
    u1_opt = {"label": " U1 (User)", "value": "u1"}
    for opts in current_options_list:
        opts = list(opts or [])
        # Remove any existing u1 option
        opts = [o for o in opts if o.get("value") != "u1"]
        # Add if user model exists
        if user_data:
            opts.append(u1_opt)
        results.append(opts)
    return results
```

- [ ] **Step 3: Verify app loads + run tests**

Run: `cd /scratch/code/bitcoinprojections && PYTHONPATH=".:btc_web:archive/btc_app" TESTING=1 btc_venv/bin/python3 -c "from btc_web.app import app; print('OK')"`

Run: `cd /scratch/code/bitcoinprojections && PYTHONPATH=".:btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_defaults.py btc_web/test_web.py --timeout=180 -q 2>&1 | tail -5`

- [ ] **Step 4: Commit**

```bash
git add btc_web/callbacks/ticker.py btc_web/callbacks/user_model.py
git commit -m "feat: U1 in ticker cycling + dynamic Display Models checklist injection"
```

---

## Task 6: Wire Other Tab Figure Builders for U1 Overlay

**Files:**
- Modify: `btc_web/callbacks/charts.py` (DCA, Retire, SC callbacks)
- Modify: `btc_web/callbacks/citadel_cb.py`

Each chart callback needs to pass `user-model-store` data to its figure builder when `"u1"` is in `active_models`.

- [ ] **Step 1: Add `State("user-model-store", "data")` to DCA, Retire, SC, Citadel callbacks**

In each callback function, add logic to pass user model data into the params dict:

```python
# In each callback, after assembling the params dict:
if "u1" in (model_show or []):
    params["user_model"] = user_model_store
```

The figure builders (`figures/dca.py`, `figures/retire.py`, `figures/supercharge.py`, `figures/citadel.py`) already handle the overlay model loop via `p.get("active_models", [])`. They look up models from `_app_ctx.PRICE_MODELS`. For `"u1"`, they need the same special handling as bubble:

```python
# In each figure builder's overlay loop:
if model_key == "u1":
    from btc_core import UserModel
    um_data = p.get("user_model")
    if not um_data:
        continue
    mdl = UserModel.from_store_dict(um_data)
    if not mdl:
        continue
else:
    mdl = _app_ctx.PRICE_MODELS.get(model_key)
```

- [ ] **Step 2: Run full test suite**

Run: `cd /scratch/code/bitcoinprojections && PYTHONPATH=".:btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_defaults.py btc_web/test_web.py --timeout=180 -q 2>&1 | tail -5`

- [ ] **Step 3: Commit**

```bash
git add btc_web/callbacks/charts.py btc_web/callbacks/citadel_cb.py btc_web/figures/dca.py btc_web/figures/retire.py btc_web/figures/supercharge.py btc_web/figures/citadel.py
git commit -m "feat: U1 overlay support on DCA, Retire, Supercharge, Citadel tabs"
```

---

## Task 7: End-to-End Verification + Polish

**Files:** Any files needing fixes

- [ ] **Step 1: Run full test suite**

Run: `cd /scratch/code/bitcoinprojections && PYTHONPATH=".:btc_web:archive/btc_app" btc_venv/bin/python3 -m pytest btc_web/test_defaults.py btc_web/test_web.py --timeout=180 -q 2>&1 | tail -10`
Expected: All tests pass (pre-existing failures excluded), 0 regressions.

- [ ] **Step 2: Manual smoke test**

Start dev server: `cd /scratch/code/bitcoinprojections && DEV=1 bash run_web.sh`

Test flow:
1. Tab 1 → tap FAB → FAB pulses, toast shows
2. Tap a point on chart → orange marker + confirm menu
3. Tap Accept → advance to point 2
4. Tap another point → orange marker + confirm menu
5. Tap Accept → line drawn with quantile bands
6. U1 appears in Display Models checklist
7. Tap ticker percentile → cycles through models, U1 included
8. Tab 3 (DCA) → check U1 in Display Models → U1 overlay appears
9. Tap FAB → Redraw/Delete menu
10. Tap Delete → model removed, U1 gone from checklists and ticker

- [ ] **Step 3: Fix any issues found**

- [ ] **Step 4: Commit fixes**

```bash
git add -u
git commit -m "fix: user model end-to-end polish"
```
