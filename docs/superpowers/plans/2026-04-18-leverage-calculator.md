# Leverage Calculator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `/leverage` — an interactive page on Quantoshi that computes the maximum rational pay-price for BTC given horizon H, target CAGR, borrow/opportunity rates, a chosen price model, and a floor quantile.

**Architecture:** Hidden 10th Dash tab (not shown in tab bar; URL-only via `/leverage` or `/10`). Pure-arithmetic calculator on top of each `PriceModel`'s existing `interp_price(q, t)` API. No cache, no payment gate. Snapshot-integrated, palette-aware, mobile-responsive.

**Tech Stack:** Dash 4.0.0, DBC 2.0.4, Python 3.12+ (dev: 3.14.3), pandas, numpy, plotly.

**Pre-read before starting:**
- Spec: `/scratch/code/bitcoinprojections/docs/superpowers/specs/2026-04-18-leverage-calculator-design.md` — especially §5 (math) and §7 (files)
- `/scratch/code/bitcoinprojections/CLAUDE.md` "Known gotchas" section (allow_duplicate, step validation, falsy-zero, tab-map drift)
- `btc_web/layout/heatmap.py::_hm_pill_bar()` — reference pill-bar pattern (we parallel it, not inherit)
- `btc_web/tab_defaults.py` — existing frozen-defaults pattern

**Shared project genesis:** `pd.Timestamp("2009-07-25")` — every `PriceModel` uses this as t₀. Do not invent per-model origins.

**Model data singleton:** `_app_ctx.M` (set at `app.py:238`). There is no `_app_ctx.MODEL_DATA`.

**Callback registration pattern:** module-level `@callback` (and `app.clientside_callback`) decorators with registration as an import side-effect. Sample: `callbacks/user_model.py`, `callbacks/scanner.py`. Do NOT write `def register(app)` wrappers.

**Color constants:** if Task 4 or Task 8 introduces any new palette-aware constants (`LEV_*`, `ALERT_*`), add them to `btc_web/colors.py` in the appropriate section (per-palette in §2 if palette-varying, palette-invariant in §1 otherwise), then regenerate CSS/JS artifacts:

```bash
btc_venv/bin/python3 tools/generate_color_artifacts.py
```

MVP uses only existing constants (`TRACE_WIDTH`, `TRACE_WIDTH_COMPOSITE`, standard palette colors) — skip this unless the implementer adds new ones.

---

### Task 1: Add `LEVERAGE_DEFAULTS` to `tab_defaults.py`

**Files:**
- Modify: `btc_web/tab_defaults.py`

- [ ] **Step 1: Write the failing test**

Create `btc_web/test_leverage.py`:

```python
"""Leverage calculator unit tests."""
from __future__ import annotations

import sys
sys.path.insert(0, "/scratch/code/bitcoinprojections/btc_web")

import datetime as _dt

import pytest


def test_leverage_defaults_has_expected_keys():
    from tab_defaults import LEVERAGE_DEFAULTS, leverage_defaults

    d = leverage_defaults()
    # Defaults used by the callback — must match the spec exactly.
    assert d["lev_price"] > 0  # seeded from model_data most recent close
    assert d["lev_date"] is not None
    assert d["lev_model"] == "bub"
    assert d["lev_floor_q"] == 0.01
    assert d["lev_rb"] == 13.0
    assert d["lev_rl"] == 4.5
    assert d["lev_horizon"] == 4.0
    assert d["lev_cagr"] == 20.0
    # Frozen-dict invariant
    assert type(LEVERAGE_DEFAULTS).__name__ == "mappingproxy"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /scratch/code/bitcoinprojections
btc_venv/bin/python3 -m pytest btc_web/test_leverage.py::test_leverage_defaults_has_expected_keys -v
```

Expected: FAIL with `ImportError` on `LEVERAGE_DEFAULTS` or `leverage_defaults`.

- [ ] **Step 3: Add LEVERAGE_DEFAULTS**

Append to `btc_web/tab_defaults.py` (after the last tab's defaults):

```python
# Leverage Calculator tab — see docs/superpowers/specs/2026-04-18-leverage-calculator-design.md
LEVERAGE_DEFAULTS = MappingProxyType({
    "lev_date":     None,   # resolved to date.today() in leverage_defaults()
    "lev_price":    None,   # resolved to most recent model_data close in leverage_defaults()
    "lev_model":    "bub",
    "lev_floor_q":  0.01,
    "lev_rb":       13.0,
    "lev_rl":       4.5,
    "lev_horizon":  4.0,
    "lev_cagr":     20.0,
})


def leverage_defaults():
    """Return a plain dict with dynamic fields resolved."""
    import datetime as _dt
    import _app_ctx
    md = _app_ctx.M  # ModelData instance — set at app.py:238
    latest_close = float(md.price_prices[-1]) if md is not None else 65000.0
    d = dict(LEVERAGE_DEFAULTS)
    d["lev_date"] = _dt.date.today().isoformat()
    d["lev_price"] = latest_close
    return d
```

- [ ] **Step 4: Run test to verify it passes**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_leverage.py::test_leverage_defaults_has_expected_keys -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add btc_web/tab_defaults.py btc_web/test_leverage.py
git commit -m "feat(leverage): add LEVERAGE_DEFAULTS tab defaults"
```

---

### Task 2: `floor_price()` helper + tests

**Files:**
- Create: `btc_web/figures/leverage.py`
- Modify: `btc_web/test_leverage.py`

- [ ] **Step 1: Write the failing tests**

Append to `btc_web/test_leverage.py`:

```python
def test_floor_price_bm_at_today():
    """BM Q1% floor at today should be a positive price."""
    from figures.leverage import floor_price
    import datetime as _dt
    price = floor_price("bub", 0.01, _dt.date.today())
    assert price > 0
    assert price < 10_000_000  # sanity upper bound


def test_floor_price_higher_q_gives_higher_price():
    """Q5% floor > Q1% floor at same date (higher quantile = more aggressive)."""
    from figures.leverage import floor_price
    import datetime as _dt
    d = _dt.date.today()
    assert floor_price("bub", 0.05, d) > floor_price("bub", 0.01, d)


def test_floor_price_future_higher_than_today():
    """Floor grows over time (power-law)."""
    from figures.leverage import floor_price
    import datetime as _dt
    today = _dt.date.today()
    future = today.replace(year=today.year + 5)
    assert floor_price("bub", 0.01, future) > floor_price("bub", 0.01, today)


def test_floor_price_rejects_s2f_silently_returning_zero_q():
    """S2F.interp_price ignores q. Not in dropdown, but guard against silent misuse."""
    from figures.leverage import floor_price
    import _app_ctx
    import datetime as _dt
    if "s2f" in _app_ctx.PRICE_MODELS:
        # Not a blocker test — documents the footgun.
        d = _dt.date.today()
        assert floor_price("s2f", 0.01, d) == floor_price("s2f", 0.50, d)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_leverage.py -v -k floor_price
```

Expected: FAIL (ImportError on `figures.leverage`).

- [ ] **Step 3: Create figures/leverage.py with floor_price**

Create `btc_web/figures/leverage.py`:

```python
"""Leverage calculator — figure builder and math helpers.

Design spec: docs/superpowers/specs/2026-04-18-leverage-calculator-design.md
"""
from __future__ import annotations

import datetime as _dt

import pandas as pd

import _app_ctx

# Shared project genesis — every PriceModel in btc_core uses this as t=0
# (CLAUDE.md: "All models use 2009-07-25 as their time origin").
_GENESIS = pd.Timestamp("2009-07-25")


def floor_price(model_short: str, q: float, target_date) -> float:
    """Return the `model_short`-q floor price at `target_date` in USD.

    Args:
        model_short: key into _app_ctx.PRICE_MODELS (e.g. "bub", "pl", "lppl").
        q: quantile in (0, 1), e.g. 0.01 for Q1%.
        target_date: datetime.date or datetime.datetime.

    Returns:
        Floor price in USD (positive float).
    """
    model = _app_ctx.PRICE_MODELS[model_short]
    t_yr = (pd.Timestamp(target_date) - _GENESIS).days / 365.25
    return float(model.interp_price(q, t_yr))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_leverage.py -v -k floor_price
```

Expected: PASS (3 or 4 tests).

- [ ] **Step 5: Commit**

```bash
git add btc_web/figures/leverage.py btc_web/test_leverage.py
git commit -m "feat(leverage): add floor_price helper wrapping interp_price"
```

---

### Task 3: Pure math helpers (`P_max`, `implied_cagr`)

**Files:**
- Modify: `btc_web/figures/leverage.py`
- Modify: `btc_web/test_leverage.py`

- [ ] **Step 1: Write the failing tests**

Append to `btc_web/test_leverage.py`:

```python
def test_P_max_basic():
    from figures.leverage import P_max
    # sell=181649, H=4, c=0.20 -> P_max ≈ 87601
    assert P_max(181649, 4, 0.20) == pytest.approx(87601, rel=1e-3)


def test_P_max_zero_cagr_equals_sell_price():
    from figures.leverage import P_max
    assert P_max(100_000, 4, 0.0) == 100_000


def test_implied_cagr_basic():
    from figures.leverage import implied_cagr
    # sell=181649, P_now=72926, H=4 -> c ≈ 0.256
    assert implied_cagr(181649, 72926, 4) == pytest.approx(0.256, rel=1e-2)


def test_implied_cagr_zero_price_returns_none():
    from figures.leverage import implied_cagr
    assert implied_cagr(181649, 0, 4) is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_leverage.py -v -k "P_max or implied_cagr"
```

Expected: FAIL (ImportError).

- [ ] **Step 3: Add helpers to figures/leverage.py**

Append to `btc_web/figures/leverage.py`:

```python
def P_max(sell_price: float, H_yr: float, target_cagr: float) -> float:
    """Max rational pay-price today for a target CAGR c over horizon H."""
    return sell_price / (1.0 + target_cagr) ** H_yr


def implied_cagr(sell_price: float, P_now: float, H_yr: float):
    """CAGR implied by buying at P_now today and selling at sell_price in H years."""
    if P_now <= 0 or H_yr <= 0:
        return None
    return (sell_price / P_now) ** (1.0 / H_yr) - 1.0
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_leverage.py -v
```

Expected: PASS (all 7-8 tests so far).

- [ ] **Step 5: Commit**

```bash
git add btc_web/figures/leverage.py btc_web/test_leverage.py
git commit -m "feat(leverage): add P_max and implied_cagr math helpers"
```

---

### Task 4: `build_leverage_figure()` — the plot

**Files:**
- Modify: `btc_web/figures/leverage.py`
- Modify: `btc_web/test_leverage.py`

- [ ] **Step 1: Write the failing test**

Append to `btc_web/test_leverage.py`:

```python
def test_build_leverage_figure_returns_plotly_figure():
    from figures.leverage import build_leverage_figure
    p = {
        "lev_price": 72926.0,
        "lev_date": "2026-04-18",
        "lev_model": "bub",
        "lev_floor_q": 0.01,
        "lev_rb": 13.0,
        "lev_rl": 4.5,
        "lev_horizon": 4.0,
        "lev_cagr": 20.0,
        "palette": "default",
    }
    fig = build_leverage_figure(p)
    # Must be a plotly Figure with 4 curves + price line + vertical line
    assert fig is not None
    assert len(fig.data) >= 4  # at least 4 curves
    # Y-axis log scale
    assert fig.layout.yaxis.type == "log"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_leverage.py::test_build_leverage_figure_returns_plotly_figure -v
```

Expected: FAIL (ImportError).

- [ ] **Step 3: Implement build_leverage_figure**

Append to `btc_web/figures/leverage.py`:

```python
import datetime as _dt
import numpy as np
import plotly.graph_objects as go

from colors import (
    TRACE_WIDTH, TRACE_WIDTH_COMPOSITE, WHITE,
    PLOT_BG_COLOR, TEXT_COLOR, GRID_MAJOR_COLOR,
)
from figures.common import _dark_layout, _apply_watermark


def _parse_date(s):
    """Accept ISO date string, date, or datetime; return datetime.date."""
    if isinstance(s, _dt.datetime):
        return s.date()
    if isinstance(s, _dt.date):
        return s
    return _dt.date.fromisoformat(str(s)[:10])


def build_leverage_figure(p: dict) -> go.Figure:
    """Build the max-pay-price plot.

    Params dict (all numeric except date/model/palette):
        lev_price, lev_date, lev_model, lev_floor_q,
        lev_rb, lev_rl, lev_horizon, lev_cagr, palette
    """
    # Guards (spec §5.5)
    H_slider = max(float(p["lev_horizon"]), 0.01)
    P_now    = max(float(p["lev_price"]), 1.0)
    c        = float(p["lev_cagr"]) / 100.0
    r_b      = float(p["lev_rb"]) / 100.0
    r_l      = float(p["lev_rl"]) / 100.0
    q        = float(p["lev_floor_q"])
    model    = str(p["lev_model"])
    buy_date = _parse_date(p["lev_date"])

    # Grid of horizons for plotting
    H_grid = np.linspace(0.25, 20.0, 400)

    # Vectorized sell-price: call floor_price once per H.
    dates = [buy_date + _dt.timedelta(days=int(round(H * 365.25))) for H in H_grid]
    try:
        sell_grid = np.array([floor_price(model, q, d) for d in dates])
    except (KeyError, AttributeError, ValueError):
        # Graceful degradation — model may not be in PRICE_MODELS or lacks interp_price
        sell_grid = np.full_like(H_grid, np.nan)

    # Four curves: 0%, r_l, r_b, your c
    def _curve(target_c):
        return sell_grid / (1.0 + target_c) ** H_grid

    curve_0   = _curve(0.0)
    curve_rl  = _curve(r_l)
    curve_rb  = _curve(r_b)
    curve_c   = _curve(c)

    fig = go.Figure()

    # Reference curves (thin)
    fig.add_trace(go.Scatter(
        x=H_grid, y=curve_0, name="Nominal breakeven (0%)",
        line=dict(width=TRACE_WIDTH, dash="dot"),
        hovertemplate="H=%{x:.2f} yr<br>Max pay=%{y:$,.0f}<extra>0%</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=H_grid, y=curve_rl, name=f"Opp cost ({r_l*100:.2f}%)",
        line=dict(width=TRACE_WIDTH),
        hovertemplate="H=%{x:.2f} yr<br>Max pay=%{y:$,.0f}<extra>r_l</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=H_grid, y=curve_rb, name=f"Borrow cost ({r_b*100:.2f}%)",
        line=dict(width=TRACE_WIDTH),
        hovertemplate="H=%{x:.2f} yr<br>Max pay=%{y:$,.0f}<extra>r_b</extra>",
    ))
    # Your target (thick, highlighted)
    fig.add_trace(go.Scatter(
        x=H_grid, y=curve_c, name=f"Your target ({c*100:.1f}%)",
        line=dict(width=TRACE_WIDTH_COMPOSITE),
        hovertemplate="H=%{x:.2f} yr<br>Max pay=%{y:$,.0f}<extra>target</extra>",
    ))

    # Current price horizontal
    fig.add_hline(y=P_now, line=dict(dash="dash"),
                  annotation_text=f"Current: ${P_now:,.0f}",
                  annotation_position="right")

    # Slider-position vertical + intersection dot (guarded — same try path as curves)
    try:
        sell_at_slider = floor_price(model, q, buy_date + _dt.timedelta(days=int(round(H_slider * 365.25))))
        y_dot = sell_at_slider / (1.0 + c) ** H_slider
        fig.add_vline(x=H_slider, line=dict(dash="dash"))
        fig.add_trace(go.Scatter(
            x=[H_slider], y=[y_dot], mode="markers",
            marker=dict(size=12, symbol="circle"),
            name="Your max pay today", showlegend=False,
            hovertemplate="H=%{x:.2f} yr<br>Max pay=%{y:$,.0f}<extra></extra>",
        ))
    except (KeyError, AttributeError, ValueError):
        pass  # fall through without the dot — curves already NaN'd out

    # Layout
    q_label = f"Q{q*100:g}%"
    fig.update_layout(
        title=dict(
            text=(
                f"<b>Max rational pay-price — reversion to "
                f"{model} {q_label} floor</b><br>"
                f"<span style='font-size:0.85em'>"
                f"Current date: {buy_date.isoformat()}  ·  "
                f"Current price: ${P_now:,.0f}</span>"
            )
        ),
        xaxis=dict(title="Horizon H (years)", range=[0.25, 20]),
        yaxis=dict(title="Max pay-price today ($)", type="log",
                   tickformat="$,.0f"),
        margin=dict(l=60, r=40, t=90, b=60),
    )
    _dark_layout(fig)
    _apply_watermark(fig)
    return fig
```

- [ ] **Step 4: Run test to verify it passes**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_leverage.py::test_build_leverage_figure_returns_plotly_figure -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add btc_web/figures/leverage.py btc_web/test_leverage.py
git commit -m "feat(leverage): add build_leverage_figure with 4-curve plot"
```

---

### Task 5: `layout/leverage.py` — `_leverage_tab()` function

**Files:**
- Create: `btc_web/layout/leverage.py`

- [ ] **Step 1: Create the layout function**

Create `btc_web/layout/leverage.py`:

```python
"""Leverage Calculator tab — control panel + output widgets.

Design spec: docs/superpowers/specs/2026-04-18-leverage-calculator-design.md
"""
from __future__ import annotations

import datetime as _dt

import dash_bootstrap_components as dbc
from dash import dcc, html

import _app_ctx
from colors import (
    UI_FONT_MD, UI_FONT_LG, UI_FONT_XL,
    MUTED_TEXT, TEXT_COLOR, LINK, WHITE,
)
from tab_defaults import leverage_defaults


# Curated flagship model list for the "Reversion target model" dropdown (§4.4).
_LEV_FLAGSHIP_MODELS = ["bub", "pl", "qr", "lppl", "hybppl", "eppl", "pca", "grdy"]
# Conditionally appended at build time: "ef" (if model present), "u1" (if drawn)


def _model_options():
    """Build dropdown options from PRICE_MODELS, respecting flagship ordering."""
    opts = []
    for key in _LEV_FLAGSHIP_MODELS:
        m = _app_ctx.PRICE_MODELS.get(key)
        if m is not None:
            opts.append({"label": m.name, "value": key})
    # Conditionals
    if "ef" in _app_ctx.PRICE_MODELS:
        opts.append({"label": _app_ctx.PRICE_MODELS["ef"].name, "value": "ef"})
    # U1 is handled at runtime (dropdown rebuilt when user draws)
    return opts


# Floor quantile pill bar IDs — hardcoded at module import (parallel to
# routing.py::_HM_PILL_IDS; cannot be inherited — see spec §4.2).
_LEV_FLOOR_QS = [0.001, 0.01, 0.05, 0.10, 0.15, 0.20]
_LEV_PILL_IDS = [f"lev-pill-q{int(q*1000):03d}" for q in _LEV_FLOOR_QS]
# i.e. lev-pill-q001 (Q0.1%), lev-pill-q010 (Q1%), lev-pill-q050 (Q5%), ...


def _floor_pill_bar():
    """Render 6 pill buttons for floor quantile selection."""
    labels = ["Q0.1%", "Q1%", "Q5%", "Q10%", "Q15%", "Q20%"]
    default_q = 0.01
    return html.Div([
        dbc.Button(
            lbl, id=pid, size="sm", className="me-1",
            outline=(q != default_q), color="primary",
            n_clicks=0,
        )
        for pid, lbl, q in zip(_LEV_PILL_IDS, labels, _LEV_FLOOR_QS)
    ], className="d-flex flex-wrap")


def _leverage_tab() -> html.Div:
    """Build the Leverage Calculator tab content."""
    d = leverage_defaults()
    return html.Div([
        # Scenario context row
        dbc.Row([
            dbc.Col([
                html.Label("Current date", style={"fontSize": UI_FONT_MD}),
                dcc.DatePickerSingle(id="lev-date", date=d["lev_date"]),
            ], md=4, xs=12, className="mb-2"),
            dbc.Col([
                html.Label("Current price ($)", style={"fontSize": UI_FONT_MD}),
                dbc.Input(id="lev-price", type="number", min=1, step=0.01,
                          value=d["lev_price"]),
            ], md=4, xs=12, className="mb-2"),
        ], className="mb-3"),

        # Reversion target row
        dbc.Row([
            dbc.Col([
                html.Label("Reversion target model", style={"fontSize": UI_FONT_MD}),
                dcc.Dropdown(id="lev-model", options=_model_options(),
                             value=d["lev_model"], clearable=False),
            ], md=6, xs=12, className="mb-2"),
            dbc.Col([
                html.Label("Floor quantile", style={"fontSize": UI_FONT_MD}),
                _floor_pill_bar(),
                dcc.Store(id="lev-floor-q-store", data=d["lev_floor_q"]),
            ], md=6, xs=12, className="mb-2"),
        ], className="mb-3"),

        # Rate environment row
        dbc.Row([
            dbc.Col([
                html.Label("Borrow rate r_b (0–50 % / yr)",
                           style={"fontSize": UI_FONT_MD}),
                dbc.Input(id="lev-rb", type="number", min=0, max=50, step=0.001,
                          value=d["lev_rb"]),
            ], md=6, xs=12, className="mb-2"),
            dbc.Col([
                html.Label("Opportunity cost r_l (0–50 % / yr)",
                           style={"fontSize": UI_FONT_MD}),
                dbc.Input(id="lev-rl", type="number", min=0, max=50, step=0.001,
                          value=d["lev_rl"]),
            ], md=6, xs=12, className="mb-2"),
        ], className="mb-3"),

        # Your-scenario row
        dbc.Row([
            dbc.Col([
                html.Label("Horizon H (years)", style={"fontSize": UI_FONT_MD}),
                dcc.Slider(id="lev-horizon", min=0.25, max=20, step=0.25,
                           value=d["lev_horizon"],
                           marks={i: str(i) for i in range(0, 21, 2)},
                           tooltip={"always_visible": True, "placement": "top"}),
            ], md=6, xs=12, className="mb-2"),
            dbc.Col([
                html.Label("Target CAGR (%)", style={"fontSize": UI_FONT_MD}),
                dcc.Slider(id="lev-cagr", min=0, max=50, step=0.5,
                           value=d["lev_cagr"],
                           marks={i: f"{i}%" for i in range(0, 51, 10)},
                           tooltip={"always_visible": True, "placement": "top"}),
            ], md=6, xs=12, className="mb-2"),
        ], className="mb-3"),

        # Outputs: readout, plot, table
        html.Div(id="lev-readout", className="mb-3"),
        dcc.Graph(id="lev-graph", style={"height": "55vh"}),
        html.Div(id="lev-table-wrap",
                 style={"overflowX": "auto", "marginTop": "12px"},
                 children=html.Div(id="lev-table")),
    ], id="leverage-tab-content", className="p-3")
```

- [ ] **Step 2: Smoke-test import**

```bash
cd /scratch/code/bitcoinprojections/btc_web
PYTHONPATH=".:.." ../btc_venv/bin/python3 -c "from layout.leverage import _leverage_tab; t = _leverage_tab(); print('OK,', len(t.children), 'rows')"
```

Expected: `OK, N rows` (no exceptions).

- [ ] **Step 3: Commit**

```bash
git add btc_web/layout/leverage.py
git commit -m "feat(leverage): add layout/leverage.py with input + output scaffold"
```

---

### Task 6: Register the tab in `layout/__init__.py`

Four edits per spec §3 and §7.2: `_PATH_TO_TAB`, `_TAB_TO_GRAPH`, hardcoded 6-tuple → 7-tuple, `_TAB_TO_FIG_FN` lazy-init, and add the hidden `dbc.Tab` to `_serve_layout`.

**Files:**
- Modify: `btc_web/layout/__init__.py:114` (`_PATH_TO_TAB`)
- Modify: `btc_web/layout/__init__.py:121` (`_TAB_TO_GRAPH`)
- Modify: `btc_web/layout/__init__.py:138-143` (`_TAB_TO_FIG_FN` lazy-init)
- Modify: `btc_web/layout/__init__.py:275` (first-render Store tuple)
- Modify: `btc_web/layout/__init__.py` (serve_layout children)

- [ ] **Step 1: Extend `_PATH_TO_TAB`**

Current (around line 114):
```python
_PATH_TO_TAB = {
    "/1": "bubble", "/2": "heatmap", "/3": "dca",
    "/4": "retire", "/5": "supercharge", "/6": "citadel",
    "/7": "stack", "/8": "model_info", "/9": "faq",
    "/faq": "faq", "/mi": "model_info",
}
```

Edit to (note: `/10` placed **before** `/leverage` so the auto-derived reverse map `_TAB_TO_PATH` picks `/leverage` as the canonical path — dict-iteration order means the later entry wins in `{v: k for k, v in _PATH_TO_TAB.items()}`):
```python
_PATH_TO_TAB = {
    "/1": "bubble", "/2": "heatmap", "/3": "dca",
    "/4": "retire", "/5": "supercharge", "/6": "citadel",
    "/7": "stack", "/8": "model_info", "/9": "faq",
    "/10": "leverage", "/leverage": "leverage",  # /leverage last → canonical
    "/faq": "faq", "/mi": "model_info",
}
```

- [ ] **Step 2: Extend `_TAB_TO_GRAPH`**

Current (around line 121):
```python
_TAB_TO_GRAPH = {
    "bubble": "bubble-graph", "heatmap": "heatmap-graph",
    "dca": "dca-graph", "retire": "retire-graph",
    "supercharge": "supercharge-graph", "citadel": "citadel-graph",
}
```

Edit to:
```python
_TAB_TO_GRAPH = {
    "bubble": "bubble-graph", "heatmap": "heatmap-graph",
    "dca": "dca-graph", "retire": "retire-graph",
    "supercharge": "supercharge-graph", "citadel": "citadel-graph",
    "leverage": "lev-graph",
}
```

- [ ] **Step 3: Extend `_TAB_TO_FIG_FN` lazy-init**

Around line 138-143, add leverage wiring. Current:
```python
from utils import (_get_bubble_fig, _get_heatmap_fig, _get_dca_fig,
                   _get_retire_fig, _get_supercharge_fig, _get_citadel_fig)
from tab_defaults import (bubble_defaults, heatmap_defaults, dca_defaults,
                          retire_defaults, supercharge_defaults, citadel_defaults)
_TAB_TO_FIG_FN["bubble"] = (_get_bubble_fig, bubble_defaults)
_TAB_TO_FIG_FN["heatmap"] = (_get_heatmap_fig, heatmap_defaults)
```

After that block (still inside the lazy-init try), add:
```python
from figures.leverage import build_leverage_figure
from tab_defaults import leverage_defaults
_TAB_TO_FIG_FN["leverage"] = (build_leverage_figure, leverage_defaults)
```

- [ ] **Step 4: Extend the first-render Store tuple (line 275)**

Current:
```python
*[dcc.Store(id=f"{tab}-first-render", storage_type="memory",
            data=1 if tab == initial_tab else 0)
  for tab in ("bubble", "heatmap", "dca", "retire", "supercharge", "citadel")],
```

Edit to add `"leverage"`:
```python
*[dcc.Store(id=f"{tab}-first-render", storage_type="memory",
            data=1 if tab == initial_tab else 0)
  for tab in ("bubble", "heatmap", "dca", "retire", "supercharge", "citadel", "leverage")],
```

- [ ] **Step 5: Add hidden `dbc.Tab` to `_serve_layout`**

Find where `dbc.Tabs(...)` is assembled in `_serve_layout` (grep for `dbc.Tab(` or `main-tabs`). Add a new `dbc.Tab` entry at the end, with `label_style={"display": "none"}` to hide the tab header:

```python
# inside dbc.Tabs(children=[...], id="main-tabs", ...)
dbc.Tab(
    _leverage_tab(),
    label="Leverage",
    tab_id="leverage",
    label_style={"display": "none"},  # hidden from tab bar; URL-only access
),
```

Also import at the top of `layout/__init__.py`:
```python
from layout.leverage import _leverage_tab
```

- [ ] **Step 6: Smoke-test that dev server starts**

```bash
cd /scratch/code/bitcoinprojections
lsof -ti :8050 | xargs kill -9 2>/dev/null || true
DEV=1 nohup bash run_web.sh > /tmp/quantoshi_dev.log 2>&1 &
sleep 4
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8050/
```

Expected: `200`.

Also:
```bash
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8050/leverage
```

Expected: `200` (layout loads; tab may not be selected yet — that's next task).

Inspect `/tmp/quantoshi_dev.log` for import errors.

- [ ] **Step 6b: Verify pre-inject path works for initial-tab=leverage**

The spec (§11) requires the leverage figure to be pre-built when the URL path on first load is `/leverage` (`initial_tab == "leverage"` in `_serve_layout`). Because `_TAB_TO_GRAPH["leverage"]` and `_TAB_TO_FIG_FN["leverage"]` are now populated (Steps 2, 3), this should work automatically:

```bash
curl -s http://127.0.0.1:8050/leverage | grep -o 'id="lev-graph"' | head -1
```

Expected: one match (graph component in initial HTML). If no match, the pre-inject path didn't fire — check `_serve_layout` for the branch that reads `_TAB_TO_GRAPH` and `_TAB_TO_FIG_FN`.

- [ ] **Step 7: Commit**

```bash
git add btc_web/layout/__init__.py
git commit -m "feat(leverage): register hidden leverage tab in layout"
```

---

### Task 7: Wire tab routing in `callbacks/routing.py` (3 sites)

**Files:**
- Modify: `btc_web/callbacks/routing.py:113` (canonical `_PATH_TO_TAB`)
- Modify: `btc_web/callbacks/routing.py:40-68` (first-render trigger clientside map)
- Modify: `btc_web/callbacks/routing.py:77-106` (snapshot-restore first-render clientside map)
- Modify: `btc_web/callbacks/routing.py:308-311` (URL routing clientside tab-map)

- [ ] **Step 1: Extend canonical `_PATH_TO_TAB` (line 113)**

Same edit as layout/__init__.py step 1.

- [ ] **Step 2: Extend tab-switch first-render clientside (lines 40-68)**

Current JS (inside the triple-quoted string):
```javascript
function(tab, bub, hm, dca, ret, sc, cp) {
    var NU = window.dash_clientside.no_update;
    var out = [NU, NU, NU, NU, NU, NU];
    var map = {bubble:0, heatmap:1, dca:2, retire:3, supercharge:4, citadel:5};
    ...
}
```

Edit to 7-wide:
```javascript
function(tab, bub, hm, dca, ret, sc, cp, lev) {
    var NU = window.dash_clientside.no_update;
    var out = [NU, NU, NU, NU, NU, NU, NU];
    var map = {bubble:0, heatmap:1, dca:2, retire:3, supercharge:4, citadel:5, leverage:6};
    var idx = map[tab];
    if (idx !== undefined) {
        var cur = [bub, hm, dca, ret, sc, cp, lev][idx];
        if (!cur) out[idx] = 1;
    }
    return out;
}
```

Also add 7th Output/State pair to the callback decorator:
```python
    Output("leverage-first-render", "data", allow_duplicate=True),
    ...
    State("leverage-first-render", "data"),
```

- [ ] **Step 3: Extend snapshot-restore first-render clientside (lines 77-106)**

Same edit pattern:
```javascript
function(state, tab, bub, hm, dca, ret, sc, cp, lev) {
    var NU = window.dash_clientside.no_update;
    var out = [NU, NU, NU, NU, NU, NU, NU];
    if (!state) return out;
    var map = {bubble:0, heatmap:1, dca:2, retire:3, supercharge:4, citadel:5, leverage:6};
    var idx = map[tab];
    if (idx === undefined) return out;
    var cur = [bub, hm, dca, ret, sc, cp, lev][idx] || 0;
    out[idx] = cur + 1;
    return out;
}
```

Add the 7th Output and 7th State.

- [ ] **Step 4: Extend URL-routing clientside tab-map (lines 308-311)**

Current:
```javascript
var map = {"/1":"bubble","/2":"heatmap","/3":"dca",
           "/4":"retire","/5":"supercharge","/6":"citadel",
           "/7":"stack","/8":"model_info","/9":"faq",
           "/faq":"faq","/mi":"model_info"};
```

Edit to add `/leverage` and `/10`:
```javascript
var map = {"/1":"bubble","/2":"heatmap","/3":"dca",
           "/4":"retire","/5":"supercharge","/6":"citadel",
           "/7":"stack","/8":"model_info","/9":"faq",
           "/10":"leverage","/leverage":"leverage",
           "/faq":"faq","/mi":"model_info"};
```

- [ ] **Step 5: Restart dev server and smoke-test**

```bash
lsof -ti :8050 | xargs kill -9 2>/dev/null || true
DEV=1 nohup bash run_web.sh > /tmp/quantoshi_dev.log 2>&1 &
sleep 4
```

Open `http://127.0.0.1:8050/leverage` in a browser. Expected: the Leverage Calculator tab's controls render (no plot yet — that's next task). No console errors in the browser devtools.

- [ ] **Step 6: Commit**

```bash
git add btc_web/callbacks/routing.py
git commit -m "feat(leverage): route /leverage + /10 via 3 tab-map sites"
```

---

### Task 8: Main callback in `callbacks/leverage_cb.py`

**Files:**
- Create: `btc_web/callbacks/leverage_cb.py`
- Modify: `btc_web/callbacks/__init__.py` (add import)

- [ ] **Step 1: Create the callback file**

Create `btc_web/callbacks/leverage_cb.py`. Uses module-level `@callback` decorator (matching the codebase pattern — see `callbacks/user_model.py`, `callbacks/scanner.py`, etc. — registration happens as an import side-effect):

```python
"""Leverage Calculator — main callback.

Wires inputs (date, price, model, floor-q store, rates, H, CAGR sliders)
to outputs (plot, readout, table).

Registered at import time (side-effect) — import from `callbacks/__init__.py`.
"""
from __future__ import annotations

import datetime as _dt

from dash import Input, Output, State, callback, html
from dash.exceptions import PreventUpdate

import _app_ctx
from figures.leverage import (
    build_leverage_figure, floor_price, P_max, implied_cagr, _parse_date,
)


def _readout(buy_date, P_now, sell_date, sell_price, H_yr, c, max_pay, implied_c, model, q):
    """Render the scenario + decision block (spec §6.2)."""
    q_label = f"Q{q*100:g}%"
    is_buy = P_now <= max_pay
    if P_now > 0:
        delta_pct = (max_pay - P_now) / P_now * 100
    else:
        delta_pct = 0
    badge_text = (
        f"✓ BUY — {delta_pct:+.1f}% under your max" if is_buy
        else f"⚠ ABOVE MAX — {-delta_pct:.1f}% over; raise H or lower target to flip"
    )
    badge_class = "alert alert-success" if is_buy else "alert alert-danger"

    implied_str = f"{implied_c*100:.1f}%" if implied_c is not None else "—"
    return html.Div([
        html.Div([
            html.Span(f"Buy:  {buy_date.isoformat()}  @ "),
            html.B(f"${P_now:,.0f}"),
            html.Span("  (current price)"),
        ]),
        html.Div([
            html.Span(f"Sell: {sell_date.isoformat()}  @ "),
            html.B(f"${sell_price:,.0f}"),
            html.Span(f"  ({model} {q_label} floor)"),
        ]),
        html.Div(f"Horizon H = {H_yr:.2f} yr"),
        html.Hr(),
        html.Div([
            html.Span("Your target: "),
            html.B(f"{c*100:.1f}% CAGR"),
        ]),
        html.Div([
            html.Span("Max pay-price today: "),
            html.B(f"${max_pay:,.0f}", style={"fontSize": "1.25em"}),
        ]),
        html.Div(badge_text, className=badge_class, style={"marginTop": "8px"}),
        html.Div(f"Implied CAGR at ${P_now:,.0f}: {implied_str}"),
    ], style={"border": "1px solid #ccc", "borderRadius": "6px", "padding": "12px"})


def _table(buy_date, model, q, r_b, r_l, c, H_slider):
    """Render the 7-row canonical table (spec §6.3)."""
    horizons = [1, 2, 3, 4, 5, 8, 10]
    header = html.Tr([html.Th(h) for h in
                      ["H (yr)", "Sell date", "Sell price",
                       "Max pay @ 0%", f"@ r_l ({r_l*100:.2f}%)",
                       f"@ r_b ({r_b*100:.2f}%)", f"@ your ({c*100:.1f}%)"]])
    rows = []
    for H in horizons:
        sell_d = buy_date + _dt.timedelta(days=int(round(H * 365.25)))
        sp = floor_price(model, q, sell_d)
        row_style = {"backgroundColor": "#ffe"} if abs(H - H_slider) < 0.5 else {}
        rows.append(html.Tr([
            html.Td(H),
            html.Td(sell_d.isoformat()),
            html.Td(f"${sp:,.0f}"),
            html.Td(f"${P_max(sp, H, 0.0):,.0f}"),
            html.Td(f"${P_max(sp, H, r_l):,.0f}"),
            html.Td(f"${P_max(sp, H, r_b):,.0f}"),
            html.Td(f"${P_max(sp, H, c):,.0f}"),
        ], style=row_style))
    return html.Table([html.Thead(header), html.Tbody(rows)],
                      className="table table-sm", style={"width": "100%"})


@callback(
    Output("lev-graph", "figure"),
    Output("lev-readout", "children"),
    Output("lev-table", "children"),
    Input("leverage-first-render", "data"),
    Input("lev-date", "date"),
    Input("lev-price", "value"),
    Input("lev-model", "value"),
    Input("lev-floor-q-store", "data"),
    Input("lev-rb", "value"),
    Input("lev-rl", "value"),
    Input("lev-horizon", "value"),
    Input("lev-cagr", "value"),
    prevent_initial_call=True,
)
def update_leverage(first_render, date_val, price_val, model, q,
                    rb_val, rl_val, H_val, c_val):
    if not first_render:
        raise PreventUpdate

    # Coerce with falsy-zero-safe pattern (CLAUDE.md §"Falsy-zero")
    price = float(price_val) if price_val is not None else 65000.0
    rb    = float(rb_val)    if rb_val    is not None else 13.0
    rl    = float(rl_val)    if rl_val    is not None else 4.5
    H_yr  = float(H_val)     if H_val     is not None else 4.0
    c_pct = float(c_val)     if c_val     is not None else 20.0
    model = str(model or "bub")
    q     = float(q) if q is not None else 0.01

    # Guards (spec §5.5)
    H_yr  = max(H_yr, 0.01)
    price = max(price, 1.0)

    # Parse date
    buy_date = _parse_date(date_val) if date_val else _dt.date.today()

    # Compute core outputs
    c_dec = c_pct / 100.0
    r_b_dec = rb / 100.0
    r_l_dec = rl / 100.0

    sell_date = buy_date + _dt.timedelta(days=int(round(H_yr * 365.25)))
    try:
        sp = floor_price(model, q, sell_date)
    except (KeyError, AttributeError, ValueError) as e:
        return (
            {}, html.Div(f"Model unavailable: {e}", className="alert alert-warning"),
            html.Div()
        )

    max_pay = P_max(sp, H_yr, c_dec)
    implied_c = implied_cagr(sp, price, H_yr)

    p = {
        "lev_price": price, "lev_date": buy_date,
        "lev_model": model, "lev_floor_q": q,
        "lev_rb": rb, "lev_rl": rl,
        "lev_horizon": H_yr, "lev_cagr": c_pct,
        "palette": "default",
    }
    fig = build_leverage_figure(p)
    ro = _readout(buy_date, price, sell_date, sp, H_yr, c_dec, max_pay, implied_c, model, q)
    tbl = _table(buy_date, model, q, r_b_dec, r_l_dec, c_dec, H_yr)
    return fig, ro, tbl
```

- [ ] **Step 2: Register in callbacks/__init__.py**

Add to `btc_web/callbacks/__init__.py` (alongside the other side-effect imports — see `callbacks.user_model`, `callbacks.scanner`, etc.):

```python
import callbacks.leverage_cb  # noqa: F401 — callbacks registered at import
```

- [ ] **Step 3: Smoke-test by restarting the dev server**

```bash
lsof -ti :8050 | xargs kill -9 2>/dev/null || true
DEV=1 nohup bash run_web.sh > /tmp/quantoshi_dev.log 2>&1 &
sleep 4
grep -iE "error|traceback" /tmp/quantoshi_dev.log | head -20
```

Expected: no tracebacks.

Open `http://127.0.0.1:8050/leverage` in a browser. Expected: chart renders, readout shows BUY/ABOVE decision, table has 7 rows.

Move the H slider. Expected: plot, readout, and table all update.

- [ ] **Step 4: Commit**

```bash
git add btc_web/callbacks/leverage_cb.py btc_web/callbacks/__init__.py
git commit -m "feat(leverage): add main callback wiring inputs to figure/readout/table"
```

---

### Task 9: Pill-bar clientside callbacks

**Files:**
- Modify: `btc_web/callbacks/leverage_cb.py` (append)

- [ ] **Step 1: Add pill-click + pill-sync clientside callbacks**

Append to `btc_web/callbacks/leverage_cb.py` (module-level, registered at import — parallel pattern to `callbacks/routing.py::_hm_pill_click`):

```python
import json as _json

from layout.leverage import _LEV_FLOOR_QS, _LEV_PILL_IDS

_lev_pill_ids_json = _json.dumps(_LEV_PILL_IDS)
_lev_floor_qs_json = _json.dumps(_LEV_FLOOR_QS)


# Click: writes selected quantile to store and updates pill outlines.
# Nearest-preset fallback in sync handles old share-links with non-preset q values.
_app_ctx.app.clientside_callback(
    f"""function() {{
        var pill_ids = {_lev_pill_ids_json};
        var qs = {_lev_floor_qs_json};
        var tid = dash_clientside.callback_context.triggered_id;
        if (!tid) throw window.dash_clientside.PreventUpdate;
        var idx = pill_ids.indexOf(tid);
        if (idx < 0) throw window.dash_clientside.PreventUpdate;
        var outlines = pill_ids.map(function(pid) {{ return pid !== tid; }});
        return [qs[idx]].concat(outlines);
    }}""",
    Output("lev-floor-q-store", "data", allow_duplicate=True),
    *[Output(pid, "outline", allow_duplicate=True) for pid in _LEV_PILL_IDS],
    *[Input(pid, "n_clicks") for pid in _LEV_PILL_IDS],
    prevent_initial_call=True,
)


# Sync: when the store changes (e.g. snapshot restore), update outlines.
_app_ctx.app.clientside_callback(
    f"""function(q) {{
        var pill_ids = {_lev_pill_ids_json};
        var qs = {_lev_floor_qs_json};
        if (q === null || q === undefined) {{
            return pill_ids.map(function() {{ return window.dash_clientside.no_update; }});
        }}
        // Find nearest preset (tolerant of old share-links with non-preset q)
        var idx = 0;
        var best_d = Infinity;
        for (var i = 0; i < qs.length; i++) {{
            var d = Math.abs(qs[i] - q);
            if (d < best_d) {{ best_d = d; idx = i; }}
        }}
        return pill_ids.map(function(pid, i) {{ return i !== idx; }});
    }}""",
    [Output(pid, "outline", allow_duplicate=True) for pid in _LEV_PILL_IDS],
    Input("lev-floor-q-store", "data"),
    prevent_initial_call=True,
)
```

`callbacks/__init__.py` does not need a second edit — the pill callbacks register via the same `import callbacks.leverage_cb` from Task 8.

- [ ] **Step 2: Restart dev server and test**

```bash
lsof -ti :8050 | xargs kill -9 2>/dev/null || true
DEV=1 nohup bash run_web.sh > /tmp/quantoshi_dev.log 2>&1 &
sleep 4
```

Browser at `/leverage`. Expected: clicking pill buttons changes the outlined styling (one solid, others outlined) and the plot/readout/table update with the new quantile.

- [ ] **Step 3: Commit**

```bash
git add btc_web/callbacks/leverage_cb.py btc_web/callbacks/__init__.py
git commit -m "feat(leverage): add pill-bar click + sync clientside callbacks"
```

---

### Task 10: Snapshot integration

**Files:**
- Modify: `btc_web/snapshot.py`

- [ ] **Step 1: Add leverage tuples to `_SNAPSHOT_CONTROLS`**

Find the end of `_SNAPSHOT_CONTROLS` list (before the closing bracket). Append:

```python
    # ── Leverage Calculator tab ──
    ("lev-date",          "date"),
    ("lev-price",         "value"),
    ("lev-model",         "value"),
    ("lev-floor-q-store", "data"),
    ("lev-rb",            "value"),
    ("lev-rl",            "value"),
    ("lev-horizon",       "value"),
    ("lev-cagr",          "value"),
```

- [ ] **Step 2: Add leverage to `_TAB_CONTROLS`**

Find `_TAB_CONTROLS` dict. Add:

```python
_TAB_CONTROLS["leverage"] = {
    "lev-date", "lev-price", "lev-model", "lev-floor-q-store",
    "lev-rb", "lev-rl", "lev-horizon", "lev-cagr",
}
```

- [ ] **Step 3: Write snapshot round-trip test**

Append to `btc_web/test_leverage.py`. Note: `_decode_snapshot` returns keys in `f"{cid}:{prop}"` form (see `snapshot.py:585-590`); `_encode_snapshot` accepts keys in the same form:

```python
def test_leverage_snapshot_roundtrip():
    """Encode + decode preserves all leverage controls."""
    from snapshot import _encode_snapshot, _decode_snapshot

    # Keys must be "cid:prop" — matching _SNAPSHOT_CONTROLS tuple layout
    state = {
        "lev-date:date":           "2026-06-01",
        "lev-price:value":         80000.0,
        "lev-model:value":         "pl",
        "lev-floor-q-store:data":  0.05,
        "lev-rb:value":            12.5,
        "lev-rl:value":            4.25,
        "lev-horizon:value":       5.5,
        "lev-cagr:value":          18.0,
    }
    hash_str = _encode_snapshot(state)
    decoded = _decode_snapshot(hash_str)
    for k, v in state.items():
        assert decoded.get(k) == v, f"{k}: expected {v}, got {decoded.get(k)}"
```

- [ ] **Step 4: Run test**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_leverage.py::test_leverage_snapshot_roundtrip -v
```

Expected: PASS.

- [ ] **Step 5: Manual test via browser**

Visit `/leverage`, adjust controls, click 📸 Share → Generate link (Current tab only) → copy → paste into new tab. Expected: all controls restored.

- [ ] **Step 6: Commit**

```bash
git add btc_web/snapshot.py btc_web/test_leverage.py
git commit -m "feat(leverage): add 8 controls to snapshot/share system"
```

---

### Task 11: Mobile CSS

**Files:**
- Modify: `btc_web/assets/style.css`

- [ ] **Step 1: Append mobile rules**

Append to `btc_web/assets/style.css` (near existing mobile rules):

```css
/* ── Leverage Calculator mobile layout (spec §9.2) ── */
@media (max-width: 767px) {
    #lev-graph { height: 55vw !important; min-height: 280px !important; }
    #lev-table-wrap { overflow-x: auto; }
}
```

Column stacking on mobile comes from the `md=6, xs=12` Bootstrap grid classes already set in `layout/leverage.py` (Task 5). The earlier `.row { flex-direction: column; }` override is unnecessary and can fight Bootstrap's own responsive logic — omitted.

- [ ] **Step 2: Visual inspection**

Browser at `/leverage`, open devtools → toggle device toolbar → iPhone/Android viewport. Expected: controls stack vertically, chart scales, table scrolls horizontally if needed.

- [ ] **Step 3: Commit**

```bash
git add btc_web/assets/style.css
git commit -m "feat(leverage): mobile layout rules for /leverage"
```

---

### Task 12: Integration smoke test + regression check

**Files:**
- Modify: `btc_web/test_leverage.py` (integration-y tests)

- [ ] **Step 1: Add edge-case tests**

Append to `btc_web/test_leverage.py`:

```python
def test_leverage_H_zero_guarded():
    """H=0 (snapshot-restored pathological) must not crash implied_cagr."""
    from figures.leverage import implied_cagr
    assert implied_cagr(100_000, 50_000, 0) is None
    assert implied_cagr(100_000, 50_000, -1) is None


def test_leverage_price_zero_guarded():
    """P_now=0 must not crash implied_cagr."""
    from figures.leverage import implied_cagr
    assert implied_cagr(100_000, 0, 4) is None


def test_leverage_s2f_not_in_flagship_dropdown():
    """S2F silently ignores q — must not appear in the leverage dropdown."""
    from layout.leverage import _LEV_FLAGSHIP_MODELS
    assert "s2f" not in _LEV_FLAGSHIP_MODELS


def test_leverage_defaults_cache_alignment():
    """LEVERAGE_DEFAULTS keys cover every callback Input."""
    from tab_defaults import LEVERAGE_DEFAULTS
    expected = {"lev_date", "lev_price", "lev_model", "lev_floor_q",
                "lev_rb", "lev_rl", "lev_horizon", "lev_cagr"}
    assert set(LEVERAGE_DEFAULTS.keys()) == expected


def test_leverage_tab_controls_snapshot_alignment():
    """_TAB_CONTROLS['leverage'] matches the leverage entries in _SNAPSHOT_CONTROLS."""
    from snapshot import _TAB_CONTROLS, _SNAPSHOT_CONTROLS
    lev_snap = {cid for cid, _ in _SNAPSHOT_CONTROLS if cid.startswith("lev-")}
    assert _TAB_CONTROLS["leverage"] == lev_snap
```

- [ ] **Step 2: Run full test suite**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_leverage.py -v
```

Expected: all leverage tests PASS.

Then run the broader suite to catch regressions:

```bash
btc_venv/bin/python3 -m pytest btc_web/ -v --ignore-glob='*_e2e.py'
```

Expected: no regressions — all existing tests still pass.

- [ ] **Step 3: Manual verification in browser**

Restart dev server if not already running:

```bash
lsof -ti :8050 | xargs kill -9 2>/dev/null || true
DEV=1 nohup bash run_web.sh > /tmp/quantoshi_dev.log 2>&1 &
sleep 4
```

Check each:
- Visit `/leverage` → page loads, defaults shown, plot rendered, readout and table populated.
- Visit `/10` → same tab loads.
- Visit `/` → bubble tab; the Leverage tab is NOT visible in the tab bar.
- Change H slider → plot, readout, table update.
- Change target CAGR slider → target curve and readout update.
- Change borrow rate to `7.125` → accepted (not null/None fallback).
- Click each of the 6 floor pills → quantile changes, plot updates.
- Click 📸 Share → Current tab only → Generate link → open in new tab → state restored.
- Switch palette → plot colors change.
- Resize browser to <767px → controls stack below plot; table scrolls.

- [ ] **Step 4: Final commit**

```bash
git add btc_web/test_leverage.py
git commit -m "test(leverage): add edge-case and alignment regression tests"
```

---

## Self-Review

**Spec coverage:**
- §3 Placement → Tasks 5-7 (layout, routing, hidden tab)
- §4 Input controls → Tasks 1 (defaults), 5 (layout), 8 (callback inputs), 9 (pill bar)
- §5 Math → Tasks 2, 3, 8 (callback uses helpers)
- §6 Output → Task 4 (figure), Task 8 (readout + table)
- §7 Files → all tasks cover the enumerated files
- §8 Snapshot → Task 10
- §9 Palette and mobile → colors from Task 4, mobile CSS from Task 11
- §10 Testing → Tasks 1-12 accumulate a full unit-test suite; E2E deferred (spec §10.2)
- §11 Caching → Task 4 (no cache by design)
- §12 Out of scope → nothing in plan exceeds scope
- §13 Acceptance criteria → Task 12 validates each
- §14 Open questions → none

**Placeholder scan:** No TBDs, no "implement later", all code shown in full, all file paths absolute.

**Type consistency:** `floor_price(model_short, q, target_date)`, `P_max(sell_price, H_yr, target_cagr)`, `implied_cagr(sell_price, P_now, H_yr)` are called with the same argument shapes wherever they appear (figure builder, callback, readout, table).

---

## Execution Handoff

Plan saved to `docs/superpowers/plans/2026-04-18-leverage-calculator.md`. Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — batch execution with checkpoints for review.

Which approach?
