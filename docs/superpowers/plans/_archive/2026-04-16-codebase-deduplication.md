# Codebase Deduplication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce server callbacks (~25 conversions to clientside JS), deduplicate figure builder boilerplate (~400 lines), and clean up engine/utility duplication (~240 lines).

**Architecture:** Three independent stages executed sequentially. Stage 1 converts trivial server callbacks to clientside JavaScript. Stage 2 extracts repeated figure-building patterns into shared helpers. Stage 3 consolidates engine tax bracket inflation, Markov return application, and utility module overlap.

**Tech Stack:** Plotly Dash 4.0.0, Python 3.14.3, JavaScript (clientside callbacks)

**Spec:** `docs/superpowers/specs/2026-04-16-codebase-deduplication-design.md`

**Test command:** `btc_venv/bin/python3 -m pytest btc_web/ -v --ignore-glob='*_e2e.py'`

---

## Execution Status (2026-04-16, master branch)

**Completed** (committed): Tasks 1, 2 (prior session — `6446232`), then Tasks 3–8 (Stage 1), 9–12 (Stage 2), 13, 14, 15, 17-partial (Stage 3). Commits `28cad79`, `a7f71d9`, `48eda2f`, `4388d48`, `a23084c`, `5bc69dd`, `180913d`, `cfd2415`, `d537283`. All changes reviewed by `dash-callback-reviewer` (Stage 1) + `feature-dev:code-reviewer` (Stages 2 & 3). Tests: 36 failed / 1271 passed — all 36 are pre-existing baseline failures (unchanged). Dev server smoke test: all 10 tab routes HTTP 200.

**Skipped** (defensible reasons, pending future sessions):

- **Task 15b** (`_apply_returns` consolidation in `citadel_step.py`): the plan's taxable → TD → TF order would change RNG consumption for the `use_markov=False` TD/TF lognormal branch, which currently interleaves `td_res[i]` / `tf_res[i]` per loop iteration. Changing order would alter MC outputs for saved seeds. No test would catch it but users with reproducibility expectations would.
- **Task 15c** (`_drain_accounts` helper): the plan template assumes a flat list-of-fields drain, but the three call sites diverge genuinely — `_compute_rmd` iterates `td_investments` in reverse, `_pay_tax_amount` applies LTCG gross-up mid-drain, `_execute_draw` is a dispatch-by-index (not a drain at all). Forcing a generic helper would either require more complexity than it removes, or change ordering/arithmetic.
- **Task 16** (Redis/cache pass-through removal): 6-file import reshuffle for ~3 lines of indirection. Net value negative; the `cache.redis_available` wrapper is documented and keeps `cache.py` a self-contained API surface.
- **Task 17 Steps 3–5** (inline `_startup_heatmap_defaults` / `_log_cache_stats` / move `_fetch_sparkline_svg`): single-caller functions are fine where they are; inlining adds churn without value.

---

## Stage 1: Clientside Callback Conversions

### Task 1: Tier 1 Trivial Returns -- nav, mc_payment, snapshot_cb

**Files:**
- Modify: `btc_web/callbacks/nav.py:145-154`
- Modify: `btc_web/callbacks/mc_payment.py:148-166,232-240`
- Modify: `btc_web/callbacks/snapshot_cb.py:226-232,268-274`

- [x] **Step 1: Convert `toggle_share_modal` to clientside**

In `btc_web/callbacks/nav.py`, replace lines 145-154:

```python
# WAS: server-side callback
# @callback(
#     Output("share-modal", "is_open"),
#     Input("share-btn", "n_clicks"),
#     Input("share-btn-mobile", "n_clicks"),
#     Input("share-modal-close", "n_clicks"),
#     State("share-modal", "is_open"),
#     prevent_initial_call=True,
# )
# def toggle_share_modal(n1, n1m, n2, is_open):
#     return not is_open

_app_ctx.app.clientside_callback(
    "function(n1, n1m, n2, is_open) { return !is_open; }",
    Output("share-modal", "is_open"),
    Input("share-btn", "n_clicks"),
    Input("share-btn-mobile", "n_clicks"),
    Input("share-modal-close", "n_clicks"),
    State("share-modal", "is_open"),
    prevent_initial_call=True,
)
```

- [x] **Step 2: Convert `_quant_proceed`, `_quant_cancel`, `_mc_payment_cancel` to clientside**

In `btc_web/callbacks/mc_payment.py`, replace `_quant_proceed` (lines 148-158):

```python
_app_ctx.app.clientside_callback(
    "function(n, t) { return [false, (t || 0) + 1]; }",
    Output("mc-quant-modal", "is_open", allow_duplicate=True),
    Output("mc-pay-trigger", "data", allow_duplicate=True),
    Input("mc-quant-proceed", "n_clicks"),
    State("mc-pay-trigger", "data"),
    prevent_initial_call=True,
)
```

Replace `_quant_cancel` (lines 160-166):

```python
_app_ctx.app.clientside_callback(
    "function(n) { return false; }",
    Output("mc-quant-modal", "is_open", allow_duplicate=True),
    Input("mc-quant-cancel", "n_clicks"),
    prevent_initial_call=True,
)
```

Replace `_mc_payment_cancel` (lines 232-240):

```python
_app_ctx.app.clientside_callback(
    "function(n) { return [false, true, '']; }",
    Output("mc-pay-modal", "is_open", allow_duplicate=True),
    Output("mc-pay-poll", "disabled", allow_duplicate=True),
    Output("mc-pay-status", "children", allow_duplicate=True),
    Input("mc-pay-cancel", "n_clicks"),
    prevent_initial_call=True,
)
```

- [x] **Step 3: Convert `restore_my_lots` and `clear_history` to clientside**

In `btc_web/callbacks/snapshot_cb.py`, replace `restore_my_lots` (lines 226-232):

```python
_app_ctx.app.clientside_callback(
    "function(n) { return null; }",
    Output("snapshot-lots", "data", allow_duplicate=True),
    Input("restore-lots-btn", "n_clicks"),
    prevent_initial_call=True,
)
```

Replace `clear_history` (lines 268-274):

```python
_app_ctx.app.clientside_callback(
    "function(n) { return []; }",
    Output("link-history", "data", allow_duplicate=True),
    Input("clear-history-btn", "n_clicks"),
    prevent_initial_call=True,
)
```

- [x] **Step 4: Run tests**

Run: `btc_venv/bin/python3 -m pytest btc_web/ -v --ignore-glob='*_e2e.py'`
Expected: All tests pass.

- [x] **Step 5: Commit**

```bash
git add btc_web/callbacks/nav.py btc_web/callbacks/mc_payment.py btc_web/callbacks/snapshot_cb.py
git commit -m "refactor(callbacks): convert 6 trivial-return callbacks to clientside JS"
```

---

### Task 2: Tier 1 Trivial Returns -- mc_controls, user_model

**Files:**
- Modify: `btc_web/callbacks/mc_controls.py:140-146`
- Modify: `btc_web/callbacks/user_model.py:100-114`

- [x] **Step 1: Convert `_close_freq_modal` to clientside**

In `btc_web/callbacks/mc_controls.py`, replace lines 140-146:

```python
_app_ctx.app.clientside_callback(
    "function(n) { return false; }",
    Output("freq-warning-modal", "is_open", allow_duplicate=True),
    Input("freq-warning-ok", "n_clicks"),
    prevent_initial_call=True,
)
```

- [x] **Step 2: Convert `delete_user_model` to clientside**

In `btc_web/callbacks/user_model.py`, replace lines 100-114:

```python
_app_ctx.app.clientside_callback(
    """function(n) {
        return [null, null, null, null, null,
                '\\u2014', '\\u2014', '\\u2014', '\\u2014'];
    }""",
    Output("user-model-store", "data", allow_duplicate=True),
    Output("um-p1-year", "data", allow_duplicate=True),
    Output("um-p1-price", "data", allow_duplicate=True),
    Output("um-p2-year", "data", allow_duplicate=True),
    Output("um-p2-price", "data", allow_duplicate=True),
    Output("um-p1-year-display", "children", allow_duplicate=True),
    Output("um-p1-price-display", "children", allow_duplicate=True),
    Output("um-p2-year-display", "children", allow_duplicate=True),
    Output("um-p2-price-display", "children", allow_duplicate=True),
    Input("um-delete-btn", "n_clicks"),
    prevent_initial_call=True,
)
```

- [x] **Step 3: Run tests**

Run: `btc_venv/bin/python3 -m pytest btc_web/ -v --ignore-glob='*_e2e.py'`
Expected: All tests pass.

- [x] **Step 4: Commit**

```bash
git add btc_web/callbacks/mc_controls.py btc_web/callbacks/user_model.py
git commit -m "refactor(callbacks): convert _close_freq_modal + delete_user_model to clientside"
```

---

### Task 3: Tier 2 -- set_p1/set_p2 to clientside

**Files:**
- Modify: `btc_web/callbacks/user_model.py:62-93`

- [x] **Step 1: Convert `set_p1` and `set_p2` to clientside**

In `btc_web/callbacks/user_model.py`, replace `set_p1` (lines 62-76) and `set_p2` (lines 79-93):

```python
_FMT_PRICE_JS = """
function _fmtPrice(p) {
    if (p >= 100) return p.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2});
    if (p >= 1) return p.toFixed(4);
    return p.toFixed(6);
}
"""

_app_ctx.app.clientside_callback(
    _FMT_PRICE_JS + """
    function(n, pt) {
        if (!pt) return [window.dash_clientside.no_update,
                         window.dash_clientside.no_update,
                         window.dash_clientside.no_update,
                         window.dash_clientside.no_update,
                         window.dash_clientside.no_update];
        var yr = pt.year;
        var pr = Math.round(pt.price * 1e6) / 1e6;
        return [yr, pr, String(yr), _fmtPrice(pr), {display: 'none'}];
    }
    """,
    Output("um-p1-year", "data", allow_duplicate=True),
    Output("um-p1-price", "data", allow_duplicate=True),
    Output("um-p1-year-display", "children"),
    Output("um-p1-price-display", "children"),
    Output("um-ctx-menu", "style", allow_duplicate=True),
    Input("um-ctx-p1", "n_clicks"),
    State("um-clicked-point", "data"),
    prevent_initial_call=True,
)

_app_ctx.app.clientside_callback(
    _FMT_PRICE_JS + """
    function(n, pt) {
        if (!pt) return [window.dash_clientside.no_update,
                         window.dash_clientside.no_update,
                         window.dash_clientside.no_update,
                         window.dash_clientside.no_update,
                         window.dash_clientside.no_update];
        var yr = pt.year;
        var pr = Math.round(pt.price * 1e6) / 1e6;
        return [yr, pr, String(yr), _fmtPrice(pr), {display: 'none'}];
    }
    """,
    Output("um-p2-year", "data", allow_duplicate=True),
    Output("um-p2-price", "data", allow_duplicate=True),
    Output("um-p2-year-display", "children"),
    Output("um-p2-price-display", "children"),
    Output("um-ctx-menu", "style", allow_duplicate=True),
    Input("um-ctx-p2", "n_clicks"),
    State("um-clicked-point", "data"),
    prevent_initial_call=True,
)
```

Note: `_fmt_price_display` (line 52) does NOT include a `$` prefix -- it returns just the formatted number. The `$` is rendered separately in the layout. The JS must also return without `$`.

- [x] **Step 2: Remove `_fmt_price_display` and `_HIDDEN` if now unused**

Check if `_fmt_price_display` (line 52) and `_HIDDEN` (line 11) are still used by other callbacks in the file (e.g., `on_data_click`, `auto_draw`). Only remove if truly unused.

- [x] **Step 3: Run tests**

Run: `btc_venv/bin/python3 -m pytest btc_web/ -v --ignore-glob='*_e2e.py'`
Expected: All tests pass.

- [x] **Step 4: Commit**

```bash
git add btc_web/callbacks/user_model.py
git commit -m "refactor(callbacks): convert set_p1/set_p2 to clientside JS"
```

---

### Task 4: Tier 2 -- Citadel scenario pills to clientside

**Files:**
- Modify: `btc_web/callbacks/citadel_scenarios.py:19-38`

- [x] **Step 1: Convert `_register_pill_cb` to clientside**

In `btc_web/callbacks/citadel_scenarios.py`, replace lines 19-38 with a clientside factory:

```python
import json as _json

def _register_pill_cb(group_name, keys):
    """Register a clientside callback that updates a store and toggles pill outlines."""
    outputs = [Output(f"cp-scenario-{group_name}", "data")]
    outputs += [Output(f"cp-pill-{k}", "outline") for k in keys]
    inputs = [Input(f"cp-pill-{k}", "n_clicks") for k in keys]

    keys_json = _json.dumps(keys)
    _app_ctx.app.clientside_callback(
        f"""function() {{
            var keys = {keys_json};
            var tid = dash_clientside.callback_context.triggered_id;
            if (!tid) return Array(1 + keys.length).fill(window.dash_clientside.no_update);
            var selected = tid.replace('cp-pill-', '');
            var outlines = keys.map(function(k) {{ return k !== selected; }});
            return [selected].concat(outlines);
        }}""",
        *outputs,
        *inputs,
        prevent_initial_call=True,
    )


_register_pill_cb("wealth", list(WEALTH_LEVELS.keys()))
_register_pill_cb("regime", list(MACRO_REGIMES.keys()))
_register_pill_cb("rules", list(RULE_SETS.keys()))
```

Add `import _app_ctx` at the top of the file (it is not currently imported there).

- [x] **Step 2: Run tests**

Run: `btc_venv/bin/python3 -m pytest btc_web/ -v --ignore-glob='*_e2e.py'`
Expected: All tests pass.

- [x] **Step 3: Commit**

```bash
git add btc_web/callbacks/citadel_scenarios.py
git commit -m "refactor(callbacks): convert citadel scenario pills to clientside"
```

---

### Task 5: Tier 2 -- apply_hm_palette to clientside

**Files:**
- Modify: `btc_web/callbacks/snapshot_cb.py:390-403`

- [x] **Step 1: Convert `apply_hm_palette` to clientside**

In `btc_web/callbacks/snapshot_cb.py`, replace lines 390-403. The `HM_PRESET_PALETTES` dict must be embedded as a JS literal since it's in `__skip_export__`:

```python
import json as _json
from colors import HM_PRESET_PALETTES

_HM_PALETTES_JS = _json.dumps({
    k: list(v) for k, v in HM_PRESET_PALETTES.items()
})

_app_ctx.app.clientside_callback(
    f"""function(preset_name) {{
        var palettes = {_HM_PALETTES_JS};
        if (!preset_name || !palettes[preset_name]) {{
            return [window.dash_clientside.no_update,
                    window.dash_clientside.no_update,
                    window.dash_clientside.no_update,
                    window.dash_clientside.no_update];
        }}
        return palettes[preset_name];
    }}""",
    Output("hm-c-lo", "value", allow_duplicate=True),
    Output("hm-c-mid1", "value", allow_duplicate=True),
    Output("hm-c-mid2", "value", allow_duplicate=True),
    Output("hm-c-hi", "value", allow_duplicate=True),
    Input("hm-palette", "value"),
    prevent_initial_call=True,
)
```

Remove the old `_HM_PALETTES = HM_PRESET_PALETTES` alias (line 390) and the server callback.

- [x] **Step 2: Run tests**

Run: `btc_venv/bin/python3 -m pytest btc_web/ -v --ignore-glob='*_e2e.py'`
Expected: All tests pass.

- [x] **Step 3: Commit**

```bash
git add btc_web/callbacks/snapshot_cb.py
git commit -m "refactor(callbacks): convert apply_hm_palette to clientside with embedded presets"
```

---

### Task 6: Tier 2 -- _restore_mc (5x) to clientside

**Files:**
- Modify: `btc_web/callbacks/mc_controls.py:248-271`

- [x] **Step 1: Convert the 5 `_restore_mc` callbacks to clientside**

In `btc_web/callbacks/mc_controls.py`, replace the loop at lines 248-271:

```python
from mc_cache import MC_DEFAULT_YEARS, MC_DEFAULT_START_YR, MC_DEFAULT_ENTRY_Q, MC_BINS, MC_SIMS

_RESTORE_MC_JS = f"""function(n_clicks, mc_cached) {{
    var nu = window.dash_clientside.no_update;
    if (!mc_cached || !mc_cached.path_key) return [nu, nu, nu, nu, nu, nu];
    var pk = mc_cached.path_key;
    return [
        pk.mc_years    !== undefined ? pk.mc_years    : {MC_DEFAULT_YEARS},
        pk.mc_start_yr !== undefined ? pk.mc_start_yr : {MC_DEFAULT_START_YR},
        pk.mc_entry_q  !== undefined ? pk.mc_entry_q  : {MC_DEFAULT_ENTRY_Q},
        pk.mc_bins     !== undefined ? pk.mc_bins     : {MC_BINS},
        pk.mc_sims     !== undefined ? pk.mc_sims     : {MC_SIMS},
        pk.mc_window   !== undefined ? pk.mc_window   : null
    ];
}}"""

for _rpfx in ("hm", "dca", "ret", "sc", "cp"):
    _app_ctx.app.clientside_callback(
        _RESTORE_MC_JS,
        Output(f"{_rpfx}-mc-years", "value", allow_duplicate=True),
        Output(f"{_rpfx}-mc-start-yr", "value", allow_duplicate=True),
        Output(f"{_rpfx}-mc-entry-q", "value", allow_duplicate=True),
        Output(f"{_rpfx}-mc-bins", "value", allow_duplicate=True),
        Output(f"{_rpfx}-mc-sims", "value", allow_duplicate=True),
        Output(f"{_rpfx}-mc-window", "value", allow_duplicate=True),
        Input(f"{_rpfx}-mc-restore-btn", "n_clicks"),
        State(f"{_rpfx}-mc-results", "data"),
        prevent_initial_call=True,
    )
```

- [x] **Step 2: Run tests**

Run: `btc_venv/bin/python3 -m pytest btc_web/ -v --ignore-glob='*_e2e.py'`
Expected: All tests pass.

- [x] **Step 3: Commit**

```bash
git add btc_web/callbacks/mc_controls.py
git commit -m "refactor(callbacks): convert 5x _restore_mc to clientside JS"
```

---

### Task 7: Tier 2 -- _toggle_freq_unlock (3x) to clientside

**Files:**
- Modify: `btc_web/callbacks/mc_controls.py:126-138`

- [x] **Step 1: Convert the 3 `_toggle_freq_unlock` callbacks to clientside**

In `btc_web/callbacks/mc_controls.py`, replace lines 126-138:

```python
for _fp in ("dca", "ret", "sc"):
    _app_ctx.app.clientside_callback(
        """function(unlock, cur_freq) {
            if (unlock && unlock.length) {
                return [false, cur_freq, true];
            }
            return [true, 'Monthly', false];
        }""",
        Output(f"{_fp}-freq", "disabled"),
        Output(f"{_fp}-freq", "value", allow_duplicate=True),
        Output("freq-warning-modal", "is_open", allow_duplicate=True),
        Input(f"{_fp}-freq-unlock", "value"),
        State(f"{_fp}-freq", "value"),
        prevent_initial_call=True,
    )
```

- [x] **Step 2: Run tests**

Run: `btc_venv/bin/python3 -m pytest btc_web/ -v --ignore-glob='*_e2e.py'`
Expected: All tests pass.

- [x] **Step 3: Commit**

```bash
git add btc_web/callbacks/mc_controls.py
git commit -m "refactor(callbacks): convert 3x _toggle_freq_unlock to clientside"
```

---

### Task 8: Tier 3 -- Heatmap pill callbacks to clientside

**Files:**
- Modify: `btc_web/callbacks/routing.py:784-836`

- [x] **Step 1: Convert `_hm_deep_link`, `_hm_pill_click`, `_hm_pill_sync` to clientside**

In `btc_web/callbacks/routing.py`, replace lines 784-836. All three need `_HM_PILL_MODELS` and `_HM_PILL_IDS` embedded at registration time:

```python
import json as _json

# _HM_PILL_MODELS, _HM_PILL_IDS, _HM_LEGACY_MODEL_FALLBACK already defined above

_pill_models_json = _json.dumps(_HM_PILL_MODELS)
_pill_ids_json = _json.dumps(_HM_PILL_IDS)
_legacy_json = _json.dumps(_HM_LEGACY_MODEL_FALLBACK)

# _hm_deep_link
_app_ctx.app.clientside_callback(
    f"""function(pathname) {{
        var nu = window.dash_clientside.no_update;
        var models = {_pill_models_json};
        if (!pathname) return nu;
        pathname = pathname.replace(/\\/+$/, '') || '/';
        if (!pathname.startsWith('/2.')) return nu;
        try {{
            var n = parseInt(pathname.substring(3));
            if (n >= 1 && n <= models.length) return models[n - 1];
        }} catch(e) {{}}
        return nu;
    }}""",
    Output("hm-active-model", "data", allow_duplicate=True),
    Input("url", "pathname"),
    prevent_initial_call=True,
)

# _hm_pill_click
_app_ctx.app.clientside_callback(
    f"""function() {{
        var pill_ids = {_pill_ids_json};
        var tid = dash_clientside.callback_context.triggered_id;
        if (!tid) throw window.dash_clientside.PreventUpdate;
        var model_key = tid.replace('hm-pill-', '');
        var outlines = pill_ids.map(function(pid) {{ return pid !== tid; }});
        return [model_key].concat(outlines);
    }}""",
    Output("hm-active-model", "data", allow_duplicate=True),
    *[Output(pid, "outline") for pid in _HM_PILL_IDS],
    *[Input(pid, "n_clicks") for pid in _HM_PILL_IDS],
    prevent_initial_call=True,
)

# _hm_pill_sync
_app_ctx.app.clientside_callback(
    f"""function(model_key) {{
        var pill_ids = {_pill_ids_json};
        var legacy = {_legacy_json};
        model_key = model_key || 'bub';
        var models = {_pill_models_json};
        if (models.indexOf(model_key) === -1) {{
            model_key = legacy[model_key] || 'bub';
        }}
        var active_id = 'hm-pill-' + model_key;
        return pill_ids.map(function(pid) {{ return pid !== active_id; }});
    }}""",
    *[Output(pid, "outline", allow_duplicate=True) for pid in _HM_PILL_IDS],
    Input("hm-active-model", "data"),
    prevent_initial_call=True,
)
```

Remove the `_norm` helper usage (the JS handles trailing slash normalization inline).

- [x] **Step 2: Run tests**

Run: `btc_venv/bin/python3 -m pytest btc_web/ -v --ignore-glob='*_e2e.py'`
Expected: All tests pass.

- [x] **Step 3: Commit**

```bash
git add btc_web/callbacks/routing.py
git commit -m "refactor(callbacks): convert heatmap pill callbacks to clientside JS"
```

---

## Stage 2: Figure Builder Deduplication

### Task 9: Add shared figure helpers to common.py

**Files:**
- Modify: `btc_web/figures/common.py`

- [x] **Step 1: Add `_parse_quantiles` helper**

Add to `btc_web/figures/common.py` (after existing imports, near line 40):

```python
def _parse_quantiles(p: dict, key: str = "selected_qs") -> list[float]:
    """Parse and sort quantile list from params dict.

    Basic form only -- callers needing model.fits filtering, reverse sort,
    or custom defaults apply post-processing themselves.
    """
    return sorted(float(q) for q in (p.get(key) or []))
```

- [x] **Step 2: Add `_format_final_value` helper**

```python
def _format_final_value(vals, prices, disp_mode: str, show_usd_parens: bool = True):
    """Format final simulation value for legend label.

    Returns (y_vals, final_label).
    show_usd_parens=False used by overlay traces where USD would duplicate.
    """
    from btc_core import fmt_price
    if disp_mode == "usd":
        y_vals = vals * prices
        return y_vals, fmt_price(float(y_vals[-1]))
    y_vals = vals
    btc_final = float(vals[-1])
    if show_usd_parens:
        usd_final = fmt_price(float(btc_final * float(prices[-1])))
        return y_vals, f"{btc_final:.4f} BTC  ({usd_final})"
    return y_vals, f"{btc_final:.4f} BTC"
```

- [x] **Step 3: Add `_quantile_trace` helper**

```python
def _quantile_trace(ts, y_vals, q: float, color: str, label: str,
                    width: float = None, shape: str = "linear",
                    **kw) -> go.Scatter:
    """Build a quantile-colored Scatter trace with standard opacity."""
    from colors import TRACE_WIDTH
    _shade = quantile_shade(color, q)
    return go.Scatter(
        x=list(ts), y=list(y_vals), mode="lines", name=label,
        line=dict(color=_shade, width=width or TRACE_WIDTH, shape=shape),
        opacity=quantile_opacity(q),
        **kw,
    )
```

- [x] **Step 4: Add `_empty_state_annotation` helper**

```python
def _empty_state_annotation(layout: dict) -> None:
    """Set the 'No models selected' fallback annotation on layout."""
    from colors import FALLBACK_MODEL_GRAY
    layout["annotations"] = [dict(
        text="No models selected \u2014 check Display Models",
        xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16, color=FALLBACK_MODEL_GRAY),
    )]
```

- [x] **Step 5: Add `_today_line_shapes` helper**

```python
def _today_line_shapes(t_today: float, y_lo, y_hi, color: str,
                       glow: bool = True, yref: str = "y") -> list[dict]:
    """Build today-line shape(s). bubble uses glow+yref='y'; residuals uses no glow+yref='paper'."""
    shapes = []
    if glow:
        shapes.append(dict(
            type="line", x0=t_today, x1=t_today, y0=y_lo, y1=y_hi,
            line=dict(color=color, width=_TODAY_GLOW_WIDTH),
            opacity=_TODAY_GLOW_OPACITY, yref=yref,
        ))
    shapes.append(dict(
        type="line", x0=t_today, x1=t_today, y0=y_lo, y1=y_hi,
        line=dict(color=color, dash="dash", width=_TODAY_LINE_WIDTH),
        opacity=_TODAY_LINE_OPACITY, yref=yref,
    ))
    return shapes
```

- [x] **Step 6: Add `_apply_final_steps` helper**

Lower-level finalization called by both `_finalize_chart()` and bubble/residuals builders:

```python
def _apply_final_steps(fig: go.Figure, p: dict, tab: str,
                       recovery: bool = False, hover_fmt: str | None = None,
                       show_qr: bool = True, show_mc: bool = False,
                       wm_pos: str = "bottom-right") -> None:
    """Apply typography, date hover, config annotation, and watermark to a figure.

    This is the lower-level helper. _finalize_chart() wraps this with
    legend positioning, MC premium, and tuple return.
    """
    _apply_sans_typography(fig.layout)
    fmt = hover_fmt or (_HOVER_FMT_BTC if p.get("disp_mode") == "btc" else _HOVER_FMT_USD)
    _add_date_hover(fig, _app_ctx.M.genesis, fmt=fmt, recovery=recovery)
    _apply_config_annotation(fig, p, tab, show_qr=show_qr, show_mc=show_mc)
    _apply_watermark(fig, pos=wm_pos)
```

Then refactor `_finalize_chart()` to call `_apply_final_steps` internally instead of inlining the same steps.

- [x] **Step 7: Run tests (no callers yet, just verify import works)**

Run: `btc_venv/bin/python3 -c "from btc_web.figures.common import _parse_quantiles, _format_final_value, _quantile_trace, _empty_state_annotation, _today_line_shapes, _apply_final_steps; print('OK')"`
Expected: `OK`

- [x] **Step 8: Commit**

```bash
git add btc_web/figures/common.py
git commit -m "refactor(figures): add 5 shared helpers to common.py"
```

---

### Task 10: Migrate figure modules to use new helpers

**Files:**
- Modify: `btc_web/figures/dca.py`
- Modify: `btc_web/figures/retire.py`
- Modify: `btc_web/figures/supercharge.py`
- Modify: `btc_web/figures/residuals.py`
- Modify: `btc_web/figures/bubble.py`
- Modify: `btc_web/figures/citadel.py`

- [x] **Step 1: Replace quantile parsing in all modules**

In each file, replace `sorted([float(q) for q in (p.get("selected_qs") or [])])` with `_parse_quantiles(p)`. Add `_parse_quantiles` to each file's import from `figures.common`.

Specific replacements:
- `bubble.py:77` -- `sel_qs = _parse_quantiles(p)`
- `dca.py:172` -- `sel_qs_raw = _parse_quantiles(p)`
- `dca.py:183` -- `sel_qs = _parse_quantiles(p)`
- `retire.py:41` -- `sel_qs_raw = _parse_quantiles(p)`
- `retire.py:52` -- `sel_qs = _parse_quantiles(p)`
- `supercharge.py:58` -- `sel_qs_raw = _parse_quantiles(p)`
- `citadel.py:145` -- keep custom default: `sel_qs = _parse_quantiles(p) or [0.01, 0.10, 0.25]`
- `common.py:442` -- `sel_qs = _parse_quantiles(p)`

Do NOT change: `supercharge.py:73` (has extra `model.fits` filter), `heatmap.py:239` (filters + reverses).

- [x] **Step 2: Replace empty-state annotations**

In each file, replace the `layout["annotations"] = [dict(text="No models selected...")]` block with `_empty_state_annotation(layout)`:
- `dca.py:339-344`
- `retire.py:169-174`
- `supercharge.py:482-487`
- `supercharge.py:599-604`
- `residuals.py:211-216`

- [x] **Step 3: Replace today-line shapes in bubble.py**

In `bubble.py`, replace lines 378-394 with:

```python
    shapes = []
    if p.get("show_today"):
        td = today_t(m.genesis)
        today_color = palette.get("today_line", _TODAY_LINE_COLOR)
        if t_lo <= td <= t_hi:
            shapes.extend(_today_line_shapes(td, y_lo, y_hi, today_color, glow=True, yref="y"))
```

- [x] **Step 4: Replace today-line shape in residuals.py**

In `residuals.py`, replace the today-line portion (keeping the zero-reference line) at lines 162-170:

```python
    if p.get("show_today"):
        td = today_t(m.genesis)
        if t_lo <= td <= t_hi:
            shapes.extend(_today_line_shapes(td, 0, 1, today_color, glow=False, yref="paper"))
```

- [x] **Step 5: Run tests**

Run: `btc_venv/bin/python3 -m pytest btc_web/ -v --ignore-glob='*_e2e.py'`
Expected: All tests pass.

- [x] **Step 6: Commit**

```bash
git add btc_web/figures/
git commit -m "refactor(figures): migrate 6 modules to shared helpers"
```

---

### Task 11: Unify HybPPL/EPPL model slots in layout

**Files:**
- Modify: `btc_web/layout/common.py:679-872`

- [x] **Step 1: Create `_two_freq_model_slot` replacing both implementations**

In `btc_web/layout/common.py`, replace `_hybppl_model_slot` (lines 679-759) and `_eppl_model_slot` (lines 792-872) with a unified function. Keep the original functions as thin wrappers:

```python
def _two_freq_model_slot(family: str, slot: str, damping_label: str = "damped") -> html.Div:
    """Build config slot (A or B) for a two-frequency model family.

    family: ID prefix, e.g. "hybppl" or "eppl"
    slot: "a" or "b"
    damping_label: "damped" or "entropy damped"
    """
    pfx = f"{family}-cfg-{slot}"
    children = []

    # Optional Model B enable checklist
    if slot == "b":
        children.append(dcc.Checklist(
            id=f"{pfx}-enable",
            options=[{"label": f" Enable Model B", "value": "yes"}],
            value=[], inputStyle=_CB_MARGIN,
        ))

    # nlog, ncal radio items
    children.append(_lbl("N log-freq"))
    children.append(dcc.RadioItems(
        id=f"{pfx}-nlog", options=[0, 1, 2], value=1,
        inline=True, inputStyle=_CB_MARGIN,
    ))
    children.append(_lbl("N cal-freq"))
    children.append(dcc.RadioItems(
        id=f"{pfx}-ncal", options=[0, 1, 2], value=1,
        inline=True, inputStyle=_CB_MARGIN,
    ))

    # Damping wrappers for log1d, log2d, cal1d, cal2d
    for freq_key in ("log1d", "log2d", "cal1d", "cal2d"):
        wrap_id = f"{pfx}-{freq_key}-wrap"
        radio_id = f"{pfx}-{freq_key}"
        children.append(html.Div(id=wrap_id, children=[
            dcc.RadioItems(
                id=radio_id,
                options=[
                    {"label": f" {damping_label}", "value": "damped"},
                    {"label": " undamped", "value": "undamped"},
                ],
                value="damped", inline=True, inputStyle=_CB_MARGIN,
            ),
        ]))

    # Status + info link footer
    children.append(html.Span(id=f"{pfx}-status", style={"fontSize": "0.85rem"}))

    return html.Div(children=children)


def _hybppl_model_slot(slot: str) -> html.Div:
    return _two_freq_model_slot("hybppl", slot, damping_label="damped")


def _eppl_model_slot(slot: str) -> html.Div:
    return _two_freq_model_slot("eppl", slot, damping_label="entropy damped")
```

IMPORTANT: Verify the exact component IDs and structure by reading the originals. The code above is a template -- the implementer MUST compare against the original `_hybppl_model_slot` to ensure every `id`, every default `value`, and every child element matches exactly.

- [x] **Step 2: Unify `_global_hybppl_modal` and `_global_eppl_modal`**

Similarly, replace the two modal wrapper functions with a shared `_global_two_freq_modal(family, title)`:

```python
def _global_two_freq_modal(family: str, title: str) -> dbc.Modal:
    """Build the global config modal for a two-frequency model family."""
    return dbc.Modal(
        id=f"{family}-cfg-modal",
        children=[
            dbc.ModalHeader(dbc.ModalTitle(title)),
            dbc.ModalBody([
                html.H6("Model A (primary)"),
                _two_freq_model_slot(family, "a"),
                html.Hr(),
                html.H6("Model B (secondary)"),
                _two_freq_model_slot(family, "b"),
            ]),
            dbc.ModalFooter(
                dbc.Button("Close", id=f"{family}-cfg-close", className="ms-auto"),
            ),
        ],
        size="lg",
        is_open=False,
    )
```

Then replace:
- `_global_hybppl_modal()` with `_global_two_freq_modal("hybppl", "Hybrid PPL Configuration")`
- `_global_eppl_modal()` with `_global_two_freq_modal("eppl", "\U0001FAE0 Entropy PPL Configuration")`

IMPORTANT: The implementer MUST read the originals to verify the modal structure (header text, body layout, footer buttons, size, etc.) matches exactly.

- [x] **Step 3: Run tests**

Run: `btc_venv/bin/python3 -m pytest btc_web/ -v --ignore-glob='*_e2e.py'`
Expected: All tests pass.

- [x] **Step 3: Commit**

```bash
git add btc_web/layout/common.py
git commit -m "refactor(layout): unify HybPPL/EPPL model slots into _two_freq_model_slot"
```

---

### Task 12: Extract _use_lots_checklist + fix citadel_tax imports

**Files:**
- Modify: `btc_web/layout/common.py`
- Modify: `btc_web/layout/bubble.py:189-191`
- Modify: `btc_web/layout/heatmap.py:82-84`
- Modify: `btc_web/layout/citadel.py:31-32`
- Modify: `btc_web/layout/citadel_tax.py:41-43`

- [x] **Step 1: Add `_use_lots_checklist` to layout/common.py**

```python
def _use_lots_checklist(prefix: str) -> dcc.Checklist:
    """Standard 'Use Stack Tracker lots' checklist."""
    return dcc.Checklist(
        id=f"{prefix}-use-lots",
        options=[{"label": " Use Stack Tracker lots", "value": "yes"}],
        value=[],
        inputStyle=_CB_MARGIN,
    )
```

- [x] **Step 2: Replace inline checklists in bubble, heatmap, citadel**

- `layout/bubble.py:189-191` -- replace with `_use_lots_checklist("bub")`
- `layout/heatmap.py:82-84` -- replace with `_use_lots_checklist("hm")`
- `layout/citadel.py:31-32` -- replace with `_use_lots_checklist("cp")`

The `layout/common.py:951-953` version already uses `f"{prefix}-use-lots"` so it just needs to call the new helper too.

- [x] **Step 3: Fix citadel_tax.py duplicate imports**

In `btc_web/layout/citadel_tax.py`, remove the local `_lbl` definition (line 43) and `_HINT` style (line 41). Import from common:

```python
from .common import _lbl, _STYLE_HINT
```

Then replace all uses of `_HINT` with `_STYLE_HINT` in the file.

- [x] **Step 4: Run tests**

Run: `btc_venv/bin/python3 -m pytest btc_web/ -v --ignore-glob='*_e2e.py'`
Expected: All tests pass.

- [x] **Step 5: Commit**

```bash
git add btc_web/layout/
git commit -m "refactor(layout): extract _use_lots_checklist + fix citadel_tax imports"
```

---

## Stage 3: Engine and Utility Cleanup

### Task 13: Extract tax bracket inflation helper

**Files:**
- Create: helper function in `btc_web/engines/citadel_waterfall.py` (top of file)
- Modify: `btc_web/engines/citadel_waterfall.py:167-201,258-281`
- Modify: `btc_web/engines/citadel_tax_integration.py:136-149`

- [x] **Step 1: Add `_inflate_tax_context` helper**

Add to `btc_web/engines/citadel_waterfall.py` near the top (before `_score_sources`):

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class TaxContext:
    """Pre-inflated tax brackets for a single simulation year."""
    ord_brackets: list
    std_ded: float
    ltcg_brackets: list
    niit_threshold: float
    sim_year: int
    infl: float


def _inflate_tax_context(state, config) -> TaxContext:
    """Compute inflated brackets for current simulation period."""
    from .tax import _inflate_brackets
    from .tax_data import (FEDERAL_BRACKETS_TCJA, FEDERAL_BRACKETS_SUNSET,
                           LTCG_BRACKETS, NIIT_THRESHOLD,
                           STANDARD_DEDUCTION_TCJA, STANDARD_DEDUCTION_SUNSET)
    ppy = FREQ_PPY.get(config.freq, 12)
    sim_year = config.start_yr + int(state.period / ppy)
    yrs = max(sim_year - 2025, 0)
    infl = config.inflation / 100

    if config.tcja_sunset:
        ord_brackets = _inflate_brackets(FEDERAL_BRACKETS_SUNSET[config.filing_status], yrs, infl)
        std_ded = STANDARD_DEDUCTION_SUNSET[config.filing_status] * (1 + infl) ** yrs
    else:
        ord_brackets = _inflate_brackets(FEDERAL_BRACKETS_TCJA[config.filing_status], yrs, infl)
        std_ded = STANDARD_DEDUCTION_TCJA[config.filing_status] * (1 + infl) ** yrs

    ltcg_brackets = _inflate_brackets(LTCG_BRACKETS[config.filing_status], yrs, infl)
    niit_threshold = NIIT_THRESHOLD[config.filing_status]
    return TaxContext(ord_brackets, std_ded, ltcg_brackets, niit_threshold, sim_year, infl)
```

- [x] **Step 2: Refactor `_score_sources` to use `TaxContext`**

Replace the bracket setup code in `_score_sources` (lines 167-201) with:

```python
    tc = _inflate_tax_context(state, config)
```

Then replace all references: `_ord_brackets` -> `tc.ord_brackets`, `_std_ded` -> `tc.std_ded`, `_ltcg_brackets` -> `tc.ltcg_brackets`, `_niit_threshold` -> `tc.niit_threshold`, `sim_year` -> `tc.sim_year`.

- [x] **Step 3: Refactor `_max_draw_before_boundary` to use `TaxContext`**

Replace the bracket setup code in `_max_draw_before_boundary` (lines 258-281) similarly.

- [x] **Step 4: Refactor `_pay_tax_amount` to use `TaxContext`**

In `btc_web/engines/citadel_tax_integration.py`, replace lines 136-144 with:

```python
    if tax_remaining > 0:
        from .citadel_waterfall import _inflate_tax_context, TaxContext
        from .tax import apply_progressive_brackets
        tc = _inflate_tax_context(state, config)
        _agi = tax_result.get("agi", 0) if tax_result else 0
        _tax_at_agi = apply_progressive_brackets(_agi, tc.ord_brackets)
        _tax_at_agi_plus = apply_progressive_brackets(_agi + 1, tc.ord_brackets)
        _marginal_fed = _tax_at_agi_plus - _tax_at_agi
```

Note: `_pay_tax_amount` uses `sim_year` directly (not via `config.start_yr + period/ppy`). Verify the `state` object has a `period` attribute here, or pass `sim_year` differently.

- [x] **Step 5: Run tests**

Run: `btc_venv/bin/python3 -m pytest btc_web/ -v --ignore-glob='*_e2e.py'`
Expected: All tests pass (especially the 92 tax-specific tests).

- [x] **Step 6: Commit**

```bash
git add btc_web/engines/citadel_waterfall.py btc_web/engines/citadel_tax_integration.py
git commit -m "refactor(engines): extract _inflate_tax_context for bracket dedup"
```

---

### Task 14: Consolidate regime initialization loop

**Files:**
- Modify: `btc_web/engines/citadel_sim.py:39-56`

- [x] **Step 1: Replace 15 regime assignments with a loop**

In `btc_web/engines/citadel_sim.py`, replace lines 39-56:

```python
    # Seed all wrapper regimes from config
    _REGIME_ATTRS = ("equity_regime", "bond_regime",
                     "res_short_regime", "res_med_regime", "res_long_regime")
    for prefix in ("", "td_", "tf_"):
        for attr in _REGIME_ATTRS:
            setattr(state, f"{prefix}{attr}", getattr(config, f"initial_{attr}"))
```

- [x] **Step 2: Run tests**

Run: `btc_venv/bin/python3 -m pytest btc_web/ -v --ignore-glob='*_e2e.py'`
Expected: All tests pass.

- [x] **Step 3: Commit**

```bash
git add btc_web/engines/citadel_sim.py
git commit -m "refactor(engines): loop-based regime initialization"
```

---

### Task 15: Deduplicate sell_lots branches

**Files:**
- Modify: `btc_web/engines/tax_lots.py:78-118`

- [x] **Step 1: Extract `_make_lot_gain` helper**

In `btc_web/engines/tax_lots.py`, add before `sell_lots`:

```python
def _make_lot_gain(btc_used: float, lot: TaxLot, sale_price: float,
                   sale_dt, holding_days: int, is_long_term: bool) -> LotGain:
    """Construct a LotGain from a lot sale (whole or partial)."""
    proceeds = btc_used * sale_price
    cost = btc_used * lot.cost_basis
    return LotGain(
        btc=btc_used,
        cost_basis=lot.cost_basis,
        sale_price=sale_price,
        proceeds=proceeds,
        cost=cost,
        gain=proceeds - cost,
        is_long_term=is_long_term,
        holding_days=holding_days,
    )
```

- [x] **Step 2: Simplify sell_lots branches**

Replace lines 78-118 with:

```python
        if lot.btc <= btc_remaining:
            btc_used = lot.btc
            btc_remaining -= btc_used
            btc_sold += btc_used
            gains.append(_make_lot_gain(btc_used, lot, sale_price, sale_dt, holding_days, is_long_term))
        else:
            btc_used = btc_remaining
            btc_sold += btc_used
            btc_remaining = 0.0
            gains.append(_make_lot_gain(btc_used, lot, sale_price, sale_dt, holding_days, is_long_term))
            remaining_lots.append(TaxLot(
                date=lot.date,
                btc=lot.btc - btc_used,
                cost_basis=lot.cost_basis,
                source=lot.source,
            ))
```

- [x] **Step 3: Run tests**

Run: `btc_venv/bin/python3 -m pytest btc_web/ -v --ignore-glob='*_e2e.py'`
Expected: All tests pass.

- [x] **Step 4: Commit**

```bash
git add btc_web/engines/tax_lots.py
git commit -m "refactor(engines): extract _make_lot_gain to deduplicate sell_lots"
```

---

### Task 15b: Consolidate Markov/lognormal return application

**Files:**
- Modify: `btc_web/engines/citadel_step.py:40-274`

- [ ] **Step 1: Rename `_lognormal_return` to accept percentages directly**

In `btc_web/engines/citadel_step.py`, modify the function signature at line 40:

```python
def _lognormal_return_pct(annual_rate_pct: float, annual_vol_pct: float, ppy: int,
                          deterministic: bool = False,
                          rng: np.random.Generator | None = None) -> float:
    """One-period return using lognormal model. Accepts rates as percentages (e.g. 7.0 for 7%)."""
    annual_rate = annual_rate_pct / 100
    annual_vol = annual_vol_pct / 100
    if deterministic:
        return (1 + annual_rate) ** (1.0 / ppy) - 1.0
    if annual_vol <= 0:
        return (1 + max(annual_rate, -0.99)) ** (1.0 / ppy) - 1.0
    annual_rate = max(annual_rate, -0.99)
    sigma_ln = math.sqrt(math.log(1 + (annual_vol / (1 + annual_rate)) ** 2))
    mu_ln = math.log(1 + annual_rate) - sigma_ln ** 2 / 2
    period_mu = mu_ln / ppy
    period_sigma = sigma_ln / math.sqrt(ppy)
    return math.exp(rng.normal(period_mu, period_sigma)) - 1.0
```

- [ ] **Step 2: Extract `_apply_returns` helper**

Add to `btc_web/engines/citadel_step.py`. The `_markov_return` signature is `(matrix, regime, rng) -> (ret, new_regime)` with 3 args. Asset matrices are looked up by key from `config.asset_matrices`:

```python
_RES_KEYS = ["tres_short", "tres_med", "tres_long"]
_INV_KEYS = ["equity", "bond"]
_RES_REGIME_SUFFIXES = ["res_short_regime", "res_med_regime", "res_long_regime"]
_INV_REGIME_SUFFIXES = ["equity_regime", "bond_regime"]


def _apply_returns(new, prefix: str, config, am: dict | None,
                   use_markov: bool, deterministic: bool, rng, ppy: int):
    """Apply asset returns (reserves + investments) for one wrapper.

    prefix: "" for taxable, "td_" for tax-deferred, "tf_" for tax-free.
    am: config.asset_matrices (may be None).
    MUST be called in order: taxable, TD, TF to preserve RNG sequence.
    """
    res_arr = getattr(new, f"{prefix}reserves" if prefix else "reserves")
    inv_arr = getattr(new, f"{prefix}investments" if prefix else "investments")

    # Reserves
    for i, mkey in enumerate(_RES_KEYS):
        if i >= len(res_arr):
            continue
        rattr = f"{prefix}{_RES_REGIME_SUFFIXES[i]}"
        if use_markov and am and mkey in am:
            ret, nr = _markov_return(am[mkey], getattr(new, rattr), rng)
            setattr(new, rattr, nr)
            res_arr[i] *= (1 + ret)
        else:
            rb = config.reserve_bins[i]
            r = _lognormal_return_pct(rb["rate"], rb["volatility"], ppy,
                                      deterministic=deterministic, rng=rng)
            res_arr[i] *= (1 + r)

    # Investments
    for i, mkey in enumerate(_INV_KEYS):
        if i >= len(inv_arr):
            continue
        rattr = f"{prefix}{_INV_REGIME_SUFFIXES[i]}"
        if use_markov and am and mkey in am:
            ret, nr = _markov_return(am[mkey], getattr(new, rattr), rng)
            setattr(new, rattr, nr)
            inv_arr[i] *= (1 + ret)
        else:
            ib = config.invest_bins[i]
            r = _lognormal_return_pct(ib["return_rate"], ib["volatility"], ppy,
                                      deterministic=deterministic, rng=rng)
            inv_arr[i] *= (1 + r)
```

- [ ] **Step 3: Replace the return blocks with 3 calls**

Keep lines 140-146 (use_markov flag + taxable cash growth) and lines 189-193 (TD/TF cash growth). Replace lines 148-187 (taxable Markov+lognormal) and lines 195-274 (TD/TF Markov+lognormal) with:

```python
    # Taxable wrapper returns (line 148 replacement)
    am = config.asset_matrices
    _apply_returns(new, "", config, am, use_markov, deterministic, rng, ppy)

    # TD/TF wrapper returns (line 195 replacement, inside `if config.tax_enabled:`)
    if config.tax_enabled:
        cash_growth = (1 + config.cash_rate / 100) ** (1.0 / ppy)
        new.td_cash *= cash_growth
        new.tf_cash *= cash_growth
        _apply_returns(new, "td_", config, am, use_markov, deterministic, rng, ppy)
        _apply_returns(new, "tf_", config, am, use_markov, deterministic, rng, ppy)
```

IMPORTANT: Lines 145-146 (taxable cash) and 191-193 (TD/TF cash) are NOT handled by `_apply_returns` -- they must remain outside the helper calls.

- [ ] **Step 4: Update all `_lognormal_return` call sites to `_lognormal_return_pct`**

Search for remaining `_lognormal_return(` calls in citadel_step.py and remove the `/100` divisions at call sites since the function now handles them internally.

- [ ] **Step 5: Add deterministic regression test**

Add to the test file covering citadel simulation:

```python
def test_return_consolidation_deterministic():
    """Verify refactored _apply_returns produces identical results to original."""
    # Run a short deterministic simulation and compare key outputs
    # (total_usd, btc_stack, depletion) against known-good values
    # captured before the refactor.
```

The implementer should capture baseline values before the refactor and assert equality after.

- [ ] **Step 6: Run tests**

Run: `btc_venv/bin/python3 -m pytest btc_web/ -v --ignore-glob='*_e2e.py'`
Expected: All tests pass.

- [ ] **Step 7: Commit**

```bash
git add btc_web/engines/citadel_step.py
git commit -m "refactor(engines): consolidate Markov/lognormal return blocks into _apply_returns"
```

---

### Task 15c: Extract account drain helper

**Files:**
- Modify: `btc_web/engines/citadel_tax_integration.py:52-78,153-164`
- Modify: `btc_web/engines/citadel_waterfall.py:342-404`

- [ ] **Step 1: Add `_drain_accounts` helper**

Add to `btc_web/engines/citadel_transactions.py` (or `citadel_tax_integration.py`):

```python
def _drain_accounts(state, field_names: list[str], remaining: float) -> float:
    """Drain accounts in priority order. Returns leftover amount.

    field_names: list of state attribute names to drain, in order.
    Mutates state balances in place.
    """
    for field in field_names:
        if remaining <= 0:
            break
        bal = getattr(state, field)
        if isinstance(bal, (list, np.ndarray)):
            # Array field (e.g., reserves, investments) -- drain each element
            for i in range(len(bal)):
                if remaining <= 0:
                    break
                draw = min(bal[i], remaining)
                bal[i] -= draw
                remaining -= draw
        else:
            draw = min(bal, remaining)
            setattr(state, field, bal - draw)
            remaining -= draw
    return remaining
```

IMPORTANT: The implementer MUST read the original drain patterns to verify whether they drain scalar fields or array fields, and whether the ordering matches. The original code in `_compute_rmd` drains TD wrapper: `td_cash -> td_reserves[i] -> td_investments[i] -> td_btc`. The helper must handle both scalar (`td_cash`) and array (`td_reserves`) fields.

- [ ] **Step 2: Refactor `_compute_rmd` to use `_drain_accounts`**

In `citadel_tax_integration.py`, replace the drain loop in `_compute_rmd` (lines 52-78) with a call to `_drain_accounts`.

- [ ] **Step 3: Refactor `_pay_tax_amount` to use `_drain_accounts`**

In `citadel_tax_integration.py`, replace the drain loop in `_pay_tax_amount` (lines 153-164).

- [ ] **Step 4: Refactor `_execute_draw` to use `_drain_accounts`**

In `citadel_waterfall.py`, replace the 3 wrapper branches in `_execute_draw` (lines 342-404).

- [ ] **Step 5: Run tests**

Run: `btc_venv/bin/python3 -m pytest btc_web/ -v --ignore-glob='*_e2e.py'`
Expected: All tests pass (especially 92 tax tests).

- [ ] **Step 6: Commit**

```bash
git add btc_web/engines/citadel_tax_integration.py btc_web/engines/citadel_waterfall.py btc_web/engines/citadel_transactions.py
git commit -m "refactor(engines): extract _drain_accounts helper"
```

---

### Task 16: Redis/cache/utils consolidation

**Files:**
- Modify: `btc_web/cache.py:22-24,84-86`
- Modify: `btc_web/utils.py` (multiple sites)
- Modify: `btc_web/app.py`
- Modify: `btc_web/api.py`
- Modify: `btc_web/generate_citadel_cache.py`

- [ ] **Step 1: Remove `redis_available` wrapper and aliases from cache.py**

In `btc_web/cache.py`, remove lines 22-24 (the copied aliases) and lines 84-86 (the pass-through wrapper). Update internal uses of `_HAS_REDIS` and `_REDIS` in cache.py to use `_app_ctx._HAS_REDIS` and `_app_ctx._REDIS` directly. Update `_MODEL_FP` references to `_app_ctx._MODEL_FP`.

- [ ] **Step 2: Update all callers to import from _app_ctx**

In each of these files, replace `from cache import redis_available` (or `from .cache import redis_available`) with `from _app_ctx import redis_available` (or `from ._app_ctx import redis_available`):

- `btc_web/utils.py` -- lines 70-71, 89-90, 276
- `btc_web/app.py` -- lines 415-416, 449-450
- `btc_web/generate_citadel_cache.py` -- lines 56, 59
- `btc_web/api.py` -- line 490

Also update any `from cache import _REDIS, _HAS_REDIS` to `from _app_ctx import _REDIS, _HAS_REDIS`.

- [ ] **Step 3: Refactor inline serialization in _make_cached_builder**

In `btc_web/utils.py`, update `_make_cached_builder` (around line 91-98) to use `_serialize_result` and `_deserialize_result` instead of inline JSON construction.

- [ ] **Step 4: Run tests**

Run: `btc_venv/bin/python3 -m pytest btc_web/ -v --ignore-glob='*_e2e.py'`
Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add btc_web/cache.py btc_web/utils.py btc_web/app.py btc_web/api.py btc_web/generate_citadel_cache.py
git commit -m "refactor(cache): remove pass-through wrappers, use _app_ctx directly"
```

---

### Task 17: Move misplaced utilities

**Files:**
- Modify: `btc_web/_app_ctx.py:102-123`
- Create: `btc_web/engines/sc_math.py`
- Modify: `btc_web/figures/dca.py`
- Modify: `btc_web/callbacks/sc_loan.py`
- Modify: `btc_web/utils.py`
- Modify: `btc_web/app.py`
- Modify: `btc_web/callbacks/ticker.py`

- [x] **Step 1: Move `_compute_sc_loan` to engines/sc_math.py**

Create `btc_web/engines/sc_math.py`:

```python
"""Stack-celerator loan math (pure functions, no app dependencies)."""


def compute_sc_loan(principal, amount, r, term_periods, loan_type):
    """Cap principal so payment <= DCA amount, compute loan payment.

    Returns (principal, pmt, capped).
    """
    capped = False
    if r > 0:
        if loan_type == "amortizing":
            max_principal = amount * (1 - (1 + r) ** (-term_periods)) / r
        else:
            max_principal = amount / r
        if principal > max_principal:
            principal = max_principal
            capped = True
    if loan_type == "amortizing":
        pmt = principal * r / (1 - (1 + r) ** (-term_periods)) if r > 0 else principal / term_periods
    else:
        pmt = principal * r
    return principal, pmt, capped
```

- [x] **Step 2: Update callers**

In `btc_web/_app_ctx.py`, remove `_compute_sc_loan` (lines 102-123). Add a re-export for backward compat if needed, but check first:

- `btc_web/figures/dca.py` -- update import to `from engines.sc_math import compute_sc_loan`
- `btc_web/callbacks/sc_loan.py` -- update import to `from engines.sc_math import compute_sc_loan`

Search test files for `_compute_sc_loan` imports and update those too.

- [ ] **Step 3: Inline `_startup_heatmap_defaults` into app.py**

In `btc_web/app.py`, find the call to `_startup_heatmap_defaults()` (around line 356). Replace the call with the inlined body:

```python
# Inline: was _startup_heatmap_defaults()
_hm_price = _fetch_btc_price()
if _hm_price is not None:
    _hm_pct = _find_lot_percentile(today_t(_app_ctx.M.genesis), _hm_price, _app_ctx.M.qr_fits)
    if _hm_pct is not None:
        _hm_entry_q = round(_hm_pct * 100, 1)
    else:
        _hm_entry_q = 50.0
else:
    _hm_entry_q = 50.0
```

Ensure the necessary imports (`_fetch_btc_price`, `_find_lot_percentile`, `today_t`) are available in app.py. Remove `_startup_heatmap_defaults` from utils.py.

- [ ] **Step 4: Inline `_log_cache_stats` into app.py**

Find the call to `_log_cache_stats()` in app.py (around line 468). Replace with the inlined body:

```python
for name, cache in _ALL_CACHES.items():
    info = cache.cache_info()
    total = info.hits + info.misses
    rate = f"{info.hits/total:.0%}" if total else "n/a"
    logger.info("cache/%s: hits=%d misses=%d size=%d/%d rate=%s",
                name, info.hits, info.misses, info.currsize, info.maxsize, rate)
```

Remove `_log_cache_stats` from utils.py.

- [ ] **Step 5: Move `_fetch_sparkline_svg` to callbacks/ticker.py**

Move the function body from `btc_web/utils.py` to `btc_web/callbacks/ticker.py` (its only caller). Update the call site in ticker.py to use the local function.

- [x] **Step 6: Run tests**

Run: `btc_venv/bin/python3 -m pytest btc_web/ -v --ignore-glob='*_e2e.py'`
Expected: All tests pass.

- [x] **Step 7: Commit**

```bash
git add btc_web/engines/sc_math.py btc_web/_app_ctx.py btc_web/figures/dca.py btc_web/callbacks/sc_loan.py btc_web/utils.py btc_web/app.py btc_web/callbacks/ticker.py
git commit -m "refactor(utils): move misplaced utilities to proper locations"
```

---

### Task 18: Final full test run

- [x] **Step 1: Run the full test suite**

Run: `btc_venv/bin/python3 -m pytest btc_web/ -v --ignore-glob='*_e2e.py'`
Expected: All 1224+ tests pass.

- [x] **Step 2: Syntax-check the web app**

Run: `cd btc_web && PYTHONPATH=".:.." ../btc_venv/bin/python3 -c "import layout, figures, callbacks, cache, engines.adapter, engines.citadel, engines.tax, engines.tax_lots, engines.tax_data, data.asset_matrices; print('OK')"`
Expected: `OK`

- [x] **Step 3: Start dev server and verify basic functionality**

Run: `DEV=1 bash run_web.sh`
Visit http://localhost:8050, click through tabs 1-6, verify:
- Share modal opens/closes (clientside)
- Heatmap pill switching works
- Citadel scenario pills toggle
- Figures render correctly
- No console errors
