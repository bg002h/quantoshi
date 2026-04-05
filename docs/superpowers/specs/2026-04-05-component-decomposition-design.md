# Component Decomposition Overlay — Design

## Goal

Let users visualize the additive components of a chosen price model on Tab 1 (Bubble + QR Overlay). Each model's formula (in log₁₀ space) is split on top-level `+` symbols (NOT inside trigonometric functions), and each component can be plotted individually on the chart. A "Σ Sum of selected" pseudo-trace plots the sum of currently-checked components.

## Scope

**Eligible models** — those with ≥2 additive terms in their log₁₀(price) formula:

| Model | Components | Count |
|---|---|---|
| BM (Bubble Model) | support, bubbles | 2 |
| EF (Empirical Floor) | support, bubbles | 2 |
| LPPL | A, B·log₁₀(t), damped osc | 3 |
| LPPL₂ | A, B·log₁₀(t), damped osc, undamped osc | 4 |
| LPPL₃ | A, B·log₁₀(t), damped osc, 2× undamped osc | 5 |
| LPPL₄ | A, B·log₁₀(t), damped osc, 3× undamped osc | 6 |
| LPPL weighted variants (LPPL_w, LPPL₂_w, LPPL₃_w, LPPL₄_w) | same counts as unweighted bases | 3–6 |
| LPPL no-13 variants (LPPL₄_N13, LPPL₄_W_N13) | same as LPPL₄ | 6 |
| LinPPL | A, B·log₁₀(t), damped calendar osc | 3 |
| HybPPL | A, B·log₁₀(t), damped log osc, undamped calendar osc | 4 |
| HybPPL (ex) | A_sup, B_sup·log₁₀(t), a₀, damped log osc, undamped calendar osc | 5 |

**Excluded:** QR, PL, U₁ (≤2 trivial components — trend line only), S2F, Exp (explicit user exclusion).

## User interactions

1. User navigates to Tab 1. New **Component Decomposition** section appears in the sidebar (after Display Models, before Projection Quantiles).
2. User picks a model family from the `bub-decomp-model` dropdown: `(none)`, `BM`, `EF`, `LPPL (family)`, `LinPPL`, `HybPPL`, `HybPPL (ex)`.
3. When a family is picked, a component checklist `bub-decomp-components` appears with one entry per component + a `"Σ Sum of selected"` pseudo-entry at the bottom.
4. User checks the components they want plotted. Each checked component renders as its own trace on the bubble chart. If `Σ Sum of selected` is checked, an additional trace plots the log-space sum of all other checked components.
5. For `LPPL (family)`, the specific variant is resolved via the existing LPPL config panel (`lppl-n-freqs`, `lppl-weighted`, `lppl-no-13`). **Decomposition requires exactly 1 entry in `lppl-n-freqs`.** If 0 or >1 are checked, the component checklist is replaced by a reminder banner: *"Pick exactly one LPPL variant in the LPPL config panel to decompose."* with a button that opens the LPPL modal.
6. Component decomposition is independent of the `bub-model-show` overlay. The user can display LPPL₃ as a full model overlay AND its component decomposition simultaneously.

## Architecture

### Model API additions (`btc_core.py`)

Each eligible model gains two attributes:

- **Class attribute** `component_names: list[str]` — human-readable names, same order as `components(t)` returns.
- **Instance method** `components(t: ndarray) -> dict[str, ndarray]` — returns each additive term's log₁₀ contribution at points `t`, keyed by name.

**Invariants (verified per-model in tests):**
- LPPL family + LinPPL + HybPPL + HybPPL(ex): `sum(components(t).values()) == _lppl_log10(t)` to within `1e-10`.
- BM / EF (composite models): `sum(components(t).values()) == _composite_log10(t)` to within `1e-10`.

**Every decomposable subclass MUST override `components()` and `component_names`.** A base-class `NotImplementedError` fallback raises if a subclass with extra `_W2`/`_W3`/`_W4` etc. parameters inherits the parent's 3-term decomposition by accident. `TestComponentDecomposition.test_components_sum_to_median` iterates over all 14 registered decomposable models and verifies the invariant — a silent inheritance bug would be caught.

**Example for `LPPLModel`:**

```python
component_names = ["A (constant)", "B·log₁₀(t)", "damped osc (ω_log)"]

def components(self, t):
    t = np.asarray(t, float)
    t_safe = np.maximum(t, 0.1)
    return {
        "A (constant)":          np.full_like(t_safe, self._A),
        "B·log₁₀(t)":            self._B * np.log10(t_safe),
        "damped osc (ω_log)":    self._C * t_safe ** (-self._D) * np.cos(
                                     self._W * np.log(t_safe) + self._PHI),
    }
```

**Example for `HybPPLExcessModel`:**

```python
component_names = ["A_sup", "B_sup·log₁₀(t)", "a₀",
                   "damped log osc", "undamped cal osc"]

def components(self, t):
    t = np.asarray(t, float)
    t_safe = np.maximum(t, 0.1)
    return {
        "A_sup":             np.full_like(t_safe, self._A_sup),
        "B_sup·log₁₀(t)":    self._B_sup * np.log10(t_safe),
        "a₀":                np.full_like(t_safe, self._a0),
        "damped log osc":    self._C1 * t_safe ** (-self._D) * np.cos(
                                  self._W_log * np.log(t_safe) + self._PHI1),
        "undamped cal osc":  self._C2 * np.cos(self._W_cal * t_safe + self._PHI2),
    }
```

**BM / EF composite models** — these are loaded from pkl grids, not analytical formulas. Composite = support + bubble contributions (in log space). The decomposition splits the composite into two pieces:

```python
component_names = ["support", "bubbles"]

def components(self, t):
    # Interpolate pkl grid (years_plot_bm, support_plot_bm for BM;
    # years_plot, support_plot for EF) onto requested t.
    log_support = self._log_support_at(t)            # direct grid interp
    log_composite = self._composite_log10(t)          # pre-existing method
    return {
        "support": log_support,
        "bubbles": log_composite - log_support,       # additive residual in log space
    }
```

**Implementation note — `BubbleModel` does NOT currently store the support grid.** Only `EmpiricalFloorModel` has `self._support_plot`. Add `self._log_support = np.log10(np.maximum(md.support_bm, 1e-10))` and `self._t_support = md.years_bm` (new, if not present) into `BubbleModel.__init__`, sourced from the existing `ModelData` attrs (`support_bm`, `years_bm`). Then `_log_support_at(t)` is a `np.interp` on those arrays. No new pkl keys required — they already ship.

### Decomposable-models registry (`btc_web/_app_ctx.py`)

```python
# Family short_names. "lppl" is resolved at render time via LPPL config.
DECOMP_FAMILIES = {
    "bub":       "BM",
    "ef":        "EF",
    "lppl":      "LPPL (family)",
    "linppl":    "LinPPL",
    "hybppl":    "HybPPL",
    "hybppl_ex": "HybPPL (ex)",
}

# Decomposition trace palette (7 colors, cycles if model has >7 components)
DECOMP_COLORS = {
    "default": ["#E64A19", "#1976D2", "#388E3C", "#7B1FA2", "#F57C00", "#00796B", "#5D4037"],
    "cb-brian": [...],   # colorblind-safe deuteranomaly palette
    "cb-rg":    [...],
    "cb-full":  [...],
}
# Dedicated color for the Σ Sum trace (distinct from any individual
# component color to avoid overlap confusion when a subset is selected)
DECOMP_SUM_COLOR = {
    "default":  "#000000",    # black
    "cb-brian": "#000000",
    "cb-rg":    "#000000",
    "cb-full":  "#000000",
}
```

### Layout — new section (`btc_web/layout/bubble.py`)

```python
_section_card("Component Decomposition",
    _lbl("Model"),
    dcc.Dropdown(id="bub-decomp-model",
                 options=[{"label": "(none)", "value": ""}] +
                         [{"label": label, "value": key}
                          for key, label in _app_ctx.DECOMP_FAMILIES.items()],
                 value="", clearable=False),
    html.Div(id="bub-decomp-body", children=[
        # Dynamic checklist populated by bub-decomp-components callback
        dcc.Checklist(id="bub-decomp-components", options=[], value=[],
                      labelStyle={"display": "block", "fontSize": "11px"},
                      inputStyle=_CB_MARGIN),
    ]),
    html.Div(id="bub-decomp-warning", children=[]),  # LPPL variant-count banner
),
```

Icon: 🧬 in `_SECTION_ICONS`.

### Callbacks (`btc_web/callbacks/charts.py`)

**Populate component checklist options** — two separate callbacks to avoid a race with `apply_snapshot` (see snapshot-race note below).

Callback A — OPTIONS only (never touches `.value`):

```python
@callback(
    Output("bub-decomp-components", "options"),
    Output("bub-decomp-warning",    "children"),
    Output("bub-decomp-body",       "style"),
    Input("bub-decomp-model",  "value"),
    Input("lppl-n-freqs",      "value"),
    Input("lppl-weighted",     "value"),
    Input("lppl-no-13",        "value"),
    prevent_initial_call=False,
)
def update_decomp_options(family, n_freqs, weighted, no_13):
    if not family:
        return [], [], {"display": "none"}
    if family == "lppl":
        if len(n_freqs or []) != 1:
            return [], _decomp_warning_banner(len(n_freqs or [])), {"display": "block"}
        resolved_key = _resolve_lppl_variant(n_freqs, weighted, no_13)
        model = _app_ctx.PRICE_MODELS[resolved_key]
    else:
        model = _app_ctx.PRICE_MODELS[family]
    opts = [{"label": f" {name}", "value": name} for name in model.component_names]
    opts.append({"label": " Σ Sum of selected", "value": "__sum__"})
    return opts, [], {"display": "block"}
```

Callback B — VALUE pruning (fires only when USER changes model dropdown, never on snapshot):

```python
@callback(
    Output("bub-decomp-components", "value", allow_duplicate=True),
    Input("bub-decomp-model",       "value"),
    State("bub-decomp-components", "options"),
    State("bub-decomp-components", "value"),
    prevent_initial_call=True,
)
def prune_decomp_value_on_model_change(family, opts, current):
    # ctx guard — only fire when bub-decomp-model was the trigger
    if ctx.triggered_id != "bub-decomp-model":
        raise dash.exceptions.PreventUpdate
    if not family:
        return []
    valid = {o["value"] for o in (opts or [])}
    return [v for v in (current or []) if v in valid]
```

**Snapshot-race note:** `apply_snapshot` restores `bub-decomp-model` and `bub-decomp-components` simultaneously via its `allow_duplicate=True` outputs. Callback A (options-only) fires when `bub-decomp-model` changes and does NOT touch `.value`, so it cannot clobber the snapshot-restored component selection. Callback B only prunes on user-driven model changes (via `ctx.triggered_id`), not snapshot restores. This mirrors the pattern already used for LPPL config restoration.

**Extend bubble chart callback** (`update_bubble` in `charts.py:247-`): add 3 new Inputs (`bub-decomp-model`, `bub-decomp-components`) + the LPPL resolution State. Pass `decomp_model` + `decomp_components` in the params dict to `build_bubble_figure(m, p)`.

### Figure builder (`btc_web/figures/bubble.py`)

```python
def _add_decomposition_traces(fig, t_grid, model, selected, palette_key):
    if not model or not selected:
        return
    comps = model.components(t_grid)
    names = [s for s in selected if s != "__sum__"]
    colors = DECOMP_COLORS[palette_key]
    for i, name in enumerate(names):
        log_vals = comps[name]
        fig.add_scatter(x=t_grid_to_years(t_grid),
                        y=10 ** log_vals,
                        mode="lines", line={"dash": "dot", "width": 1.5,
                                             "color": colors[i % len(colors)]},
                        name=f"{model.legend_name} | {name}",
                        hovertemplate="...")
    if "__sum__" in selected and names:
        sum_log = sum(comps[n] for n in names)
        fig.add_scatter(x=..., y=10 ** sum_log, mode="lines",
                        line={"dash": "solid", "width": 3,
                               "color": DECOMP_SUM_COLOR[palette_key]},
                        name=f"{model.legend_name} | Σ ({len(names)} components)")
```

### Snapshot / share link (`btc_web/snapshot.py`)

Append two entries to `_SNAPSHOT_CONTROLS` (appended to END — backward compat):

```python
("bub-decomp-model",      "value"),
("bub-decomp-components", "value"),
```

Add `"bub-decomp-model"` and `"bub-decomp-components"` to `_TAB_CONTROLS["bubble"]` set in `callbacks/routing.py`. LPPL-family decomposition requires `lppl-n-freqs` / `lppl-weighted` / `lppl-no-13` — all three are ALREADY in `_TAB_CONTROLS["bubble"]`, so no additional entries needed.

**Bitmask encoding:** `bub-decomp-components` is NOT added to `_CHECKLIST_OPTIONS` because its option set is dynamic per model. Store as plain list[str] (JSON-native). Decoder already handles both `list` and `int` (bitmask) variants via `isinstance(val, int)` check in `_apply_snapshot_value`.

Snapshot scheme stays at `q3:` (backward-compatible addition — older links without these trailing entries decode to `None` → defaults).

### Trace styling details

- Individual components: `dash="dot"`, `width=1.5`, color from `DECOMP_COLORS[palette_key]`.
- Sum trace: `dash="solid"`, `width=3`, one color position past the last individual component.
- Hover tooltip: `"{component_name}: $X at year Y"` (USD on hover; log value not shown).
- `showlegend=True` (respects the global Show Legend toggle via layout `showlegend`).

### Auto-Y interaction

When decomposition is active, `auto_bubble_yrange` callback MUST include component y-values when computing the fit range. Otherwise a constant like `A = -1.15` (horizontal line at $0.07) would force the Y-range to include `10^(-1.15)` which may be far below the model bands.

**Implementation:** `auto_bubble_yrange` gains two new **Inputs** (not States — otherwise the callback doesn't refire when decomposition changes):

- `Input("bub-decomp-model", "value")`
- `Input("bub-decomp-components", "value")`

When decomposition is active AND Auto-Y is on, extend the computed `p_lo`/`p_hi` with component min/max values over `[t_lo, t_hi]` before log10+floor/ceil rounding.

Edge case: the constant `A = -1.15` is always far below bands. The auto-fit will expand the low end to cover it, making the chart less readable. This is the expected behavior — if the user enables `A (constant)` they want to see it.

## Testing

### Unit tests (`btc_web/test_web.py` — new class `TestComponentDecomposition`)

1. **`test_components_sum_to_median`** — for each decomposable model, assert `sum(components(t).values()) ≈ _lppl_log10(t)` to 1e-10 at t ∈ {1, 5, 10, 16, 30, 50}.
2. **`test_component_names_match_components`** — each model's `len(component_names) == len(components(t))`.
3. **`test_decompose_families_registered`** — `_app_ctx.DECOMP_FAMILIES` contains exactly 6 entries: bub, ef, lppl, linppl, hybppl, hybppl_ex.
4. **`test_lppl_config_resolves`** — `_resolve_lppl_variant([3], [], [])` → "lp3"; `_resolve_lppl_variant([3], ["weighted"], [])` → "lp3_w"; `_resolve_lppl_variant([4], [], ["no13"])` → "lp4_n13".
5. **`test_decomp_component_count`** — each model has expected component count (LPPL₃=5, HybPPL(ex)=5, etc.)
6. **`test_decomp_warning_shown`** — `update_decomp_options("lppl", [], [], [])` returns a warning banner; `update_decomp_options("lppl", [3], [], [])` returns options.
7. **`test_decomp_snapshot_roundtrip`** — encode a state with decomp active, decode, verify dropdown + component list recovered.

### E2E (manual Playwright check)

Navigate to /1, pick "LPPL (family)" in decomposition dropdown, verify warning appears (since default n_freqs=[3] which is 1 — actually no warning). Change n_freqs to [1,2,3,4] → verify warning renders. Change back to [3] → verify checklist has 5 components + Σ Sum entry. Check "A (constant)" and "damped osc" → verify 2 new dotted traces. Check "Σ Sum of selected" → verify 1 additional solid trace appears.

## Files touched (summary)

| File | Change |
|---|---|
| `btc_core.py` | +14 class attributes (`component_names`) + 14 `components()` methods (~150 lines) |
| `btc_web/_app_ctx.py` | `DECOMP_FAMILIES` dict + `DECOMP_COLORS` palette dict (~30 lines) |
| `btc_web/layout/bubble.py` | New `_section_card("Component Decomposition", …)` (~25 lines) |
| `btc_web/layout/common.py` | Add `"Component Decomposition": "🧬"` to `_SECTION_ICONS` |
| `btc_web/callbacks/charts.py` | New `update_decomp_options` callback (~60 lines); extend `update_bubble` and `auto_bubble_yrange` Inputs/States |
| `btc_web/figures/bubble.py` | `_add_decomposition_traces()` helper (~40 lines); call site in `build_bubble_figure` |
| `btc_web/snapshot.py` | +2 entries in `_SNAPSHOT_CONTROLS`; +2 in `_TAB_CONTROLS["bubble"]` |
| `btc_web/test_web.py` | `TestComponentDecomposition` class (~100 lines) |

**Estimate:** ~450 lines across 8 files. No new files required.

## Out of scope

- Decomposing S2F, Exp, QR, PL, U₁.
- Per-quantile shifts on component traces (always at median).
- Per-component hover annotations beyond the standard Plotly tooltip.
- Saving/loading custom component combinations as presets.
- Animating component buildup ("add them one at a time").
