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

**Invariant (verified in tests):** `sum(components(t).values()) == _lppl_log10(t)` (or equivalent for BM/EF) to within `1e-10`.

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

**BM / EF composite models** — components read from pkl grid:

```python
component_names = ["support", "bubbles"]

def components(self, t):
    # Interpolate pkl grid to requested t
    log_support = self._log_support_at(t)
    log_total   = np.log10(self._price_at_median(t))  # median of composite
    return {
        "support": log_support,
        "bubbles": log_total - log_support,  # composite minus support in log space
    }
```

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

**Populate component checklist** (server-side, fires on model dropdown + LPPL config change):

```python
@callback(
    Output("bub-decomp-components", "options"),
    Output("bub-decomp-components", "value"),
    Output("bub-decomp-warning",    "children"),
    Output("bub-decomp-body",       "style"),
    Input("bub-decomp-model",  "value"),
    Input("lppl-n-freqs",      "value"),
    Input("lppl-weighted",     "value"),
    Input("lppl-no-13",        "value"),
    State("bub-decomp-components", "value"),
    prevent_initial_call=False,
)
def update_decomp_options(family, n_freqs, weighted, no_13, current_checked):
    if not family:
        return [], [], [], {"display": "none"}

    if family == "lppl":
        n_checked = len(n_freqs or [])
        if n_checked != 1:
            warning = _decomp_warning_banner(n_checked)
            return [], [], warning, {"display": "block"}
        # Resolve to specific variant
        resolved_key = _resolve_lppl_variant(n_freqs, weighted, no_13)
        model = _app_ctx.PRICE_MODELS[resolved_key]
    else:
        model = _app_ctx.PRICE_MODELS[family]

    names = model.component_names
    opts = [{"label": f" {name}", "value": name} for name in names]
    opts.append({"label": " Σ Sum of selected",
                 "value": "__sum__"})

    # Preserve user's checked state where names still exist
    valid = {o["value"] for o in opts}
    kept = [v for v in (current_checked or []) if v in valid]
    return opts, kept, [], {"display": "block"}
```

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
                               "color": colors[len(names) % len(colors)]},
                        name=f"{model.legend_name} | Σ ({len(names)} components)")
```

### Snapshot / share link (`btc_web/snapshot.py`)

Append two entries to `_SNAPSHOT_CONTROLS`:

```python
("bub-decomp-model",      "value"),
("bub-decomp-components", "value"),
```

Add to `_TAB_CONTROLS["bubble"]` set. Checklist is bitmask-encoded via `_CHECKLIST_OPTIONS` — but the option list is dynamic per model, so we cannot use bitmask here. Store the list of string values as-is (plain JSON); no bitmask.

Snapshot scheme stays at `q3:` (backward-compatible addition).

### Trace styling details

- Individual components: `dash="dot"`, `width=1.5`, color from `DECOMP_COLORS[palette_key]`.
- Sum trace: `dash="solid"`, `width=3`, one color position past the last individual component.
- Hover tooltip: `"{component_name}: $X at year Y"` (USD on hover; log value not shown).
- `showlegend=True` (respects the global Show Legend toggle via layout `showlegend`).

### Auto-Y interaction

When decomposition is active, `auto_bubble_yrange` callback MUST include component y-values when computing the fit range. Otherwise a constant like `A = -1.15` (horizontal line at $0.07) would force the Y-range to include `10^(-1.15)` which may be far below the model bands.

Modification: `auto_bubble_yrange` takes new States for `bub-decomp-model` and `bub-decomp-components`, and extends the computed `p_lo`/`p_hi` with component min/max values over `[t_lo, t_hi]` when decomposition is active.

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
