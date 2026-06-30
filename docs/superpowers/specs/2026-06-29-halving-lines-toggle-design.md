# Halving-line toggle on Tab 1 (Price & Model Overlays)

**Date:** 2026-06-29
**Status:** Approved
**Scope:** Single additive control on the Bubble tab.

## Goal

Add one checkbox, **"Show halving lines"**, to the Display section of Tab 1.
When on, draw vertical lines at each Bitcoin halving — past (known) and future
(estimated) — over the price chart, each with a small year label.

## Decisions (from brainstorm)

1. **Date source:** hardcoded. Known dates for the 4 past halvings; future
   halvings generated at the nominal ~4-year cadence anchored on the
   2024-04-20 halving. No runtime file I/O (ALARA — keep server light). At the
   chart's decade-wide scale a hardcoded estimate is visually identical to a
   live block-extrapolated one.
2. **Style:** single muted, colorblind-safe color (indigo — reliable on the
   blue–yellow axis for deuteranomaly). Past = solid; future = dashed +
   lower opacity (conveys certainty vs. estimate). Each line carries a small
   `⛏ YYYY` label.
3. **Default:** off (opt-in), like the Unfairly Cheap Line / OLS toggles.

## Data

Known halvings (block 210 000·n), actual dates:
`2012-11-28, 2016-07-09, 2020-05-11, 2024-04-20`.

Future halvings: anchor `2024-04-20` + `4·k` years (`pd.DateOffset`), generated
out to year 2080 (covers the max x-range). Marked `is_estimated=True`.

Each date → chart `t` via `(date − genesis).days / 365.25` (the same
conversion lot markers already use). A line is drawn only when
`t_lo ≤ t ≤ t_hi`, exactly like the today line — off-screen halvings add
nothing.

## Components

- **`colors.py`** (appearance SSOT): add `HALVING_LINE_COLOR` (`#3F51B5`,
  muted indigo) in section 1 near `TODAY_LINE_COLOR`; add
  `TRACE_WIDTH_HALVING = 1.4` near the trace-width block; add
  `HALVING_PAST_OPACITY = 0.55` and `HALVING_FUTURE_OPACITY = 0.35` in the
  opacities block. Regenerate `_colors_generated.{css,js}`
  (`HALVING_LINE_COLOR` is a hex string → exported like `TODAY_LINE_COLOR`).
- **`figures/common.py`**: add the halving epoch table
  (`_HALVING_EPOCHS`, computed once at import) and a pure helper
  `_halving_line_shapes(epochs, genesis, t_lo, t_hi, y_lo, y_hi, color,
  past_op, future_op, width, yref="y")` next to `_today_line_shapes`. Returns
  Plotly vertical-line shape dicts; future lines dashed at `future_op`, past
  solid at `past_op`; each shape carries a `label` (`⛏ year`, top-center,
  `CHART_FONT_LEGEND`).
- **`figures/bubble.py`**: after the today-line block,
  `if p.get("show_halvings"): shapes.extend(_halving_line_shapes(...))`.
  Imports the helper + constants. Shapes already flow into `layout["shapes"]`.
- **`layout/bubble.py`**: add `{"label":" Show halving lines","value":
  "show_halvings"}` to the `bub-toggles` checklist (Display card). Not in the
  default `value` list → off by default.
- **`callbacks/charts/__init__.py`**: add
  `show_halvings = "show_halvings" in toggles` to **both** bubble param dicts
  (MC-aware builder and fast path). Residuals chart untouched (halvings are a
  price-chart feature).
- **`tab_defaults.py`**: add `"show_halvings": "show_halvings" in toggles` to
  the bubble defaults so the prewarm L0 cache key matches the runtime key
  (enforced by `test_cache_key_alignment.py`).
- **`snapshot.py`**: append `"show_halvings"` to the **end** of
  `_CHECKLIST_OPTIONS["bub-toggles"]` so existing share-link bitmasks stay
  valid (bit positions are append-only).

## Testing

`test_halving_lines.py`:
- Unit: `_halving_line_shapes` — count, dash (solid past / dashed future),
  opacity, label text; range clipping drops off-screen halvings.
- Integration: `build_bubble_figure` with `show_halvings=True` adds halving
  shapes; `False` adds none; today-line + halving shapes coexist.

Plus `test_cache_key_alignment.py` and the existing bubble figure tests.
(Per project convention, no test for the pure layout checkbox.)

## Out of scope (YAGNI)

Live block extrapolation; halving lines on residuals / other tabs; per-line
color customization; user-editable cadence.
