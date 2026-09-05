# Item 4 — one table drives Tab-1's five view pills

**Date:** 2026-09-04
**Scope:** `btc_web/bub_views.py` (new), `toggle_bub_view` + the three
`bub-view-mode` clientside callbacks in `btc_web/callbacks/charts/__init__.py`,
`deep_link_bub_view` in `btc_web/callbacks/routing.py`,
`btc_web/test_bub_view_modes.py` (new), one test body in
`btc_web/test_occupancy.py`.
**Behaviour change:** none intended, none observed.

## Problem

Tab 1's per-view UI state — which graph wrapper is visible, which pill is
filled, whether the scale controls and bubble panel show, which view-specific
control span shows, and whether the view is historical-only — was written out
by hand in three places that had to agree:

| # | Producer | Outputs | Shape |
|---|---|---|---|
| 1 | `callbacks/charts/__init__.py::toggle_bub_view` | 16 | five-branch `if/elif` on `ctx.triggered_id` |
| 2 | the clientside sync callback right below it | 14 | five hand-written `return [...]` JS arrays |
| 3 | `callbacks/routing.py::deep_link_bub_view` | 20 | four `pathname.startswith` branches |

Positional outputs, so a row that disagreed would mis-assign styles rather than
raise. Two reviewers had to verify agreement by eye. A sixth pill meant editing
all three plus the tests.

## The table

`btc_web/bub_views.py` — import-light (stdlib only; no Dash, no layout), so both
callback modules and the tests can import it without a cycle.

```python
class ViewMode(NamedTuple):
    pill: str              # pill button id      (drives `outline`)
    wrap: str              # graph wrapper id    (drives `style`)
    historical: bool       # x-range capped at next year, N-future hidden
    scale_controls: bool   # show the axis-scale controls block
    bubble_panel: bool     # show the bubble-composite panel
    ctl: str | None        # view-specific control span id
    deep_link: str | None  # URL prefix that opens this view
```

| mode | pill | wrap | historical | scale | panel | ctl | deep_link |
|---|---|---|---|---|---|---|---|
| `price` | `bub-view-price` | `bub-price-wrap` | no | yes | yes | — | — |
| `cagr` | `bub-view-cagr` | `bub-cagr-wrap` | no | **no** | **no** | `bub-cagr-fwd-wrap` | `/1.2` |
| `resid` | `bub-view-resid` | `bub-resid-wrap` | **yes** | yes | yes | — | `/1.3` |
| `percentile` | `bub-view-pctile` | `bub-pctile-wrap` | **yes** | yes | yes | — | `/1.4` |
| `occupancy` | `bub-view-occ` | `bub-occ-wrap` | **yes** | yes | yes | `bub-occ-ctl-wrap` | `/1.5` |

Derived from it, in table order (never hand-maintained):

* `WRAP_IDS`, `PILL_IDS`, `CTL_IDS`, `PANEL_IDS = ("bub-scale-controls", "bub-bubble-panel")`
* `STYLE_OUTPUT_IDS = WRAP_IDS + PILL_IDS + PANEL_IDS + CTL_IDS` — the 14 outputs, in the order all three callbacks declare them
* `HISTORICAL_MODES = {"occupancy", "percentile", "resid"}`
* `mode_styles(mode) -> tuple` — the 14 values; unknown modes fall back to `DEFAULT_MODE` (`"price"`), matching the JS `T[mode] || T["price"]`
* `mode_for_path(pathname) -> mode | None` — longest `deep_link` prefix match, on an already-`_norm`-ed path
* `styles_table_json()` / `historical_modes_js()` — the generated JS

Values emitted are byte-identical to the old hand-written ones: wrappers `{}` /
`{"display": "none"}`, pill `outline` `False` for the active pill, control spans
`{"display": "inline"}` when shown.

## What each producer now does

**1. `toggle_bub_view`** — 34 lines of branches became:

```python
mode = _PILL_TO_MODE.get(ctx.triggered_id, DEFAULT_MODE)
if mode == "cagr":
    xr = CAGR_DEFAULT_XRANGE if cur_xrange == [2010, 2033] else dash.no_update
else:
    xr = [2010, 2033] if cur_xrange == CAGR_DEFAULT_XRANGE else dash.no_update
return (mode, *mode_styles(mode), xr)
```

The x-range swap is unchanged, including the conditions: each direction only
fires when the slider still sits on the *other* view's default, so a
user-chosen range is never clobbered. `_PILL_TO_MODE` is derived from the table;
an unknown trigger falls through to `price`, exactly as the old trailing
`# Price` branch did.

**2. the clientside sync** — five `return [...]` arrays became one generated
JSON table:

```js
function(mode) {
    /* QS_BUB_VIEW_SYNC_TABLE — generated from bub_views.VIEW_MODES;
       one array of 14 values per mode, in this callback's Output order. */
    var T = {"price": [...], "cagr": [...], ...};
    return T[mode] || T["price"];
}
```

`SYNC_JS_MARKER` is the stable marker tests use to find this exact inline
script.

**3. `deep_link_bub_view`** — four `startswith` branches became
`mode_for_path` + `mode_styles`, with only the five trailing control values
still per-view (CAGR `N`/`B` parsing, occupancy `T`/`W` parsing, x-range).
`_pick`'s 1-based indexing, its `ValueError`/out-of-range → `no_update`
handling, and the `(no_update,) * 20` fallback are untouched.

The clientside tab-map's `/^\/1\.\d+/` regex was already generic — unchanged.

## Deviation from the brief (one)

The brief asked the two historical-only clientside callbacks to embed
`json.dumps(sorted(HISTORICAL_MODES))` and test membership. They instead embed
a **generated** `mode === 'occupancy' || mode === 'percentile' || mode === 'resid'`
chain (`bub_views.historical_modes_js()`).

Reason: `test_occupancy.py::TestOccupancyWiring::test_historical_only_clientside_checks_include_occupancy`
asserts exactly two inline scripts contain the literal `mode === 'occupancy'`,
and the brief allowed editing only *one* test in that file
(`test_clientside_sync_returns_one_value_per_output`). A JSON array would have
broken a test I was not permitted to touch. The chain is still generated from
`HISTORICAL_MODES` — a sixth historical pill needs no JS edit — so the "one
table" property holds either way, and the emitted JS is now semantically
identical to what shipped before. `test_bub_view_modes.py::TestHistoricalOnlyScripts`
pins the chain to `sorted(HISTORICAL_MODES)` exactly (and that non-historical
modes are absent), which is requirement (d) in a different encoding.

## Diff summary

```
 btc_web/callbacks/charts/__init__.py | 95 ++++++++++++++++--------------------
 btc_web/callbacks/routing.py         | 46 ++++++-----------
 btc_web/test_occupancy.py            | 22 ++++++---
 3 files changed, 72 insertions(+), 91 deletions(-)
```

New files: `btc_web/bub_views.py` (164 lines), `btc_web/test_bub_view_modes.py` (304 lines).

### Full diff — `btc_web/callbacks/charts/__init__.py` and `btc_web/callbacks/routing.py`

```diff
diff --git i/btc_web/callbacks/charts/__init__.py w/btc_web/callbacks/charts/__init__.py
index 810c2e2..68312df 100644
--- i/btc_web/callbacks/charts/__init__.py
+++ w/btc_web/callbacks/charts/__init__.py
@@ -17,6 +17,7 @@ Public API preserved: ``from callbacks.charts import update_bubble, ...``
 still works exactly as before.
 """
 
+import json
 import math
 
 import dash
@@ -61,6 +62,9 @@ from figures.common import apply_zoom_lock
 from tab_defaults import BUBBLE, HEATMAP, DCA, RETIRE, SUPERCHARGE
 from layout.common import _bands_to_qs
 from layout.bubble import CAGR_DEFAULT_XRANGE
+from bub_views import (VIEW_MODES, DEFAULT_MODE, SYNC_JS_MARKER,
+                       mode_styles, styles_table_json,
+                       historical_modes_js)
 from callbacks.coerce import _ci, _cf
 from callbacks.timemachine import _asof_frame
 from callbacks.mc_helpers import (_mc_setup, _mc_finalize, _mc_status,
@@ -538,6 +542,12 @@ def update_bubble(_first_render, sel_qs, adv_qs, toggles, bubble_toggles,
 
 # ── Price/CAGR view pill bar ─────────────────────────────────────────────────
 
+# Pill button id -> view mode.  Derived from the table so a new pill needs no
+# edit here; an unknown trigger falls back to the default view, exactly as the
+# old if/elif chain's trailing "# Price" branch did.
+_PILL_TO_MODE = {v.pill: m for m, v in VIEW_MODES.items()}
+
+
 @callback(
     Output("bub-view-mode", "data"),
     Output("bub-price-wrap", "style"),
@@ -565,64 +575,40 @@ def update_bubble(_first_render, sel_qs, adv_qs, toggles, bubble_toggles,
 )
 def toggle_bub_view(price_clicks, cagr_clicks, resid_clicks, pctile_clicks,
                     occ_clicks, cur_xrange):
-    triggered = ctx.triggered_id
-    _hide = {"display": "none"}
-    _show_inline = {"display": "inline"}
-    if triggered == "bub-view-cagr":
-        xr = CAGR_DEFAULT_XRANGE if cur_xrange == [2010, 2033] else dash.no_update
-        return ("cagr", _hide, {}, _hide, _hide, _hide,
-                True, False, True, True, True,
-                _hide, _hide, _show_inline, _hide, xr)
-    if triggered == "bub-view-resid":
-        # Residuals: keep same x-range as price view, keep bubble panel visible
-        xr = [2010, 2033] if cur_xrange == CAGR_DEFAULT_XRANGE else dash.no_update
-        return ("resid", _hide, _hide, {}, _hide, _hide,
-                True, True, False, True, True,
-                {}, {}, _hide, _hide, xr)
-    if triggered == "bub-view-pctile":
-        # Percentile oscillator: time x-axis like residuals; historical only.
-        xr = [2010, 2033] if cur_xrange == CAGR_DEFAULT_XRANGE else dash.no_update
-        return ("percentile", _hide, _hide, _hide, {}, _hide,
-                True, True, True, False, True,
-                {}, {}, _hide, _hide, xr)
-    if triggered == "bub-view-occ":
-        # Occupancy (time in the fan's tails): historical only, like percentile.
-        xr = [2010, 2033] if cur_xrange == CAGR_DEFAULT_XRANGE else dash.no_update
-        return ("occupancy", _hide, _hide, _hide, _hide, {},
-                True, True, True, True, False,
-                {}, {}, _hide, _show_inline, xr)
-    # Price
-    xr = [2010, 2033] if cur_xrange == CAGR_DEFAULT_XRANGE else dash.no_update
-    return ("price", {}, _hide, _hide, _hide, _hide,
-            False, True, True, True, True,
-            {}, {}, _hide, _hide, xr)
+    """Pill click -> view mode + the 14 view-state values + an x-range swap.
 
+    Every UI value except the x-range comes from ``bub_views.mode_styles``; the
+    Output order above mirrors ``bub_views.STYLE_OUTPUT_IDS`` (pinned by
+    test_bub_view_modes.py).  Only the x-range is per-view logic: CAGR wants a
+    forward window, every other view wants the price window back — and each
+    swap only fires when the slider is still sitting on the *other* view's
+    default, so a user-chosen range is never clobbered.
+    """
+    mode = _PILL_TO_MODE.get(ctx.triggered_id, DEFAULT_MODE)
+    if mode == "cagr":
+        xr = CAGR_DEFAULT_XRANGE if cur_xrange == [2010, 2033] else dash.no_update
+    else:
+        xr = [2010, 2033] if cur_xrange == CAGR_DEFAULT_XRANGE else dash.no_update
+    return (mode, *mode_styles(mode), xr)
+
+
+# Generated from bub_views.VIEW_MODES: one 14-value array per mode, in the
+# Output order below.  The marker in the comment is how tests find this exact
+# inline script among all the others.
+_VIEW_SYNC_JS = (
+    "function(mode) {\n"
+    "    /* " + SYNC_JS_MARKER + " — generated from bub_views.VIEW_MODES;\n"
+    "       one array of 14 values per mode, in this callback's Output order. */\n"
+    "    var T = " + styles_table_json() + ";\n"
+    "    return T[mode] || T[" + json.dumps(DEFAULT_MODE) + "];\n"
+    "}\n"
+)
 
 # Sync view-mode wrappers + button outlines when bub-view-mode.data changes
 # (e.g., from snapshot restore — button clicks set it directly in toggle_bub_view,
 # but snapshot sets it via apply_tab_bubble without clicking buttons).
 _app_ctx.app.clientside_callback(
-    """
-    function(mode) {
-        var _h = {"display": "none"};
-        var _i = {"display":"inline"};
-        /* [price,cagr,resid,pctile,occ wraps][5 outlines][scale,panel,cagr-fwd,occ-ctl] */
-        if (mode === "cagr") {
-            return [_h, {}, _h, _h, _h, true, false, true, true, true, _h, _h, _i, _h];
-        }
-        if (mode === "resid") {
-            return [_h, _h, {}, _h, _h, true, true, false, true, true, {}, {}, _h, _h];
-        }
-        if (mode === "percentile") {
-            return [_h, _h, _h, {}, _h, true, true, true, false, true, {}, {}, _h, _h];
-        }
-        if (mode === "occupancy") {
-            return [_h, _h, _h, _h, {}, true, true, true, true, false, {}, {}, _h, _i];
-        }
-        /* price (default) */
-        return [{}, _h, _h, _h, _h, false, true, true, true, true, {}, {}, _h, _h];
-    }
-    """,
+    _VIEW_SYNC_JS,
     Output("bub-price-wrap", "style", allow_duplicate=True),
     Output("bub-cagr-wrap", "style", allow_duplicate=True),
     Output("bub-resid-wrap", "style", allow_duplicate=True),
@@ -643,7 +629,8 @@ _app_ctx.app.clientside_callback(
 
 # Hide "N future bubbles" slider in residuals/percentile/occupancy views (doesn't apply to past data)
 _app_ctx.app.clientside_callback(
-    "function(mode) { return (mode === 'resid' || mode === 'percentile' || mode === 'occupancy') ? {display: 'none'} : {}; }",
+    "function(mode) { return (" + historical_modes_js()
+    + ") ? {display: 'none'} : {}; }",
     Output("bub-n-future-wrap", "style"),
     Input("bub-view-mode", "data"),
 )
@@ -655,7 +642,7 @@ _app_ctx.app.clientside_callback(
     """
     function(mode, cur_range) {
         var hist_max = (new Date()).getFullYear() + 1;
-        var new_max = (mode === 'resid' || mode === 'percentile' || mode === 'occupancy') ? hist_max : 2080;
+        var new_max = (""" + historical_modes_js() + """) ? hist_max : 2080;
         // Cap current value if it exceeds the new max
         var r = (cur_range || [2010, 2033]).slice();
         if (r[1] > new_max) r[1] = new_max;
diff --git i/btc_web/callbacks/routing.py w/btc_web/callbacks/routing.py
index dcb6ff7..4aa9ce4 100644
--- i/btc_web/callbacks/routing.py
+++ w/btc_web/callbacks/routing.py
@@ -6,6 +6,7 @@ from dash import Input, Output, State, callback, ctx, no_update, ALL
 import _app_ctx
 from layout.faq import _FAQ
 from layout.bubble import CAGR_DEFAULT_XRANGE
+from bub_views import mode_for_path, mode_styles
 
 
 def _norm(pathname: str | None) -> str | None:
@@ -406,16 +407,18 @@ def deep_link_bub_view(pathname):
     /1.3          Residuals
     /1.4          Percentile
     /1.5[.T[.W]]  Occupancy     (T -> tail 5/10/25 %, W -> window 1/2/4 yr)
-    Indices are 1-based; '-' is accepted for '.' (see _norm). Output order
-    mirrors toggle_bub_view, then the CAGR / occupancy control values.
+    Indices are 1-based; '-' is accepted for '.' (see _norm).  Which path opens
+    which view is `deep_link` in bub_views.VIEW_MODES; the first 15 outputs are
+    the mode plus bub_views.mode_styles(mode) — the same values toggle_bub_view
+    returns, in the same order — and only the five trailing control values are
+    per-view logic.
     """
     from dash import no_update
     NU = no_update
     pathname = _norm(pathname)
-    if not pathname:
+    mode = mode_for_path(pathname)
+    if mode is None:
         return (NU,) * 20
-    _hide = {"display": "none"}
-    _inline = {"display": "inline"}
     parts = pathname[1:].split(".")
 
     def _pick(i, options):
@@ -428,25 +431,9 @@ def deep_link_bub_view(pathname):
             return NU
         return options[n - 1] if 1 <= n <= len(options) else NU
 
-    if pathname.startswith("/1.3"):
-        return ("resid", _hide, _hide, {}, _hide, _hide,
-                True, True, False, True, True,
-                {}, {}, _hide, _hide,
-                NU, NU, NU, NU, NU)
-
-    if pathname.startswith("/1.4"):
-        return ("percentile", _hide, _hide, _hide, {}, _hide,
-                True, True, True, False, True,
-                {}, {}, _hide, _hide,
-                NU, NU, NU, NU, NU)
-
-    if pathname.startswith("/1.5"):
-        return ("occupancy", _hide, _hide, _hide, _hide, {},
-                True, True, True, True, False,
-                {}, {}, _hide, _inline,
-                NU, NU, NU, _pick(2, [5, 10, 25]), _pick(3, [1, 2, 4]))
-
-    if pathname.startswith("/1.2"):
+    # xrange, cagr fwd-yrs, cagr hover-today, occ tail, occ window
+    extras = (NU, NU, NU, NU, NU)
+    if mode == "cagr":
         hover_today = NU
         if len(parts) >= 4:
             try:
@@ -454,13 +441,12 @@ def deep_link_bub_view(pathname):
                     hover_today = True
             except ValueError:
                 pass
-        return ("cagr", _hide, {}, _hide, _hide, _hide,
-                True, False, True, True, True,
-                _hide, _hide, _inline, _hide,
-                CAGR_DEFAULT_XRANGE, _pick(2, [1, 2, 4, 10, 20, 30]), hover_today,
-                NU, NU)
+        extras = (CAGR_DEFAULT_XRANGE, _pick(2, [1, 2, 4, 10, 20, 30]),
+                  hover_today, NU, NU)
+    elif mode == "occupancy":
+        extras = (NU, NU, NU, _pick(2, [5, 10, 25]), _pick(3, [1, 2, 4]))
 
-    return (NU,) * 20
+    return (mode, *mode_styles(mode), *extras)
 
 
 # ══════════════════════════════════════════════════════════════════════════════
```

### `btc_web/test_occupancy.py` — the one permitted test-body replacement

```diff
diff --git i/btc_web/test_occupancy.py w/btc_web/test_occupancy.py
index 424d7f5..fb39f37 100644
--- i/btc_web/test_occupancy.py
+++ w/btc_web/test_occupancy.py
@@ -284,15 +284,23 @@ class TestOccupancyWiring:
         assert out[14] == {"display": "none"}
 
     def test_clientside_sync_returns_one_value_per_output(self):
-        # The view-mode sync JS returns positional arrays; adding an Output
-        # without extending every `return [...]` silently mis-assigns styles.
+        # The view-mode sync JS returns positional arrays; a mode whose row is
+        # short silently mis-assigns styles.  The five hand-written
+        # `return [...]` branches are now one generated JSON table keyed by
+        # mode (bub_views.VIEW_MODES), so this checks the table instead.
+        # Fuller coverage — every mode, against every producer — lives in
+        # test_bub_view_modes.py::TestClientsideSyncTable.
+        import json
+
+        from bub_views import SYNC_JS_MARKER, VIEW_MODES
         scripts = [s for s in _app_ctx.app._inline_scripts
-                   if 'mode === "occupancy"' in s and 'mode === "percentile"' in s]
+                   if SYNC_JS_MARKER in s]
         assert len(scripts) == 1, "view-mode sync clientside callback not found"
-        returns = re.findall(r"return \[(.*?)\];", scripts[0], flags=re.S)
-        assert len(returns) == 5   # cagr, resid, percentile, occupancy, price
-        counts = {len([e for e in r.split(",") if e.strip()]) for r in returns}
-        assert counts == {14}, counts
+        table = json.loads(
+            re.search(r"var T = (\{.*\});", scripts[0], flags=re.S).group(1))
+        assert set(table) == set(VIEW_MODES)   # price, cagr, resid, pctile, occ
+        assert {len(row) for row in table.values()} == {14}
+        assert table["occupancy"][4] == {}     # bub-occ-wrap shown
 
     def test_historical_only_clientside_checks_include_occupancy(self):
         hits = [s for s in _app_ctx.app._inline_scripts
```

## Tests

`btc_web/test_bub_view_modes.py` — 45 tests (43 pass, 2 skip: `price` has no
deep link, so the two parametrized deep-link cases skip for it). Grouped by what
they pin:

* **`TestTableShape`** — derived id lists follow table order and are unique;
  `STYLE_OUTPUT_IDS` is 14 long; for every mode, `mode_styles` shows exactly one
  wrapper, fills exactly one pill, and sets scale/panel/ctl from the row;
  unknown mode falls back to the default; the returned dicts are *fresh objects*
  (a shared instance would let one caller's mutation leak into every later view
  switch); `HISTORICAL_MODES` matches the table.
* **`TestModeForPath`** — each deep link resolves to its own mode (bare and with
  a `.N.B` suffix); non-view paths (`/1`, `/2`, `/9.3`, `/10`, `/faq.2`, `None`,
  `""`) are `None`; longest-prefix wins over listing order (checked by patching
  in a hypothetical `/1.2.9` row).
* **`TestToggleBubView`** — for every mode, positions 1..14 `== mode_styles(mode)`;
  unknown trigger → default; the x-range swap fires in both directions only off
  the other view's default; the callback's registered Output order is
  `["bub-view-mode", *STYLE_OUTPUT_IDS, "bub-xrange"]`.
* **`TestClientsideSyncTable`** — the embedded JSON decodes to exactly
  `{mode: list(mode_styles(mode))}`; every row is 14 long; the fallback is
  `DEFAULT_MODE`; the sync callback's Output order `== STYLE_OUTPUT_IDS`.
* **`TestHistoricalOnlyScripts`** — exactly two scripts gate on modes, each
  listing exactly `sorted(HISTORICAL_MODES)`, with no non-historical mode named.
* **`TestDeepLinkUsesTheTable`** — arity 20, positions 1..14 `== mode_styles(mode)`
  for every mode with a deep link; modes without one are URL-unreachable; Output
  order is `[bub-view-mode, *STYLE_OUTPUT_IDS, bub-xrange, bub-cagr-fwd-yrs,
  bub-cagr-hover-today, bub-occ-tail, bub-occ-window]`.
* **`TestTableIdsExistInLayout`** — every `pill` / `wrap` / `ctl` / panel id
  exists in the served layout, reusing `test_no_orphan_callbacks._collect_layout_ids()`
  (builds every tab's content directly — no server, no browser).

Output order is read from `dash._callback.GLOBAL_CALLBACK_MAP` merged with
`_app_ctx.app.callback_map`, the same lookup `test_bub_deep_links::_n_outputs`
uses. All three view callbacks drive the same 14 components, so `_find_callback`
takes a `must_not_contain` tuple to tell them apart by their *extra* outputs.

### Mutation check (guard against a false PASS)

Reversing the pill order inside `mode_styles` (`[::-1]`) — a change that must be
caught, since the table now feeds all three producers — turned **14 tests red**
across `test_bub_view_modes.py`, `test_occupancy.py` and `test_bub_deep_links.py`.
Reverted immediately.

### Run output

Targeted set (`test_bub_view_modes`, `test_occupancy`, `test_bub_deep_links`,
`test_bub_view_gating`, `test_axes_presets`, `test_no_orphan_callbacks`,
`test_infrastructure`), `-q -p no:randomly -n0`:

```
4 failed, 133 passed, 2 skipped, 3 warnings in 4.27s
FAILED btc_web/test_occupancy.py::TestOccupancyFigure::test_strip_marks_first_model_only
FAILED btc_web/test_occupancy.py::TestOccupancyFigure::test_strip_hover_trace_covers_every_displayed_day
FAILED btc_web/test_occupancy.py::TestOccupancyFigure::test_strip_hover_text_matches_line_values
FAILED btc_web/test_occupancy.py::TestOccupancyFigure::test_strip_hover_before_full_window_says_so
```

Full suite (`btc_venv/bin/python3 -m pytest -q`):

```
5 failed, 2928 passed, 12 skipped, 19 warnings in 22.38s
FAILED btc_web/test_callbacks.py::TestBTCPayPricing::test_free_tier_all_models
FAILED btc_web/test_occupancy.py::TestOccupancyFigure::test_strip_hover_trace_covers_every_displayed_day
FAILED btc_web/test_occupancy.py::TestOccupancyFigure::test_strip_hover_text_matches_line_values
FAILED btc_web/test_occupancy.py::TestOccupancyFigure::test_strip_hover_before_full_window_says_so
FAILED btc_web/test_occupancy.py::TestOccupancyFigure::test_strip_marks_first_model_only
```

**None of the failures are from this work.**

* `test_free_tier_all_models` is the one known pre-existing failure named in the brief.
* The four `TestOccupancyFigure` strip/hover failures come from a concurrent
  agent's in-flight edits to `btc_web/figures/occupancy.py` and
  `btc_web/figures/common.py`. They fail inside `build_occupancy_figure`
  (`assert len(strip) == 3` now sees 4 traces) — code this work never touches.
  They were transiently 5 during one run and back to 4 on the next, tracking
  that agent's live edits. `TestOccupancyWiring` and `TestOccupancySnapshot`
  (the classes this work does touch) are 10/10 green.

## Browser click-through

`PORT=8062 DEV=1 bash run_web.sh`, Playwright MCP driving Firefox. After each
pill click, `getComputedStyle(...).display` of the five wrappers, the two
control spans, `bub-scale-controls`, `bub-bubble-panel`, `bub-n-future-wrap`,
and the pill fill state (`btn-outline-*` class present or not).

| action | wraps (price/cagr/resid/pctile/occ) | filled pill | scale | panel | cagr ctl | occ ctl | N-future |
|---|---|---|---|---|---|---|---|
| load `/1` | block · none · none · none · none | price | block | none¹ | none | none | block |
| click `#bub-view-cagr` | none · **block** · none · none · none | cagr | **none** | **none** | **block** | none | block |
| click `#bub-view-resid` | none · none · **block** · none · none | resid | block | block | none | none | **none** |
| click `#bub-view-pctile` | none · none · none · **block** · none | pctile | block | block | none | none | **none** |
| click `#bub-view-occ` | none · none · none · none · **block** | occ | block | block | none | **block** | **none** |
| click `#bub-view-price` | **block** · none · none · none · none | price | block | block | none | none | block |

¹ `bub-bubble-panel` starts hidden from the *layout* default; the sync callback
is `prevent_initial_call=True`, so nothing has driven it on first paint. First
pill click sets it per the table. Unchanged from before.

Deep links:

| URL | result |
|---|---|
| `/1.2.5.1` | tab `Price Models`; CAGR wrap shown, CAGR pill filled, scale + panel hidden, CAGR ctl shown, N-future shown; **`bub-cagr-fwd-yrs` = `20yr`** (index 5 of `[1,2,4,10,20,30]`) |
| `/1.5.3.1` | Occupancy wrap shown, occupancy pill filled, scale + panel shown, occ ctl shown, N-future hidden; **`bub-occ-tail` = `tail 25%`**, **`bub-occ-window` = `1yr`**; occupancy graph rendered |

Two notes from the walk, both pre-existing:

* Control spans emit `display: inline` (raw `style` attribute confirmed as
  `display: inline;`), but `getComputedStyle` reports `block` — the parent is a
  flex container, which blockifies `inline`. The emitted value is byte-identical
  to what shipped before; only the *used* value differs, and it always has.
* Ten console errors on every page load — orphan `{dca,ret,sc,cp}-mc-years`
  `State` refs on lazy tabs, a `citadel-graph` 500 in DEV, and two React
  warnings. None mentions any `bub-view-*` id. Pre-existing.

Server killed afterwards (`lsof -ti :8062 | xargs -r kill -9`); port confirmed free.

## How to add a sixth pill

1. **Layout** (`layout/bubble.py`): add the pill button, the graph wrapper div,
   and — if the view needs its own controls — a control span, following the
   `bub-view-*` / `bub-*-wrap` naming.
2. **Table** (`bub_views.py`): add one `VIEW_MODES` row. Order matters — it is
   the Output order. **Append it** unless you also reorder the three Output
   lists; appending keeps every existing position stable.
3. **Outputs**: add the new wrapper `style`, pill `outline`, and (if any)
   control-span `style` to all three Output lists —
   `toggle_bub_view`, the sync clientside callback, and `deep_link_bub_view` —
   in `STYLE_OUTPUT_IDS` order. This is the one mechanical step the table does
   not do for you, and `test_bub_view_modes.py`'s three
   `test_output_order_is_the_table_order*` tests fail loudly if you get it wrong.
   Nothing else in any of the three callback *bodies* changes.
4. **Figure callback**: add `update_bub_<view>` with `Input("bub-view-mode", "data")`
   and the standard `if view_mode != "<mode>": return dash.no_update` gate —
   `test_bub_view_gating.py` pins that an unlisted view never builds.
5. **Deep link** (optional): set `deep_link="/1.6"` on the row. `mode_for_path`
   picks it up; the clientside tab-map's `/^\/1\.\d+/` already routes it. Add
   trailing `_pick(...)` extras in `deep_link_bub_view` only if the view has
   URL-addressable control values.
6. **Snapshot** (only if the view adds controls): append `(id, prop)` pairs to
   the **absolute tail** of `_SNAPSHOT_CONTROLS`, add defaults to
   `snapshot_defaults.py`, add the ids to `_TAB_CONTROLS["bubble"]`, and re-pin
   the fingerprint with `tools/update_defaults_registry.py`.
7. **Tests**: none to add for the view-state wiring —
   `test_bub_view_modes.py` is parametrized over `VIEW_MODES`, so the new row is
   covered by all three producers, the generated JS, the historical-mode
   scripts, and the layout-id check the moment it exists.

Steps 1, 4 and 6 are per-view work that no table can absorb. Steps 2, 3 and 5
are what used to be "edit three hand-written state machines and hope they agree".
