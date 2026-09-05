# Item 5 — Occupancy "when"-strip hover payload

Agent report, 2026-09-04. Scope: `btc_web/figures/occupancy.py`,
`_add_date_hover` in `btc_web/figures/common.py`, the hover/strip tests in
`btc_web/test_occupancy.py`, plus a new `btc_web/test_date_hover.py`.
Nothing committed — the controller commits.

## Headline

The strip hover trace went from **362,720 → 194,976 bytes (1.86×)** of shipped
JSON, with the rendered label byte-identical and daily hover resolution intact.

**The 4× target is arithmetically unreachable under the stated constraints, and
the report shows why.** 4× is a budget of 90,680 bytes; the per-day
`"Mon DD, YYYY"` strings alone cost **85,877**, leaving 4,803 for the x
positions, the two shares, the y row and all trace scaffolding. Section
"Why not 4×" below gives the two levers that do reach it, both of which change
something the brief froze.

A second result, arguably worth more than the bytes: **the browser check caught
a real bug I had introduced.** See "The dy=0 trap".

## 1. Evidence — every existing `customdata=` user, checked

`_add_date_hover` used to *replace* `trace.customdata`. Step 1 makes it
*prepend* when a trace already carries one row per point. Before changing it I
enumerated every `customdata=` assignment that could reach the helper.

`_add_date_hover` is reached from exactly two places
(`grep -rn "_add_date_hover" btc_web/`):

- `figures/common.py:_apply_final_steps` — used by `bubble`, `residuals`,
  `percentile`, `occupancy` directly, and by `dca`, `retire`, `supercharge`,
  `citadel` via `_finalize_chart`.
- `figures/bubble.py:702` (`recovery=True`).

| # | Site | Reaches the helper? | Verdict |
|---|---|---|---|
| 1 | `figures/heatmap.py:640` — `customdata=list(zip(cagr_max, cagr_min, peak_dates, trough_dates, cagr_mult, peak_mult, trough_mult))` | **No.** `heatmap.py` never calls `_add_date_hover`, `_apply_final_steps` or `_finalize_chart` — it builds `go.Figure` and calls `_apply_watermark` only. | Unaffected. Belt and braces: its x are calendar years (≈2010–2075), which the helper's `x_max > 120` guard skips anyway. |
| 2 | `figures/supercharge.py:607` and `:677` — `customdata=qlbls`, a **flat** list of quantile-label strings with a `%{customdata}` (unindexed) template, Mode B `chart_layout == 2` | **Sometimes.** supercharge reaches the helper via `_finalize_chart`. Its x are quantile *fractions*; with the default SC quantiles (Q0.1% + Q10% → min 0.001) `x_min < 0.3` skips it, but a selection whose lowest quantile is ≥ 0.3 (e.g. Q50% + Q90%) passes the guard and the helper does act. | **Unchanged by design.** The new code preserves customdata only when *every* element is a row (`list`/`tuple`/`ndarray`); a flat list of scalars takes the old replace-with-`[[date]]` path. Pinned by `test_date_hover.py::test_flat_scalar_customdata_is_replaced`. |
| 3 | `callbacks/custom_time.py:300, 379, 390` — `customdata=[[d] for d in d_scatter]` / `trace_customdata` | **No.** The Custom Time Axis figure is built in `callbacks/custom_time.py::_build_figure`; `_add_date_hover` appears nowhere under `btc_web/callbacks/`. | Unaffected. |
| 4 | `figures/common.py:900,902` — the `recovery=True` branch's own `[[d, r]]` rows | It *is* the helper. | Left exactly as it was; the only caller is `figures/bubble.py:702`, whose traces carry no pre-existing customdata. |
| 5 | `figures/occupancy.py` (new) | Yes — the only production consumer of the preserve path. | The point of the change. |

Uncertainty worth flagging: row 2 documents a **pre-existing** cosmetic defect
I did not touch — in that quantile selection the SC Mode B hover already
renders `%{customdata}` against a `[[date]]` row, i.e. an array. Out of scope,
but someone should file it.

## 2. What changed

**`figures/common.py`** — new `_date_customdata(existing, dates)`; the
non-recovery branch of `_add_date_hover` now calls it instead of writing
`[[d] for d in dates]` unconditionally. A trace with one row per point keeps
its values and the date is inserted at index 0, so `customdata[0]` is still
the date on every trace in the app and the trace's own values land at `[1:]`.
No customdata, a length mismatch, or flat scalar customdata all fall back to
the historical behaviour verbatim.

**`figures/occupancy.py`** — the single 5,725-point hover trace became two:

- full-window days carry `customdata = np.round(column_stack(above, below), 1)`
  and format it in the template (`%{customdata[1]:.1f}`, `%{customdata[2]:.1f}`);
- the days before the first full trailing window are their own trace with a
  fixed `window not yet full` template and no numbers at all;
- the `text` array (one pre-rendered label string per day) is gone;
- strip `x` is narrowed to **float32 after** `_apply_final_steps`, and the
  constant y rows are plain Python lists rather than `np.full`.

Both transport changes exist because **Dash serialises numpy arrays as base64
typed arrays** (`dash._utils.to_json` → `plotly.io.json.to_json_plotly`;
verified: `'"bdata"' in dash_to_json(fig)` is `True`). That has two
consequences the obvious optimisations get backwards:

- rounding a float64 array is **worthless** — base64 f8 is a flat 10.7 bytes
  per point whatever the digits are. Narrowing the dtype is what pays: f4 is
  5.3. (My first attempt rounded x to 5 dp and saved nothing on the wire.)
- a constant `np.full(n, 1.5)` costs **10.7** bytes/day, while the plain
  Python list `[1.5] * n` costs **4**. Here the numpy array is the pessimisation.

The float32 narrowing must run *after* `_add_date_hover`, which names each
point's calendar day from x: `t * 365.25` is an exact integer here, so a t
nudged **down** by 30 s falls back over midnight and every marker gets labelled
with the previous day. `test_strip_hover_dates_are_the_displayed_days` pins the
ordering — it failed loudly when I first wrote the check the naive way.

## 3. The dy=0 trap (the browser check earned its keep)

I first shipped the constant y rows as `y0=_STRIP_HOVER_Y, dy=0`, which is free
in JSON (two numbers instead of one per day) and looked right in every Python
assertion: the full suite was green and the figure was 22 KB smaller.

In the browser, hovering the strip did nothing at all. `gd.calcdata` showed the
reason: point 0 at y=1.5, point 1 at **y=2.5**, and the tick rows climbing 2, 3,
4… **plotly.js reads the step as `trace['d' + axLetter] || 1`**, so `dy = 0` is
falsy and silently becomes 1. Every marker but the first was ramping off the
top of the y2 strip and out of hover range. (This project's own CLAUDE.md has a
"Falsy-zero in callbacks" section; same shape, different language.)

Fixed by shipping plain constant lists. `test_strip_hover_payload_carries_numbers_not_rendered_text`
now asserts `t.dy is None and t.y0 is None` alongside the payload shape, so the
trap can't be re-entered quietly, and the comment in `occupancy.py` says why.

Note this was **not** detectable from Python: `y0`/`dy` are valid plotly.py
attributes, the figure validated, and the only symptom was in plotly.js's
calcdata.

## 4. Measurements

Default view params — `xmin=2011, xmax=2033, active_models=["bub"], occ_tail=10,
occ_window=4, sigma_mode="constant"`, remaining keys as in
`test_occupancy.py::_p`. 5,725 displayed days (4,432 with a full trailing
window, 1,293 before it).

Sizes are measured on **the encoder Dash actually ships** — `fig.to_json()`,
which base64-encodes numpy arrays — by locating the hover trace objects inside
the parsed figure JSON and re-serialising them compactly. `trace.to_json()`
does *not* base64-encode and therefore under-reports x and y; the brief's
suggested `len(json.dumps(trace.to_plotly_json(), default=str))` under-reports
them further (it renders a numpy array through `str()`, i.e. a truncated repr,
scoring 5,725 floats as 81 bytes). Numbers below are from the shipped form.

### Hover trace

| | BEFORE (HEAD) | AFTER |
|---|---|---|
| traces | 1 | 2 (full-window + pre-window) |
| points | 5,725 | 5,725 (unchanged) |
| `text` | 142,940 | — |
| `customdata` | 97,326 | 140,908 |
| `x` | 61,093 (f8 base64) | 30,586 (f4 base64) |
| `y` | 61,093 (f8 base64) | 22,902 (plain list) |
| **total raw** | **362,720** | **194,976** → **1.86×** |
| **total gzip -6** | **49,793** | **40,300** → **1.24×** |

### Whole figure

| | BEFORE | AFTER |
|---|---|---|
| `fig.to_json()` | 803,596 | 578,452 → 1.39× |
| gzip -6 | 162,079 | 144,663 → 1.12× |

The gzip ratios are much flatter than the raw ones, and that is the honest
number for wire bandwidth. The old `text` array was ~5,700 near-identical
strings and compressed ~18:1; nothing that replaces it can do as well. Two
smaller effects, measured: interleaving dates with numbers in one customdata
array costs **2,463 gzipped bytes** versus keeping them in separate homogeneous
runs (unavoidable — one `customdata` array, and the helper owns index 0), and
the float32 x is base64, i.e. incompressible, so it gzips at only 2.3:1.

### Why not 4×

Composition of the 194,976, measured not estimated:

| component | bytes |
|---|---|
| date strings | 85,877 |
| the two shares | 52,446 |
| `x` | 30,586 |
| `y` | 22,902 |
| trace scaffolding (names, templates, marker) | 3,165 |

4× of 362,720 is 90,680. The dates alone are 85,877 of it. **No arrangement of
the remaining fields fits**, so 4× requires giving up either the per-day
calendar date in the label or the per-day point. Both are frozen by the brief.
The reachable variants, for the controller to price:

| variant | bytes | ratio | what it costs |
|---|---|---|---|
| as shipped | 194,976 | 1.86× | — |
| + `x0`/`dx` instead of an x array | 164,390 | 2.21× | see below |
| + drop the date from the strip label | 109,099 | 3.32× | label text changes (the date is still one row below, on the line traces' own hover) |
| + both | 78,513 | 4.62× | both of the above |
| every 3rd day instead of daily | ≈65,000 | ≈5.6× | the reported date can be up to a day off the hovered pixel — though at the default range the strip already draws ~6 days per pixel |

On `x0`/`dx`: it is exactly correct here. The daily price grid is contiguous —
over all 5,893 historical days, `max |t − (t₀ + i·step)|` is **1.3e-12 days**
and there are **zero** gaps > 1.5 steps. I did not take it because
`test_occupancy.py::test_historical_only_no_future` reads `.x` on every strip
trace (`np.asarray(t.x, float).max()`), and that test is outside the ownership
the brief gave me. It is a 30,586-byte, four-line follow-up for whoever owns
that test — with a uniformity guard, since a future gap in the CSV would
silently displace every later marker.

## 5. Tests

New file `btc_web/test_date_hover.py`, 8 tests on the shared helper: rows
preserved with the date prepended; numeric rows stay numeric (a string here
renders `NaN` through a `%{...:.1f}` template); no customdata → `[[date]]`;
flat scalar customdata replaced (the supercharge case); length mismatch
replaced; `hoverinfo="skip"` untouched; out-of-range x untouched.

`test_occupancy.py` — the strip/hover tests keep their intent and were rewritten
for the new representation. `_hover(fig)` now returns both hover traces in x
order; `_hover_full` / `_hover_rows` are new helpers. Renamed
`test_strip_hover_text_matches_line_values` → `..._values_match_line_values`.
Two new tests: `test_strip_hover_payload_carries_numbers_not_rendered_text`
(no `text`; numeric customdata; plain-list y; the `dy`/`y0` guard) and
`test_strip_hover_dates_are_the_displayed_days` (every shipped date equals the
true calendar day of that price row — the float32-ordering regression guard).
No other test was touched.

```
btc_venv/bin/python3 -m pytest btc_web/test_occupancy.py btc_web/test_date_hover.py -q -p no:randomly -n0
  43 passed

btc_venv/bin/python3 -m pytest btc_web/test_occupancy.py btc_web/test_date_hover.py \
                               btc_web/test_percentile.py btc_web/test_figures.py -q -p no:randomly -n0
  225 passed

btc_venv/bin/python3 -m pytest -q
  1 failed, 2942 passed, 12 skipped
  FAILED btc_web/test_callbacks.py::TestBTCPayPricing::test_free_tier_all_models   (the known pre-existing failure)
```

## 6. Browser check

`PORT=8061 DEV=1 bash run_web.sh`, Playwright, viewport 1280×900,
`http://127.0.0.1:8061/1.5`. Real `page.mouse.move` over the y2 strip's own
drag rect — note the strip is a **separate `xy2` subplot** with its own
`.nsewdrag`, above the main panel's; my first attempts hovered the `xy` rect
and found nothing, which is a measurement artefact, not a defect.

Rendered labels (`.hoverlayer` text, `/` = line break):

| position | label |
|---|---|
| 62% across | `BM · Apr 27, 2015 / ≥Q90 8.1% · ≤Q10 18.4% / of trailing 4 yr` |
| 80% across | `BM · Apr 05, 2019 / ≥Q90 1.4% · ≤Q10 12.5% / of trailing 4 yr` |
| 5% across (pre-window) | `BM · Aug 29, 2010 / window not yet full / of trailing 4 yr` |

Same three-line shape and wording as before, with the numbers now assembled by
d3 from the numeric customdata. `≤Q10 12.0%` renders from a JSON `12` — d3's
`.1f` and Python's `:.1f` agree, and the values are pre-rounded to 1 dp so the
two cannot diverge (asserted).

`gd.calcdata` after the fix: tick rows flat at y=2 and y=1, both hover traces
flat at y=1.5, first point to last. A screenshot of the strip confirms the
ticks sit on their rows. The Dash dev-server error overlay shows
`citadel-graph.figure` callback errors and "nonexistent object … State"
warnings — the known DEV=1 lazy-tab noise, unrelated to this change.

## 7. Uncertain / worth a second look

1. **The gzip picture is much weaker than the raw one** (1.24× on the hover
   trace, 1.12× on the figure). If ALARA here means wire bytes, this change is
   worth ~9.5 KB per occupancy render, not ~168 KB. If it means browser memory,
   parse time and Dash diffing, the raw number is the right one. Both are
   above; I did not assume which the operator meant.
2. **The `x0`/`dx` follow-up** (§4) is blocked only by test ownership.
3. **The line traces are now the fat part of this figure** — each ships 4,432
   full-precision f8 x values (~47 KB base64) plus its own `[[date]]`
   customdata (~75 KB). Narrowing their x to float32 is the same trick for
   ~47 KB more, but `test_line_starts_one_window_after_first_price` asserts a
   1e-6 margin against a float32 epsilon of ~4.8e-7 — it would pass, tightly.
   I left it alone deliberately.
4. **Pre-existing SC Mode B customdata defect**, §1 row 2.

## 8. Diff

```diff
diff --git i/btc_web/figures/common.py w/btc_web/figures/common.py
index c66ecac..6da253c 100644
--- i/btc_web/figures/common.py
+++ w/btc_web/figures/common.py
@@ -862,6 +862,36 @@ def _compute_recovery(x_arr, y_arr, genesis=None):
     return result
 
 
+def _date_customdata(existing, dates):
+    """Build ``customdata`` rows for a date-hover trace, PRESERVING any rows
+    the trace already carries.
+
+    Historically this helper simply replaced ``customdata`` with
+    ``[[date], ...]``, so a trace could not ship its own per-point values —
+    they had to be smuggled through ``text`` as a pre-formatted string, which
+    is several times larger on the wire than the numbers behind it.
+
+    A trace that already has one *row* (list/tuple/array) per point keeps it:
+    the date is inserted at index 0, so ``customdata[0]`` is the date on every
+    trace as before and the trace's own values land at ``customdata[1:]``.
+    Anything else — no customdata, a length mismatch, or flat scalar
+    customdata (e.g. supercharge Mode B's quantile labels) — gets the plain
+    ``[[date], ...]`` of the original behaviour.
+    """
+    if existing is None:
+        return [[d] for d in dates]
+    try:
+        if len(existing) != len(dates):
+            return [[d] for d in dates]
+    except TypeError:
+        return [[d] for d in dates]
+    rows = list(existing)
+    if not all(isinstance(r, (list, tuple, np.ndarray)) for r in rows):
+        return [[d] for d in dates]
+    return [[d, *(r.tolist() if isinstance(r, np.ndarray) else list(r))]
+            for d, r in zip(dates, rows)]
+
+
 def _add_date_hover(fig, genesis, fmt=None, recovery=False):
     """Add calendar-date hover to all traces whose x-axis is t (years since genesis).
 
@@ -870,6 +900,10 @@ def _add_date_hover(fig, genesis, fmt=None, recovery=False):
     supercharge uses delay-years or quantile fractions as x).
 
     recovery=True: compute price recovery time and append to hover.
+
+    A trace that already carries one customdata ROW per point keeps it — the
+    date is prepended, so customdata[0] is the date and the trace's own values
+    shift to customdata[1:].  See _date_customdata.
     """
     if fmt is None:
         fmt = _HOVER_FMT_USD
@@ -905,7 +939,8 @@ def _add_date_hover(fig, genesis, fmt=None, recovery=False):
             elif getattr(trace, "hovertemplate", None) == fmt:
                 trace.hovertemplate = _fmt_with_rec
         else:
-            trace.customdata = [[d] for d in dates]
+            trace.customdata = _date_customdata(
+                getattr(trace, "customdata", None), dates)
             if not getattr(trace, "hovertemplate", None):
                 trace.hovertemplate = fmt
 
diff --git i/btc_web/figures/occupancy.py w/btc_web/figures/occupancy.py
index 958bc25..9c448ec 100644
--- i/btc_web/figures/occupancy.py
+++ w/btc_web/figures/occupancy.py
@@ -145,7 +145,13 @@ def build_occupancy_figure(m: ModelData, p: dict[str, Any]) -> go.Figure:
             ):
                 days = t_all[cond & disp]
                 traces.append(go.Scatter(
-                    x=days, y=np.full(days.shape, y_row),
+                    # A plain Python list of the constant row, NOT np.full:
+                    # Dash base64-encodes numpy arrays, and "1.0," repeated is
+                    # 4 bytes a day against float64's 10.7.  y0/dy would be
+                    # free, but plotly.js reads dy as `trace.dy || 1`, so
+                    # dy=0 silently becomes a step of 1 and the ticks climb
+                    # off the strip.
+                    x=days, y=[y_row] * days.shape[0],
                     mode="markers", yaxis="y2",
                     name=f"{name} days {lbl}", showlegend=False,
                     marker=dict(symbol="line-ns", size=_STRIP_MARKER_SIZE,
@@ -155,27 +161,49 @@ def build_occupancy_figure(m: ModelData, p: dict[str, Any]) -> go.Figure:
                     hoverinfo="skip",
                 ))
             # Invisible marker on EVERY displayed day: hovering anywhere along
-            # the bar reports the trailing-window shares at that date. Values
-            # ride on `text` because _add_date_hover overwrites customdata.
+            # the bar reports the trailing-window shares at that date.
+            #
+            # Bandwidth: the two shares ride as NUMBERS on customdata (index
+            # 1/2 — _add_date_hover prepends the date at index 0) and are
+            # formatted by the hovertemplate, instead of one pre-formatted
+            # label string per day.  The days before the first full window
+            # carry no numbers at all, so they get their own trace with a
+            # fixed template.  Rendered label text is unchanged.
             t_disp = t_all[disp]
-            pos = np.searchsorted(t_out, t_disp)
-            hover_text = []
-            for k in range(t_disp.shape[0]):
-                j = pos[k]
-                if j < t_out.shape[0] and abs(t_out[j] - t_disp[k]) < 1e-9:
-                    hover_text.append(f"≥Q{q_hi} {above[j]:.1f}% · ≤Q{tail} {below[j]:.1f}%")
-                else:
-                    hover_text.append("window not yet full")
-            traces.append(go.Scatter(
-                x=t_disp, y=np.full(t_disp.shape, _STRIP_HOVER_Y),
+            n_out = t_out.shape[0]
+            if n_out:
+                pos = np.searchsorted(t_out, t_disp)
+                pos_c = np.clip(pos, 0, n_out - 1)
+                hit = (pos < n_out) & (np.abs(t_out[pos_c] - t_disp) < 1e-9)
+            else:
+                pos = np.zeros(t_disp.shape, dtype=int)
+                hit = np.zeros(t_disp.shape, dtype=bool)
+            hover_kw = dict(
                 mode="markers", yaxis="y2",
                 name=f"{name} trailing {window:g} yr", showlegend=False,
                 # Invisible, but the hover label takes the model colour.
                 marker=dict(size=_STRIP_MARKER_SIZE, color=color, opacity=0),
-                text=hover_text,
-                hovertemplate=(f"<b>{name}</b> · %{{customdata[0]}}<br>%{{text}}"
-                               f"<br>of trailing {window:g} yr<extra></extra>"),
-            ))
+            )
+            n_hit = int(hit.sum())
+            if n_hit:
+                traces.append(go.Scatter(
+                    x=t_disp[hit], y=[_STRIP_HOVER_Y] * n_hit,
+                    customdata=np.round(np.column_stack(
+                        (above[pos[hit]], below[pos[hit]])), 1),
+                    hovertemplate=(
+                        f"<b>{name}</b> · %{{customdata[0]}}<br>"
+                        f"≥Q{q_hi} %{{customdata[1]:.1f}}% · "
+                        f"≤Q{tail} %{{customdata[2]:.1f}}%"
+                        f"<br>of trailing {window:g} yr<extra></extra>"),
+                    **hover_kw))
+            n_miss = int(t_disp.shape[0] - n_hit)
+            if n_miss:
+                traces.append(go.Scatter(
+                    x=t_disp[~hit], y=[_STRIP_HOVER_Y] * n_miss,
+                    hovertemplate=(f"<b>{name}</b> · %{{customdata[0]}}"
+                                   f"<br>window not yet full"
+                                   f"<br>of trailing {window:g} yr<extra></extra>"),
+                    **hover_kw))
 
     if traces:
         traces.append(go.Scatter(
@@ -242,4 +270,15 @@ def build_occupancy_figure(m: ModelData, p: dict[str, Any]) -> go.Figure:
                    f"%{{y:.1f}}% of trailing {window:g} yr<extra></extra>"),
         show_qr=False, show_mc=False, wm_pos="bottom-right",
     )
+    # Transport-only.  Dash serialises numpy arrays as base64 typed arrays, so
+    # a float64 x costs a flat 10.7 bytes/day no matter how round the numbers
+    # are — float32 halves that.  Its worst-case error over this t range is
+    # ~1e-6 yr (a third of a second) against a 1-day point spacing and a strip
+    # drawn at ~6 days per pixel, so it can neither move a marker visibly nor
+    # change which day a hover snaps to.  It has to happen AFTER
+    # _apply_final_steps: _add_date_hover names each point's calendar day from
+    # x, and a t nudged DOWNWARD would fall back over midnight.
+    for tr in fig.data:
+        if getattr(tr, "yaxis", None) == "y2" and tr.x is not None:
+            tr.x = np.asarray(tr.x, dtype=np.float32)
     return fig
diff --git i/btc_web/test_occupancy.py w/btc_web/test_occupancy.py
index fb39f37..476e199 100644
--- i/btc_web/test_occupancy.py
+++ w/btc_web/test_occupancy.py
@@ -44,11 +44,31 @@ def _ticks(fig):
 
 
 def _hover(fig):
+    """The invisible full-coverage hover traces, in x order.
+
+    Two of them: the days with a full trailing window (numeric customdata) and
+    the days before it (date only).  Either may be absent.
+    """
     h = [t for t in _strip(fig) if t.marker.opacity == 0]
+    return sorted(h, key=lambda t: float(np.asarray(t.x, float).min()))
+
+
+def _hover_full(fig):
+    """The hover trace carrying the two shares (customdata is 3 wide)."""
+    h = [t for t in _hover(fig) if t.customdata is not None
+         and len(t.customdata[0]) == 3]
     assert len(h) <= 1
     return h[0] if h else None
 
 
+def _hover_rows(trace):
+    """(t, date, above, below) per point; above/below are None pre-window."""
+    x = np.asarray(trace.x, float)
+    return [(x[i], r[0], (r[1] if len(r) > 1 else None),
+             (r[2] if len(r) > 2 else None))
+            for i, r in enumerate(trace.customdata)]
+
+
 def _daily_t(years):
     return np.arange(0.0, years, 1.0 / 365.25)
 
@@ -178,8 +198,9 @@ class TestOccupancyFigure:
     def test_strip_marks_first_model_only(self):
         fig = build_occupancy_figure(M, _p(active_models=["bub", "qr"]))
         strip = _strip(fig)
-        assert len(strip) == 3                    # 2 tick rows + 1 hover trace
-        assert len(_ticks(fig)) == 2
+        assert len(strip) == 4        # 2 tick rows + 2 hover traces (pre/post
+        assert len(_ticks(fig)) == 2  # the first full trailing window)
+        assert len(_hover(fig)) == 2
         assert all(t.yaxis == "y2" for t in strip)
         assert all("BM" in t.name for t in strip)
         assert all(t.showlegend is False for t in strip)
@@ -192,39 +213,82 @@ class TestOccupancyFigure:
 
     def test_strip_hover_trace_covers_every_displayed_day(self):
         fig = build_occupancy_figure(M, _p(xmin=2016, active_models=["bub"]))
-        h = _hover(fig)
-        assert h is not None and h.yaxis == "y2"
+        hs = _hover(fig)
+        assert hs and all(t.yaxis == "y2" for t in hs)
         from btc_core import yr_to_t
         td = today_t(M.genesis)
         t_lo = yr_to_t(2016, M.genesis)
         n_days = int(((M.price_years >= t_lo) & (M.price_years <= td)).sum())
         # covers every displayed day (2016-01-01 .. today), not just tail days
-        assert len(h.x) == n_days
-        assert len(h.text) == len(h.x)
-        assert "%{text}" in h.hovertemplate and "%{customdata[0]}" in h.hovertemplate
+        assert sum(len(t.x) for t in hs) == n_days
+        for t in hs:
+            assert len(t.customdata) == len(t.x)
+            assert "%{customdata[0]}" in t.hovertemplate
 
-    def test_strip_hover_text_matches_line_values(self):
+    def test_strip_hover_payload_carries_numbers_not_rendered_text(self):
+        # Bandwidth: the label is assembled by the hovertemplate from numeric
+        # customdata, not shipped as one pre-formatted string per day.  The
+        # constant y2 row must stay a plain Python sequence — Dash base64s a
+        # numpy array, which costs 10.7 bytes/day instead of 4.  (y0/dy would
+        # be free but plotly.js reads dy as `trace.dy || 1`, so dy=0 becomes a
+        # step of 1 and the markers ramp off the strip.)
+        fig = build_occupancy_figure(M, _p(active_models=["bub"]))
+        for t in _strip(fig):
+            assert not isinstance(t.y, np.ndarray)
+            assert set(np.asarray(t.y, float)) <= {1.0, 1.5, 2.0}
+            assert t.dy is None and t.y0 is None
+        for t in _hover(fig):
+            assert t.text is None
+            assert len(t.y) == len(t.x)
+        h = _hover_full(fig)
+        assert "%{customdata[1]:.1f}" in h.hovertemplate
+        assert "%{customdata[2]:.1f}" in h.hovertemplate
+        assert all(isinstance(r[1], float) and isinstance(r[2], float)
+                   for r in h.customdata)
+
+    def test_strip_hover_values_match_line_values(self):
         fig = build_occupancy_figure(M, _p(active_models=["bub"], occ_tail=10))
-        h = _hover(fig)
+        h = _hover_full(fig)
         above = [t for t in _lines(fig) if "≥" in t.name][0]
         below = [t for t in _lines(fig) if "≤" in t.name][0]
         x_last = float(np.asarray(above.x, float)[-1])
         i = int(np.argmin(np.abs(np.asarray(h.x, float) - x_last)))
-        txt = h.text[i]
-        a = float(re.search(r"≥Q90 ([\d.]+)%", txt).group(1))
-        b = float(re.search(r"≤Q10 ([\d.]+)%", txt).group(1))
+        _, _, a, b = _hover_rows(h)[i]
         assert abs(a - float(above.y[-1])) < 0.06
         assert abs(b - float(below.y[-1])) < 0.06
+        # Shipped already rounded to 1 dp, so d3's ".1f" in the hovertemplate
+        # renders exactly the string Python's ":.1f" used to bake into `text`.
+        assert all(v == round(v, 1) for _t, _d, v, _b in _hover_rows(h))
+        assert all(v == round(v, 1) for _t, _d, _a, v in _hover_rows(h))
+
+    def test_strip_hover_dates_are_the_displayed_days(self):
+        # _add_date_hover names each point's day from x, and the transport-only
+        # float32 narrowing of x runs AFTER it — done in the other order, a t
+        # nudged down by 30 s falls back over midnight and every marker is
+        # labelled with the previous day.
+        from figures.common import _t_to_datestr
+        from btc_core import yr_to_t
+        fig = build_occupancy_figure(M, _p(xmin=2016, active_models=["bub"]))
+        t_lo, td = yr_to_t(2016, M.genesis), today_t(M.genesis)
+        days = np.sort(M.price_years[(M.price_years >= t_lo)
+                                     & (M.price_years <= td)])
+        want = [_t_to_datestr(t, M.genesis) for t in days]
+        got = [r[0] for h in _hover(fig) for r in h.customdata]
+        assert got == want
 
     def test_strip_hover_before_full_window_says_so(self):
         fig = build_occupancy_figure(M, _p(xmin=2010, active_models=["bub"], occ_window=4))
-        h = _hover(fig)
+        early, late = _hover(fig)            # x-ordered: pre-window first
         first_line_t = float(np.asarray([t.x for t in _lines(fig)][0]).min())
-        hx = np.asarray(h.x, float)
-        early = [h.text[i] for i in range(len(hx)) if hx[i] < first_line_t - 1e-9]
-        assert early and all("not yet full" in t for t in early)
-        late = [h.text[i] for i in range(len(hx)) if hx[i] >= first_line_t]
-        assert late and all("≥Q90" in t for t in late)
+        assert "window not yet full" in early.hovertemplate
+        assert "%{customdata[1]" not in early.hovertemplate
+        assert all(len(r) == 1 for r in early.customdata)   # date only
+        assert np.asarray(early.x, float).max() < first_line_t
+        assert "≥Q90 %{customdata[1]:.1f}%" in late.hovertemplate
+        assert np.asarray(late.x, float).min() >= first_line_t - 1e-4
+        # Same trailing clause on both, as before.
+        for t in (early, late):
+            assert t.hovertemplate.endswith("<br>of trailing 4 yr<extra></extra>")
 
     def test_strip_days_are_the_tail_days(self):
         # For QR (quantile regression on ALL data) the share of days above Q90
```
