# Phase 2 — `/3` (DCA) Fast Modal Close Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the bubble fast-modal-close pattern (Phase 1, commit `b4895ec`) to `/3` (DCA) share links so the modal closes in ~3-5 s instead of falling back to the 7 s timer.

**Architecture:** Extend `restore_builder.py` with `_build_dca_figure_from_state(state)` that calls `_get_dca_fig` directly. Route the figure through a new always-mounted `dcc.Store("restore-dca-fig")` + clientside `set_props` relay (mirrors Phase 1's `restore-bubble-fig` pattern). Add `update_dca` post-restore short-circuit to suppress phantom rebuilds. MC-enabled and Saylor-live DCA shares fall back to the existing 7 s timer (option a in spec).

**Tech Stack:** Python 3.14 (dev) / 3.12 (prod), Plotly Dash 4.0.0, `dcc.Store`, clientside_callback, `set_props`, Playwright (Firefox) for E2E, gunicorn 5-worker prod.

**Spec:** `docs/superpowers/specs/2026-04-25-restore-dca-fast-modal-close-design.md` (architect-approved, all 10 prior review issues resolved).

**Phase 1 precedent:** `docs/superpowers/plans/2026-04-25-restore-from-url-dispatch-fix-plan.md`. Mirrors its structure (single commit at end, CHECKPOINT before prod, prod-verify Playwright probe).

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `btc_web/layout/__init__.py` | modify | Add 2 always-mounted Stores: `restore-dca-fig`, `dca-build-count` |
| `btc_web/restore_builder.py` | modify | Add `_build_dca_figure_from_state(state)` (~150 lines, mirrors `_build_bubble_figure_from_state`) |
| `btc_web/callbacks/snapshot_cb.py` | modify | `restore_from_url`: 6th Output + DCA branch; 2 new clientside callbacks (figure relay + build counter) |
| `btc_web/callbacks/charts/__init__.py` | modify | `update_dca`: 2 new States + post-restore short-circuit |
| `btc_web/test_restore_builder.py` | extend | 5 unit tests for `_build_dca_figure_from_state` |
| `btc_web/test_restore_phase2_dca_e2e.py` | create | 6 Playwright E2E tests (DCA fast restore, MC fallback, Saylor-live fallback, no phantom rebuild via Store, yr-range correctness, /1 regression) |
| `memory/restore_callback_architecture.md` | modify (in `~/.claude/projects/.../memory/`) | Add Phase 2 entry after Phase 1 section |
| `docs/architecture.md` | modify | Update restore-architecture section to list `/3` as fast-path |

---

## Task 1: Add `restore-dca-fig` and `dca-build-count` Stores to layout

**Files:**
- Modify: `btc_web/layout/__init__.py:299-300`

- [ ] **Step 1: Edit layout to add the two new Stores**

Find the existing `dcc.Store(id="restore-bubble-fig", ...)` line (added in Phase 1). Insert the two new Stores immediately after it:

```python
    dcc.Store(id="restore-bubble-fig", storage_type="memory", data=None),
    # Phase 2 (2026-04-25): DCA figure delivery via Store + set_props relay,
    # same pattern as restore-bubble-fig. dca-graph is inside dca-lazy on /1
    # /2/4/5/6/7 initial loads — directly outputting to dca-graph.figure
    # would re-introduce the Phase 1 lazy-Output dispatch-drop bug.
    dcc.Store(id="restore-dca-fig", storage_type="memory", data=None),
    # Phase 2 phantom-rebuild detector: clientside callback below increments
    # this on dca-graph.figure mutation. E2E test 9 reads it via page.evaluate
    # and asserts count==1 after restore (single delivery via relay; >=2 means
    # the post-restore guard failed and the cascade rebuilt the figure).
    dcc.Store(id="dca-build-count", storage_type="memory", data=0),
```

- [ ] **Step 2: Syntax check**

Run:
```bash
cd /scratch/code/bitcoinprojections && PYTHONPATH="btc_web:." btc_venv/bin/python3 -c "
import sys; sys.path.insert(0, 'btc_web')
import os; os.environ['DEV'] = '1'
import app  # initializes _app_ctx
import layout
print('OK')
"
```

Expected: `OK` (no ImportError, no SyntaxError, no Dash component-id collision).

---

## Task 2: Implement `_build_dca_figure_from_state` in `restore_builder.py`

**Files:**
- Modify: `btc_web/restore_builder.py` (append after `_build_bubble_figure_from_state`)

- [ ] **Step 1: Append the new helper**

Add at the end of `btc_web/restore_builder.py`:

```python
def _build_dca_figure_from_state(state: dict):
    """Build a DCA-tab Plotly figure from a snapshot state dict.

    Mirrors `update_dca` (callbacks/charts/__init__.py:958) param construction
    line-for-line, but reads from the decoded snapshot dict instead of widget
    Inputs.

    Returns:
        - go.Figure on the standard fast path
        - None when MC is enabled (snapshot has dca-mc-enable=["yes"]) — caller
          falls back to the existing chart-callback path (Phase 2 doesn't
          replicate the MC pipeline; modal closes via 7 s timer for those).
        - None when Saylor-live mode is active (sc-enable + sc-entry-mode=
          "live") — builder has no live BTC price source. Caller falls back.

    Critical: the builder must NOT attempt to read btc-price-store from
    state. There is no such key in snapshots. sc_live_price is passed as 0
    (unused on non-live SC paths and on non-SC paths).
    """
    # ── Gates: return None for MC and Saylor-live, fall back to cascade ──
    mc_enable = _v(state, "dca-mc-enable", default=[]) or []
    if "yes" in mc_enable:
        return None
    sc_enable = _v(state, "dca-sc-enable", default=[]) or []
    sc_entry_mode = _v(state, "dca-sc-entry-mode", default="live")
    if "yes" in sc_enable and sc_entry_mode == "live":
        return None

    # ── DCA widget values (Inputs) ──
    stack    = _v(state, "dca-stack",    default=0.0)
    use_lots = _v(state, "dca-use-lots", default=[])
    amount   = _v(state, "dca-amount",   default=100)
    freq     = _v(state, "dca-freq",     default="Monthly")
    dca_infl = _v(state, "dca-infl",     default=0.0)
    yr_range = _v(state, "dca-yr-range", default=[2024, 2034])
    disp     = _v(state, "dca-disp",     default="btc")
    toggles  = _v(state, "dca-toggles",  default=[])
    legend_pos = _v(state, "dca-legend-pos", default="outside")
    sel_qs   = _v(state, "dca-qs",       default=[0.5])
    adv_qs   = _v(state, "dca-qs-adv",   default=[])
    qs_mode  = _v(state, "dca-qs-mode",  default=[])  # State at line 953
    model_show = _v(state, "dca-model-show", default=[])

    # ── Saylor-mode (Stack-celerator) widget values ──
    sc_loan        = _v(state, "dca-sc-loan",        default=0.0)
    sc_rate        = _v(state, "dca-sc-rate",        default=8.0)
    sc_term        = _v(state, "dca-sc-term",        default=120)
    sc_type        = _v(state, "dca-sc-type",        default="interest_only")
    sc_repeats     = _v(state, "dca-sc-repeats",     default=0)
    sc_custom_price = _v(state, "dca-sc-custom-price", default=70000.0)
    sc_tax         = _v(state, "dca-sc-tax",         default=33.0)
    sc_rollover    = _v(state, "dca-sc-rollover",    default=False)

    # ── Lots resolution (clientside cascade replicated, mirrors bubble) ──
    _lots = state.get("_lots") or []
    use_lots_bool = bool("yes" in (use_lots or []))

    # ── Shared model-config States (LPPL/HybPPL/EPPL) ──
    lppl_n_freqs  = _v(state, "lppl-n-freqs",  default=[])
    lppl_weighted = _v(state, "lppl-weighted", default=[])
    lppl_no_13    = _v(state, "lppl-no-13",    default=[])

    hyb_a_nlog  = _v(state, "hybppl-cfg-a-nlog",  default=1)
    hyb_a_ncal  = _v(state, "hybppl-cfg-a-ncal",  default=1)
    hyb_a_log1d = _v(state, "hybppl-cfg-a-log1d", default="d")
    hyb_a_log2d = _v(state, "hybppl-cfg-a-log2d", default="d")
    hyb_a_cal1d = _v(state, "hybppl-cfg-a-cal1d", default="u")
    hyb_a_cal2d = _v(state, "hybppl-cfg-a-cal2d", default="u")

    hyb_b_enabled = _v(state, "hybppl-cfg-b-enabled", default=[])
    hyb_b_nlog  = _v(state, "hybppl-cfg-b-nlog",  default=0)
    hyb_b_ncal  = _v(state, "hybppl-cfg-b-ncal",  default=0)
    hyb_b_log1d = _v(state, "hybppl-cfg-b-log1d", default="d")
    hyb_b_log2d = _v(state, "hybppl-cfg-b-log2d", default="d")
    hyb_b_cal1d = _v(state, "hybppl-cfg-b-cal1d", default="u")
    hyb_b_cal2d = _v(state, "hybppl-cfg-b-cal2d", default="u")

    ep_a_nlog  = _v(state, "eppl-cfg-a-nlog",  default=1)
    ep_a_ncal  = _v(state, "eppl-cfg-a-ncal",  default=1)
    ep_a_log1d = _v(state, "eppl-cfg-a-log1d", default="d")
    ep_a_log2d = _v(state, "eppl-cfg-a-log2d", default="d")
    ep_a_cal1d = _v(state, "eppl-cfg-a-cal1d", default="u")
    ep_a_cal2d = _v(state, "eppl-cfg-a-cal2d", default="u")

    ep_b_enabled = _v(state, "eppl-cfg-b-enabled", default=[])
    ep_b_nlog  = _v(state, "eppl-cfg-b-nlog",  default=0)
    ep_b_ncal  = _v(state, "eppl-cfg-b-ncal",  default=0)
    ep_b_log1d = _v(state, "eppl-cfg-b-log1d", default="d")
    ep_b_log2d = _v(state, "eppl-cfg-b-log2d", default="d")
    ep_b_cal1d = _v(state, "eppl-cfg-b-cal1d", default="u")
    ep_b_cal2d = _v(state, "eppl-cfg-b-cal2d", default="u")

    palette_key = _v(state, "palette-store", "data", default="default")
    user_model_store = state.get("user-model-store:data")

    # ── Master-key resolution (mirrors update_dca) ──
    from callbacks.charts._resolvers import (
        _resolve_lppl_master, _resolve_hybppl_master, _resolve_eppl_master,
    )
    from callbacks.coerce import _ci, _cf
    model_show = list(model_show or [])
    model_show = _resolve_lppl_master(
        model_show, lppl_n_freqs, lppl_weighted, lppl_no_13)
    model_show = _resolve_hybppl_master(
        model_show,
        hyb_a_nlog, hyb_a_ncal, hyb_a_log1d, hyb_a_log2d, hyb_a_cal1d, hyb_a_cal2d,
        hyb_b_enabled, hyb_b_nlog, hyb_b_ncal, hyb_b_log1d, hyb_b_log2d,
        hyb_b_cal1d, hyb_b_cal2d)
    model_show = _resolve_eppl_master(
        model_show,
        ep_a_nlog, ep_a_ncal, ep_a_log1d, ep_a_log2d, ep_a_cal1d, ep_a_cal2d,
        ep_b_enabled, ep_b_nlog, ep_b_ncal, ep_b_log1d, ep_b_log2d,
        ep_b_cal1d, ep_b_cal2d)

    # ── Effective quantiles (mirrors update_dca:983-984) ──
    yr_range = yr_range or [2024, 2034]
    toggles = toggles or []
    _advanced = "advanced" in (qs_mode or [])
    _effective_qs = (adv_qs or []) if _advanced else (
        _bands_to_qs(sel_qs) if sel_qs and isinstance(sel_qs[0], str) else (sel_qs or []))

    # ── Build figure via _get_dca_fig. mc_enabled=False is sufficient;
    # _get_mc_or_cached strips all mc_* keys before quantizing the cache key
    # (utils.py:166-168). No _mc_setup, no mc_p stub needed. ──
    from utils import _get_dca_fig
    from tab_defaults import DCA
    try:
        params = dict(
            start_stack    = _cf(stack, DCA["start_stack"]),
            use_lots       = use_lots_bool,
            amount         = _ci(amount, DCA["amount"]),
            freq           = freq or "Monthly",
            inflation      = _cf(dca_infl, DCA["inflation"]),
            start_yr       = int(yr_range[0]),
            end_yr         = int(yr_range[1]),
            disp_mode      = disp or "btc",
            log_y          = "log_y"      in toggles,
            annotate       = "annotate"   in toggles,
            discrete       = "discrete"   in toggles,
            shade          = "shade"      in toggles,
            show_today     = "show_today" in toggles,
            show_legend    = "show_legend" in toggles,
            legend_pos     = legend_pos or "outside",
            minor_grid     = "minor_grid" in toggles,
            selected_qs    = _effective_qs,
            lots           = _lots,
            sc_enabled     = bool("yes" in (sc_enable or [])),
            sc_loan_amount = _cf(sc_loan, 0),
            sc_rate        = _cf(sc_rate, DCA["sc_rate"]),
            sc_loan_type   = sc_type or "interest_only",
            sc_term_months = _cf(sc_term, DCA["sc_term_months"]),
            sc_repeats     = _ci(sc_repeats, 0),
            sc_live_price   = 0,  # builder has no live source — gated above for sc-live
            sc_entry_mode   = sc_entry_mode or "live",
            sc_custom_price = _cf(sc_custom_price, DCA["sc_custom_price"]),
            sc_tax_rate     = _cf(sc_tax, 33, lo=0, hi=100) / 100.0,
            sc_rollover     = bool(sc_rollover),
            show_qr        = "bub" in model_show,
            show_mc        = False,
            active_models  = [k for k in model_show if k != "mc"],
            palette        = palette_key or "default",
            user_model     = user_model_store,
            mc_enabled     = False,  # sufficient on its own — see utils.py:166-168
        )
        result = _get_dca_fig(params)
        fig = result[0] if isinstance(result, tuple) else result
        return fig
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(
            "_build_dca_figure_from_state failed: %s; caller will fall back", e)
        return None
```

- [ ] **Step 2: Syntax check**

Run:
```bash
cd /scratch/code/bitcoinprojections && PYTHONPATH="btc_web:." btc_venv/bin/python3 -c "
import sys; sys.path.insert(0, 'btc_web')
import os; os.environ['DEV'] = '1'
import app
from restore_builder import _build_bubble_figure_from_state, _build_dca_figure_from_state
print('OK')
"
```

Expected: `OK`.

---

## Task 3: Add 5 unit tests for the DCA builder

**Files:**
- Modify: `btc_web/test_restore_builder.py` (append after existing bubble tests)

- [ ] **Step 1: Append unit tests**

Append to `btc_web/test_restore_builder.py`:

```python
# ════════════════════════════════════════════════════════════════════════════
# Phase 2 (2026-04-25): _build_dca_figure_from_state tests.
# ════════════════════════════════════════════════════════════════════════════

class TestBuildDcaFigureFromState:
    def test_dca_basic_returns_figure(self):
        """Minimal state — builder returns a figure with ≥1 quantile trace."""
        from restore_builder import _build_dca_figure_from_state
        state = {
            "main-tabs:active_tab": "dca",
            "dca-amount:value": 100,
            "dca-yr-range:value": [2024, 2034],
            "dca-qs:value": [0.5],
        }
        fig = _build_dca_figure_from_state(state)
        assert fig is not None, "builder returned None for non-MC non-SC-live state"
        # Convert to dict if Figure object
        fig_dict = fig.to_dict() if hasattr(fig, "to_dict") else fig
        assert "data" in fig_dict
        # At least one trace whose name matches quantile pattern (e.g., "Q50%", "Q1%")
        import re
        q_re = re.compile(r"Q\d")
        has_q = any(q_re.search(str(t.get("name", ""))) for t in fig_dict["data"])
        assert has_q, (
            f"no quantile trace in figure data: "
            f"{[t.get('name') for t in fig_dict['data']]}"
        )

    def test_dca_mc_enabled_returns_none(self):
        """dca-mc-enable=['yes'] (decoded list, not bitmask int) — return None."""
        from restore_builder import _build_dca_figure_from_state
        state = {
            "main-tabs:active_tab": "dca",
            "dca-mc-enable:value": ["yes"],
            "dca-amount:value": 100,
        }
        fig = _build_dca_figure_from_state(state)
        assert fig is None, "MC-enabled snapshot must fall back to cascade path"

    def test_dca_sc_live_returns_none(self):
        """sc-enable + sc-entry-mode=live — return None, no exception, no
        attempt to resolve btc-price-store from state."""
        from restore_builder import _build_dca_figure_from_state
        state = {
            "main-tabs:active_tab": "dca",
            "dca-sc-enable:value": ["yes"],
            "dca-sc-entry-mode:value": "live",
            "dca-amount:value": 100,
        }
        # Must not raise — even though btc-price-store key is absent.
        fig = _build_dca_figure_from_state(state)
        assert fig is None, "Saylor-live snapshot must fall back to cascade path"

    def test_dca_sc_custom_returns_figure(self):
        """sc-enable + sc-entry-mode=custom + custom price — returns figure."""
        from restore_builder import _build_dca_figure_from_state
        state = {
            "main-tabs:active_tab": "dca",
            "dca-sc-enable:value": ["yes"],
            "dca-sc-entry-mode:value": "custom",
            "dca-sc-custom-price:value": 50000.0,
            "dca-sc-loan:value": 100000.0,
            "dca-amount:value": 100,
        }
        fig = _build_dca_figure_from_state(state)
        assert fig is not None, "SC-custom snapshot should produce a figure"

    def test_dca_with_lots_returns_figure(self):
        """Snapshot with _lots + dca-use-lots=['yes'] — figure builds without
        error (lots-resolution mirrors bubble builder)."""
        from restore_builder import _build_dca_figure_from_state
        state = {
            "main-tabs:active_tab": "dca",
            "dca-use-lots:value": ["yes"],
            "dca-amount:value": 100,
            "_lots": [
                {"date": "2020-01-01", "btc": 0.5, "price_usd": 7000.0,
                 "fee_pct": 0.0, "label": "test", "uuid": "lot-1"},
            ],
        }
        fig = _build_dca_figure_from_state(state)
        assert fig is not None
```

- [ ] **Step 2: Run unit tests**

Run:
```bash
cd /scratch/code/bitcoinprojections && \
  btc_venv/bin/python3 -m pytest btc_web/test_restore_builder.py::TestBuildDcaFigureFromState -v
```

Expected: `5 passed`.

If any test fails: read the error, check the corresponding state key in the builder. Most likely cause is a typo in a `_v(state, "dca-X")` key not matching `_SNAPSHOT_CONTROLS` format (`{cid}:{prop}`).

---

## Task 4: Modify `restore_from_url` for 6th Output and DCA branch

**Files:**
- Modify: `btc_web/callbacks/snapshot_cb.py:44-100`

- [ ] **Step 1: Add 6th Output**

Edit the `@callback` decorator at line 44-51 to add the new Output:

```python
@callback(
    Output("snapshot-state-store", "data"),
    Output("loaded-hash-store",    "data"),
    Output("snapshot-pending",     "data", allow_duplicate=True),
    Output("restore-bubble-fig",   "data", allow_duplicate=True),
    Output("active-chart-committed","data",   allow_duplicate=True),
    Output("restore-dca-fig",      "data", allow_duplicate=True),
    Input("url", "hash"),
    prevent_initial_call='initial_duplicate',
)
```

- [ ] **Step 2: Update both early-exit returns to 6-tuples**

Find the two early-exit `return` statements at lines 62 and 67. Both currently return 5 `no_update`s. Change to 6:

Line 62:
```python
    if not hash_str:
        return no_update, no_update, no_update, no_update, no_update, no_update
```

Line 67:
```python
    if not state:
        logger.warning("Snapshot decode failed for hash: %s…", hash_str[:20])
        return no_update, no_update, no_update, no_update, no_update, no_update
```

- [ ] **Step 3: Initialize new variable + add DCA branch**

Find the existing bubble-only block at lines 79-99. After the closing of that block (the `_committed_out = hash_str` etc. line ~97-99), add a parallel DCA branch and a new variable:

```python
    # Build the active tab's figure server-side. Bubble path (Phase 1) and
    # DCA path (Phase 2) handle their own builders. For other tabs (or when
    # the builder returns None — e.g. CTA-active for bubble, MC-enabled or
    # Saylor-live for DCA), fall back to the existing callback path.
    _fig_out = no_update
    _committed_out = no_update
    _dca_out = no_update
    active_tab = state.get("main-tabs:active_tab", "bubble")
    if active_tab == "bubble":
        _t1 = _time.perf_counter()
        try:
            from restore_builder import _build_bubble_figure_from_state
            fig = _build_bubble_figure_from_state(state)
        except Exception as e:
            logger.warning("restore_builder failed: %s; falling back to "
                           "callback path", e)
            fig = None
        if fig is not None:
            _fig_out = fig
            _committed_out = hash_str
            print(f"[trace] restore-direct-build BUILT "
                  f"{(_time.perf_counter() - _t1) * 1000:.1f}ms", flush=True)
    elif active_tab == "dca":
        _t1 = _time.perf_counter()
        try:
            from restore_builder import _build_dca_figure_from_state
            fig = _build_dca_figure_from_state(state)
        except Exception as e:
            logger.warning("restore_builder (dca) failed: %s; falling back to "
                           "callback path", e)
            fig = None
        if fig is not None:
            _dca_out = fig
            _committed_out = hash_str
            print(f"[trace] restore-dca-build BUILT "
                  f"{(_time.perf_counter() - _t1) * 1000:.1f}ms", flush=True)
    return state, hash_str, True, _fig_out, _committed_out, _dca_out
```

(This replaces the existing block from line ~79 through the existing `return state, hash_str, True, _fig_out, _committed_out` at line ~100.)

- [ ] **Step 4: Syntax check**

Run:
```bash
cd /scratch/code/bitcoinprojections && PYTHONPATH="btc_web:." btc_venv/bin/python3 -c "
import sys; sys.path.insert(0, 'btc_web')
import os; os.environ['DEV'] = '1'
import app
import callbacks.snapshot_cb
print('OK')
"
```

Expected: `OK`.

---

## Task 5: Add clientside relay + `dca-build-count` incrementer

**Files:**
- Modify: `btc_web/callbacks/snapshot_cb.py` (insert near existing `restore-bubble-fig` relay block)

- [ ] **Step 1: Add the DCA figure relay + build-count incrementer**

Find the existing comment block "Bubble figure relay via set_props (Phase 1, 2026-04-25)" and the clientside_callback that follows it. Immediately AFTER that block (still before the "Direct modal close on active-chart-committed" block), add:

```python
# ── DCA figure relay via set_props (Phase 2, 2026-04-25) ──────────────────
# Same pattern as the bubble relay above. dca-graph is inside dca-lazy and
# absent on /1/2/4/5/6/7 initial loads — so we cannot use Output("dca-graph",
# "figure") directly (would re-introduce the Phase 1 dispatch-drop bug).
# Instead, restore_from_url writes to the always-mounted restore-dca-fig
# Store, and this callback uses set_props to push the figure into dca-graph.
_app_ctx.app.clientside_callback(
    """
    function(fig) {
        var NU = window.dash_clientside.no_update;
        if (fig == null) return NU;
        try {
            window.dash_clientside.set_props('dca-graph', {figure: fig});
        } catch (e) {
            console.warn('restore-dca-fig: set_props failed', e);
        }
        if (window.__qsTrace) window.__qsTrace('restore-dca-fig delivered');
        return null;  // self-clear
    }
    """,
    Output("restore-dca-fig", "data", allow_duplicate=True),
    Input("restore-dca-fig", "data"),
    prevent_initial_call=True,
)


# ── DCA build-count phantom-rebuild detector (Phase 2 test infra) ─────────
# Increments dca-build-count whenever dca-graph.figure mutates (relay
# delivery OR cascade rebuild). E2E test 9 reads the Store and asserts
# count == 1 after restore. count >= 2 means the post-restore guard failed.
_app_ctx.app.clientside_callback(
    """function(fig, cur) { return (cur || 0) + 1; }""",
    Output("dca-build-count", "data", allow_duplicate=True),
    Input("dca-graph", "figure"),
    State("dca-build-count", "data"),
    prevent_initial_call=True,
)
```

- [ ] **Step 2: Syntax + import check**

Run:
```bash
cd /scratch/code/bitcoinprojections && PYTHONPATH="btc_web:." btc_venv/bin/python3 -c "
import sys; sys.path.insert(0, 'btc_web')
import os; os.environ['DEV'] = '1'
import app
import callbacks.snapshot_cb
print('OK')
"
```

Expected: `OK`.

---

## Task 6: Add `update_dca` post-restore short-circuit

**Files:**
- Modify: `btc_web/callbacks/charts/__init__.py:871-1054`

- [ ] **Step 1: Add 2 new States to the `@callback` decorator**

Find the `@callback` for `update_dca` (line 871-955). The existing State list ends at line 955 with `State("snapshot-pending", "data"),`. Add immediately after that, BEFORE `prevent_initial_call=True`:

```python
    State("snapshot-pending",   "data"),
    State("active-chart-committed", "data"),
    State("loaded-hash-store",  "data"),
    prevent_initial_call=True,
)
```

- [ ] **Step 2: Update the function signature**

Find the `def update_dca(...)` signature (line 958-975). It currently ends with `snapshot_pending=False):`. Add the two new keyword args at the end:

```python
def update_dca(_first_render, stack, use_lots, amount, freq, dca_infl, yr_range, disp, toggles, legend_pos, sel_qs, adv_qs,
               lppl_n_freqs, lppl_weighted, lppl_no_13,
               hyb_a_nlog, hyb_a_ncal, hyb_a_log1d, hyb_a_log2d,
               hyb_a_cal1d, hyb_a_cal2d,
               hyb_b_enabled, hyb_b_nlog, hyb_b_ncal, hyb_b_log1d, hyb_b_log2d,
               hyb_b_cal1d, hyb_b_cal2d,
               ep_a_nlog, ep_a_ncal, ep_a_log1d, ep_a_log2d,
               ep_a_cal1d, ep_a_cal2d,
               ep_b_enabled, ep_b_nlog, ep_b_ncal, ep_b_log1d, ep_b_log2d,
               ep_b_cal1d, ep_b_cal2d,
               _hybppl_commit, _eppl_commit,
               lots_data,
               sc_enable, sc_loan, sc_rate, sc_term, sc_type, sc_repeats,
               sc_entry_mode, sc_custom_price, sc_tax, sc_rollover,
               mc_enable, mc_bins, mc_regime, mc_sims, mc_years, mc_window,
               mc_start_yr, mc_entry_q, _mc_loaded, _pay_trigger, model_show, mc_model_src,
               price_data, mc_cached, pay_token, mc_unblocked, mc_auth, palette_key,
               qs_mode=None, user_model_store=None, snapshot_pending=False,
               active_chart_committed=None, loaded_hash=None):
```

- [ ] **Step 3: Define `_POST_RESTORE_TRIGGERS_DCA` + add guard**

Find the existing `snapshot_pending` gate at line 977-978:
```python
    # Snapshot gate — see spec 2026-04-24-single-redraw-per-snapshot.
    if snapshot_pending:
        return (dash.no_update,) * 8
```

Insert IMMEDIATELY AFTER that block (before line 979 `toggles = toggles or []`):

```python
    # Phase 2 (2026-04-25) post-restore short-circuit: when restore_from_url
    # has delivered the figure via set_props (active-chart-committed ==
    # loaded-hash), suppress the apply_tab_dca cascade's phantom rebuild.
    # The clear-on-user-input listener (snapshot_cb.py:863-889) clears
    # active-chart-committed on first DOM interaction, so steady-state edits
    # proceed normally. dca-mc-loaded is deliberately excluded so MC async
    # completion can rebuild the chart after the post-restore window.
    # Tuple-size invariant: keep `(dash.no_update,) * 8` aligned with this
    # callback's 8 Outputs (figure, mc-results, mc-status, mc-rendered-key,
    # mc-save-modal, mc-save-tab, mc-unblocked, yr-range). If a 9th Output
    # is added, this guard's return tuple must grow too.
    _POST_RESTORE_TRIGGERS_DCA = {
        "dca-first-render", "dca-stack", "dca-use-lots", "dca-amount",
        "dca-freq", "dca-infl", "dca-yr-range", "dca-disp", "dca-toggles",
        "dca-legend-pos", "dca-qs", "dca-qs-adv",
        "lppl-n-freqs", "lppl-weighted", "lppl-no-13",
        "hybppl-commit-trigger", "eppl-commit-trigger",
        "dca-sc-enable", "dca-sc-loan", "dca-sc-rate", "dca-sc-term",
        "dca-sc-type", "dca-sc-repeats", "dca-sc-entry-mode",
        "dca-sc-custom-price", "dca-sc-tax", "dca-sc-rollover",
        "dca-mc-enable", "dca-mc-bins", "dca-mc-regime", "dca-mc-sims",
        "dca-mc-years", "dca-mc-window", "dca-mc-start-yr", "dca-mc-entry-q",
        "dca-model-show", "dca-mc-model-src",
    }
    _trg = ctx.triggered_id
    if active_chart_committed and active_chart_committed == loaded_hash \
            and _trg in _POST_RESTORE_TRIGGERS_DCA:
        return (dash.no_update,) * 8
```

(Note: `ctx` is already imported at the top of `callbacks/charts/__init__.py`. If not — grep first to confirm — add `from dash import ctx`.)

- [ ] **Step 4: Syntax + import check**

Run:
```bash
cd /scratch/code/bitcoinprojections && PYTHONPATH="btc_web:." btc_venv/bin/python3 -c "
import sys; sys.path.insert(0, 'btc_web')
import os; os.environ['DEV'] = '1'
import app
import callbacks.charts
print('OK')
"
```

Expected: `OK`.

---

## Task 7: Run full unit test suite (regression check)

**Files:** No changes — confirm nothing else broke.

- [ ] **Step 1: Run all non-E2E tests**

Run:
```bash
cd /scratch/code/bitcoinprojections && \
  btc_venv/bin/python3 -m pytest btc_web/ -v --ignore-glob='*_e2e.py' 2>&1 | tail -15
```

Expected: all tests pass except the pre-existing `test_no_hex_literals_outside_colors_module` failure (same as Phase 1 — unrelated to this work). Total: ~1525 passed, 1 pre-existing failed.

If any OTHER test fails, do NOT proceed. Most likely candidates:
- A test that hard-codes `update_dca`'s signature length (unlikely — none did in Phase 1).
- The Phase 1 invariant test `test_restore_from_url_does_not_output_bubble_graph` should still pass (we still don't output to `bubble-graph.figure`; the swap to `restore-bubble-fig.data` from Phase 1 stays intact).

---

## Task 8: Add 6 E2E tests + run them

**Files:**
- Create: `btc_web/test_restore_phase2_dca_e2e.py`

- [ ] **Step 1: Create the E2E test file**

```python
"""Phase 2 end-to-end Playwright tests for /3 (DCA) fast modal close.

Verifies the bubble fast-modal-close pattern correctly extends to /3:
- Non-MC, non-Saylor-live shares deliver the figure via set_props relay
  and modal closes via the active-chart-committed listener (~3-5 s).
- MC-enabled and Saylor-live shares fall back to the existing 7 s timer.
- Post-restore guard suppresses phantom rebuilds (verified via the
  dca-build-count Store + clientside increment on dca-graph.figure mutation).

Requires:
  pip install playwright && python -m playwright install firefox
  Dev server must be running on :8050
Run:
  cd /scratch/code/bitcoinprojections && btc_venv/bin/python3 \\
      -m pytest btc_web/test_restore_phase2_dca_e2e.py -v --timeout=60
"""
import pytest
import time

try:
    from playwright.sync_api import sync_playwright
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False

pytestmark = pytest.mark.skipif(not HAS_PLAYWRIGHT, reason="playwright not installed")

BASE_URL = "http://localhost:8050"


def _make_share_url(state, path):
    """Encode state dict into q4 share link at the given path."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    os.environ.setdefault("DEV", "1")
    import app  # noqa: F401
    from snapshot import _encode_snapshot_v4
    blob = _encode_snapshot_v4(state)
    return f"{BASE_URL}{path}#q4:{blob}"


def _read_dca_build_count(page):
    """Read dca-build-count Store value via Dash's internal Redux store.

    Dash 4 exposes _dashprivate_layout but its store path is implementation
    detail — instead, use the rendered DOM data attribute that Dash writes
    for dcc.Store components when their value changes. dcc.Store renders
    nothing visible but Dash's store registry exposes the value via a
    JS read of the layout tree. Simplest reliable approach: ask Dash via
    its internal API.
    """
    return page.evaluate("""() => {
        // Dash 4 exposes the layout via _dashprivate_layout. Walk it
        // recursively to find the Store with id 'dca-build-count'.
        function find(comp) {
            if (!comp) return null;
            if (comp.props && comp.props.id === 'dca-build-count') {
                return comp.props.data;
            }
            var ch = comp.props && comp.props.children;
            if (Array.isArray(ch)) {
                for (var i = 0; i < ch.length; i++) {
                    var r = find(ch[i]);
                    if (r !== null && r !== undefined) return r;
                }
            } else if (ch) {
                return find(ch);
            }
            return null;
        }
        return find(window._dashprivate_layout);
    }""")


def test_dca_share_fast_modal_close():
    """/3 share with dca-amount=999 renders chart fast and restores widget."""
    url = _make_share_url(
        {"main-tabs:active_tab": "dca", "dca-amount:value": 999},
        "/3",
    )
    with sync_playwright() as p:
        browser = p.firefox.launch(headless=True)
        page = browser.new_page()
        t0 = time.perf_counter()
        page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        page.wait_for_function(
            "() => { var gd = document.querySelector('#dca-graph .js-plotly-plot'); "
            "return gd && gd.data && gd.data.length > 0; }",
            timeout=20_000,
        )
        t_chart = (time.perf_counter() - t0) * 1000
        amount = page.evaluate("() => document.getElementById('dca-amount').value")
        browser.close()
    assert amount == "999", f"dca-amount={amount}, expected 999"
    assert t_chart < 5000, (
        f"DCA chart took {t_chart:.0f}ms (expected <5000ms with fast path)"
    )


def test_dca_yr_range_restored():
    """/3 share with non-default yr-range — slider restores to exact values."""
    url = _make_share_url(
        {
            "main-tabs:active_tab": "dca",
            "dca-yr-range:value": [2030, 2040],
            "dca-amount:value": 100,
        },
        "/3",
    )
    with sync_playwright() as p:
        browser = p.firefox.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        page.wait_for_function(
            "() => { var gd = document.querySelector('#dca-graph .js-plotly-plot'); "
            "return gd && gd.data && gd.data.length > 0; }",
            timeout=20_000,
        )
        # Read the slider value via Dash's internal store
        time.sleep(1)  # let cascade settle
        yr_range = page.evaluate("""() => {
            function find(comp) {
                if (!comp) return null;
                if (comp.props && comp.props.id === 'dca-yr-range') {
                    return comp.props.value;
                }
                var ch = comp.props && comp.props.children;
                if (Array.isArray(ch)) {
                    for (var i = 0; i < ch.length; i++) {
                        var r = find(ch[i]);
                        if (r !== null && r !== undefined) return r;
                    }
                } else if (ch) { return find(ch); }
                return null;
            }
            return find(window._dashprivate_layout);
        }""")
        browser.close()
    assert yr_range == [2030, 2040], f"yr_range={yr_range}, expected [2030, 2040]"


def test_dca_mc_share_falls_back():
    """MC-enabled /3 share — modal eventually closes (cascade path).

    Asserts a generous upper bound (12 s) without a brittle lower bound.
    The cascade path is correct as long as the chart eventually renders.
    """
    url = _make_share_url(
        {
            "main-tabs:active_tab": "dca",
            "dca-mc-enable:value": ["yes"],
            "dca-amount:value": 100,
        },
        "/3",
    )
    with sync_playwright() as p:
        browser = p.firefox.launch(headless=True)
        page = browser.new_page()
        t0 = time.perf_counter()
        page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        page.wait_for_function(
            "() => { var gd = document.querySelector('#dca-graph .js-plotly-plot'); "
            "return gd && gd.data && gd.data.length > 0; }",
            timeout=15_000,
        )
        t_chart = (time.perf_counter() - t0) * 1000
        browser.close()
    assert t_chart < 12000, (
        f"MC fallback DCA took {t_chart:.0f}ms (>12s — fallback path broken)"
    )


def test_dca_sc_live_falls_back():
    """Saylor-live /3 share — modal eventually closes via cascade path."""
    url = _make_share_url(
        {
            "main-tabs:active_tab": "dca",
            "dca-sc-enable:value": ["yes"],
            "dca-sc-entry-mode:value": "live",
            "dca-sc-loan:value": 100000.0,
            "dca-amount:value": 100,
        },
        "/3",
    )
    with sync_playwright() as p:
        browser = p.firefox.launch(headless=True)
        page = browser.new_page()
        t0 = time.perf_counter()
        page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        page.wait_for_function(
            "() => { var gd = document.querySelector('#dca-graph .js-plotly-plot'); "
            "return gd && gd.data && gd.data.length > 0; }",
            timeout=15_000,
        )
        t_chart = (time.perf_counter() - t0) * 1000
        browser.close()
    assert t_chart < 12000, (
        f"SC-live fallback DCA took {t_chart:.0f}ms (>12s — fallback broken)"
    )


def test_dca_no_phantom_rebuild():
    """Post-restore guard suppresses cascade rebuilds.

    dca-build-count Store + clientside increment fires on every
    dca-graph.figure mutation. Single delivery via the relay = count==1.
    Cascade rebuild = count>=2 (guard failed).
    """
    url = _make_share_url(
        {
            "main-tabs:active_tab": "dca",
            "dca-amount:value": 999,
            "dca-yr-range:value": [2025, 2035],
        },
        "/3",
    )
    with sync_playwright() as p:
        browser = p.firefox.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        page.wait_for_function(
            "() => { var gd = document.querySelector('#dca-graph .js-plotly-plot'); "
            "return gd && gd.data && gd.data.length > 0; }",
            timeout=20_000,
        )
        # Wait for any post-restore cascade fires to settle.
        time.sleep(1.5)
        count = _read_dca_build_count(page)
        browser.close()
    assert count == 1, (
        f"dca-build-count={count} after restore (expected 1). "
        f">=2 means the post-restore guard failed and the cascade "
        f"rebuilt the figure unnecessarily."
    )


def test_bubble_share_still_restores():
    """Phase 1 regression: /1 bubble share still works fast."""
    url = _make_share_url(
        {"main-tabs:active_tab": "bubble", "bub-qs:value": ["median"]},
        "/1",
    )
    with sync_playwright() as p:
        browser = p.firefox.launch(headless=True)
        page = browser.new_page()
        t0 = time.perf_counter()
        page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        page.wait_for_function(
            "() => { var gd = document.querySelector('#bubble-graph .js-plotly-plot'); "
            "return gd && gd.data && gd.data.length > 0; }",
            timeout=20_000,
        )
        t_chart = (time.perf_counter() - t0) * 1000
        browser.close()
    assert t_chart < 5000, f"/1 bubble took {t_chart:.0f}ms (Phase 1 regression)"
```

- [ ] **Step 2: Start dev server**

Run:
```bash
cd /scratch/code/bitcoinprojections && \
  lsof -ti :8050 | xargs -r kill -9 2>/dev/null; \
  sleep 1 && \
  DEV=1 nohup bash run_web.sh > /tmp/quantoshi_dev.log 2>&1 &
sleep 6 && tail -3 /tmp/quantoshi_dev.log
```

Expected: log shows `Dash is running on http://0.0.0.0:8050/`.

- [ ] **Step 3: Run E2E tests**

Run:
```bash
cd /scratch/code/bitcoinprojections && \
  btc_venv/bin/python3 -m pytest btc_web/test_restore_phase2_dca_e2e.py -v --timeout=60
```

Expected: `6 passed`.

If `test_dca_no_phantom_rebuild` fails with `count >= 2`: the post-restore guard is not catching one of the 37 Inputs in `_POST_RESTORE_TRIGGERS_DCA`. Capture the dev journal during the test, look for a `[trace]` line indicating which Input fired.

If `test_dca_yr_range_restored` fails with stale yr-range: the cascade has rebuilt yr-range; check that apply_tab_dca's writes are properly flowing.

- [ ] **Step 4: Stop dev server (will restart for Task 9)**

Run:
```bash
lsof -ti :8050 | xargs -r kill -9 2>/dev/null
```

---

## Task 9: Cold-cache timing probe

**Files:** No code changes — observation step.

The architect flagged DCA cold-cache build time as an unknown (could be 50-600 ms; Phase 1 bubble was 40-150 ms). Threshold per spec is 500 ms median; if exceeded, narrow option (a) to also include SC-enabled.

- [ ] **Step 1: Force a cold L1 cache + start dev server**

Run:
```bash
cd /scratch/code/bitcoinprojections && \
  lsof -ti :8050 | xargs -r kill -9 2>/dev/null; \
  sleep 1 && \
  DEV=1 nohup bash run_web.sh > /tmp/quantoshi_dev.log 2>&1 &
sleep 6
```

Expected: server started, L1 LRU cache empty.

- [ ] **Step 2: Generate 5 representative `/3` URLs (varying yr-range + SC)**

```bash
cd /scratch/code/bitcoinprojections && PYTHONPATH="btc_web:." btc_venv/bin/python3 -c "
import sys; sys.path.insert(0, 'btc_web')
import os; os.environ['DEV'] = '1'
import app
from snapshot import _encode_snapshot_v4

scenarios = [
    {'name': 'short_no_sc',    'st': {'main-tabs:active_tab': 'dca', 'dca-yr-range:value': [2026, 2030], 'dca-amount:value': 100}},
    {'name': 'long_no_sc',     'st': {'main-tabs:active_tab': 'dca', 'dca-yr-range:value': [2026, 2050], 'dca-amount:value': 100}},
    {'name': 'short_sc_custom','st': {'main-tabs:active_tab': 'dca', 'dca-yr-range:value': [2026, 2030], 'dca-sc-enable:value': ['yes'], 'dca-sc-entry-mode:value': 'custom', 'dca-sc-custom-price:value': 50000, 'dca-sc-loan:value': 50000}},
    {'name': 'long_sc_custom', 'st': {'main-tabs:active_tab': 'dca', 'dca-yr-range:value': [2026, 2050], 'dca-sc-enable:value': ['yes'], 'dca-sc-entry-mode:value': 'custom', 'dca-sc-custom-price:value': 50000, 'dca-sc-loan:value': 50000}},
    {'name': 'multi_q_long',   'st': {'main-tabs:active_tab': 'dca', 'dca-yr-range:value': [2026, 2050], 'dca-qs:value': [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99], 'dca-amount:value': 100}},
]
for s in scenarios:
    blob = _encode_snapshot_v4(s['st'])
    print(f'{s[\"name\"]}: http://localhost:8050/3#q4:{blob}')
"
```

- [ ] **Step 3: Load each URL with cold cache via Playwright + grep dev log**

```bash
cat > /tmp/cold_cache_probe.py <<'EOF'
"""Cold-cache timing probe for /3 build."""
import sys, time, os, subprocess
sys.path.insert(0, "/scratch/code/bitcoinprojections/btc_web")
os.environ["DEV"] = "1"
import app
from snapshot import _encode_snapshot_v4
from playwright.sync_api import sync_playwright

scenarios = [
    ("short_no_sc",     {"main-tabs:active_tab": "dca", "dca-yr-range:value": [2026, 2030], "dca-amount:value": 100}),
    ("long_no_sc",      {"main-tabs:active_tab": "dca", "dca-yr-range:value": [2026, 2050], "dca-amount:value": 100}),
    ("short_sc_custom", {"main-tabs:active_tab": "dca", "dca-yr-range:value": [2026, 2030], "dca-sc-enable:value": ["yes"], "dca-sc-entry-mode:value": "custom", "dca-sc-custom-price:value": 50000.0, "dca-sc-loan:value": 50000.0}),
    ("long_sc_custom",  {"main-tabs:active_tab": "dca", "dca-yr-range:value": [2026, 2050], "dca-sc-enable:value": ["yes"], "dca-sc-entry-mode:value": "custom", "dca-sc-custom-price:value": 50000.0, "dca-sc-loan:value": 50000.0}),
    ("multi_q_long",    {"main-tabs:active_tab": "dca", "dca-yr-range:value": [2026, 2050], "dca-qs:value": [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99], "dca-amount:value": 100}),
]

# Restart dev server to ensure cold L1 cache
subprocess.run(["bash", "-c", "lsof -ti :8050 | xargs -r kill -9 2>/dev/null"], check=False)
time.sleep(1)
subprocess.Popen(
    ["bash", "-c", "DEV=1 nohup bash /scratch/code/bitcoinprojections/run_web.sh > /tmp/quantoshi_dev.log 2>&1 &"],
)
time.sleep(7)

with sync_playwright() as p:
    browser = p.firefox.launch(headless=True)
    for name, state in scenarios:
        blob = _encode_snapshot_v4(state)
        url = f"http://localhost:8050/3#q4:{blob}"
        ctx = browser.new_context()
        page = ctx.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        page.wait_for_function(
            "() => { var gd = document.querySelector('#dca-graph .js-plotly-plot'); return gd && gd.data && gd.data.length > 0; }",
            timeout=20_000,
        )
        ctx.close()
        time.sleep(0.5)
    browser.close()

# Read trace lines from dev log
with open("/tmp/quantoshi_dev.log") as f:
    lines = [l for l in f if "[trace] restore-dca-build BUILT" in l]
print(f"\n=== restore-dca-build BUILT trace lines ({len(lines)}) ===")
for line in lines:
    print(f"  {line.rstrip()}")
EOF
btc_venv/bin/python3 /tmp/cold_cache_probe.py
```

Expected: 5 trace lines like `[trace] restore-dca-build BUILT 245.3ms`. Note the ms values.

- [ ] **Step 4: Evaluate threshold**

Compute median of the 5 BUILT times.

- **Median < 500 ms**: green-light. Proceed to Task 10.
- **Median >= 500 ms** AND most slowness is in SC scenarios: revisit spec — narrow option (a) to also include SC-enabled by adding `if "yes" in sc_enable: return None` to the gate at the top of `_build_dca_figure_from_state`. Re-run probe.
- **Median >= 500 ms** for non-SC scenarios too: stop. Phase 2 needs deeper rethink (e.g., async build via Celery, or partial figure).

- [ ] **Step 5: Stop dev server**

```bash
lsof -ti :8050 | xargs -r kill -9 2>/dev/null
```

---

## Task 10: Update memory + docs

**Files:**
- Modify: `/home/bcg/.claude/projects/-scratch-code-bitcoinprojections/memory/restore_callback_architecture.md`
- Modify: `/scratch/code/bitcoinprojections/docs/architecture.md`

- [ ] **Step 1: Append Phase 2 entry to memory**

Find the section heading `## Phase 1 dispatch fix (2026-04-25 — SHIPPED + WORKING)` and the `## Files involved` heading after it. Insert immediately before `## Files involved`:

```markdown
## Phase 2 (DCA fast modal close, 2026-04-25 — SHIPPED + WORKING)

After Phase 1 fixed `restore_from_url` dispatch for /2-/7, the next step
was extending the bubble fast-modal-close pattern (compute figure inside
`restore_from_url`, deliver via Store + set_props) to per-tab paths. /3
(DCA) is the first ship in the user-suggested order 3 → 4 → 5 → 7 → 2 → 6.

**Pattern (per-tab Store relay):**
- New `dcc.Store(id="restore-dca-fig")` in layout, always-mounted.
- `_build_dca_figure_from_state(state)` in `restore_builder.py` mirrors
  `update_dca`'s param construction. Returns None for MC-enabled and
  Saylor-live snapshots (fall back to 7s timer); otherwise calls
  `_get_dca_fig` directly with `mc_enabled=False` (sufficient on its own;
  `_get_mc_or_cached` strips all `mc_*` keys before quantizing the cache key).
- `restore_from_url` gains a 6th Output (`restore-dca-fig.data`) and a
  DCA branch alongside the existing bubble branch. Both early-exit returns
  become 6-tuples.
- Clientside relay watches `restore-dca-fig.data` and uses
  `set_props('dca-graph', {figure: fig})` to push the figure when dca-graph
  is mounted (works for /3 share links; no-ops on other paths).
- `update_dca` post-restore guard mirrors `update_bubble`'s pattern:
  `_POST_RESTORE_TRIGGERS_DCA` set (37 entries — every Input EXCEPT
  `dca-mc-loaded`), `(dash.no_update,) * 8` short-circuit when
  `active_chart_committed == loaded_hash` AND `_trg in set`. The existing
  clear-on-user-input listener handles both /1 and /3.

**Phantom-rebuild detection (test infra):** new `dca-build-count` Store +
pure-clientside callback that increments on `dca-graph.figure` mutation.
E2E test 9 reads via `page.evaluate` and asserts count==1 after restore
(single delivery via relay; >=2 means the cascade rebuilt the figure).

**Out of scope:** /4 retire, /5 supercharge, /7 leverage, /2 heatmap, /6 citadel.
Each ships independently in subsequent commits.

**Spec:** `docs/superpowers/specs/2026-04-25-restore-dca-fast-modal-close-design.md`
**Plan:** `docs/superpowers/plans/2026-04-25-restore-dca-fast-modal-close-plan.md`
**Measured prod latency:** TBD-after-prod-verify (target <5s).
```

(After prod-verify completes in Task 13, replace `TBD-after-prod-verify` with the actual measured value.)

- [ ] **Step 2: Update docs/architecture.md**

Find the existing "Restore performance architecture" section. Find the sentence:
```
Citadel + other non-bubble share-link tabs fall back to the existing
callback cascade (`restore_builder` only handles bubble).
```

Replace with:
```
/3 (DCA) joins /1 (bubble) on the fast path via the same pattern: per-tab
Store relay + clientside set_props + post-restore short-circuit (Phase 2,
2026-04-25). Each new tab requires (a) a new `restore-{tab}-fig` Store,
(b) a `_build_{tab}_figure_from_state` helper in `restore_builder.py`,
(c) a new branch in `restore_from_url`, (d) a clientside relay, and
(e) a post-restore guard inside the tab's chart callback. Citadel +
heatmap + retire + supercharge + leverage still fall back to the existing
callback cascade (per-tab ships in subsequent commits).
```

---

## Task 11: Single commit

**Files:** all probe edits + tests + memory + docs.

- [ ] **Step 1: Review diff**

Run:
```bash
cd /scratch/code/bitcoinprojections && git status --short && echo "---" && git diff --stat
```

Expected files modified:
- `btc_web/layout/__init__.py`
- `btc_web/restore_builder.py`
- `btc_web/callbacks/snapshot_cb.py`
- `btc_web/callbacks/charts/__init__.py`
- `btc_web/test_restore_builder.py`
- `btc_web/test_restore_phase2_dca_e2e.py` (new)
- `docs/architecture.md`

NOT in commit:
- `model_data_ef.pkl`, `model_data_resqr_diagnostics.json` (still-modified data files; leave in working tree)
- `dash_req*.json`, `tools/*.py` (unrelated WIP)
- The memory file (lives outside the repo)

- [ ] **Step 2: Stage exactly the Phase 2 files**

```bash
cd /scratch/code/bitcoinprojections && \
  git add btc_web/layout/__init__.py \
          btc_web/restore_builder.py \
          btc_web/callbacks/snapshot_cb.py \
          btc_web/callbacks/charts/__init__.py \
          btc_web/test_restore_builder.py \
          btc_web/test_restore_phase2_dca_e2e.py \
          docs/architecture.md \
          docs/superpowers/plans/2026-04-25-restore-dca-fast-modal-close-plan.md \
          docs/superpowers/specs/2026-04-25-restore-dca-fast-modal-close-design.md
```

- [ ] **Step 3: Verify staged files**

```bash
git diff --cached --stat
```

Expected: 9 files staged (7 code/test/doc + 2 spec/plan markdown).

- [ ] **Step 4: Commit**

```bash
git commit -m "$(cat <<'EOF'
feat(restore): /3 (DCA) fast modal close via per-tab Store relay

Extends the Phase 1 bubble fast-modal-close pattern to /3 share links.
restore_from_url now calls _build_dca_figure_from_state for active_tab=
"dca" and delivers the figure via the new restore-dca-fig Store +
clientside set_props relay. update_dca gains a post-restore guard with
_POST_RESTORE_TRIGGERS_DCA (37 entries — every Input except dca-mc-loaded)
to suppress phantom rebuilds during the post-restore window.

MC-enabled and Saylor-live snapshots fall back to the existing 7s timer
(option a) — Phase 2 first ship doesn't replicate the MC pipeline or
fetch a live BTC price inside the builder.

Test infra: new dca-build-count Store + pure-clientside increment on
dca-graph.figure mutation. E2E test asserts count==1 after restore
(single delivery via relay; >=2 means the post-restore guard failed).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
git log --oneline -3
```

Expected: top commit is the Phase 2 fix; ~9 files changed.

---

## Task 12: CHECKPOINT — confirm with user before prod deploy

**Files:** No changes — gating step.

- [ ] **Step 1: Report local verification status to user**

Tell the user:
- Phase 2 commit landed locally with hash `<git rev-parse HEAD>`.
- Local verification passed (Tasks 7 + 8: full unit suite, all 6 E2E tests, cold-cache probe under 500 ms median).
- About to push to origin and deploy to prod.

- [ ] **Step 2: Wait for explicit user "ok" before Task 13**

Per Phase 1 precedent: do NOT proceed to push/deploy until the user explicitly approves.

---

## Task 13: Prod deploy

**Files:** No changes — deploy step.

- [ ] **Step 1: Push + deploy**

```bash
cd /scratch/code/bitcoinprojections && git push origin master
ssh root@89.167.70.45 "cd /opt/quantoshi && git pull && redis-cli FLUSHDB && systemctl restart quantoshi"
sleep 8
ssh root@89.167.70.45 "systemctl status quantoshi --no-pager | head -8"
```

Expected: `Active: active (running)`.

- [ ] **Step 2: Wait + check journal for clean startup**

```bash
sleep 5 && ssh root@89.167.70.45 "journalctl -u quantoshi --since '30 sec ago' --no-pager | grep -E 'ERROR|CRITICAL|Traceback|exit' | head -10"
```

Expected: no errors, no traceback. (Empty grep output is normal.)

---

## Task 14: Prod verify

**Files:** No changes — verification step.

- [ ] **Step 1: Generate prod URL + run Playwright probe**

```bash
cat > /tmp/prod_phase2_verify.py <<'EOF'
"""Prod verification for Phase 2 /3 fast modal close."""
import sys, os, time
sys.path.insert(0, "/scratch/code/bitcoinprojections/btc_web")
os.environ["DEV"] = "1"
import app
from snapshot import _encode_snapshot_v4
from playwright.sync_api import sync_playwright

dca_blob = _encode_snapshot_v4({
    "main-tabs:active_tab": "dca",
    "dca-amount:value": 999,
    "dca-yr-range:value": [2030, 2040],
})
bub_blob = _encode_snapshot_v4({"main-tabs:active_tab": "bubble", "bub-qs:value": ["median"]})
dca_url = f"https://quantoshi.xyz/3#q4:{dca_blob}"
bub_url = f"https://quantoshi.xyz/1#q4:{bub_blob}"

results = {}
with sync_playwright() as p:
    browser = p.firefox.launch(headless=True)

    # /3 DCA fast-path
    ctx = browser.new_context()
    page = ctx.new_page()
    t0 = time.perf_counter()
    page.goto(dca_url, wait_until="domcontentloaded", timeout=30_000)
    page.wait_for_function(
        "() => { var gd = document.querySelector('#dca-graph .js-plotly-plot'); return gd && gd.data && gd.data.length > 0; }",
        timeout=30_000,
    )
    t_dca = (time.perf_counter() - t0) * 1000
    dca_amount = page.evaluate("() => document.getElementById('dca-amount').value")
    results["dca_chart_ms"] = t_dca
    results["dca_amount"] = dca_amount
    ctx.close()

    # /1 bubble regression
    ctx = browser.new_context()
    page = ctx.new_page()
    t0 = time.perf_counter()
    page.goto(bub_url, wait_until="domcontentloaded", timeout=30_000)
    page.wait_for_function(
        "() => { var gd = document.querySelector('#bubble-graph .js-plotly-plot'); return gd && gd.data && gd.data.length > 0; }",
        timeout=30_000,
    )
    t_bub = (time.perf_counter() - t0) * 1000
    results["bub_chart_ms"] = t_bub
    ctx.close()
    browser.close()

print(f"\n=== PROD VERIFY ===")
print(f"  /3 DCA fast: dca-amount={results['dca_amount']!r} (expected '999')")
print(f"  /3 DCA chart: {results['dca_chart_ms']:.0f}ms (expect <5000ms)")
print(f"  /1 bubble:    {results['bub_chart_ms']:.0f}ms (expect <6000ms, no regression)")

assert results["dca_amount"] == "999", f"PROD FAIL dca-amount: {results['dca_amount']!r}"
assert results["dca_chart_ms"] < 5000, f"PROD FAIL /3 too slow: {results['dca_chart_ms']:.0f}ms"
assert results["bub_chart_ms"] < 6000, f"PROD FAIL /1 regression: {results['bub_chart_ms']:.0f}ms"
print("\nPROD VERIFY: PASS")
EOF
btc_venv/bin/python3 /tmp/prod_phase2_verify.py
```

Expected: `PROD VERIFY: PASS` with /3 chart <5000ms and /1 chart <6000ms.

- [ ] **Step 2: Confirm prod journal traces**

```bash
ssh root@89.167.70.45 "journalctl -u quantoshi --since '3 min ago' --no-pager | grep -E '\\[trace\\]' | tail -15"
```

Expected: at least
- `[trace] restore_from_url prefix=q4: controls=N` for both `/3` and `/1`
- `[trace] restore-dca-build BUILT Xms` for the `/3` load (NEW)
- `[trace] restore-direct-build BUILT Xms` for the `/1` load (Phase 1 unchanged)

- [ ] **Step 3: Update memory file with measured prod latency**

Edit `/home/bcg/.claude/projects/-scratch-code-bitcoinprojections/memory/restore_callback_architecture.md`. Replace `TBD-after-prod-verify` (added in Task 10 Step 1) with the actual measured `/3` chart_ms value, e.g., "/3 chart_ms ~3500ms (prod, post-deploy)". This is a memory file edit only — not committed.

---

## Done — Phase 2 `/3` complete

Phase 2 next ship is `/4` (Retire). The pattern is mechanical: new Store, new builder helper, new restore_from_url branch, new relay, new post-restore guard. Same plan structure with `dca` → `retire` substitutions.
