# MC Spaghetti Fan on Tab 1 — Design Spec

**Date:** 2026-04-26
**Status:** Brainstormed, awaiting user review before implementation plan.

## Goal

Render Markov Monte Carlo simulation paths as a "spaghetti fan" overlay on the Bubble chart (Tab 1), giving the user a visual sample of regime-aware path uncertainty alongside the existing analytical price models. Sourced from the user's selected MC model + start year via the same MC config panel used on Tabs 3-5.

## Visualization

Mockup: https://quantoshi.xyz/mcideas — option **(C) Spaghetti fan**.

- Subsample of N=100 paths drawn from the cached/computed (n_sims, n_steps) array.
- Each path is a thin Plotly `Scatter` line (mode=`lines`, width=0.6, opacity=0.45).
- Color: matplotlib `RdYlGn` cmap, normalized by terminal price (lowest-final = red, highest-final = green).
- Layer order: between price scatter (lowest) and model lines (highest) so paths don't obscure the analytical curves.
- Single bundle of N traces, not regime-tinted (regime tinting deferred to a future iteration; see "out of scope").

## Architecture

Reuse the existing MC machinery — same panel, same payment flow, same callback pattern. Tab 1 becomes a peer of Tabs 3-5 in MC plumbing.

### Files touched

| File | Change |
|---|---|
| `btc_web/layout/bubble.py` | Insert `_mc_controls("bub", show_amount=False, show_inflation=False, show_stack=False)` after the Projection Quantiles card |
| `btc_web/snapshot_defaults.py` | Add `bub-mc-*` defaults (mirrors existing `dca-mc-*`) |
| `btc_web/snapshot.py` | Append `bub-mc-*` to `_SNAPSHOT_CONTROLS` **at the END of the bubble section** (after `bub-use-lots`), not in the middle — minimizes bit-position drift for downstream entries; add to `_TAB_CONTROLS["bubble"]` |
| `btc_web/tab_defaults.py` | Add MC keys to `BUBBLE` dict via `sd("bub-mc-*:value", ...)` AND to `bubble_defaults()` so the prewarm cache key matches runtime (test_cache_key_alignment.py) |
| `btc_web/callbacks/mc_payment.py` | **Append `"bub"` to the END of `_MC_TABS = ("dca", "ret", "hm", "sc", "cp", "bub")`** — the callback's positional indexing (`tab_idx * 3` etc.) requires existing offsets stay stable. Append the 5 new States (`bub-mc-years/start-yr/entry-q/model-src/price-val`) **at the END of the existing State block, before the cfg-modal States added in commit 7820a84** so they fit the existing `state_base + tab_idx * 3` arithmetic. |
| `btc_web/callbacks/mc_controls.py` | Add `"bub"` to **every** prefix-iteration loop in this file. Audit shows 7 loops: (1) body-toggle (`for _mc_tog`), (2) Display-Models-MC-injection (`for _mc_auto`), (3) advanced-toggle (`for _mc_adv`), (4) regime-options (`for _mc_reg`), (5) `_MC_MATCH_JS_TPL` rendered_key matcher, (6) restore-button loop (`for _mc_rst` if present), (7) MC cost (`for _cost_pfx`). Plan must enumerate explicitly — silently missing any one breaks a piece of the UI. |
| `btc_web/callbacks/charts/__init__.py` | Extend `update_bubble` with MC Inputs/States; call `_resolve_mc_model_src` then `_mc_setup`; route the figure build through a new `_get_mc_bubble_fig` (see `utils.py` row); render spaghetti via the new helper; **add every new `bub-mc-*` Input ID to the `_POST_RESTORE_TRIGGERS` set** so post-restore writes don't trigger phantom rebuilds. |
| `btc_web/utils.py` | New `_get_mc_bubble_fig(p)` wrapping `_get_mc_or_cached(p, build_bubble_figure, _cached_bubble_fig)` (parallel to `_get_dca_fig`/`_get_retire_fig`). Without this, `mc_cached` dict pollutes the JSON cache key and serialization fails. `update_bubble` calls `_get_mc_bubble_fig` only when `mc_enabled` is truthy; otherwise stays on `_get_bubble_fig` (the existing fast non-MC path). |
| `btc_web/figures/bubble.py` | New `_add_mc_spaghetti(fig, paths, t_axis, ...)` helper. Confirm `build_bubble_figure` returns `(fig, mc_result)` tuple when called via `_get_mc_or_cached` (compare `build_dca_figure` / `build_retire_figure`); if it currently returns `go.Figure` only, update its signature for the MC branch. |
| `btc_web/restore_builder.py` | `_build_bubble_figure_from_state` gates: when `bub-mc-enable` is truthy in the snapshot state, return `None` (fall back to cascade), matching the CTA-active gate already in that builder. Without this, MC-enabled share links restore an empty bubble fig and never render the fan. |

### Data flow

```
[user enables bub-mc-enable] → [MC card body uncollapses (clientside)]
[user picks model/year/q] → [Run Simulation button] → [_mc_payment_initiate]
                                                       │
                              ┌── free tier ──────────┤
                              │   trigger increments  │
                              │                       └── paid ──── invoice modal
                              ▼                                     │
                         [update_bubble fires]                      ▼
                         [_mc_setup → mc_p w/ paths]            [user pays]
                         [_add_mc_spaghetti → fig.data]              │
                                                              [token issued]
                                                              [trigger increments]
                                                              [update_bubble fires]
```

`_resolve_mc_model_src` runs in `update_bubble` before `_mc_setup`, identical to the pattern in `update_dca/retire/sc` and `_mc_payment_initiate`. So the master→variant resolution stays symmetric across all consumers (per `feedback_master_resolver_symmetry.md`).

## UI layout

The new MC card sits **immediately after Projection Quantiles** in the bubble controls column. Default state: `bub-mc-enable=[]` (collapsed body, opt-in). Existing clientside toggle handles the show/hide. Three controls hidden via `show_amount=False, show_inflation=False, show_stack=False` since Tab 1 is price-space only (no withdrawal amount, no inflation, no stack accumulation).

The MC card displays:

- Enable Monte Carlo Simulation checkbox
- Model source (master dropdown — bub/pl/lppl/hybppl/eppl/etc.)
- MC start year (cache-aligned dropdown: 2028/2031/2035 + custom)
- Entry percentile (cache-aligned: 10/20/.../90 + custom)
- MC years (default 40)
- Bins / sims / regime / window (advanced collapsible)
- Run Simulation button
- Status text (free tier badge / payment status)

When the user enables MC, the existing dynamic clientside callback in `mc_controls.py:40` adds `"MC Simulation"` to `bub-model-show.options` — needs the loop extended to include `"bub"`.

## Free vs paid

Same flow as Tabs 3-5. Cached params (model ∈ {bub, ef, exp, lppl, pl, qr} × start_yr ∈ {2028, 2031, 2035} × 40-yr horizon × bin-aligned entry-q) → free; non-cached → BTCPay Lightning invoice.

Tab 1 thereby gains its first paywall surface. Mitigation for the "public on-ramp" character: MC is **disabled by default**. First-paint UX is unchanged; users have to opt in.

**Acknowledged paywall trap on master selection:** the `bub-mc-model-src` dropdown shows masters (per commit e634b54). When the user picks `lppl` master, `_resolve_mc_model_src` translates to `lp3` (default LPPL config has n_freqs=[3]), which is NOT in `_CACHED_MODEL_KEYS` — so the user lands in paid mode silently. This is the same UX contract as Tabs 3-5 and we accept it for parity. Users who want free MC on Tab 1 can pick `bub` / `ef` / `exp` / `pl` / `qr` directly, or pick `lppl` master with config n_freqs=[1] (the cached single-frequency variant).

## Snapshot / share-link contract

`bub-mc-*` keys join `_SNAPSHOT_CONTROLS` and `_TAB_CONTROLS["bubble"]`. Snapshot fingerprint changes — pin both old and new fingerprints via `tools/update_defaults_registry.py` per the existing workflow (see `CLAUDE.md` "Change a snapshot default").

## Path rendering helper (`figures/bubble.py`)

```python
def _add_mc_spaghetti(fig, paths, t_axis, n_display=100, layer_above=False):
    """Add N=n_display sample paths from a (n_sims, n_steps) array.

    Args:
        fig: go.Figure to mutate.
        paths: np.ndarray (n_sims, n_steps).
        t_axis: np.ndarray (n_steps,) — time values matching paths' x dim.
        n_display: target trace count (deterministic stride).
        layer_above: True to draw above all other traces; False (default)
            inserts at the index after price scatter, below model lines.
    """
    if paths is None or paths.size == 0:
        return
    stride = max(1, paths.shape[0] // n_display)
    sample = paths[::stride][:n_display]   # deterministic, reproducible
    finals = sample[:, -1]
    norm = (finals - finals.min()) / max(np.ptp(finals), 1e-12)
    cmap = matplotlib.cm.RdYlGn
    for i, path in enumerate(sample):
        rgba = cmap(norm[i])
        color = f"rgba({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)},0.45)"
        fig.add_trace(go.Scatter(
            x=t_axis, y=path,
            mode="lines",
            line=dict(color=color, width=0.6),
            showlegend=(i == 0),  # one legend entry for the bundle
            name="MC paths" if i == 0 else None,
            hoverinfo="skip",     # avoid hover-clutter on 100 traces
            legendgroup="mc-spaghetti",
        ))
```

`hoverinfo="skip"` makes each spaghetti trace transparent to Plotly's hover proximity calculations entirely (distinct from `"none"` which would show an empty tooltip). The price-scatter and model-line traces underneath remain interactive normally — paths are pure visual context.

## Failure modes

Match Tabs 3-5 exactly — no Tab-1-specific handling.

- `_HAS_MARKOV=False` → MC card hidden via existing placeholder pattern
- `bub-mc-enable=[]` → no fan, MC card body collapsed
- Cache miss + no payment token → invoice modal (existing payment flow)
- Resolver lands on non-master non-cached key → free-tier check fails → invoice
- Status messages reuse existing `mc-pay-status` infrastructure

## Performance

`_add_mc_spaghetti` adds ~100 traces to the bubble figure. Existing chart caches (L0/L1/L2) key on the full params dict including `mc_*` so subsequent identical requests cache-hit. Per-tab first-render trigger architecture means the spaghetti only computes when Tab 1 is active — no cross-tab overhead.

**Concrete perf risk:** `chart_responsive.js` iterates `g.data[i]` and applies trace-width tuning per trace. Per-trace work is O(N_traces) at first paint and on relayout events. The `_applied[id]` cache makes per-event work essentially free after the first iteration, so the dominant cost is **first-paint latency**, not per-event latency. With 100 spaghetti traces added on top of the bubble's existing ~30 traces (price scatter + model lines + bands), first-paint may increase by ~2-3× on mobile.

**Benchmark target before commit:** Playwright cold-load probe of `/1` with MC enabled (`?trace=1`); `[trace] update_bubble BUILT` should stay under 1500ms on the dev box (matches Tab 4 retire's typical cold render). If exceeded, reduce `n_display` to 50 and re-measure. Real prod-mobile measurement happens post-deploy.

## Testing

| Test | What it covers |
|---|---|
| `test_figures.py::test_add_mc_spaghetti_returns_n_traces` | Helper produces correct trace count given a synthetic paths array |
| `test_figures.py::test_add_mc_spaghetti_color_gradient` | Terminal-value normalization gives RdYlGn span |
| `test_callbacks.py::TestUpdateBubbleCallback::test_mc_enabled_renders_paths` | Callback wiring — enabling MC + valid token → traces appear in figure data |
| `test_snapshot.py::TestSnapshotDefaultsConsistency::test_defaults_match_widget_defaults` | Existing test — catches `bub-mc-*` defaults divergence |
| `test_cache_key_alignment.py::test_prewarm_key_matches_runtime_key[_get_bubble_fig-bub]` | Existing test — `bub` prewarm dict matches runtime kwargs after MC params added |
| Manual smoke (`?trace=1`) | `[trace] update_bubble BUILT Xms` line under target latency for cached run |

## Out of scope (defer to follow-ups)

- **Per-regime path tinting** — color paths by current regime bin instead of terminal value. Visually informative but adds a legend dimension. Decide after live feedback.
- **Path interactivity** — hover for individual path's terminal value / regime sequence. Plotly-default hover is OK for MVP.
- **Spaghetti density slider** — let user choose 50/100/200 paths. Defer until mobile-perf data justifies.
- **Options A and B from `/mcideas`** — already in `todo_backlog.md` as #42 and #43.

## Self-review checklist

- [x] Spec covers all 6 open questions listed in `/mcideas`? Yes — Q1=C, Q2=mc-model-src dropdown, Q3=full parity, Q4=panel's years control, Q5=after Projection Quantiles, Q6=same as other tabs.
- [x] No placeholders or "TBD"? Verified.
- [x] Internally consistent (no contradictions)? Verified — payment-on-Tab-1 acknowledged, default-disabled mitigation noted.
- [x] Scope decomposable into a single implementation plan? Yes — ~10 file touches, ~250 LOC including tests (revised upward after architect review).
- [x] Ambiguity check? Two minor unknowns flagged ("out of scope"): per-regime tinting and density slider. Both deferred deliberately.

### Architect review (2026-04-26) — all 4 HIGH and 3 MEDIUM issues addressed in this revision

| # | Severity | Issue | Resolution |
|---|---|---|---|
| 1 | HIGH | `_mc_payment_initiate` positional indexing breaks if `"bub"` inserted mid-`_MC_TABS` | "Files touched" row now mandates **append** at END; State block must also append |
| 2 | HIGH | `_mc_controls.py` has 7 prefix loops, only 1 originally noted | "Files touched" row now enumerates all 7 |
| 3 | HIGH | `_get_bubble_fig` doesn't route through `_get_mc_or_cached`; `mc_cached` dict would break JSON serialization | New `_get_mc_bubble_fig` wrapper required; `update_bubble` switches based on `mc_enabled` |
| 4 | HIGH | `_POST_RESTORE_TRIGGERS` won't catch new `bub-mc-*` Inputs | Plan must add every new bub-mc-* Input ID to the set |
| 5 | MEDIUM | `_build_bubble_figure_from_state` doesn't gate on MC | New row in "Files touched" — gate to None when `bub-mc-enable` truthy |
| 6 | MEDIUM | LPPL master + default n_freqs=[3] silently lands in paid mode | "Free vs paid" section now explicitly acknowledges and accepts the trap |
| 7 | MEDIUM | Perf claim "<2s" is unverified and conflates first-paint vs per-event | "Performance" section now distinguishes the two and gives a concrete benchmark target (1500ms cold) |
| 8 | LOW | `hoverinfo="skip"` semantics framing | Helper section now clarifies "skip" vs "none" |
| 9 | LOW | Snapshot bit-drift wording | "Files touched" row now mandates append at END of bubble section |
