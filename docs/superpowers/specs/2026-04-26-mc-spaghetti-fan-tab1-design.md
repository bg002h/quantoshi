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
| `btc_web/layout/bubble.py` | Insert `_mc_controls("bub", ...)` after the Projection Quantiles card |
| `btc_web/snapshot_defaults.py` | Add `bub-mc-*` defaults (mirrors existing `dca-mc-*`) |
| `btc_web/snapshot.py` | Add `bub-mc-*` to `_SNAPSHOT_CONTROLS`; add to `_TAB_CONTROLS["bubble"]` |
| `btc_web/tab_defaults.py` | Add MC keys to `BUBBLE` dict via `sd("bub-mc-*:value", ...)` |
| `btc_web/callbacks/mc_payment.py` | Add `"bub"` to `_MC_TABS`; add `bub-mc-years/start-yr/entry-q/model-src/price-val` States |
| `btc_web/callbacks/mc_controls.py` | Add `"bub"` to the MC body-toggle loop and the Display-Models-MC-injection loop |
| `btc_web/callbacks/charts/__init__.py` | Extend `update_bubble` with MC Inputs/States, call `_mc_setup`, render spaghetti via new helper |
| `btc_web/figures/bubble.py` | New `_add_mc_spaghetti(fig, paths, t_axis, ...)` helper |

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

`hoverinfo="skip"` keeps the price-scatter hover authoritative — paths are visual context, not interactive data.

## Failure modes

Match Tabs 3-5 exactly — no Tab-1-specific handling.

- `_HAS_MARKOV=False` → MC card hidden via existing placeholder pattern
- `bub-mc-enable=[]` → no fan, MC card body collapsed
- Cache miss + no payment token → invoice modal (existing payment flow)
- Resolver lands on non-master non-cached key → free-tier check fails → invoice
- Status messages reuse existing `mc-pay-status` infrastructure

## Performance

`_add_mc_spaghetti` adds ~100 traces to the bubble figure. Existing chart caches (L0/L1/L2) key on the full params dict including `mc_*` so subsequent identical requests cache-hit. Per-tab first-render trigger architecture means the spaghetti only computes when Tab 1 is active — no cross-tab overhead.

Mobile budget concern: 100 thin lines × Plotly first paint should stay <2s on iPhone Safari (existing chart_responsive.js handles trace-width tuning). Measure post-deploy; reduce `n_display` to 50 if needed.

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

- [ ] Spec covers all 6 open questions listed in `/mcideas`? Yes — Q1=C, Q2=mc-model-src dropdown, Q3=full parity, Q4=panel's years control, Q5=after Projection Quantiles, Q6=same as other tabs.
- [ ] No placeholders or "TBD"? Verified.
- [ ] Internally consistent (no contradictions)? Verified — payment-on-Tab-1 acknowledged, default-disabled mitigation noted.
- [ ] Scope decomposable into a single implementation plan? Yes — ~10 file touches, ~150 LOC including tests.
- [ ] Ambiguity check? Two minor unknowns flagged ("out of scope"): per-regime tinting and density slider. Both deferred deliberately.
