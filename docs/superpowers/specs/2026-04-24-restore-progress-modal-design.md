# Restore-Progress Modal — Design

**Date:** 2026-04-24
**Status:** Approved to implement
**Depends on:** `docs/superpowers/specs/2026-04-24-single-redraw-per-snapshot-design.md` (shipped)
**Prior art:** welcome splash modal (`dbc.Modal(id="splash-modal", …)` in `layout/__init__.py`)

## Goal

When a user opens a share-link URL (e.g. `/1#q3:…`), show a blocking modal that prevents UI interaction until the active tab has been fully restored and its chart has rendered with restored state. The modal auto-dismisses when the gate (`snapshot-pending`) releases + one animation frame.

## Motivation

Restore on prod takes ~3–4 seconds end-to-end (decode + `apply_globals` + `apply_tab_{active}` + chart render). During that window, stale pre-injected controls look "broken" and the chart re-renders 1–2 times (gate + uirevision mitigate the visual churn but don't eliminate the delay). A blocking modal converts the delay from "broken" into "loading" — the perceptual Gulf-of-Execution fix.

## Non-goals

- No progress count ("34 of 69"), no phase labels, no per-tab list. Single spinner + short copy. If restore routinely exceeds 1 s after this ships, a follow-up can upgrade to per-control progress on top of the same modal.
- No changes to `apply_globals`, `apply_tab_*`, `restore_from_url`, the gate, or the safety timer. Zero new server Outputs.
- No blocking for non-share-link loads (plain `/1`, `/2`, etc.). Modal only appears for `#q[1-3]:…` hashes.
- No support for hash changes that happen *after* initial load (e.g. user pastes new share link mid-session). First-load-only.

## User-visible behavior

### What changes
- Visiting any Quantoshi URL with a valid share-hash (`#q1:…`, `#q2:…`, `#q3:…`) shows a centered, backdrop-static modal ~150 ms after navigation begins. Modal content: Quantoshi logo + small spinner + "Restoring your shared view…" one-liner.
- Modal is dismissible **only by the app itself** — `backdrop="static"`, `keyboard=False`. User cannot close it.
- Modal auto-dismisses when restore is complete (gate release + one animation frame). Fade-out takes ~250 ms (Bootstrap modal default).

### What stays the same
- Non-hash loads: no modal (layout is unchanged).
- Gate, safety timer, uirevision, all current restore internals unchanged.

### Fast-restore behavior (no flash)
- Modal open is **150 ms debounced**. If restore completes within 150 ms of navigation, the modal never opens. Sub-second restores show nothing — correct UX.

### Minimum display time
- Once opened, modal stays visible for at least **500 ms** regardless of when the gate releases. Prevents sub-50-ms flash-dismiss on near-cached restores that slip past the 150 ms open debounce.

### Hard fallback
- 4-second timer unconditionally closes the modal and clears local progress state. Protects against any path where the gate doesn't release (cold cache extremes, network errors, Dash callback failures).

## Architecture

Two clientside callbacks and one new modal. No Python callbacks, no new Stores, no new Outputs on existing server callbacks.

### Modal declaration (`btc_web/layout/__init__.py`)

Near the existing splash modal block, add:

```python
dbc.Modal([
    dbc.ModalBody([
        html.Div([
            html.Img(src="/assets/quantoshi_logo_wm.png",
                     style={"width": "80px", "height": "auto",
                            "marginBottom": "16px", "opacity": "0.9"}),
            dbc.Spinner(size="sm", color="primary", spinner_style={"margin": "0 8px"}),
            html.Div("Restoring your shared view…",
                     style={"fontSize": UI_FONT_BASE, "marginTop": "12px",
                            "color": theme.TEXT_COLOR}),
        ], style={"textAlign": "center", "padding": "24px"}),
    ]),
], id="restore-progress-modal", is_open=False, centered=True, size="sm",
   backdrop="static", keyboard=False),
```

### Clientside callback 1 — open-on-share-hash

Fires on `Input("url","hash")`. Uses `prevent_initial_call='initial_duplicate'` because of the allow_duplicate Output. Reads `window.location.hash` directly (reliable across Dash's pathname/hash initialization quirks).

```python
_app_ctx.app.clientside_callback(
    """
    function(hash) {
        var h = (hash || window.location.hash || '').replace(/^#/, '');
        var isShare = h.indexOf('q1:') === 0 ||
                      h.indexOf('q2:') === 0 ||
                      h.indexOf('q3:') === 0;
        if (!isShare) return window.dash_clientside.no_update;

        // Record nav time for min-display enforcement.
        window.__restoreNavTime = performance.now();

        // Debounce open by 150ms — fast restores never show the modal.
        if (window.__restoreOpenTimer) clearTimeout(window.__restoreOpenTimer);
        window.__restoreOpenTimer = setTimeout(function () {
            window.dash_clientside.set_props(
                'restore-progress-modal', { is_open: true });
            window.__restoreOpenTime = performance.now();
        }, 150);

        // Hard fallback: 4s close.
        if (window.__restoreFallback) clearTimeout(window.__restoreFallback);
        window.__restoreFallback = setTimeout(function () {
            window.dash_clientside.set_props(
                'restore-progress-modal', { is_open: false });
        }, 4000);

        return window.dash_clientside.no_update;
    }
    """,
    Output("restore-progress-modal", "is_open", allow_duplicate=True),
    Input("url", "hash"),
    prevent_initial_call='initial_duplicate',
)
```

### Clientside callback 2 — close-on-gate-release

Fires on `Input("snapshot-pending","data")`. When the gate flips to `False` (i.e. `apply_tab_{active}` has written its batch), close the modal after one animation frame + min-display enforcement.

```python
_app_ctx.app.clientside_callback(
    """
    function(pending) {
        if (pending === true) return window.dash_clientside.no_update;
        // Only close if we opened. Skip if debounce cancelled the open.
        if (!window.__restoreOpenTime) {
            // Cancel pending open if gate released before 150ms debounce.
            if (window.__restoreOpenTimer) {
                clearTimeout(window.__restoreOpenTimer);
                window.__restoreOpenTimer = null;
            }
            if (window.__restoreFallback) {
                clearTimeout(window.__restoreFallback);
                window.__restoreFallback = null;
            }
            return window.dash_clientside.no_update;
        }
        var elapsed = performance.now() - window.__restoreOpenTime;
        var minDisplay = 500;
        var delay = Math.max(0, minDisplay - elapsed);
        setTimeout(function () {
            requestAnimationFrame(function () {
                window.dash_clientside.set_props(
                    'restore-progress-modal', { is_open: false });
                window.__restoreOpenTime = null;
                if (window.__restoreFallback) {
                    clearTimeout(window.__restoreFallback);
                    window.__restoreFallback = null;
                }
            });
        }, delay);
        return window.dash_clientside.no_update;
    }
    """,
    Output("restore-progress-modal", "is_open", allow_duplicate=True),
    Input("snapshot-pending", "data"),
    prevent_initial_call=True,
)
```

### Data flow

1. User navigates to `/1#q3:<payload>`.
2. Layout renders. Modal is in DOM with `is_open=False`.
3. Callback 1 fires (`url.hash` initial fire via `'initial_duplicate'`). Detects share hash. Starts 150 ms open-timer + 4 s fallback-timer.
4. `restore_from_url` (existing server callback) decodes and writes `snapshot-state-store` + `snapshot-pending=True`.
5. `apply_globals` writes globals. Gate stays `True`.
6. Safety bump writes `{active}-first-render += 1`. `apply_tab_{active}` writes tab controls + `snapshot-pending=False`.
7. Callback 2 fires on `snapshot-pending=False`. Reads `window.__restoreOpenTime`.
   - If unset (restore finished before 150 ms debounce): cancel open-timer + fallback, do nothing.
   - Else: compute `max(0, 500 - elapsed)` delay, schedule `requestAnimationFrame` → close modal.
8. Modal fades out, user interacts.

### Fallback / failure paths

| Scenario | Outcome |
|---|---|
| Restore completes in <150 ms (cached, tiny payload) | Open-timer cancelled before it fires; modal never appears. |
| Restore completes between 150 ms and 500 ms | Modal opens; min-display delay holds it open to 500 ms total. |
| Restore completes between 500 ms and 4 s | Modal opens; closes when gate releases + 1 rAF. |
| Decode fails (`restore_from_url` returns `no_update` for state) | Gate never armed True → never flips False. 4 s fallback closes modal. |
| Any path where `snapshot-pending` never flips False | 4 s fallback closes modal. |
| User navigates mid-restore (route change) | Single-page-app: `url.hash` changes. Callback 1 re-runs; if still share hash, restarts timers. If not, no new timer set; existing timer still runs but `apply_tab_*` won't fire for stale state, so fallback closes. |

## Files to change

| File | Change |
|---|---|
| `btc_web/layout/__init__.py` | Add `dbc.Modal(id="restore-progress-modal", …)` near the splash modal. |
| `btc_web/callbacks/snapshot_cb.py` | Append two clientside callbacks at the end of the file (next to the existing safety-timer clientside). |

## Tests

| Test | What it checks |
|---|---|
| `test_restore_progress_modal_in_layout` | `restore-progress-modal` id appears in rendered layout. |
| `test_restore_progress_modal_backdrop_static` | Layout inspection confirms `backdrop="static"` and `keyboard=False`. |
| `test_open_callback_has_initial_duplicate` | Source-level grep on `snapshot_cb.py` confirms the open-clientside uses `prevent_initial_call='initial_duplicate'` (otherwise it won't fire on initial load with hash). |
| `test_close_callback_has_min_display` | Grep confirms `500` and `Math.max` appear in the close callback's JS body. |
| `test_open_callback_debounce` | Grep confirms `150` appears in the open callback's JS body. |
| `test_fallback_timer_4000ms` | Grep confirms `4000` appears in the open callback's JS body (setTimeout second arg). |
| `test_no_server_output_on_restore_progress_modal` | Walk `app.callback_map`; assert the modal's `is_open` Output is only written by clientside callbacks, no server callback writers. |

No existing tests should break. No Python server callback changes.

## Rollout

- Single commit. `dash-callback-reviewer` on the diff before push (per today's pattern).
- Deploy via `git push origin master && ssh root@89.167.70.45 "cd /opt/quantoshi && git pull && redis-cli FLUSHDB && systemctl restart quantoshi"`.
- Verify on prod: share link → modal appears briefly → dismisses when chart settles. Non-share-link navigation → no modal.
- Watch prod logs for any `nonexistent object` or new Dash errors.

## Out of scope

- Per-control progress (deferred; cheap upgrade on top of this modal).
- Phase labels / multi-step progress indicator (deferred).
- Non-first-load hash changes (user pastes new share link mid-session) — rare enough to punt.
- Reducing actual restore time (covered by separate specs / earlier work).
