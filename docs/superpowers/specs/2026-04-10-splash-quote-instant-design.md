# Instant Splash Modal Quote Population

**Date:** 2026-04-10
**Status:** Approved

## Problem

The Quantoshi welcome modal shows a quote + attribution. The attribution (`html.Div`) populates instantly via an inline `<script>` in `<head>`. The quote text (`dcc.Markdown`) takes 1-2 seconds to populate because:

1. `dcc.Markdown` renders with `children=None` from the server.
2. React renders an empty markdown container.
3. An inline script tries to populate it via DOM manipulation, but React re-renders the component when the Dash clientside splash callback eventually fires with `children` set, wiping our injection.

## Solution

Pick a quote **server-side** in `_serve_layout()` (same hour-based seeded shuffle the JS uses) and set it as the initial `children` of the `dcc.Markdown`. The layout JSON arrives with the quote baked in. React renders it immediately on first paint. The clientside callback still runs later but sets the same value idempotently.

## Design

### Quote selection at layout time

In `btc_web/layout/__init__.py`, `_serve_layout()`:

```python
import time as _time, random as _rnd
from layout.splash import _SPLASH_QUOTES_JS
import json as _json

_all_quotes = _json.loads(_SPLASH_QUOTES_JS)  # [[quote, attr], ...]
# Same seed logic as the JS: floor(now_ms / 6h)
_seed = int(_time.time() * 1000 // (6 * 3600 * 1000))
_rng = _rnd.Random(_seed)
# Genesis is quotes[0]; shuffle the rest
_rest = _all_quotes[1:].copy()
_rng.shuffle(_rest)
_initial_quote = [_all_quotes[0]] + _rest  # first shown is always genesis
_qtext, _qattr = _initial_quote[0]
```

Then when building the layout, set:

```python
dcc.Markdown(id="splash-quote-text", children=f'"{_qtext}"', ...)
html.Div(id="splash-quote-attr", children=f"\u2014 {_qattr}", ...)
```

### Remove the inline script

Delete the inline `<script>` pre-population block from `index_string`. It's no longer needed — the quote arrives pre-populated in the layout JSON.

### Keep the clientside callback

The existing splash modal callback in `btc_web/callbacks/splash.py` still:
- Decides whether to show the modal (6-hour gate, dev skip, deep link skip, share hash skip)
- Sets `splash-modal.is_open` to True/False
- Sets the quote text via `Output("splash-quote-text", "children")` — same value the server already set, so no visible change

The callback becomes a no-op for the quote text on initial load (same value), but still works for:
- The "Next" button cycling (different callback)
- Subsequent visits

### Seed consistency

The JS seed is `Math.floor(Date.now() / (6*3600*1000))`. The Python seed uses the same formula. Both run on the same hour bucket, so they pick the same quote (most of the time — they could drift by 1 bucket across an hour boundary, but this is cosmetic).

Simpler alternative: skip seed matching entirely. Server picks any quote at layout time. If the callback later overrides with a different quote, that's acceptable (still no visible flash since both values arrive via React, not via DOM mutation).

**We'll use the simpler alternative.** Server picks a quote deterministically per request (fresh random each time), callback may override later with its own seeded pick — but since they both write to `children` via React props, there's no flash.

## Files

| File | Action |
|------|--------|
| `btc_web/layout/__init__.py` | Modify: pick quote in `_serve_layout`, pass as `children` to the Markdown and attribution components; remove inline `<script>` pre-population block |
| `btc_web/callbacks/splash.py` | No change — still populates quote on its own schedule |
| `btc_web/layout/splash.py` | No change — already exports `_SPLASH_QUOTES_JS` |

## Success criteria

- Welcome modal opens with quote text fully populated on first paint
- No visible flash or "pop" of text appearing after modal is visible
- Attribution still populates immediately (unchanged)
- Clickable links in quotes still work (dcc.Markdown still parses them)
- Deep links (/faq, /9.4, etc.) still skip the modal entirely
- Share hash links still skip the modal
