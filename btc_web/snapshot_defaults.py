# btc_web/snapshot_defaults.py
"""Single source of truth for control default values across all 206 entries
in _SNAPSHOT_CONTROLS.

Phase 1 (2026-04-24): consolidation only. q3: encoding unchanged.

Conventions
-----------
- Keys are "{component_id}:{property}" matching _SNAPSHOT_CONTROLS exactly.
- Values are the WIDGET representation (what dcc.X(value=...) accepts).
- Translation widget->figure-params lives in tab_defaults.py adapters
  (e.g. bub-xrange:value [2010,2033] -> xmin=2010, xmax=2033;
        bub-yrange:value [-1.5, 6.05] -> ymin=10**-1.5, ymax=10**6.05).
- Dynamic-default fields (current year, today's date, live BTC price)
  use a static placeholder here; ALWAYS_ENCODE forces emission in q4:
  encoding so the link author's value at link-creation time is preserved.
- See spec docs/superpowers/specs/2026-04-24-snapshot-defaults-ssot-
  and-v4-encoding-design.md.
"""

from __future__ import annotations
import hashlib
import json
from typing import Any

# Populated in Task 2 from the live layout.
SNAPSHOT_DEFAULTS: dict[str, Any] = {}

# Controls whose default at link-creation time is genuinely dynamic
# (current_year, today, live BTC price). q4: encoder emits these
# unconditionally even when matching the static placeholder.
ALWAYS_ENCODE: frozenset[str] = frozenset({
    "hm-entry-yr:value",
    "hm-entry-q:value",
    "lev-date:date",
    "lev-price:value",
})


def _compute_snapshot_defaults_fingerprint() -> str:
    """8-char SHA256 over SNAPSHOT_DEFAULTS values, ordered by
    _SNAPSHOT_CONTROLS. Stable under benign dict-literal reorderings."""
    from snapshot import _SNAPSHOT_CONTROLS
    h = hashlib.sha256()
    for cid, prop in _SNAPSHOT_CONTROLS:
        val = SNAPSHOT_DEFAULTS.get(f"{cid}:{prop}")
        h.update(json.dumps(val, sort_keys=True).encode())
        h.update(b"\x00")
    return h.hexdigest()[:8]


def get(key: str, fallback: Any = None) -> Any:
    return SNAPSHOT_DEFAULTS.get(key, fallback)
