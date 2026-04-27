"""Site-wide time-axis configuration (Phase 1 plumbing).

Loads quantoshi.toml at import time and exposes axis-derived constants
plus calendar-to-t conversion helpers. This module is the single source
of truth for which axis the canonical site model was trained on.

Imported by: btc_core, tools/build_*.py, btc_web/_app_ctx, btc_web/cache,
btc_web/snapshot_defaults.

NOT imported by: btc_web/engines/custom_fit (CTA stays per-fit
user-controlled; verified by integration test).

Spec: docs/superpowers/specs/2026-04-26-time-basis-toggle-design.md §3.2
"""
from __future__ import annotations

import datetime as _dt
import logging
import sys
from pathlib import Path
from typing import Literal, Optional

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover — prod is Py 3.12, dev is 3.14
    import tomli as tomllib

_LOG = logging.getLogger("time_basis")

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TOML_PATH = _REPO_ROOT / "quantoshi.toml"

_DEFAULTS = {
    "time_basis": "calendar",
    "block_origin": 20188,        # last block of 2009-07-25 UTC (RPC-verified)
    "blocks_per_year": 52596,     # 144 × 365.25
}


def _load_config(path: Optional[Path] = None) -> dict:
    """Load quantoshi.toml, falling back to _DEFAULTS if missing.

    Public for testing — production callers should use the module-level
    constants below, which are computed once at import time.
    """
    p = path if path is not None else _TOML_PATH
    if not p.exists():
        _LOG.warning("time_basis: %s not found; using defaults (calendar)", p)
        return dict(_DEFAULTS)
    with open(p, "rb") as f:
        cfg = tomllib.load(f)
    return {**_DEFAULTS, **cfg}


_cfg = _load_config()

TIME_BASIS: Literal["calendar", "block"] = _cfg["time_basis"]
T_ORIGIN_DATE: _dt.date = _dt.date(2009, 7, 25)
T_ORIGIN_BLOCK: int = int(_cfg["block_origin"])
T_LABEL: str = "years" if TIME_BASIS == "calendar" else "blocks"
T_PER_YEAR: float = 1.0 if TIME_BASIS == "calendar" else float(_cfg["blocks_per_year"])
T_MIN: float = 1.0 if TIME_BASIS == "calendar" else float(_cfg["blocks_per_year"])

_LOG.info(
    "time_basis: %s (T_LABEL=%s, T_PER_YEAR=%g, T_ORIGIN_BLOCK=%d)",
    TIME_BASIS, T_LABEL, T_PER_YEAR, T_ORIGIN_BLOCK,
)


def calendar_to_t(d: _dt.date) -> float:
    """Convert a calendar date to t in the active basis.

    Calendar mode: t = years since 2009-07-25 (`days / 365.25`).
    Block mode: t = projected block offset, using protocol-target rate
                (T_PER_YEAR blocks per year) since the calendar origin.

    Used at the simulator boundary (Citadel, MC year-stepping) to convert
    user-entered calendar years into model-axis t.
    """
    days = (d - T_ORIGIN_DATE).days
    years = days / 365.25
    return years if TIME_BASIS == "calendar" else years * T_PER_YEAR


def t_to_calendar(t: float) -> _dt.date:
    """Convert t (active basis) back to calendar date for chart display.

    Inverse of calendar_to_t (modulo integer-day rounding).
    """
    years = t if TIME_BASIS == "calendar" else t / T_PER_YEAR
    days = years * 365.25
    return T_ORIGIN_DATE + _dt.timedelta(days=days)
