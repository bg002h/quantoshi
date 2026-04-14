#!/usr/bin/env python3
"""Pre-compute the Custom Time Axis $1M projection table.

The table shows, for every combination of
  - 6 calendar t₀ presets (whitepaper, genesis, optimal, NLS, pizza, mtgox)
  - 4 weightings (unweighted, 1/t, 1/√t, log-t density)
  - 4 models (PL, QR Q50%, BM floor, Exp)
the fit's log-log exponent (`b`) and the projected calendar month at which
Bitcoin price first reaches $1,000,000 USD.

Output is written to btc_web/_projection_table.json. The modal at
btc_web/callbacks/custom_time.py::_build_projection_table loads this file
instead of running 96 fits on every user click.

By default the script skips the rebuild if the existing JSON is less than
REBUILD_INTERVAL_DAYS old (28 days → roughly monthly). Pass --force to
override. The daily-update cadence (daily_update.sh) calls this without
--force so the table only refreshes on the first run of a new month.

Usage:
    python3 tools/build_projection_table.py          # monthly-ish cadence
    python3 tools/build_projection_table.py --force  # always rebuild
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

_LOG = logging.getLogger("build_projection_table")

_ROOT = Path(__file__).resolve().parent.parent
_OUTPUT = _ROOT / "btc_web" / "_projection_table.json"
_REBUILD_INTERVAL_DAYS = 28

_TARGET_LOG_P = 6.0
_MAX_YEARS_OUT = 991

_WEIGHTINGS = [
    ("none",        "Unweighted"),
    ("inv_t",       "1/t"),
    ("inv_sqrt_t",  "1/\u221at"),
    ("log_density", "log-t density"),
]


def _extract(r, t0_ts):
    """Return {'b': float_or_none, 'date': 'YYYY-MM'-or-marker} for a fit."""
    import pandas as pd
    if r is None or not r.params:
        return {"b": None, "date": "\u2014"}

    if r.name == "Exp":
        slope = r.params.get("slope")
        intercept = r.params.get("intercept")
        if slope is None:
            return {"b": None, "date": "\u2014"}
        if slope <= 0:
            return {"b": float(slope), "date": "\u2014"}
        t_hit = (_TARGET_LOG_P - intercept) / slope
    elif r.name == "QR":
        slopes = r.params.get("slopes", {})
        intercepts = r.params.get("intercepts", {})
        if 0.5 in slopes:
            q = 0.5
        elif slopes:
            q = min(slopes.keys(), key=lambda x: abs(x - 0.5))
        else:
            return {"b": None, "date": "\u2014"}
        slope = slopes[q]
        intercept = intercepts[q]
        if slope is None or slope <= 0:
            return {"b": float(slope) if slope is not None else None, "date": "\u2014"}
        t_hit = 10 ** ((_TARGET_LOG_P - intercept) / slope)
    else:
        slope = r.params.get("slope")
        intercept = r.params.get("intercept")
        if slope is None:
            return {"b": None, "date": "\u2014"}
        if slope <= 0:
            return {"b": float(slope), "date": "\u2014"}
        t_hit = 10 ** ((_TARGET_LOG_P - intercept) / slope)

    if t_hit > _MAX_YEARS_OUT:
        target_year = int(t0_ts.year + t_hit)
        return {"b": float(slope), "date": f">{min(target_year, 9999)}"}
    try:
        dt = t0_ts + pd.to_timedelta(t_hit * 365.25, unit="D")
    except Exception:
        return {"b": float(slope), "date": ">9999"}
    if dt.year < 2010:
        return {"b": float(slope), "date": "<2010"}
    return {"b": float(slope), "date": dt.strftime("%Y-%m")}


def _compute_table():
    """Run all 96 fits and return the structured dict."""
    # Minimal app bootstrap so custom_fit can eager-load
    sys.path.insert(0, str(_ROOT / "btc_web"))
    sys.path.insert(0, str(_ROOT))
    import pandas as pd  # noqa: F401 — used via _extract
    import app  # noqa: F401 — populates _app_ctx.app
    from engines import custom_fit as cf
    from _custom_time_presets import CAL_PRESETS

    models = [
        ("PL",       lambda fi: cf.fit_pl(fi)),
        ("QR Q50",   lambda fi: cf.fit_qr(fi, quantiles=(0.5,))),
        ("BM floor", lambda fi: cf.fit_bm_floor(fi)),
        ("Exp",      lambda fi: cf.fit_exp(fi)),
    ]

    weightings = []
    for wkey, wlbl in _WEIGHTINGS:
        rows = []
        for preset_key, d, preset_lbl in CAL_PRESETS:
            t0_iso = d.isoformat()
            t0_ts = pd.Timestamp(t0_iso)
            fi = cf.build_fit_input(
                scale="calendar", t0=t0_iso, weighting=wkey)
            cells = {}
            for mname, fn in models:
                try:
                    r = fn(fi)
                    cells[mname] = _extract(r, t0_ts)
                except Exception as exc:
                    cells[mname] = {"b": None,
                                     "date": f"err:{type(exc).__name__}"}
            rows.append({
                "preset": preset_key,
                "preset_label": preset_lbl,
                "t0": t0_iso,
                "cells": cells,
            })
        weightings.append({"key": wkey, "label": wlbl, "rows": rows})

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target_price_usd": int(10 ** _TARGET_LOG_P),
        "models": [m[0] for m in models],
        "weightings": weightings,
    }


def _fresh_enough() -> bool:
    """True if the existing output file is younger than REBUILD_INTERVAL_DAYS."""
    if not _OUTPUT.exists():
        return False
    try:
        blob = json.loads(_OUTPUT.read_text())
        ts = datetime.fromisoformat(blob["generated_at"])
        age_days = (datetime.now(timezone.utc) - ts).days
        _LOG.info("existing projection table is %d days old", age_days)
        return age_days < _REBUILD_INTERVAL_DAYS
    except Exception as exc:
        _LOG.warning("failed to read existing table (%s); rebuilding", exc)
        return False


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true",
                    help="rebuild even if the existing table is fresh")
    args = ap.parse_args()

    if not args.force and _fresh_enough():
        _LOG.info("projection table is fresh (<%d days); skipping",
                  _REBUILD_INTERVAL_DAYS)
        return

    _LOG.info("computing projection table (6 t\u2080 × 4 weightings × 4 models)")
    blob = _compute_table()

    tmp = _OUTPUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(blob, indent=2))
    tmp.replace(_OUTPUT)
    _LOG.info("wrote %s", _OUTPUT)


if __name__ == "__main__":
    main()
