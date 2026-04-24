"""Idempotent registry updater. Phase 1.

Computes today's fingerprint from SNAPSHOT_DEFAULTS. If the fingerprint
is already in btc_web/snapshot_defaults_registry.json, exits 0 without
modification. Else appends a new entry and trims oldest if length > 20.
"""
from __future__ import annotations
import copy
import datetime as dt
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "btc_web"))

from snapshot_defaults import (SNAPSHOT_DEFAULTS,
                               _compute_snapshot_defaults_fingerprint)

REGISTRY_PATH = os.path.join(ROOT, "btc_web",
                             "snapshot_defaults_registry.json")
CAP = 20


def main() -> int:
    fp = _compute_snapshot_defaults_fingerprint()
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH) as f:
            registry = json.load(f)
    else:
        registry = []
    fps = {entry["fp"] for entry in registry}
    if fp in fps:
        print(f"fingerprint {fp} already in registry; no change")
        return 0
    entry = {
        "fp": fp,
        "created_at": dt.date.today().isoformat(),
        "defaults": copy.deepcopy(SNAPSHOT_DEFAULTS),
    }
    registry.append(entry)
    if len(registry) > CAP:
        dropped = len(registry) - CAP
        registry = registry[-CAP:]
        print(f"dropped {dropped} oldest entries")
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2, sort_keys=True)
    print(f"appended fingerprint {fp}; registry now has {len(registry)} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
