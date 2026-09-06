#!/usr/bin/env python3
"""Sabotage-test the callback-introspection guards: can they still fail?

Between 2026-04 and 2026-09-06 three of these tests were fully vacuous and
two inspected a fraction of what they claimed, because they walked only
`app.callback_map` (clientside registrations) and parsed multi-output keys
with a naive `key.split("...")`.  They passed for free.  See
`docs/superpowers/agent-reports/2026-09-06-orphan-guard-and-syntax-check-recon.md`.

The tests now carry non-vacuity assertions, which catch a regression to
*zero matches*.  This script checks the stronger property: that each guard
actually FAILS when the defect it exists to catch is present.  A guard that
inspects the right callbacks but no longer detects the defect would pass
every non-vacuity assertion and still be worthless.

Each case injects one defect into `dash._callback.GLOBAL_CALLBACK_MAP`,
runs the guard, and asserts it raises.  Every injection is undone.

Run it after touching `_all_callbacks`, `_split_output_key`, or any of the
guards themselves:

    btc_venv/bin/python3 scripts/sabotage_callback_guards.py

Exits 0 if every guard behaved as expected, 1 otherwise.  Not part of the
pytest suite on purpose: it mutates a process-global registry, and the
suite runs under `-n logical` where that would be shared with whatever else
lands in the same worker.
"""
import os
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
os.environ.setdefault("TESTING", "1")
sys.path.insert(0, str(REPO / "btc_web"))
sys.path.insert(0, str(REPO))

import app  # noqa: F401,E402  — registers every callback
import _app_ctx  # noqa: E402
from dash import _callback as dash_callback  # noqa: E402

import test_no_orphan_callbacks as TNO  # noqa: E402
import test_callbacks as TC  # noqa: E402
import test_snapshot as TS  # noqa: E402

GM = dash_callback.GLOBAL_CALLBACK_MAP
RESULTS: list[bool] = []


def check(name, fn, expect_fail=True):
    from _pytest.outcomes import Failed          # pytest.fail() -> BaseException
    try:
        fn()
    except (AssertionError, Failed) as exc:
        got, detail = "FAILED", str(exc).splitlines()[0][:70]
    except Exception as exc:                      # noqa: BLE001
        got, detail = f"ERROR({type(exc).__name__})", str(exc)[:70]
    else:
        got, detail = "passed", ""
    want = "FAILED" if expect_fail else "passed"
    ok = got == want
    RESULTS.append(ok)
    print(f"  [{'OK ' if ok else 'BAD'}] {name}: expected {want}, got {got}  {detail}")


def entry(inputs=(), state=()):
    return {"inputs": [{"id": i, "property": "value"} for i in inputs],
            "state": [{"id": s, "property": "data"} for s in state]}


GUARDS = {
    "orphan refs": TNO.test_no_orphan_callback_refs,
    "introspection sees both": TNO.test_introspection_sees_both_callback_registries,
    "dup outputs":
        TC.TestNoDuplicateCallbackOutputs().test_no_unguarded_duplicate_outputs,
    "restore intermediate store":
        TC.TestNoDuplicateCallbackOutputs().test_restore_from_url_uses_intermediate_store,
    "nine chart gate":
        TC.TestSnapshotPendingChartGate()
          .test_nine_chart_callbacks_have_snapshot_pending_state,
    "pending writers":
        TS.TestSnapshotPendingGate().test_snapshot_pending_writers_have_allow_duplicate,
    "apply_globals gate":
        TS.TestSnapshotPendingGate().test_apply_globals_does_not_output_snapshot_pending,
}

print("Baseline — every guard must pass BEFORE sabotage:")
for _name, _fn in GUARDS.items():
    check(_name, _fn, expect_fail=False)

print("\nSabotage 1 — a SERVER callback referencing a component not in the layout:")
GM["sabotage-out.data"] = entry(inputs=["totally-bogus-id-xyz"])
check("orphan refs catches it", GUARDS["orphan refs"])
del GM["sabotage-out.data"]

print("\nSabotage 2 — two SERVER callbacks on one output, neither allow_duplicate:")
GM["collide-me.data"] = entry(inputs=["url"])
GM["..collide-me.data...second-out.data.."] = entry(inputs=["url"])
check("dup outputs catches it", GUARDS["dup outputs"])
del GM["collide-me.data"], GM["..collide-me.data...second-out.data.."]

print("\nSabotage 3 — a chart callback loses State('snapshot-pending'):")
victim = next(k for k in GM
              if TNO._split_output_key(k)[0].split("@")[0] == "bubble-graph.figure")
saved = GM[victim]["state"]
GM[victim] = {**GM[victim],
              "state": [d for d in saved
                        if (d.get("id") if isinstance(d, dict) else None)
                        != "snapshot-pending"]}
check("chart gate catches it", GUARDS["nine chart gate"])
GM[victim] = {**GM[victim], "state": saved}

print("\nSabotage 4 — a snapshot-pending writer without allow_duplicate:")
GM["snapshot-pending.data"] = entry(inputs=["url"])
check("pending writers catches it", GUARDS["pending writers"])
del GM["snapshot-pending.data"]

print("\nSabotage 5 — restore_from_url writing a snapshot control directly:")
GM["..loaded-hash-store.data...bub-xrange.value.."] = entry(inputs=["url"])
check("restore guard catches it", GUARDS["restore intermediate store"])
del GM["..loaded-hash-store.data...bub-xrange.value.."]

print("\nSabotage 6 — apply_globals gaining a snapshot-pending output:")
_k = "..main-tabs.active_tab...palette-store.data...snapshot-pending.data.."
GM[_k] = entry(inputs=["x"])
check("apply_globals gate catches it", GUARDS["apply_globals gate"])
del GM[_k]

print("\nSabotage 7 — the key parser regresses to a naive split (defect D2):")
_real_split = TNO._split_output_key
TNO._split_output_key = lambda k: k.split("...")
check("non-vacuity guard catches parser regression", GUARDS["introspection sees both"])
TNO._split_output_key = _real_split

print("\nSabotage 8 — _all_callbacks regresses to clientside-only (defect D1):")
_real_all = TNO._all_callbacks
TNO._all_callbacks = lambda: dict(_app_ctx.app.callback_map)
check("non-vacuity guard catches map regression", GUARDS["introspection sees both"])
TNO._all_callbacks = _real_all

print(f"\n{sum(RESULTS)}/{len(RESULTS)} checks behaved as expected")
sys.exit(0 if all(RESULTS) else 1)
