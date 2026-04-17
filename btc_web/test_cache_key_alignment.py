"""Regression guard for cache-key alignment between tab_defaults and
the chart callbacks.

The L0/L1 figure cache relies on `json.dumps(params, sort_keys=True)` as
the cache key. `_prewarm_caches()` at worker startup computes the key from
`*_defaults()`; the chart callback computes it from State/Input values.
If the two param dicts have different keys, the prewarm figure is orphaned
and the user's first tab visit recomputes from scratch.

This test AST-parses each callback's `_get_*_fig(dict(...))` call to
extract the set of kwarg names the runtime passes in, and compares against
the set of keys the `*_defaults()` function emits (after the prewarm's
strip-dynamic-keys logic in `_compute_cache_key`).

A difference on ANY tab means first-visit cache hit rate is < 100%.
"""
import ast
import pathlib

import pytest

_CHARTS_INIT = pathlib.Path(__file__).parent / "callbacks" / "charts" / "__init__.py"


def _dict_call_keys(dict_call: ast.Call, locals_map: dict) -> set:
    """Extract the kwarg names from a `dict(k1=..., k2=..., *other_dict)` call.
    Positional args that reference another dict-assignment (e.g. `dict(shared_params, ...)`)
    are resolved via locals_map, which maps name → set of keys."""
    names = set()
    for pos in dict_call.args:
        if isinstance(pos, ast.Name) and pos.id in locals_map:
            names |= locals_map[pos.id]
        elif isinstance(pos, ast.Call):
            if (isinstance(pos.func, ast.Name) and pos.func.id == "dict"):
                names |= _dict_call_keys(pos, locals_map)
    for kw in dict_call.keywords:
        if kw.arg is not None:
            names.add(kw.arg)
        elif isinstance(kw.value, ast.Name):
            if kw.value.id == "mc_p":
                names.add("_MC_P_KEYS_")
            elif kw.value.id in locals_map:
                names |= locals_map[kw.value.id]
    return names


def _extract_kwargs(call_name: str) -> set:
    """Find `<call_name>(dict(...))` in charts/__init__.py, walking enclosing
    function body to resolve any `shared_params = dict(...)` assignments that
    feed into the call."""
    src = _CHARTS_INIT.read_text()
    tree = ast.parse(src)
    # Locate the call site first, then back up to its enclosing FunctionDef
    # so we can resolve locals like `shared_params = dict(...)`.
    for func in ast.walk(tree):
        if not isinstance(func, ast.FunctionDef):
            continue
        locals_map = {}
        found = None
        for stmt in ast.walk(func):
            # Capture `name = dict(...)` into locals_map
            if isinstance(stmt, ast.Assign):
                if (len(stmt.targets) == 1
                        and isinstance(stmt.targets[0], ast.Name)
                        and isinstance(stmt.value, ast.Call)
                        and isinstance(stmt.value.func, ast.Name)
                        and stmt.value.func.id == "dict"):
                    locals_map[stmt.targets[0].id] = _dict_call_keys(
                        stmt.value, locals_map)
            if (isinstance(stmt, ast.Call)
                    and isinstance(stmt.func, ast.Name)
                    and stmt.func.id == call_name
                    and stmt.args
                    and isinstance(stmt.args[0], ast.Call)):
                found = stmt
        if found is not None:
            # Also union any OTHER call sites to `call_name` in the same
            # function (some tabs have a fallback branch with a different
            # kwarg shape, e.g. heatmap's mc-fallback vs. default paths).
            keys = set()
            for stmt in ast.walk(func):
                if (isinstance(stmt, ast.Call)
                        and isinstance(stmt.func, ast.Name)
                        and stmt.func.id == call_name
                        and stmt.args
                        and isinstance(stmt.args[0], ast.Call)):
                    keys |= _dict_call_keys(stmt.args[0], locals_map)
            return keys
    raise AssertionError(f"Could not find {call_name}(dict(...)) in charts/__init__.py")


# The keys that `**mc_p` injects (from callbacks/mc_helpers.py::_mc_setup).
_MC_P_KEYS = {
    "mc_enabled", "mc_free_tier", "mc_stale",
    "mc_bins", "mc_regime", "mc_sims", "mc_years", "mc_freq",
    "mc_window", "mc_start_yr", "mc_entry_q", "mc_amount", "mc_infl",
    "mc_cached", "mc_model_src",
}

# Keys that appear in `_compute_cache_key` stripping logic — these are
# intentionally removed from the cache key even if the runtime passes them.
_KEY_STRIPS_BY_PREFIX = {
    "bub": set(),
    "hm": {"live_price"} | {f"mc_{k}" for k in ()},  # live_price stripped in _get_heatmap_fig
    "dca": set(),
    "ret": set(),
    "sc": set(),
}


def _prewarm_keys(pfx: str) -> set:
    """What keys does _compute_cache_key(pfx, defaults) produce?"""
    import os
    os.environ.setdefault("DEV", "1")
    import sys
    for p in ["btc_web", "."]:
        if p not in sys.path:
            sys.path.insert(0, p)
    from tab_defaults import (bubble_defaults, heatmap_defaults,
                              dca_defaults, retire_defaults, supercharge_defaults)
    from utils import _compute_cache_key
    defs = {
        "bub": bubble_defaults, "hm": heatmap_defaults, "dca": dca_defaults,
        "ret": retire_defaults, "sc": supercharge_defaults,
    }[pfx]()
    import json
    key = _compute_cache_key(pfx, dict(defs))
    return set(json.loads(key).keys()) - {"_day"}  # _day added by _get_bubble_fig only


def _runtime_keys(call_name: str, pfx: str) -> set:
    names = _extract_kwargs(call_name)
    if "_MC_P_KEYS_" in names:
        names.discard("_MC_P_KEYS_")
        names |= _MC_P_KEYS
    # Apply the same mc_* stripping logic _compute_cache_key does when
    # mc_enabled is falsy: all mc_* keys except none.
    names = {n for n in names if not n.startswith("mc_")}
    # Other per-tab strips
    names -= _KEY_STRIPS_BY_PREFIX.get(pfx, set())
    return names


@pytest.mark.parametrize("call_name,pfx", [
    ("_get_bubble_fig",      "bub"),
    ("_get_heatmap_fig",     "hm"),
    ("_get_dca_fig",         "dca"),
    ("_get_retire_fig",      "ret"),
    ("_get_supercharge_fig", "sc"),
])
def test_prewarm_key_matches_runtime_key(call_name, pfx):
    """The set of kwargs in _get_*_fig(dict(...)) must equal the keys that
    _compute_cache_key produces from *_defaults(). Asymmetry → cache miss
    on first tab visit."""
    runtime = _runtime_keys(call_name, pfx)
    prewarm = _prewarm_keys(pfx)

    runtime_only = runtime - prewarm
    prewarm_only = prewarm - runtime

    msg = []
    if runtime_only:
        msg.append(f"Keys in runtime but missing from defaults: {sorted(runtime_only)}")
    if prewarm_only:
        msg.append(f"Keys in defaults but not in runtime: {sorted(prewarm_only)}")
    assert not msg, f"Cache key drift on tab={pfx!r}: " + " | ".join(msg)
