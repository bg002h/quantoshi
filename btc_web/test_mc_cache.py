"""Tests for MC cache SSOT layer (_parse_cache_filename, _INTENDED_KEYS,
intended_models, MASTER_TO_CACHED_FALLBACK, is_master_cached, stash/commit/
restore)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))


def test_parse_cache_filename_handles_multi_underscore():
    """Model keys with internal underscores (ecfg_1d_1u, cfg_2d_1u) parse correctly."""
    from mc_cache import _parse_cache_filename

    assert _parse_cache_filename("paths_ecfg_1d_1u_2028.npz") == ("paths", "ecfg_1d_1u", 2028)
    assert _parse_cache_filename("overlays_cfg_2d_1u_2031.npz.bak") == ("overlays", "cfg_2d_1u", 2031)
    assert _parse_cache_filename("paths_bub_2028.npz") == ("paths", "bub", 2028)
    assert _parse_cache_filename("overlays_lppl_2035.npz") == ("overlays", "lppl", 2035)


def test_parse_cache_filename_rejects_garbage():
    """Non-cache filenames return None."""
    from mc_cache import _parse_cache_filename

    assert _parse_cache_filename("paths_2028.npz") is None       # no model key
    assert _parse_cache_filename("random.txt") is None
    assert _parse_cache_filename("") is None
    assert _parse_cache_filename("paths_bub_2028") is None       # missing .npz
    assert _parse_cache_filename("paths_bub_abcd.npz") is None   # year not 4 digits


def test_intended_models_keys_match_intended_set():
    """Drift guard: intended_models(M).keys() == _INTENDED_KEYS."""
    import _app_ctx, app  # noqa: F401  - boots the app, registers PRICE_MODELS
    from btc_core import load_model_data
    from mc_cache import intended_models, _INTENDED_KEYS

    M = load_model_data(str(Path(__file__).resolve().parent.parent / "model_data.pkl"))
    assert set(intended_models(M).keys()) == _INTENDED_KEYS


def test_intended_models_post_swap_keys():
    """Spec contract: _INTENDED_KEYS reflects the post-swap target."""
    from mc_cache import _INTENDED_KEYS

    assert _INTENDED_KEYS == frozenset({"bub", "qr", "pl", "ecfg_1d_1u", "ef"})


def test_intended_models_short_names_match_dict_keys():
    """Each instantiated model's .short_name must equal its dict key
    (cache files are named from short_name; mismatch would break lookup)."""
    import _app_ctx, app  # noqa: F401
    from btc_core import load_model_data
    from mc_cache import intended_models

    M = load_model_data(str(Path(__file__).resolve().parent.parent / "model_data.pkl"))
    for key, model in intended_models(M).items():
        assert model.short_name == key, (
            f"short_name mismatch: dict key {key!r} vs model.short_name {model.short_name!r}")


def test_master_to_cached_fallback_keys_in_dropdown():
    """Every fallback key must be a valid dropdown master."""
    import _app_ctx, app  # noqa: F401  - layout.heatmap transitively needs _app_ctx initialized
    from mc_cache import MASTER_TO_CACHED_FALLBACK
    from layout.heatmap import _HM_PILL_MODELS_BASE

    valid_masters = set(_HM_PILL_MODELS_BASE) | {"ef", "u1"}
    for master in MASTER_TO_CACHED_FALLBACK:
        assert master in valid_masters, (
            f"{master!r} in MASTER_TO_CACHED_FALLBACK but not in dropdown options")


def test_master_to_cached_fallback_values_in_intended():
    """Every fallback value must be in _INTENDED_KEYS, EXCEPT the
    'lppl' transition artifact (see _lppl_transition_exemption)."""
    from mc_cache import MASTER_TO_CACHED_FALLBACK, _INTENDED_KEYS

    for master, fallback in MASTER_TO_CACHED_FALLBACK.items():
        if master == "lppl":
            continue  # transition artifact; tracked by separate test
        assert fallback in _INTENDED_KEYS, (
            f"MASTER_TO_CACHED_FALLBACK[{master!r}] = {fallback!r} "
            f"is not in _INTENDED_KEYS = {_INTENDED_KEYS}")


def test_master_to_cached_fallback_lppl_transition_exemption():
    """The 'lppl' entry is a transition artifact: lppl is no longer in
    _INTENDED_KEYS but the entry stays until the rebuild purges
    paths_lppl_*.npz from prod. After purge, edit the dict to remove
    this entry (and delete this test).
    """
    from mc_cache import MASTER_TO_CACHED_FALLBACK, _INTENDED_KEYS

    # Today: "lppl" → "lppl" remains. This test passes iff the comment
    # invariant in mc_cache.py is upheld.
    assert MASTER_TO_CACHED_FALLBACK.get("lppl") == "lppl"
    assert "lppl" not in _INTENDED_KEYS


@pytest.mark.skipif(
    not (Path(__file__).resolve().parent / "mc_cache").exists(),
    reason="mc_cache/ dir does not exist (fresh clone or test env)",
)
def test_cached_model_keys_match_disk_glob():
    """Live _CACHED_MODEL_KEYS matches what _parse_cache_filename
    extracts from real overlays_*.npz files in mc_cache/."""
    from mc_cache import _CACHED_MODEL_KEYS, CACHE_DIR, _parse_cache_filename

    expected = set()
    for f in CACHE_DIR.glob("overlays_*.npz"):
        parsed = _parse_cache_filename(f.name)
        if parsed is not None and parsed[0] == "overlays":
            expected.add(parsed[1])
    assert _CACHED_MODEL_KEYS == frozenset(expected), (
        f"disk has {expected}, _CACHED_MODEL_KEYS = {set(_CACHED_MODEL_KEYS)}")


def test_is_master_cached_direct_key(monkeypatch):
    """Sanity baseline: bub is bold when bub is in _CACHED_MODEL_KEYS."""
    import mc_cache
    monkeypatch.setattr(mc_cache, "_CACHED_MODEL_KEYS",
                        frozenset({"bub", "pl"}))
    assert mc_cache.is_master_cached("bub") is True


def test_is_master_cached_via_fallback(monkeypatch):
    """Master 'eppl' is bold when its alias 'ecfg_1d_1u' is on disk."""
    import mc_cache
    monkeypatch.setattr(mc_cache, "_CACHED_MODEL_KEYS",
                        frozenset({"ecfg_1d_1u"}))
    assert mc_cache.is_master_cached("eppl") is True


def test_is_master_cached_returns_false_when_uncached(monkeypatch):
    """Master with no direct cache and no usable fallback → not bold."""
    import mc_cache
    monkeypatch.setattr(mc_cache, "_CACHED_MODEL_KEYS", frozenset())
    assert mc_cache.is_master_cached("eppl") is False
    assert mc_cache.is_master_cached("hybppl") is False  # no fallback entry
    assert mc_cache.is_master_cached("bub") is False     # direct miss
