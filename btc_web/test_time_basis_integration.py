"""Phase 1 integration tests — time_basis plumbing across modules."""
from __future__ import annotations
import hashlib

import pytest


def test_cache_l1_prefix_includes_time_basis():
    """Cache key prefix carries the axis so calendar/block won't collide.

    cache.py uses neighbor-import (`import _app_ctx`, no btc_web. prefix)
    because gunicorn runs with btc_web/ on sys.path. Tests must match.
    """
    import sys
    sys.path.insert(0, "btc_web")
    import cache  # noqa: E402
    from time_basis import TIME_BASIS  # noqa: E402
    key = cache._cache_key("bub", '{"a": 1, "b": 2}')
    assert key.startswith(f"fig:{TIME_BASIS}:"), (
        f"cache key {key!r} should start with fig:{TIME_BASIS}:")


def test_cache_l0_fingerprint_includes_time_basis():
    """L0 pinned fingerprint hash input includes TIME_BASIS.

    Slice is [:12] (matches existing cache.py:134 — do NOT shorten to [:8]).
    """
    import sys
    sys.path.insert(0, "btc_web")
    import cache  # noqa: E402
    from time_basis import TIME_BASIS  # noqa: E402
    from tab_defaults import _DEFAULTS_HASH  # noqa: E402
    expected_input = f"{TIME_BASIS}:{cache._MODEL_FP}:{_DEFAULTS_HASH}"
    expected_fp = hashlib.md5(expected_input.encode()).hexdigest()[:12]
    assert cache._L0_FINGERPRINT == expected_fp


def test_calendar_block_cache_keys_differ(monkeypatch):
    """Same params but different TIME_BASIS yield different cache keys."""
    import sys
    sys.path.insert(0, "btc_web")
    import cache  # noqa: E402
    monkeypatch.setattr(cache, "TIME_BASIS", "calendar", raising=False)
    cal = cache._cache_key("bub", '{"a": 1}')
    monkeypatch.setattr(cache, "TIME_BASIS", "block", raising=False)
    blk = cache._cache_key("bub", '{"a": 1}')
    assert cal != blk
    assert ":calendar:" in cal
    assert ":block:" in blk


def test_snapshot_fingerprint_changes_when_time_basis_changes(monkeypatch):
    """Reserving the TIME_BASIS slot in the snapshot fingerprint hash."""
    from btc_web import snapshot_defaults as sd
    from btc_web import time_basis as tb
    monkeypatch.setattr(tb, "TIME_BASIS", "calendar")
    monkeypatch.setattr(sd, "TIME_BASIS", "calendar", raising=False)
    cal_fp = sd._compute_snapshot_defaults_fingerprint()
    monkeypatch.setattr(tb, "TIME_BASIS", "block")
    monkeypatch.setattr(sd, "TIME_BASIS", "block", raising=False)
    blk_fp = sd._compute_snapshot_defaults_fingerprint()
    assert cal_fp != blk_fp
    assert len(cal_fp) == 8
    assert len(blk_fp) == 8


def test_snapshot_fingerprint_calendar_value_is_stable():
    """Calendar mode fingerprint is deterministic given the current registry."""
    from btc_web import snapshot_defaults as sd
    fp1 = sd._compute_snapshot_defaults_fingerprint()
    fp2 = sd._compute_snapshot_defaults_fingerprint()
    assert fp1 == fp2


def test_model_data_meta_json_exists_and_has_required_fields():
    """Phase 1 emits a JSON metadata sidecar alongside model_data.pkl."""
    import json
    from pathlib import Path
    repo_root = Path(__file__).resolve().parent.parent
    meta_path = repo_root / "model_data_meta.json"
    assert meta_path.exists(), (
        f"{meta_path} should be written by tools/build_bm_model.py")
    with open(meta_path) as f:
        meta = json.load(f)
    assert meta["time_basis"] in ("calendar", "block")
    assert meta["t_label"] in ("years", "blocks")
    assert meta["t_per_year"] in (1.0, 52596.0)
    assert meta["t_origin"] is not None
    # Default deployment is calendar
    if meta["time_basis"] == "calendar":
        assert meta["t_label"] == "years"
        assert meta["t_per_year"] == 1.0
        assert meta["t_origin"] == "2009-07-25"


def test_model_data_meta_matches_active_time_basis():
    """The on-disk pkl metadata reflects the current TIME_BASIS config."""
    import json
    import sys
    from pathlib import Path
    sys.path.insert(0, "btc_web")
    from time_basis import TIME_BASIS  # noqa: E402
    repo_root = Path(__file__).resolve().parent.parent
    meta_path = repo_root / "model_data_meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    assert meta["time_basis"] == TIME_BASIS, (
        f"sidecar reports {meta['time_basis']!r} but TIME_BASIS is "
        f"{TIME_BASIS!r} — rebuild model_data.pkl after editing "
        f"quantoshi.toml")


def test_custom_fit_does_not_import_time_basis():
    """CTA stays per-fit user-controlled — must not couple to site axis.

    Spec §3.2 + §3.7. Design decision: the per-fit scale dropdown on
    Tab 1's Custom Time Axis panel is independent of the site-wide
    TIME_BASIS. Coupling would mean changing the admin TOML would
    silently change CTA fits — surprising and wrong.

    Two-layer check:
      1. AST walker catches static `import` / `from … import` forms.
      2. Substring scan catches dynamic forms (importlib.import_module,
         __import__, getattr(sys.modules,…)) that the AST walker misses.
    """
    import ast
    from pathlib import Path
    repo_root = Path(__file__).resolve().parent.parent
    src = (repo_root / "btc_web" / "engines" / "custom_fit.py").read_text()

    # Layer 1: static AST scan
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            assert mod != "time_basis" and mod != "btc_web.time_basis", (
                f"engines/custom_fit.py must not import time_basis "
                f"(found 'from {mod} import …') — spec §3.2"
            )
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "time_basis" not in alias.name, (
                    f"engines/custom_fit.py must not import time_basis "
                    f"(found 'import {alias.name}')"
                )

    # Layer 2: substring scan (catches dynamic imports the AST misses).
    # Strip comments and docstrings first so the assertion message itself
    # — which mentions time_basis — doesn't trip the test.
    import io, tokenize
    code_only = []
    for tok in tokenize.tokenize(io.BytesIO(src.encode()).readline):
        if tok.type not in (tokenize.COMMENT, tokenize.STRING,
                            tokenize.ENCODING, tokenize.NL,
                            tokenize.NEWLINE):
            code_only.append(tok.string)
    code_str = " ".join(code_only)
    assert "time_basis" not in code_str, (
        "engines/custom_fit.py must not reference time_basis even "
        "dynamically (importlib, __import__, etc.)"
    )
