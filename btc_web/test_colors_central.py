"""Lint test enforcing the color centralization invariant.

After the color centralization migration, no hex literal should appear
in btc_web/ except in:
  - btc_web/colors.py (the source of truth)
  - btc_web/assets/_colors_generated.css (generated artifact)
  - btc_web/assets/_colors_generated.js (generated artifact)
  - btc_web/test_*.py (test fixtures)
  - btc_web/assets/.deferred/*.js (easter-egg JS files, allowlisted)

Plus the generator script at tools/generate_color_artifacts.py and
this test file itself.

Spec: docs/superpowers/specs/2026-04-10-color-centralization-design.md
"""
import ast
import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BTC_WEB = _REPO_ROOT / "btc_web"
_TOOLS = _REPO_ROOT / "tools"

# Files that may legitimately contain hex literals
_ALLOWLIST = {
    _BTC_WEB / "colors.py",
    _BTC_WEB / "assets" / "_colors_generated.css",
    _BTC_WEB / "assets" / "_colors_generated.js",
    _BTC_WEB / "assets" / "bootstrap_flatly.min.css",  # vendor bundle (~331 hex literals)
    _TOOLS / "generate_color_artifacts.py",
    _BTC_WEB / "test_colors_central.py",
    # model_info.py contains one user-visible documentation string that spells out
    # a hex value for users: html.Span("orange (#e67e22), 3px for the drawn line").
    # This is genuinely unmigrable — removing the hex would change user-facing text.
    _BTC_WEB / "layout" / "model_info.py",
}

_ALLOWLIST_DIRS = {
    _BTC_WEB / "assets" / ".deferred",
    _BTC_WEB / "__pycache__",
}

# Vendor file patterns — any file matching is allowlisted regardless of path.
# Catches future minified vendor bundles dropped into assets/.
_VENDOR_PATTERNS = (
    re.compile(r"\.min\.css$"),
    re.compile(r"\.min\.js$"),
    re.compile(r"\.bundle\.css$"),
    re.compile(r"\.bundle\.js$"),
)

# Test fixtures with hardcoded color assertions are allowlisted as a class
_TEST_FILE_PATTERN = re.compile(r"^test_.*\.py$")

# Catches both #abcdef (6-digit) and #abc (3-digit) forms.
# Negative lookbehind/lookahead ensures we don't match a 6-digit form
# as the leading 3 chars of a longer string.
_HEX_PATTERN = re.compile(
    r"(?<![0-9a-fA-F#])#(?:[0-9a-fA-F]{6}|[0-9a-fA-F]{3})(?![0-9a-fA-F])"
)
_RGBA_PATTERN = re.compile(r'\brgba?\(\s*\d+\s*,\s*\d+\s*,\s*\d+(?:\s*,\s*[\d.]+)?\s*\)')


def _walk_btc_web():
    """Yield Path objects for every .py / .css / .js file in btc_web/
    that is NOT in the allowlist."""
    for path in _BTC_WEB.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in (".py", ".css", ".js"):
            continue
        if path in _ALLOWLIST:
            continue
        if any(parent in _ALLOWLIST_DIRS for parent in path.parents):
            continue
        if _TEST_FILE_PATTERN.match(path.name):
            continue
        if any(pat.search(path.name) for pat in _VENDOR_PATTERNS):
            continue
        yield path


def _strip_css_comments(src: str) -> str:
    """Remove /* ... */ comment blocks from CSS source."""
    return re.sub(r'/\*.*?\*/', '', src, flags=re.DOTALL)


def _strip_js_comments(src: str) -> str:
    """Remove /* ... */ and // ... comments from JS source.

    Strings are NOT stripped — hex literals legitimately live inside JS
    string defaults (e.g. plot_appearance.js DEFAULTS dict) and the lint
    must catch them.
    """
    src = re.sub(r'/\*.*?\*/', '', src, flags=re.DOTALL)
    src = re.sub(r'//[^\n]*', '', src)
    return src


def _find_hex_literals_outside_string_constants(path: Path) -> list[tuple[int, str]]:
    """Find hex literals in a file, excluding allowed contexts.

    For .py: hex literals INSIDE string constants are still flagged
    (because that's where they live as Python source). However hex
    inside docstrings (which Python represents as Constant nodes
    immediately under FunctionDef/ClassDef/Module) is excluded.

    For .css/.js: comments are stripped before scanning.
    """
    src = path.read_text()
    if path.suffix == ".py":
        # For .py files, walk the AST and find Constant(value=str) nodes.
        # Skip any string constant whose parent is a docstring slot.
        try:
            tree = ast.parse(src)
        except SyntaxError:
            return []
        hits = []
        # Build a set of (lineno, col) of docstring constants to exclude.
        doc_locations = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if (node.body and isinstance(node.body[0], ast.Expr)
                        and isinstance(node.body[0].value, ast.Constant)
                        and isinstance(node.body[0].value.value, str)):
                    doc_locations.add(node.body[0].value.lineno)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.lineno in doc_locations:
                    continue
                if _HEX_PATTERN.search(node.value):
                    for m in _HEX_PATTERN.finditer(node.value):
                        hits.append((node.lineno, m.group()))
        return hits
    elif path.suffix == ".js":
        cleaned = _strip_js_comments(src)
        hits = []
        for i, line in enumerate(cleaned.splitlines(), 1):
            for m in _HEX_PATTERN.finditer(line):
                hits.append((i, m.group()))
        return hits
    elif path.suffix == ".css":
        cleaned = _strip_css_comments(src)
        hits = []
        for i, line in enumerate(cleaned.splitlines(), 1):
            for m in _HEX_PATTERN.finditer(line):
                hits.append((i, m.group()))
        return hits
    return []


def _find_rgba_literals_in_python(path: Path) -> list[tuple[int, str]]:
    """Find rgba()/rgb() string literals in Python files via AST.

    The lint requires literal rgba(...) strings to be moved into colors.py
    OR converted to use _hex_alpha(constant, alpha) which produces the
    rgba() at runtime as a function return value, not as a source literal.

    AST inspection: walk every Constant(value=str) node and reject if its
    value matches the rgba/rgb pattern. This catches literal string forms
    only — function returns from _hex_alpha() are fine because they
    aren't string constants in the source.
    """
    if path.suffix != ".py":
        return []
    src = path.read_text()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            for m in _RGBA_PATTERN.finditer(node.value):
                hits.append((node.lineno, m.group()))
    return hits


def test_no_hex_literals_outside_colors_module():
    """No hex literal should appear outside colors.py + generated files."""
    leaks = []
    for path in _walk_btc_web():
        hits = _find_hex_literals_outside_string_constants(path)
        for lineno, hex_str in hits:
            leaks.append(f"{path.relative_to(_REPO_ROOT)}:{lineno} {hex_str}")
    assert not leaks, (
        "Hex literals found outside the centralized colors module:\n"
        + "\n".join(leaks)
        + "\n\nMove these to btc_web/colors.py and import."
    )


def test_no_rgba_literals_in_python():
    """No literal rgba(...) string in Python code. Use _hex_alpha(constant)
    or define a baked-alpha named constant in colors.py."""
    leaks = []
    for path in _walk_btc_web():
        if path.suffix != ".py":
            continue
        hits = _find_rgba_literals_in_python(path)
        for lineno, lit in hits:
            leaks.append(f"{path.relative_to(_REPO_ROOT)}:{lineno} {lit}")
    assert not leaks, (
        "Literal rgba()/rgb() strings found in Python source:\n"
        + "\n".join(leaks)
        + "\n\nReplace with _hex_alpha(named_constant, alpha) or add a "
        "baked-alpha named constant to btc_web/colors.py."
    )


def test_no_rgba_literals_in_css():
    """No literal rgba(...) in .css files. Use var(--qs-*) which references
    a baked-alpha named constant in colors.py."""
    leaks = []
    for path in _walk_btc_web():
        if path.suffix != ".css":
            continue
        cleaned = _strip_css_comments(path.read_text())
        for i, line in enumerate(cleaned.splitlines(), 1):
            for m in _RGBA_PATTERN.finditer(line):
                leaks.append(f"{path.relative_to(_REPO_ROOT)}:{i} {m.group()}")
    assert not leaks, (
        "Literal rgba()/rgb() values found in CSS:\n"
        + "\n".join(leaks)
        + "\n\nMove to btc_web/colors.py as a named constant and reference "
        "via var(--qs-...) from the generated _colors_generated.css."
    )


def test_generator_check_mode_passes():
    """Running tools/generate_color_artifacts.py --check should exit 0."""
    import subprocess
    result = subprocess.run(
        ["python", str(_TOOLS / "generate_color_artifacts.py"), "--check"],
        capture_output=True, text=True, cwd=str(_REPO_ROOT),
    )
    assert result.returncode == 0, (
        f"generator --check failed (drift detected):\n{result.stderr}"
    )


def test_palette_key_parity():
    """Every palette must have an identical TOP-LEVEL key set AND
    identical inner model_colors key set."""
    import sys
    sys.path.insert(0, str(_BTC_WEB))
    import colors

    # Top-level parity
    key_sets = {pkey: set(pdict.keys()) for pkey, pdict in colors.PALETTES.items()}
    all_keys = set.union(*key_sets.values())
    top_divergences = {}
    for pkey, keys in key_sets.items():
        missing = all_keys - keys
        if missing:
            top_divergences[pkey] = sorted(missing)
    assert not top_divergences, (
        "Palette top-level key divergences:\n"
        + "\n".join(f"  {pkey}: missing {keys}" for pkey, keys in top_divergences.items())
    )

    # Inner model_colors parity — every palette must have the same set
    # of model keys. Catches drift like "lp4 added to default but not cb-rg".
    mc_sets = {pkey: set(pdict["model_colors"].keys())
               for pkey, pdict in colors.PALETTES.items()}
    all_models = set.union(*mc_sets.values())
    mc_divergences = {}
    for pkey, keys in mc_sets.items():
        missing = all_models - keys
        if missing:
            mc_divergences[pkey] = sorted(missing)
    assert not mc_divergences, (
        "Palette model_colors key divergences:\n"
        + "\n".join(f"  {pkey}: missing {keys}" for pkey, keys in mc_divergences.items())
    )


def test_css_var_consistency():
    """Every var(--qs-*) referenced in style.css must be defined in
    _colors_generated.css."""
    gen_css = (_BTC_WEB / "assets" / "_colors_generated.css").read_text()
    style_css = (_BTC_WEB / "assets" / "style.css").read_text()
    defined = set(re.findall(r'(--qs-[a-z0-9-]+):', gen_css))
    referenced = set(re.findall(r'var\((--qs-[a-z0-9-]+)\)', style_css))
    undefined = referenced - defined
    assert not undefined, (
        f"style.css references {len(undefined)} undefined CSS variables: "
        + ", ".join(sorted(undefined))
    )


def test_constant_export_coverage():
    """Every uppercase string/dict/list constant in colors.py must be
    either exported (matched by generator) or in __skip_export__."""
    import sys
    sys.path.insert(0, str(_BTC_WEB))
    import colors
    skip = getattr(colors, "__skip_export__", frozenset())
    src = (_BTC_WEB / "colors.py").read_text()
    tree = ast.parse(src)
    declared_uppercase = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    val = node.value
                    if isinstance(val, ast.Constant) and isinstance(val.value, str):
                        declared_uppercase.add(target.id)
                    elif isinstance(val, (ast.Dict, ast.List, ast.Tuple)):
                        declared_uppercase.add(target.id)
    # Every declared uppercase name should be either in skip or accessible from colors module
    missing = []
    for name in declared_uppercase:
        if name in skip:
            continue
        if not hasattr(colors, name):
            missing.append(name)
    assert not missing, f"Constants not accessible from colors module: {missing}"
