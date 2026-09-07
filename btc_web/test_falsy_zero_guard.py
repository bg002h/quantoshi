"""`x or <non-zero number>` silently replaces a legitimate 0.

This class has bitten this codebase twice:

* `dca-sc-rate` / `sc-infl` — `float(x or default)` turned a 0 % rate into the
  default rate (recorded in CLAUDE.md's "Falsy-zero in callbacks" gotcha);
* `cfg_a_nlog or 1` in the HybPPL/EPPL resolvers — asking for **zero**
  oscillators rendered the one-oscillator model, so the config panel appeared
  to do nothing (2026-09-07, fixed in `_resolvers.py`).

Both were invisible: no error, no warning, just a wrong number or an
unchanged chart.

Scope is deliberately narrow. `x or []`, `x or ""`, `x or None` and
`x or 0` are all idiomatic and harmless — an empty list really is "nothing
selected", and a 0 fallback cannot mask a 0. Only a NON-ZERO NUMERIC default
can swallow a meaningful zero, and that is the single shape flagged here.
Measured over `btc_web/` when this guard was written: 563 `or` expressions
in total, of which 18 are this shape.

Every existing site is acknowledged below with the reason 0 cannot occur
there. A NEW one fails this test until someone writes down why it is safe —
which is the whole point: the two bugs above would both have had to be
justified in writing, and neither could have been.

Acknowledgements are keyed by (file, expression), not line number, so
unrelated edits do not churn them; changing an expression forces a fresh look.
"""
import ast
import os
import pathlib

import pytest

os.environ.setdefault("TESTING", "1")

_BTC_WEB = pathlib.Path(__file__).resolve().parent


# ── the detector ────────────────────────────────────────────────────────────

def _falsy_numeric_defaults(source: str, filename: str = "<str>"):
    """[(lineno, expression_source), ...] for every `A or <non-zero number>`.

    Booleans are excluded: `x or True` is not a numeric default despite
    `bool` being a subclass of `int`.
    """
    out = []
    for node in ast.walk(ast.parse(source, filename=filename)):
        if not (isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or)):
            continue
        right = node.values[-1]
        if not isinstance(right, ast.Constant):
            continue
        v = right.value
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        if v == 0:
            continue
        out.append((node.lineno, ast.unparse(node)))
    return out


def _scan_repo():
    """{(relative_path, expression): [line, ...]} across btc_web, sans tests."""
    found = {}
    for path in sorted(_BTC_WEB.rglob("*.py")):
        rel = path.relative_to(_BTC_WEB).as_posix()
        if path.name.startswith("test_") or rel.startswith(("mc_cache/", "assets/")):
            continue
        try:
            src = path.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        try:
            hits = _falsy_numeric_defaults(src, rel)
        except SyntaxError:
            continue
        for lineno, expr in hits:
            found.setdefault((rel, expr), []).append(lineno)
    return found


# ── acknowledged sites: why a 0 cannot arrive here ──────────────────────────
# Removing a site from the code means removing its entry: test_no_stale_
# acknowledgements keeps this list honest, the way _KNOWN_ORPHANS does.

_ACKNOWLEDGED: dict[tuple[str, str], str] = {
    # Degenerate-range guards: the operand is a computed span, and 0 span is
    # the thing being defended against (division / zero-height axis).
    ("figures/common.py", "abs(y_max) * 0.1 or 1.0"):
        "zero-span guard: a 0 range would collapse the axis",
    ("tasks.py", "mx - mn or 1"):
        "zero-span guard: mx == mn would divide by zero",
    ("utils.py", "hi - lo or 1"):
        "zero-span guard: hi == lo would divide by zero",

    # Calendar years — no control can emit year 0.
    ("callbacks/charts/__init__.py", "entry_yr or 2025"):
        "calendar year; the entry-year control cannot produce 0",
    ("callbacks/charts/__init__.py", "start_yr or 2033"):
        "calendar year; the start-year dropdown cannot produce 0",
    ("callbacks/citadel_scenarios.py", "start_yr or 2035"):
        "calendar year; the scenario start-year cannot produce 0",

    # Dropdown-backed counts whose option lists start at 1 or higher.
    ("callbacks/charts/__init__.py", "mc_p.get('mc_sims') or 100"):
        "sims dropdown offers 1-3200; 0 is not selectable",
    ("callbacks/charts/__init__.py", "mc_p.get('mc_bins') or 5"):
        "bins are >= 1; 0 bins is not an option",
    ("mc_overlay.py", "p.get('mc_sims') or 200"):
        "sims dropdown offers 1-3200; 0 is not selectable",
    ("mc_overlay.py", "p.get('mc_entry_q') or 10"):
        "entry-q options are [1, 10, 20, ... 90]; 0 is not selectable",
    ("callbacks/citadel_scenarios.py", "quantile or 0.25"):
        "quantiles are strictly between 0 and 1; exact 0 is not a quantile",

    # Tab-1 Occupancy: both controls are fixed option sets excluding 0.
    ("figures/occupancy.py", "p.get('occ_tail', 10) or 10"):
        "tail options are 5/10/25 %; 0 is not offered",
    ("figures/occupancy.py", "p.get('occ_window', 4) or 4"):
        "window options are 1/2/4 yr; 0 is not offered",

    # Environment / viewport values that are never legitimately 0.
    ("callbacks/charts/__init__.py", "viewport_width or 1200"):
        "a 0-px viewport is degenerate; fallback is intended",
    ("mc_cache.py", "os.cpu_count() or 4"):
        "os.cpu_count() returns None or >= 1, never 0",
    ("layout/mc_controls.py",
     "round(_app_ctx._HM_ENTRY_Q_DEFAULT / 10) * 10 or 50"):
        "a percentile rounding to 0 is below every offered option",
}


# ── the guard ───────────────────────────────────────────────────────────────

def test_no_unacknowledged_falsy_numeric_defaults():
    found = _scan_repo()
    unknown = sorted(k for k in found if k not in _ACKNOWLEDGED)
    assert not unknown, (
        "new `x or <non-zero number>` site(s) — a real 0 would be silently "
        "replaced. Either use `x if x is not None else default`, or add an "
        "entry to _ACKNOWLEDGED saying why 0 cannot occur here:\n  "
        + "\n  ".join(f"{f}:{found[(f, e)]}  {e}" for f, e in unknown))


def test_no_stale_acknowledgements():
    """An entry for a site that no longer exists means the list is rotting —
    and a future reader would trust a justification for nothing."""
    found = _scan_repo()
    stale = sorted(k for k in _ACKNOWLEDGED if k not in found)
    assert not stale, (
        "acknowledged sites that no longer exist — delete these entries:\n  "
        + "\n  ".join(f"{f}  {e}" for f, e in stale))


@pytest.mark.parametrize("reason", _ACKNOWLEDGED.values())
def test_every_acknowledgement_states_a_reason(reason):
    assert reason and len(reason) > 15, (
        "an acknowledgement must explain why 0 cannot occur, not just exist")


# ── the detector must actually detect ───────────────────────────────────────

def test_detector_catches_the_resolver_bug_shape():
    """The exact expression that shipped the 2026-09-07 config-panel bug."""
    src = ("def f(cfg_a_nlog, cfg_a_ncal):\n"
           "    return key(cfg_a_nlog or 1, cfg_a_ncal or 1)\n")
    exprs = [e for _, e in _falsy_numeric_defaults(src)]
    assert "cfg_a_nlog or 1" in exprs
    assert "cfg_a_ncal or 1" in exprs


def test_detector_catches_the_documented_callback_shape():
    """The other historical instance: float(x or default) on a rate."""
    src = "rate = float(sc_rate or 13.0)\n"
    assert [e for _, e in _falsy_numeric_defaults(src)] == ["sc_rate or 13.0"]


@pytest.mark.parametrize("src", [
    "v = toggles or []",
    "v = name or ''",
    "v = data or {}",
    "v = x or None",
    "v = infl or 0",
    "v = infl or 0.0",
    "v = flag or True",
    "v = a or b",
    "v = x or compute()",
])
def test_detector_ignores_harmless_shapes(src):
    assert _falsy_numeric_defaults(src) == [], (
        f"{src!r} is idiomatic and cannot mask a meaningful 0")


def test_the_fixed_resolver_is_clean():
    """Regression: the site that caused the bug must stay fixed."""
    src = (_BTC_WEB / "callbacks" / "charts" / "_resolvers.py").read_text()
    exprs = [e for _, e in _falsy_numeric_defaults(src)]
    assert not [e for e in exprs if "nlog" in e or "ncal" in e], (
        f"a component-count fallback is back: {exprs}")
