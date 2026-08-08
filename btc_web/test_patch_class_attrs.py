"""Tests for `tools/_patch_class_attrs.py`, the shared `--update` patcher.

THE BUG THIS FILE EXISTS FOR. Attribute names in `btc_core/_simple.py` are not
unique: `GompertzModel`, `LogisticSCurveModel` and `SaturatingPowerLawModel`
all define `_t0`. A fit tool whose class scoping is missing, or one line too
generous at the upper edge, rewrites a *different* model's fitted parameters.
Nothing raises and nothing logs -- that model just quietly starts fitting
worse. `test_scoping_leaves_sibling_class_byte_identical` is the test for it,
and `test_unscoped_regex_corrupts_the_sibling_class` is its counterfactual:
it runs the naive whole-file regex that the pre-migration tools used and
asserts that it *does* corrupt the sibling, so the first test cannot pass for
some reason other than the scoping.
"""
from __future__ import annotations

import os
import re
import shutil
import stat
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "tools"))
import _patch_class_attrs as P  # noqa: E402

# Two classes with COLLIDING attribute names, in the same shape the real file
# uses: four-space class-body assignments, a `self._t0` reference inside a
# method (which must never be touched), padding around `=`, a trailing
# comment, and a third class after them so the section has an upper edge.
FIXTURE = '''\
"""Fixture module -- mirrors btc_core/_simple.py's shape."""


class AlphaModel:
    """Alpha. Mentions class BetaModel in prose, which must not open a window."""

    name = "Alpha"

    # Fitted parameters
    _K  =             1.111111
    _r  =             2.222222
    _t0 =             3.333333

    def value(self, t):
        self._t0 = t          # 8-space: not a class attribute
        return self._K * self._r * self._t0


class BetaModel:
    """Beta."""

    name = "Beta"

    _K  =                7.777777
    _r  =                8.888888  # growth rate
    _t0 =                9.999999


class GammaModel:
    """Gamma has no fitted attributes at all."""

    name = "Gamma"
'''


@pytest.fixture()
def src_file(tmp_path: Path) -> Path:
    p = tmp_path / "_fixture.py"
    p.write_text(FIXTURE, encoding="utf-8")
    return p


def _section_of(text: str, class_name: str) -> str:
    """The raw text of one class, for byte-identity assertions.

    Computed independently of the module under test, but note the `^`: the
    first draft of this helper used `text.index(f"class {class_name}")` and
    matched the prose "class BetaModel" inside AlphaModel's docstring -- the
    very hazard `locate_class_section` anchors against.
    """
    m = re.search(rf"^class {re.escape(class_name)}\b", text, re.M)
    assert m is not None, f"no top-level class {class_name}"
    nxt = text.find("\nclass ", m.start() + 1)
    return text[m.start():nxt if nxt != -1 else len(text)]


# --------------------------------------------------------------------------
# The collision: the reason this module exists
# --------------------------------------------------------------------------

def test_scoping_leaves_sibling_class_byte_identical(src_file: Path):
    before = src_file.read_text(encoding="utf-8")
    changed = P.patch_class_attrs(
        str(src_file), "AlphaModel",
        {"_K": 4.5, "_r": 5.5, "_t0": 6.5})
    after = src_file.read_text(encoding="utf-8")

    assert len(changed) == 3
    # BetaModel also defines _K, _r and _t0. It must be untouched, byte for
    # byte -- including its trailing comment and its padding.
    assert _section_of(after, "BetaModel") == _section_of(before, "BetaModel")
    assert _section_of(after, "GammaModel") == _section_of(before, "GammaModel")
    # ...and Alpha really did change.
    assert "_t0 =             6.500000" in _section_of(after, "AlphaModel")
    assert "9.999999" in after          # Beta's _t0 survived verbatim


def test_unscoped_regex_corrupts_the_sibling_class(src_file: Path):
    """The counterfactual, so the test above cannot pass for the wrong reason.

    This is the pre-migration shape: one `re.sub` over the WHOLE file, with no
    class window. It rewrites BetaModel's `_t0` with AlphaModel's value and
    raises nothing. If a future edit lets the real patcher behave this way,
    the test above goes red while this one stays green.
    """
    before = src_file.read_text(encoding="utf-8")
    naive = before
    for name, val in (("_K", 4.5), ("_r", 5.5), ("_t0", 6.5)):
        naive = re.sub(rf"(    {name}\s*=\s*)[^#\n]+",
                       lambda m, _v=val: f"{m.group(1)}{_v:.6f}", naive)

    assert _section_of(naive, "BetaModel") != _section_of(before, "BetaModel")
    assert "9.999999" not in naive      # Beta's _t0 was silently overwritten

    # And the guarded patcher does not do that.
    P.patch_class_attrs(str(src_file), "AlphaModel",
                        {"_K": 4.5, "_r": 5.5, "_t0": 6.5})
    assert "9.999999" in src_file.read_text(encoding="utf-8")


def test_can_patch_the_second_of_two_colliding_classes(src_file: Path):
    before = src_file.read_text(encoding="utf-8")
    changed = P.patch_class_attrs(str(src_file), "BetaModel", {"_t0": 1.25})
    after = src_file.read_text(encoding="utf-8")

    assert len(changed) == 1
    assert _section_of(after, "AlphaModel") == _section_of(before, "AlphaModel")
    assert "_t0 =                1.250000" in _section_of(after, "BetaModel")


def test_method_body_assignment_is_not_a_class_attribute(src_file: Path):
    P.patch_class_attrs(str(src_file), "AlphaModel", {"_t0": 6.5})
    after = src_file.read_text(encoding="utf-8")
    assert "        self._t0 = t          # 8-space: not a class attribute" in after


# --------------------------------------------------------------------------
# Window arithmetic
# --------------------------------------------------------------------------

def test_window_stops_one_line_before_the_next_class(src_file: Path):
    src = src_file.read_text(encoding="utf-8")
    sec = P.locate_class_section(src, "AlphaModel")
    lines = src.splitlines()

    assert lines[sec.lo_line] == "class AlphaModel:"
    assert not lines[sec.hi_line].startswith("class ")
    assert lines[sec.hi_line + 1] == "class BetaModel:"


def test_last_class_in_file_has_a_window(src_file: Path):
    src = src_file.read_text(encoding="utf-8")
    sec = P.locate_class_section(src, "GammaModel")
    assert sec.end == len(src)
    assert sec.hi_line == src.count("\n")


def test_prose_mention_of_a_class_does_not_open_a_window(src_file: Path):
    """AlphaModel's docstring says "class BetaModel"; the window must not
    start there. An unanchored `src.find("class BetaModel")` would."""
    src = src_file.read_text(encoding="utf-8")
    sec = P.locate_class_section(src, "BetaModel")
    assert src[sec.start:].startswith("class BetaModel:")
    assert "_K  =                7.777777" in src[sec.start:sec.end]


def test_stray_class_guard_reports_offenders():
    """Defence in depth: unreachable under LF endings, tested directly."""
    assert P._stray_class_lines("class A:\n    x = 1\n") == []
    assert P._stray_class_lines("class A:\n    x = 1\nclass B:\n") == ["class B:"]


def test_missing_class_raises(src_file: Path):
    with pytest.raises(P.PatchError, match="could not find"):
        P.patch_class_attrs(str(src_file), "NoSuchModel", {"_t0": 1.0})


def test_duplicate_class_definition_raises(tmp_path: Path):
    p = tmp_path / "dup.py"
    p.write_text("class A:\n    _t0 = 1.0\n\n\nclass A:\n    _t0 = 2.0\n",
                 encoding="utf-8")
    with pytest.raises(P.PatchError, match="2 top-level definitions"):
        P.patch_class_attrs(str(p), "A", {"_t0": 3.0})


# --------------------------------------------------------------------------
# Change accounting
# --------------------------------------------------------------------------

def test_rewriting_identical_digits_is_a_no_op(src_file: Path):
    before = src_file.read_text(encoding="utf-8")
    changed = P.patch_class_attrs(
        str(src_file), "AlphaModel",
        {"_K": 1.111111, "_r": 2.222222, "_t0": 3.333333})
    assert changed == []
    assert src_file.read_text(encoding="utf-8") == before


def test_partial_change_is_allowed_not_a_failure(src_file: Path):
    """Fewer changes than values is idempotence, not a scoping failure."""
    changed = P.patch_class_attrs(
        str(src_file), "AlphaModel",
        {"_K": 1.111111, "_r": 2.222222, "_t0": 6.5})
    assert len(changed) == 1 < 3


def test_repeated_patches_do_not_creep_the_column(src_file: Path):
    """The bug the old `f"{v:>11.6f}"` rendering had: `pre` already absorbs the
    padding, so a pre-padded value added three more columns every run."""
    def offset(text: str) -> int:
        line = next(ln for ln in _section_of(text, "AlphaModel").splitlines()
                    if ln.startswith("    _t0"))
        return line.index("=")

    start = offset(src_file.read_text(encoding="utf-8"))
    for v in (4.0, 5.0, 6.0):
        P.patch_class_attrs(str(src_file), "AlphaModel", {"_t0": v})
        text = src_file.read_text(encoding="utf-8")
        line = next(ln for ln in text.splitlines() if ln.startswith("    _t0 ="))
        assert offset(text) == start
        assert line == f"    _t0 =             {v:.6f}"


def test_trailing_comment_and_spacing_survive(src_file: Path):
    P.patch_class_attrs(str(src_file), "BetaModel", {"_r": 1.5})
    after = src_file.read_text(encoding="utf-8")
    assert "    _r  =                1.500000  # growth rate" in after


def test_dry_run_reports_but_writes_nothing(src_file: Path):
    before = src_file.read_text(encoding="utf-8")
    changed = P.patch_class_attrs(str(src_file), "AlphaModel",
                                  {"_t0": 6.5}, dry_run=True)
    assert len(changed) == 1
    assert src_file.read_text(encoding="utf-8") == before


def test_changed_line_carries_index_old_and_new(src_file: Path):
    src = src_file.read_text(encoding="utf-8")
    changed = P.patch_class_attrs(str(src_file), "AlphaModel",
                                  {"_t0": 6.5}, dry_run=True)
    (c,) = changed
    assert src.splitlines()[c.index] == c.old
    assert c.old.strip() == "_t0 =             3.333333"
    assert c.new.strip() == "_t0 =             6.500000"
    assert P.format_changes(changed) == [
        f"    L{c.index + 1} -{c.old}",
        f"    L{c.index + 1} +{c.new}",
    ]


# --------------------------------------------------------------------------
# Value guards
# --------------------------------------------------------------------------

def test_missing_attribute_raises(src_file: Path):
    with pytest.raises(P.PatchError, match="expected exactly 1 `_nope`"):
        P.patch_class_attrs(str(src_file), "AlphaModel", {"_nope": 1.0})


def test_duplicate_attribute_in_one_class_raises(tmp_path: Path):
    p = tmp_path / "dup_attr.py"
    p.write_text("class A:\n    _t0 = 1.0\n    _t0 = 2.0\n", encoding="utf-8")
    with pytest.raises(P.PatchError, match="found 2"):
        P.patch_class_attrs(str(p), "A", {"_t0": 3.0})


def test_no_values_raises(src_file: Path):
    with pytest.raises(P.PatchError, match="no values"):
        P.patch_class_attrs(str(src_file), "AlphaModel", {})


@pytest.mark.parametrize("bad, msg", [
    ("1.0\n2.0", "spans lines"),
    ("1.0  # sneaky", "contains '#'"),
    ("   1.0", "whitespace"),
    ("1.0   ", "whitespace"),
    ("", "empty value"),
    (True, "bool"),
])
def test_bad_values_are_refused(src_file: Path, bad, msg):
    before = src_file.read_text(encoding="utf-8")
    with pytest.raises(P.PatchError, match=msg):
        P.patch_class_attrs(str(src_file), "AlphaModel", {"_t0": bad})
    assert src_file.read_text(encoding="utf-8") == before


def test_value_that_breaks_the_syntax_is_refused(src_file: Path):
    before = src_file.read_text(encoding="utf-8")
    with pytest.raises(P.PatchError, match="does not compile"):
        P.patch_class_attrs(str(src_file), "AlphaModel", {"_t0": "1.0)"})
    assert src_file.read_text(encoding="utf-8") == before


def test_string_values_pass_through_verbatim(src_file: Path):
    P.patch_class_attrs(str(src_file), "AlphaModel", {"_t0": "np.nan"})
    assert "_t0 =             np.nan" in src_file.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# Atomic write
# --------------------------------------------------------------------------

def test_file_mode_is_preserved(src_file: Path):
    os.chmod(src_file, 0o640)
    P.patch_class_attrs(str(src_file), "AlphaModel", {"_t0": 6.5})
    assert stat.S_IMODE(os.stat(src_file).st_mode) == 0o640


def test_failed_replace_leaves_no_debris_and_no_damage(src_file: Path,
                                                       monkeypatch):
    before = src_file.read_text(encoding="utf-8")

    def boom(*a, **k):
        raise OSError("simulated failure at the rename")

    monkeypatch.setattr(P.os, "replace", boom)
    with pytest.raises(OSError, match="simulated failure"):
        P.patch_class_attrs(str(src_file), "AlphaModel", {"_t0": 6.5})

    assert src_file.read_text(encoding="utf-8") == before
    assert [p.name for p in src_file.parent.iterdir()] == [src_file.name]


# --------------------------------------------------------------------------
# apply_and_report -- the CLI front end the fit tools call
# --------------------------------------------------------------------------

def test_apply_and_report_prints_scope_and_diff(src_file: Path, capsys):
    changed = P.apply_and_report(str(src_file), "AlphaModel", {"_t0": 6.5})
    out = capsys.readouterr().out
    assert len(changed) == 1
    assert "(class AlphaModel), 1 line(s) changed" in out
    assert "-    _t0 =             3.333333" in out
    assert "+    _t0 =             6.500000" in out


def test_apply_and_report_says_so_when_nothing_changed(src_file: Path, capsys):
    assert P.apply_and_report(str(src_file), "AlphaModel",
                              {"_t0": 3.333333}) == []
    assert "already carries this fit" in capsys.readouterr().out


def test_apply_and_report_dry_run_writes_nothing(src_file: Path, capsys):
    before = src_file.read_text(encoding="utf-8")
    assert len(P.apply_and_report(str(src_file), "AlphaModel",
                                  {"_t0": 6.5}, dry_run=True)) == 1
    assert "would patch" in capsys.readouterr().out
    assert src_file.read_text(encoding="utf-8") == before


def test_apply_and_report_turns_a_guard_into_systemexit(src_file: Path):
    """A refusal should read as one line, not a traceback -- and must still
    be a non-zero exit, so a wrapper script cannot mistake it for success."""
    with pytest.raises(SystemExit) as exc:
        P.apply_and_report(str(src_file), "NoSuchModel", {"_t0": 1.0})
    assert "--update refused" in str(exc.value)
    assert exc.value.code != 0


# --------------------------------------------------------------------------
# Against the real file (copied; the original is never written)
# --------------------------------------------------------------------------

_REAL = _ROOT / "btc_core" / "_simple.py"


@pytest.mark.parametrize("class_name, attrs", [
    ("GompertzModel", ("_K", "_r", "_t0")),
    ("LogisticSCurveModel", ("_K", "_r", "_t0")),
    ("SaturatingPowerLawModel", ("_log10_L", "_t0", "_beta")),
])
def test_real_file_round_trip_is_a_no_op(tmp_path: Path, class_name, attrs):
    """Feeding back the values already in the file must change nothing.

    Guards the formatting contract as much as the scoping: if `_render` or the
    `pre`/`post` preservation drifted, this would report changes.
    """
    copy = tmp_path / "_simple.py"
    shutil.copy2(_REAL, copy)
    src = copy.read_text(encoding="utf-8")
    sec = P.locate_class_section(src, class_name)
    section = src[sec.start:sec.end]

    values = {}
    for a in attrs:
        m = P._attr_re(a).search(section)
        assert m is not None, f"{a} not found in {class_name}"
        values[a] = float(m.group("val"))

    assert P.patch_class_attrs(str(copy), class_name, values) == []
    assert copy.read_text(encoding="utf-8") == src


def test_real_file_windows_of_the_three_t0_classes_are_disjoint():
    """The three classes that share `_t0` must not overlap by even one line."""
    src = _REAL.read_text(encoding="utf-8")
    secs = {n: P.locate_class_section(src, n) for n in
            ("GompertzModel", "LogisticSCurveModel", "SaturatingPowerLawModel")}
    ranges = sorted((s.lo_line, s.hi_line) for s in secs.values())
    for (a_lo, a_hi), (b_lo, b_hi) in zip(ranges, ranges[1:]):
        assert a_hi < b_lo, f"windows overlap: {a_lo}-{a_hi} vs {b_lo}-{b_hi}"

    lines = src.splitlines()
    for name, sec in secs.items():
        assert lines[sec.lo_line] == f"class {name}(_ShrinkingBandsMixin):"
        assert not lines[sec.hi_line].startswith("class ")
        # exactly one `_t0` at class-body indentation inside each window
        assert len(P._attr_re("_t0").findall(src[sec.start:sec.end])) == 1
