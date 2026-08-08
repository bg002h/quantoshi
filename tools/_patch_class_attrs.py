"""Scoped, atomic in-place rewriting of fitted class attributes in `btc_core/`.

**Every new `tools/fit_*.py` that supports `--update` MUST use this module
rather than growing its own regex.** Roughly twenty fit tools each carried a
private copy of this logic, and a private copy is where the hazard lives:

    THE HAZARD. Attribute names in `btc_core/_simple.py` are NOT unique.
    `GompertzModel`, `LogisticSCurveModel` and `SaturatingPowerLawModel` all
    define `_t0`; `_K` and `_r` are shared by the first two; `_beta` is shared
    by `SaturatingPowerLawModel` and `StretchedExponentialModel`. A patch
    whose class scoping is absent, or is one line too generous at either edge,
    rewrites a DIFFERENT model's fitted parameters with this model's numbers.
    Nothing raises. Nothing logs. That model simply starts fitting worse, and
    the next person to notice is looking at a chart months later.

So the guards below are the point of the module, not decoration:

  * the search window runs from the `class {name}` line (matched at a line
    start, so a mention inside a docstring cannot open a window) to the
    newline immediately before the next top-level `class `;
  * that window must contain no other `class ` line;
  * every attribute regex is anchored to a line start at exactly four spaces
    of indentation, so an `self._t0 = ...` inside a method body is invisible
    to it, and each attribute must match exactly once inside the window;
  * the patched text must have the same line count as the original, at most
    `len(values)` lines may differ, and every differing line must lie inside
    the window;
  * the result must still compile;
  * and the write is atomic.

Every one of those raises `PatchError` explicitly. None is an `assert`:
`python -O` strips assert statements, and a stripped guard here is worse than
no guard, because it silently restores the exact failure mode the guard was
written to prevent.

FEWER CHANGES THAN EXPECTED IS NOT A FAILURE. Re-running a fit on unchanged
price data legitimately rewrites the same digits, so the ceiling is one-sided:
`len(changed) > len(values)` is a refusal, `len(changed) < len(values)` is
idempotence working.

WHITESPACE IS PRESERVED, NOT REGENERATED. The replacement substitutes only
the value text; the indentation, the padding around `=`, any trailing spaces
and any trailing comment are carried across byte-for-byte. This is what makes
a no-op patch a genuine no-op, and it is also why a rendered value may not
carry its own leading or trailing padding -- see `_render`.

Public API:

    locate_class_section(src, class_name) -> ClassSection
    patch_class_attrs(path, class_name, values, *, dry_run=False)
        -> list[ChangedLine]
    format_changes(changed) -> list[str]
    apply_and_report(path, class_name, values, *, dry_run=False)
        -> list[ChangedLine]      # the CLI front end: prints, never traces

`patch_class_attrs` is print-free so tests and library callers can use it.
`apply_and_report` is what a `--update` flag should call: it prints the scope
and the diff, and turns a `PatchError` into a one-line `SystemExit` instead of
a traceback.

ADOPTED BY: `tools/fit_spl.py`, `tools/fit_gompertz.py`,
`tools/fit_logistic.py` -- the three tools that patch a `_t0`. The other 18
`--update` tools still inline their own regex; migrating them is follow-up
**F-8** in `docs/superpowers/followups.md`, which also records that two of
them (`fit_lppl.py`, `fit_grdy.py`) have no class scoping at all and are
correct today only because their target class happens to come first in its
file. Tests: `btc_web/test_patch_class_attrs.py`.
"""
from __future__ import annotations

import os
import re
import tempfile
from typing import NamedTuple

__all__ = [
    "ChangedLine",
    "ClassSection",
    "PatchError",
    "apply_and_report",
    "format_changes",
    "locate_class_section",
    "patch_class_attrs",
]


class PatchError(RuntimeError):
    """Raised by every guard in this module.

    A distinct type so callers can catch it and exit with a message instead of
    a traceback, and so a test can assert that a guard -- rather than an
    incidental `ValueError` from somewhere else -- is what fired.
    """


class ClassSection(NamedTuple):
    """The byte range and line range owned by one top-level class.

    `start`/`end` are byte offsets into the source: `src[start:end]` is the
    class body, ending just before the newline that precedes the next
    top-level `class ` (or at end-of-file for the last class).

    `lo_line`/`hi_line` are 0-BASED line INDICES and the interval is CLOSED:
    `lo_line` is the `class ...` line itself, `hi_line` is the last line of
    the section -- typically a blank separator, and always one line BEFORE the
    next class. Measured on `btc_core/_simple.py` at 2026-08-07:

        356  class SaturatingPowerLawModel     <- lo_line
        437  ''                                 } trailing blanks
        438  ''                                 } hi_line
        439  class BrokenPowerLawModel          <- excluded, and must stay so

    The upper edge is the one that matters. `_t0` is not a unique attribute
    name, so a window that reached line 439 would let `--update` silently
    rewrite a different model's parameters. Do not loosen it.
    """

    start: int
    end: int
    lo_line: int
    hi_line: int


class ChangedLine(NamedTuple):
    """One rewritten line: 0-based index, plus its text before and after.

    A tuple rather than a bare string so callers can print a diff-ish summary
    (`format_changes`) *and* report line numbers, which a plain list of
    strings cannot support.
    """

    index: int
    old: str
    new: str


# A top-level class definition, anchored at a line start. Anchoring is not
# cosmetic: `src.find("class Foo")` also matches the words "class Foo" inside
# a docstring, and `SaturatingPowerLawModel`'s docstring names
# `LogisticSCurveModel` -- a prose mention that an unanchored search could
# mistake for a definition and open a window on.
def _class_re(class_name: str) -> re.Pattern[str]:
    return re.compile(rf"^class {re.escape(class_name)}\b", re.M)


def _attr_re(name: str) -> re.Pattern[str]:
    """Match `    <name> = <value>[  # comment]` at class-body indentation.

    Exactly four leading spaces, so `        self._t0 = ...` inside a method
    cannot match. `pre` absorbs all padding around `=`, `post` keeps any
    trailing whitespace and comment, and only `val` is replaced.
    """
    return re.compile(
        rf"^(?P<pre>    {re.escape(name)}[ \t]*=[ \t]*)"
        rf"(?P<val>[^#\n]*?)"
        rf"(?P<post>[ \t]*(?:#[^\n]*)?)$",
        re.M,
    )


def _render(value: str | float) -> str:
    """Value text to substitute. Floats get 6 decimals; strings pass through.

    A rendered value may NOT carry leading or trailing padding. That is not
    fussiness -- it is a bug this module was written to stop. Because `pre`
    already absorbs the whitespace that follows `=`, a value pre-padded to a
    fixed width (the `f"{v:>11.6f}"` that `tools/fit_gompertz.py` and
    `tools/fit_logistic.py` both used before migrating) ADDS its padding to
    the padding already in the file. Every `--update` run then pushed the
    number three columns further right than the last one.
    """
    if isinstance(value, bool):        # bool is an int; almost surely a typo
        raise PatchError(f"refusing to write a bool as a fitted value: {value!r}")
    text = f"{value:.6f}" if isinstance(value, (int, float)) else str(value)
    if not text:
        raise PatchError("refusing to write an empty value")
    if "\n" in text or "\r" in text:
        raise PatchError(f"value {text!r} spans lines; that changes the line count")
    if "#" in text:
        raise PatchError(f"value {text!r} contains '#'; it would comment out code")
    if text != text.strip():
        raise PatchError(
            f"value {text!r} carries leading/trailing whitespace. The existing "
            f"padding around '=' is preserved for you; adding more makes the "
            f"column creep on every run."
        )
    return text


def locate_class_section(src: str, class_name: str) -> ClassSection:
    """Byte + line range of `class {class_name}`, or raise `PatchError`."""
    hits = list(_class_re(class_name).finditer(src))
    if not hits:
        raise PatchError(f"could not find a top-level `class {class_name}`")
    if len(hits) > 1:
        raise PatchError(
            f"found {len(hits)} top-level definitions of `class {class_name}`; "
            f"refusing to guess which one carries the fitted parameters")
    start = hits[0].start()

    # "\nclass " is inherently line-anchored, and pointing `end` AT that
    # newline (not past it) is what keeps `hi_line` one line short of the next
    # class. See ClassSection's docstring.
    nxt = src.find("\nclass ", start + 1)
    end = nxt if nxt != -1 else len(src)
    section = src[start:end]

    stray = _stray_class_lines(section)
    if stray:
        raise PatchError(
            f"section scoping is wrong: the window for {class_name} also spans "
            f"{stray}")

    return ClassSection(start=start, end=end,
                        lo_line=src[:start].count("\n"),
                        hi_line=src[:end].count("\n"))


def _stray_class_lines(section: str) -> list[str]:
    """Any `class ` line inside the section other than its own header.

    Defence in depth: `locate_class_section` ends the window at the next
    `\\nclass `, so under LF line endings this cannot fire. It exists for the
    cases that reasoning misses -- a `\\r`-only line ending, a future change
    to how `end` is computed -- because the cost of being wrong here is a
    silently mis-patched model rather than an error.
    """
    return [ln for ln in section.splitlines()[1:] if ln.startswith("class ")]


def patch_class_attrs(
    path: str,
    class_name: str,
    values: dict[str, str | float],
    *,
    dry_run: bool = False,
) -> list[ChangedLine]:
    """Rewrite `values` as class attributes of `class_name` in `path`.

    Args:
        path:       source file to patch, e.g. `btc_core/_simple.py`.
        class_name: the top-level class that owns the attributes.
        values:     `{attr_name: value}`. Floats render with 6 decimals;
                    strings are substituted verbatim (see `_render`).
        dry_run:    run every guard and report the changes, write nothing.

    Returns:
        The changed lines, oldest-first by line number. EMPTY IS A SUCCESS:
        it means the file already carried these digits.

    Raises:
        PatchError: on any guard. The file is untouched whenever this raises.
    """
    if not values:
        raise PatchError("no values to patch")

    with open(path, encoding="utf-8") as f:
        src = f.read()

    sec = locate_class_section(src, class_name)
    section = src[sec.start:sec.end]

    for name, value in values.items():
        pat = _attr_re(name)
        hits = pat.findall(section)
        if len(hits) != 1:
            raise PatchError(
                f"expected exactly 1 `{name}` assignment in {class_name}, "
                f"found {len(hits)}")
        text = _render(value)
        section = pat.sub(
            lambda m, _t=text: f"{m.group('pre')}{_t}{m.group('post')}",
            section, count=1)

    new_src = src[:sec.start] + section + src[sec.end:]

    old_lines, new_lines = src.splitlines(), new_src.splitlines()
    if len(old_lines) != len(new_lines):
        raise PatchError(
            f"patch changed the line count "
            f"({len(old_lines)} -> {len(new_lines)}); refusing to write")

    changed = [ChangedLine(i, a, b)
               for i, (a, b) in enumerate(zip(old_lines, new_lines)) if a != b]
    outside = [c.index for c in changed
               if not (sec.lo_line <= c.index <= sec.hi_line)]
    if outside:
        raise PatchError(
            f"refusing to write: lines {outside} lie outside class "
            f"{class_name} (0-based lines {sec.lo_line}-{sec.hi_line}, "
            f"inclusive) -- this is the cross-model overwrite guard")
    if len(changed) > len(values):
        raise PatchError(
            f"refusing to write: expected at most {len(values)} changed lines, "
            f"got {[c.index for c in changed]}")

    try:
        compile(new_src, path, "exec")
    except SyntaxError as exc:
        raise PatchError(f"patched text does not compile: {exc}") from exc

    if changed and not dry_run:
        _atomic_write(path, new_src)
    return changed


def format_changes(changed: list[ChangedLine]) -> list[str]:
    """`changed` as unified-diff-ish lines, 1-based, ready to print."""
    out: list[str] = []
    for c in changed:
        out.append(f"    L{c.index + 1} -{c.old}")
        out.append(f"    L{c.index + 1} +{c.new}")
    return out


def apply_and_report(
    path: str,
    class_name: str,
    values: dict[str, str | float],
    *,
    dry_run: bool = False,
) -> list[ChangedLine]:
    """`patch_class_attrs` with the reporting a `--update` flag wants.

    Prints the window it is allowed to write to (so a reader of the log can
    see the scoping actually applied, not just trust that it did) and the
    per-line diff, and converts `PatchError` into `SystemExit` so a refusal
    reads as one line of explanation rather than a traceback.
    """
    try:
        with open(path, encoding="utf-8") as f:
            sec = locate_class_section(f.read(), class_name)
        changed = patch_class_attrs(path, class_name, values, dry_run=dry_run)
    except PatchError as exc:
        raise SystemExit(f"--update refused: {exc}") from exc

    verb = "would patch" if dry_run else "patching"
    print(f"\n--update: {verb} {path}")
    print(f"    scope: lines {sec.lo_line + 1}-{sec.hi_line + 1} "
          f"(class {class_name}), {len(changed)} line(s) changed")
    if not changed:
        print("    (no change -- the file already carries this fit)")
    for line in format_changes(changed):
        print(line)
    return changed


def _atomic_write(path: str, text: str) -> None:
    """Replace `path`'s contents with `text`, all-or-nothing.

    `open(path, "w")` truncates the target the instant it succeeds, so a
    Ctrl-C, a full disk, or a crash between truncate and flush leaves
    `btc_core/_simple.py` TRUNCATED -- every model in it gone, not just the
    three constants being rewritten. Writing a sibling temp file and renaming
    it over the target makes the swap a single atomic operation: the file is
    either the old text or the new one, never a prefix of either.

    The temp file is created in the target's own directory because os.replace
    is only atomic within a filesystem. fsync before the rename so a power
    loss cannot leave the renamed inode pointing at unflushed data.
    """
    d = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = tempfile.mkstemp(dir=d, prefix=os.path.basename(path) + ".",
                               suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        # mkstemp creates 0600; carry the original file's mode across.
        try:
            os.chmod(tmp, os.stat(path).st_mode & 0o7777)
        except OSError:
            pass
        os.replace(tmp, path)
    except BaseException:
        # Includes KeyboardInterrupt -- the whole point is that an interrupted
        # run leaves the original untouched and no debris behind.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
