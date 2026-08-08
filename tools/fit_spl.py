#!/usr/bin/env python3
"""Fit the Saturating Power Law (spl) model to Bitcoin price history.

    price   = L / (1 + (t/t0)^(-beta))
    log10 p = log10(L) - log10(1 + (t/t0)^(-beta))

Regenerates the three class attributes on `btc_core._simple.
SaturatingPowerLawModel`: `_log10_L`, `_t0`, `_beta`.

Usage:
    btc_venv/bin/python3 tools/fit_spl.py             # fit and print
    btc_venv/bin/python3 tools/fit_spl.py --update    # fit and patch btc_core/

THE OPTIMISER IS NOT DEFINED HERE. `tools/analyze_spl.py::fit_spl` already
implements exactly this objective and these bounds, and it is the tool that
regenerates every quantitative claim in the spec. Two optimisers over one
model drift apart silently, so this tool imports that one. What it adds is
(a) the `--update` write-back and (b) the bound-activity report, neither of
which the analysis tool has.

BOUNDS (spec section 4). `differential_evolution` box-bounds *coordinates*,
and the fit is parameterised on (A, beta, log10 t0) with
A = log10 L - beta*log10 t0. So `L` is a DERIVED quantity, not a coordinate,
and its admissible range [max observed price, $1000T/21e6] is imposed inside
`analyze_spl.fit_spl` as a penalty rather than a box.

WHY THE BOUND REPORT IS NOT OPTIONAL. A parameter resting on a bound is an
artifact of the bound, not a measurement of the data. That has to be visible
in the output rather than inferred by someone eyeballing the numbers, so
`--update` REFUSES to write when any bound is active: a pinned fit is a
finding about the spec, not something to work around by widening the box.
"""
from __future__ import annotations

import difflib
import inspect
import os
import re
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "btc_web"))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from model_toolkit.data import load_prices                      # noqa: E402
from scipy.optimize import curve_fit                            # noqa: E402

# The shared fitter and its constants. Do not re-declare these.
from analyze_spl import (                                       # noqa: E402
    BETA_MAX, CAP_FIT_USD, SUPPLY, T0_HI, T0_LO, T_MIN,
    fit_spl, spl_log10,
)

CORE_PATH = os.path.join(ROOT, "btc_core", "_simple.py")
CLASS_NAME = "SaturatingPowerLawModel"
ATTRS = ("_log10_L", "_t0", "_beta")

# Mirrors the box on the nuisance coordinate A inside analyze_spl.fit_spl.
# A is not one of the four bounds the spec asks about (it is a wide box on a
# reparameterisation artifact), but an active bound there would still be a
# finding, so it is reported. _check_mirrors_analyze_spl() below turns a
# future edit to analyze_spl into a loud failure instead of a stale report.
A_LO, A_HI = -10.0, 10.0

# Fit-coordinate boxes, exactly as differential_evolution receives them.
BOX = {
    "A":        (A_LO, A_HI),
    "beta":     (0.01, BETA_MAX),
    "log10_t0": (np.log10(T0_LO), np.log10(T0_HI)),
}
BOUND_TOL = 1e-6      # fraction of the box width that counts as "on the bound"


def _check_mirrors_analyze_spl() -> None:
    """Fail loudly if analyze_spl.fit_spl's boxes drift away from BOX.

    The A box is the only literal this file has to duplicate (fit_spl builds
    its bounds list inline). Checking the source text is cheap and converts
    silent report drift into a hard stop.

    Deliberately NOT an `assert`: `python -O` strips assert statements, and a
    stripped check here would let the bound report go quietly stale -- exactly
    the failure this guard exists to prevent. So it raises explicitly.
    """
    src = inspect.getsource(fit_spl)
    for frag in ("(-10.0, 10.0)", "(0.01, BETA_MAX)",
                 "(np.log10(T0_LO), np.log10(T0_HI))"):
        if frag not in src:
            raise SystemExit(
                f"analyze_spl.fit_spl no longer contains the bound {frag!r}; "
                f"tools/fit_spl.py's BOX table is stale and its bound report "
                f"would be wrong. Reconcile them before trusting this run."
            )


def load_fit_data():
    """The exact (t, log10 price) pair SaturatingPowerLawModel.__init__ sees.

    The class masks `price_years >= T_MIN`; so does this. Fitting on a wider
    mask would optimise a different residual set than the one
    `_init_shrinking_bands` derives the sigma bands from.
    """
    try:
        from time_basis import T_MIN as CLASS_T_MIN
    except Exception as exc:                                # pragma: no cover
        raise SystemExit(f"cannot import btc_web/time_basis.py: {exc}")
    if float(CLASS_T_MIN) != float(T_MIN):
        raise SystemExit(
            f"T_MIN mismatch: the model class masks at {CLASS_T_MIN} but this "
            f"fit masks at {T_MIN}. analyze_spl.py's bounds (t0 in [1,100] "
            f"YEARS) assume the calendar basis; a block-basis fit needs its "
            f"own bounds, not this tool."
        )
    prices = load_prices("BitcoinPricesDaily.csv")
    t = prices.df_full["years"].values
    lp = prices.df_full["log_price"].values
    mask = t >= T_MIN
    return t[mask], lp[mask]


def sse(t, lp, theta) -> float:
    """Unpenalised sum of squared log10 residuals."""
    return float(np.sum((lp - spl_log10(t, *theta)) ** 2))


def derived(theta):
    """(log10_L, t0) implied by the fit coordinates (A, beta, log10_t0)."""
    A, beta, lt0 = theta
    return A + beta * lt0, 10.0 ** lt0


def polish(t, lp, theta, lo_L, hi_L):
    """curve_fit refinement of the DE optimum; rejected unless it helps.

    The DE run already polishes with L-BFGS-B. This is the spec's second
    polish. curve_fit cannot see the L penalty (it takes box bounds only),
    so the result is accepted only if it both lowers the SSE and leaves the
    derived L inside its admissible range.
    """
    lo = [BOX["A"][0], BOX["beta"][0], BOX["log10_t0"][0]]
    hi = [BOX["A"][1], BOX["beta"][1], BOX["log10_t0"][1]]
    try:
        popt, _ = curve_fit(spl_log10, t, lp, p0=theta, bounds=(lo, hi),
                            maxfev=40000)
    except Exception as exc:
        return theta, f"curve_fit failed ({type(exc).__name__}), DE kept"
    l10L, _ = derived(popt)
    if not (lo_L <= l10L <= hi_L):
        return theta, "curve_fit left L outside its range, DE kept"
    if sse(t, lp, popt) >= sse(t, lp, theta):
        return theta, "curve_fit did not improve the SSE, DE kept"
    return np.asarray(popt, float), "curve_fit polish accepted"


def bound_report(theta, lo_L, hi_L):
    """Rows of (label, value, lo, hi, active?) for every bound in the fit.

    L is measured in log10 space because that is the space its penalty
    operates in (`lo_L <= A + beta*lt0 <= hi_L` in analyze_spl.fit_spl). On
    the linear-dollar scale the cap dwarfs the floor and every margin reads
    as "nearly at the floor" regardless of the fit.
    """
    A, beta, lt0 = theta
    l10L, t0 = derived(theta)
    rows = [
        ("A (nuisance)", A,    *BOX["A"]),
        ("beta",         beta, *BOX["beta"]),
        ("t0 (yr)",      t0,   T0_LO, T0_HI),
        ("log10 L",      l10L, lo_L, hi_L),
    ]
    out = []
    for label, val, lo, hi in rows:
        span = hi - lo
        d_lo, d_hi = val - lo, hi - val
        at_lo = d_lo <= BOUND_TOL * span
        at_hi = d_hi <= BOUND_TOL * span
        out.append((label, val, lo, hi, d_lo, d_hi, at_lo, at_hi))
    return out


def print_bounds(rows) -> bool:
    print("\nBound activity  (a parameter on its bound is an artifact of the "
          "bound, not a measurement)")
    print(f"    {'parameter':<14} {'fitted':>16} {'lower':>16} {'upper':>16}"
          f"   {'margin (% of box)':>20}  status")
    any_active = False
    for label, val, lo, hi, d_lo, d_hi, at_lo, at_hi in rows:
        span = hi - lo
        m_lo, m_hi = 100 * d_lo / span, 100 * d_hi / span
        if at_lo or at_hi:
            any_active = True
            status = f"*** AT {'LOWER' if at_lo else 'UPPER'} BOUND ***"
        else:
            status = "interior"
        print(f"    {label:<14} {val:>16,.6f} {lo:>16,.6f} {hi:>16,.6f}"
              f"   {m_lo:>9.4f} / {m_hi:<8.4f}  {status}")
        if label == "log10 L":
            print(f"    {'  in $/coin':<14} {10**val:>16,.0f} "
                  f"{10**lo:>16,.0f} {10**hi:>16,.0f}"
                  f"   floor = max observed price, cap = $1000T / 21e6")
    print(f"    bounds active: {'YES' if any_active else 'NONE'}"
          f"   (tolerance: margin <= {BOUND_TOL:g} x the box width)")
    return any_active


def _atomic_write(path: str, text: str) -> None:
    """Replace `path`'s contents with `text`, all-or-nothing.

    `open(path, "w")` truncates the target the instant it succeeds, so a
    Ctrl-C, a full disk, or a crash between truncate and flush leaves
    btc_core/_simple.py TRUNCATED -- every model in it gone, not just SPL's
    three constants. Writing a sibling temp file and renaming it over the
    target makes the swap a single atomic operation: the file is either the
    old text or the new one, never a prefix of either.

    The temp file is created in the target's own directory because os.replace
    is only atomic within a filesystem. fsync before the rename so a power
    loss cannot leave the renamed inode pointing at unflushed data.
    """
    d = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = tempfile.mkstemp(dir=d, prefix=os.path.basename(path) + ".",
                               suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
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


def patch_core(values: dict) -> str:
    """Rewrite the three class attrs, scoped to SaturatingPowerLawModel.

    THE HAZARD THIS GUARDS. Neither `_t0` nor `_beta` is a unique attribute
    name in btc_core/_simple.py: LogisticSCurveModel and GompertzModel both
    define `_t0`, and StretchedExponentialModel defines `_beta`. An unscoped
    or loosely scoped patch would rewrite another model's fitted parameters
    with SPL's, and nothing would raise -- that model would just quietly
    start fitting worse. So:

      * the search window is the byte range from `class SaturatingPowerLaw-
        Model` to the next top-level `class `, and every regex is anchored to
        a line start at exactly four spaces of indentation;
      * each attr must match exactly once inside that window;
      * the window must contain no other `class ` line;
      * and the finished text is diffed against the original: unless exactly
        three lines changed and all three lie inside the window, nothing is
        written.

    Returns the new source text.
    """
    with open(CORE_PATH) as f:
        src = f.read()

    start = src.find(f"class {CLASS_NAME}")
    if start == -1:
        raise SystemExit(f"could not find `class {CLASS_NAME}` in {CORE_PATH}")
    nxt = src.find("\nclass ", start + 1)
    end = nxt if nxt != -1 else len(src)
    section = src[start:end]

    stray = [ln for ln in section.splitlines()[1:] if ln.startswith("class ")]
    if stray:
        raise SystemExit(f"section scoping is wrong: it also spans {stray}")

    for name in ATTRS:
        pat = re.compile(rf"^(    {re.escape(name)}\s*=\s*)[^#\n]+", re.M)
        hits = pat.findall(section)
        if len(hits) != 1:
            raise SystemExit(
                f"expected exactly 1 `{name}` assignment in {CLASS_NAME}, "
                f"found {len(hits)}")
        section = pat.sub(
            lambda m: f"{m.group(1)}{values[name]:.6f}", section, count=1)

    new_src = src[:start] + section + src[end:]

    old_lines, new_lines = src.splitlines(), new_src.splitlines()
    if len(old_lines) != len(new_lines):
        raise SystemExit("patch changed the line count; refusing to write")
    changed = [i for i, (a, b) in enumerate(zip(old_lines, new_lines)) if a != b]
    # Both are 0-based line INDICES. `start` sits on the `class ...` line;
    # `hi_line` lands on the blank separator that trails the section, one
    # line before the NEXT `class`. Measured 2026-08-07:
    #   356  class SaturatingPowerLawModel     <- lo_line
    #   437  ''                                 } trailing blanks
    #   438  ''                                 } hi_line
    #   439  class BrokenPowerLawModel          <- excluded, and must stay so
    #
    # `<=` and `<` are BOTH safe here: the attr lines sit far above the
    # trailing blanks, so no write can land on 437/438 either way, and the
    # next class is excluded under both. `<=` is kept only because it makes
    # the interval read as closed on both ends, matching the variable names.
    #
    # What actually matters is the upper edge: `_t0` is NOT a unique
    # attribute name (GompertzModel and LogisticSCurveModel both have one),
    # so a window that reached line 439 would let --update silently rewrite
    # a different model's parameters. Do not loosen this bound.
    lo_line = src[:start].count("\n")
    hi_line = src[:end].count("\n")
    # At most the three attr lines, and every one of them inside the class.
    # Fewer than three is legitimate: re-running on unchanged price data
    # rewrites the same digits, so the tool must be idempotent rather than
    # treating a no-op as a scoping failure.
    if len(changed) > len(ATTRS) or not all(lo_line <= i <= hi_line
                                            for i in changed):
        raise SystemExit(
            f"refusing to write: expected at most {len(ATTRS)} changed lines "
            f"inside 0-based lines {lo_line}-{hi_line} (inclusive), "
            f"got {changed}")

    print("\n--update: patching btc_core/_simple.py")
    print(f"    scope: lines {lo_line + 1}-{hi_line + 1} (class {CLASS_NAME})")
    if not changed:
        print("    (no change -- the file already carries this fit)")
    for i in changed:
        print(f"    -{old_lines[i]}")
        print(f"    +{new_lines[i]}")
    diff = list(difflib.unified_diff(old_lines, new_lines, n=0, lineterm=""))
    print(f"    unified diff: {sum(1 for d in diff if d.startswith('+') and not d.startswith('+++'))} "
          f"additions / "
          f"{sum(1 for d in diff if d.startswith('-') and not d.startswith('---'))} deletions")
    return new_src


def main() -> None:
    update = "--update" in sys.argv
    _check_mirrors_analyze_spl()

    t, lp = load_fit_data()
    n = len(t)
    print(f"n={n}   t {t.min():.4f}..{t.max():.4f} yr   "
          f"price ${10**lp.min():,.4g}..${10**lp.max():,.0f}   "
          f"(mask t >= {T_MIN}, the same mask {CLASS_NAME} uses)")

    lo_L = float(np.log10(np.max(10 ** lp)))       # floor: max observed price
    hi_L = float(np.log10(CAP_FIT_USD / SUPPLY))   # cap:   $1000T market cap
    print(f"L admissible: ${10**lo_L:,.0f}..${10**hi_L:,.0f} per coin "
          f"(= ${10**lo_L*SUPPLY/1e12:,.3f}T..${10**hi_L*SUPPLY/1e12:,.0f}T cap)")

    # differential_evolution over several seeds -- a global optimum that moves
    # with the seed is not a global optimum, and the report should say so.
    print("\nRunning differential_evolution (tools/analyze_spl.py::fit_spl) ...")
    runs = []
    for seed in (0, 1, 2):
        r = fit_spl(t, lp, seed=seed)
        l10L, t0 = derived(r.x)
        runs.append((seed, r))
        print(f"    seed {seed}: SSE {r.fun:.10f}   beta {r.x[1]:.6f}   "
              f"t0 {t0:.4f} yr   log10 L {l10L:.6f}")
    seed, res = min(runs, key=lambda sr: sr[1].fun)
    spread = max(r.fun for _, r in runs) - min(r.fun for _, r in runs)
    print(f"    best seed {seed};  SSE spread across seeds {spread:.3e}")

    theta, note = polish(t, lp, np.asarray(res.x, float), lo_L, hi_L)
    print(f"    polish: {note}")

    l10L, t0 = derived(theta)
    beta = float(theta[1])
    s = sse(t, lp, theta)
    rmse = float(np.sqrt(s / n))
    r2 = 1.0 - s / float(np.sum((lp - lp.mean()) ** 2))
    sigma = float(np.std(lp - spl_log10(t, *theta)))

    print("\nFitted Saturating Power Law parameters")
    print(f"  _log10_L = {l10L:.6f}   (L = ${10**l10L:,.0f}/coin  "
          f"= ${10**l10L*SUPPLY/1e12:,.2f}T market cap)")
    print(f"  _t0      = {t0:.6f}   (years since 2009-07-25; price = L/2 here)")
    print(f"  _beta    = {beta:.6f}   (early-time power-law exponent)")
    print(f"  RMSE     = {rmse:.6f}   (log10 units)   SSE {s:.6f}")
    print(f"  R^2      = {r2:.6f}     sigma {sigma:.6f}")

    any_active = print_bounds(bound_report(theta, lo_L, hi_L))

    if not update:
        print("\nRun with --update to write these into btc_core/_simple.py")
        return

    if any_active:
        raise SystemExit(
            "\nREFUSING --update: a bound is active. The spec's section 4 "
            "claims the $1000T cap never binds; a pinned parameter is a "
            "finding about the spec, not something to fix by widening the "
            "box. Report the numbers above before writing anything."
        )

    new_src = patch_core({"_log10_L": l10L, "_t0": t0, "_beta": beta})
    _atomic_write(CORE_PATH, new_src)
    print("btc_core/_simple.py updated. Verify with:\n"
          "    git diff btc_core/_simple.py\n"
          "    btc_venv/bin/python3 -m pytest btc_web/test_models.py "
          "-k Saturating -v")


if __name__ == "__main__":
    main()
