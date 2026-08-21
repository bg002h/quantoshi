#!/usr/bin/env python3
"""Model-registration linter for Quantoshi price models.

Registering a new price model touches ~26 checklist steps plus several
registries that are not on the checklist at all.  Most of the misses fail
*silently*: the model renders correctly on every chart whether or not the
step was done, and every existing test still passes.  Each check below turns
one of those steps into an assertion.

Two front ends:

* ``btc_web/test_model_registration.py`` parameterises every check over every
  entry in ``_app_ctx.PRICE_MODELS``, so CI enforces this instead of
  discipline.
* ``btc_venv/bin/python3 tools/check_model_registration.py [short_name]``
  prints a per-registry table -- what someone adding a model actually wants.

Design rules
------------
**Fail closed.**  A model not named on an exemption list must satisfy every
check.  A newly registered model therefore starts out failing everything,
which is the entire point.

**Exemptions are data, not control flow.**  Every skip is an ``Exemption``
entry carrying a written reason, and the reason is printed next to the row.
Where the exemption can be *derived from the code it excuses* (the scanner's
own filter expressions, the two ``for`` loops in ``build_bm_model.py``, the
family colour fallback in ``figures/common.py``) it is derived rather than
hand-copied, so it cannot drift into excusing a real hole.

**Known holes stay visible.**  ``KNOWN_HOLES`` maps ``(model, check)`` to the
follow-up that owns the fix.  They report as ``KNOWN``, never as a pass.  An
entry that stops reproducing is itself a failure (``STALE``), so fixing the
follow-up forces the entry's deletion -- that deletion is the regression test.
"""
from __future__ import annotations

import argparse
import ast
import functools
import inspect
import os
import pathlib
import re
import sys
from dataclasses import dataclass

REPO = pathlib.Path(__file__).resolve().parents[1]


# ══════════════════════════════════════════════════════════════════════════
# Registry access
# ══════════════════════════════════════════════════════════════════════════

def _prepare_sys_path() -> None:
    for p in (str(REPO), str(REPO / "btc_web")):
        if p not in sys.path:
            sys.path.insert(0, p)


@functools.lru_cache(maxsize=1)
def load_price_models():
    """Return the populated ``_app_ctx.PRICE_MODELS`` registry.

    ``_app_ctx`` starts empty; ``app.py`` populates it as an import side
    effect.  Under pytest, conftest has already done that, so this is a
    no-op there.
    """
    _prepare_sys_path()
    import _app_ctx

    if not _app_ctx.PRICE_MODELS:
        os.environ.setdefault("TESTING", "1")
        # app.py fetches a live price at import; block the socket so the CLI
        # does not stall on a network timeout (conftest does the same).
        import urllib.request

        _real = urllib.request.urlopen
        urllib.request.urlopen = lambda *a, **k: (_ for _ in ()).throw(
            OSError("blocked by check_model_registration")
        )
        try:
            import app  # noqa: F401  -- registration side effect
        finally:
            urllib.request.urlopen = _real
    return _app_ctx.PRICE_MODELS


@functools.lru_cache(maxsize=None)
def _src(rel: str) -> str:
    return (REPO / rel).read_text()


def _strip_noise(text: str) -> str:
    """Drop comments and import lines.

    Both are why a bare ``"T_MIN" in window`` substring test stays green
    through exactly the deletion it claims to prevent: the identifier
    survives in a ``from time_basis import T_MIN`` line, or in prose that
    merely mentions it.
    """
    out = []
    for line in text.splitlines():
        line = re.sub(r"#.*$", "", line)
        if re.match(r"\s*(from|import)\s", line):
            continue
        out.append(line)
    return "\n".join(out)


def _module_assign(rel: str, name: str):
    """``ast.literal_eval`` the value assigned to ``name`` anywhere in a file.

    Used instead of importing, so a constant defined *inside a function*
    (``scanner.py``'s ``_HYBPPL_FAMILY_EXTRAS``) is still readable.
    """
    for node in ast.walk(ast.parse(_src(rel))):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            try:
                return ast.literal_eval(node.value)
            except ValueError:
                return None
    return None


# ══════════════════════════════════════════════════════════════════════════
# Result model
# ══════════════════════════════════════════════════════════════════════════

OK, FAIL, EXEMPT, KNOWN, STALE = "OK", "FAIL", "EXEMPT", "KNOWN", "STALE"

#: Statuses that must not fail the pytest suite.
GREEN = (OK, EXEMPT, KNOWN)

_GLYPH = {OK: "✓", FAIL: "✗", EXEMPT: "–",
          KNOWN: "!", STALE: "⚠"}


@dataclass(frozen=True)
class Result:
    check: str
    model: str
    status: str
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status in GREEN


@dataclass(frozen=True)
class Exemption:
    """One documented reason a check does not apply to some models.

    ``models`` are literal short_names; ``prefixes`` cover whole generated
    families.  ``derive`` is an optional zero-arg callable returning extra
    short_names read out of the very source the exemption excuses -- so the
    exemption shrinks automatically when that source stops filtering them.
    ``verify`` is an optional ``(short) -> (bool, detail)`` that PROVES the
    claim rather than trusting it; if it returns False the row is a FAIL,
    because an exemption resting on a premise that stopped holding is exactly
    the hole this linter exists to find.
    """
    reason: str
    models: frozenset = frozenset()
    prefixes: tuple = ()
    derive: object = None
    verify: object = None

    def members(self) -> frozenset:
        extra = frozenset(self.derive() or ()) if self.derive else frozenset()
        return self.models | extra

    def covers(self, short: str) -> bool:
        return short in self.members() or short.startswith(self.prefixes)


# ══════════════════════════════════════════════════════════════════════════
# Exemptions -- every one of these is a decision, not an accident
# ══════════════════════════════════════════════════════════════════════════

# Families generated in bulk.  36 HybPPL configs + 36 Entropy-PPL configs are
# selected by resolving the `hybppl` / `eppl` MASTER checkbox to a concrete
# variant key at chart-build time; the variant key is never offered in the UI
# and never round-trips through a share link.
_CFG_FAMILIES = ("cfg_", "ecfg_")

# LPPL variants reachable only through the LPPL config modal (n_freqs /
# weighted / no-1/3 radio buttons), not as standalone Display Models entries.
_LPPL_VARIANTS = frozenset({
    "lppl_w", "lp2_w", "lp3_w", "lp4_w", "lp4_n13", "lp4_w_n13",
})
_LPPL_FAMILY = _LPPL_VARIANTS | {"lppl", "lp2", "lp3", "lp4"}


def _scanner_popped() -> frozenset:
    """Short_names ``scanner.py`` pops before ordering, read from its source.

    Derived rather than copied: if the pop is ever removed, this exemption
    silently shrinks and the model becomes *required* in ``_SCANNER_ORDER``
    -- the safe direction.
    """
    return frozenset(re.findall(r"_models\.pop\(\"([a-z0-9_]+)\"",
                                _src("btc_web/callbacks/scanner.py")))


def _scanner_hybppl_extras() -> frozenset:
    """``_HYBPPL_FAMILY_EXTRAS``, a function-local in ``scanner.py``."""
    return frozenset(_module_assign(
        "btc_web/callbacks/scanner.py", "_HYBPPL_FAMILY_EXTRAS") or ())


EXEMPTIONS: dict[str, tuple[Exemption, ...]] = {

    # -- 1. PRICE_MODELS ---------------------------------------------------
    # No exemptions: the check is parameterised over PRICE_MODELS itself.
    "price_models": (),

    # -- 2. colours --------------------------------------------------------
    "colors": (
        Exemption(
            reason="master-gate family variant; figures/common.py::"
                   "_get_model_color inherits the master colour by key "
                   "prefix (verified per palette, not assumed)",
            models=_LPPL_VARIANTS, prefixes=_CFG_FAMILIES,
            verify=lambda short: _c_colors_exempt_verify(short),
        ),
    ),

    # -- 3. snapshot checklist bitmasks ------------------------------------
    "checklist": (
        Exemption(
            reason="never an individually selectable Display Models entry -- "
                   "the hybppl/eppl master checkbox resolves to the variant "
                   "key after the snapshot is decoded, so it owns no bit",
            prefixes=_CFG_FAMILIES,
        ),
    ),

    # -- 4. build_bm_model.py instances ------------------------------------
    "build_bm": (
        Exemption(
            reason="instantiated by the `for cfg_key in _HYBPPL_CONFIG_PARAMS`"
                   " / `_EPPL_CONFIG_PARAMS` loops rather than a literal "
                   "assignment (membership verified against those dicts)",
            prefixes=_CFG_FAMILIES,
            verify=lambda short: _c_build_bm_exempt_verify(short),
        ),
        Exemption(
            reason="no model instance to build: QR fits are read straight off "
                   "model_data.pkl (md_obj.qr_fits)",
            models=frozenset({"qr"}),
        ),
        Exemption(
            reason="not residual-fitted: S2F is a single issuance-derived "
                   "trajectory with no residual bands",
            models=frozenset({"s2f", "s2f_inst"}),
        ),
        Exemption(
            reason="built by tools/build_ef_model.py into model_data_ef.pkl, "
                   "a separate artifact",
            models=frozenset({"ef"}),
        ),
        Exemption(
            reason="LPPL family is outside _build_model_instances' scope "
                   "(RESQR_FLAGSHIP_MODELS names no LPPL entry); its bands "
                   "come from _init_shrinking_bands at construction",
            models=_LPPL_FAMILY,
        ),
    ),

    # -- 5. refit_all_ppl.py -----------------------------------------------
    "refit": (
        Exemption(
            reason="no free parameters to refit: coefficients are read from "
                   "model_data.pkl, which update_prices.py rebuilds daily",
            models=frozenset({"bub", "qr"}),
        ),
        Exemption(
            reason="closed-form OLS at construction -- there is no fit script "
                   "and nothing a monthly refit could move",
            models=frozenset({"pl", "exp"}),
        ),
        Exemption(
            reason="derived model: PCA basis is recomputed from the already-"
                   "fitted HybPPL family at construction",
            models=frozenset({"pca"}),
        ),
        Exemption(
            reason="not fitted: S2F is parameterised from the issuance "
                   "schedule",
            models=frozenset({"s2f", "s2f_inst"}),
        ),
        Exemption(
            reason="refit by tools/build_ef_model.py, not the monthly PPL job",
            models=frozenset({"ef"}),
        ),
    ),

    # -- 6/7. fit_shrinking_sigma.py ---------------------------------------
    # Scope is masters and standalone models.  Composites and family variants
    # derive sigma at construction via _init_shrinking_bands from their own
    # residuals, so there is nothing for the offline sigma fit to write.
    "sigma": (
        Exemption(
            reason="sigma derived at construction by _init_shrinking_bands; "
                   "no class attrs for the offline fit to patch",
            models=frozenset({"bub", "qr", "s2f", "s2f_inst", "ef", "pca",
                              "linppl", "hyb2l", "hyb2c", "hyb2b", "hyb4d"})
                   | (_LPPL_FAMILY - {"lppl"}),
            prefixes=_CFG_FAMILIES,
        ),
    ),

    # -- 8. scanner ordering -----------------------------------------------
    "scanner": (
        Exemption(
            reason="popped by scanner.py before ordering (derived from its "
                   "own `_models.pop(...)` calls)",
            derive=_scanner_popped,
        ),
        Exemption(
            reason="excluded by scanner.py's _HYBPPL_FAMILY_EXTRAS filter "
                   "(derived from that literal)",
            derive=_scanner_hybppl_extras,
            prefixes=_CFG_FAMILIES,
        ),
        Exemption(
            reason="LPPL family variants collapse onto the `lppl` slot via "
                   "_scanner_sort_key, so only the master needs an entry",
            models=_LPPL_FAMILY - {"lppl"},
        ),
    ),

    # -- 9. docs/architecture.md -------------------------------------------
    "arch_doc": (
        Exemption(
            reason="documented as a pattern row (`cfg_<...>` / `ecfg_...`) "
                   "rather than 36 individual rows",
            prefixes=_CFG_FAMILIES,
            verify=lambda short: _c_arch_doc_exempt_verify(short),
        ),
    ),

    # -- 10. Model Info card ------------------------------------------------
    "model_info": (
        Exemption(
            reason="covered by the shared `mi-lp2` (multi-frequency) and "
                   "`mi-lppl-weighting` cards, not a card of its own",
            models=_LPPL_VARIANTS | {"lp3", "lp4"},
        ),
        Exemption(
            reason="covered by the HybPPL / EPPL family cards; 72 config "
                   "variants do not get 72 accordion items",
            prefixes=_CFG_FAMILIES,
        ),
    ),
}


# ══════════════════════════════════════════════════════════════════════════
# Known holes -- real defects, kept visible instead of silently exempted
# ══════════════════════════════════════════════════════════════════════════

KNOWN_HOLES: dict[tuple[str, str], str] = {
    # F-1 -- the SCRIPTS entry stores "--mode=de" inside the path string, so
    # refit_all_ppl.py's os.path.exists() never resolves it and grdy is
    # silently skipped by every monthly refit.
    ("grdy", "refit"): "F-1",

    # F-2 -- filed against `logi`; the follow-up notes 9 further branches were
    # unaudited.  This linter audits them: all 9 have the same defect.
    ("logi", "sigma_mask"): "F-2",
    ("pl", "sigma_mask"): "F-2",
    ("exp", "sigma_mask"): "F-2",
    ("lppl", "sigma_mask"): "F-2",
    ("hybppl", "sigma_mask"): "F-2",
    ("hybppl_dd", "sigma_mask"): "F-2",
    ("gomp", "sigma_mask"): "F-2",
    ("bpl", "sigma_mask"): "F-2",
    ("plo", "sigma_mask"): "F-2",
    ("sexp", "sigma_mask"): "F-2",

    # F-6 -- reach the scanner (no filter drops them) but are absent from
    # _SCANNER_ORDER, so _scanner_sort_key's `except ValueError` sinks them
    # to the bottom instead of their Display Models slot.
    ("plo", "scanner"): "F-6",
    ("sexp", "scanner"): "F-6",
    ("logi", "scanner"): "F-6",
}


# ══════════════════════════════════════════════════════════════════════════
# Checks
# ══════════════════════════════════════════════════════════════════════════

def _c_price_models(short, model):
    """1. Registered, and its short_name agrees with its registry key."""
    pm = load_price_models()
    if short not in pm:
        return False, "absent from _app_ctx.PRICE_MODELS"
    actual = getattr(pm[short], "short_name", None)
    if actual != short:
        return False, f"short_name is {actual!r}, registry key is {short!r}"
    return True, f"short_name == {short!r}"


def _c_colors(short, model):
    """2. In MODEL_TRACE_COLORS and all four palettes' model_colors.

    A missing entry renders that one palette grey and nothing else changes.
    """
    load_price_models()
    from colors import PALETTES, MODEL_TRACE_COLORS

    missing = [] if short in MODEL_TRACE_COLORS else ["MODEL_TRACE_COLORS"]
    missing += [n for n, p in PALETTES.items() if short not in p["model_colors"]]
    if missing:
        return False, "missing from " + ", ".join(missing)
    return True, f"{1 + len(PALETTES)} colour registries"


def _c_colors_exempt_verify(short):
    """Verify a colour exemption instead of trusting it.

    The exemption claims the family prefix fallback supplies a colour.  Prove
    it: resolve through the real ``_get_model_color`` in every palette and
    reject the fallback grey.
    """
    load_price_models()
    from colors import PALETTES, FALLBACK_MODEL_GRAY
    from figures.common import _get_model_color

    grey = [n for n in PALETTES
            if _get_model_color(short, {"palette": n}) == FALLBACK_MODEL_GRAY]
    if grey:
        return False, f"family fallback still renders grey in: {grey}"
    sample = _get_model_color(short, {"palette": "default"})
    return True, f"family fallback -> {sample} in all {len(PALETTES)} palettes"


def _c_checklist(short, model):
    """3. In all five ``*-model-show`` lists, appended (never inserted).

    ``_CHECKLIST_OPTIONS`` positions ARE the share-link bitmask bit indices.
    A missing entry makes the encoder drop the model silently; an insert
    re-points every already-published link's bits.
    """
    load_price_models()
    from snapshot import _CHECKLIST_OPTIONS

    lists = {k: v for k, v in _CHECKLIST_OPTIONS.items()
             if k.endswith("-model-show")}
    absent = sorted(k for k, v in lists.items() if short not in v)
    if absent:
        return False, "missing from " + ", ".join(absent)
    # Append consistency: a model that is last in one list must be last in
    # all of them.  A half-done append is an insert in the other four.
    last_in = {k for k, v in lists.items() if v[-1] == short}
    if last_in and len(last_in) != len(lists):
        return False, (f"appended last only in {sorted(last_in)} -- it is "
                       f"mid-list in the rest, which shifts their bitmasks")
    pos = {k: lists[k].index(short) for k in lists}
    return True, f"in all {len(lists)} lists, bit {sorted(set(pos.values()))}"


def _c_build_bm(short, model):
    """4. ``instances["{short}"]`` in tools/build_bm_model.py.

    update_prices.py runs build_bm_model.py after every daily price append,
    so a model missing here is absent from every future model_data.pkl.
    """
    if f'instances["{short}"]' in _src("tools/build_bm_model.py"):
        return True, "literal instances[...] assignment"
    return False, "no instances[...] entry in _build_model_instances"


def _c_build_bm_exempt_verify(short):
    """Verify the cfg_/ecfg_ loop actually produces this key."""
    _prepare_sys_path()
    import btc_core as bc

    if short in getattr(bc, "_HYBPPL_CONFIG_PARAMS", {}):
        return True, "built by the _HYBPPL_CONFIG_PARAMS loop"
    if short in getattr(bc, "_EPPL_CONFIG_PARAMS", {}):
        return True, "built by the _EPPL_CONFIG_PARAMS loop"
    return False, "claimed loop-built but in neither config-params dict"


#: short_name -> the refit script that owns its parameters.  Kept explicit
#: because refit_all_ppl.py keys its entries by DISPLAY name ("SatPL",
#: "Logistic"), which no rule maps back to a short_name.
REFIT_SCRIPTS: dict[str, str] = {
    "lppl": "tools/fit_lppl.py",
    "lp2": "tools/fit_lppl2.py",
    "lp3": "tools/fit_lppl3.py",
    "lp4": "tools/fit_lppl4.py",
    "lppl_w": "tools/fit_lppl_variants.py",
    "lp2_w": "tools/fit_lppl_variants.py",
    "lp3_w": "tools/fit_lppl_variants.py",
    "lp4_w": "tools/fit_lppl_variants.py",
    "lp4_n13": "tools/fit_lppl_variants.py",
    "lp4_w_n13": "tools/fit_lppl_variants.py",
    "linppl": "tools/fit_linppl.py",
    "hybppl": "tools/fit_hybppl.py",
    "hybppl_dd": "tools/fit_hybppl_dd.py",
    "hyb2l": "tools/fit_hyb2l.py",
    "hyb2c": "tools/fit_hyb2c.py",
    "hyb2b": "tools/fit_hyb2b.py",
    "hyb4d": "tools/fit_hyb4d.py",
    "eppl": "tools/fit_all_eppl_configs.py",
    "grdy": "tools/fit_grdy.py",
    "gomp": "tools/fit_gompertz.py",
    "bpl": "tools/fit_bpl.py",
    "plo": "tools/fit_plo.py",
    "sexp": "tools/fit_sexp.py",
    "logi": "tools/fit_logistic.py",
    "spl": "tools/fit_spl.py",
}


def refit_scripts_entries():
    """AST-parse ``SCRIPTS`` out of refit_all_ppl.py."""
    return _module_assign("tools/refit_all_ppl.py", "SCRIPTS") or []


def _c_refit(short, model):
    """5. A refit_all_ppl.py entry exists AND its path resolves on disk.

    Substring-matching the source is not enough, and that is not a
    hypothetical: ``("Greedy Select", "tools/fit_grdy.py --mode=de")`` stores
    a flag inside the path, so refit_all_ppl.py's ``os.path.exists`` fails,
    the entry is skipped with no error, and the model is never refit.  Parse
    the entry and resolve the stored string exactly as the runner does.
    """
    if short.startswith(_CFG_FAMILIES):
        expected = ("tools/fit_all_hybppl_configs.py"
                    if short.startswith("cfg_")
                    else "tools/fit_all_eppl_configs.py")
    else:
        expected = REFIT_SCRIPTS.get(short)
    if expected is None:
        return False, (f"no REFIT_SCRIPTS mapping for {short!r} -- add one, "
                       f"or add a documented exemption")

    hits = [stored for _, stored in refit_scripts_entries()
            if stored.split()[0] == expected]
    if not hits:
        return False, f"no SCRIPTS entry for {expected}"
    if len(hits) > 1:
        return False, f"{len(hits)} SCRIPTS entries for {expected}"

    stored = hits[0]
    if not (REPO / stored).exists():
        return False, (f"SCRIPTS path {stored!r} does not resolve on disk; "
                       f"refit_all_ppl.py prints 'NOT FOUND -- skipping' and "
                       f"never refits {short}")
    return True, stored


def sigma_branches() -> dict[str, str]:
    """Map ``model_name == "X"`` branch -> its body source."""
    src = _src("tools/fit_shrinking_sigma.py")
    hits = list(re.finditer(r'model_name == "([a-z0-9_]+)"', src))
    end_of_chain = src.index('        raise ValueError(f"Unknown model')
    out = {}
    for i, m in enumerate(hits):
        stop = hits[i + 1].start() if i + 1 < len(hits) else end_of_chain
        out[m.group(1)] = src[m.start():stop]
    return out


def sigma_models_list() -> list[str]:
    """The ``models = [...]`` list inside ``main()``."""
    m = re.search(r"^\s*models\s*=\s*(\[[^\]]*\])",
                  _src("tools/fit_shrinking_sigma.py"), re.M)
    return ast.literal_eval(m.group(1)) if m else []


def _c_sigma(short, model):
    """6. BOTH a ``model_name == "{short}"`` branch AND a ``models`` entry.

    Either one alone is a silent no-op: a branch nothing dispatches to, or a
    list entry that raises ``Unknown model`` inside a worker process whose
    failure is printed and swallowed.
    """
    has_branch = short in sigma_branches()
    in_list = short in sigma_models_list()
    if has_branch and in_list:
        return True, "branch + models[] entry"
    if has_branch:
        return False, "has a branch but is absent from main()'s models[] -- "\
                      "the branch is never dispatched"
    if in_list:
        return False, "in main()'s models[] but has no branch -- raises "\
                      "'Unknown model' in the worker and is skipped"
    return False, "neither a branch nor a models[] entry"


def _c_sigma_mask(short, model):
    """7. If the constructor masks its fit window, the branch must too.

    ``_init_shrinking_bands`` derives the bands from residuals over
    ``price_years >= T_MIN``.  A branch that omits the mask optimises sigma
    over a *different* residual set -- including exactly the early, large
    residuals the constructor excludes.  Silent: the bands still render.
    """
    branch = sigma_branches().get(short)
    if branch is None:
        return True, "n/a -- no sigma branch (see the `sigma` check)"
    try:
        ctor = inspect.getsource(type(model).__init__)
    except (OSError, TypeError):
        return True, "n/a -- constructor source unavailable"
    if "T_MIN" not in ctor:
        return True, "constructor applies no T_MIN mask; none required"
    if "T_MIN" not in _strip_noise(branch):
        return False, ("constructor masks price_years >= T_MIN but the sigma "
                       "branch does not -- sigma is fitted on a different "
                       "residual set than the bands")
    return True, "both constructor and branch mask on T_MIN"


def _c_scanner(short, model):
    """8. Present in ``callbacks/scanner.py::_SCANNER_ORDER``.

    Benign-ish -- ``_scanner_sort_key`` sinks unknown keys via
    ``except ValueError`` -- but the row then lands at the bottom of the
    scanner instead of its Display Models slot.
    """
    load_price_models()
    from callbacks.scanner import _SCANNER_ORDER

    if short in _SCANNER_ORDER:
        return True, f"position {_SCANNER_ORDER.index(short)}"
    return False, ("absent from _SCANNER_ORDER -- _scanner_sort_key sinks it "
                   "to the bottom of the scanner table")


def _arch_first_cells() -> list[str]:
    return [ln.split("|")[1] for ln in _src("docs/architecture.md").splitlines()
            if ln.startswith("|") and ln.count("|") >= 2]


def _c_arch_doc(short, model):
    """9. A row in docs/architecture.md's model tables.

    That file is SERVED at /docs/architecture (btc_web/api.py), so staleness
    here is publicly visible.  Matches the row's first cell, which lets one
    row legitimately cover several keys (``hyb2l``, ``hyb2c``, ...).
    """
    hits = [c for c in _arch_first_cells() if f"`{short}`" in c]
    if not hits:
        return False, "no row in the model table"
    return True, f"row: {hits[0].strip()[:48]}"


def _c_arch_doc_exempt_verify(short):
    pattern = "`cfg_<" if short.startswith("cfg_") else "`ecfg_"
    if any(pattern in c for c in _arch_first_cells()):
        return True, f"covered by the {pattern}...` pattern row"
    return False, f"claimed pattern row {pattern}...` is not in the table"


def mi_item_id(short: str) -> str:
    """Accordion item_id for a model card (``hybppl_dd`` -> ``mi-hybppl-dd``)."""
    return "mi-" + short.replace("_", "-")


def _c_model_info(short, model):
    """10. ``mi-{short}`` in BOTH item lists, and the lists are identical.

    Deep links resolve positionally (``_MODEL_INFO_ITEMS[n - 1]``), so the
    two hand-maintained lists drifting re-points every ``/mi.N`` link at a
    different card with no error.
    """
    load_price_models()
    from layout.model_info import _MODEL_INFO_ITEM_IDS
    from callbacks.routing import _MODEL_INFO_ITEMS

    item = mi_item_id(short)
    in_layout = item in _MODEL_INFO_ITEM_IDS
    in_routing = item in _MODEL_INFO_ITEMS
    if not in_layout and not in_routing:
        return False, (f"{item} is in neither list -- write a Model Info card, "
                       f"or add a documented exemption")
    if in_layout != in_routing:
        present = "layout/model_info" if in_layout else "callbacks/routing"
        return False, (f"{item} is only in {present}; the lists must be "
                       f"byte-identical or /mi.N shifts")
    if list(_MODEL_INFO_ITEM_IDS) != list(_MODEL_INFO_ITEMS):
        return False, "the two item lists disagree (see the global check)"
    return True, f"{item} at index {_MODEL_INFO_ITEM_IDS.index(item)} in both"


@dataclass(frozen=True)
class Check:
    id: str
    title: str
    fn: object


CHECKS: tuple[Check, ...] = (
    Check("price_models", "PRICE_MODELS registration", _c_price_models),
    Check("colors", "colors.py + 4 palettes", _c_colors),
    Check("checklist", "snapshot bitmask lists", _c_checklist),
    Check("build_bm", "build_bm_model.py instances", _c_build_bm),
    Check("refit", "refit_all_ppl.py (path resolves)", _c_refit),
    Check("sigma", "fit_shrinking_sigma branch+list", _c_sigma),
    Check("sigma_mask", "sigma branch mask parity", _c_sigma_mask),
    Check("scanner", "_SCANNER_ORDER", _c_scanner),
    Check("arch_doc", "docs/architecture.md row", _c_arch_doc),
    Check("model_info", "Model Info card (both lists)", _c_model_info),
)

CHECK_IDS = tuple(c.id for c in CHECKS)
CHECK_BY_ID = {c.id: c for c in CHECKS}


# ══════════════════════════════════════════════════════════════════════════
# Runner
# ══════════════════════════════════════════════════════════════════════════

def run_check(check_id: str, short: str) -> Result:
    check = CHECK_BY_ID[check_id]
    model = load_price_models()[short]

    for ex in EXEMPTIONS.get(check_id, ()):
        if not ex.covers(short):
            continue
        if ex.verify is not None:
            ok, detail = ex.verify(short)
            if not ok:
                return Result(check_id, short, FAIL,
                              f"exemption does not hold: {detail}")
            return Result(check_id, short, EXEMPT, f"{ex.reason} [{detail}]")
        return Result(check_id, short, EXEMPT, ex.reason)

    passed, detail = check.fn(short, model)
    hole = KNOWN_HOLES.get((short, check_id))
    if hole is not None:
        if passed:
            return Result(check_id, short, STALE,
                          f"KNOWN_HOLES entry no longer reproduces ({hole}) "
                          f"-- delete it; that deletion is the regression test")
        return Result(check_id, short, KNOWN, f"{hole} | {detail}")
    return Result(check_id, short, OK if passed else FAIL, detail)


def check_model(short: str) -> list[Result]:
    return [run_check(cid, short) for cid in CHECK_IDS]


def check_all() -> list[Result]:
    return [r for short in load_price_models() for r in check_model(short)]


# ── Global (not per-model) invariants ──────────────────────────────────────

def check_global() -> list[Result]:
    load_price_models()
    from snapshot import _CHECKLIST_OPTIONS
    from layout.model_info import _MODEL_INFO_ITEM_IDS
    from callbacks.routing import _MODEL_INFO_ITEMS

    out = []

    lists = [k for k in _CHECKLIST_OPTIONS if k.endswith("-model-show")]
    out.append(Result(
        "global:checklist_count", "-", OK if len(lists) == 5 else FAIL,
        f"{len(lists)} *-model-show lists (expect 5): {sorted(lists)}"))

    same = list(_MODEL_INFO_ITEM_IDS) == list(_MODEL_INFO_ITEMS)
    out.append(Result(
        "global:model_info_mirror", "-", OK if same else FAIL,
        "layout/model_info::_MODEL_INFO_ITEM_IDS == "
        "callbacks/routing::_MODEL_INFO_ITEMS"
        + ("" if same else " -- MISMATCH re-points every /mi.N deep link")))

    # An exemption naming a model that no longer exists is a stale exemption:
    # it excuses nothing today and will silently excuse the wrong thing when
    # the name is reused.
    pm = load_price_models()
    stale = sorted({
        m for exs in EXEMPTIONS.values() for ex in exs
        for m in ex.members() if m not in pm
    })
    out.append(Result(
        "global:exemptions_live", "-", OK if not stale else FAIL,
        "every exemption names a live model"
        if not stale else f"exemptions name unknown models: {stale}"))

    return out


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════

def _print_model(short: str) -> int:
    pm = load_price_models()
    cls = type(pm[short]).__name__
    print(f"\nModel registration: {short}  ({cls})\n")
    w = max(len(c.title) for c in CHECKS)
    bad = 0
    for r in check_model(short):
        title = CHECK_BY_ID[r.check].title
        print(f"  {_GLYPH[r.status]} {title:<{w}}  {r.status:<7} {r.detail}")
        bad += 0 if r.ok else 1
    print()
    return bad


def _print_all(show_ok: bool) -> int:
    results = check_all()
    bad = [r for r in results if not r.ok]
    known = [r for r in results if r.status == KNOWN]
    exempt = [r for r in results if r.status == EXEMPT]
    n_models = len(load_price_models())

    print(f"\n{n_models} models x {len(CHECKS)} checks = {len(results)} "
          f"assertions\n")
    if show_ok:
        for r in results:
            print(f"  {_GLYPH[r.status]} {r.model:<12} {r.check:<14} {r.detail}")
    else:
        for r in known + bad:
            print(f"  {_GLYPH[r.status]} {r.model:<12} {r.check:<14} {r.detail}")

    print(f"\n  OK     {sum(1 for r in results if r.status == OK)}")
    print(f"  EXEMPT {len(exempt)}")
    print(f"  KNOWN  {len(known)}   (open follow-ups, not passes)")
    print(f"  FAIL   {len(bad)}\n")
    return len(bad)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Check that a price model is registered everywhere it "
                    "has to be. Silent-failure registries only.")
    ap.add_argument("short_name", nargs="?",
                    help="model to check; omit to check every model")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="with no short_name, print passing rows too")
    args = ap.parse_args(argv)

    bad = 0
    for r in check_global():
        if not r.ok:
            print(f"  {_GLYPH[r.status]} GLOBAL  {r.check:<28} {r.detail}")
            bad += 1

    if args.short_name:
        pm = load_price_models()
        if args.short_name not in pm:
            print(f"unknown model {args.short_name!r}; known: "
                  f"{', '.join(sorted(pm))}")
            return 2
        bad += _print_model(args.short_name)
    else:
        bad += _print_all(args.verbose)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
