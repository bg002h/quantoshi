"""Model-registration linter, run as a test over every registered model.

The logic lives in ``tools/check_model_registration.py`` so the same checks
back a CLI (``btc_venv/bin/python3 tools/check_model_registration.py spl``).
This file only parameterises it, because the point of the linter is that it
runs in CI instead of relying on someone remembering a 26-step checklist.

Every check maps to a registry whose omission fails SILENTLY -- the model
renders correctly on every chart either way, and every other test still
passes.  Do not "verify" any of these by looking at a chart.

Statuses:
  OK      registered
  EXEMPT  documented, reasoned non-applicability (printed, never hidden)
  KNOWN   a real open hole, pinned to its follow-up ID so the suite stays
          green while the hole stays visible
  FAIL    what this file exists to catch
  STALE   a KNOWN_HOLES entry that no longer reproduces -- delete it
"""
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# conftest imports app, which populates _app_ctx.PRICE_MODELS.
from conftest import _app_ctx  # noqa: E402,F401

from tools.check_model_registration import (  # noqa: E402
    CHECK_IDS, EXEMPTIONS, KNOWN_HOLES, REFIT_SCRIPTS,
    EXEMPT, KNOWN, STALE,
    check_global, load_price_models, refit_scripts_entries, run_check,
    sigma_branches, sigma_models_list,
)

MODELS = sorted(load_price_models())

# Guards against the parameterisation quietly collapsing to nothing -- a
# zero-case parametrize is a green test that checks the empty set.
assert len(MODELS) > 100, f"only {len(MODELS)} models discovered"
assert len(CHECK_IDS) == 10, CHECK_IDS


@pytest.mark.parametrize("short", MODELS)
@pytest.mark.parametrize("check_id", CHECK_IDS)
def test_model_registration(check_id, short):
    r = run_check(check_id, short)
    assert r.ok, f"[{short}] {check_id}: {r.detail}"


@pytest.mark.parametrize("r", check_global(), ids=lambda r: r.check)
def test_global_invariants(r):
    assert r.ok, f"{r.check}: {r.detail}"


class TestLinterIntegrity:
    """The linter's own failure modes. A linter that cannot fail is decor."""

    def test_known_holes_still_reproduce(self):
        """A KNOWN_HOLES entry that has been fixed must be deleted.

        That deletion is the regression test for the follow-up: once F-1 is
        fixed, leaving its entry in place turns this red, and removing it
        makes the real check load-bearing again.
        """
        stale = [f"{m}/{c} ({fid})"
                 for (m, c), fid in KNOWN_HOLES.items()
                 if run_check(c, m).status == STALE]
        assert not stale, (
            "these KNOWN_HOLES entries no longer reproduce -- delete them: "
            + ", ".join(stale))

    def test_known_holes_name_live_models_and_checks(self):
        pm = load_price_models()
        bad = [(m, c) for (m, c) in KNOWN_HOLES
               if m not in pm or c not in CHECK_IDS]
        assert not bad, f"KNOWN_HOLES references unknown model/check: {bad}"

    def test_every_exemption_is_used(self):
        """An exemption matching no model is dead text.

        Dead exemptions are how a stale allow-list keeps excusing a registry
        that has since changed shape.
        """
        unused = []
        for check_id, exs in EXEMPTIONS.items():
            for ex in exs:
                if not any(ex.covers(s) for s in MODELS):
                    unused.append(f"{check_id}: {ex.reason[:60]}")
        assert not unused, f"exemptions matching no model: {unused}"

    def test_refit_scripts_map_points_at_real_files(self):
        """Every path in the linter's own mapping must exist.

        Same class of bug as F-1, one level up: a typo here would make the
        `refit` check demand a SCRIPTS entry that could never be satisfied.
        """
        missing = sorted({p for p in REFIT_SCRIPTS.values()
                          if not (_ROOT / p).exists()})
        assert not missing, missing

    def test_a_deliberately_broken_model_fails_every_check(self):
        """Prove the checks can go red.

        A short_name registered nowhere must fail every registry check.  If
        this passes, the checks are matching something they should not.

        ``sigma_mask`` is excluded on purpose: it is conditional on a sigma
        branch existing, and a model with no branch is the `sigma` check's
        finding, not a second copy of it.
        """
        conditional = {"sigma_mask"}
        pm = load_price_models()
        sentinel = "zz_not_a_model"
        assert sentinel not in pm
        pm[sentinel] = pm["spl"]          # a real model object, wrong key
        try:
            statuses = {c: run_check(c, sentinel).status
                        for c in CHECK_IDS if c not in conditional}
        finally:
            del pm[sentinel]
        assert len(statuses) == len(CHECK_IDS) - len(conditional)
        passing = [c for c, s in statuses.items() if s != "FAIL"]
        assert not passing, (
            f"an unregistered model still passed {passing} -- those checks "
            f"cannot fail")


class TestSigmaRegistrySymmetry:
    """fit_shrinking_sigma.py's two halves must name the same models.

    Either half alone is a no-op: a branch nothing dispatches to, or a list
    entry that raises 'Unknown model' inside a worker whose exception is
    caught, printed, and dropped.
    """

    def test_branches_and_models_list_agree(self):
        branches = set(sigma_branches())
        listed = set(sigma_models_list())
        assert branches == listed, (
            f"branch-only: {sorted(branches - listed)}; "
            f"list-only: {sorted(listed - branches)}")

    def test_models_list_has_no_duplicates(self):
        listed = sigma_models_list()
        assert len(listed) == len(set(listed)), listed


class TestRefitScriptPaths:
    """F-3: generalise the SCRIPTS path assertion beyond the one entry.

    ``test_spl_registration.py`` deliberately scoped its assertion to the
    ``spl`` entry so it would not go red on F-1.  This is the general form,
    written so that F-1 -- and only F-1 -- is tolerated, by name.  When F-1
    lands, deleting the exception below is the whole of F-3.
    """

    F1_UNRESOLVED = "tools/fit_grdy.py --mode=de"

    def test_every_scripts_path_resolves_except_f1(self):
        broken = sorted({stored for _, stored in refit_scripts_entries()
                         if not (_ROOT / stored).exists()})
        assert broken == [self.F1_UNRESOLVED], (
            f"unresolvable SCRIPTS paths: {broken}. refit_all_ppl.py "
            f"os.path.exists()-checks the whole stored string, so an entry "
            f"that grows a flag is skipped with no error and its model is "
            f"never refit again.")


def test_summary_counts_are_stable():
    """Pin the shape of the sweep so a silent collapse is visible.

    If a future refactor made every check return EXEMPT, every other test
    here would still pass.
    """
    results = [run_check(c, s) for s in MODELS for c in CHECK_IDS]
    n_ok = sum(1 for r in results if r.status == "OK")
    n_known = sum(1 for r in results if r.status == KNOWN)
    assert n_ok > 400, f"only {n_ok} OK results -- checks may have collapsed"
    assert n_known == len(KNOWN_HOLES), (
        f"{n_known} KNOWN results vs {len(KNOWN_HOLES)} KNOWN_HOLES entries")
    assert sum(1 for r in results if r.status == EXEMPT) > 0
