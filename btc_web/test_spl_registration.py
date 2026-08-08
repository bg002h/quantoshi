"""Registration surface for spl.

Every check here guards a step that fails SILENTLY -- the model renders fine
whether or not the step was done. Do not replace these with a visual check.
"""
import ast
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[1]


def _src(rel):
    return (REPO / rel).read_text()


class TestPipelineIntegration:
    def test_in_build_bm_model_instances(self):
        """Step 19. update_prices.py runs build_bm_model.py after every price
        append; a model missing here is absent from every pkl rebuild."""
        assert 'instances["spl"]' in _src("tools/build_bm_model.py")

    def test_in_refit_all_ppl(self):
        """Step 20. Otherwise spl is never refit and its params freeze.

        Substring-matching the source is not enough. refit_all_ppl.py
        os.path.exists()-checks the whole path string, so an entry written
        ("SatPL", "tools/fit_spl.py --update") satisfies `"fit_spl.py" in
        src` while resolving to nothing -- and the script is silently
        skipped on every live run, which is the exact freeze this test
        exists to catch. That failure mode is LIVE a few lines away in the
        same list, so parse the entry and assert its path resolves.
        """
        tree = ast.parse(_src("tools/refit_all_ppl.py"))
        scripts = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "SCRIPTS"
                for t in node.targets
            ):
                scripts = ast.literal_eval(node.value)
        assert scripts is not None, "SCRIPTS not found in refit_all_ppl.py"

        paths = [p for _, p in scripts if "fit_spl.py" in p]
        assert len(paths) == 1, f"expected exactly one spl entry, got {paths}"
        # Deliberately scoped to the spl entry only. Generalising this to
        # every SCRIPTS entry would go red on a pre-existing unrelated
        # entry whose path does not resolve; that one is out of scope.
        assert (REPO / paths[0]).exists(), (
            f"SCRIPTS entry {paths[0]!r} does not resolve on disk; "
            "refit_all_ppl.py would print 'NOT FOUND -- skipping' and "
            "never refit spl"
        )

    def test_in_fit_shrinking_sigma(self):
        """Step 21: both the branch and the models list."""
        s = _src("tools/fit_shrinking_sigma.py")
        assert 'model_name == "spl"' in s
        assert re.search(r"models\s*=\s*\[[^\]]*\"spl\"", s, re.S)

    def test_shrinking_sigma_branch_uses_the_class_mask(self):
        """The branch must use t >= T_MIN, matching the constructor. A
        different mask optimises a different residual set than the one
        _init_shrinking_bands derives sigma from.

        Match the executable mask, not the identifier. A bare `"T_MIN" in
        <window>` is satisfied by the import line and by prose comments
        that merely mention T_MIN -- it stays green with the mask deleted
        or replaced by a wrong literal, i.e. green through exactly the
        "cleanup" it claims to prevent.
        """
        s = _src("tools/fit_shrinking_sigma.py")
        i = s.index('model_name == "spl"')
        assert re.search(r"np\.where\(\s*t\s*>=\s*T_MIN", s[i:])
