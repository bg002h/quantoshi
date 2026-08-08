"""Registration surface for spl.

Every check here guards a step that fails SILENTLY -- the model renders fine
whether or not the step was done. Do not replace these with a visual check.
"""
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
        """Step 20. Otherwise spl is never refit and its params freeze."""
        assert "fit_spl.py" in _src("tools/refit_all_ppl.py")

    def test_in_fit_shrinking_sigma(self):
        """Step 21: both the branch and the models list."""
        s = _src("tools/fit_shrinking_sigma.py")
        assert 'model_name == "spl"' in s
        assert re.search(r"models\s*=\s*\[[^\]]*\"spl\"", s, re.S)

    def test_shrinking_sigma_branch_uses_the_class_mask(self):
        """The branch must use t >= T_MIN, matching the constructor. A
        different mask optimises a different residual set than the one
        _init_shrinking_bands derives sigma from."""
        s = _src("tools/fit_shrinking_sigma.py")
        i = s.index('model_name == "spl"')
        assert "T_MIN" in s[i:i + 800]
