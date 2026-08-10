"""Task 6: per-request param overrides on EPPLConfigModel.

Verifies that cfg_override / sigma_override shadow the shared
_EPPL_CONFIG_PARAMS global (used by the Task-5 runtime "as-of" loader)
without mutating it.
"""
import numpy as np
from btc_core._hybppl_eppl import EPPLConfigModel, _EPPL_CONFIG_PARAMS


def test_cfg_override_bypasses_global_dict():
    base = _EPPL_CONFIG_PARAMS["ecfg_1d_1u"]
    ov = {**base, "params": {**base["params"], "B": base["params"]["B"] + 0.5}}
    t = np.linspace(1, 16, 500); p = 10.0 ** (0.0 + 5.0 * np.log10(t))
    m = EPPLConfigModel("ecfg_1d_1u", t, p, [0.5],
                        cfg_override=ov, sigma_override=0.13)
    assert m._params["B"] == base["params"]["B"] + 0.5
    assert m._sigma == 0.13
    assert _EPPL_CONFIG_PARAMS["ecfg_1d_1u"]["params"]["B"] == base["params"]["B"]
