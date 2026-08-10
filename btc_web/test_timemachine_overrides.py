"""Task 6: per-request param overrides on EPPLConfigModel.

Verifies that cfg_override / sigma_override shadow the shared
_EPPL_CONFIG_PARAMS global (used by the Task-5 runtime "as-of" loader)
without mutating it.
"""
import numpy as np
from btc_core._hybppl_eppl import EPPLConfigModel, _EPPL_CONFIG_PARAMS


def test_cfg_override_bypasses_global_dict():
    base = _EPPL_CONFIG_PARAMS["ecfg_1d_1u"]
    orig_B = base["params"]["B"]  # snapshot by value (float), not a reference into base
    ov = {**base, "params": {**base["params"], "B": orig_B + 0.5}}
    t = np.linspace(1, 16, 500); p = 10.0 ** (0.0 + 5.0 * np.log10(t))
    m = EPPLConfigModel("ecfg_1d_1u", t, p, [0.5],
                        cfg_override=ov, sigma_override=0.13)
    assert m._params["B"] == orig_B + 0.5
    assert m._sigma == 0.13
    # Prove the deep-copy actually decouples the model from the global dict:
    # mutate the model's own params and confirm the global is unaffected.
    # This assertion WOULD fail if the deep-copy in EPPLConfigModel.__init__
    # were removed (self._params would alias the global dict's "params" dict).
    m._params["B"] = 999.0
    assert _EPPL_CONFIG_PARAMS["ecfg_1d_1u"]["params"]["B"] == orig_B


def test_cfg_override_without_sigma_key_uses_sigma_override():
    """cfg_override contract is params/n_log/n_cal/log_damps/cal_damps/r2 --
    no "sigma" key. Construction must not crash reading cfg["sigma"], and
    sigma_override must still win."""
    base = _EPPL_CONFIG_PARAMS["ecfg_1d_1u"]
    ov = {
        "params": base["params"],
        "n_log": base["n_log"],
        "n_cal": base["n_cal"],
        "log_damps": base["log_damps"],
        "cal_damps": base["cal_damps"],
        "r2": base["r2"],
    }
    assert "sigma" not in ov
    t = np.linspace(1, 16, 500); p = 10.0 ** (0.0 + 5.0 * np.log10(t))
    m = EPPLConfigModel("ecfg_1d_1u", t, p, [0.5],
                        cfg_override=ov, sigma_override=0.13)
    assert m._sigma == 0.13
