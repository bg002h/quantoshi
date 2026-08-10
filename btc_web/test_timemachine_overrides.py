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
    orig_log_damps = list(base["log_damps"])  # snapshot by value (new list)
    ov = {**base, "params": {**base["params"], "B": orig_B + 0.5}}
    t = np.linspace(1, 16, 500); p = 10.0 ** (0.0 + 5.0 * np.log10(t))
    m = EPPLConfigModel("ecfg_1d_1u", t, p, [0.5],
                        cfg_override=ov, sigma_override=0.13)
    assert m._params["B"] == orig_B + 0.5
    assert m._sigma == 0.13

    # Genuine deep-copy regression guard. `ov = {**base, "params": {...}}`
    # only reconstructs the top-level dict and "params" -- "log_damps" is
    # NOT rebuilt by that spread, so ov["log_damps"] is literally
    # base["log_damps"] (the actual global list) going into __init__.
    # If EPPLConfigModel.__init__ did not deep-copy cfg_override,
    # self._log_damps would alias that same global list, and mutating it
    # here would corrupt _EPPL_CONFIG_PARAMS. Verified directly: with
    # copy.deepcopy monkeypatched to identity, this exact sequence leaks
    # "MUTATED" into the global list; with the real deepcopy (as shipped)
    # it does not. (NOTE: mutating m._params here would NOT be a valid
    # guard for this override shape -- ov["params"] is already a fresh
    # dict built by this test's own `{**base["params"], ...}` spread, so
    # it can never alias base["params"] regardless of __init__'s
    # behaviour. log_damps is the field this test leaves un-rebuilt.)
    assert m._log_damps is not _EPPL_CONFIG_PARAMS["ecfg_1d_1u"]["log_damps"]
    m._log_damps.append("MUTATED")
    assert _EPPL_CONFIG_PARAMS["ecfg_1d_1u"]["log_damps"] == orig_log_damps


def test_cfg_override_without_sigma_key_uses_sigma_override():
    """cfg_override contract is params/n_log/n_cal/log_damps/cal_damps/r2 --
    no "sigma" key. Construction must not crash reading cfg["sigma"], and
    sigma_override must still win. This override dict reuses base's nested
    values as-is (no test-side reconstruction of any field), so it also
    doubles as the cleanest deep-copy regression guard: every key aliases
    the global going in, so mutating m._params afterward and finding the
    global untouched is unconfounded proof of the deep-copy in __init__.
    """
    base = _EPPL_CONFIG_PARAMS["ecfg_1d_1u"]
    orig_A = base["params"]["A"]  # snapshot by value (float)
    ov = {
        "params": base["params"],
        "n_log": base["n_log"],
        "n_cal": base["n_cal"],
        "log_damps": base["log_damps"],
        "cal_damps": base["cal_damps"],
        "r2": base["r2"],
    }
    assert "sigma" not in ov
    assert ov["params"] is base["params"]  # aliased going in -- the real test
    t = np.linspace(1, 16, 500); p = 10.0 ** (0.0 + 5.0 * np.log10(t))
    m = EPPLConfigModel("ecfg_1d_1u", t, p, [0.5],
                        cfg_override=ov, sigma_override=0.13)
    assert m._sigma == 0.13

    assert m._params is not _EPPL_CONFIG_PARAMS["ecfg_1d_1u"]["params"]
    m._params["A"] = 999.0
    assert _EPPL_CONFIG_PARAMS["ecfg_1d_1u"]["params"]["A"] == orig_A
