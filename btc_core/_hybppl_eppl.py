"""Hybrid PPL + Entropy PPL families.

HybPPLModel / HybPPLDDModel / Hyb2L / Hyb2C / Hyb2B / Hyb4D subclass LPPLModel.
HybPPLConfigModel + EntropyPPLModel + EPPLConfigModel use _ShrinkingBandsMixin
directly. Kept together because they share the same dispatch pattern and often
co-vary in development.
"""

import copy

import numpy as np

from btc_core._helpers import _lazy_norm, _DEFAULT_QS
from btc_core._base import _ShrinkingBandsMixin
from btc_core._lppl import LPPLModel
from time_basis import T_MIN


class HybPPLModel(LPPLModel):
    """Hybrid Log+Linear PPL: log-periodic damped + linear-periodic undamped.

    Fits: log10(price) = A + B*log10(t) + C1*t^(-D)*cos(ω_log*ln(t)+φ1)
                       + C2*cos(ω_cal*t+φ2)

    Combines LPPL's log-periodic damped oscillation (captures early-Bitcoin
    self-similarity) with a linear-periodic undamped term (captures the
    halving cycle). 9 parameters — same count as LPPL₂.

    _W is the log-time angular frequency (like LPPL).
    _W2 is the calendar angular frequency in rad/yr (like LinPPL).
    """
    name = "HybPPL"
    short_name = "hybppl"
    legend_name = "HybPPL"
    dash_style = "dashdot"

    # Fitted parameters (will be overwritten by fit_hybppl.py --update)
    _A   = -1.150969  
    _B   =                                      5.057785  
    _C   =                                      0.691402  
    _W   =                                      7.434710  
    _PHI =                                      1.443941  
    _D   =                                      0.712061  
    _C2  =                                      0.230793  
    _W2  =                                      1.733428  
    _PHI2 = -1.933169  

    def _lppl_log10(self, t):
        """Evaluate hybrid model: log-periodic damped + linear-periodic undamped."""
        t = np.asarray(t, float)
        t_safe = np.maximum(t, 0.1)
        damped = self._C * t_safe ** (-self._D) * np.cos(self._W * np.log(t_safe) + self._PHI)
        undamped = self._C2 * np.cos(self._W2 * t_safe + self._PHI2)
        return self._A + self._B * np.log10(t_safe) + damped + undamped

    component_names = [
        "A (constant)",
        "B\u00b7log\u2081\u2080(t)",
        "damped log osc (\u03c9_log)",
        "undamped cal osc (\u03c9_cal)",
    ]
    formula_log10_latex = (
        r"A + B \log_{10}(t) + C_1 t^{-D} \cos(\omega_{\text{log}} \ln t + \varphi_1)"
        r" + C_2 \cos(\omega_{\text{cal}} t + \varphi_2)"
    )
    formula_product_latex = (
        r"10^A \cdot t^B"
        r" \cdot 10^{\,C_1 t^{-D} \cos(\omega_{\text{log}} \ln t + \varphi_1)}"
        r" \cdot 10^{\,C_2 \cos(\omega_{\text{cal}} t + \varphi_2)}"
    )
    component_details = {
        "A (constant)":           ("A",                         [("A", "_A")]),
        "B\u00b7log\u2081\u2080(t)": ("B\u00b7log\u2081\u2080(t)", [("B", "_B")]),
        "damped log osc (\u03c9_log)": (
            "C\u2081\u00b7t^(-D)\u00b7cos(\u03c9_log\u00b7ln(t)+\u03c6\u2081)",
            [("C\u2081", "_C"), ("D", "_D"),
             ("\u03c9_log", "_W"), ("\u03c6\u2081", "_PHI")]),
        "undamped cal osc (\u03c9_cal)": (
            "C\u2082\u00b7cos(\u03c9_cal\u00b7t+\u03c6\u2082)",
            [("C\u2082", "_C2"), ("\u03c9_cal", "_W2"),
             ("\u03c6\u2082", "_PHI2")]),
    }

    def components(self, t):
        """Hybrid: log-periodic damped + linear-periodic undamped."""
        t = np.asarray(t, float)
        t_safe = np.maximum(t, 0.1)
        return {
            "A (constant)":                        np.full_like(t_safe, self._A),
            "B\u00b7log\u2081\u2080(t)":            self._B * np.log10(t_safe),
            "damped log osc (\u03c9_log)":          self._C * t_safe ** (-self._D) * np.cos(
                self._W * np.log(t_safe) + self._PHI),
            "undamped cal osc (\u03c9_cal)":        self._C2 * np.cos(
                self._W2 * t_safe + self._PHI2),
        }


class HybPPLDDModel(LPPLModel):
    """HybPPL (DD — Double Damped): both oscillators damped, non-excess.

    Fits: log10(price) = A + B*log10(t)
                       + C1*t^(-D1)*cos(W_log*ln(t) + PHI1)
                       + C2*t^(-D2)*cos(W_cal*t + PHI2)

    Like HybPPL but with an independent damping exponent on each oscillator.
    Tests whether the halving cycle is permanent (D2 near 0) or decaying.
    10 parameters — one more than HybPPL's 9.
    """
    name = "HybPPL (DD)"
    short_name = "hybppl_dd"
    legend_name = "HybPPL (DD)"
    dash_style = "dashdot"

    # Fitted parameters (will be overwritten by fit_hybppl_dd.py --update)
    _A     = -1.160848  
    _B     =                             5.072492  
    _C1    =                             1.009027  
    _W_log =                             7.350446  
    _PHI1  =                             1.685164  
    _D1    =                             0.756430  
    _C2    =                             0.865470  
    _W_cal =                             7.849093  
    _PHI2  =          2.883853  
    _D2    =                             1.782217  

    def _lppl_log10(self, t):
        """Evaluate double-damped hybrid model."""
        t = np.asarray(t, float)
        t_safe = np.maximum(t, 0.1)
        damped_log = self._C1 * t_safe ** (-self._D1) * np.cos(
            self._W_log * np.log(t_safe) + self._PHI1)
        damped_cal = self._C2 * t_safe ** (-self._D2) * np.cos(
            self._W_cal * t_safe + self._PHI2)
        return self._A + self._B * np.log10(t_safe) + damped_log + damped_cal

    component_names = [
        "A (constant)",
        "B\u00b7log\u2081\u2080(t)",
        "damped log osc (\u03c9_log)",
        "damped cal osc (\u03c9_cal)",
    ]
    support_component_names = []
    formula_log10_latex = (
        r"A + B \log_{10}(t)"
        r" + C_1 t^{-D_1} \cos(\omega_{\text{log}} \ln t + \varphi_1)"
        r" + C_2 t^{-D_2} \cos(\omega_{\text{cal}} t + \varphi_2)"
    )
    formula_product_latex = (
        r"10^A \cdot t^B"
        r" \cdot 10^{\,C_1 t^{-D_1} \cos(\omega_{\text{log}} \ln t + \varphi_1)}"
        r" \cdot 10^{\,C_2 t^{-D_2} \cos(\omega_{\text{cal}} t + \varphi_2)}"
    )
    component_details = {
        "A (constant)":           ("A",                         [("A", "_A")]),
        "B\u00b7log\u2081\u2080(t)": ("B\u00b7log\u2081\u2080(t)", [("B", "_B")]),
        "damped log osc (\u03c9_log)": (
            "C\u2081\u00b7t^(-D\u2081)\u00b7cos(\u03c9_log\u00b7ln(t)+\u03c6\u2081)",
            [("C\u2081", "_C1"), ("D\u2081", "_D1"),
             ("\u03c9_log", "_W_log"), ("\u03c6\u2081", "_PHI1")]),
        "damped cal osc (\u03c9_cal)": (
            "C\u2082\u00b7t^(-D\u2082)\u00b7cos(\u03c9_cal\u00b7t+\u03c6\u2082)",
            [("C\u2082", "_C2"), ("D\u2082", "_D2"),
             ("\u03c9_cal", "_W_cal"), ("\u03c6\u2082", "_PHI2")]),
    }

    def components(self, t):
        """Double-damped hybrid: both oscillators have independent damping."""
        t = np.asarray(t, float)
        t_safe = np.maximum(t, 0.1)
        return {
            "A (constant)":                        np.full_like(t_safe, self._A),
            "B\u00b7log\u2081\u2080(t)":            self._B * np.log10(t_safe),
            "damped log osc (\u03c9_log)":          self._C1 * t_safe ** (-self._D1) * np.cos(
                self._W_log * np.log(t_safe) + self._PHI1),
            "damped cal osc (\u03c9_cal)":          self._C2 * t_safe ** (-self._D2) * np.cos(
                self._W_cal * t_safe + self._PHI2),
        }


class Hyb2LModel(LPPLModel):
    """HybPPL + 2nd log-periodic oscillation.

    Fits: log10(price) = A + B*log10(t)
                       + C1*t^(-D1)*cos(W1*ln(t)+PHI1)
                       + C2*cos(Wc*t+PHI2)
                       + C3*t^(-D2)*cos(W2*ln(t)+PHI3)

    Adds a second damped log-periodic harmonic to the baseline HybPPL.
    13 parameters.
    """
    name = "HybPPL +2L"
    short_name = "hyb2l"
    legend_name = "Hyb2L"
    dash_style = "dashdot"
    quantized = True

    # Fitted parameters (will be overwritten by fit_hyb2l.py --update)
    _A    = -1.121326  
    _B    =                          5.026795  
    _C1   =                          0.418118  
    _W1   =                     15.986763  
    _PHI1 =                          1.897151  
    _D1   =                          1.037490  
    _C2   =                          0.250158  
    _Wc   =                          1.720558  
    _PHI2 = -1.763151  
    _C3   =                          0.771853  
    _W2   =                      7.507615  
    _PHI3 =                          1.272938  
    _D2   =                          0.784387  

    def _lppl_log10(self, t):
        t = np.asarray(t, float)
        t_safe = np.maximum(t, 0.1)
        osc1 = self._C1 * t_safe ** (-self._D1) * np.cos(self._W1 * np.log(t_safe) + self._PHI1)
        cal  = self._C2 * np.cos(self._Wc * t_safe + self._PHI2)
        osc2 = self._C3 * t_safe ** (-self._D2) * np.cos(self._W2 * np.log(t_safe) + self._PHI3)
        return self._A + self._B * np.log10(t_safe) + osc1 + cal + osc2

    component_names = [
        "A (constant)",
        "B\u00b7log\u2081\u2080(t)",
        "damped log osc 1 (\u03c9\u2081)",
        "undamped cal osc (\u03c9_cal)",
        "damped log osc 2 (\u03c9\u2082)",
    ]
    formula_log10_latex = (
        r"A + B \log_{10}(t)"
        r" + C_1 t^{-D_1} \cos(\omega_1 \ln t + \varphi_1)"
        r" + C_2 \cos(\omega_c t + \varphi_2)"
        r" + C_3 t^{-D_2} \cos(\omega_2 \ln t + \varphi_3)"
    )
    formula_product_latex = (
        r"10^A \cdot t^B"
        r" \cdot 10^{\,C_1 t^{-D_1} \cos(\omega_1 \ln t + \varphi_1)}"
        r" \cdot 10^{\,C_2 \cos(\omega_c t + \varphi_2)}"
        r" \cdot 10^{\,C_3 t^{-D_2} \cos(\omega_2 \ln t + \varphi_3)}"
    )
    component_details = {
        "A (constant)":           ("A", [("A", "_A")]),
        "B\u00b7log\u2081\u2080(t)": ("B\u00b7log\u2081\u2080(t)", [("B", "_B")]),
        "damped log osc 1 (\u03c9\u2081)": (
            "C\u2081\u00b7t^(-D\u2081)\u00b7cos(\u03c9\u2081\u00b7ln(t)+\u03c6\u2081)",
            [("C\u2081", "_C1"), ("D\u2081", "_D1"),
             ("\u03c9\u2081", "_W1"), ("\u03c6\u2081", "_PHI1")]),
        "undamped cal osc (\u03c9_cal)": (
            "C\u2082\u00b7cos(\u03c9_c\u00b7t+\u03c6\u2082)",
            [("C\u2082", "_C2"), ("\u03c9_c", "_Wc"), ("\u03c6\u2082", "_PHI2")]),
        "damped log osc 2 (\u03c9\u2082)": (
            "C\u2083\u00b7t^(-D\u2082)\u00b7cos(\u03c9\u2082\u00b7ln(t)+\u03c6\u2083)",
            [("C\u2083", "_C3"), ("D\u2082", "_D2"),
             ("\u03c9\u2082", "_W2"), ("\u03c6\u2083", "_PHI3")]),
    }

    def components(self, t):
        t = np.asarray(t, float)
        t_safe = np.maximum(t, 0.1)
        return {
            "A (constant)":                np.full_like(t_safe, self._A),
            "B\u00b7log\u2081\u2080(t)":    self._B * np.log10(t_safe),
            "damped log osc 1 (\u03c9\u2081)": self._C1 * t_safe ** (-self._D1) * np.cos(
                self._W1 * np.log(t_safe) + self._PHI1),
            "undamped cal osc (\u03c9_cal)": self._C2 * np.cos(
                self._Wc * t_safe + self._PHI2),
            "damped log osc 2 (\u03c9\u2082)": self._C3 * t_safe ** (-self._D2) * np.cos(
                self._W2 * np.log(t_safe) + self._PHI3),
        }


class Hyb2CModel(LPPLModel):
    """HybPPL + 2nd calendar-periodic oscillation.

    Fits: log10(price) = A + B*log10(t)
                       + C1*t^(-D)*cos(W1*ln(t)+PHI1)
                       + C2*cos(Wc1*t+PHI2)
                       + C3*cos(Wc2*t+PHI3)

    Adds a second undamped calendar-periodic term. The 2nd frequency
    (~1.88yr) is roughly half the halving cycle — may capture
    sub-halving market structure.
    12 parameters.
    """
    name = "HybPPL +2C"
    short_name = "hyb2c"
    legend_name = "Hyb2C"
    dash_style = "dashdot"
    quantized = True

    # Fitted parameters (will be overwritten by fit_hyb2c.py --update)
    _A    = -1.147802  
    _B    =                          5.058465  
    _C1   =                          0.765491  
    _W1   =                          7.413681  
    _PHI1 =                          1.594622  
    _D    =                          0.758388  
    _C2   =                          0.226821  
    _Wc1  =                          1.744751  
    _PHI2 = -2.063416  
    _C3   =                          0.104482  
    _Wc2  =                          3.299584  
    _PHI3 = -2.603517  

    def _lppl_log10(self, t):
        t = np.asarray(t, float)
        t_safe = np.maximum(t, 0.1)
        osc  = self._C1 * t_safe ** (-self._D) * np.cos(self._W1 * np.log(t_safe) + self._PHI1)
        cal1 = self._C2 * np.cos(self._Wc1 * t_safe + self._PHI2)
        cal2 = self._C3 * np.cos(self._Wc2 * t_safe + self._PHI3)
        return self._A + self._B * np.log10(t_safe) + osc + cal1 + cal2

    component_names = [
        "A (constant)",
        "B\u00b7log\u2081\u2080(t)",
        "damped log osc (\u03c9_log)",
        "undamped cal osc 1 (\u03c9_c\u2081)",
        "undamped cal osc 2 (\u03c9_c\u2082)",
    ]
    formula_log10_latex = (
        r"A + B \log_{10}(t)"
        r" + C_1 t^{-D} \cos(\omega_1 \ln t + \varphi_1)"
        r" + C_2 \cos(\omega_{c1} t + \varphi_2)"
        r" + C_3 \cos(\omega_{c2} t + \varphi_3)"
    )
    formula_product_latex = (
        r"10^A \cdot t^B"
        r" \cdot 10^{\,C_1 t^{-D} \cos(\omega_1 \ln t + \varphi_1)}"
        r" \cdot 10^{\,C_2 \cos(\omega_{c1} t + \varphi_2)}"
        r" \cdot 10^{\,C_3 \cos(\omega_{c2} t + \varphi_3)}"
    )
    component_details = {
        "A (constant)":           ("A", [("A", "_A")]),
        "B\u00b7log\u2081\u2080(t)": ("B\u00b7log\u2081\u2080(t)", [("B", "_B")]),
        "damped log osc (\u03c9_log)": (
            "C\u2081\u00b7t^(-D)\u00b7cos(\u03c9\u2081\u00b7ln(t)+\u03c6\u2081)",
            [("C\u2081", "_C1"), ("D", "_D"),
             ("\u03c9\u2081", "_W1"), ("\u03c6\u2081", "_PHI1")]),
        "undamped cal osc 1 (\u03c9_c\u2081)": (
            "C\u2082\u00b7cos(\u03c9_c\u2081\u00b7t+\u03c6\u2082)",
            [("C\u2082", "_C2"), ("\u03c9_c\u2081", "_Wc1"), ("\u03c6\u2082", "_PHI2")]),
        "undamped cal osc 2 (\u03c9_c\u2082)": (
            "C\u2083\u00b7cos(\u03c9_c\u2082\u00b7t+\u03c6\u2083)",
            [("C\u2083", "_C3"), ("\u03c9_c\u2082", "_Wc2"), ("\u03c6\u2083", "_PHI3")]),
    }

    def components(self, t):
        t = np.asarray(t, float)
        t_safe = np.maximum(t, 0.1)
        return {
            "A (constant)":                    np.full_like(t_safe, self._A),
            "B\u00b7log\u2081\u2080(t)":        self._B * np.log10(t_safe),
            "damped log osc (\u03c9_log)":      self._C1 * t_safe ** (-self._D) * np.cos(
                self._W1 * np.log(t_safe) + self._PHI1),
            "undamped cal osc 1 (\u03c9_c\u2081)": self._C2 * np.cos(
                self._Wc1 * t_safe + self._PHI2),
            "undamped cal osc 2 (\u03c9_c\u2082)": self._C3 * np.cos(
                self._Wc2 * t_safe + self._PHI3),
        }


class Hyb2BModel(LPPLModel):
    """HybPPL + 2nd log-periodic + 2nd calendar-periodic.

    Fits: log10(price) = A + B*log10(t)
                       + C1*t^(-D1)*cos(W1*ln(t)+PHI1)
                       + C2*cos(Wc1*t+PHI2)
                       + C3*t^(-D2)*cos(W2*ln(t)+PHI3)
                       + C4*cos(Wc2*t+PHI4)

    Full second-frequency model: both log-periodic and calendar-periodic
    get a second harmonic. 16 parameters — highest R² in the family.
    """
    name = "HybPPL +2B"
    short_name = "hyb2b"
    legend_name = "Hyb2B"
    dash_style = "dashdot"
    quantized = True

    # Fitted parameters (will be overwritten by fit_hyb2b.py --update)
    _A    = -1.137965  
    _B    =                          5.044992  
    _C1   =                          0.792407  
    _W1   =                         7.404269  
    _PHI1 =                          1.552133  
    _D1   =                          0.780730  
    _C2   =                          0.233860  
    _Wc1  =                          1.733446  
    _PHI2 = -1.959773  
    _C3   =                          0.122235  
    _W2   =                  36.864687  
    _PHI3 =                       2.288604  
    _D2   =                          0.258743  
    _C4   =                          0.089123  
    _Wc2  =                          3.351444  
    _PHI4 =           -2.898941  

    def _lppl_log10(self, t):
        t = np.asarray(t, float)
        t_safe = np.maximum(t, 0.1)
        osc1 = self._C1 * t_safe ** (-self._D1) * np.cos(self._W1 * np.log(t_safe) + self._PHI1)
        cal1 = self._C2 * np.cos(self._Wc1 * t_safe + self._PHI2)
        osc2 = self._C3 * t_safe ** (-self._D2) * np.cos(self._W2 * np.log(t_safe) + self._PHI3)
        cal2 = self._C4 * np.cos(self._Wc2 * t_safe + self._PHI4)
        return self._A + self._B * np.log10(t_safe) + osc1 + cal1 + osc2 + cal2

    component_names = [
        "A (constant)",
        "B\u00b7log\u2081\u2080(t)",
        "damped log osc 1 (\u03c9_l\u2081)",
        "undamped cal osc 1 (\u03c9_c\u2081)",
        "damped log osc 2 (\u03c9_l\u2082)",
        "undamped cal osc 2 (\u03c9_c\u2082)",
    ]
    formula_log10_latex = (
        r"A + B \log_{10}(t)"
        r" + C_1 t^{-D_1} \cos(\omega_{l1} \ln t + \varphi_1)"
        r" + C_2 \cos(\omega_{c1} t + \varphi_2)"
        r" + C_3 t^{-D_2} \cos(\omega_{l2} \ln t + \varphi_3)"
        r" + C_4 \cos(\omega_{c2} t + \varphi_4)"
    )
    formula_product_latex = (
        r"10^A \cdot t^B"
        r" \cdot 10^{\,C_1 t^{-D_1} \cos(\omega_{l1} \ln t + \varphi_1)}"
        r" \cdot 10^{\,C_2 \cos(\omega_{c1} t + \varphi_2)}"
        r" \cdot 10^{\,C_3 t^{-D_2} \cos(\omega_{l2} \ln t + \varphi_3)}"
        r" \cdot 10^{\,C_4 \cos(\omega_{c2} t + \varphi_4)}"
    )
    component_details = {
        "A (constant)":           ("A", [("A", "_A")]),
        "B\u00b7log\u2081\u2080(t)": ("B\u00b7log\u2081\u2080(t)", [("B", "_B")]),
        "damped log osc 1 (\u03c9_l\u2081)": (
            "C\u2081\u00b7t^(-D\u2081)\u00b7cos(\u03c9_l\u2081\u00b7ln(t)+\u03c6\u2081)",
            [("C\u2081", "_C1"), ("D\u2081", "_D1"),
             ("\u03c9_l\u2081", "_W1"), ("\u03c6\u2081", "_PHI1")]),
        "undamped cal osc 1 (\u03c9_c\u2081)": (
            "C\u2082\u00b7cos(\u03c9_c\u2081\u00b7t+\u03c6\u2082)",
            [("C\u2082", "_C2"), ("\u03c9_c\u2081", "_Wc1"), ("\u03c6\u2082", "_PHI2")]),
        "damped log osc 2 (\u03c9_l\u2082)": (
            "C\u2083\u00b7t^(-D\u2082)\u00b7cos(\u03c9_l\u2082\u00b7ln(t)+\u03c6\u2083)",
            [("C\u2083", "_C3"), ("D\u2082", "_D2"),
             ("\u03c9_l\u2082", "_W2"), ("\u03c6\u2083", "_PHI3")]),
        "undamped cal osc 2 (\u03c9_c\u2082)": (
            "C\u2084\u00b7cos(\u03c9_c\u2082\u00b7t+\u03c6\u2084)",
            [("C\u2084", "_C4"), ("\u03c9_c\u2082", "_Wc2"), ("\u03c6\u2084", "_PHI4")]),
    }

    def components(self, t):
        t = np.asarray(t, float)
        t_safe = np.maximum(t, 0.1)
        return {
            "A (constant)":                        np.full_like(t_safe, self._A),
            "B\u00b7log\u2081\u2080(t)":            self._B * np.log10(t_safe),
            "damped log osc 1 (\u03c9_l\u2081)":   self._C1 * t_safe ** (-self._D1) * np.cos(
                self._W1 * np.log(t_safe) + self._PHI1),
            "undamped cal osc 1 (\u03c9_c\u2081)": self._C2 * np.cos(
                self._Wc1 * t_safe + self._PHI2),
            "damped log osc 2 (\u03c9_l\u2082)":   self._C3 * t_safe ** (-self._D2) * np.cos(
                self._W2 * np.log(t_safe) + self._PHI3),
            "undamped cal osc 2 (\u03c9_c\u2082)": self._C4 * np.cos(
                self._Wc2 * t_safe + self._PHI4),
        }


class Hyb4DModel(LPPLModel):
    """HybPPL 4D — all 4 oscillatory components damped.

    Fits: log10(price) = A + B*log10(t)
                       + C1*t^(-D1)*cos(W1*ln(t)+PHI1)
                       + C2*t^(-Dc1)*cos(Wc1*t+PHI2)
                       + C3*t^(-D2)*cos(W2*ln(t)+PHI3)
                       + C4*t^(-Dc2)*cos(Wc2*t+PHI4)

    All four oscillators carry damping exponents. 18 parameters.
    Compared to Hyb2B (16 params, R²=0.993), adding 2 extra D params
    yields WORSE fit (R²=0.992, BIC=-22624 vs -23203). The calendar
    terms resist damping — Dc2≈0.076 is near zero, meaning the 2nd
    calendar oscillator WANTS to be undamped.
    """
    name = "HybPPL 4D"
    short_name = "hyb4d"
    legend_name = "Hyb4D"
    dash_style = "dashdot"
    quantized = True

    # Fitted parameters (will be overwritten by fit_hyb4d.py --update)
    _A    = -1.083720  
    _B    =                          4.967562  
    _C1   =                          0.454158  
    _W1   =                         4.478189  
    _PHI1 =                          1.361262  
    _D1   =                          0.764935  
    _C2   =                          0.535458  
    _Wc1  =                          1.745567  
    _PHI2 =  -2.106597  
    _Dc1  =                          0.329941  
    _C3   =                          1.008994  
    _W2   =                    7.522535  
    _PHI3 =                       1.352950  
    _D2   =                          1.086903  
    _C4   =                          0.646195  
    _Wc2  =                     10.309498  
    _PHI4 =       -1.220980  
    _Dc2  =                          1.364291  

    def _lppl_log10(self, t):
        t = np.asarray(t, float)
        t_safe = np.maximum(t, 0.1)
        osc1 = self._C1 * t_safe ** (-self._D1) * np.cos(self._W1 * np.log(t_safe) + self._PHI1)
        cal1 = self._C2 * t_safe ** (-self._Dc1) * np.cos(self._Wc1 * t_safe + self._PHI2)
        osc2 = self._C3 * t_safe ** (-self._D2) * np.cos(self._W2 * np.log(t_safe) + self._PHI3)
        cal2 = self._C4 * t_safe ** (-self._Dc2) * np.cos(self._Wc2 * t_safe + self._PHI4)
        return self._A + self._B * np.log10(t_safe) + osc1 + cal1 + osc2 + cal2

    component_names = [
        "A (constant)",
        "B\u00b7log\u2081\u2080(t)",
        "damped log osc 1 (\u03c9_l\u2081)",
        "damped cal osc 1 (\u03c9_c\u2081)",
        "damped log osc 2 (\u03c9_l\u2082)",
        "damped cal osc 2 (\u03c9_c\u2082)",
    ]
    formula_log10_latex = (
        r"A + B \log_{10}(t)"
        r" + C_1 t^{-D_1} \cos(\omega_{l1} \ln t + \varphi_1)"
        r" + C_2 t^{-D_{c1}} \cos(\omega_{c1} t + \varphi_2)"
        r" + C_3 t^{-D_2} \cos(\omega_{l2} \ln t + \varphi_3)"
        r" + C_4 t^{-D_{c2}} \cos(\omega_{c2} t + \varphi_4)"
    )
    formula_product_latex = (
        r"10^A \cdot t^B"
        r" \cdot 10^{\,C_1 t^{-D_1} \cos(\omega_{l1} \ln t + \varphi_1)}"
        r" \cdot 10^{\,C_2 t^{-D_{c1}} \cos(\omega_{c1} t + \varphi_2)}"
        r" \cdot 10^{\,C_3 t^{-D_2} \cos(\omega_{l2} \ln t + \varphi_3)}"
        r" \cdot 10^{\,C_4 t^{-D_{c2}} \cos(\omega_{c2} t + \varphi_4)}"
    )
    component_details = {
        "A (constant)":           ("A", [("A", "_A")]),
        "B\u00b7log\u2081\u2080(t)": ("B\u00b7log\u2081\u2080(t)", [("B", "_B")]),
        "damped log osc 1 (\u03c9_l\u2081)": (
            "C\u2081\u00b7t^(-D\u2081)\u00b7cos(\u03c9_l\u2081\u00b7ln(t)+\u03c6\u2081)",
            [("C\u2081", "_C1"), ("D\u2081", "_D1"),
             ("\u03c9_l\u2081", "_W1"), ("\u03c6\u2081", "_PHI1")]),
        "damped cal osc 1 (\u03c9_c\u2081)": (
            "C\u2082\u00b7t^(-D_c\u2081)\u00b7cos(\u03c9_c\u2081\u00b7t+\u03c6\u2082)",
            [("C\u2082", "_C2"), ("D_c\u2081", "_Dc1"),
             ("\u03c9_c\u2081", "_Wc1"), ("\u03c6\u2082", "_PHI2")]),
        "damped log osc 2 (\u03c9_l\u2082)": (
            "C\u2083\u00b7t^(-D\u2082)\u00b7cos(\u03c9_l\u2082\u00b7ln(t)+\u03c6\u2083)",
            [("C\u2083", "_C3"), ("D\u2082", "_D2"),
             ("\u03c9_l\u2082", "_W2"), ("\u03c6\u2083", "_PHI3")]),
        "damped cal osc 2 (\u03c9_c\u2082)": (
            "C\u2084\u00b7t^(-D_c\u2082)\u00b7cos(\u03c9_c\u2082\u00b7t+\u03c6\u2084)",
            [("C\u2084", "_C4"), ("D_c\u2082", "_Dc2"),
             ("\u03c9_c\u2082", "_Wc2"), ("\u03c6\u2084", "_PHI4")]),
    }

    def components(self, t):
        t = np.asarray(t, float)
        t_safe = np.maximum(t, 0.1)
        return {
            "A (constant)":                        np.full_like(t_safe, self._A),
            "B\u00b7log\u2081\u2080(t)":            self._B * np.log10(t_safe),
            "damped log osc 1 (\u03c9_l\u2081)":   self._C1 * t_safe ** (-self._D1) * np.cos(
                self._W1 * np.log(t_safe) + self._PHI1),
            "damped cal osc 1 (\u03c9_c\u2081)":   self._C2 * t_safe ** (-self._Dc1) * np.cos(
                self._Wc1 * t_safe + self._PHI2),
            "damped log osc 2 (\u03c9_l\u2082)":   self._C3 * t_safe ** (-self._D2) * np.cos(
                self._W2 * np.log(t_safe) + self._PHI3),
            "damped cal osc 2 (\u03c9_c\u2082)":   self._C4 * t_safe ** (-self._Dc2) * np.cos(
                self._Wc2 * t_safe + self._PHI4),
        }


# ── EPPL config params (auto-generated) ──
_EPPL_CONFIG_PARAMS = {
    "ecfg_0_0": {"n_log": 0, "n_cal": 0, "log_damps": [], "cal_damps": [], "params": {"A": -1.154625, "B": 5.050620}, "r2": 0.963450, "sigma": 0.292985},
    "ecfg_0_1d": {"n_log": 0, "n_cal": 1, "log_damps": [], "cal_damps": ['d'], "params": {"A": -1.168203, "B": 5.063956, "C_cal": 0.366877, "W_cal": 1.746196, "PHI_cal": -2.100014, "w_cal": 0.053092}, "r2": 0.981439, "sigma": 0.208788},
    "ecfg_0_1u": {"n_log": 0, "n_cal": 1, "log_damps": [], "cal_damps": ['u'], "params": {"A": -1.212876, "B": 5.110816, "C_cal": 0.275969, "W_cal": 1.766224, "PHI_cal": -2.286279}, "r2": 0.979724, "sigma": 0.218217},
    "ecfg_0_2dd": {"n_log": 0, "n_cal": 2, "log_damps": [], "cal_damps": ['d', 'd'], "params": {"A": -1.078368, "B": 4.971764, "C_cal1": 0.609075, "W_cal1": 4.210510, "PHI_cal1": -2.129473, "w_cal1": 0.330101, "C_cal2": 0.387366, "W_cal2": 1.718987, "PHI_cal2": -1.844772, "w_cal2": 0.055281}, "r2": 0.986628, "sigma": 0.177212},
    "ecfg_0_2du": {"n_log": 0, "n_cal": 2, "log_damps": [], "cal_damps": ['d', 'u'], "params": {"A": -1.148031, "B": 5.043941, "C_cal1": 0.373021, "W_cal1": 1.740326, "PHI_cal1": -2.039736, "w_cal1": 0.054456, "C_cal2": 0.125060, "W_cal2": 3.151931, "PHI_cal2": -0.846450}, "r2": 0.984773, "sigma": 0.189108},
    "ecfg_0_2uu": {"n_log": 0, "n_cal": 2, "log_damps": [], "cal_damps": ['u', 'u'], "params": {"A": -1.328513, "B": 5.242820, "C_cal1": 0.152629, "W_cal1": 0.811031, "PHI_cal1": -2.732733, "C_cal2": 0.293660, "W_cal2": 1.781992, "PHI_cal2": -2.418297}, "r2": 0.983826, "sigma": 0.194901},
    "ecfg_1d_0": {"n_log": 1, "n_cal": 0, "log_damps": ['d'], "cal_damps": [], "params": {"A": -1.119705, "B": 5.026169, "C_log": 0.534952, "W_log": 7.701080, "PHI_log": 1.202356, "w_log": 0.098469}, "r2": 0.983355, "sigma": 0.197717},
    "ecfg_1d_1d": {"n_log": 1, "n_cal": 1, "log_damps": ['d'], "cal_damps": ['d'], "params": {"A": -1.121022, "B": 5.028819, "C_log": 0.544158, "W_log": 7.685388, "PHI_log": 1.225845, "w_log": 0.099832, "C_cal": 0.223933, "W_cal": 10.000000, "PHI_cal": -0.432545, "w_cal": 0.172894}, "r2": 0.985201, "sigma": 0.186429},
    "ecfg_1d_1u": {"n_log": 1, "n_cal": 1, "log_damps": ['d'], "cal_damps": ['u'], "params": {"A": -1.186514, "B": 5.096203, "C_log": 0.478491, "W_log": 7.799395, "PHI_log": 1.320001, "w_log": 0.105503, "C_cal": 0.197496, "W_cal": 1.858877, "PHI_cal": 2.787863}, "r2": 0.989710, "sigma": 0.155457},
    "ecfg_1d_2dd": {"n_log": 1, "n_cal": 2, "log_damps": ['d'], "cal_damps": ['d', 'd'], "params": {"A": -1.076756, "B": 4.956668, "C_log": 0.250531, "W_log": 7.170997, "PHI_log": 2.031400, "w_log": 0.071533, "C_cal1": 0.282834, "W_cal1": 1.742122, "PHI_cal1": -2.034056, "w_cal1": 0.044679, "C_cal2": 0.482887, "W_cal2": 4.579858, "PHI_cal2": -2.912572, "w_cal2": 0.373287}, "r2": 0.990607, "sigma": 0.148524},
    "ecfg_1d_2du": {"n_log": 1, "n_cal": 2, "log_damps": ['d'], "cal_damps": ['d', 'u'], "params": {"A": -1.088254, "B": 4.968863, "C_log": 0.250636, "W_log": 7.080722, "PHI_log": 2.270158, "w_log": 0.072023, "C_cal1": 0.619186, "W_cal1": 4.258707, "PHI_cal1": -2.511183, "w_cal1": 0.387739, "C_cal2": 0.245329, "W_cal2": 1.750423, "PHI_cal2": -2.136096}, "r2": 0.990475, "sigma": 0.149568},
    "ecfg_1d_2uu": {"n_log": 1, "n_cal": 2, "log_damps": ['d'], "cal_damps": ['u', 'u'], "params": {"A": -1.182030, "B": 5.097717, "C_log": 0.544373, "W_log": 7.724853, "PHI_log": 1.477158, "w_log": 0.106630, "C_cal1": 0.199457, "W_cal1": 1.881938, "PHI_cal1": 2.521008, "C_cal2": 0.112063, "W_cal2": 3.350333, "PHI_cal2": 3.113888}, "r2": 0.992147, "sigma": 0.135807},
    "ecfg_1u_0": {"n_log": 1, "n_cal": 0, "log_damps": ['u'], "cal_damps": [], "params": {"A": -1.212520, "B": 5.148940, "C_log": 0.229673, "W_log": 7.501284, "PHI_log": 1.463241}, "r2": 0.974742, "sigma": 0.243557},
    "ecfg_1u_1d": {"n_log": 1, "n_cal": 1, "log_damps": ['u'], "cal_damps": ['d'], "params": {"A": -1.194226, "B": 5.112388, "C_log": 0.174706, "W_log": 7.322228, "PHI_log": 1.641241, "C_cal": 0.296028, "W_cal": 1.739303, "PHI_cal": -2.022812, "w_cal": 0.044616}, "r2": 0.987359, "sigma": 0.172300},
    "ecfg_1u_1u": {"n_log": 1, "n_cal": 1, "log_damps": ['u'], "cal_damps": ['u'], "params": {"A": -1.249162, "B": 5.177557, "C_log": 0.182661, "W_log": 7.463541, "PHI_log": 1.526733, "C_cal": 0.240511, "W_cal": 1.762299, "PHI_cal": -2.276707}, "r2": 0.986615, "sigma": 0.177299},
    "ecfg_1u_2dd": {"n_log": 1, "n_cal": 2, "log_damps": ['u'], "cal_damps": ['d', 'd'], "params": {"A": -1.177301, "B": 5.085247, "C_log": 0.143932, "W_log": 7.088725, "PHI_log": 2.090359, "C_cal1": 0.303992, "W_cal1": 1.737076, "PHI_cal1": -2.004054, "w_cal1": 0.045656, "C_cal2": 0.213137, "W_cal2": 2.850995, "PHI_cal2": 0.089770, "w_cal2": 0.100316}, "r2": 0.990152, "sigma": 0.152083},
    "ecfg_1u_2du": {"n_log": 1, "n_cal": 2, "log_damps": ['u'], "cal_damps": ['d', 'u'], "params": {"A": -1.171489, "B": 5.084231, "C_log": 0.161429, "W_log": 7.169349, "PHI_log": 1.940487, "C_cal1": 0.305052, "W_cal1": 1.741425, "PHI_cal1": -2.038780, "w_cal1": 0.046564, "C_cal2": 0.106786, "W_cal2": 3.147191, "PHI_cal2": -1.025256}, "r2": 0.989671, "sigma": 0.155754},
    "ecfg_1u_2uu": {"n_log": 1, "n_cal": 2, "log_damps": ['u'], "cal_damps": ['u', 'u'], "params": {"A": -1.220917, "B": 5.142771, "C_log": 0.169761, "W_log": 7.310818, "PHI_log": 1.803148, "C_cal1": 0.247284, "W_cal1": 1.757691, "PHI_cal1": -2.213124, "C_cal2": 0.109246, "W_cal2": 3.135365, "PHI_cal2": -1.009632}, "r2": 0.989022, "sigma": 0.160569},
    "ecfg_2dd_0": {"n_log": 2, "n_cal": 0, "log_damps": ['d', 'd'], "cal_damps": [], "params": {"A": -1.100858, "B": 4.990648, "C_log1": 0.516321, "W_log1": 7.699786, "PHI_log1": 1.208440, "w_log1": 0.093798, "C_log2": 0.183801, "W_log2": 20.999593, "PHI_log2": -1.297342, "w_log2": 0.029031}, "r2": 0.988467, "sigma": 0.164582},
    "ecfg_2dd_1d": {"n_log": 2, "n_cal": 1, "log_damps": ['d', 'd'], "cal_damps": ['d'], "params": {"A": -1.136427, "B": 5.029571, "C_log1": 0.609911, "W_log1": 8.718596, "PHI_log1": -0.530151, "w_log1": 0.104234, "C_log2": 0.188144, "W_log2": 20.878965, "PHI_log2": -1.116640, "w_log2": 0.040534, "C_cal": 0.700885, "W_cal": 4.423915, "PHI_cal": -1.331334, "w_cal": 0.268566}, "r2": 0.990687, "sigma": 0.147893},
    "ecfg_2dd_1u": {"n_log": 2, "n_cal": 1, "log_damps": ['d', 'd'], "cal_damps": ['u'], "params": {"A": -1.174924, "B": 5.084216, "C_log1": 0.485148, "W_log1": 7.864106, "PHI_log1": 1.223782, "w_log1": 0.105625, "C_log2": 0.262910, "W_log2": 16.656447, "PHI_log2": 1.542029, "w_log2": 0.251354, "C_cal": 0.194908, "W_cal": 1.853415, "PHI_cal": 2.845581}, "r2": 0.991219, "sigma": 0.143609},
    "ecfg_2dd_2dd": {"n_log": 2, "n_cal": 2, "log_damps": ['d', 'd'], "cal_damps": ['d', 'd'], "params": {"A": -1.076600, "B": 4.955777, "C_log1": 0.265817, "W_log1": 7.350310, "PHI_log1": 1.698002, "w_log1": 0.073639, "C_log2": 0.113215, "W_log2": 37.205539, "PHI_log2": 1.825743, "w_log2": 0.050741, "C_cal1": 0.412053, "W_cal1": 4.603006, "PHI_cal1": -2.842755, "w_cal1": 0.357738, "C_cal2": 0.272462, "W_cal2": 1.737824, "PHI_cal2": -1.982436, "w_cal2": 0.038240}, "r2": 0.992324, "sigma": 0.134266},
    "ecfg_2dd_2du": {"n_log": 2, "n_cal": 2, "log_damps": ['d', 'd'], "cal_damps": ['d', 'u'], "params": {"A": -1.183134, "B": 5.093868, "C_log1": 0.110639, "W_log1": 21.541721, "PHI_log1": -2.524589, "w_log1": 0.065340, "C_log2": 0.495134, "W_log2": 7.764643, "PHI_log2": 1.344845, "w_log2": 0.106518, "C_cal1": 0.225239, "W_cal1": 10.000000, "PHI_cal1": -0.239643, "w_cal1": 0.168112, "C_cal2": 0.173772, "W_cal2": 1.856002, "PHI_cal2": 2.777918}, "r2": 0.992694, "sigma": 0.130990},
    "ecfg_2dd_2uu": {"n_log": 2, "n_cal": 2, "log_damps": ['d', 'd'], "cal_damps": ['u', 'u'], "params": {"A": -1.172266, "B": 5.087527, "C_log1": 0.554093, "W_log1": 7.801516, "PHI_log1": 1.362827, "w_log1": 0.106798, "C_log2": 0.251089, "W_log2": 16.819794, "PHI_log2": 1.461993, "w_log2": 0.251550, "C_cal1": 0.196766, "W_cal1": 1.879318, "PHI_cal1": 2.537884, "C_cal2": 0.109234, "W_cal2": 3.361624, "PHI_cal2": 2.988717}, "r2": 0.993512, "sigma": 0.123438},
    "ecfg_2du_0": {"n_log": 2, "n_cal": 0, "log_damps": ['d', 'u'], "cal_damps": [], "params": {"A": -1.105944, "B": 4.997912, "C_log1": 0.517500, "W_log1": 7.691650, "PHI_log1": 1.221941, "w_log1": 0.094324, "C_log2": 0.153587, "W_log2": 20.964083, "PHI_log2": -1.208050}, "r2": 0.988107, "sigma": 0.167126},
    "ecfg_2du_1d": {"n_log": 2, "n_cal": 1, "log_damps": ['d', 'u'], "cal_damps": ['d'], "params": {"A": -1.138158, "B": 5.051562, "C_log1": 0.489271, "W_log1": 7.764289, "PHI_log1": 1.077129, "w_log1": 0.103091, "C_log2": 0.060645, "W_log2": 6.896841, "PHI_log2": 2.582250, "C_cal": 0.222810, "W_cal": 10.000000, "PHI_cal": -0.422637, "w_cal": 0.172772}, "r2": 0.985642, "sigma": 0.183632},
    "ecfg_2du_1u": {"n_log": 2, "n_cal": 1, "log_damps": ['d', 'u'], "cal_damps": ['u'], "params": {"A": -1.170544, "B": 5.062589, "C_log1": 0.624761, "W_log1": 6.258790, "PHI_log1": 1.586219, "w_log1": 0.329648, "C_log2": 0.151273, "W_log2": 6.615522, "PHI_log2": 3.044371, "C_cal": 0.276052, "W_cal": 1.721047, "PHI_cal": -1.866803}, "r2": 0.989514, "sigma": 0.156931},
    "ecfg_2du_2dd": {"n_log": 2, "n_cal": 2, "log_damps": ['d', 'u'], "cal_damps": ['d', 'd'], "params": {"A": -1.132339, "B": 5.017840, "C_log1": 0.563840, "W_log1": 8.640349, "PHI_log1": -0.409832, "w_log1": 0.096772, "C_log2": 0.156312, "W_log2": 21.171626, "PHI_log2": -1.740206, "C_cal1": 0.644916, "W_cal1": 4.414646, "PHI_cal1": -1.362998, "w_cal1": 0.269557, "C_cal2": 0.180378, "W_cal2": 10.096036, "PHI_cal2": -0.368274, "w_cal2": 0.161812}, "r2": 0.991627, "sigma": 0.140227},
    "ecfg_2du_2du": {"n_log": 2, "n_cal": 2, "log_damps": ['d', 'u'], "cal_damps": ['d', 'u'], "params": {"A": -1.148610, "B": 5.060264, "C_log1": 0.527820, "W_log1": 7.738887, "PHI_log1": 1.406019, "w_log1": 0.107080, "C_log2": 0.064714, "W_log2": 37.148986, "PHI_log2": 1.967414, "C_cal1": 0.230749, "W_cal1": 1.861921, "PHI_cal1": 2.716473, "w_cal1": 0.031927, "C_cal2": 0.087508, "W_cal2": 3.368530, "PHI_cal2": 3.006750}, "r2": 0.992944, "sigma": 0.128731},
    "ecfg_2du_2uu": {"n_log": 2, "n_cal": 2, "log_damps": ['d', 'u'], "cal_damps": ['u', 'u'], "params": {"A": -1.179572, "B": 5.095088, "C_log1": 0.503443, "W_log1": 7.622851, "PHI_log1": 1.636436, "w_log1": 0.106736, "C_log2": 0.084000, "W_log2": 20.232999, "PHI_log2": -0.716489, "C_cal1": 0.200929, "W_cal1": 1.898552, "PHI_cal1": 2.624986, "C_cal2": 0.110923, "W_cal2": 3.309232, "PHI_cal2": -2.601723}, "r2": 0.993073, "sigma": 0.127547},
    "ecfg_2uu_0": {"n_log": 2, "n_cal": 0, "log_damps": ['u', 'u'], "cal_damps": [], "params": {"A": -1.178043, "B": 5.087724, "C_log1": 0.240808, "W_log1": 7.187865, "PHI_log1": 1.860414, "C_log2": 0.179903, "W_log2": 20.890309, "PHI_log2": -1.107652}, "r2": 0.981189, "sigma": 0.210188},
    "ecfg_2uu_1d": {"n_log": 2, "n_cal": 1, "log_damps": ['u', 'u'], "cal_damps": ['d'], "params": {"A": -1.177172, "B": 5.085507, "C_log1": 0.183375, "W_log1": 7.220524, "PHI_log1": 1.743807, "C_log2": 0.089807, "W_log2": 21.002644, "PHI_log2": -1.326174, "C_cal": 0.265616, "W_cal": 1.725361, "PHI_cal": -1.899099, "w_cal": 0.052660}, "r2": 0.988391, "sigma": 0.165117},
    "ecfg_2uu_1u": {"n_log": 2, "n_cal": 1, "log_damps": ['u', 'u'], "cal_damps": ['u'], "params": {"A": -1.238299, "B": 5.159757, "C_log1": 0.181313, "W_log1": 7.387103, "PHI_log1": 1.603227, "C_log2": 0.096148, "W_log2": 37.182623, "PHI_log2": 1.911055, "C_cal": 0.253293, "W_cal": 1.759701, "PHI_cal": -2.249329}, "r2": 0.988546, "sigma": 0.164011},
    "ecfg_2uu_2dd": {"n_log": 2, "n_cal": 2, "log_damps": ['u', 'u'], "cal_damps": ['d', 'd'], "params": {"A": -1.077617, "B": 4.960257, "C_log1": 0.099187, "W_log1": 21.018168, "PHI_log1": -1.320408, "C_log2": 0.148688, "W_log2": 6.434495, "PHI_log2": -2.975839, "C_cal1": 0.615623, "W_cal1": 4.207033, "PHI_cal1": -2.383437, "w_cal1": 0.339705, "C_cal2": 0.309043, "W_cal2": 1.714629, "PHI_cal2": -1.869871, "w_cal2": 0.058078}, "r2": 0.991025, "sigma": 0.145181},
    "ecfg_2uu_2du": {"n_log": 2, "n_cal": 2, "log_damps": ['u', 'u'], "cal_damps": ['d', 'u'], "params": {"A": -1.123095, "B": 5.010996, "C_log1": 0.209799, "W_log1": 8.369069, "PHI_log1": 1.382220, "C_log2": 0.286261, "W_log2": 7.068998, "PHI_log2": 1.652544, "C_cal1": 0.245342, "W_cal1": 1.777858, "PHI_cal1": -2.485781, "w_cal1": 0.042604, "C_cal2": 0.107829, "W_cal2": 3.313094, "PHI_cal2": -2.739747}, "r2": 0.992487, "sigma": 0.132836},
    "ecfg_2uu_2uu": {"n_log": 2, "n_cal": 2, "log_damps": ['u', 'u'], "cal_damps": ['u', 'u'], "params": {"A": -1.166922, "B": 5.059894, "C_log1": 0.209031, "W_log1": 8.410201, "PHI_log1": 1.317022, "C_log2": 0.281935, "W_log2": 7.096555, "PHI_log2": 1.659882, "C_cal1": 0.203831, "W_cal1": 1.793838, "PHI_cal1": -2.674157, "C_cal2": 0.108761, "W_cal2": 3.299630, "PHI_cal2": -2.639374}, "r2": 0.991857, "sigma": 0.138289},
}

class EntropyPPLModel(_ShrinkingBandsMixin):
    """Entropy PPL — HybPPL variant with Shannon entropy envelope damping.

    Replaces the t^(-D) power-law damping of HybPPL with a normalized
    Shannon entropy envelope E(w*t) = max(-w*t*ln(w*t), 0) / (1/e).

    The entropy envelope peaks when adoption uncertainty is maximal
    (w*t = 1/e) and decays to zero when adoption is "resolved" (w*t = 1).

    Formula (2+2 version, 16 params):
        log10(price) = A + B*log10(t)
            + C1*E(w1*t)*cos(W1*ln(t)+P1)     # entropy-damped log-periodic 1
            + C3*E(w2*t)*cos(W2*ln(t)+P3)     # entropy-damped log-periodic 2
            + C2*cos(Wc1*t+P2)                 # undamped halving cycle
            + C4*cos(Wc2*t+P4)                 # undamped sub-halving

    R²=0.993320, σ=0.125028
    """
    name = "Entropy PPL"
    short_name = "eppl"
    legend_name = "EPPL"
    dash_style = "dot"
    quantized = True

    # ── Fitted parameters (EPPL 2+2) ────────────────────────────────────
    _A    = -1.167364
    _B    =  5.079560
    _C1   =  0.250431    # log osc 1 amplitude
    _W1   = 16.823756    # log osc 1 frequency
    _P1   =  1.460422    # log osc 1 phase
    _w1   =  0.251550    # log osc 1 entropy rate
    _C3   =  0.556269    # log osc 2 amplitude
    _W2   =  7.803554    # log osc 2 frequency
    _P3   =  1.373041    # log osc 2 phase
    _w2   =  0.107049    # log osc 2 entropy rate
    _C2   =  0.202747    # cal osc 1 amplitude
    _Wc1  =  1.881312    # cal osc 1 frequency (T=3.34yr)
    _P2   =  2.520900    # cal osc 1 phase
    _C4   =  0.113542    # cal osc 2 amplitude
    _Wc2  =  3.355482    # cal osc 2 frequency (T=1.87yr)
    _P4   =  3.033230    # cal osc 2 phase
    _sigma0_up   = 0.094900
    _alpha_up    = 0.346400
    _sigma0_down = 0.094900
    _alpha_down  = 0.434700
    _sigma       = 0.125028  # backward compat

    def __init__(self, price_years, price_prices, quantiles):
        self.fits = {}
        for q in quantiles:
            self.fits[q] = {"z": float(_lazy_norm().ppf(q))}
        self.quantiles = sorted(self.fits.keys())
        self._build_colors()

    @staticmethod
    def entropy_env(t, w):
        """Normalized Shannon entropy envelope: E(x) = max(-x*ln(x), 0) / (1/e)."""
        x = w * t
        raw = -x * np.log(np.maximum(x, 1e-30))
        return np.maximum(raw, 0.0) / (1.0 / np.e)

    def _model_log10(self, t):
        """Evaluate the 2+2 entropy PPL formula."""
        t_arr = np.asarray(t, float)
        scalar = t_arr.ndim == 0
        if scalar:
            t_arr = t_arr.reshape(1)
        t_safe = np.maximum(t_arr, 0.1)

        result = self._A + self._B * np.log10(t_safe)
        # Entropy-damped log-periodic term 1
        result += self._C1 * self.entropy_env(t_safe, self._w1) * np.cos(
            self._W1 * np.log(t_safe) + self._P1)
        # Entropy-damped log-periodic term 2
        result += self._C3 * self.entropy_env(t_safe, self._w2) * np.cos(
            self._W2 * np.log(t_safe) + self._P3)
        # Undamped halving cycle
        result += self._C2 * np.cos(self._Wc1 * t_safe + self._P2)
        # Undamped sub-halving
        result += self._C4 * np.cos(self._Wc2 * t_safe + self._P4)

        return float(result[0]) if scalar else result

    # price_at, interp_price, find_percentile inherited from _ShrinkingBandsMixin

    # ── Decomposition ────────────────────────────────────────────────────

    component_names = [
        "A (constant)",
        "B\u00b7log\u2081\u2080(t)",
        "entropy log osc 1 (\u03c9\u2081)",
        "entropy log osc 2 (\u03c9\u2082)",
        "undamped cal osc 1 (\u03c9_c\u2081)",
        "undamped cal osc 2 (\u03c9_c\u2082)",
    ]

    formula_log10_latex = (
        r"A + B \log_{10}(t)"
        r" + C_1 \cdot E(w_1 t) \cos(\omega_1 \ln t + \varphi_1)"
        r" + C_3 \cdot E(w_2 t) \cos(\omega_2 \ln t + \varphi_3)"
        r" + C_2 \cos(\omega_{c1} t + \varphi_2)"
        r" + C_4 \cos(\omega_{c2} t + \varphi_4)"
    )
    formula_product_latex = None  # too complex for product form

    @property
    def component_details(self):
        return {
            "A (constant)": (
                "A",
                [("A", "_A")],
            ),
            "B\u00b7log\u2081\u2080(t)": (
                "B\u00b7log\u2081\u2080(t)",
                [("B", "_B")],
            ),
            "entropy log osc 1 (\u03c9\u2081)": (
                "C\u2081\u00b7E(w\u2081\u00b7t)\u00b7cos(\u03c9\u2081\u00b7ln(t)+\u03c6\u2081)",
                [("C\u2081", "_C1"), ("\u03c9\u2081", "_W1"),
                 ("\u03c6\u2081", "_P1"), ("w\u2081", "_w1")],
            ),
            "entropy log osc 2 (\u03c9\u2082)": (
                "C\u2083\u00b7E(w\u2082\u00b7t)\u00b7cos(\u03c9\u2082\u00b7ln(t)+\u03c6\u2083)",
                [("C\u2083", "_C3"), ("\u03c9\u2082", "_W2"),
                 ("\u03c6\u2083", "_P3"), ("w\u2082", "_w2")],
            ),
            "undamped cal osc 1 (\u03c9_c\u2081)": (
                "C\u2082\u00b7cos(\u03c9_c\u2081\u00b7t+\u03c6\u2082)",
                [("C\u2082", "_C2"), ("\u03c9_c\u2081", "_Wc1"),
                 ("\u03c6\u2082", "_P2")],
            ),
            "undamped cal osc 2 (\u03c9_c\u2082)": (
                "C\u2084\u00b7cos(\u03c9_c\u2082\u00b7t+\u03c6\u2084)",
                [("C\u2084", "_C4"), ("\u03c9_c\u2082", "_Wc2"),
                 ("\u03c6\u2084", "_P4")],
            ),
        }

    def components(self, t):
        """Decompose into constant + trend + 4 oscillatory terms."""
        t_arr = np.asarray(t, float)
        scalar = t_arr.ndim == 0
        if scalar:
            t_arr = t_arr.reshape(1)
        t_safe = np.maximum(t_arr, 0.1)

        result = {
            "A (constant)":                    np.full_like(t_safe, self._A),
            "B\u00b7log\u2081\u2080(t)":        self._B * np.log10(t_safe),
            "entropy log osc 1 (\u03c9\u2081)": self._C1 * self.entropy_env(t_safe, self._w1) * np.cos(
                self._W1 * np.log(t_safe) + self._P1),
            "entropy log osc 2 (\u03c9\u2082)": self._C3 * self.entropy_env(t_safe, self._w2) * np.cos(
                self._W2 * np.log(t_safe) + self._P3),
            "undamped cal osc 1 (\u03c9_c\u2081)": self._C2 * np.cos(
                self._Wc1 * t_safe + self._P2),
            "undamped cal osc 2 (\u03c9_c\u2082)": self._C4 * np.cos(
                self._Wc2 * t_safe + self._P4),
        }
        if scalar:
            result = {k: float(v[0]) for k, v in result.items()}
        return result

    def _build_colors(self):
        """Warm amber/orange palette — entropy PPL model."""
        self.colors = {}
        n = len(self.quantiles)
        for i, q in enumerate(self.quantiles):
            frac = i / max(n - 1, 1)
            r = int(180 + 40 * frac)     # 180 → 220
            g = int(120 + 50 * frac)     # 120 → 170
            b = int(30 + 40 * frac)      # 30 → 70
            self.colors[q] = f"#{r:02x}{g:02x}{b:02x}"


class EPPLConfigModel(_ShrinkingBandsMixin):
    """Generic EPPL config model -- loads pre-fitted params for any config.

    Config key format: ecfg_{log_spec}_{cal_spec}
    where spec = "0" or "{count}{damps}" e.g. "2du" = 2 freqs, first damped,
    second undamped.

    Model: log10(price) = A + B*log10(t) + sum(log_osc_i) + sum(cal_osc_i)
    where:
      entropy-damped log: C * E(w*t) * cos(W * ln(t) + PHI)
      undamped log:       C * cos(W * ln(t) + PHI)
      entropy-damped cal: C * E(w*t) * cos(W * t + PHI)
      undamped cal:       C * cos(W * t + PHI)
    with E(x) = max(-x*ln(x), 0) / (1/e)   (normalized Shannon entropy envelope)
    """
    quantized = True

    @staticmethod
    def entropy_env(t, w):
        """Normalized Shannon entropy envelope: E(x) = max(-x*ln(x), 0) / (1/e)."""
        x = w * t
        raw = -x * np.log(np.maximum(x, 1e-30))
        return np.maximum(raw, 0.0) / (1.0 / np.e)

    def __init__(self, config_key, price_years, price_prices, quantiles,
                 *, cfg_override=None, sigma_override=None):
        if cfg_override is not None:
            cfg = copy.deepcopy(cfg_override)
        else:
            cfg = _EPPL_CONFIG_PARAMS.get(config_key)
            if cfg is None:
                raise ValueError(f"Unknown EPPL config: {config_key}")
            cfg = copy.deepcopy(cfg)
        self._config_key = config_key
        self._cfg = cfg
        self._params = cfg["params"]
        self._sigma = cfg.get("sigma")
        self._n_log = cfg["n_log"]
        self._n_cal = cfg["n_cal"]
        self._log_damps = cfg["log_damps"]
        self._cal_damps = cfg["cal_damps"]
        self.r2 = cfg["r2"]

        # Readable names
        self.name = config_key
        self.short_name = config_key
        spec = config_key.replace("ecfg_", "")
        self.legend_name = spec.upper()
        self.dash_style = "dot"

        if sigma_override is not None:
            # Per-request override: constant sigma, skip residual-based band fit.
            self._sigma = sigma_override
            self.fits = {q: {"z": float(_lazy_norm().ppf(q))} for q in quantiles}
            self.quantiles = sorted(self.fits.keys())
        else:
            # Build shrinking quantile bands from residuals
            mask = price_years >= T_MIN
            t_fit = price_years[mask]
            lp_fit = np.log10(price_prices[mask])
            residuals = lp_fit - self._model_log10(t_fit)
            self._init_shrinking_bands(t_fit, residuals, quantiles)
        self._build_colors()

    def _model_log10(self, t):
        """Evaluate the model at time t using stored params."""
        t = np.asarray(t, float)
        ts = np.maximum(t, 0.1)
        p = self._params
        result = p["A"] + p["B"] * np.log10(ts)

        # Log-periodic terms
        for i in range(self._n_log):
            suffix = str(i + 1) if self._n_log > 1 else ""
            C = p[f"C_log{suffix}"]
            W = p[f"W_log{suffix}"]
            PHI = p[f"PHI_log{suffix}"]
            if self._log_damps[i] == "d":
                w = p[f"w_log{suffix}"]
                result = result + C * self.entropy_env(ts, w) * np.cos(W * np.log(ts) + PHI)
            else:
                result = result + C * np.cos(W * np.log(ts) + PHI)

        # Calendar terms
        for i in range(self._n_cal):
            suffix = str(i + 1) if self._n_cal > 1 else ""
            C = p[f"C_cal{suffix}"]
            W = p[f"W_cal{suffix}"]
            PHI = p[f"PHI_cal{suffix}"]
            if self._cal_damps[i] == "d":
                w = p[f"w_cal{suffix}"]
                result = result + C * self.entropy_env(ts, w) * np.cos(W * ts + PHI)
            else:
                result = result + C * np.cos(W * ts + PHI)

        return result

    # price_at, interp_price, find_percentile inherited from _ShrinkingBandsMixin

    @property
    def component_names(self):
        names = ["A (constant)", "B\u00b7log\u2081\u2080(t)"]
        for i in range(self._n_log):
            d = self._log_damps[i]
            names.append(f"log osc {i+1} ({'entropy damped' if d == 'd' else 'undamped'})")
        for i in range(self._n_cal):
            d = self._cal_damps[i]
            names.append(f"cal osc {i+1} ({'entropy damped' if d == 'd' else 'undamped'})")
        return names

    @property
    def formula_log10_latex(self):
        parts = [r"A + B \log_{10}(t)"]
        for i in range(self._n_log):
            d = self._log_damps[i]
            idx = i + 1
            if d == "d":
                parts.append(rf"C_{{l{idx}}} E(w_{{l{idx}}} t) \cos(\omega_{{l{idx}}} \ln t + \varphi_{{l{idx}}})")
            else:
                parts.append(rf"C_{{l{idx}}} \cos(\omega_{{l{idx}}} \ln t + \varphi_{{l{idx}}})")
        for i in range(self._n_cal):
            d = self._cal_damps[i]
            idx = i + 1
            if d == "d":
                parts.append(rf"C_{{c{idx}}} E(w_{{c{idx}}} t) \cos(\omega_{{c{idx}}} t + \varphi_{{c{idx}}})")
            else:
                parts.append(rf"C_{{c{idx}}} \cos(\omega_{{c{idx}}} t + \varphi_{{c{idx}}})")
        return " + ".join(parts)

    @property
    def formula_product_latex(self):
        return None  # too complex for product form

    @property
    def component_details(self):
        det = {
            "A (constant)": ("A", [("A", "A")]),
            "B\u00b7log\u2081\u2080(t)": ("B\u00b7log\u2081\u2080(t)", [("B", "B")]),
        }
        for i in range(self._n_log):
            d = self._log_damps[i]
            name = f"log osc {i+1} ({'entropy damped' if d == 'd' else 'undamped'})"
            if d == "d":
                det[name] = (
                    f"C\u00b7E(w\u00b7t)\u00b7cos(\u03c9\u00b7ln(t)+\u03c6)",
                    [],
                )
            else:
                det[name] = ("C\u00b7cos(\u03c9\u00b7ln(t)+\u03c6)", [])
        for i in range(self._n_cal):
            d = self._cal_damps[i]
            name = f"cal osc {i+1} ({'entropy damped' if d == 'd' else 'undamped'})"
            if d == "d":
                det[name] = (
                    f"C\u00b7E(w\u00b7t)\u00b7cos(\u03c9\u00b7t+\u03c6)",
                    [],
                )
            else:
                det[name] = ("C\u00b7cos(\u03c9\u00b7t+\u03c6)", [])
        return det

    def components(self, t):
        """Decompose into individual additive terms."""
        t = np.asarray(t, float)
        ts = np.maximum(t, 0.1)
        p = self._params
        result = {
            "A (constant)": np.full_like(ts, p["A"]),
            "B\u00b7log\u2081\u2080(t)": p["B"] * np.log10(ts),
        }
        for i in range(self._n_log):
            suffix = str(i + 1) if self._n_log > 1 else ""
            d = self._log_damps[i]
            C = p[f"C_log{suffix}"]; W = p[f"W_log{suffix}"]; PHI = p[f"PHI_log{suffix}"]
            name = f"log osc {i+1} ({'entropy damped' if d == 'd' else 'undamped'})"
            if d == "d":
                w = p[f"w_log{suffix}"]
                result[name] = C * self.entropy_env(ts, w) * np.cos(W * np.log(ts) + PHI)
            else:
                result[name] = C * np.cos(W * np.log(ts) + PHI)
        for i in range(self._n_cal):
            suffix = str(i + 1) if self._n_cal > 1 else ""
            d = self._cal_damps[i]
            C = p[f"C_cal{suffix}"]; W = p[f"W_cal{suffix}"]; PHI = p[f"PHI_cal{suffix}"]
            name = f"cal osc {i+1} ({'entropy damped' if d == 'd' else 'undamped'})"
            if d == "d":
                w = p[f"w_cal{suffix}"]
                result[name] = C * self.entropy_env(ts, w) * np.cos(W * ts + PHI)
            else:
                result[name] = C * np.cos(W * ts + PHI)
        return result

    def _build_colors(self):
        """Teal-cyan palette -- distinct from HybPPL's gray-blue."""
        self.colors = {}
        n = len(self.quantiles)
        for i, q in enumerate(self.quantiles):
            frac = i / max(n - 1, 1)
            r = int(20 + 60 * frac)      # 20 -> 80
            g = int(140 + 50 * frac)     # 140 -> 190
            b = int(140 + 40 * frac)     # 140 -> 180
            self.colors[q] = f"#{r:02x}{g:02x}{b:02x}"


# ── HybPPL config params (auto-generated) ──
_HYBPPL_CONFIG_PARAMS = {
    "cfg_0_0": {"n_log": 0, "n_cal": 0, "log_damps": [], "cal_damps": [], "params": {"A": -1.154625, "B": 5.050620}, "r2": 0.963450, "sigma": 0.292985},
    "cfg_0_1d": {"n_log": 0, "n_cal": 1, "log_damps": [], "cal_damps": ['d'], "params": {"A": -1.213360, "B": 5.111328, "C_cal": 0.277600, "W_cal": 1.766372, "PHI_cal": -2.287640, "D_cal": 0.002936}, "r2": 0.979725, "sigma": 0.218216},
    "cfg_0_1u": {"n_log": 0, "n_cal": 1, "log_damps": [], "cal_damps": ['u'], "params": {"A": -1.212876, "B": 5.110816, "C_cal": 0.275969, "W_cal": 1.766224, "PHI_cal": -2.286279}, "r2": 0.979724, "sigma": 0.218217},
    "cfg_0_2dd": {"n_log": 0, "n_cal": 2, "log_damps": [], "cal_damps": ['d', 'd'], "params": {"A": -1.043960, "B": 4.937381, "C_cal1": 0.740230, "W_cal1": 1.720698, "PHI_cal1": -1.805348, "D_cal1": 0.434908, "C_cal2": 1.361779, "W_cal2": 3.072793, "PHI_cal2": -0.635548, "D_cal2": 1.549961}, "r2": 0.987159, "sigma": 0.173659},
    "cfg_0_2du": {"n_log": 0, "n_cal": 2, "log_damps": [], "cal_damps": ['d', 'u'], "params": {"A": -1.089972, "B": 4.980926, "C_cal1": 0.740811, "W_cal1": 3.053998, "PHI_cal1": -0.368848, "D_cal1": 1.046829, "C_cal2": 0.298177, "W_cal2": 1.730645, "PHI_cal2": -1.906732}, "r2": 0.986107, "sigma": 0.180632},
    "cfg_0_2uu": {"n_log": 0, "n_cal": 2, "log_damps": [], "cal_damps": ['u', 'u'], "params": {"A": -1.328513, "B": 5.242820, "C_cal1": 0.152629, "W_cal1": 0.811031, "PHI_cal1": -2.732733, "C_cal2": 0.293660, "W_cal2": 1.781992, "PHI_cal2": -2.418297}, "r2": 0.983826, "sigma": 0.194901},
    "cfg_1d_0": {"n_log": 1, "n_cal": 0, "log_damps": ['d'], "cal_damps": [], "params": {"A": -1.139207, "B": 5.055828, "C_log": 0.729416, "W_log": 7.486114, "PHI_log": 1.450450, "D_log": 0.597466}, "r2": 0.978649, "sigma": 0.223928},
    "cfg_1d_1d": {"n_log": 1, "n_cal": 1, "log_damps": ['d'], "cal_damps": ['d'], "params": {"A": -1.136737, "B": 5.052869, "C_log": 0.863338, "W_log": 7.538416, "PHI_log": 1.380642, "D_log": 0.679656, "C_cal": 0.533028, "W_cal": 9.774442, "PHI_cal": 0.069225, "D_cal": 1.261622}, "r2": 0.980762, "sigma": 0.212563},
    "cfg_1d_1u": {"n_log": 1, "n_cal": 1, "log_damps": ['d'], "cal_damps": ['u'], "params": {"A": -1.150969, "B": 5.057785, "C_log": 0.691402, "W_log": 7.434710, "PHI_log": 1.443941, "D_log": 0.712062, "C_cal": 0.230793, "W_cal": 1.733428, "PHI_cal": -1.933169}, "r2": 0.989248, "sigma": 0.158909},
    "cfg_1d_2dd": {"n_log": 1, "n_cal": 2, "log_damps": ['d'], "cal_damps": ['d', 'd'], "params": {"A": -1.135308, "B": 5.044952, "C_log": 0.804138, "W_log": 7.357560, "PHI_log": 1.727691, "D_log": 0.778536, "C_cal1": 0.220708, "W_cal1": 3.285534, "PHI_cal1": -2.487909, "D_cal1": 0.347704, "C_cal2": 0.230094, "W_cal2": 1.748172, "PHI_cal2": -2.109057, "D_cal2": 0.000000}, "r2": 0.991455, "sigma": 0.141667},
    "cfg_1d_2du": {"n_log": 1, "n_cal": 2, "log_damps": ['d'], "cal_damps": ['d', 'u'], "params": {"A": -1.147801, "B": 5.058464, "C_log": 0.765491, "W_log": 7.413681, "PHI_log": 1.594623, "D_log": 0.758388, "C_cal1": 0.226821, "W_cal1": 1.744751, "PHI_cal1": -2.063417, "D_cal1": 0.000000, "C_cal2": 0.104482, "W_cal2": 3.299584, "PHI_cal2": -2.603517}, "r2": 0.991350, "sigma": 0.142530},
    "cfg_1d_2uu": {"n_log": 1, "n_cal": 2, "log_damps": ['d'], "cal_damps": ['u', 'u'], "params": {"A": -1.147801, "B": 5.058464, "C_log": 0.765491, "W_log": 7.413681, "PHI_log": 1.594622, "D_log": 0.758389, "C_cal1": 0.226821, "W_cal1": 1.744751, "PHI_cal1": -2.063415, "C_cal2": 0.104482, "W_cal2": 3.299584, "PHI_cal2": -2.603519}, "r2": 0.991350, "sigma": 0.142530},
    "cfg_1u_0": {"n_log": 1, "n_cal": 0, "log_damps": ['u'], "cal_damps": [], "params": {"A": -1.212520, "B": 5.148940, "C_log": 0.229673, "W_log": 7.501285, "PHI_log": 1.463240}, "r2": 0.974742, "sigma": 0.243557},
    "cfg_1u_1d": {"n_log": 1, "n_cal": 1, "log_damps": ['u'], "cal_damps": ['d'], "params": {"A": -1.249162, "B": 5.177557, "C_log": 0.182661, "W_log": 7.463542, "PHI_log": 1.526733, "C_cal": 0.240511, "W_cal": 1.762299, "PHI_cal": -2.276707, "D_cal": 0.000000}, "r2": 0.986615, "sigma": 0.177299},
    "cfg_1u_1u": {"n_log": 1, "n_cal": 1, "log_damps": ['u'], "cal_damps": ['u'], "params": {"A": -1.249162, "B": 5.177557, "C_log": 0.182661, "W_log": 7.463541, "PHI_log": 1.526733, "C_cal": 0.240511, "W_cal": 1.762299, "PHI_cal": -2.276706}, "r2": 0.986615, "sigma": 0.177299},
    "cfg_1u_2dd": {"n_log": 1, "n_cal": 2, "log_damps": ['u'], "cal_damps": ['d', 'd'], "params": {"A": -0.993312, "B": 4.853066, "C_log": 0.157245, "W_log": 5.105284, "PHI_log": 0.092157, "C_cal1": 1.000394, "W_cal1": 1.726319, "PHI_cal1": -1.958132, "D_cal1": 0.575183, "C_cal2": 1.935012, "W_cal2": 3.060738, "PHI_cal2": -0.626159, "D_cal2": 1.814822}, "r2": 0.990505, "sigma": 0.149331},
    "cfg_1u_2du": {"n_log": 1, "n_cal": 2, "log_damps": ['u'], "cal_damps": ['d', 'u'], "params": {"A": -1.115765, "B": 5.015978, "C_log": 0.148506, "W_log": 6.828637, "PHI_log": 2.617504, "C_cal1": 0.559436, "W_cal1": 3.093573, "PHI_cal1": -0.769452, "D_cal1": 0.881258, "C_cal2": 0.267786, "W_cal2": 1.729813, "PHI_cal2": -1.954865}, "r2": 0.990078, "sigma": 0.152654},
    "cfg_1u_2uu": {"n_log": 1, "n_cal": 2, "log_damps": ['u'], "cal_damps": ['u', 'u'], "params": {"A": -1.220917, "B": 5.142771, "C_log": 0.169761, "W_log": 7.310817, "PHI_log": 1.803150, "C_cal1": 0.247284, "W_cal1": 1.757691, "PHI_cal1": -2.213123, "C_cal2": 0.109246, "W_cal2": 3.135366, "PHI_cal2": -1.009633}, "r2": 0.989022, "sigma": 0.160569},
    "cfg_2dd_0": {"n_log": 2, "n_cal": 0, "log_damps": ['d', 'd'], "cal_damps": [], "params": {"A": -1.106604, "B": 4.984542, "C_log1": 0.180054, "W_log1": 9.655232, "PHI_log1": -1.473618, "D_log1": 0.010000, "C_log2": 0.553186, "W_log2": 7.005817, "PHI_log2": 2.031685, "D_log2": 0.329642}, "r2": 0.983005, "sigma": 0.199786},
    "cfg_2dd_1d": {"n_log": 2, "n_cal": 1, "log_damps": ['d', 'd'], "cal_damps": ['d'], "params": {"A": -1.121326, "B": 5.026795, "C_log1": 0.771853, "W_log1": 7.507614, "PHI_log1": 1.272938, "D_log1": 0.784386, "C_log2": 0.418117, "W_log2": 15.986762, "PHI_log2": 1.897151, "D_log2": 1.037488, "C_cal": 0.250158, "W_cal": 1.720558, "PHI_cal": -1.763149, "D_cal": 0.000000}, "r2": 0.991083, "sigma": 0.144713},
    "cfg_2dd_1u": {"n_log": 2, "n_cal": 1, "log_damps": ['d', 'd'], "cal_damps": ['u'], "params": {"A": -1.121326, "B": 5.026795, "C_log1": 0.771853, "W_log1": 7.507615, "PHI_log1": 1.272938, "D_log1": 0.784387, "C_log2": 0.418118, "W_log2": 15.986765, "PHI_log2": 1.897149, "D_log2": 1.037489, "C_cal": 0.250158, "W_cal": 1.720558, "PHI_cal": -1.763150}, "r2": 0.991083, "sigma": 0.144713},
    "cfg_2dd_2dd": {"n_log": 2, "n_cal": 2, "log_damps": ['d', 'd'], "cal_damps": ['d', 'd'], "params": {"A": -1.111580, "B": 5.013376, "C_log1": 0.861391, "W_log1": 7.558978, "PHI_log1": 1.186249, "D_log1": 0.846309, "C_log2": 0.086320, "W_log2": 36.984800, "PHI_log2": 2.304401, "D_log2": 0.010000, "C_cal1": 0.247075, "W_cal1": 1.724814, "PHI_cal1": -1.814700, "D_cal1": 0.000000, "C_cal2": 0.526006, "W_cal2": 10.311622, "PHI_cal2": -1.306639, "D_cal2": 1.323328}, "r2": 0.992963, "sigma": 0.128560},
    "cfg_2dd_2du": {"n_log": 2, "n_cal": 2, "log_damps": ['d', 'd'], "cal_damps": ['d', 'u'], "params": {"A": -1.118448, "B": 5.019492, "C_log1": 0.892914, "W_log1": 7.506087, "PHI_log1": 1.281188, "D_log1": 0.865944, "C_log2": 0.091640, "W_log2": 36.912332, "PHI_log2": 2.493280, "D_log2": 0.010000, "C_cal1": 0.523655, "W_cal1": 9.830941, "PHI_cal1": -0.349797, "D_cal1": 1.352447, "C_cal2": 0.244896, "W_cal2": 1.725206, "PHI_cal2": -1.822174}, "r2": 0.992827, "sigma": 0.129792},
    "cfg_2dd_2uu": {"n_log": 2, "n_cal": 2, "log_damps": ['d', 'd'], "cal_damps": ['u', 'u'], "params": {"A": -1.127722, "B": 5.038828, "C_log1": 0.921550, "W_log1": 7.548775, "PHI_log1": 1.309747, "D_log1": 0.855091, "C_log2": 0.465981, "W_log2": 16.302768, "PHI_log2": 1.857379, "D_log2": 1.283872, "C_cal1": 0.231104, "W_cal1": 1.733293, "PHI_cal1": -1.895978, "C_cal2": 0.099464, "W_cal2": 3.385528, "PHI_cal2": 2.791233}, "r2": 0.992771, "sigma": 0.130300},
    "cfg_2du_0": {"n_log": 2, "n_cal": 0, "log_damps": ['d', 'u'], "cal_damps": [], "params": {"A": -1.122898, "B": 5.025181, "C_log1": 0.698418, "W_log1": 7.342309, "PHI_log1": 1.621428, "D_log1": 0.551562, "C_log2": 0.170191, "W_log2": 20.962129, "PHI_log2": -1.253205}, "r2": 0.984521, "sigma": 0.190667},
    "cfg_2du_1d": {"n_log": 2, "n_cal": 1, "log_damps": ['d', 'u'], "cal_damps": ['d'], "params": {"A": -1.162246, "B": 5.070765, "C_log1": 0.668980, "W_log1": 7.436934, "PHI_log1": 1.453970, "D_log1": 0.679908, "C_log2": 0.087251, "W_log2": 20.200946, "PHI_log2": -0.279983, "C_cal": 0.211995, "W_cal": 1.773391, "PHI_cal": -2.130658, "D_cal": 0.000000}, "r2": 0.990252, "sigma": 0.151310},
    "cfg_2du_1u": {"n_log": 2, "n_cal": 1, "log_damps": ['d', 'u'], "cal_damps": ['u'], "params": {"A": -1.143265, "B": 5.045936, "C_log1": 0.693101, "W_log1": 7.413626, "PHI_log1": 1.448385, "D_log1": 0.717637, "C_log2": 0.093461, "W_log2": 37.105122, "PHI_log2": 2.022251, "C_cal": 0.243518, "W_cal": 1.733693, "PHI_cal": -1.926151}, "r2": 0.991092, "sigma": 0.144643},
    "cfg_2du_2dd": {"n_log": 2, "n_cal": 2, "log_damps": ['d', 'u'], "cal_damps": ['d', 'd'], "params": {"A": -1.111571, "B": 5.013337, "C_log1": 0.861315, "W_log1": 7.558662, "PHI_log1": 1.186571, "D_log1": 0.846283, "C_log2": 0.084636, "W_log2": 36.985829, "PHI_log2": 2.302447, "C_cal1": 0.247164, "W_cal1": 1.724843, "PHI_cal1": -1.814874, "D_cal1": 0.000000, "C_cal2": 0.526021, "W_cal2": 10.311695, "PHI_cal2": -1.306286, "D_cal2": 1.322600}, "r2": 0.992964, "sigma": 0.128546},
    "cfg_2du_2du": {"n_log": 2, "n_cal": 2, "log_damps": ['d', 'u'], "cal_damps": ['d', 'u'], "params": {"A": -1.118438, "B": 5.019452, "C_log1": 0.892857, "W_log1": 7.505781, "PHI_log1": 1.281504, "D_log1": 0.865938, "C_log2": 0.089812, "W_log2": 36.912855, "PHI_log2": 2.492361, "C_cal1": 0.523820, "W_cal1": 9.831183, "PHI_cal1": -0.349552, "D_cal1": 1.352650, "C_cal2": 0.244987, "W_cal2": 1.725232, "PHI_cal2": -1.822337}, "r2": 0.992827, "sigma": 0.129789},
    "cfg_2du_2uu": {"n_log": 2, "n_cal": 2, "log_damps": ['d', 'u'], "cal_damps": ['u', 'u'], "params": {"A": -1.137121, "B": 5.043341, "C_log1": 0.791321, "W_log1": 7.395566, "PHI_log1": 1.558095, "D_log1": 0.780911, "C_log2": 0.075034, "W_log2": 36.870858, "PHI_log2": 2.271156, "C_cal1": 0.235874, "W_cal1": 1.732325, "PHI_cal1": -1.949909, "C_cal2": 0.087758, "W_cal2": 3.357777, "PHI_cal2": -2.921746}, "r2": 0.992385, "sigma": 0.133731},
    "cfg_2uu_0": {"n_log": 2, "n_cal": 0, "log_damps": ['u', 'u'], "cal_damps": [], "params": {"A": -1.106945, "B": 4.975243, "C_log1": 0.308456, "W_log1": 6.836352, "PHI_log1": 2.188467, "C_log2": 0.221339, "W_log2": 9.090116, "PHI_log2": -0.405799}, "r2": 0.982589, "sigma": 0.202213},
    "cfg_2uu_1d": {"n_log": 2, "n_cal": 1, "log_damps": ['u', 'u'], "cal_damps": ['d'], "params": {"A": -1.179692, "B": 5.072625, "C_log1": 0.254603, "W_log1": 7.162909, "PHI_log1": 1.578918, "C_log2": 0.170032, "W_log2": 8.636794, "PHI_log2": 0.835524, "C_cal": 0.206227, "W_cal": 1.777835, "PHI_cal": -2.503620, "D_cal": 0.000000}, "r2": 0.989624, "sigma": 0.156106},
    "cfg_2uu_1u": {"n_log": 2, "n_cal": 1, "log_damps": ['u', 'u'], "cal_damps": ['u'], "params": {"A": -1.179692, "B": 5.072625, "C_log1": 0.254603, "W_log1": 7.162907, "PHI_log1": 1.578921, "C_log2": 0.170031, "W_log2": 8.636795, "PHI_log2": 0.835520, "C_cal": 0.206227, "W_cal": 1.777835, "PHI_cal": -2.503620}, "r2": 0.989624, "sigma": 0.156106},
    "cfg_2uu_2dd": {"n_log": 2, "n_cal": 2, "log_damps": ['u', 'u'], "cal_damps": ['d', 'd'], "params": {"A": -1.459656, "B": 5.520922, "C_log1": 0.196335, "W_log1": 7.252482, "PHI_log1": 2.054862, "C_log2": 0.208285, "W_log2": 2.149814, "PHI_log2": -1.792333, "C_cal1": 0.253145, "W_cal1": 3.179520, "PHI_cal1": -1.375710, "D_cal1": 0.444922, "C_cal2": 0.377884, "W_cal2": 1.769810, "PHI_cal2": -2.293122, "D_cal2": 0.178734}, "r2": 0.991331, "sigma": 0.142690},
    "cfg_2uu_2du": {"n_log": 2, "n_cal": 2, "log_damps": ['u', 'u'], "cal_damps": ['d', 'u'], "params": {"A": -1.094072, "B": 4.985844, "C_log1": 0.149414, "W_log1": 6.696999, "PHI_log1": 2.850724, "C_log2": 0.103360, "W_log2": 37.277505, "PHI_log2": 1.847559, "C_cal1": 0.653372, "W_cal1": 2.995130, "PHI_cal1": -0.564640, "D_cal1": 1.039087, "C_cal2": 0.283157, "W_cal2": 1.732354, "PHI_cal2": -1.965992}, "r2": 0.991853, "sigma": 0.138328},
    "cfg_2uu_2uu": {"n_log": 2, "n_cal": 2, "log_damps": ['u', 'u'], "cal_damps": ['u', 'u'], "params": {"A": -1.166923, "B": 5.059895, "C_log1": 0.209031, "W_log1": 8.410201, "PHI_log1": 1.317022, "C_log2": 0.281935, "W_log2": 7.096555, "PHI_log2": 1.659882, "C_cal1": 0.203831, "W_cal1": 1.793837, "PHI_cal1": -2.674156, "C_cal2": 0.108761, "W_cal2": 3.299630, "PHI_cal2": -2.639371}, "r2": 0.991857, "sigma": 0.138289},
}

class HybPPLConfigModel(_ShrinkingBandsMixin):
    """Generic HybPPL config model -- loads pre-fitted params for any config.

    Config key format: cfg_{log_spec}_{cal_spec}
    where spec = "0" or "{count}{damps}" e.g. "2du" = 2 freqs, first damped,
    second undamped.

    Model: log10(price) = A + B*log10(t) + sum(log_osc_i) + sum(cal_osc_i)
    where:
      damped log:   C * t^(-D) * cos(W * ln(t) + PHI)
      undamped log: C * cos(W * ln(t) + PHI)
      damped cal:   C * t^(-D) * cos(W * t + PHI)
      undamped cal: C * cos(W * t + PHI)
    """
    quantized = True

    def __init__(self, config_key, price_years, price_prices, quantiles):
        cfg = _HYBPPL_CONFIG_PARAMS.get(config_key)
        if cfg is None:
            raise ValueError(f"Unknown HybPPL config: {config_key}")
        self._config_key = config_key
        self._cfg = cfg
        self._params = cfg["params"]
        self._sigma = cfg["sigma"]
        self._n_log = cfg["n_log"]
        self._n_cal = cfg["n_cal"]
        self._log_damps = cfg["log_damps"]
        self._cal_damps = cfg["cal_damps"]
        self.r2 = cfg["r2"]

        # Readable names
        self.name = config_key
        self.short_name = config_key
        spec = config_key.replace("cfg_", "")
        self.legend_name = spec.upper()
        self.dash_style = "solid"

        # Build shrinking quantile bands from residuals
        mask = price_years >= T_MIN
        t_fit = price_years[mask]
        lp_fit = np.log10(price_prices[mask])
        residuals = lp_fit - self._model_log10(t_fit)
        self._init_shrinking_bands(t_fit, residuals, quantiles)
        self._build_colors()

    def _model_log10(self, t):
        """Evaluate the model at time t using stored params."""
        t = np.asarray(t, float)
        ts = np.maximum(t, 0.1)
        p = self._params
        result = p["A"] + p["B"] * np.log10(ts)

        # Log-periodic terms
        for i in range(self._n_log):
            suffix = str(i + 1) if self._n_log > 1 else ""
            C = p[f"C_log{suffix}"]
            W = p[f"W_log{suffix}"]
            PHI = p[f"PHI_log{suffix}"]
            if self._log_damps[i] == "d":
                D = p[f"D_log{suffix}"]
                result = result + C * ts**(-D) * np.cos(W * np.log(ts) + PHI)
            else:
                result = result + C * np.cos(W * np.log(ts) + PHI)

        # Calendar terms
        for i in range(self._n_cal):
            suffix = str(i + 1) if self._n_cal > 1 else ""
            C = p[f"C_cal{suffix}"]
            W = p[f"W_cal{suffix}"]
            PHI = p[f"PHI_cal{suffix}"]
            if self._cal_damps[i] == "d":
                D = p[f"D_cal{suffix}"]
                result = result + C * ts**(-D) * np.cos(W * ts + PHI)
            else:
                result = result + C * np.cos(W * ts + PHI)

        return result

    # price_at, interp_price, find_percentile inherited from _ShrinkingBandsMixin

    @property
    def component_names(self):
        names = ["A (constant)", "B\u00b7log\u2081\u2080(t)"]
        for i in range(self._n_log):
            d = self._log_damps[i]
            names.append(f"log osc {i+1} ({'damped' if d == 'd' else 'undamped'})")
        for i in range(self._n_cal):
            d = self._cal_damps[i]
            names.append(f"cal osc {i+1} ({'damped' if d == 'd' else 'undamped'})")
        return names

    @property
    def formula_log10_latex(self):
        parts = [r"A + B \log_{10}(t)"]
        for i in range(self._n_log):
            d = self._log_damps[i]
            idx = i + 1
            if d == "d":
                parts.append(rf"C_{{l{idx}}} t^{{-D_{{l{idx}}}}} \cos(\omega_{{l{idx}}} \ln t + \varphi_{{l{idx}}})")
            else:
                parts.append(rf"C_{{l{idx}}} \cos(\omega_{{l{idx}}} \ln t + \varphi_{{l{idx}}})")
        for i in range(self._n_cal):
            d = self._cal_damps[i]
            idx = i + 1
            if d == "d":
                parts.append(rf"C_{{c{idx}}} t^{{-D_{{c{idx}}}}} \cos(\omega_{{c{idx}}} t + \varphi_{{c{idx}}})")
            else:
                parts.append(rf"C_{{c{idx}}} \cos(\omega_{{c{idx}}} t + \varphi_{{c{idx}}})")
        return " + ".join(parts)

    @property
    def formula_product_latex(self):
        return None  # too complex for product form

    @property
    def component_details(self):
        det = {
            "A (constant)": ("A", [("A", "A")]),
            "B\u00b7log\u2081\u2080(t)": ("B\u00b7log\u2081\u2080(t)", [("B", "B")]),
        }
        for i in range(self._n_log):
            d = self._log_damps[i]
            name = f"log osc {i+1} ({'damped' if d == 'd' else 'undamped'})"
            if d == "d":
                det[name] = (
                    f"C\u00b7t^(\u2212D)\u00b7cos(\u03c9\u00b7ln(t)+\u03c6)",
                    [],
                )
            else:
                det[name] = ("C\u00b7cos(\u03c9\u00b7ln(t)+\u03c6)", [])
        for i in range(self._n_cal):
            d = self._cal_damps[i]
            name = f"cal osc {i+1} ({'damped' if d == 'd' else 'undamped'})"
            if d == "d":
                det[name] = (
                    f"C\u00b7t^(\u2212D)\u00b7cos(\u03c9\u00b7t+\u03c6)",
                    [],
                )
            else:
                det[name] = ("C\u00b7cos(\u03c9\u00b7t+\u03c6)", [])
        return det

    def components(self, t):
        """Decompose into individual additive terms."""
        t = np.asarray(t, float)
        ts = np.maximum(t, 0.1)
        p = self._params
        result = {
            "A (constant)": np.full_like(ts, p["A"]),
            "B\u00b7log\u2081\u2080(t)": p["B"] * np.log10(ts),
        }
        for i in range(self._n_log):
            suffix = str(i + 1) if self._n_log > 1 else ""
            d = self._log_damps[i]
            C = p[f"C_log{suffix}"]; W = p[f"W_log{suffix}"]; PHI = p[f"PHI_log{suffix}"]
            name = f"log osc {i+1} ({'damped' if d == 'd' else 'undamped'})"
            if d == "d":
                D = p[f"D_log{suffix}"]
                result[name] = C * ts**(-D) * np.cos(W * np.log(ts) + PHI)
            else:
                result[name] = C * np.cos(W * np.log(ts) + PHI)
        for i in range(self._n_cal):
            suffix = str(i + 1) if self._n_cal > 1 else ""
            d = self._cal_damps[i]
            C = p[f"C_cal{suffix}"]; W = p[f"W_cal{suffix}"]; PHI = p[f"PHI_cal{suffix}"]
            name = f"cal osc {i+1} ({'damped' if d == 'd' else 'undamped'})"
            if d == "d":
                D = p[f"D_cal{suffix}"]
                result[name] = C * ts**(-D) * np.cos(W * ts + PHI)
            else:
                result[name] = C * np.cos(W * ts + PHI)
        return result

    def _build_colors(self):
        """Neutral gray-blue palette -- distinct from other model families."""
        self.colors = {}
        n = len(self.quantiles)
        for i, q in enumerate(self.quantiles):
            frac = i / max(n - 1, 1)
            r = int(70 + 80 * frac)
            g = int(100 + 60 * frac)
            b = int(140 + 50 * frac)
            self.colors[q] = f"#{r:02x}{g:02x}{b:02x}"


