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
    _A   = -1.146871  
    _B   =                 5.051440  
    _C   =                 0.689800  
    _W   =                 7.420028  
    _PHI =                 1.453362  
    _D   =                 0.708113  
    _C2  =                 0.233047  
    _W2  =                 1.733178  
    _PHI2 = -1.923186  

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
    _A     = -1.146940  
    _B     =        5.051521  
    _C1    =        0.690016  
    _W_log =        7.420125  
    _PHI1  =        1.453219  
    _D1    =        0.708418  
    _C2    =        0.233494  
    _W_cal =        1.733171  
    _PHI2  = -1.923130  
    _D2    =        0.001000  

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
    _A    = -1.113051  
    _B    =     5.013919  
    _C1   =     0.765444  
    _W1   =     7.471808  
    _PHI1 =     1.297984  
    _D1   =     0.773452  
    _C2   =     0.257516  
    _Wc   =     1.720228  
    _PHI2 = -1.736955  
    _C3   =     0.392739  
    _W2   =   15.993374  
    _PHI3 =     1.889585  
    _D2   =     0.932751  

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
    _A    = -1.135475  
    _B    =     5.037834  
    _C1   =     0.738861  
    _W1   =     7.356028  
    _PHI1 =     1.659079  
    _D    =     0.730244  
    _C2   =     0.235258  
    _Wc1  =     1.750651  
    _PHI2 = -2.086733  
    _C3   =     0.114575  
    _Wc2  =     3.280654  
    _PHI3 = -2.452119  

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
    _A    = -1.114180  
    _B    =     5.017427  
    _C1   =     0.890964  
    _W1   =     7.483988  
    _PHI1 =     1.389285  
    _D1   =     0.832962  
    _C2   =     0.242031  
    _Wc1  =     1.739799  
    _PHI2 = -1.918563  
    _C3   =     0.422538  
    _W2   =   16.237963  
    _PHI3 =     1.885419  
    _D2   =     1.166351  
    _C4   =     0.105464  
    _Wc2  =     3.340729  
    _PHI4 =     3.135899  

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
    _A    = -1.113156  
    _B    =     5.016722  
    _C1   =     0.921541  
    _W1   =     7.482817  
    _PHI1 =     1.403663  
    _D1   =     0.847676  
    _C2   =     0.240589  
    _Wc1  =     1.740838  
    _PHI2 =  -1.935091  
    _Dc1  =     0.000000  
    _C3   =     0.433531  
    _W2   =   16.252921  
    _PHI3 =     1.891523  
    _D2   =     1.205977  
    _C4   =     0.134349  
    _Wc2  =     3.342578  
    _PHI4 =    3.121447  
    _Dc2  =     0.109722  

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
# DO NOT MOVE: tools/fit_all_eppl_configs.py regex-matches this marker line
# exactly so monthly refits replace the dict in-place rather than prepending
# a duplicate.
_EPPL_CONFIG_PARAMS = {
    "ecfg_0_0": {"n_log": 0, "n_cal": 0, "log_damps": [], "cal_damps": [], "params": {"A": -1.173784, "B": 5.081215}, "r2": 0.962654, "sigma": 0.295620},
    "ecfg_0_1d": {"n_log": 0, "n_cal": 1, "log_damps": [], "cal_damps": ['d'], "params": {"A": -1.185489, "B": 5.092057, "C_cal": 0.376542, "W_cal": 1.739438, "PHI_cal": -2.060581, "w_cal": 0.056790}, "r2": 0.981127, "sigma": 0.210151},
    "ecfg_0_1u": {"n_log": 0, "n_cal": 1, "log_damps": [], "cal_damps": ['u'], "params": {"A": -1.211635, "B": 5.108941, "C_cal": 0.276894, "W_cal": 1.765383, "PHI_cal": -2.280318}, "r2": 0.978941, "sigma": 0.221991},
    "ecfg_0_2dd": {"n_log": 0, "n_cal": 2, "log_damps": [], "cal_damps": ['d', 'd'], "params": {"A": -1.096556, "B": 5.001078, "C_cal1": 0.614571, "W_cal1": 4.147145, "PHI_cal1": -2.025236, "w_cal1": 0.333507, "C_cal2": 0.399038, "W_cal2": 1.711301, "PHI_cal2": -1.800361, "w_cal2": 0.058408}, "r2": 0.986551, "sigma": 0.177400},
    "ecfg_0_2du": {"n_log": 0, "n_cal": 2, "log_damps": [], "cal_damps": ['d', 'u'], "params": {"A": -1.289916, "B": 5.212834, "C_cal1": 0.387360, "W_cal1": 1.753223, "PHI_cal1": -2.186095, "w_cal1": 0.058384, "C_cal2": 0.137250, "W_cal2": 0.834382, "PHI_cal2": -2.702990}, "r2": 0.984474, "sigma": 0.190610},
    "ecfg_0_2uu": {"n_log": 0, "n_cal": 2, "log_damps": [], "cal_damps": ['u', 'u'], "params": {"A": -1.183331, "B": 5.075917, "C_cal1": 0.133621, "W_cal1": 3.119343, "PHI_cal1": -0.727837, "C_cal2": 0.285132, "W_cal2": 1.763443, "PHI_cal2": -2.221823}, "r2": 0.982750, "sigma": 0.200914},
    "ecfg_1d_0": {"n_log": 1, "n_cal": 0, "log_damps": ['d'], "cal_damps": [], "params": {"A": -1.140066, "B": 5.060594, "C_log": 0.542874, "W_log": 7.680129, "PHI_log": 1.224824, "w_log": 0.101156}, "r2": 0.983419, "sigma": 0.196980},
    "ecfg_1d_1d": {"n_log": 1, "n_cal": 1, "log_damps": ['d'], "cal_damps": ['d'], "params": {"A": -1.157398, "B": 5.069443, "C_log": 0.472459, "W_log": 7.769475, "PHI_log": 1.320545, "w_log": 0.106177, "C_cal": 0.219876, "W_cal": 1.843120, "PHI_cal": 2.974032, "w_cal": 0.036873}, "r2": 0.989615, "sigma": 0.155888},
    "ecfg_1d_1u": {"n_log": 1, "n_cal": 1, "log_damps": ['d'], "cal_damps": ['u'], "params": {"A": -1.189358, "B": 5.101294, "C_log": 0.478427, "W_log": 7.794515, "PHI_log": 1.320053, "w_log": 0.105442, "C_cal": 0.193897, "W_cal": 1.857282, "PHI_cal": 2.802240}, "r2": 0.989340, "sigma": 0.157939},
    "ecfg_1d_2dd": {"n_log": 1, "n_cal": 2, "log_damps": ['d'], "cal_damps": ['d', 'd'], "params": {"A": -1.152768, "B": 5.064707, "C_log": 0.484537, "W_log": 7.809597, "PHI_log": 1.264030, "w_log": 0.106425, "C_cal1": 0.219245, "W_cal1": 1.846572, "PHI_cal1": 2.924649, "w_cal1": 0.036499, "C_cal2": 0.236297, "W_cal2": 10.292570, "PHI_cal2": -1.198334, "w_cal2": 0.177048}, "r2": 0.991712, "sigma": 0.139260},
    "ecfg_1d_2du": {"n_log": 1, "n_cal": 2, "log_damps": ['d'], "cal_damps": ['d', 'u'], "params": {"A": -1.189530, "B": 5.101966, "C_log": 0.491673, "W_log": 7.787010, "PHI_log": 1.337354, "w_log": 0.105686, "C_cal1": 0.223446, "W_cal1": 10.000000, "PHI_cal1": -0.385570, "w_cal1": 0.173351, "C_cal2": 0.193789, "W_cal2": 1.862757, "PHI_cal2": 2.731482}, "r2": 0.991245, "sigma": 0.143135},
    "ecfg_1d_2uu": {"n_log": 1, "n_cal": 2, "log_damps": ['d'], "cal_damps": ['u', 'u'], "params": {"A": -1.176691, "B": 5.088978, "C_log": 0.546098, "W_log": 7.726491, "PHI_log": 1.488667, "w_log": 0.106829, "C_cal1": 0.205753, "W_cal1": 1.883481, "PHI_cal1": 2.507911, "C_cal2": 0.116652, "W_cal2": 3.344440, "PHI_cal2": -3.125422}, "r2": 0.991906, "sigma": 0.137621},
    "ecfg_1u_0": {"n_log": 1, "n_cal": 0, "log_damps": ['u'], "cal_damps": [], "params": {"A": -1.232539, "B": 5.182900, "C_log": 0.230726, "W_log": 7.721562, "PHI_log": 1.188723}, "r2": 0.973981, "sigma": 0.246753},
    "ecfg_1u_1d": {"n_log": 1, "n_cal": 1, "log_damps": ['u'], "cal_damps": ['d'], "params": {"A": -1.180404, "B": 5.090632, "C_log": 0.182428, "W_log": 7.252503, "PHI_log": 1.714152, "C_cal": 0.293675, "W_cal": 1.737834, "PHI_cal": -1.997991, "w_cal": 0.041179}, "r2": 0.986943, "sigma": 0.174798},
    "ecfg_1u_1u": {"n_log": 1, "n_cal": 1, "log_damps": ['u'], "cal_damps": ['u'], "params": {"A": -1.234885, "B": 5.154289, "C_log": 0.184747, "W_log": 7.370084, "PHI_log": 1.623173, "C_cal": 0.245727, "W_cal": 1.760606, "PHI_cal": -2.246028}, "r2": 0.986226, "sigma": 0.179535},
    "ecfg_1u_2dd": {"n_log": 1, "n_cal": 2, "log_damps": ['u'], "cal_damps": ['d', 'd'], "params": {"A": -1.168494, "B": 5.071364, "C_log": 0.150704, "W_log": 7.043406, "PHI_log": 2.148976, "C_cal1": 0.300974, "W_cal1": 1.737710, "PHI_cal1": -2.003796, "w_cal1": 0.043205, "C_cal2": 0.212469, "W_cal2": 2.854918, "PHI_cal2": 0.062348, "w_cal2": 0.100288}, "r2": 0.989822, "sigma": 0.154328},
    "ecfg_1u_2du": {"n_log": 1, "n_cal": 2, "log_damps": ['u'], "cal_damps": ['d', 'u'], "params": {"A": -1.089246, "B": 4.968562, "C_log": 0.161827, "W_log": 6.419797, "PHI_log": -2.864354, "C_cal1": 0.805182, "W_cal1": 4.250424, "PHI_cal1": -2.519881, "w_cal1": 0.379993, "C_cal2": 0.286664, "W_cal2": 1.742011, "PHI_cal2": -2.076675}, "r2": 0.989673, "sigma": 0.155450},
    "ecfg_1u_2uu": {"n_log": 1, "n_cal": 2, "log_damps": ['u'], "cal_damps": ['u', 'u'], "params": {"A": -1.195381, "B": 5.101300, "C_log": 0.178136, "W_log": 7.144865, "PHI_log": 2.014151, "C_cal1": 0.256178, "W_cal1": 1.760221, "PHI_cal1": -2.212661, "C_cal2": 0.121642, "W_cal2": 3.167675, "PHI_cal2": -1.212915}, "r2": 0.989170, "sigma": 0.159195},
    "ecfg_2dd_0": {"n_log": 2, "n_cal": 0, "log_damps": ['d', 'd'], "cal_damps": [], "params": {"A": -1.127776, "B": 5.048206, "C_log1": 0.553513, "W_log1": 7.728906, "PHI_log1": 1.158299, "w_log1": 0.101405, "C_log2": 0.277842, "W_log2": 16.516411, "PHI_log2": 1.634666, "w_log2": 0.249992}, "r2": 0.985204, "sigma": 0.186074},
    "ecfg_2dd_1d": {"n_log": 2, "n_cal": 1, "log_damps": ['d', 'd'], "cal_damps": ['d'], "params": {"A": -1.143612, "B": 5.055210, "C_log1": 0.484316, "W_log1": 7.829766, "PHI_log1": 1.229580, "w_log1": 0.106270, "C_log2": 0.266809, "W_log2": 16.646059, "PHI_log2": 1.554957, "w_log2": 0.250686, "C_cal": 0.216854, "W_cal": 1.840976, "PHI_cal": 2.989631, "w_cal": 0.035055}, "r2": 0.991243, "sigma": 0.143154},
    "ecfg_2dd_1u": {"n_log": 2, "n_cal": 1, "log_damps": ['d', 'd'], "cal_damps": ['u'], "params": {"A": -1.191233, "B": 5.102466, "C_log1": 0.473390, "W_log1": 7.807929, "PHI_log1": 1.312068, "w_log1": 0.104927, "C_log2": 0.109168, "W_log2": 37.276563, "PHI_log2": 1.716159, "w_log2": 0.044386, "C_cal": 0.201474, "W_cal": 1.850313, "PHI_cal": 2.844831}, "r2": 0.991176, "sigma": 0.143701},
    "ecfg_2dd_2dd": {"n_log": 2, "n_cal": 2, "log_damps": ['d', 'd'], "cal_damps": ['d', 'd'], "params": {"A": -0.628613, "B": 4.607373, "C_log1": 3.912672, "W_log1": 3.008530, "PHI_log1": 1.981470, "w_log1": 0.500000, "C_log2": 0.831928, "W_log2": 5.804256, "PHI_log2": -1.324804, "w_log2": 0.105199, "C_cal1": 1.516885, "W_cal1": 1.513709, "PHI_cal1": -2.167326, "w_cal1": 0.176575, "C_cal2": 0.205880, "W_cal2": 10.232352, "PHI_cal2": -0.970630, "w_cal2": 0.166852}, "r2": 0.987254, "sigma": 0.172702},
    "ecfg_2dd_2du": {"n_log": 2, "n_cal": 2, "log_damps": ['d', 'd'], "cal_damps": ['d', 'u'], "params": {"A": -1.189596, "B": 5.102710, "C_log1": 0.487108, "W_log1": 7.786044, "PHI_log1": 1.328379, "w_log1": 0.104547, "C_log2": 0.124018, "W_log2": 30.429686, "PHI_log2": -1.408445, "w_log2": 0.065882, "C_cal1": 0.233721, "W_cal1": 10.000000, "PHI_cal1": -0.591390, "w_cal1": 0.172940, "C_cal2": 0.191346, "W_cal2": 1.868342, "PHI_cal2": 2.722372}, "r2": 0.992844, "sigma": 0.129400},
    "ecfg_2dd_2uu": {"n_log": 2, "n_cal": 2, "log_damps": ['d', 'd'], "cal_damps": ['u', 'u'], "params": {"A": -1.167364, "B": 5.079560, "C_log1": 0.250431, "W_log1": 16.823754, "PHI_log1": 1.460423, "w_log1": 0.251550, "C_log2": 0.556269, "W_log2": 7.803554, "PHI_log2": 1.373041, "w_log2": 0.107049, "C_cal1": 0.202747, "W_cal1": 1.881312, "PHI_cal1": 2.520901, "C_cal2": 0.113542, "W_cal2": 3.355482, "PHI_cal2": 3.033229}, "r2": 0.993320, "sigma": 0.125028},
    "ecfg_2du_0": {"n_log": 2, "n_cal": 0, "log_damps": ['d', 'u'], "cal_damps": [], "params": {"A": -1.124220, "B": 5.029270, "C_log1": 0.521293, "W_log1": 7.684369, "PHI_log1": 1.227815, "w_log1": 0.095766, "C_log2": 0.151372, "W_log2": 20.773683, "PHI_log2": -0.896309}, "r2": 0.988298, "sigma": 0.165479},
    "ecfg_2du_1d": {"n_log": 2, "n_cal": 1, "log_damps": ['d', 'u'], "cal_damps": ['d'], "params": {"A": -1.159884, "B": 5.064082, "C_log1": 0.618147, "W_log1": 8.736504, "PHI_log1": -0.566654, "w_log1": 0.104475, "C_log2": 0.163302, "W_log2": 20.724695, "PHI_log2": -0.867348, "C_cal": 0.723176, "W_cal": 4.397533, "PHI_cal": -1.275783, "w_cal": 0.267779}, "r2": 0.991035, "sigma": 0.144844},
    "ecfg_2du_1u": {"n_log": 2, "n_cal": 1, "log_damps": ['d', 'u'], "cal_damps": ['u'], "params": {"A": -1.184380, "B": 5.101305, "C_log1": 0.479821, "W_log1": 7.731005, "PHI_log1": 1.344117, "w_log1": 0.105839, "C_log2": 0.109760, "W_log2": 20.299527, "PHI_log2": -0.436882, "C_cal": 0.157679, "W_cal": 1.902087, "PHI_cal": 2.682141}, "r2": 0.990968, "sigma": 0.145379},
    "ecfg_2du_2dd": {"n_log": 2, "n_cal": 2, "log_damps": ['d', 'u'], "cal_damps": ['d', 'd'], "params": {"A": -1.158361, "B": 5.068582, "C_log1": 0.403762, "W_log1": 7.901512, "PHI_log1": 1.339118, "w_log1": 0.106518, "C_log2": 0.088706, "W_log2": 7.595651, "PHI_log2": 0.660092, "C_cal1": 0.235665, "W_cal1": 1.820294, "PHI_cal1": -3.068852, "w_cal1": 0.033592, "C_cal2": 0.239068, "W_cal2": 10.308671, "PHI_cal2": -1.251776, "w_cal2": 0.177478}, "r2": 0.992519, "sigma": 0.132307},
    "ecfg_2du_2du": {"n_log": 2, "n_cal": 2, "log_damps": ['d', 'u'], "cal_damps": ['d', 'u'], "params": {"A": -1.131272, "B": 5.012643, "C_log1": 0.270433, "W_log1": 7.953887, "PHI_log1": 0.262641, "w_log1": 0.069334, "C_log2": 0.112180, "W_log2": 16.454993, "PHI_log2": 1.452554, "C_cal1": 0.651536, "W_cal1": 2.993211, "PHI_cal1": 0.385461, "w_cal1": 0.330616, "C_cal2": 0.290289, "W_cal2": 1.677997, "PHI_cal2": -1.076457}, "r2": 0.991493, "sigma": 0.141092},
    "ecfg_2du_2uu": {"n_log": 2, "n_cal": 2, "log_damps": ['d', 'u'], "cal_damps": ['u', 'u'], "params": {"A": -1.183241, "B": 5.102046, "C_log1": 0.507664, "W_log1": 7.620463, "PHI_log1": 1.622911, "w_log1": 0.106671, "C_log2": 0.092300, "W_log2": 20.256307, "PHI_log2": -0.722984, "C_cal1": 0.193756, "W_cal1": 1.907184, "PHI_cal1": 2.575084, "C_cal2": 0.106908, "W_cal2": 3.323490, "PHI_cal2": -2.693162}, "r2": 0.992866, "sigma": 0.129202},
    "ecfg_2uu_0": {"n_log": 2, "n_cal": 0, "log_damps": ['u', 'u'], "cal_damps": [], "params": {"A": -1.176813, "B": 5.085663, "C_log1": 0.180520, "W_log1": 20.892431, "PHI_log1": -1.111269, "C_log2": 0.241183, "W_log2": 7.180144, "PHI_log2": 1.870043}, "r2": 0.980460, "sigma": 0.213831},
    "ecfg_2uu_1d": {"n_log": 2, "n_cal": 1, "log_damps": ['u', 'u'], "cal_damps": ['d'], "params": {"A": -1.124459, "B": 5.004040, "C_log1": 0.265000, "W_log1": 7.067953, "PHI_log1": 1.781609, "C_log2": 0.159753, "W_log2": 8.815924, "PHI_log2": 0.461503, "C_cal": 0.247165, "W_cal": 1.763631, "PHI_cal": -2.340795, "w_cal": 0.029502}, "r2": 0.990087, "sigma": 0.152303},
    "ecfg_2uu_1u": {"n_log": 2, "n_cal": 1, "log_damps": ['u', 'u'], "cal_damps": ['u'], "params": {"A": -1.184602, "B": 5.073838, "C_log1": 0.159905, "W_log1": 8.867766, "PHI_log1": 0.497450, "C_log2": 0.257808, "W_log2": 7.160579, "PHI_log2": 1.707531, "C_cal": 0.211546, "W_cal": 1.788972, "PHI_cal": -2.646276}, "r2": 0.989315, "sigma": 0.158123},
    "ecfg_2uu_2dd": {"n_log": 2, "n_cal": 2, "log_damps": ['u', 'u'], "cal_damps": ['d', 'd'], "params": {"A": -1.274573, "B": 5.177999, "C_log1": 0.132538, "W_log1": 21.168160, "PHI_log1": -1.467086, "C_log2": 0.130871, "W_log2": 17.071574, "PHI_log2": 1.327892, "C_cal1": 0.569327, "W_cal1": 2.146082, "PHI_cal1": 2.441430, "w_cal1": 0.130633, "C_cal2": 0.284337, "W_cal2": 1.131472, "PHI_cal2": 1.920439, "w_cal2": 0.085599}, "r2": 0.990184, "sigma": 0.151560},
    "ecfg_2uu_2du": {"n_log": 2, "n_cal": 2, "log_damps": ['u', 'u'], "cal_damps": ['d', 'u'], "params": {"A": -1.087292, "B": 4.964856, "C_log1": 0.161126, "W_log1": 6.347032, "PHI_log1": -2.708822, "C_log2": 0.100394, "W_log2": 37.186575, "PHI_log2": 1.932340, "C_cal1": 0.824688, "W_cal1": 4.212409, "PHI_cal1": -2.462779, "w_cal1": 0.375861, "C_cal2": 0.298069, "W_cal2": 1.742661, "PHI_cal2": -2.090698}, "r2": 0.991786, "sigma": 0.138637},
    "ecfg_2uu_2uu": {"n_log": 2, "n_cal": 2, "log_damps": ['u', 'u'], "cal_damps": ['u', 'u'], "params": {"A": -1.167409, "B": 5.049106, "C_log1": 0.293024, "W_log1": 7.098379, "PHI_log1": 1.931776, "C_log2": 0.185402, "W_log2": 8.846058, "PHI_log2": 0.759028, "C_cal1": 0.224986, "W_cal1": 1.814174, "PHI_cal1": -2.934866, "C_cal2": 0.125693, "W_cal2": 3.283288, "PHI_cal2": -2.543245}, "r2": 0.992145, "sigma": 0.135577},
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
        self._sigma = cfg["sigma"]
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
    "cfg_0_0": {"n_log": 0, "n_cal": 0, "log_damps": [], "cal_damps": [], "params": {"A": -1.173875, "B": 5.081360}, "r2": 0.962650, "sigma": 0.295634},
    "cfg_0_1d": {"n_log": 0, "n_cal": 1, "log_damps": [], "cal_damps": ['d'], "params": {"A": -1.211645, "B": 5.108969, "C_cal": 0.276881, "W_cal": 1.765340, "PHI_cal": -2.280041, "D_cal": 0.000000}, "r2": 0.978937, "sigma": 0.222008},
    "cfg_0_1u": {"n_log": 0, "n_cal": 1, "log_damps": [], "cal_damps": ['u'], "params": {"A": -1.211646, "B": 5.108970, "C_cal": 0.276881, "W_cal": 1.765341, "PHI_cal": -2.280043}, "r2": 0.978937, "sigma": 0.222008},
    "cfg_0_2dd": {"n_log": 0, "n_cal": 2, "log_damps": [], "cal_damps": ['d', 'd'], "params": {"A": -1.047754, "B": 4.943547, "C_cal1": 1.352713, "W_cal1": 3.066049, "PHI_cal1": -0.612299, "D_cal1": 1.531193, "C_cal2": 0.746118, "W_cal2": 1.716575, "PHI_cal2": -1.783384, "D_cal2": 0.442203}, "r2": 0.986731, "sigma": 0.176210},
    "cfg_0_2du": {"n_log": 0, "n_cal": 2, "log_damps": [], "cal_damps": ['d', 'u'], "params": {"A": -1.087910, "B": 4.978008, "C_cal1": 0.738656, "W_cal1": 3.058029, "PHI_cal1": -0.383182, "D_cal1": 1.035207, "C_cal2": 0.299985, "W_cal2": 1.729902, "PHI_cal2": -1.901978}, "r2": 0.985651, "sigma": 0.183242},
    "cfg_0_2uu": {"n_log": 0, "n_cal": 2, "log_damps": [], "cal_damps": ['u', 'u'], "params": {"A": -1.328861, "B": 5.243841, "C_cal1": 0.292755, "W_cal1": 1.781646, "PHI_cal1": -2.416708, "C_cal2": 0.152796, "W_cal2": 0.813027, "PHI_cal2": -2.741282}, "r2": 0.983187, "sigma": 0.198349},
    "cfg_1d_0": {"n_log": 1, "n_cal": 0, "log_damps": ['d'], "cal_damps": [], "params": {"A": -1.153820, "B": 5.079271, "C_log": 0.733974, "W_log": 7.557911, "PHI_log": 1.377121, "D_log": 0.607966}, "r2": 0.978055, "sigma": 0.226611},
    "cfg_1d_1d": {"n_log": 1, "n_cal": 1, "log_damps": ['d'], "cal_damps": ['d'], "params": {"A": -1.146875, "B": 5.051453, "C_log": 0.689836, "W_log": 7.420057, "PHI_log": 1.453329, "D_log": 0.708168, "C_cal": 0.233044, "W_cal": 1.733155, "PHI_cal": -1.923037, "D_cal": 0.000000}, "r2": 0.988870, "sigma": 0.161384},
    "cfg_1d_1u": {"n_log": 1, "n_cal": 1, "log_damps": ['d'], "cal_damps": ['u'], "params": {"A": -1.146875, "B": 5.051454, "C_log": 0.689837, "W_log": 7.420057, "PHI_log": 1.453329, "D_log": 0.708168, "C_cal": 0.233044, "W_cal": 1.733155, "PHI_cal": -1.923038}, "r2": 0.988870, "sigma": 0.161384},
    "cfg_1d_2dd": {"n_log": 1, "n_cal": 2, "log_damps": ['d'], "cal_damps": ['d', 'd'], "params": {"A": -1.119963, "B": 5.023515, "C_log": 0.894153, "W_log": 7.537078, "PHI_log": 1.247825, "D_log": 0.859867, "C_cal1": 0.575944, "W_cal1": 10.000000, "PHI_cal1": -0.600808, "D_cal1": 1.461041, "C_cal2": 0.234815, "W_cal2": 1.724073, "PHI_cal2": -1.813160, "D_cal2": 0.000000}, "r2": 0.990862, "sigma": 0.146232},
    "cfg_1d_2du": {"n_log": 1, "n_cal": 2, "log_damps": ['d'], "cal_damps": ['d', 'u'], "params": {"A": -1.135472, "B": 5.037822, "C_log": 0.738890, "W_log": 7.356010, "PHI_log": 1.659095, "D_log": 0.730226, "C_cal1": 0.235259, "W_cal1": 1.750674, "PHI_cal1": -2.086894, "D_cal1": 0.000000, "C_cal2": 0.114588, "W_cal2": 3.280720, "PHI_cal2": -2.452578}, "r2": 0.991324, "sigma": 0.142482},
    "cfg_1d_2uu": {"n_log": 1, "n_cal": 2, "log_damps": ['d'], "cal_damps": ['u', 'u'], "params": {"A": -1.135472, "B": 5.037822, "C_log": 0.738890, "W_log": 7.356010, "PHI_log": 1.659096, "D_log": 0.730225, "C_cal1": 0.235259, "W_cal1": 1.750674, "PHI_cal1": -2.086893, "C_cal2": 0.114588, "W_cal2": 3.280720, "PHI_cal2": -2.452577}, "r2": 0.991324, "sigma": 0.142482},
    "cfg_1u_0": {"n_log": 1, "n_cal": 0, "log_damps": ['u'], "cal_damps": [], "params": {"A": -1.232639, "B": 5.183069, "C_log": 0.230735, "W_log": 7.722702, "PHI_log": 1.187301}, "r2": 0.973977, "sigma": 0.246768},
    "cfg_1u_1d": {"n_log": 1, "n_cal": 1, "log_damps": ['u'], "cal_damps": ['d'], "params": {"A": -1.234864, "B": 5.154251, "C_log": 0.184756, "W_log": 7.369940, "PHI_log": 1.623339, "C_cal": 0.245734, "W_cal": 1.760620, "PHI_cal": -2.246105, "D_cal": 0.000000}, "r2": 0.986223, "sigma": 0.179550},
    "cfg_1u_1u": {"n_log": 1, "n_cal": 1, "log_damps": ['u'], "cal_damps": ['u'], "params": {"A": -1.234864, "B": 5.154251, "C_log": 0.184756, "W_log": 7.369940, "PHI_log": 1.623339, "C_cal": 0.245734, "W_cal": 1.760620, "PHI_cal": -2.246106}, "r2": 0.986223, "sigma": 0.179550},
    "cfg_1u_2dd": {"n_log": 1, "n_cal": 2, "log_damps": ['u'], "cal_damps": ['d', 'd'], "params": {"A": -1.006185, "B": 4.865911, "C_log": 0.161247, "W_log": 5.310100, "PHI_log": -0.347192, "C_cal1": 1.772314, "W_cal1": 3.074611, "PHI_cal1": -0.657795, "D_cal1": 1.706594, "C_cal2": 0.848744, "W_cal2": 1.734324, "PHI_cal2": -2.025364, "D_cal2": 0.492016}, "r2": 0.990270, "sigma": 0.150893},
    "cfg_1u_2du": {"n_log": 1, "n_cal": 2, "log_damps": ['u'], "cal_damps": ['d', 'u'], "params": {"A": -1.094170, "B": 4.981448, "C_log": 0.161787, "W_log": 6.664948, "PHI_log": 2.924047, "C_cal1": 0.615604, "W_cal1": 3.121978, "PHI_cal1": -0.910641, "D_cal1": 0.898565, "C_cal2": 0.276107, "W_cal2": 1.740379, "PHI_cal2": -2.038337}, "r2": 0.990144, "sigma": 0.151865},
    "cfg_1u_2uu": {"n_log": 1, "n_cal": 2, "log_damps": ['u'], "cal_damps": ['u', 'u'], "params": {"A": -1.195363, "B": 5.101264, "C_log": 0.178158, "W_log": 7.144729, "PHI_log": 2.014367, "C_cal1": 0.256183, "W_cal1": 1.760251, "PHI_cal1": -2.212877, "C_cal2": 0.121646, "W_cal2": 3.167757, "PHI_cal2": -1.213413}, "r2": 0.989168, "sigma": 0.159207},
    "cfg_2dd_0": {"n_log": 2, "n_cal": 0, "log_damps": ['d', 'd'], "cal_damps": [], "params": {"A": -1.130959, "B": 5.038805, "C_log1": 0.705568, "W_log1": 7.377517, "PHI_log1": 1.583198, "D_log1": 0.566186, "C_log2": 0.171854, "W_log2": 20.903808, "PHI_log2": -1.156632, "D_log2": 0.010000}, "r2": 0.983997, "sigma": 0.193516},
    "cfg_2dd_1d": {"n_log": 2, "n_cal": 1, "log_damps": ['d', 'd'], "cal_damps": ['d'], "params": {"A": -1.113052, "B": 5.013922, "C_log1": 0.765452, "W_log1": 7.471817, "PHI_log1": 1.297976, "D_log1": 0.773463, "C_log2": 0.392752, "W_log2": 15.993349, "PHI_log2": 1.889603, "D_log2": 0.932804, "C_cal": 0.257514, "W_cal": 1.720226, "PHI_cal": -1.736945, "D_cal": 0.000000}, "r2": 0.990843, "sigma": 0.146384},
    "cfg_2dd_1u": {"n_log": 2, "n_cal": 1, "log_damps": ['d', 'd'], "cal_damps": ['u'], "params": {"A": -1.113052, "B": 5.013922, "C_log1": 0.392751, "W_log1": 15.993349, "PHI_log1": 1.889602, "D_log1": 0.932802, "C_log2": 0.765451, "W_log2": 7.471817, "PHI_log2": 1.297977, "D_log2": 0.773462, "C_cal": 0.257514, "W_cal": 1.720226, "PHI_cal": -1.736945}, "r2": 0.990843, "sigma": 0.146384},
    "cfg_2dd_2dd": {"n_log": 2, "n_cal": 2, "log_damps": ['d', 'd'], "cal_damps": ['d', 'd'], "params": {"A": -1.113752, "B": 5.018172, "C_log1": 0.090187, "W_log1": 36.963949, "PHI_log1": 2.332185, "D_log1": 0.010000, "C_log2": 0.872584, "W_log2": 7.574685, "PHI_log2": 1.172171, "D_log2": 0.860020, "C_cal1": 0.245779, "W_cal1": 1.720867, "PHI_cal1": -1.794831, "D_cal1": 0.000000, "C_cal2": 0.538482, "W_cal2": 10.310219, "PHI_cal2": -1.305199, "D_cal2": 1.360139}, "r2": 0.992770, "sigma": 0.130072},
    "cfg_2dd_2du": {"n_log": 2, "n_cal": 2, "log_damps": ['d', 'd'], "cal_damps": ['d', 'u'], "params": {"A": -1.088318, "B": 4.986522, "C_log1": 0.879333, "W_log1": 7.206259, "PHI_log1": 1.974339, "D_log1": 0.804974, "C_log2": 0.366188, "W_log2": 13.038626, "PHI_log2": -2.676943, "D_log2": 0.983059, "C_cal1": 0.394977, "W_cal1": 3.259796, "PHI_cal1": -2.213342, "D_cal1": 0.532655, "C_cal2": 0.230152, "W_cal2": 1.743372, "PHI_cal2": -1.994561}, "r2": 0.992768, "sigma": 0.130093},
    "cfg_2dd_2uu": {"n_log": 2, "n_cal": 2, "log_damps": ['d', 'd'], "cal_damps": ['u', 'u'], "params": {"A": -1.114174, "B": 5.017407, "C_log1": 0.891032, "W_log1": 7.483928, "PHI_log1": 1.389298, "D_log1": 0.832947, "C_log2": 0.422419, "W_log2": 16.238168, "PHI_log2": 1.885353, "D_log2": 1.165713, "C_cal1": 0.242036, "W_cal1": 1.739827, "PHI_cal1": -1.918656, "C_cal2": 0.105488, "W_cal2": 3.340840, "PHI_cal2": 3.134976}, "r2": 0.992744, "sigma": 0.130305},
    "cfg_2du_0": {"n_log": 2, "n_cal": 0, "log_damps": ['d', 'u'], "cal_damps": [], "params": {"A": -1.130641, "B": 5.038329, "C_log1": 0.705462, "W_log1": 7.376939, "PHI_log1": 1.583467, "D_log1": 0.565812, "C_log2": 0.168837, "W_log2": 20.904199, "PHI_log2": -1.157210}, "r2": 0.984022, "sigma": 0.193364},
    "cfg_2du_1d": {"n_log": 2, "n_cal": 1, "log_damps": ['d', 'u'], "cal_damps": ['d'], "params": {"A": -1.163916, "B": 5.073560, "C_log1": 0.670547, "W_log1": 7.443535, "PHI_log1": 1.448851, "D_log1": 0.682897, "C_log2": 0.088414, "W_log2": 20.195363, "PHI_log2": -0.269939, "C_cal": 0.210935, "W_cal": 1.773495, "PHI_cal": -2.132973, "D_cal": 0.000000}, "r2": 0.989885, "sigma": 0.153848},
    "cfg_2du_1u": {"n_log": 2, "n_cal": 1, "log_damps": ['d', 'u'], "cal_damps": ['u'], "params": {"A": -1.144272, "B": 5.049050, "C_log1": 0.701091, "W_log1": 7.420752, "PHI_log1": 1.440933, "D_log1": 0.729824, "C_log2": 0.096745, "W_log2": 37.086656, "PHI_log2": 2.047683, "C_cal": 0.242614, "W_cal": 1.729290, "PHI_cal": -1.897870}, "r2": 0.990826, "sigma": 0.146514},
    "cfg_2du_2dd": {"n_log": 2, "n_cal": 2, "log_damps": ['d', 'u'], "cal_damps": ['d', 'd'], "params": {"A": -1.192589, "B": 5.080377, "C_log1": 0.585018, "W_log1": 7.242775, "PHI_log1": 1.464854, "D_log1": 0.549788, "C_log2": 0.098792, "W_log2": 37.128293, "PHI_log2": 1.998761, "C_cal1": 0.364974, "W_cal1": 1.077836, "PHI_cal1": -3.124365, "D_cal1": 0.690705, "C_cal2": 0.271926, "W_cal2": 1.712807, "PHI_cal2": -1.726335, "D_cal2": 0.000000}, "r2": 0.992295, "sigma": 0.134276},
    "cfg_2du_2du": {"n_log": 2, "n_cal": 2, "log_damps": ['d', 'u'], "cal_damps": ['d', 'u'], "params": {"A": -1.138230, "B": 5.015692, "C_log1": 0.251273, "W_log1": 8.651516, "PHI_log1": 1.092978, "D_log1": 0.130090, "C_log2": 0.292981, "W_log2": 6.957905, "PHI_log2": 2.159159, "C_cal1": 0.320649, "W_cal1": 3.264976, "PHI_cal1": -2.369649, "D_cal1": 0.442434, "C_cal2": 0.233646, "W_cal2": 1.811678, "PHI_cal2": -2.922371}, "r2": 0.992315, "sigma": 0.134099},
    "cfg_2du_2uu": {"n_log": 2, "n_cal": 2, "log_damps": ['d', 'u'], "cal_damps": ['u', 'u'], "params": {"A": -1.135646, "B": 5.040344, "C_log1": 0.778133, "W_log1": 7.385649, "PHI_log1": 1.573670, "D_log1": 0.769954, "C_log2": 0.070961, "W_log2": 36.898377, "PHI_log2": 2.249526, "C_cal1": 0.237345, "W_cal1": 1.735501, "PHI_cal1": -1.970635, "C_cal2": 0.088877, "W_cal2": 3.339989, "PHI_cal2": -2.812188}, "r2": 0.992105, "sigma": 0.135923},
    "cfg_2uu_0": {"n_log": 2, "n_cal": 0, "log_damps": ['u', 'u'], "cal_damps": [], "params": {"A": -1.114110, "B": 4.989677, "C_log1": 0.292255, "W_log1": 6.750135, "PHI_log1": 2.263445, "C_log2": 0.232793, "W_log2": 8.884470, "PHI_log2": -0.115770}, "r2": 0.982124, "sigma": 0.204525},
    "cfg_2uu_1d": {"n_log": 2, "n_cal": 1, "log_damps": ['u', 'u'], "cal_damps": ['d'], "params": {"A": -1.649681, "B": 5.791759, "C_log1": 0.214124, "W_log1": 7.529246, "PHI_log1": 1.453833, "C_log2": 0.296423, "W_log2": 2.000000, "PHI_log2": -1.599010, "C_cal": 0.334831, "W_cal": 1.772901, "PHI_cal": -2.321013, "D_cal": 0.141312}, "r2": 0.988897, "sigma": 0.161190},
    "cfg_2uu_1u": {"n_log": 2, "n_cal": 1, "log_damps": ['u', 'u'], "cal_damps": ['u'], "params": {"A": -1.184615, "B": 5.073845, "C_log1": 0.159895, "W_log1": 8.868180, "PHI_log1": 0.496797, "C_log2": 0.257814, "W_log2": 7.160547, "PHI_log2": 1.707803, "C_cal": 0.211553, "W_cal": 1.789004, "PHI_cal": -2.646637}, "r2": 0.989313, "sigma": 0.158137},
    "cfg_2uu_2dd": {"n_log": 2, "n_cal": 2, "log_damps": ['u', 'u'], "cal_damps": ['d', 'd'], "params": {"A": -1.126108, "B": 5.022372, "C_log1": 0.144599, "W_log1": 6.637244, "PHI_log1": 3.009907, "C_log2": 0.098846, "W_log2": 19.941469, "PHI_log2": -0.463206, "C_cal1": 0.291305, "W_cal1": 1.772604, "PHI_cal1": -2.166151, "D_cal1": 0.000000, "C_cal2": 0.583542, "W_cal2": 3.139937, "PHI_cal2": -0.865760, "D_cal2": 0.776542}, "r2": 0.991229, "sigma": 0.143265},
    "cfg_2uu_2du": {"n_log": 2, "n_cal": 2, "log_damps": ['u', 'u'], "cal_damps": ['d', 'u'], "params": {"A": -1.167439, "B": 5.049097, "C_log1": 0.293113, "W_log1": 7.098274, "PHI_log1": 1.932680, "C_log2": 0.185429, "W_log2": 8.847566, "PHI_log2": 0.756730, "C_cal1": 0.225036, "W_cal1": 1.814325, "PHI_cal1": -2.936559, "D_cal1": 0.000000, "C_cal2": 0.125748, "W_cal2": 3.283480, "PHI_cal2": -2.544719}, "r2": 0.992145, "sigma": 0.135576},
    "cfg_2uu_2uu": {"n_log": 2, "n_cal": 2, "log_damps": ['u', 'u'], "cal_damps": ['u', 'u'], "params": {"A": -1.167439, "B": 5.049097, "C_log1": 0.293113, "W_log1": 7.098275, "PHI_log1": 1.932678, "C_log2": 0.185429, "W_log2": 8.847567, "PHI_log2": 0.756729, "C_cal1": 0.125748, "W_cal1": 3.283480, "PHI_cal1": -2.544720, "C_cal2": 0.225036, "W_cal2": 1.814325, "PHI_cal2": -2.936560}, "r2": 0.992145, "sigma": 0.135576},
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


