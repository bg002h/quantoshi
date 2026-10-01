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
    _A   = -1.148775  
    _B   =                                   5.054512  
    _C   =                                   0.691128  
    _W   =                                   7.427131  
    _PHI =                                   1.448595  
    _D   =                                   0.710778  
    _C2  =                                   0.231917  
    _W2  =                                   1.732922  
    _PHI2 = -1.925232  

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
    _A     = -1.136722  
    _B     =                          5.054667  
    _C1    =                          1.120118  
    _W_log =                          7.435226  
    _PHI1  =                          1.559802  
    _D1    =                          0.768805  
    _C2    =                          0.365617  
    _W_cal =                          3.408694  
    _PHI2  =       2.627990  
    _D2    =                          0.544582  

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
    _A    = -1.118049  
    _B    =                       5.022016  
    _C1   =                       0.771299  
    _W1   =                   7.494727  
    _PHI1 =                       1.281440  
    _D1   =                       0.782566  
    _C2   =                       0.252634  
    _Wc   =                       1.719544  
    _PHI2 = -1.748519  
    _C3   =                       0.411283  
    _W2   =                   15.982035  
    _PHI3 =                       1.899031  
    _D2   =                       1.008117  

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
    _A    = -1.143903  
    _B    =                       5.052580  
    _C1   =                       0.750230  
    _W1   =                       7.395362  
    _PHI1 =                       1.615818  
    _D    =                       0.748869  
    _C2   =                       0.229575  
    _Wc1  =                       1.744692  
    _PHI2 = -2.056637  
    _C3   =                       0.105778  
    _Wc2  =                       3.282689  
    _PHI3 = -2.477937  

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
    _A    = -1.124488  
    _B    =                       5.034328  
    _C1   =                       0.461323  
    _W1   =                      16.273503  
    _PHI1 =                       1.868990  
    _D1   =                       1.288072  
    _C2   =                       0.233429  
    _Wc1  =                       1.733712  
    _PHI2 = -1.898174  
    _C3   =                       0.911182  
    _W2   =                7.535859  
    _PHI3 =                    1.329053  
    _D2   =                       0.850695  
    _C4   =                       0.099007  
    _Wc2  =                       3.367809  
    _PHI4 =           2.930029  

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
    _A    = -1.128713  
    _B    =                       5.039920  
    _C1   =                       0.904741  
    _W1   =                      7.584730  
    _PHI1 =                       1.184166  
    _D1   =                       0.891370  
    _C2   =                       0.249791  
    _Wc1  =                       1.731970  
    _PHI2 =  -1.789764  
    _Dc1  =                       0.000000  
    _C3   =                       0.085374  
    _W2   =                 17.217446  
    _PHI3 =                    -0.241147  
    _D2   =                       0.107018  
    _C4   =                       0.595135  
    _Wc2  =                   10.288616  
    _PHI4 =       -1.047058  
    _Dc2  =                       1.297382  

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
    "ecfg_0_0": {"n_log": 0, "n_cal": 0, "log_damps": [], "cal_damps": [], "params": {"A": -1.157034, "B": 5.054441}, "r2": 0.963296, "sigma": 0.293492},
    "ecfg_0_1d": {"n_log": 0, "n_cal": 1, "log_damps": [], "cal_damps": ['d'], "params": {"A": -1.169732, "B": 5.066382, "C_cal": 0.367937, "W_cal": 1.745733, "PHI_cal": -2.097261, "w_cal": 0.053534}, "r2": 0.981349, "sigma": 0.209212},
    "ecfg_0_1u": {"n_log": 0, "n_cal": 1, "log_damps": [], "cal_damps": ['u'], "params": {"A": -1.211880, "B": 5.109151, "C_cal": 0.276816, "W_cal": 1.766225, "PHI_cal": -2.285958}, "r2": 0.979616, "sigma": 0.218717},
    "ecfg_0_2dd": {"n_log": 0, "n_cal": 2, "log_damps": [], "cal_damps": ['d', 'd'], "params": {"A": -1.080037, "B": 4.974495, "C_cal1": 0.389115, "W_cal1": 1.717915, "PHI_cal1": -1.838212, "w_cal1": 0.055773, "C_cal2": 0.610354, "W_cal2": 4.202423, "PHI_cal2": -2.116755, "w_cal2": 0.330632}, "r2": 0.986577, "sigma": 0.177488},
    "ecfg_0_2du": {"n_log": 0, "n_cal": 2, "log_damps": [], "cal_damps": ['d', 'u'], "params": {"A": -1.280150, "B": 5.191764, "C_cal1": 0.372299, "W_cal1": 1.758285, "PHI_cal1": -2.203167, "w_cal1": 0.052392, "C_cal2": 0.130627, "W_cal2": 0.777452, "PHI_cal2": -2.418156}, "r2": 0.984388, "sigma": 0.191414},
    "ecfg_0_2uu": {"n_log": 0, "n_cal": 2, "log_damps": [], "cal_damps": ['u', 'u'], "params": {"A": -1.186777, "B": 5.081296, "C_cal1": 0.130707, "W_cal1": 3.112609, "PHI_cal1": -0.695070, "C_cal2": 0.282517, "W_cal2": 1.762771, "PHI_cal2": -2.219230}, "r2": 0.983202, "sigma": 0.198549},
    "ecfg_1d_0": {"n_log": 1, "n_cal": 0, "log_damps": ['d'], "cal_damps": [], "params": {"A": -1.122212, "B": 5.030304, "C_log": 0.535503, "W_log": 7.699217, "PHI_log": 1.204344, "w_log": 0.098645}, "r2": 0.983328, "sigma": 0.197806},
    "ecfg_1d_1d": {"n_log": 1, "n_cal": 1, "log_damps": ['d'], "cal_damps": ['d'], "params": {"A": -1.155110, "B": 5.058081, "C_log": 0.595571, "W_log": 8.588720, "PHI_log": -0.328794, "w_log": 0.101292, "C_cal": 0.632061, "W_cal": 4.375460, "PHI_cal": -1.185055, "w_cal": 0.272371}, "r2": 0.985249, "sigma": 0.186059},
    "ecfg_1d_1u": {"n_log": 1, "n_cal": 1, "log_damps": ['d'], "cal_damps": ['u'], "params": {"A": -1.186586, "B": 5.096343, "C_log": 0.478544, "W_log": 7.799202, "PHI_log": 1.320115, "w_log": 0.105503, "C_cal": 0.197403, "W_cal": 1.858952, "PHI_cal": 2.787325}, "r2": 0.989647, "sigma": 0.155872},
    "ecfg_1d_2dd": {"n_log": 1, "n_cal": 2, "log_damps": ['d'], "cal_damps": ['d', 'd'], "params": {"A": -1.076623, "B": 4.956428, "C_log": 0.250864, "W_log": 7.173432, "PHI_log": 2.026955, "w_log": 0.071562, "C_cal1": 0.481924, "W_cal1": 4.580564, "PHI_cal1": -2.912672, "w_cal1": 0.373212, "C_cal2": 0.282588, "W_cal2": 1.742137, "PHI_cal2": -2.033881, "w_cal2": 0.044556}, "r2": 0.990550, "sigma": 0.148920},
    "ecfg_1d_2du": {"n_log": 1, "n_cal": 2, "log_damps": ['d'], "cal_damps": ['d', 'u'], "params": {"A": -1.086247, "B": 4.966012, "C_log": 0.250053, "W_log": 7.065484, "PHI_log": 2.300008, "w_log": 0.071900, "C_cal1": 0.626950, "W_cal1": 4.253473, "PHI_cal1": -2.509132, "w_cal1": 0.388341, "C_cal2": 0.246892, "W_cal2": 1.750967, "PHI_cal2": -2.141865}, "r2": 0.990439, "sigma": 0.149791},
    "ecfg_1d_2uu": {"n_log": 1, "n_cal": 2, "log_damps": ['d'], "cal_damps": ['u', 'u'], "params": {"A": -1.181572, "B": 5.096973, "C_log": 0.543792, "W_log": 7.723803, "PHI_log": 1.480244, "w_log": 0.106642, "C_cal1": 0.199987, "W_cal1": 1.881502, "PHI_cal1": 2.525354, "C_cal2": 0.112199, "W_cal2": 3.348087, "PHI_cal2": 3.130682}, "r2": 0.992102, "sigma": 0.136144},
    "ecfg_1u_0": {"n_log": 1, "n_cal": 0, "log_damps": ['u'], "cal_damps": [], "params": {"A": -1.214560, "B": 5.152404, "C_log": 0.229711, "W_log": 7.522709, "PHI_log": 1.436538}, "r2": 0.974604, "sigma": 0.244132},
    "ecfg_1u_1d": {"n_log": 1, "n_cal": 1, "log_damps": ['u'], "cal_damps": ['d'], "params": {"A": -1.190994, "B": 5.107436, "C_log": 0.176174, "W_log": 7.304866, "PHI_log": 1.658317, "C_cal": 0.295394, "W_cal": 1.738105, "PHI_cal": -2.010624, "w_cal": 0.043846}, "r2": 0.987307, "sigma": 0.172596},
    "ecfg_1u_1u": {"n_log": 1, "n_cal": 1, "log_damps": ['u'], "cal_damps": ['u'], "params": {"A": -1.245311, "B": 5.171564, "C_log": 0.182751, "W_log": 7.438856, "PHI_log": 1.551007, "C_cal": 0.241867, "W_cal": 1.760525, "PHI_cal": -2.258807}, "r2": 0.986575, "sigma": 0.177500},
    "ecfg_1u_2dd": {"n_log": 1, "n_cal": 2, "log_damps": ['u'], "cal_damps": ['d', 'd'], "params": {"A": -1.090194, "B": 4.978651, "C_log": 0.143883, "W_log": 6.482464, "PHI_log": -3.064287, "C_cal1": 0.656141, "W_cal1": 4.528600, "PHI_cal1": -2.916089, "w_cal1": 0.369399, "C_cal2": 0.336951, "W_cal2": 1.734622, "PHI_cal2": -2.005042, "w_cal2": 0.050162}, "r2": 0.989789, "sigma": 0.154804},
    "ecfg_1u_2du": {"n_log": 1, "n_cal": 2, "log_damps": ['u'], "cal_damps": ['d', 'u'], "params": {"A": -1.102025, "B": 4.990346, "C_log": 0.151298, "W_log": 6.512557, "PHI_log": -3.063702, "C_cal1": 0.757227, "W_cal1": 4.283323, "PHI_cal1": -2.549265, "w_cal1": 0.377246, "C_cal2": 0.279668, "W_cal2": 1.731995, "PHI_cal2": -2.000040}, "r2": 0.989708, "sigma": 0.155413},
    "ecfg_1u_2uu": {"n_log": 1, "n_cal": 2, "log_damps": ['u'], "cal_damps": ['u', 'u'], "params": {"A": -1.212994, "B": 5.130582, "C_log": 0.170576, "W_log": 7.260412, "PHI_log": 1.861029, "C_cal1": 0.249861, "W_cal1": 1.755341, "PHI_cal1": -2.188680, "C_cal2": 0.112888, "W_cal2": 3.137893, "PHI_cal2": -1.028341}, "r2": 0.989122, "sigma": 0.159781},
    "ecfg_2dd_0": {"n_log": 2, "n_cal": 0, "log_damps": ['d', 'd'], "cal_damps": [], "params": {"A": -1.102482, "B": 4.993311, "C_log1": 0.182248, "W_log1": 20.979541, "PHI_log1": -1.258862, "w_log1": 0.030133, "C_log2": 0.516382, "W_log2": 7.698957, "PHI_log2": 1.209320, "w_log2": 0.093846}, "r2": 0.988417, "sigma": 0.164873},
    "ecfg_2dd_1d": {"n_log": 2, "n_cal": 1, "log_damps": ['d', 'd'], "cal_damps": ['d'], "params": {"A": -1.138288, "B": 5.047753, "C_log1": 0.486151, "W_log1": 7.822409, "PHI_log1": 1.229355, "w_log1": 0.106146, "C_log2": 0.267283, "W_log2": 16.642618, "PHI_log2": 1.557350, "w_log2": 0.250514, "C_cal": 0.218824, "W_cal": 1.842067, "PHI_cal": 2.983533, "w_cal": 0.031368}, "r2": 0.991488, "sigma": 0.141333},
    "ecfg_2dd_1u": {"n_log": 2, "n_cal": 1, "log_damps": ['d', 'd'], "cal_damps": ['u'], "params": {"A": -1.067261, "B": 4.947612, "C_log1": 0.599677, "W_log1": 5.323868, "PHI_log1": 2.014599, "w_log1": 0.422254, "C_log2": 0.257926, "W_log2": 7.451675, "PHI_log2": 1.551982, "w_log2": 0.073226, "C_cal": 0.241845, "W_cal": 1.741493, "PHI_cal": -2.014601}, "r2": 0.990154, "sigma": 0.152009},
    "ecfg_2dd_2dd": {"n_log": 2, "n_cal": 2, "log_damps": ['d', 'd'], "cal_damps": ['d', 'd'], "params": {"A": -1.149502, "B": 5.059817, "C_log1": 0.481287, "W_log1": 7.796956, "PHI_log1": 1.269338, "w_log1": 0.105199, "C_log2": 0.122327, "W_log2": 30.360507, "PHI_log2": -1.295891, "w_log2": 0.066156, "C_cal1": 0.219647, "W_cal1": 1.854061, "PHI_cal1": 2.895917, "w_cal1": 0.034202, "C_cal2": 0.244419, "W_cal2": 10.233312, "PHI_cal2": -1.234157, "w_cal2": 0.175406}, "r2": 0.993428, "sigma": 0.124192},
    "ecfg_2dd_2du": {"n_log": 2, "n_cal": 2, "log_damps": ['d', 'd'], "cal_damps": ['d', 'u'], "params": {"A": -1.149150, "B": 5.064419, "C_log1": 0.573707, "W_log1": 7.856112, "PHI_log1": 1.325607, "w_log1": 0.112212, "C_log2": 0.257659, "W_log2": 16.817472, "PHI_log2": 1.479569, "w_log2": 0.250859, "C_cal1": 0.250383, "W_cal1": 1.878484, "PHI_cal1": 2.550194, "w_cal1": 0.044686, "C_cal2": 0.114742, "W_cal2": 3.363763, "PHI_cal2": 2.963754}, "r2": 0.993719, "sigma": 0.121411},
    "ecfg_2dd_2uu": {"n_log": 2, "n_cal": 2, "log_damps": ['d', 'd'], "cal_damps": ['u', 'u'], "params": {"A": -1.171906, "B": 5.086949, "C_log1": 0.251026, "W_log1": 16.817250, "PHI_log1": 1.463296, "w_log1": 0.251550, "C_log2": 0.553675, "W_log2": 7.800751, "PHI_log2": 1.365206, "w_log2": 0.106818, "C_cal1": 0.197208, "W_cal1": 1.878985, "PHI_cal1": 2.541328, "C_cal2": 0.109326, "W_cal2": 3.359755, "PHI_cal2": 3.002785}, "r2": 0.993475, "sigma": 0.123746},
    "ecfg_2du_0": {"n_log": 2, "n_cal": 0, "log_damps": ['d', 'u'], "cal_damps": [], "params": {"A": -1.107322, "B": 5.000234, "C_log1": 0.517709, "W_log1": 7.691155, "PHI_log1": 1.222325, "w_log1": 0.094406, "C_log2": 0.152942, "W_log2": 20.950763, "PHI_log2": -1.185769}, "r2": 0.988061, "sigma": 0.167389},
    "ecfg_2du_1d": {"n_log": 2, "n_cal": 1, "log_damps": ['d', 'u'], "cal_damps": ['d'], "params": {"A": -1.173822, "B": 5.094163, "C_log1": 0.545849, "W_log1": 8.050397, "PHI_log1": 1.055621, "w_log1": 0.106487, "C_log2": 0.091328, "W_log2": 9.138187, "PHI_log2": -3.091285, "C_cal": 0.234260, "W_cal": 1.835499, "PHI_cal": 3.028488, "w_cal": 0.035644}, "r2": 0.990835, "sigma": 0.146662},
    "ecfg_2du_1u": {"n_log": 2, "n_cal": 1, "log_damps": ['d', 'u'], "cal_damps": ['u'], "params": {"A": -1.184042, "B": 5.089882, "C_log1": 0.475240, "W_log1": 7.826162, "PHI_log1": 1.307520, "w_log1": 0.105290, "C_log2": 0.087317, "W_log2": 37.251846, "PHI_log2": 1.824246, "C_cal": 0.212668, "W_cal": 1.856695, "PHI_cal": 2.789706}, "r2": 0.991245, "sigma": 0.143339},
    "ecfg_2du_2dd": {"n_log": 2, "n_cal": 2, "log_damps": ['d', 'u'], "cal_damps": ['d', 'd'], "params": {"A": -1.143409, "B": 5.051343, "C_log1": 0.492493, "W_log1": 7.736223, "PHI_log1": 1.316188, "w_log1": 0.106283, "C_log2": 0.070237, "W_log2": 20.818239, "PHI_log2": -1.554104, "C_cal1": 0.187039, "W_cal1": 1.872012, "PHI_cal1": 2.829064, "w_cal1": 0.032263, "C_cal2": 0.222893, "W_cal2": 10.218835, "PHI_cal2": -0.885112, "w_cal2": 0.173680}, "r2": 0.992594, "sigma": 0.131840},
    "ecfg_2du_2du": {"n_log": 2, "n_cal": 2, "log_damps": ['d', 'u'], "cal_damps": ['d', 'u'], "params": {"A": -1.178750, "B": 5.094947, "C_log1": 0.506504, "W_log1": 7.523443, "PHI_log1": 1.731441, "w_log1": 0.095891, "C_log2": 0.106021, "W_log2": 13.969342, "PHI_log2": -2.894884, "C_cal1": 0.130445, "W_cal1": 3.292003, "PHI_cal1": -2.436098, "w_cal1": 0.046886, "C_cal2": 0.189568, "W_cal2": 1.899933, "PHI_cal2": 2.289202}, "r2": 0.993677, "sigma": 0.121817},
    "ecfg_2du_2uu": {"n_log": 2, "n_cal": 2, "log_damps": ['d', 'u'], "cal_damps": ['u', 'u'], "params": {"A": -1.158558, "B": 5.045381, "C_log1": 0.657231, "W_log1": 6.612252, "PHI_log1": 1.602735, "w_log1": 0.326408, "C_log2": 0.155666, "W_log2": 6.408657, "PHI_log2": -2.786115, "C_cal1": 0.274476, "W_cal1": 1.726724, "PHI_cal1": -1.916335, "C_cal2": 0.105676, "W_cal2": 3.229209, "PHI_cal2": -2.047719}, "r2": 0.991636, "sigma": 0.140104},
    "ecfg_2uu_0": {"n_log": 2, "n_cal": 0, "log_damps": ['u', 'u'], "cal_damps": [], "params": {"A": -1.176194, "B": 5.084631, "C_log1": 0.241518, "W_log1": 7.177743, "PHI_log1": 1.872984, "C_log2": 0.180475, "W_log2": 20.895218, "PHI_log2": -1.115699}, "r2": 0.981087, "sigma": 0.210681},
    "ecfg_2uu_1d": {"n_log": 2, "n_cal": 1, "log_damps": ['u', 'u'], "cal_damps": ['d'], "params": {"A": -1.079911, "B": 4.945744, "C_log1": 0.231030, "W_log1": 9.764666, "PHI_log1": -1.817452, "C_log2": 0.295337, "W_log2": 6.882026, "PHI_log2": 2.220107, "C_cal": 0.368784, "W_cal": 5.567807, "PHI_cal": 2.160540, "w_cal": 0.265830}, "r2": 0.984066, "sigma": 0.193374},
    "ecfg_2uu_1u": {"n_log": 2, "n_cal": 1, "log_damps": ['u', 'u'], "cal_damps": ['u'], "params": {"A": -1.180016, "B": 5.072000, "C_log1": 0.255620, "W_log1": 7.166341, "PHI_log1": 1.599078, "C_log2": 0.167270, "W_log2": 8.684047, "PHI_log2": 0.772801, "C_cal": 0.207624, "W_cal": 1.778754, "PHI_cal": -2.520445}, "r2": 0.989596, "sigma": 0.156259},
    "ecfg_2uu_2dd": {"n_log": 2, "n_cal": 2, "log_damps": ['u', 'u'], "cal_damps": ['d', 'd'], "params": {"A": -1.164083, "B": 5.064852, "C_log1": 0.155317, "W_log1": 7.035065, "PHI_log1": 2.146400, "C_log2": 0.092558, "W_log2": 37.182924, "PHI_log2": 1.919208, "C_cal1": 0.303085, "W_cal1": 1.736738, "PHI_cal1": -2.001712, "w_cal1": 0.037722, "C_cal2": 0.208284, "W_cal2": 2.840344, "PHI_cal2": 0.119747, "w_cal2": 0.101064}, "r2": 0.991799, "sigma": 0.138733},
    "ecfg_2uu_2du": {"n_log": 2, "n_cal": 2, "log_damps": ['u', 'u'], "cal_damps": ['d', 'u'], "params": {"A": -1.120028, "B": 5.006140, "C_log1": 0.284805, "W_log1": 7.059328, "PHI_log1": 1.704487, "C_log2": 0.202732, "W_log2": 8.413918, "PHI_log2": 1.315294, "C_cal1": 0.245751, "W_cal1": 1.777938, "PHI_cal1": -2.491658, "w_cal1": 0.041310, "C_cal2": 0.108161, "W_cal2": 3.299318, "PHI_cal2": -2.642959}, "r2": 0.992489, "sigma": 0.132764},
    "ecfg_2uu_2uu": {"n_log": 2, "n_cal": 2, "log_damps": ['u', 'u'], "cal_damps": ['u', 'u'], "params": {"A": -1.513083, "B": 5.606643, "C_log1": 0.210280, "W_log1": 7.368633, "PHI_log1": 1.781778, "C_log2": 0.238354, "W_log2": 2.000000, "PHI_log2": -1.508758, "C_cal1": 0.255531, "W_cal1": 1.771394, "PHI_cal1": -2.296050, "C_cal2": 0.100712, "W_cal2": 3.200653, "PHI_cal2": -1.579682}, "r2": 0.991137, "sigma": 0.144222},
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
    "cfg_0_0": {"n_log": 0, "n_cal": 0, "log_damps": [], "cal_damps": [], "params": {"A": -1.157034, "B": 5.054441}, "r2": 0.963296, "sigma": 0.293492},
    "cfg_0_1d": {"n_log": 0, "n_cal": 1, "log_damps": [], "cal_damps": ['d'], "params": {"A": -1.211880, "B": 5.109151, "C_cal": 0.276816, "W_cal": 1.766225, "PHI_cal": -2.285958, "D_cal": 0.000000}, "r2": 0.979616, "sigma": 0.218717},
    "cfg_0_1u": {"n_log": 0, "n_cal": 1, "log_damps": [], "cal_damps": ['u'], "params": {"A": -1.211880, "B": 5.109151, "C_cal": 0.276816, "W_cal": 1.766225, "PHI_cal": -2.285958}, "r2": 0.979616, "sigma": 0.218717},
    "cfg_0_2dd": {"n_log": 0, "n_cal": 2, "log_damps": [], "cal_damps": ['d', 'd'], "params": {"A": -1.043529, "B": 4.936476, "C_cal1": 0.734780, "W_cal1": 1.720985, "PHI_cal1": -1.806886, "D_cal1": 0.430257, "C_cal2": 1.354680, "W_cal2": 3.073899, "PHI_cal2": -0.636889, "D_cal2": 1.544632}, "r2": 0.987088, "sigma": 0.174075},
    "cfg_0_2du": {"n_log": 0, "n_cal": 2, "log_damps": [], "cal_damps": ['d', 'u'], "params": {"A": -1.087872, "B": 4.977789, "C_cal1": 0.740856, "W_cal1": 3.058107, "PHI_cal1": -0.383717, "D_cal1": 1.043210, "C_cal2": 0.299708, "W_cal2": 1.731287, "PHI_cal2": -1.910322}, "r2": 0.986069, "sigma": 0.180813},
    "cfg_0_2uu": {"n_log": 0, "n_cal": 2, "log_damps": [], "cal_damps": ['u', 'u'], "params": {"A": -1.186777, "B": 5.081296, "C_cal1": 0.130707, "W_cal1": 3.112609, "PHI_cal1": -0.695070, "C_cal2": 0.282517, "W_cal2": 1.762771, "PHI_cal2": -2.219230}, "r2": 0.983202, "sigma": 0.198549},
    "cfg_1d_0": {"n_log": 1, "n_cal": 0, "log_damps": ['d'], "cal_damps": [], "params": {"A": -1.140887, "B": 5.058520, "C_log": 0.730043, "W_log": 7.494583, "PHI_log": 1.441745, "D_log": 0.598700}, "r2": 0.978542, "sigma": 0.224406},
    "cfg_1d_1d": {"n_log": 1, "n_cal": 1, "log_damps": ['d'], "cal_damps": ['d'], "params": {"A": -1.162145, "B": 5.074806, "C_log": 1.009259, "W_log": 7.357535, "PHI_log": 1.677781, "D_log": 0.757690, "C_cal": 0.861875, "W_cal": 7.851108, "PHI_cal": 2.876003, "D_cal": 1.772680}, "r2": 0.980872, "sigma": 0.211871},
    "cfg_1d_1u": {"n_log": 1, "n_cal": 1, "log_damps": ['d'], "cal_damps": ['u'], "params": {"A": -1.148775, "B": 5.054512, "C_log": 0.691127, "W_log": 7.427131, "PHI_log": 1.448596, "D_log": 0.710778, "C_cal": 0.231917, "W_cal": 1.732922, "PHI_cal": -1.925232}, "r2": 0.989208, "sigma": 0.159143},
    "cfg_1d_2dd": {"n_log": 1, "n_cal": 2, "log_damps": ['d'], "cal_damps": ['d', 'd'], "params": {"A": -1.117823, "B": 5.022361, "C_log": 0.859059, "W_log": 7.571008, "PHI_log": 1.190604, "D_log": 0.837448, "C_cal1": 0.235980, "W_cal1": 1.724592, "PHI_cal1": -1.818065, "D_cal1": 0.000000, "C_cal2": 0.527732, "W_cal2": 10.326154, "PHI_cal2": -1.284730, "D_cal2": 1.216465}, "r2": 0.991446, "sigma": 0.141690},
    "cfg_1d_2du": {"n_log": 1, "n_cal": 2, "log_damps": ['d'], "cal_damps": ['d', 'u'], "params": {"A": -1.143903, "B": 5.052580, "C_log": 0.750230, "W_log": 7.395362, "PHI_log": 1.615819, "D_log": 0.748869, "C_cal1": 0.229576, "W_cal1": 1.744692, "PHI_cal1": -2.056638, "D_cal1": 0.000000, "C_cal2": 0.105778, "W_cal2": 3.282688, "PHI_cal2": -2.477933}, "r2": 0.991369, "sigma": 0.142323},
    "cfg_1d_2uu": {"n_log": 1, "n_cal": 2, "log_damps": ['d'], "cal_damps": ['u', 'u'], "params": {"A": -1.143903, "B": 5.052580, "C_log": 0.750230, "W_log": 7.395363, "PHI_log": 1.615818, "D_log": 0.748869, "C_cal1": 0.229576, "W_cal1": 1.744692, "PHI_cal1": -2.056638, "C_cal2": 0.105778, "W_cal2": 3.282688, "PHI_cal2": -2.477935}, "r2": 0.991369, "sigma": 0.142323},
    "cfg_1u_0": {"n_log": 1, "n_cal": 0, "log_damps": ['u'], "cal_damps": [], "params": {"A": -1.214560, "B": 5.152404, "C_log": 0.229711, "W_log": 7.522710, "PHI_log": 1.436538}, "r2": 0.974604, "sigma": 0.244132},
    "cfg_1u_1d": {"n_log": 1, "n_cal": 1, "log_damps": ['u'], "cal_damps": ['d'], "params": {"A": -1.245311, "B": 5.171564, "C_log": 0.182751, "W_log": 7.438856, "PHI_log": 1.551007, "C_cal": 0.241867, "W_cal": 1.760525, "PHI_cal": -2.258807, "D_cal": 0.000000}, "r2": 0.986575, "sigma": 0.177500},
    "cfg_1u_1u": {"n_log": 1, "n_cal": 1, "log_damps": ['u'], "cal_damps": ['u'], "params": {"A": -1.245311, "B": 5.171564, "C_log": 0.182751, "W_log": 7.438856, "PHI_log": 1.551007, "C_cal": 0.241867, "W_cal": 1.760525, "PHI_cal": -2.258808}, "r2": 0.986575, "sigma": 0.177500},
    "cfg_1u_2dd": {"n_log": 1, "n_cal": 2, "log_damps": ['u'], "cal_damps": ['d', 'd'], "params": {"A": -0.997270, "B": 4.856964, "C_log": 0.158282, "W_log": 5.161217, "PHI_log": -0.026973, "C_cal1": 0.952993, "W_cal1": 1.728154, "PHI_cal1": -1.974739, "D_cal1": 0.550085, "C_cal2": 1.881161, "W_cal2": 3.064227, "PHI_cal2": -0.630168, "D_cal2": 1.778389}, "r2": 0.990507, "sigma": 0.149259},
    "cfg_1u_2du": {"n_log": 1, "n_cal": 2, "log_damps": ['u'], "cal_damps": ['d', 'u'], "params": {"A": -1.212994, "B": 5.130582, "C_log": 0.170576, "W_log": 7.260411, "PHI_log": 1.861030, "C_cal1": 0.249861, "W_cal1": 1.755341, "PHI_cal1": -2.188682, "D_cal1": 0.000000, "C_cal2": 0.112888, "W_cal2": 3.137894, "PHI_cal2": -1.028343}, "r2": 0.989122, "sigma": 0.159781},
    "cfg_1u_2uu": {"n_log": 1, "n_cal": 2, "log_damps": ['u'], "cal_damps": ['u', 'u'], "params": {"A": -1.212994, "B": 5.130582, "C_log": 0.170576, "W_log": 7.260412, "PHI_log": 1.861029, "C_cal1": 0.249861, "W_cal1": 1.755341, "PHI_cal1": -2.188681, "C_cal2": 0.112888, "W_cal2": 3.137893, "PHI_cal2": -1.028342}, "r2": 0.989122, "sigma": 0.159781},
    "cfg_2dd_0": {"n_log": 2, "n_cal": 0, "log_damps": ['d', 'd'], "cal_damps": [], "params": {"A": -1.123170, "B": 5.025582, "C_log1": 0.173234, "W_log1": 20.961580, "PHI_log1": -1.252121, "D_log1": 0.010000, "C_log2": 0.698534, "W_log2": 7.342602, "PHI_log2": 1.621463, "D_log2": 0.551927}, "r2": 0.984401, "sigma": 0.191334},
    "cfg_2dd_1d": {"n_log": 2, "n_cal": 1, "log_damps": ['d', 'd'], "cal_damps": ['d'], "params": {"A": -1.147162, "B": 5.049628, "C_log1": 0.436147, "W_log1": 7.550082, "PHI_log1": 1.239835, "D_log1": 0.561340, "C_log2": 0.737461, "W_log2": 3.619379, "PHI_log2": 2.719225, "D_log2": 1.193088, "C_cal": 0.600715, "W_cal": 1.729934, "PHI_cal": -1.889899, "D_cal": 0.373035}, "r2": 0.990441, "sigma": 0.149779},
    "cfg_2dd_1u": {"n_log": 2, "n_cal": 1, "log_damps": ['d', 'd'], "cal_damps": ['u'], "params": {"A": -1.118049, "B": 5.022015, "C_log1": 0.411283, "W_log1": 15.982033, "PHI_log1": 1.899032, "D_log1": 1.008117, "C_log2": 0.771300, "W_log2": 7.494727, "PHI_log2": 1.281440, "D_log2": 0.782566, "C_cal": 0.252634, "W_cal": 1.719544, "PHI_cal": -1.748519}, "r2": 0.991076, "sigma": 0.144720},
    "cfg_2dd_2dd": {"n_log": 2, "n_cal": 2, "log_damps": ['d', 'd'], "cal_damps": ['d', 'd'], "params": {"A": -1.107543, "B": 5.013261, "C_log1": 0.867743, "W_log1": 7.564553, "PHI_log1": 1.185286, "D_log1": 0.836456, "C_log2": 0.080231, "W_log2": 31.168526, "PHI_log2": -2.556200, "D_log2": 0.010000, "C_cal1": 0.227427, "W_cal1": 1.700399, "PHI_cal1": -1.645623, "D_cal1": 0.000000, "C_cal2": 0.539153, "W_cal2": 10.306927, "PHI_cal2": -1.316656, "D_cal2": 1.265284}, "r2": 0.992586, "sigma": 0.131907},
    "cfg_2dd_2du": {"n_log": 2, "n_cal": 2, "log_damps": ['d', 'd'], "cal_damps": ['d', 'u'], "params": {"A": -1.124488, "B": 5.034328, "C_log1": 0.911183, "W_log1": 7.535859, "PHI_log1": 1.329053, "D_log1": 0.850695, "C_log2": 0.461323, "W_log2": 16.273503, "PHI_log2": 1.868991, "D_log2": 1.288074, "C_cal1": 0.233429, "W_cal1": 1.733712, "PHI_cal1": -1.898175, "D_cal1": 0.000000, "C_cal2": 0.099007, "W_cal2": 3.367809, "PHI_cal2": 2.930030}, "r2": 0.992761, "sigma": 0.130343},
    "cfg_2dd_2uu": {"n_log": 2, "n_cal": 2, "log_damps": ['d', 'd'], "cal_damps": ['u', 'u'], "params": {"A": -1.137227, "B": 5.043742, "C_log1": 0.125788, "W_log1": 36.880010, "PHI_log1": 2.278761, "D_log1": 0.286135, "C_log2": 0.785690, "W_log2": 7.400371, "PHI_log2": 1.558453, "D_log2": 0.776123, "C_cal1": 0.089185, "W_cal1": 3.341074, "PHI_cal1": -2.834937, "C_cal2": 0.234500, "W_cal2": 1.734463, "PHI_cal2": -1.966046}, "r2": 0.992386, "sigma": 0.133676},
    "cfg_2du_0": {"n_log": 2, "n_cal": 0, "log_damps": ['d', 'u'], "cal_damps": [], "params": {"A": -1.176638, "B": 5.085312, "C_log1": 0.183641, "W_log1": 20.894887, "PHI_log1": -1.114996, "D_log1": 0.010000, "C_log2": 0.241331, "W_log2": 7.179184, "PHI_log2": 1.871440}, "r2": 0.981056, "sigma": 0.210852},
    "cfg_2du_1d": {"n_log": 2, "n_cal": 1, "log_damps": ['d', 'u'], "cal_damps": ['d'], "params": {"A": -1.180281, "B": 5.072318, "C_log1": 0.169963, "W_log1": 8.684287, "PHI_log1": 0.762940, "D_log1": 0.010000, "C_log2": 0.253902, "W_log2": 7.160450, "PHI_log2": 1.609210, "C_cal": 0.207659, "W_cal": 1.779010, "PHI_cal": -2.523333, "D_cal": 0.000000}, "r2": 0.989594, "sigma": 0.156271},
    "cfg_2du_1u": {"n_log": 2, "n_cal": 1, "log_damps": ['d', 'u'], "cal_damps": ['u'], "params": {"A": -1.142149, "B": 5.044196, "C_log1": 0.692455, "W_log1": 7.409711, "PHI_log1": 1.450783, "D_log1": 0.716222, "C_log2": 0.093157, "W_log2": 37.117641, "PHI_log2": 2.004310, "C_cal": 0.244148, "W_cal": 1.733687, "PHI_cal": -1.923785}, "r2": 0.991047, "sigma": 0.144954},
    "cfg_2du_2dd": {"n_log": 2, "n_cal": 2, "log_damps": ['d', 'u'], "cal_damps": ['d', 'd'], "params": {"A": -1.118964, "B": 5.023027, "C_log1": 0.794230, "W_log1": 7.291374, "PHI_log1": 1.821260, "D_log1": 0.775453, "C_log2": 0.074095, "W_log2": 37.085360, "PHI_log2": 2.062923, "C_cal1": 0.311380, "W_cal1": 3.274908, "PHI_cal1": -2.367894, "D_cal1": 0.631669, "C_cal2": 0.244574, "W_cal2": 1.747934, "PHI_cal2": -2.108679, "D_cal2": 0.000000}, "r2": 0.992492, "sigma": 0.132740},
    "cfg_2du_2du": {"n_log": 2, "n_cal": 2, "log_damps": ['d', 'u'], "cal_damps": ['d', 'u'], "params": {"A": -1.099844, "B": 4.995303, "C_log1": 0.216041, "W_log1": 15.479657, "PHI_log1": 2.042643, "D_log1": 0.686561, "C_log2": 0.150705, "W_log2": 6.932695, "PHI_log2": 2.359407, "C_cal1": 0.532472, "W_cal1": 3.107680, "PHI_cal1": -0.769203, "D_cal1": 0.898874, "C_cal2": 0.286132, "W_cal2": 1.721100, "PHI_cal2": -1.884138}, "r2": 0.991109, "sigma": 0.144448},
    "cfg_2du_2uu": {"n_log": 2, "n_cal": 2, "log_damps": ['d', 'u'], "cal_damps": ['u', 'u'], "params": {"A": -1.136659, "B": 5.042504, "C_log1": 0.787289, "W_log1": 7.392779, "PHI_log1": 1.561974, "D_log1": 0.778073, "C_log2": 0.074091, "W_log2": 36.881293, "PHI_log2": 2.262460, "C_cal1": 0.087574, "W_cal1": 3.352130, "PHI_cal1": -2.886965, "C_cal2": 0.236362, "W_cal2": 1.732922, "PHI_cal2": -1.953413}, "r2": 0.992344, "sigma": 0.134043},
    "cfg_2uu_0": {"n_log": 2, "n_cal": 0, "log_damps": ['u', 'u'], "cal_damps": [], "params": {"A": -1.107312, "B": 4.975982, "C_log1": 0.307643, "W_log1": 6.832722, "PHI_log1": 2.191370, "C_log2": 0.221748, "W_log2": 9.078811, "PHI_log2": -0.389220}, "r2": 0.982488, "sigma": 0.202726},
    "cfg_2uu_1d": {"n_log": 2, "n_cal": 1, "log_damps": ['u', 'u'], "cal_damps": ['d'], "params": {"A": -1.246428, "B": 5.172096, "C_log1": 0.185667, "W_log1": 7.400972, "PHI_log1": 1.599057, "C_log2": 0.088936, "W_log2": 20.297545, "PHI_log2": -0.458549, "C_cal": 0.220948, "W_cal": 1.795904, "PHI_cal": -2.414062, "D_cal": 0.000000}, "r2": 0.987612, "sigma": 0.170510},
    "cfg_2uu_1u": {"n_log": 2, "n_cal": 1, "log_damps": ['u', 'u'], "cal_damps": ['u'], "params": {"A": -1.180016, "B": 5.072000, "C_log1": 0.255621, "W_log1": 7.166343, "PHI_log1": 1.599074, "C_log2": 0.167270, "W_log2": 8.684045, "PHI_log2": 0.772807, "C_cal": 0.207624, "W_cal": 1.778753, "PHI_cal": -2.520444}, "r2": 0.989596, "sigma": 0.156259},
    "cfg_2uu_2dd": {"n_log": 2, "n_cal": 2, "log_damps": ['u', 'u'], "cal_damps": ['d', 'd'], "params": {"A": -0.999772, "B": 4.857211, "C_log1": 0.162047, "W_log1": 5.229864, "PHI_log1": -0.169094, "C_log2": 0.093381, "W_log2": 37.214097, "PHI_log2": 1.906477, "C_cal1": 0.850857, "W_cal1": 1.732514, "PHI_cal1": -2.003964, "D_cal1": 0.476422, "C_cal2": 1.748256, "W_cal2": 2.996382, "PHI_cal2": -0.502756, "D_cal2": 1.723574}, "r2": 0.992202, "sigma": 0.135278},
    "cfg_2uu_2du": {"n_log": 2, "n_cal": 2, "log_damps": ['u', 'u'], "cal_damps": ['d', 'u'], "params": {"A": -1.164997, "B": 5.056126, "C_log1": 0.199707, "W_log1": 8.472375, "PHI_log1": 1.253162, "C_log2": 0.281319, "W_log2": 7.102784, "PHI_log2": 1.700880, "C_cal1": 0.208220, "W_cal1": 1.793016, "PHI_cal1": -2.673696, "D_cal1": 0.000000, "C_cal2": 0.110230, "W_cal2": 3.281415, "PHI_cal2": -2.510872}, "r2": 0.991897, "sigma": 0.137904},
    "cfg_2uu_2uu": {"n_log": 2, "n_cal": 2, "log_damps": ['u', 'u'], "cal_damps": ['u', 'u'], "params": {"A": -1.513083, "B": 5.606643, "C_log1": 0.210280, "W_log1": 7.368633, "PHI_log1": 1.781778, "C_log2": 0.238353, "W_log2": 2.000000, "PHI_log2": -1.508758, "C_cal1": 0.100712, "W_cal1": 3.200653, "PHI_cal1": -1.579681, "C_cal2": 0.255531, "W_cal2": 1.771394, "PHI_cal2": -2.296050}, "r2": 0.991137, "sigma": 0.144222},
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


