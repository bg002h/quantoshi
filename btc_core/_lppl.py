"""LPPL family: 10 LPPL variants (1–4 freqs, weighted/unweighted,
no-omega=13 ablations) + LinPPL.

All classes share _ShrinkingBandsMixin for the band-fitting machinery.
The 1-freq LPPLModel is the base; every other class ultimately derives from it.
"""

import numpy as np

from btc_core._helpers import (
    _lazy_norm, _lazy_linregress, _DEFAULT_QS,
)
from btc_core._base import _ShrinkingBandsMixin
from time_basis import T_MIN


class LPPLModel(_ShrinkingBandsMixin):
    """Damped Log-Periodic Power Law model with shrinking Gaussian quantile bands.

    Fits: log10(price) = A + B*log10(t) + C*t^(-d)*cos(w*ln(t) + phi)

    The oscillatory term captures Bitcoin's ~4-year bubble cycles in log-time,
    with damping (d > 0) reflecting decreasing volatility over time. Quantile
    bands use σ(t) = σ₀·t^(-α) — wider in early era, narrower in late era.
    """
    name = "LPPL"
    short_name = "lppl"
    legend_name = "LPPL"
    dash_style = "dashdot"
    quantized = True

    # Best-fit parameters from differential evolution on full BTC history
    # (genesis = 2009-07-25)
    _A   = -1.142474
    _B   =                                   5.061067
    _C   =                                   0.730627
    _W   =                                   7.502512
    _PHI =                                   1.433610
    _D   =                                   0.599894

    def __init__(self, price_years, price_prices, quantiles):
        mask = price_years >= T_MIN
        t = price_years[mask]
        lp = np.log10(price_prices[mask])
        residuals = lp - self._lppl_log10(t)
        self._init_shrinking_bands(t, residuals, quantiles)
        self._build_colors()

    def _model_log10(self, t):
        """Alias for mixin compatibility."""
        return self._lppl_log10(t)

    def _lppl_log10(self, t):
        """Evaluate damped LPPL median in log10 space."""
        t = np.asarray(t, float)
        t_safe = np.maximum(t, 0.1)
        envelope = self._C * t_safe ** (-self._D)
        return self._A + self._B * np.log10(t_safe) + envelope * np.cos(
            self._W * np.log(t_safe) + self._PHI)

    component_names = [
        "A (constant)",
        "B\u00b7log\u2081\u2080(t)",
        "damped osc (\u03c9_log)",
    ]
    # Components that together define the "support" (trend) line.
    # Used by reference/cumulative decomposition rendering modes.
    support_component_names = ["A (constant)", "B\u00b7log\u2081\u2080(t)"]
    formula_log10_latex = (
        r"A + B \log_{10}(t) + C \cdot t^{-D} \cos(\omega \ln t + \varphi)"
    )
    formula_product_latex = (
        r"10^A \cdot t^B \cdot 10^{\,C \cdot t^{-D} \cos(\omega \ln t + \varphi)}"
    )
    # (plain-text formula, [(param_name, attr_name), ...])
    component_details = {
        "A (constant)":           ("A",                         [("A", "_A")]),
        "B\u00b7log\u2081\u2080(t)": ("B\u00b7log\u2081\u2080(t)", [("B", "_B")]),
        "damped osc (\u03c9_log)": ("C\u00b7t^(-D)\u00b7cos(\u03c9\u00b7ln(t)+\u03c6)",
                                    [("C", "_C"), ("D", "_D"),
                                     ("\u03c9", "_W"), ("\u03c6", "_PHI")]),
    }

    def components(self, t):
        """Additive terms in log10 space. sum(values) == _lppl_log10(t)."""
        t = np.asarray(t, float)
        t_safe = np.maximum(t, 0.1)
        return {
            "A (constant)":            np.full_like(t_safe, self._A),
            "B\u00b7log\u2081\u2080(t)": self._B * np.log10(t_safe),
            "damped osc (\u03c9_log)":  self._C * t_safe ** (-self._D) * np.cos(
                self._W * np.log(t_safe) + self._PHI),
        }

    # price_at, interp_price, find_percentile inherited from _ShrinkingBandsMixin

    def _build_colors(self):
        """Green/teal palette — visually distinct from Bubble and PL."""
        self.colors = {}
        n = len(self.quantiles)
        for i, q in enumerate(self.quantiles):
            frac = i / max(n - 1, 1)
            r = int(20 + 80 * frac)     # 20 → 100
            g = int(160 + 60 * frac)    # 160 → 220
            b = int(120 + 40 * frac)    # 120 → 160
            self.colors[q] = f"#{r:02x}{g:02x}{b:02x}"


class LPPL2Model(LPPLModel):
    """Two-frequency LPPL with independent oscillation frequencies.

    Fits: log10(price) = A + B*log10(t) + C1*t^(-D)*cos(W1*ln(t)+φ1) + C2*cos(W2*ln(t)+φ2)

    Primary oscillation (W1) is damped by t^(-D); secondary oscillation (W2)
    is undamped — it persists as a permanent structural feature. W2 is
    independent of W1 (not constrained to 2×W1).
    """
    name = "LPPL\u2082"
    short_name = "lp2"
    legend_name = "LPPL\u2082"
    dash_style = "dashdot"

    # All 9 params jointly fitted by tools/fit_lppl2.py
    _A   = -1.123081
    _B   =                                      5.025490
    _C   =                                      0.698589
    _W   =                                      7.343171
    _PHI =                                      1.620479
    _D   =                                      0.551888
    _C2  =                                      0.170118
    _W2  =                          20.960875
    _PHI2 = -1.251115

    def _lppl_log10(self, t):
        """Evaluate two-frequency LPPL median in log10 space.

        Primary oscillation is damped; secondary oscillation is undamped.
        """
        t = np.asarray(t, float)
        t_safe = np.maximum(t, 0.1)
        term1 = self._C * t_safe ** (-self._D) * np.cos(self._W * np.log(t_safe) + self._PHI)
        term2 = self._C2 * np.cos(self._W2 * np.log(t_safe) + self._PHI2)
        return self._A + self._B * np.log10(t_safe) + term1 + term2

    component_names = [
        "A (constant)",
        "B\u00b7log\u2081\u2080(t)",
        "damped osc (\u03c9\u2081)",
        "undamped osc (\u03c9\u2082)",
    ]
    formula_log10_latex = (
        r"A + B \log_{10}(t) + C_1 t^{-D} \cos(\omega_1 \ln t + \varphi_1)"
        r" + C_2 \cos(\omega_2 \ln t + \varphi_2)"
    )
    formula_product_latex = (
        r"10^A \cdot t^B"
        r" \cdot 10^{\,C_1 t^{-D} \cos(\omega_1 \ln t + \varphi_1)}"
        r" \cdot 10^{\,C_2 \cos(\omega_2 \ln t + \varphi_2)}"
    )
    component_details = {
        "A (constant)":           ("A",                         [("A", "_A")]),
        "B\u00b7log\u2081\u2080(t)": ("B\u00b7log\u2081\u2080(t)", [("B", "_B")]),
        "damped osc (\u03c9\u2081)": ("C\u2081\u00b7t^(-D)\u00b7cos(\u03c9\u2081\u00b7ln(t)+\u03c6\u2081)",
                                      [("C\u2081", "_C"), ("D", "_D"),
                                       ("\u03c9\u2081", "_W"), ("\u03c6\u2081", "_PHI")]),
        "undamped osc (\u03c9\u2082)": ("C\u2082\u00b7cos(\u03c9\u2082\u00b7ln(t)+\u03c6\u2082)",
                                        [("C\u2082", "_C2"), ("\u03c9\u2082", "_W2"),
                                         ("\u03c6\u2082", "_PHI2")]),
    }

    def components(self, t):
        t = np.asarray(t, float)
        t_safe = np.maximum(t, 0.1)
        return {
            "A (constant)":                np.full_like(t_safe, self._A),
            "B\u00b7log\u2081\u2080(t)":    self._B * np.log10(t_safe),
            "damped osc (\u03c9\u2081)":    self._C * t_safe ** (-self._D) * np.cos(
                self._W * np.log(t_safe) + self._PHI),
            "undamped osc (\u03c9\u2082)":  self._C2 * np.cos(
                self._W2 * np.log(t_safe) + self._PHI2),
        }


class LPPL3Model(LPPL2Model):
    """Three-frequency LPPL: damped primary + two undamped secondary oscillations.

    Fits: log10(price) = A + B*log10(t) + C1*t^(-D)*cos(W1*ln(t)+φ1)
                       + C2*cos(W2*ln(t)+φ2) + C3*cos(W3*ln(t)+φ3)

    Initial search started at W3=13.3 (from FFT residual analysis), but
    the optimizer settled at W3≈10 — the peak at ω≈13 turned out to be
    an intermodulation product (W2 − W1 ≈ 13.5), not a third oscillation.
    W3/W1 ≈ √2, W3/W2 ≈ 0.49.
    """
    name = "LPPL\u2083"
    short_name = "lp3"
    legend_name = "LPPL\u2083"
    dash_style = "dashdot"

    # All 12 params jointly fitted by tools/fit_lppl3.py
    _A   = -1.092318
    _B   =                                4.962643
    _C   =                                0.605321
    _W   =                                7.126016
    _PHI =                                1.885918
    _D   =                                0.351888
    _C2  =                                0.179585
    _W2  =                      20.828065
    _PHI2 = -1.032021
    _C3  =                                0.172767
    _W3  =                      10.109899
    _PHI3 = -2.203281

    def _lppl_log10(self, t):
        """Evaluate three-frequency LPPL median in log10 space."""
        t = np.asarray(t, float)
        t_safe = np.maximum(t, 0.1)
        term1 = self._C * t_safe ** (-self._D) * np.cos(self._W * np.log(t_safe) + self._PHI)
        term2 = self._C2 * np.cos(self._W2 * np.log(t_safe) + self._PHI2)
        term3 = self._C3 * np.cos(self._W3 * np.log(t_safe) + self._PHI3)
        return self._A + self._B * np.log10(t_safe) + term1 + term2 + term3

    component_names = [
        "A (constant)",
        "B\u00b7log\u2081\u2080(t)",
        "damped osc (\u03c9\u2081)",
        "undamped osc (\u03c9\u2082)",
        "undamped osc (\u03c9\u2083)",
    ]
    formula_log10_latex = (
        r"A + B \log_{10}(t) + C_1 t^{-D} \cos(\omega_1 \ln t + \varphi_1)"
        r" + C_2 \cos(\omega_2 \ln t + \varphi_2)"
        r" + C_3 \cos(\omega_3 \ln t + \varphi_3)"
    )
    formula_product_latex = (
        r"10^A \cdot t^B"
        r" \cdot 10^{\,C_1 t^{-D} \cos(\omega_1 \ln t + \varphi_1)}"
        r" \cdot 10^{\,C_2 \cos(\omega_2 \ln t + \varphi_2)}"
        r" \cdot 10^{\,C_3 \cos(\omega_3 \ln t + \varphi_3)}"
    )
    component_details = {
        "A (constant)":           ("A",                         [("A", "_A")]),
        "B\u00b7log\u2081\u2080(t)": ("B\u00b7log\u2081\u2080(t)", [("B", "_B")]),
        "damped osc (\u03c9\u2081)": ("C\u2081\u00b7t^(-D)\u00b7cos(\u03c9\u2081\u00b7ln(t)+\u03c6\u2081)",
                                      [("C\u2081", "_C"), ("D", "_D"),
                                       ("\u03c9\u2081", "_W"), ("\u03c6\u2081", "_PHI")]),
        "undamped osc (\u03c9\u2082)": ("C\u2082\u00b7cos(\u03c9\u2082\u00b7ln(t)+\u03c6\u2082)",
                                        [("C\u2082", "_C2"), ("\u03c9\u2082", "_W2"),
                                         ("\u03c6\u2082", "_PHI2")]),
        "undamped osc (\u03c9\u2083)": ("C\u2083\u00b7cos(\u03c9\u2083\u00b7ln(t)+\u03c6\u2083)",
                                        [("C\u2083", "_C3"), ("\u03c9\u2083", "_W3"),
                                         ("\u03c6\u2083", "_PHI3")]),
    }

    def components(self, t):
        t = np.asarray(t, float)
        t_safe = np.maximum(t, 0.1)
        return {
            "A (constant)":                np.full_like(t_safe, self._A),
            "B\u00b7log\u2081\u2080(t)":    self._B * np.log10(t_safe),
            "damped osc (\u03c9\u2081)":    self._C * t_safe ** (-self._D) * np.cos(
                self._W * np.log(t_safe) + self._PHI),
            "undamped osc (\u03c9\u2082)":  self._C2 * np.cos(
                self._W2 * np.log(t_safe) + self._PHI2),
            "undamped osc (\u03c9\u2083)":  self._C3 * np.cos(
                self._W3 * np.log(t_safe) + self._PHI3),
        }


class LPPLModelW(LPPLModel):
    """LPPL with log-time-uniform (1/t) weighting.

    Same formula as LPPLModel, but fitted with residuals weighted by 1/t
    so each unit of ln(t) contributes equally to the loss. Emphasizes
    early-history structure over the 2020-2024 bubble era.
    """
    name = "LPPL (weighted)"
    short_name = "lppl_w"
    legend_name = "LPPL\u1d65\u1d65"
    _A   = -1.086405
    _B   =                  4.986164
    _C   =                  0.552357
    _W   =                  7.413713
    _PHI =                  1.534947
    _D   =                  0.356071


class LPPL2ModelW(LPPL2Model):
    """LPPL\u2082 with log-time-uniform weighting."""
    name = "LPPL\u2082 (weighted)"
    short_name = "lp2_w"
    legend_name = "LPPL\u2082\u1d65\u1d65"
    _A   = -1.113845
    _B   =                  4.985830
    _C   =                  0.218055
    _W   =                8.781733
    _PHI =                  0.042550
    _D   =                  0.000000
    _C2  =                  0.306948
    _W2  =                  6.687838
    _PHI2 =                2.350981


class LPPL3ModelW(LPPL3Model):
    """LPPL\u2083 with log-time-uniform weighting."""
    name = "LPPL\u2083 (weighted)"
    short_name = "lp3_w"
    legend_name = "LPPL\u2083\u1d65\u1d65"
    _A   = -1.097320
    _B   =                  4.998266
    _C   =                  0.567157
    _W   =                7.363708
    _PHI =                  1.629461
    _D   =                  0.370323
    _C2  =                  0.146603
    _W2  =               20.887663
    _PHI2 =          -1.077466
    _C3  =                  0.140480
    _W3  =                13.359291
    _PHI3 =        -2.626026


class LPPL4Model(LPPL3Model):
    """Four-frequency LPPL: damped primary + three undamped secondary oscillations.

    Fits: log10(price) = A + B*log10(t) + C1*t^(-D)*cos(W1*ln(t)+φ1)
                       + C2*cos(W2*ln(t)+φ2) + C3*cos(W3*ln(t)+φ3) + C4*cos(W4*ln(t)+φ4)

    CAUTION: The 4th frequency is NOT stable under log-time weighting.
    Unweighted fits find W2≈13 (close to 2×W1, looks like 2nd harmonic)
    but weighted fits find ~17 (ratio 2.5, non-harmonic). Use with awareness
    that one of the 4 frequencies may be a recent-era artifact. See the
    "LPPL Weighting & Regime Shifts" section for details.
    """
    name = "LPPL\u2084"
    short_name = "lp4"
    legend_name = "LPPL\u2084"
    dash_style = "dashdot"

    # All 15 params jointly fitted by tools/fit_lppl4.py (unweighted)
    _A   = -1.108612
    _B   =                                4.971057
    _C   =                                0.104254
    _W   =                             31.017350
    _PHI =                       -2.562749
    _D   =                                0.050000
    _C2  =                                0.169132
    _W2  =                          20.673474
    _PHI2 =       -0.734225
    _C3  =                                0.209504
    _W3  =                          9.440425
    _PHI3 =          -0.960305
    _C4  =                                0.325684
    _W4  =                        6.929980
    _PHI4 =       2.121904

    def _lppl_log10(self, t):
        """Evaluate four-frequency LPPL median in log10 space."""
        t = np.asarray(t, float)
        t_safe = np.maximum(t, 0.1)
        term1 = self._C * t_safe ** (-self._D) * np.cos(self._W * np.log(t_safe) + self._PHI)
        term2 = self._C2 * np.cos(self._W2 * np.log(t_safe) + self._PHI2)
        term3 = self._C3 * np.cos(self._W3 * np.log(t_safe) + self._PHI3)
        term4 = self._C4 * np.cos(self._W4 * np.log(t_safe) + self._PHI4)
        return self._A + self._B * np.log10(t_safe) + term1 + term2 + term3 + term4

    component_names = [
        "A (constant)",
        "B\u00b7log\u2081\u2080(t)",
        "damped osc (\u03c9\u2081)",
        "undamped osc (\u03c9\u2082)",
        "undamped osc (\u03c9\u2083)",
        "undamped osc (\u03c9\u2084)",
    ]
    formula_log10_latex = (
        r"A + B \log_{10}(t) + C_1 t^{-D} \cos(\omega_1 \ln t + \varphi_1)"
        r" + C_2 \cos(\omega_2 \ln t + \varphi_2)"
        r" + C_3 \cos(\omega_3 \ln t + \varphi_3)"
        r" + C_4 \cos(\omega_4 \ln t + \varphi_4)"
    )
    formula_product_latex = (
        r"10^A \cdot t^B"
        r" \cdot 10^{\,C_1 t^{-D} \cos(\omega_1 \ln t + \varphi_1)}"
        r" \cdot 10^{\,C_2 \cos(\omega_2 \ln t + \varphi_2)}"
        r" \cdot 10^{\,C_3 \cos(\omega_3 \ln t + \varphi_3)}"
        r" \cdot 10^{\,C_4 \cos(\omega_4 \ln t + \varphi_4)}"
    )
    component_details = {
        "A (constant)":           ("A",                         [("A", "_A")]),
        "B\u00b7log\u2081\u2080(t)": ("B\u00b7log\u2081\u2080(t)", [("B", "_B")]),
        "damped osc (\u03c9\u2081)": ("C\u2081\u00b7t^(-D)\u00b7cos(\u03c9\u2081\u00b7ln(t)+\u03c6\u2081)",
                                      [("C\u2081", "_C"), ("D", "_D"),
                                       ("\u03c9\u2081", "_W"), ("\u03c6\u2081", "_PHI")]),
        "undamped osc (\u03c9\u2082)": ("C\u2082\u00b7cos(\u03c9\u2082\u00b7ln(t)+\u03c6\u2082)",
                                        [("C\u2082", "_C2"), ("\u03c9\u2082", "_W2"),
                                         ("\u03c6\u2082", "_PHI2")]),
        "undamped osc (\u03c9\u2083)": ("C\u2083\u00b7cos(\u03c9\u2083\u00b7ln(t)+\u03c6\u2083)",
                                        [("C\u2083", "_C3"), ("\u03c9\u2083", "_W3"),
                                         ("\u03c6\u2083", "_PHI3")]),
        "undamped osc (\u03c9\u2084)": ("C\u2084\u00b7cos(\u03c9\u2084\u00b7ln(t)+\u03c6\u2084)",
                                        [("C\u2084", "_C4"), ("\u03c9\u2084", "_W4"),
                                         ("\u03c6\u2084", "_PHI4")]),
    }

    def components(self, t):
        t = np.asarray(t, float)
        t_safe = np.maximum(t, 0.1)
        return {
            "A (constant)":                np.full_like(t_safe, self._A),
            "B\u00b7log\u2081\u2080(t)":    self._B * np.log10(t_safe),
            "damped osc (\u03c9\u2081)":    self._C * t_safe ** (-self._D) * np.cos(
                self._W * np.log(t_safe) + self._PHI),
            "undamped osc (\u03c9\u2082)":  self._C2 * np.cos(
                self._W2 * np.log(t_safe) + self._PHI2),
            "undamped osc (\u03c9\u2083)":  self._C3 * np.cos(
                self._W3 * np.log(t_safe) + self._PHI3),
            "undamped osc (\u03c9\u2084)":  self._C4 * np.cos(
                self._W4 * np.log(t_safe) + self._PHI4),
        }


class LPPL4ModelW(LPPL4Model):
    """LPPL\u2084 with log-time-uniform weighting."""
    name = "LPPL\u2084 (weighted)"
    short_name = "lp4_w"
    legend_name = "LPPL\u2084\u1d65\u1d65"
    _A   = -1.115466
    _B   =                  4.993713
    _C   =                  0.189559
    _W   =               13.708773
    _PHI =            -2.964215
    _D   =                  0.227009
    _C2  =                  0.275109
    _W2  =                6.562995
    _PHI2 =          2.410944
    _C3  =                  0.137470
    _W3  =             20.876036
    _PHI3 = -1.053513
    _C4  =                  0.240792
    _W4  =                8.427213
    _PHI4 =           0.562557


class LPPL4ModelN13(LPPL4Model):
    """LPPL\u2084 with ω=13 intermod band excluded (secondary frequencies constrained to avoid 11.5-14.5).

    Secondaries at ω≈9.9, 17.5, 20.9 (ratios 1.41, 2.48, 2.96). R²=0.9905.
    """
    name = "LPPL\u2084 (no \u03c9\u224813)"
    short_name = "lp4_n13"
    legend_name = "LPPL\u2084-n13"
    _A   = -1.110160
    _B   =                  4.973231
    _C   =                  0.228744
    _W   =                 9.386961
    _PHI =         -0.877716
    _D   =                  0.050000
    _C2  =                  0.168195
    _W2  =             20.665975
    _PHI2 =    -0.717718
    _C3  =                  0.095346
    _W3  =              31.057791
    _PHI3 =    -2.643874
    _C4  =                  0.320054
    _W4  =              6.902560
    _PHI4 =        2.162170


class LPPL4ModelWN13(LPPL4ModelN13):
    """LPPL\u2084 weighted + ω=13 excluded.

    Secondaries at ω≈9.3, 17.1, 20.9 (same as LP4 weighted, since
    LP4 weighted's secondaries already avoid the 11.5-14.5 band).
    """
    name = "LPPL\u2084 (weighted, no \u03c9\u224813)"
    short_name = "lp4_w_n13"
    legend_name = "LPPL\u2084\u1d65\u1d65-n13"
    _A   = -1.095260
    _B   =                  4.964047
    _C   =                  0.500266
    _W   =                6.921294
    _PHI =            2.122265
    _D   =                  0.265744
    _C2  =                  0.180255
    _W2  =                9.430558
    _PHI2 =           -1.040059
    _C3  =                  0.104922
    _W3  =               17.064190
    _PHI3 =                1.364193
    _C4  =                  0.122225
    _W4  =             20.983760
    _PHI4 = -1.271991


class LinPPLModel(LPPLModel):
    """Linear-periodic Power Law: oscillation in CALENDAR time, not log-time.

    Fits: log10(price) = A + B*log10(t) + C*t^(-D)*cos(W_cal*t + φ)

    Unlike LPPL (log-periodic with ω·ln(t)), LinPPL uses ω·t so the cycle
    period is constant in calendar years — matching Bitcoin's ~4-year halving
    cycle rather than LPPL's ever-lengthening log-time cycles.

    W_cal is the angular frequency in radians per year. A halving cycle of
    T calendar years corresponds to W_cal = 2π/T. For T=4yr, W_cal ≈ 1.57.
    """
    name = "LinPPL"
    short_name = "linppl"
    legend_name = "LinPPL"
    dash_style = "dash"

    # Fitted parameters (W_cal in radians/year; T_years = 2π/W_cal)
    _A   = -1.212845  
    _B   =                                5.109690  
    _C   =                                0.283075  
    _W   =                                1.766924  # ≈ 2π/4 (4-year halving cycle, will refit)
    _PHI =  -2.291677  
    _D   =                                0.010000  

    def _lppl_log10(self, t):
        """Evaluate LinPPL median in log10 space — oscillation in calendar t, not ln(t)."""
        t = np.asarray(t, float)
        t_safe = np.maximum(t, 0.1)
        envelope = self._C * t_safe ** (-self._D)
        return (self._A + self._B * np.log10(t_safe)
                + envelope * np.cos(self._W * t_safe + self._PHI))

    component_names = [
        "A (constant)",
        "B\u00b7log\u2081\u2080(t)",
        "damped osc (\u03c9_cal\u00b7t)",
    ]
    formula_log10_latex = (
        r"A + B \log_{10}(t) + C \cdot t^{-D} \cos(\omega_{\text{cal}} \cdot t + \varphi)"
    )
    component_details = {
        "A (constant)":           ("A",                         [("A", "_A")]),
        "B\u00b7log\u2081\u2080(t)": ("B\u00b7log\u2081\u2080(t)", [("B", "_B")]),
        "damped osc (\u03c9_cal\u00b7t)": (
            "C\u00b7t^(-D)\u00b7cos(\u03c9_cal\u00b7t+\u03c6)",
            [("C", "_C"), ("D", "_D"),
             ("\u03c9_cal", "_W"), ("\u03c6", "_PHI")]),
    }

    def components(self, t):
        """Calendar-time oscillation (cos(W_cal * t + phi)), not log-time."""
        t = np.asarray(t, float)
        t_safe = np.maximum(t, 0.1)
        return {
            "A (constant)":                    np.full_like(t_safe, self._A),
            "B\u00b7log\u2081\u2080(t)":        self._B * np.log10(t_safe),
            "damped osc (\u03c9_cal\u00b7t)":   self._C * t_safe ** (-self._D) * np.cos(
                self._W * t_safe + self._PHI),
        }


