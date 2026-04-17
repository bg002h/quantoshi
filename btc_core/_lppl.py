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
    _A   = -1.153754
    _B   =                    5.079165
    _C   =                    0.733958
    _W   =                    7.557598
    _PHI =                    1.377438
    _D   =                    0.607919

    def __init__(self, price_years, price_prices, quantiles):
        mask = price_years >= 1.0
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
    _A   = -1.130574
    _B   =                       5.038215
    _C   =                       0.705403
    _W   =                       7.376654
    _PHI =                       1.583777
    _D   =                       0.565686
    _C2  =                       0.168834
    _W2  =                20.904698
    _PHI2 = -1.158033

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
    _A   = -1.094293
    _B   =                 4.966525
    _C   =                 0.613915
    _W   =                 7.122560
    _PHI =                 1.890154
    _D   =                 0.365686
    _C2  =                 0.178607
    _W2  =            20.805576
    _PHI2 = -0.997242
    _C3  =                 0.171206
    _W3  =            10.083261
    _PHI3 = -2.167289

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
    _A   = -1.086800
    _B   =   4.987100
    _C   =   0.552500
    _W   =   7.416000
    _PHI =   1.533100
    _D   =   0.356800


class LPPL2ModelW(LPPL2Model):
    """LPPL\u2082 with log-time-uniform weighting."""
    name = "LPPL\u2082 (weighted)"
    short_name = "lp2_w"
    legend_name = "LPPL\u2082\u1d65\u1d65"
    _A   = -1.102500
    _B   =   4.980300
    _C   =   0.500900
    _W   =   6.857500
    _PHI =   2.196600
    _D   =   0.301900
    _C2  =   0.188000
    _W2  =   9.259900
    _PHI2 = -0.846900


class LPPL3ModelW(LPPL3Model):
    """LPPL\u2083 with log-time-uniform weighting."""
    name = "LPPL\u2083 (weighted)"
    short_name = "lp3_w"
    legend_name = "LPPL\u2083\u1d65\u1d65"
    _A   = -1.103800
    _B   =   4.984500
    _C   =   0.443200
    _W   =   6.760600
    _PHI =   2.284100
    _D   =   0.245900
    _C2  =   0.209000
    _W2  =   9.083400
    _PHI2 = -0.530300
    _C3  =   0.132000
    _W3  =  17.258100
    _PHI3 =  1.054900


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
    _A   = -1.099422
    _B   =                 4.971727
    _C   =                 0.581476
    _W   =                 7.103090
    _PHI =                 1.920446
    _D   =                 0.331580
    _C2  =                 0.172529
    _W2  =              10.018587
    _PHI2 = -2.052668
    _C3  =                 0.095754
    _W3  =             30.953441
    _PHI3 =    -2.440706
    _C4  =                 0.170618
    _W4  =           20.754990
    _PHI4 = -0.911211

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
    _A   = -1.101500
    _B   =   4.976300
    _C   =   0.491600
    _W   =   6.861300
    _PHI =   2.190700
    _D   =   0.276400
    _C2  =   0.190000
    _W2  =   9.313500
    _PHI2 = -0.889300
    _C3  =   0.119700
    _W3  =  20.921200
    _PHI3 = -1.203000
    _C4  =   0.108400
    _W4  =  17.087700
    _PHI4 =  1.336900


class LPPL4ModelN13(LPPL4Model):
    """LPPL\u2084 with ω=13 intermod band excluded (secondary frequencies constrained to avoid 11.5-14.5).

    Secondaries at ω≈9.9, 17.5, 20.9 (ratios 1.41, 2.48, 2.96). R²=0.9905.
    """
    name = "LPPL\u2084 (no \u03c9\u224813)"
    short_name = "lp4_n13"
    legend_name = "LPPL\u2084-n13"
    _A   = -1.096074
    _B   =   4.966763
    _C   =   0.557848
    _W   =   7.053735
    _PHI =   1.972293
    _D   =   0.306513
    _C2  =   0.146182
    _W2  =  20.900338
    _PHI2 = -1.214332
    _C3  =   0.186290
    _W3  =   9.931483
    _PHI3 = -1.893553
    _C4  =   0.093978
    _W4  =  17.456360
    _PHI4 =  0.827859


class LPPL4ModelWN13(LPPL4ModelN13):
    """LPPL\u2084 weighted + ω=13 excluded.

    Secondaries at ω≈9.3, 17.1, 20.9 (same as LP4 weighted, since
    LP4 weighted's secondaries already avoid the 11.5-14.5 band).
    """
    name = "LPPL\u2084 (weighted, no \u03c9\u224813)"
    short_name = "lp4_w_n13"
    legend_name = "LPPL\u2084\u1d65\u1d65-n13"
    _A   = -1.101466
    _B   =   4.976301
    _C   =   0.491568
    _W   =   6.861297
    _PHI =   2.190663
    _D   =   0.276438
    _C2  =   0.108395
    _W2  =  17.087659
    _PHI2 =  1.336939
    _C3  =   0.119697
    _W3  =  20.921188
    _PHI3 = -1.202956
    _C4  =   0.189996
    _W4  =   9.313489
    _PHI4 = -0.889292


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
    _A   = -1.213406  
    _B   =                 5.110908  
    _C   =                 0.282358  
    _W   =                 1.765788  # ≈ 2π/4 (4-year halving cycle, will refit)
    _PHI =  -2.284355  
    _D   =                 0.010000  

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


