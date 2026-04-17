"""Simple price models: BubbleModel, PowerLaw, Exponential, Logistic,
BrokenPowerLaw, EmpiricalFloor, S2F, UserModel.

All non-LPPL/HybPPL/EPPL concrete models live here. Classes that subclass
_FitsBasedModel or _CompositeModel use the base types; the other four
use _ShrinkingBandsMixin directly.
"""

import numpy as np

from btc_core._helpers import (
    _lazy_norm, _lazy_linregress, _DEFAULT_QS, _compute_log_r2,
)
from btc_core._base import (
    _ShrinkingBandsMixin, _FitsBasedModel, _CompositeModel,
)


class BubbleModel(_CompositeModel):
    """Bubble model with asymmetric shrinking Gaussian bands around composite."""
    name = "Bubble Model"
    short_name = "bub"
    legend_name = "BM"
    dash_style = "solid"

    def __init__(self, md):
        # Composite curve (max future bubbles)
        self._t_grid = np.asarray(md.years_plot_bm, float)
        comp = md.comp_by_n[-1]
        self._log_comp = np.log10(np.maximum(np.asarray(comp, float), 1e-10))

        # Support line (log10 USD) for component decomposition
        self._log_support = np.log10(np.maximum(
            np.asarray(md.support_bm, float), 1e-10))


        # Shrinking σ parameters (from pkl, fitted by tools/fit_sigma.py)
        self._init_bands(
            getattr(md, 'bm_sigma0_up', 0.085),
            getattr(md, 'bm_alpha_up', 0.132),
            getattr(md, 'bm_sigma0_down', 0.075),
            getattr(md, 'bm_alpha_down', 0.218),
            md.QR_QUANTILES,
        )

        # Colors: from pkl if present, otherwise generate thermal defaults.
        # app.py overwrites these with the full thermal palette at startup.
        if md.qr_colors:
            self.colors = dict(md.qr_colors)
        else:
            self.colors = {q: f"#{int(255*q):02x}80{int(255*(1-q)):02x}"
                           for q in self.quantiles}


class PowerLawModel(_FitsBasedModel):
    """OLS power law with Gaussian quantile bands.

    All bands share the same slope (OLS slope) but have different intercepts
    shifted by z_q * sigma where sigma is the OLS residual standard deviation.
    This means the bands are parallel lines in log-log space.
    """
    name = "Power Law"
    short_name = "pl"
    legend_name = "PL"
    dash_style = "dot"

    def __init__(self, ols_intercept, ols_slope, price_years, price_prices,
                 genesis, quantiles):
        # Compute OLS residual sigma
        mask = price_years >= 1.0  # skip very early data
        ly = np.log10(price_years[mask])
        lp = np.log10(price_prices[mask])
        predicted = ols_intercept + ols_slope * ly
        residuals = lp - predicted
        sigma = float(np.std(residuals))

        # Build fits: each quantile is the OLS line shifted by z_q * sigma
        self.fits = {}
        for q in quantiles:
            z = _lazy_norm().ppf(q)
            self.fits[q] = {
                "intercept": ols_intercept + z * sigma,
                "slope": ols_slope,
            }
        self.quantiles = sorted(self.fits.keys())

        # Cool blue/purple palette — visually distinct from Bubble's warm colors
        self._build_colors()

    def _build_colors(self):
        self.colors = {}
        n = len(self.quantiles)
        for i, q in enumerate(self.quantiles):
            frac = i / max(n - 1, 1)
            r = int(40 + 140 * frac)    # 40 → 180
            g = int(60 + 40 * frac)     # 60 → 100
            b = int(200 - 30 * frac)    # 200 → 170
            self.colors[q] = f"#{r:02x}{g:02x}{b:02x}"



class ExponentialModel(_ShrinkingBandsMixin):
    """Exponential growth model with shrinking Gaussian quantile bands.

    Fits log10(price) = a + b*t (linear in time, exponential in price).
    Poor fit (R²~0.87) — included for comparison to show why power law
    is preferred over exponential for Bitcoin.
    """
    name = "Exponential"
    short_name = "exp"
    legend_name = "Exp"
    dash_style = "longdashdot"
    quantized = True

    def __init__(self, price_years, price_prices, quantiles):
        mask = price_years >= 1.0
        t = price_years[mask]
        lp = np.log10(price_prices[mask])
        slope, intercept, r, _, _ = _lazy_linregress()(t, lp)
        self._intercept = intercept
        self._slope = slope
        residuals = lp - (intercept + slope * t)
        self._init_shrinking_bands(t, residuals, quantiles)
        self._build_colors()

    def _model_log10(self, t):
        t_arr = np.asarray(t, float)
        return self._intercept + self._slope * t_arr

    def _build_colors(self):
        """Red/pink palette — visually distinct, signals 'caution'."""
        self.colors = {}
        n = len(self.quantiles)
        for i, q in enumerate(self.quantiles):
            frac = i / max(n - 1, 1)
            r = int(200 + 55 * frac)     # 200 → 255
            g = int(60 + 80 * frac)      # 60 → 140
            b = int(80 + 60 * frac)      # 80 → 140
            self.colors[q] = f"#{r:02x}{g:02x}{b:02x}"


# ── EPPL config params (auto-generated) ──
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



class LogisticModel(_ShrinkingBandsMixin):
    """Logistic/Gompertz growth model with Gaussian quantile bands.

    Gompertz: log10(price) = K * exp(-exp(-r * (t - t0)))
    where K = carrying capacity (log10 of max price), r = growth rate,
    t0 = inflection point.

    Provides an upper saturation bound that power law models lack.
    """
    name = "Logistic Growth"
    short_name = "gomp"
    legend_name = "Gomp"
    dash_style = "dot"
    quantized = True

    # Fitted parameters (will be overwritten by fit_logistic.py --update)
    _K  =             4.888545  
    _r  =             0.302367  
    _t0 =             4.373878  

    def __init__(self, price_years, price_prices, quantiles):
        mask = price_years >= 1.0
        t = price_years[mask]
        lp = np.log10(price_prices[mask])
        predicted = self._model_log10(t)
        residuals = lp - predicted
        self._init_shrinking_bands(t, residuals, quantiles)
        self.quantiles = sorted(self.fits.keys())
        self._build_colors()

    def _model_log10(self, t):
        t = np.asarray(t, float)
        return self._K * np.exp(-np.exp(-self._r * (t - self._t0)))

    # price_at, interp_price, find_percentile inherited from _ShrinkingBandsMixin

    def _build_colors(self):
        """Steel blue palette — saturation model."""
        self.colors = {}
        n = len(self.quantiles)
        for i, q in enumerate(self.quantiles):
            frac = i / max(n - 1, 1)
            r = int(50 + 60 * frac)
            g = int(90 + 70 * frac)
            b = int(150 + 50 * frac)
            self.colors[q] = f"#{r:02x}{g:02x}{b:02x}"



class BrokenPowerLawModel(_ShrinkingBandsMixin):
    """Broken (two-segment) power law with Gaussian quantile bands.

    For t < t_break: log10(price) = a1 + b1 * log10(t)
    For t >= t_break: log10(price) = a2 + b2 * log10(t)
    Continuity constraint: a2 = a1 + (b1 - b2) * log10(t_break)
    """
    name = "Broken Power Law"
    short_name = "bpl"
    legend_name = "BPL"
    dash_style = "longdash"
    quantized = True

    # Fitted parameters (will be overwritten by fit_bpl.py --update)
    _a1      = -1.092244  
    _b1      =             4.920330  
    _t_break =             6.694045  
    _b2      =             5.318074  

    def __init__(self, price_years, price_prices, quantiles):
        mask = price_years >= 1.0
        t = price_years[mask]
        lp = np.log10(price_prices[mask])
        predicted = self._model_log10(t)
        residuals = lp - predicted
        self._init_shrinking_bands(t, residuals, quantiles)
        self.quantiles = sorted(self.fits.keys())
        self._build_colors()

    @property
    def _a2(self):
        return self._a1 + (self._b1 - self._b2) * np.log10(self._t_break)

    def _model_log10(self, t):
        t = np.asarray(t, float)
        t_safe = np.maximum(t, 0.1)
        lt = np.log10(t_safe)
        return np.where(
            t_safe < self._t_break,
            self._a1 + self._b1 * lt,
            self._a2 + self._b2 * lt,
        )

    # price_at, interp_price, find_percentile inherited from _ShrinkingBandsMixin

    def _build_colors(self):
        """Amber/tan palette — regime-shift model."""
        self.colors = {}
        n = len(self.quantiles)
        for i, q in enumerate(self.quantiles):
            frac = i / max(n - 1, 1)
            r = int(160 + 60 * frac)
            g = int(110 + 50 * frac)
            b = int(40 + 40 * frac)
            self.colors[q] = f"#{r:02x}{g:02x}{b:02x}"



class EmpiricalFloorModel(_CompositeModel):
    """BM Empirical Floor with asymmetric shrinking Gaussian bands."""
    name = "BM Empirical Floor"
    short_name = "ef"
    legend_name = "EF"
    dash_style = "longdash"

    def __init__(self, pkl_path):
        import pickle
        with open(pkl_path, "rb") as f:
            d = pickle.load(f)

        self._slope = d["ef_support_slope"]
        self._intercept = d["ef_support_intercept"]
        self._t_grid = np.asarray(d["years_plot"], float)
        self._support_plot = np.asarray(d["support_plot"], float)
        self._comp_by_n = d["comp_by_n"]
        self._bm_r2 = d["bm_r2"]
        self._n_future_max = d["n_future_max"]

        comp = self._comp_by_n[-1]
        self._log_comp = np.log10(np.maximum(np.asarray(comp, float), 1e-10))

        # Support line (log10 USD) for component decomposition
        self._log_support = np.log10(np.maximum(self._support_plot, 1e-10))

        # Shrinking σ parameters
        quantiles = d.get("QR_QUANTILES", [
            0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3,
            0.5, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 0.999])

        self._init_bands(
            d.get("sigma0_up", 0.093),
            d.get("alpha_up", 0.297),
            d.get("sigma0_down", 0.085),
            d.get("alpha_down", 0.295),
            quantiles,
        )
        self._build_colors()

    def _build_colors(self):
        """Amber/warm palette."""
        self.colors = {}
        n = len(self.quantiles)
        for i, q in enumerate(self.quantiles):
            frac = i / max(n - 1, 1)
            r = int(139 + 100 * frac)
            g = int(105 + 87 * frac)
            b = int(20 + 44 * frac)
            self.colors[q] = f"#{r:02x}{g:02x}{b:02x}"



class S2FModel:
    """Stock-to-Flow model — single price trajectory (not quantized).

    Fits log10(price) = a + b * log10(S2F) from historical data, where
    S2F = stock / annual_flow based on the Bitcoin halving schedule.
    """
    name = "Stock-to-Flow"
    short_name = "s2f"
    legend_name = "S2F"
    dash_style = "dot"
    quantized = False
    fits = None
    quantiles = []
    colors = {}

    _HALVING_BLOCKS = 210_000
    _BLOCKS_PER_DAY = 144
    _INITIAL_REWARD = 50.0

    def __init__(self, price_years, price_prices, genesis):
        self.genesis = genesis
        # Fit log10(price) = a + b * log10(S2F) from historical data
        mask = price_years >= 1.0
        yrs = price_years[mask]
        prices = price_prices[mask]

        s2f_vals = np.array([self._s2f_at_t(t) for t in yrs])
        valid = s2f_vals > 0
        log_s2f = np.log10(s2f_vals[valid])
        log_p = np.log10(prices[valid])

        slope, intercept, *_ = _lazy_linregress()(log_s2f, log_p)
        self._s2f_intercept = intercept
        self._s2f_slope = slope

    def _s2f_at_t(self, t):
        """Compute stock-to-flow ratio at years-since-genesis t."""
        days = t * 365.25
        total_blocks = days * self._BLOCKS_PER_DAY
        n_halvings = int(total_blocks // self._HALVING_BLOCKS)
        reward = self._INITIAL_REWARD / (2 ** n_halvings)

        # Cumulative stock
        stock = 0.0
        for h in range(n_halvings):
            stock += self._HALVING_BLOCKS * self._INITIAL_REWARD / (2 ** h)
        remaining = total_blocks - n_halvings * self._HALVING_BLOCKS
        stock += remaining * reward

        # Annual flow
        annual_flow = reward * self._BLOCKS_PER_DAY * 365.25
        if annual_flow <= 0:
            return 1e10  # effectively infinite S2F after all BTC mined
        return stock / annual_flow

    def price_at(self, q, t, sigma_mode="constant"):
        """S2F model price (ignores quantile — single trajectory)."""
        t_arr = np.asarray(t, float)
        scalar = t_arr.ndim == 0
        t_flat = t_arr.ravel()
        s2f_vals = np.array([self._s2f_at_t(ti) for ti in t_flat])
        log_p = self._s2f_intercept + self._s2f_slope * np.log10(s2f_vals)
        result = 10.0 ** log_p
        return float(result[0]) if scalar else result.reshape(t_arr.shape)

    def interp_price(self, q, t, sigma_mode="constant"):
        return float(self.price_at(q, t))

    def find_percentile(self, t, price, sigma_mode="constant"):
        return 0.5  # meaningless for non-quantized model



class UserModel(_FitsBasedModel):
    """User-defined power law model from two clicked points on log-log chart.

    Fully quantized: parallel lines (same slope, shifted intercepts) derived
    from the empirical residual distribution against historical prices.
    """
    name = "User Model"
    short_name = "u1"
    legend_name = "U\u2081"
    dash_style = "solid"
    quantized = True

    def __init__(self, slope, intercept, shifts, quantiles, r2_per_quantile, own_quantile):
        self.fits = {q: {"intercept": intercept + shifts[q], "slope": slope}
                     for q in quantiles}
        self.quantiles = sorted(quantiles)
        self.r2_per_quantile = r2_per_quantile or {}
        self.own_quantile = own_quantile
        self.colors = {q: "#e67e22" for q in self.quantiles}

    @classmethod
    def from_points(cls, t1, p1, t2, p2, price_years, price_prices, quantiles):
        """Factory: two chart points + historical data → fully quantized model."""
        log_t1, log_p1 = np.log10(max(t1, 0.01)), np.log10(max(p1, 1e-10))
        log_t2, log_p2 = np.log10(max(t2, 0.01)), np.log10(max(p2, 1e-10))
        denom = log_t2 - log_t1
        if abs(denom) < 1e-12:
            denom = 1e-12
        slope = (log_p2 - log_p1) / denom
        intercept = log_p1 - slope * log_t1

        mask = price_years >= 0.5
        t_hist = np.asarray(price_years[mask], float)
        p_hist = np.asarray(price_prices[mask], float)
        predicted = intercept + slope * np.log10(np.maximum(t_hist, 0.01))
        residuals = np.log10(np.maximum(p_hist, 1e-10)) - predicted

        own_quantile = float(np.mean(residuals <= 0))
        shifts = {q: float(np.percentile(residuals, q * 100)) for q in quantiles}

        # Ensure own_quantile is in the quantile list with shift=0
        # (the user's drawn line passes exactly through the two points)
        if own_quantile not in shifts:
            shifts[own_quantile] = 0.0
        else:
            shifts[own_quantile] = 0.0  # force exact zero even if percentile is close
        all_quantiles = sorted(set(quantiles) | {own_quantile})

        r2 = {}
        for q in all_quantiles:
            pred_q = 10.0 ** (intercept + shifts.get(q, 0) + slope * np.log10(np.maximum(t_hist, 0.01)))
            r2_val = _compute_log_r2(p_hist, pred_q)
            if r2_val is not None:
                r2[q] = r2_val

        return cls(slope, intercept, shifts, all_quantiles, r2, own_quantile)

    def to_store_dict(self):
        """Serialize to JSON-safe dict for dcc.Store."""
        slope = self.fits[self.quantiles[0]]["slope"]
        # base_intercept: the user's drawn line (shift=0, passes through both points)
        base_intercept = self.fits[self.own_quantile]["intercept"]
        return {
            "slope": slope,
            "base_intercept": base_intercept,
            "intercepts": {str(q): self.fits[q]["intercept"] for q in self.quantiles},
            "r2": {str(q): v for q, v in self.r2_per_quantile.items()},
            "own_quantile": self.own_quantile,
            "quantiles": [float(q) for q in self.quantiles],
        }

    @classmethod
    def from_store_dict(cls, d):
        """Reconstruct from dcc.Store dict."""
        if not d:
            return None
        quantiles = [float(q) for q in d["quantiles"]]
        slope = d["slope"]
        intercepts = {float(q): v for q, v in d["intercepts"].items()}
        r2 = {float(q): v for q, v in d["r2"].items()} if d.get("r2") else {}
        model = cls.__new__(cls)
        model.fits = {q: {"intercept": intercepts[q], "slope": slope} for q in quantiles}
        model.quantiles = sorted(quantiles)
        model.r2_per_quantile = r2
        model.own_quantile = d["own_quantile"]
        model.colors = {q: "#e67e22" for q in quantiles}
        return model
