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



class GompertzModel(_ShrinkingBandsMixin):
    """Gompertz growth model with Gaussian quantile bands.

    log10(price) = K * exp(-exp(-r * (t - t0)))
    where K = carrying capacity (log10 of max price), r = growth rate,
    t0 = inflection point.

    Provides an upper saturation bound that power law models lack.

    Note: Previously named `LogisticModel`, but the formula is Gompertz
    (asymmetric S-curve, not the symmetric logistic L/(1+exp(-k(t-t0)))).
    Renamed 2026-04-17 to eliminate confusion; short_name remains "gomp"
    so share links and cached snapshots continue to resolve.
    """
    name = "Gompertz"
    short_name = "gomp"
    legend_name = "Gomp"
    dash_style = "dot"
    quantized = True

    # Fitted parameters (will be overwritten by fit_gompertz.py --update)
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
