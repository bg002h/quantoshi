"""Base protocol + mixin types for price models.

PriceModel Protocol defines the interface every model satisfies.
_FitsBasedModel/QuantileRegressionModel/_CompositeModel/_ShrinkingBandsMixin
are the shared building blocks the concrete model classes extend.
"""

from typing import Protocol, runtime_checkable

import numpy as np

from btc_core._helpers import (
    _lazy_norm, _fit_shrinking_sigma, _resqr_price_at, yr_to_t,
)


class _ShrinkingBandsMixin:
    """Mixin providing time-dependent σ bands for models with a _model_log10 method.

    Replaces the constant z*σ pattern with σ(t) = σ₀·t^(-α), asymmetric
    for above/below median. Models must set _sigma0_up, _alpha_up,
    _sigma0_down, _alpha_down (via _init_shrinking_bands or class attrs).
    """

    def _init_shrinking_bands(self, t, residuals, quantiles):
        """Fit shrinking σ from residuals and build fits dict."""
        s0u, au, s0d, ad = _fit_shrinking_sigma(t, residuals)
        self._sigma0_up = s0u
        self._alpha_up = au
        self._sigma0_down = s0d
        self._alpha_down = ad
        self._sigma = float(np.std(residuals))  # backward compat
        self.fits = {}
        for q in quantiles:
            self.fits[q] = {"z": float(_lazy_norm().ppf(q))}
        self.quantiles = sorted(self.fits.keys())

    def _sigma_at(self, t, q):
        """Constant σ for quantile bands (shrinking σ reverted — too narrow at late times)."""
        return self._sigma

    def price_at(self, q, t, sigma_mode="constant"):
        t_arr = np.asarray(t, float)
        log_median = self._model_log10(t_arr)
        if sigma_mode == "resqr":
            out = _resqr_price_at(self, q, t_arr, log_median)
            if out is not None:
                return out
        z = self.fits[q]["z"]
        sigma = self._sigma_at(t_arr, q)
        return 10.0 ** (log_median + z * sigma)

    def interp_price(self, q, t, sigma_mode="constant"):
        if q in self.fits:
            return float(self.price_at(q, t, sigma_mode=sigma_mode))
        sorted_qs = self.quantiles
        lo = max((qq for qq in sorted_qs if qq <= q), default=sorted_qs[0])
        hi = min((qq for qq in sorted_qs if qq >= q), default=sorted_qs[-1])
        if lo == hi:
            return float(self.price_at(lo, t, sigma_mode=sigma_mode))
        frac = (q - lo) / (hi - lo)
        p_lo = np.log10(float(self.price_at(lo, t, sigma_mode=sigma_mode)))
        p_hi = np.log10(float(self.price_at(hi, t, sigma_mode=sigma_mode)))
        return 10.0 ** (p_lo + frac * (p_hi - p_lo))

    def find_percentile(self, t, price, sigma_mode="constant"):
        sorted_qs = self.quantiles
        if not sorted_qs:
            return 0.5
        t_safe = max(float(t), 0.5)
        log_p = np.log10(max(float(price), 1e-10))
        log_ps = [np.log10(max(float(self.price_at(q, t_safe, sigma_mode=sigma_mode)), 1e-10))
                  for q in sorted_qs]
        if log_p <= log_ps[0]:
            return sorted_qs[0]
        if log_p >= log_ps[-1]:
            return sorted_qs[-1]
        for i in range(len(sorted_qs) - 1):
            if log_ps[i] <= log_p <= log_ps[i + 1]:
                frac = (log_p - log_ps[i]) / (log_ps[i + 1] - log_ps[i] + 1e-30)
                return sorted_qs[i] + frac * (sorted_qs[i + 1] - sorted_qs[i])
        return sorted_qs[-1]


@runtime_checkable
class PriceModel(Protocol):
    """Protocol all price models must satisfy.

    Implement this to add a new model — then register in app.py and the UI
    auto-discovers it via PRICE_MODELS iteration.
    """
    name: str              # human-readable name ("Bubble Model", "Power Law")
    short_name: str        # registry key ("bub", "pl", "s2f")
    quantized: bool        # True → has fits dict → MC-compatible
    quantiles: list        # sorted list of available quantiles
    colors: dict           # {q: "#hex"} for trace coloring
    fits: dict | None      # {q: {"intercept","slope"}} or None
    dash_style: str        # Plotly dash pattern ("solid", "dot", "longdash")

    def price_at(self, q, t, sigma_mode="constant"): ...
    def interp_price(self, q, t, sigma_mode="constant"): ...
    def find_percentile(self, t, price, sigma_mode="constant"): ...


class _FitsBasedModel:
    """Base for models whose quantile bands are log-linear in log10(t).

    Subclasses must set self.fits, self.quantiles, and self.colors.
    """
    quantized = True

    def _model_log10(self, t):
        """Median log10 curve for resqr sigma mode — use the Q50 fit (nearest if absent)."""
        qs = sorted(self.fits.keys())
        q_med = 0.5 if 0.5 in self.fits else min(qs, key=lambda qq: abs(qq - 0.5))
        f = self.fits[q_med]
        return f["intercept"] + f["slope"] * np.log10(np.maximum(np.asarray(t, float), 1e-6))

    def price_at(self, q, t, sigma_mode="constant"):
        """Price at quantile q, time t (years since genesis)."""
        if sigma_mode == "resqr":
            t_arr = np.asarray(t, float)
            log_median = self._model_log10(t_arr)
            out = _resqr_price_at(self, q, t_arr, log_median)
            if out is not None:
                return out
        f = self.fits[q]
        return 10.0 ** (f["intercept"] + f["slope"] * np.log10(np.asarray(t, float)))

    def interp_price(self, q, t, sigma_mode="constant"):
        """Log-space interpolated price for arbitrary quantile (e.g. Q7.5%)."""
        if q in self.fits:
            return float(self.price_at(q, t, sigma_mode=sigma_mode))
        sorted_qs = self.quantiles
        lo = max((qq for qq in sorted_qs if qq <= q), default=sorted_qs[0])
        hi = min((qq for qq in sorted_qs if qq >= q), default=sorted_qs[-1])
        if lo == hi:
            return float(self.price_at(lo, t, sigma_mode=sigma_mode))
        frac = (q - lo) / (hi - lo)
        p_lo = np.log10(float(self.price_at(lo, t, sigma_mode=sigma_mode)))
        p_hi = np.log10(float(self.price_at(hi, t, sigma_mode=sigma_mode)))
        return 10.0 ** (p_lo + frac * (p_hi - p_lo))

    def find_percentile(self, t, price, sigma_mode="constant"):
        """Interpolate the QR percentile (0–1) for a given time and price."""
        sorted_qs = self.quantiles
        if not sorted_qs:
            return 0.5
        t_safe = max(float(t), 0.5)
        log_p = np.log10(max(float(price), 1e-10))
        log_ps = [np.log10(max(float(self.price_at(q, t_safe, sigma_mode=sigma_mode)), 1e-10))
                  for q in sorted_qs]
        if log_p <= log_ps[0]:
            return sorted_qs[0]
        if log_p >= log_ps[-1]:
            return sorted_qs[-1]
        for i in range(len(sorted_qs) - 1):
            if log_ps[i] <= log_p <= log_ps[i + 1]:
                frac = (log_p - log_ps[i]) / (log_ps[i + 1] - log_ps[i] + 1e-30)
                return sorted_qs[i] + frac * (sorted_qs[i + 1] - sorted_qs[i])
        return sorted_qs[-1]


class QuantileRegressionModel(_FitsBasedModel):
    """Raw quantile regression fits — model-free, purely empirical.

    Straight lines in log-log space. Each quantile has its own independently
    fitted slope and intercept. This is what BubbleModel used to be before
    the shrinking Gaussian conversion.
    """
    name = "Quantile Regression"
    short_name = "qr"
    legend_name = "QR"
    dash_style = "dash"

    def __init__(self, md):
        self.fits = md.qr_fits
        self.colors = dict(md.qr_colors) if md.qr_colors else {}
        self.quantiles = sorted(md.qr_fits.keys())


class _CompositeModel:
    """Base for models with a shaped composite median and asymmetric shrinking
    Gaussian bands.

    σ_up(t) = σ₀_up × t^(-α_up) for quantiles ≥ 0.5
    σ_down(t) = σ₀_down × t^(-α_down) for quantiles < 0.5

    Subclasses must set self._t_grid, self._log_comp, and call self._init_bands().
    """
    quantized = True

    def _composite_log10(self, t):
        """Interpolate composite curve in log10 space at arbitrary t."""
        t = np.asarray(t, float)
        return np.interp(t, self._t_grid, self._log_comp)

    def _sigma_at(self, t, q):
        """Compute σ at time t for quantile q (asymmetric, shrinking)."""
        t = np.maximum(np.asarray(t, float), 0.5)
        if q >= 0.5:
            return self._sigma0_up * t ** (-self._alpha_up)
        else:
            return self._sigma0_down * t ** (-self._alpha_down)

    def _model_log10(self, t):
        """Median log10 curve for resqr mode — the full composite (support +
        bubble cycles). Fitting resqr residuals against this shaped curve
        means the bands track the composite's bubble peaks rather than the
        straight support line, so they parallel the visible median."""
        return self._composite_log10(t)

    def price_at(self, q, t, sigma_mode="constant"):
        """Price at quantile q, time t (years since genesis)."""
        t_arr = np.asarray(t, float)
        if sigma_mode == "resqr":
            log_median_resqr = self._model_log10(t_arr)
            out = _resqr_price_at(self, q, t_arr, log_median_resqr)
            if out is not None:
                return out
        log_median = self._composite_log10(t_arr)
        z = _lazy_norm().ppf(q)
        sigma = self._sigma_at(t_arr, q)
        return 10.0 ** (log_median + z * sigma)

    def interp_price(self, q, t, sigma_mode="constant"):
        """Log-space interpolated price for arbitrary quantile."""
        if q in self.fits:
            return float(self.price_at(q, t, sigma_mode=sigma_mode))
        sorted_qs = self.quantiles
        lo = max((qq for qq in sorted_qs if qq <= q), default=sorted_qs[0])
        hi = min((qq for qq in sorted_qs if qq >= q), default=sorted_qs[-1])
        if lo == hi:
            return float(self.price_at(lo, t, sigma_mode=sigma_mode))
        frac = (q - lo) / (hi - lo)
        p_lo = np.log10(float(self.price_at(lo, t, sigma_mode=sigma_mode)))
        p_hi = np.log10(float(self.price_at(hi, t, sigma_mode=sigma_mode)))
        return 10.0 ** (p_lo + frac * (p_hi - p_lo))

    def find_percentile(self, t, price, sigma_mode="constant"):
        """Reverse lookup: time + price → quantile."""
        sorted_qs = self.quantiles
        if not sorted_qs:
            return 0.5
        t_safe = max(float(t), 0.5)
        log_p = np.log10(max(float(price), 1e-10))
        log_ps = [np.log10(max(float(self.price_at(q, t_safe, sigma_mode=sigma_mode)), 1e-10))
                  for q in sorted_qs]
        if log_p <= log_ps[0]:
            return sorted_qs[0]
        if log_p >= log_ps[-1]:
            return sorted_qs[-1]
        for i in range(len(sorted_qs) - 1):
            if log_ps[i] <= log_p <= log_ps[i + 1]:
                frac = (log_p - log_ps[i]) / (log_ps[i + 1] - log_ps[i] + 1e-30)
                return sorted_qs[i] + frac * (sorted_qs[i + 1] - sorted_qs[i])
        return sorted_qs[-1]

    def _init_bands(self, sigma0_up, alpha_up, sigma0_down, alpha_down, quantiles):
        """Initialize σ parameters and build fits dict."""
        self._sigma0_up = sigma0_up
        self._alpha_up = alpha_up
        self._sigma0_down = sigma0_down
        self._alpha_down = alpha_down
        self.fits = {}
        for q in quantiles:
            self.fits[q] = {"z": float(_lazy_norm().ppf(q))}
        self.quantiles = sorted(self.fits.keys())

    component_names = ["support", "bubbles"]
    support_component_names = ["support"]
    formula_log10_latex = (
        r"\log_{10}(\text{support}(t)) + \sum_i \text{bubble}_i(t)"
    )
    component_details = {
        "support": ("log\u2081\u2080(support(t))", []),
        "bubbles": ("\u03a3 bubble_i(t)", []),
    }

    def components(self, t):
        """Composite decomposition: support + bubbles (both in log10 space).

        sum(components(t)) == self._composite_log10(t).
        """
        t = np.asarray(t, float)
        log_support = np.interp(t, self._t_grid, self._log_support)
        log_composite = self._composite_log10(t)
        return {
            "support": log_support,
            "bubbles": log_composite - log_support,
        }

    @property
    def comp_by_n(self):
        return self._comp_by_n

    @property
    def support_plot(self):
        return self._support_plot

    @property
    def t_grid(self):
        return self._t_grid

    @property
    def bm_r2(self):
        return self._bm_r2

    @property
    def n_future_max(self):
        return self._n_future_max

