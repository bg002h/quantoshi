"""Basis-function models: PCA and Greedy.

PCA fits an ortho basis via singular value decomposition of log-space residuals;
Greedy iteratively adds the single best basis function from a large dictionary.
Both share _ShrinkingBandsMixin.
"""

import numpy as np

from btc_core._helpers import _lazy_norm, _DEFAULT_QS
from btc_core._base import _ShrinkingBandsMixin


class PCAModel(_ShrinkingBandsMixin):
    """PCA-based model: principal components from HybPPL-family component basis.

    Takes the ~30 component time series from all HybPPL-family models,
    runs PCA (SVD) to find orthogonal directions, then OLS-regresses
    log10(price) on the top k principal components.

    Result: R²=0.993 with 7 params (6 PCs + intercept) — beats Hyb2B
    (16 params) on BIC. The 30 correlated components collapse into ~6
    orthogonal directions that capture all the signal.

    At prediction time, evaluates all source basis functions at t,
    applies pre-computed weight vector (no matrix ops needed).
    """
    name = "PCA (HybPPL basis)"
    short_name = "pca"
    legend_name = "PCA"
    dash_style = "dot"
    quantized = True

    # Source model keys whose components form the basis
    _SOURCE_KEYS = ("hybppl", "hybppl_dd", "hyb2l", "hyb2c", "hyb2b", "hyb4d")
    _N_PCS = 6  # number of principal components to use

    def __init__(self, price_years, price_prices, quantiles, source_models=None):
        if source_models is None:
            source_models = {}
        mask = price_years >= 1.0
        t = price_years[mask]
        lp = np.log10(price_prices[mask])
        n = len(t)

        # Build component matrix from all source models
        self._basis_info = []  # [(model_key, comp_name), ...] for each column
        columns = []
        for key in self._SOURCE_KEYS:
            mdl = source_models.get(key)
            if mdl is None:
                continue
            comps = mdl.components(t)
            for cname, vals in comps.items():
                columns.append(np.asarray(vals, float))
                self._basis_info.append((key, cname))

        if not columns:
            # Fallback: degenerate model
            self._intercept = float(np.mean(lp))
            self._weights = np.array([])
            self._sigma = float(np.std(lp))
            self._X_mean = np.array([])
            self._V_k = np.array([]).reshape(0, 0)
            self._beta = np.array([self._intercept])
            self._explained = np.array([])
        else:
            X = np.column_stack(columns)
            X_mean = X.mean(axis=0)
            Xc = X - X_mean

            # SVD-based PCA
            U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
            k = min(self._N_PCS, len(S))
            total_var = np.sum(S ** 2)
            self._explained = (S ** 2 / total_var)[:k]

            # PC scores and OLS regression
            scores = (U * S)[:, :k]
            X_reg = np.column_stack([np.ones(n), scores])
            beta = np.linalg.lstsq(X_reg, lp, rcond=None)[0]

            # Collapse PCA + OLS into a single weight vector on components
            V_k = Vt[:k, :].T  # (n_components x k)
            w = V_k @ beta[1:]  # (n_components,)
            intercept = beta[0] - float(X_mean @ w)

            self._intercept = intercept
            self._weights = w
            self._X_mean = X_mean
            self._V_k = V_k
            self._beta = beta
            residuals = lp - (intercept + X @ w)
            self._sigma = float(np.std(residuals))  # backward compat

        # Store source models for component evaluation at prediction time
        self._source_models = {k: source_models[k] for k in self._SOURCE_KEYS
                               if k in source_models}

        # Build shrinking quantile bands
        self._init_shrinking_bands(t, lp - self._model_log10(t), quantiles)
        self._build_colors()

    def _eval_basis(self, t):
        """Evaluate all source basis functions at time t, return column vector."""
        t = np.asarray(t, float)
        columns = []
        for key, cname in self._basis_info:
            mdl = self._source_models.get(key)
            if mdl is None:
                continue
            comps = mdl.components(t)
            columns.append(np.asarray(comps[cname], float))
        if not columns:
            return np.zeros_like(t)
        return np.column_stack(columns)

    def _model_log10(self, t):
        """Evaluate: intercept + X @ weights."""
        t_arr = np.asarray(t, float)
        scalar = t_arr.ndim == 0
        if scalar:
            t_arr = t_arr.reshape(1)
        X = self._eval_basis(t_arr)
        if X.ndim == 1 or len(self._weights) == 0:
            result = np.full_like(t_arr, self._intercept)
        else:
            result = self._intercept + X @ self._weights
        return float(result[0]) if scalar else result

    # price_at, interp_price, find_percentile inherited from _ShrinkingBandsMixin

    # Decomposition: group the 30 weighted basis functions by physical role
    _COMP_GROUPS = [
        ("intercept",         "intercept",     None),
        ("power law trend",   "B\u00b7log\u2081\u2080(t)", "B"),
        ("log-periodic osc",  "log osc",       "log"),
        ("calendar-periodic", "cal osc",       "cal"),
    ]

    component_names = [
        "intercept",
        "power law trend",
        "log-periodic osc",
        "calendar-periodic",
    ]

    formula_log10_latex = (
        r"\text{intercept} + \sum_{j} w_j \cdot f_j(t)"
    )
    formula_product_latex = (
        r"10^{\,\text{intercept}} \cdot \prod_{j} 10^{\,w_j \cdot f_j(t)}"
    )

    @property
    def component_details(self):
        return {
            "intercept": (
                "\u03b1 (constant)",
                [("const", "_intercept")],
            ),
            "power law trend": (
                "\u03a3 w\u2c7c\u00b7B\u2c7c\u00b7log\u2081\u2080(t)",
                [],
            ),
            "log-periodic osc": (
                "\u03a3 w\u2c7c\u00b7C\u2c7c\u00b7t^(\u2212D)\u00b7cos(\u03c9\u00b7ln(t)+\u03c6)",
                [],
            ),
            "calendar-periodic": (
                "\u03a3 w\u2c7c\u00b7C\u2c7c\u00b7cos(\u03c9\u00b7t+\u03c6)",
                [],
            ),
        }

    def components(self, t):
        """Decompose into intercept + grouped basis function contributions."""
        t_arr = np.asarray(t, float)
        scalar = t_arr.ndim == 0
        if scalar:
            t_arr = t_arr.reshape(1)
        X = self._eval_basis(t_arr)
        n = len(t_arr)
        intercept = np.full(n, self._intercept)
        trend = np.zeros(n)
        log_osc = np.zeros(n)
        cal_osc = np.zeros(n)

        if X.ndim > 1 and len(self._weights) > 0:
            for i, ((key, cname), w) in enumerate(zip(self._basis_info, self._weights)):
                contrib = w * X[:, i]
                cl = cname.lower()
                if "log" in cl and ("osc" in cl or "cos" in cl):
                    log_osc += contrib
                elif "cal" in cl and ("osc" in cl or "cos" in cl):
                    cal_osc += contrib
                elif "log" in cl and "t" in cl:
                    # B·log₁₀(t) — power law trend
                    trend += contrib
                elif "constant" in cl or cname.startswith("A "):
                    intercept += contrib
                else:
                    trend += contrib  # fallback: lump into trend

        result = {
            "intercept": intercept,
            "power law trend": trend,
            "log-periodic osc": log_osc,
            "calendar-periodic": cal_osc,
        }
        if scalar:
            result = {k: float(v[0]) for k, v in result.items()}
        return result

    def _build_colors(self):
        """Indigo palette — PCA model."""
        self.colors = {}
        n = len(self.quantiles)
        for i, q in enumerate(self.quantiles):
            frac = i / max(n - 1, 1)
            r = int(60 + 50 * frac)
            g = int(40 + 60 * frac)
            b = int(120 + 60 * frac)
            self.colors[q] = f"#{r:02x}{g:02x}{b:02x}"


class GreedyModel(_ShrinkingBandsMixin):
    """Greedy forward BIC-selected model: 5 oscillatory terms from LPPL/HybPPL.

    Selects components via greedy forward BIC minimisation from the pool
    of individual oscillatory terms in existing LPPL/HybPPL models.
    Result: R²=0.9928, σ=0.130, BIC=-23,319 with only 7 parameters
    (intercept + slope + 5 weighted oscillatory terms).

    v2: uses entropy-damped E(w·t) and EPPL model components.
    All parameters are hardcoded — no runtime dependency on other models.

    Formula:
        log₁₀(price) = α + β·log₁₀(t) + Σᵢ wᵢ·fᵢ(t)

    where fᵢ are 5 oscillatory basis functions selected by greedy
    BIC minimization from a dictionary of entropy-damped, power-law-
    damped, undamped, and EPPL model components.
    """
    name = "Greedy Select"
    short_name = "grdy"
    legend_name = "Greedy"
    dash_style = "dashdot"
    quantized = True

    # ── OLS intercept and slope ──────────────────────────────────────────
    _alpha = -1.166405
    _beta  =  5.078858
    _sigma       = 0.123652  # backward compat
    _sigma0_up   = 0.093000
    _alpha_up    = 0.343500
    _sigma0_down = 0.106900
    _alpha_down  = 0.498500

    # ── 5 selected oscillatory terms (v2: entropy-damped) ────────────────
    # f₁: E(0.10)·sin(7.5·ln(t)) — entropy-damped log-periodic
    _w1 = 0.015686;  _we1 = 0.10;  _W1 = 7.5
    # f₂: undamped halving cycle (from EPPL: C2·cos(Wc1·t+P2))
    _w2 = 0.997744;  _C2 = 0.202747;  _Wc2 = 1.881312;  _P2 = 2.520900
    # f₃: E(0.05)·cos(2π/1.88·t) — entropy-damped sub-halving
    _w3 = -0.139415;  _we3 = 0.05;  _Wc3 = 3.340840  # 2π/1.88
    # f₄: EPPL entropy log osc 1 (C1·E(w1·t)·cos(W1·ln(t)+P1))
    _w4 = 0.981907;  _C4 = 0.250431;  _W4 = 16.823756;  _P4 = 1.460422;  _we4 = 0.251550
    # f₅: EPPL entropy log osc 2 (C3·E(w2·t)·cos(W2·ln(t)+P3))
    _w5 = 1.007897;  _C5 = 0.556269;  _W5 = 7.803554;  _P5 = 1.373041;  _we5 = 0.107049

    def __init__(self, price_years, price_prices, quantiles):
        # Build quantile bands via shrinking σ(t) (z stored, σ computed at eval)
        self.fits = {}
        for q in quantiles:
            self.fits[q] = {"z": float(_lazy_norm().ppf(q))}
        self.quantiles = sorted(self.fits.keys())
        self._build_colors()

    @staticmethod
    def _entropy_env(t, w):
        """Shannon entropy envelope: max(-x·ln(x), 0)/(1/e) where x=w·t."""
        x = w * t
        raw = -x * np.log(np.maximum(x, 1e-30))
        return np.maximum(raw, 0) * np.e

    def _model_log10(self, t):
        """Evaluate: α + β·log₁₀(t) + Σ wᵢ·fᵢ(t)."""
        t_arr = np.asarray(t, float)
        scalar = t_arr.ndim == 0
        if scalar:
            t_arr = t_arr.reshape(1)
        ts = np.maximum(t_arr, 0.1)
        ln_t = np.log(ts)

        result = self._alpha + self._beta * np.log10(ts)
        # f₁: entropy-damped sin(7.5·ln(t))
        result += self._w1 * self._entropy_env(ts, self._we1) * np.sin(self._W1 * ln_t)
        # f₂: undamped halving cycle
        result += self._w2 * self._C2 * np.cos(self._Wc2 * ts + self._P2)
        # f₃: entropy-damped cos(sub-halving)
        result += self._w3 * self._entropy_env(ts, self._we3) * np.cos(self._Wc3 * ts)
        # f₄: EPPL entropy log osc 1
        result += self._w4 * self._C4 * self._entropy_env(ts, self._we4) * np.cos(self._W4 * ln_t + self._P4)
        # f₅: EPPL entropy log osc 2
        result += self._w5 * self._C5 * self._entropy_env(ts, self._we5) * np.cos(self._W5 * ln_t + self._P5)

        return float(result[0]) if scalar else result

    # price_at, interp_price, find_percentile inherited from _ShrinkingBandsMixin

    # ── Decomposition ────────────────────────────────────────────────────

    component_names = [
        "\u03b1 (intercept)",
        "\u03b2\u00b7log\u2081\u2080(t)",
        "f\u2081 entropy log-periodic",
        "f\u2082 halving cycle",
        "f\u2083 entropy sub-halving",
        "f\u2084 entropy log osc 1",
        "f\u2085 entropy log osc 2",
    ]

    formula_log10_latex = (
        r"\alpha + \beta \log_{10}(t) + \sum_{i=1}^{5} w_i \cdot f_i(t)"
    )
    formula_product_latex = (
        r"10^{\,\alpha} \cdot t^{\beta} \cdot \prod_{i=1}^{5} 10^{\,w_i \cdot f_i(t)}"
    )

    @property
    def component_details(self):
        return {
            "\u03b1 (intercept)": (
                "\u03b1",
                [("\u03b1", "_alpha")],
            ),
            "\u03b2\u00b7log\u2081\u2080(t)": (
                "\u03b2\u00b7log\u2081\u2080(t)",
                [("\u03b2", "_beta")],
            ),
            "f\u2081 entropy log-periodic": (
                "w\u2081\u00b7E(w_e\u2081\u00b7t)\u00b7sin(\u03c9\u2081\u00b7ln(t))",
                [("w\u2081", "_w1"), ("w_e\u2081", "_we1"), ("\u03c9\u2081", "_W1")],
            ),
            "f\u2082 halving cycle": (
                "w\u2082\u00b7C\u2082\u00b7cos(\u03c9_c\u00b7t+\u03c6\u2082)",
                [("w\u2082", "_w2"), ("C\u2082", "_C2"),
                 ("\u03c9_c", "_Wc2"), ("\u03c6\u2082", "_P2")],
            ),
            "f\u2083 entropy sub-halving": (
                "w\u2083\u00b7E(w_e\u2083\u00b7t)\u00b7cos(\u03c9_c\u2083\u00b7t)",
                [("w\u2083", "_w3"), ("w_e\u2083", "_we3"), ("\u03c9_c\u2083", "_Wc3")],
            ),
            "f\u2084 entropy log osc 1": (
                "w\u2084\u00b7C\u2084\u00b7E(w_e\u2084\u00b7t)\u00b7cos(\u03c9\u2084\u00b7ln(t)+\u03c6\u2084)",
                [("w\u2084", "_w4"), ("C\u2084", "_C4"),
                 ("w_e\u2084", "_we4"), ("\u03c9\u2084", "_W4"), ("\u03c6\u2084", "_P4")],
            ),
            "f\u2085 entropy log osc 2": (
                "w\u2085\u00b7C\u2085\u00b7E(w_e\u2085\u00b7t)\u00b7cos(\u03c9\u2085\u00b7ln(t)+\u03c6\u2085)",
                [("w\u2085", "_w5"), ("C\u2085", "_C5"),
                 ("w_e\u2085", "_we5"), ("\u03c9\u2085", "_W5"), ("\u03c6\u2085", "_P5")],
            ),
        }

    def components(self, t):
        """Decompose into intercept + trend + 5 individual oscillatory terms."""
        t_arr = np.asarray(t, float)
        scalar = t_arr.ndim == 0
        if scalar:
            t_arr = t_arr.reshape(1)
        ts = np.maximum(t_arr, 0.1)
        ln_t = np.log(ts)

        result = {
            "\u03b1 (intercept)":      np.full_like(ts, self._alpha),
            "\u03b2\u00b7log\u2081\u2080(t)": self._beta * np.log10(ts),
            "f\u2081 entropy log-periodic": self._w1 * self._entropy_env(ts, self._we1) * np.sin(self._W1 * ln_t),
            "f\u2082 halving cycle":   self._w2 * self._C2 * np.cos(self._Wc2 * ts + self._P2),
            "f\u2083 entropy sub-halving": self._w3 * self._entropy_env(ts, self._we3) * np.cos(self._Wc3 * ts),
            "f\u2084 entropy log osc 1": self._w4 * self._C4 * self._entropy_env(ts, self._we4) * np.cos(self._W4 * ln_t + self._P4),
            "f\u2085 entropy log osc 2": self._w5 * self._C5 * self._entropy_env(ts, self._we5) * np.cos(self._W5 * ln_t + self._P5),
        }
        if scalar:
            result = {k: float(v[0]) for k, v in result.items()}
        return result

    def _build_colors(self):
        """Forest green palette — greedy select model."""
        self.colors = {}
        n = len(self.quantiles)
        for i, q in enumerate(self.quantiles):
            frac = i / max(n - 1, 1)
            r = int(30 + 50 * frac)      # 30 → 80
            g = int(120 + 60 * frac)     # 120 → 180
            b = int(50 + 50 * frac)      # 50 → 100
            self.colors[q] = f"#{r:02x}{g:02x}{b:02x}"


