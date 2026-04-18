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
    """Greedy forward BIC-selected model: 5 oscillatory terms from a
    dictionary of undamped / hybrid-damped / entropy-damped log- and
    cal-space oscillations.

    v3 (2026-04-17): the candidate dictionary is
      * 3 log frequencies (from LPPL₃'s best-fit triplet)
      * 3 cal frequencies (from a new 3-freq cal-space DE fit)
      * each paired with 3 dampings (none / hybrid / entropy) × 2 phases (sin, cos)
    = 36 candidates. Greedy forward-BIC picks the best 5.

    v2 used entropy-only damping; v3 re-opens the space to allow any mix
    of damping / space / phase. Basis is now stored as a generic
    ``_BASIS`` tuple — fit_grdy.py writes it directly, no per-slot
    hardcoded arithmetic in this class.

    Formula:
        log₁₀(price) = α + β·log₁₀(t) + Σᵢ wᵢ·Dᵢ(t)·φᵢ(t)

    where each (Dᵢ, φᵢ) is chosen from the dictionary.
    """
    name = "Greedy Select"
    short_name = "grdy"
    legend_name = "Greedy"
    dash_style = "dashdot"
    quantized = True

    # ── OLS intercept and slope ──────────────────────────────────────────
    _alpha = -1.103025  
    _beta  =                  4.974213  
    _sigma       = 0.123652  # backward compat
    _sigma0_up   = 0.093000
    _alpha_up    = 0.343500
    _sigma0_down = 0.106900
    _alpha_down  = 0.498500

    # ── Selected basis (written by tools/fit_grdy.py --update) ───────────
    # Each term is a tuple: (space, damping, freq, phase, weight, d_param)
    #   space   : "log" or "cal"
    #   damping : "none" | "hybrid" | "entropy"
    #   freq    : angular freq (rad/ln(t) for log; rad/yr for cal)
    #   phase   : "sin" or "cos"
    #   weight  : OLS-fitted multiplier
    #   d_param : None for undamped; D for hybrid (t^-D); w_e for entropy
    _BASIS = (
        ('log', 'entropy', 6.436000, 'sin', -0.625083, 0.320630),
        ('cal', 'none', 1.699697, 'sin', 0.326520, None),
        ('log', 'none', 15.970474, 'sin', -0.095033, None),
        ('log', 'none', 6.550895, 'cos', -0.163392, None),
        ('cal', 'hybrid', 3.192910, 'sin', 0.102907, 0.050000),
    )

    def __init__(self, price_years, price_prices, quantiles):
        # Build quantile bands via shrinking σ(t) (z stored, σ computed at eval)
        self.fits = {}
        for q in quantiles:
            self.fits[q] = {"z": float(_lazy_norm().ppf(q))}
        self.quantiles = sorted(self.fits.keys())
        self._build_colors()

    @staticmethod
    def _entropy_env(t, w):
        """Shannon entropy envelope: max(-x·ln(x), 0) / (1/e) where x=w·t."""
        x = w * t
        raw = -x * np.log(np.maximum(x, 1e-30))
        return np.maximum(raw, 0.0) / (1.0 / np.e)

    @classmethod
    def _eval_term(cls, ts, ln_t, term):
        """Evaluate one basis term (weight × damping × oscillation)."""
        space, damping, freq, phase, weight, d_param = term
        arg = freq * (ln_t if space == "log" else ts)
        osc = np.sin(arg) if phase == "sin" else np.cos(arg)
        if damping == "none":
            env = 1.0
        elif damping == "hybrid":
            env = ts ** (-d_param)
        else:  # entropy
            env = cls._entropy_env(ts, d_param)
        return weight * env * osc

    def _model_log10(self, t):
        """Evaluate: α + β·log₁₀(t) + Σ wᵢ·Dᵢ(t)·φᵢ(t)."""
        t_arr = np.asarray(t, float)
        scalar = t_arr.ndim == 0
        if scalar:
            t_arr = t_arr.reshape(1)
        ts = np.maximum(t_arr, 0.1)
        ln_t = np.log(ts)

        result = self._alpha + self._beta * np.log10(ts)
        for term in self._BASIS:
            result = result + self._eval_term(ts, ln_t, term)

        return float(result[0]) if scalar else result

    # price_at, interp_price, find_percentile inherited from _ShrinkingBandsMixin

    # ── Decomposition ────────────────────────────────────────────────────

    formula_log10_latex = (
        r"\alpha + \beta \log_{10}(t) + \sum_i w_i \cdot D_i(t) \cdot \varphi_i(t)"
    )
    formula_product_latex = (
        r"10^{\,\alpha} \cdot t^{\beta} \cdot \prod_i 10^{\,w_i \cdot D_i(t) \cdot \varphi_i(t)}"
    )

    @staticmethod
    def _term_label(i, term):
        """Short label for the i-th selected term, used in the Model Info
        decomposition panel."""
        space, damping, freq, phase, _weight, _dp = term
        d_tag = {"none": "undamped",
                 "hybrid": "hybrid",
                 "entropy": "entropy"}[damping]
        space_tag = "log" if space == "log" else "cal"
        return f"f{i} {d_tag} {space_tag} ({phase} ω≈{freq:.2f})"

    @property
    def component_names(self):
        names = [
            "\u03b1 (intercept)",
            "\u03b2\u00b7log\u2081\u2080(t)",
        ]
        for i, term in enumerate(self._BASIS, 1):
            names.append(self._term_label(i, term))
        return names

    @property
    def component_details(self):
        """Per-term metadata for the Model Info decomposition panel.

        For each basis term we show the generic form string (with damping
        envelope substituted) and the concrete numeric params. Class-attr
        names use the index (e.g. 'f1') to stay stable when _BASIS changes.
        """
        details = {
            "\u03b1 (intercept)": ("\u03b1", [("\u03b1", "_alpha")]),
            "\u03b2\u00b7log\u2081\u2080(t)":
                ("\u03b2\u00b7log\u2081\u2080(t)", [("\u03b2", "_beta")]),
        }
        for i, term in enumerate(self._BASIS, 1):
            space, damping, freq, phase, weight, d_param = term
            # Generic form string
            env_str = {
                "none": "",
                "hybrid": "t\u207b\u1d40 \u00b7 ",
                "entropy": "E(w_e\u00b7t) \u00b7 ",
            }[damping]
            arg_str = ("\u03c9\u00b7ln(t)" if space == "log"
                       else "\u03c9\u00b7t")
            form = f"w{i}\u00b7{env_str}{phase}({arg_str})"
            # Params list — use generic names
            plist = [(f"w{i}", f"__basis_weight_{i}"),
                     (f"\u03c9{i}", f"__basis_freq_{i}")]
            if damping == "hybrid":
                plist.append((f"D{i}", f"__basis_dparam_{i}"))
            elif damping == "entropy":
                plist.append((f"w_e{i}", f"__basis_dparam_{i}"))
            details[self._term_label(i, term)] = (form, plist)
        return details

    def __getattr__(self, name):
        """Virtual attributes for component_details: __basis_weight_1, _freq_1,
        _dparam_1, etc. Lets the Model Info panel pull per-term numeric
        values without a parallel dict."""
        if name.startswith("__basis_"):
            kind, idx = name[len("__basis_"):].rsplit("_", 1)
            try:
                i = int(idx) - 1
            except ValueError:
                raise AttributeError(name)
            if i < 0 or i >= len(self._BASIS):
                raise AttributeError(name)
            term = self._BASIS[i]
            if kind == "weight":
                return term[4]
            if kind == "freq":
                return term[2]
            if kind == "dparam":
                return term[5]
        raise AttributeError(name)

    def components(self, t):
        """Decompose into intercept + trend + individual oscillatory terms."""
        t_arr = np.asarray(t, float)
        scalar = t_arr.ndim == 0
        if scalar:
            t_arr = t_arr.reshape(1)
        ts = np.maximum(t_arr, 0.1)
        ln_t = np.log(ts)

        result = {
            "\u03b1 (intercept)":      np.full_like(ts, self._alpha),
            "\u03b2\u00b7log\u2081\u2080(t)": self._beta * np.log10(ts),
        }
        for i, term in enumerate(self._BASIS, 1):
            result[self._term_label(i, term)] = self._eval_term(ts, ln_t, term)
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


