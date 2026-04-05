"""Composite model: support + bubbles, R2, comp_by_n."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .bubble_shape import bubble_shape

PLOT_GRID_POINTS = 3000
PLOT_YEARS_MIN = 1.0
PLOT_YEARS_MAX = 72.0


@dataclass
class CompositeResult:
    composite_at_grid: np.ndarray   # USD prices on t_grid
    composite_at_data: np.ndarray   # log10 composite at data timestamps (for R2)
    r2: float
    r2_support: float
    hist_K_max: float
    total_plot: np.ndarray          # raw log-excess total on t_grid
    t_grid: np.ndarray
    log_support_grid: np.ndarray
    support_grid: np.ndarray        # USD prices on t_grid
    support_intercept: float        # A_sup in log_price = A_sup + B_sup*log10(t)
    support_slope: float            # B_sup


def _total_bubble(t, fitted_bubbles, budget, slope_sup):
    """Sum bubble_shape contributions, optionally capped by a budget.

    Mirrors ``bm_total_bubble`` from Cell 0 lines 729-762.

    Parameters
    ----------
    t : ndarray
        Time values (years since genesis).
    fitted_bubbles : list[dict]
        Bubble parameter dicts (need t_rise, r, t_plateau, t_decay, d, plat_pow).
    budget : ndarray or None
        Per-point budget cap.  When provided, bubbles are processed
        chronologically and each contribution is capped to the remaining
        budget at every time point.  None = plain additive sum.
    slope_sup : float
        Support line slope, passed through to ``bubble_shape``.
    """
    if not fitted_bubbles:
        return np.zeros(len(t))

    if budget is None:
        total = np.zeros(len(t))
        for bp in fitted_bubbles:
            total += bubble_shape(t, bp['t_rise'], bp['r'],
                                  bp['t_plateau'], bp['t_decay'], bp['d'],
                                  bp.get('plat_pow', 0.0),
                                  slope_sup=slope_sup)
        return total

    # Sequential capped sum
    ordered = sorted(fitted_bubbles, key=lambda b: b['t_rise'])
    total = np.zeros(len(t))
    remaining = np.maximum(0.0, budget.copy())
    for bp in ordered:
        contrib = bubble_shape(t, bp['t_rise'], bp['r'],
                               bp['t_plateau'], bp['t_decay'], bp['d'],
                               bp.get('plat_pow', 0.0),
                               slope_sup=slope_sup)
        contrib = np.minimum(contrib, remaining)
        total += contrib
        remaining = np.maximum(0.0, remaining - contrib)
    return total


def build_composite(support, fitted, price_data, t_grid=None, cap_overlap=True):
    """Build composite model on a plotting grid + compute R2.

    Parameters
    ----------
    support : SupportLine
        Fitted support line.
    fitted : list[dict]
        All fitted bubble parameter dicts (major + minor combined).
    price_data : PriceData
        Loaded price data (for R2 computation on the full daily series).
    t_grid : ndarray or None
        Plotting grid in years since genesis.  None = default linspace.
    cap_overlap : bool
        Apply sequential budget cap (default True, matches CAP_COMPOSITE_OVERLAP).

    Returns
    -------
    CompositeResult
    """
    if t_grid is None:
        t_grid = np.linspace(PLOT_YEARS_MIN, PLOT_YEARS_MAX, PLOT_GRID_POINTS)

    slope_sup = support.slope

    # ── hist_K_max: maximum log-excess in the historical fit residuals ────
    hist_K_max = float(np.max(np.maximum(0.0, support.log_excess)))

    # ── Plotting-grid composite ───────────────────────────────────────────
    log_t_grid = np.log10(t_grid)
    log_support_grid = support.intercept + slope_sup * log_t_grid
    support_grid = 10.0 ** log_support_grid

    budget_plot = np.full(len(t_grid), hist_K_max) if cap_overlap else None
    all_fitted = sorted(fitted, key=lambda b: b['t_rise'])
    total_plot = _total_bubble(t_grid, all_fitted, budget_plot, slope_sup)
    composite_at_grid = 10.0 ** (log_support_grid + total_plot)

    # ── R2 on the full daily time series ──────────────────────────────────
    years_all = price_data.df_full["years"].values
    log_p_all = price_data.df_full["log_price"].values
    log_support_all = support.log_support_all

    budget_all = np.full(len(years_all), hist_K_max) if cap_overlap else None
    total_all = _total_bubble(years_all, all_fitted, budget_all, slope_sup)
    composite_all = log_support_all + total_all

    ss_tot = np.sum((log_p_all - np.mean(log_p_all)) ** 2)
    r2_support = 1.0 - np.sum((log_p_all - log_support_all) ** 2) / ss_tot
    r2_comp = 1.0 - np.sum((log_p_all - composite_all) ** 2) / ss_tot

    return CompositeResult(
        composite_at_grid=composite_at_grid,
        composite_at_data=composite_all,
        r2=r2_comp,
        r2_support=r2_support,
        hist_K_max=hist_K_max,
        total_plot=total_plot,
        t_grid=t_grid,
        log_support_grid=log_support_grid,
        support_grid=support_grid,
        support_intercept=float(support.intercept),
        support_slope=float(support.slope),
    )


def build_comp_by_n(support, fitted, future, t_grid, hist_K_max,
                    total_plot, cap_overlap=True):
    """Precompute composite USD prices for N = 0 .. len(future) future bubbles.

    Mirrors Cell 2 lines 1-36 logic.

    Parameters
    ----------
    support : SupportLine
        Fitted support line.
    fitted : list[dict]
        All historical fitted bubbles.
    future : list[dict]
        Future bubbles sorted by t_rise.
    t_grid : ndarray
        Plotting grid (years since genesis).
    hist_K_max : float
        Budget cap per point.
    total_plot : ndarray
        Log-excess total from historical fitted bubbles on t_grid.
    cap_overlap : bool
        Apply sequential budget cap (default True).

    Returns
    -------
    list[list[float]]
        comp_by_n[n] = Python list of USD prices on t_grid for N=n future
        bubbles added.  Length = len(future) + 1.
    """
    slope_sup = support.slope
    log_t_grid = np.log10(t_grid)
    log_support_grid = support.intercept + slope_sup * log_t_grid

    all_fitted = sorted(fitted, key=lambda b: b['t_rise'])
    max_n = len(future)
    comp_by_n = []

    for n in range(max_n + 1):
        subset = future[:n]
        if cap_overlap:
            # Start with full budget, subtract historical fitted contributions
            budget = np.full(len(t_grid), hist_K_max)
            for bp in all_fitted:
                c = bubble_shape(t_grid, bp['t_rise'], bp['r'],
                                 bp['t_plateau'], bp['t_decay'], bp['d'],
                                 bp.get('plat_pow', 0.0),
                                 slope_sup=slope_sup)
                budget = np.maximum(0.0, budget - np.minimum(c, budget))
            # Add future bubbles under remaining budget
            tot = total_plot.copy()
            rem = budget.copy()
            for fb in subset:
                c = bubble_shape(t_grid, fb['t_rise'], fb['r'],
                                 fb['t_plateau'], fb['t_decay'], fb['d'],
                                 fb.get('plat_pow', 0.0),
                                 slope_sup=slope_sup)
                cap = np.minimum(c, rem)
                tot = tot + cap
                rem = np.maximum(0.0, rem - cap)
        else:
            tot = total_plot.copy()
            for fb in subset:
                tot += bubble_shape(t_grid, fb['t_rise'], fb['r'],
                                    fb['t_plateau'], fb['t_decay'], fb['d'],
                                    fb.get('plat_pow', 0.0),
                                    slope_sup=slope_sup)
        # E8: return Python lists, not numpy arrays
        comp_by_n.append((10.0 ** (log_support_grid + tot)).tolist())

    return comp_by_n
