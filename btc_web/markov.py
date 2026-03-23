# Copyright (c) 2026 Quantoshi / bg002h. All rights reserved.
# This file is proprietary and NOT covered by the project's BSD-2-Clause license.
# Unauthorized copying, distribution, or use is prohibited.

"""markov.py — Markov chain Monte Carlo engine for Bitcoin price simulations.

Builds a transition probability matrix from historical percentile data,
then runs Monte Carlo simulations to produce price path distributions
for DCA and retirement scenarios.
"""

import numpy as np
from btc_core import yr_to_t

# ── Frequency constants ──────────────────────────────────────────────────────

FREQ_PPY = {"Daily": 365, "Weekly": 52, "Monthly": 12, "Quarterly": 4, "Annually": 1}

# ── Cost constants (sats) ────────────────────────────────────────────────────

COST_PER_STEP        = 0.01   # sats per simulation step
COST_PER_ENTRY       = 1.0    # sats per non-zero transition matrix entry
COST_PER_WINDOW_YEAR = 2.0    # sats per year of training window beyond minimum
MIN_WINDOW_YEARS     = 4      # shortest viable window (for 5×5 non-sparse)
MIN_TRANSITIONS_PER_CELL = 1.5  # sparsity threshold for max_bins calculation


def max_bins_for_window(window_years, step_days=30):
    """Max non-sparse bin count for a given training window and step size.

    Requires at least MIN_TRANSITIONS_PER_CELL transitions per matrix cell
    on average.  Returns int clamped to [5, 10].
    """
    transitions = window_years * (365.0 / max(step_days, 1))
    raw = int(np.sqrt(transitions / MIN_TRANSITIONS_PER_CELL))
    return max(5, min(raw, 10))


def compute_cost(n_steps, n_sims, n_nonzero_entries, window_years):
    """Compute total cost in sats for an MC run.

    Returns dict with matrix_cost, sim_cost, window_cost, total.
    """
    matrix_cost = n_nonzero_entries * COST_PER_ENTRY
    sim_cost    = n_steps * COST_PER_STEP * n_sims
    window_cost = max(0.0, window_years - MIN_WINDOW_YEARS) * COST_PER_WINDOW_YEAR
    total       = matrix_cost + sim_cost + window_cost
    return {
        "matrix_cost": matrix_cost,
        "sim_cost":    sim_cost,
        "window_cost": window_cost,
        "total":       total,
    }


def count_nonzero_entries(trans_matrix):
    """Count non-zero entries in a transition matrix."""
    return int(np.count_nonzero(trans_matrix))


# ── Build transition matrix ──────────────────────────────────────────────────

def _prices_to_percentiles(prices, years, model):
    """Convert price series to percentile series (0–1) using a price model."""
    pcts = np.empty(len(prices))
    for i in range(len(prices)):
        t = max(float(years[i]), 0.5)
        pcts[i] = model.find_percentile(t, float(prices[i]))
    return pcts


def _bin_index(pct, bin_edges):
    """Map a percentile (0–1) to a bin index."""
    n_bins = len(bin_edges) - 1
    idx = int(pct * n_bins)
    return min(max(idx, 0), n_bins - 1)


def build_transition_matrix(prices, years, model, n_bins=5,
                            window_start_yr=None, window_end_yr=None,
                            step_days=30):
    """Build Markov transition matrix from historical price percentiles.

    Parameters
    ----------
    prices : array-like — daily prices
    years : array-like — years since genesis for each price
    model : PriceModel — model with find_percentile(t, price) method
    n_bins : int — number of percentile bins (e.g., 5 → 20% each)
    window_start_yr, window_end_yr : float or None — filter window (years since genesis)
    step_days : int — transition step size in days (30 = monthly)

    Returns
    -------
    trans_matrix : ndarray (n_bins, n_bins) — row-stochastic transition matrix
    bin_edges : ndarray (n_bins + 1,) — percentile bin edges [0, 0.2, 0.4, ..., 1.0]
    pct_series : ndarray — full percentile series for the window
    """
    prices = np.asarray(prices, dtype=float)
    years = np.asarray(years, dtype=float)

    # Apply time window filter
    mask = np.ones(len(prices), dtype=bool)
    if window_start_yr is not None:
        mask &= years >= window_start_yr
    if window_end_yr is not None:
        mask &= years <= window_end_yr
    w_prices = prices[mask]
    w_years = years[mask]

    if len(w_prices) < step_days + 1:
        # Not enough data — return uniform matrix
        bin_edges = np.linspace(0, 1, n_bins + 1)
        return np.ones((n_bins, n_bins)) / n_bins, bin_edges, np.array([])

    # Convert to percentiles
    pct_series = _prices_to_percentiles(w_prices, w_years, model)

    # Build transition counts
    bin_edges = np.linspace(0, 1, n_bins + 1)
    trans = np.zeros((n_bins, n_bins), dtype=float)

    for i in range(0, len(pct_series) - step_days, step_days):
        src = _bin_index(pct_series[i], bin_edges)
        dst = _bin_index(pct_series[i + step_days], bin_edges)
        trans[src, dst] += 1.0

    # Normalize rows (add small epsilon to avoid zero-row issues)
    row_sums = trans.sum(axis=1)
    for r in range(n_bins):
        if row_sums[r] > 0:
            trans[r] /= row_sums[r]
        else:
            # No observations from this state — use uniform
            trans[r] = 1.0 / n_bins

    return trans, bin_edges, pct_series


# ── Monte Carlo simulation ───────────────────────────────────────────────────

def monte_carlo_prices(trans_matrix, bin_edges, start_pctile, n_steps,
                       n_sims, model, start_t, dt):
    """Generate Monte Carlo price paths using the Markov transition matrix.

    Parameters
    ----------
    trans_matrix : ndarray (n_bins, n_bins)
    bin_edges : ndarray (n_bins + 1,)
    start_pctile : float — starting percentile (0–1)
    n_steps : int — number of time steps to simulate
    n_sims : int — number of simulation runs
    model : PriceModel — model with interp_price(q, t) method
    start_t : float — starting time (years since genesis)
    dt : float — time step in years (e.g., 1/12 for monthly)

    Returns
    -------
    price_paths : ndarray (n_sims, n_steps) — simulated price paths
    pctile_paths : ndarray (n_sims, n_steps) — simulated percentile paths
    """
    n_bins = len(bin_edges) - 1
    rng = np.random.default_rng()

    # Precompute cumulative probabilities for efficient sampling
    cum_probs = np.cumsum(trans_matrix, axis=1)

    # Start bin
    start_bin = _bin_index(start_pctile, bin_edges)

    price_paths = np.empty((n_sims, n_steps))
    pctile_paths = np.empty((n_sims, n_steps))

    for sim in range(n_sims):
        current_bin = start_bin
        for step in range(n_steps):
            # Sample next bin from transition probabilities
            r = rng.random()
            next_bin = np.searchsorted(cum_probs[current_bin], r)
            next_bin = min(next_bin, n_bins - 1)

            # Convert bin to a percentile (sample uniformly within bin)
            bin_lo = bin_edges[next_bin]
            bin_hi = bin_edges[next_bin + 1]
            pctile = bin_lo + rng.random() * (bin_hi - bin_lo)

            # Convert percentile to price at the correct time
            t = start_t + (step + 1) * dt
            price = model.interp_price(pctile, t)

            pctile_paths[sim, step] = pctile
            price_paths[sim, step] = price
            current_bin = next_bin

    return price_paths, pctile_paths


# ── DCA simulation on MC paths ──────────────────────────────────────────────

def mc_dca(price_paths, amount, start_stack=0.0):
    """Run DCA simulation on Monte Carlo price paths.

    Parameters
    ----------
    price_paths : ndarray (n_sims, n_steps)
    amount : float — USD to invest per period
    start_stack : float — initial BTC holdings

    Returns
    -------
    btc_paths : ndarray (n_sims, n_steps) — BTC balance over time
    usd_paths : ndarray (n_sims, n_steps) — USD value over time
    """
    n_sims, n_steps = price_paths.shape
    btc_paths = np.empty_like(price_paths)
    usd_paths = np.empty_like(price_paths)

    for sim in range(n_sims):
        stack = start_stack
        for step in range(n_steps):
            price = price_paths[sim, step]
            if price > 0:
                stack += amount / price
            btc_paths[sim, step] = stack
            usd_paths[sim, step] = stack * price

    return btc_paths, usd_paths


# ── Retirement simulation on MC paths ────────────────────────────────────────

def mc_retire(price_paths, start_stack, withdrawal, inflation_rate, dt):
    """Run retirement withdrawal simulation on Monte Carlo price paths.

    Parameters
    ----------
    price_paths : ndarray (n_sims, n_steps)
    start_stack : float — initial BTC holdings
    withdrawal : float — USD withdrawal per period (initial, before inflation)
    inflation_rate : float — annual inflation rate (e.g., 0.04 for 4%)
    dt : float — time step in years

    Returns
    -------
    btc_paths : ndarray (n_sims, n_steps) — BTC balance over time
    usd_paths : ndarray (n_sims, n_steps) — USD value over time
    depletion_steps : ndarray (n_sims,) — step at which BTC runs out (-1 if never)
    """
    n_sims, n_steps = price_paths.shape
    btc_paths = np.empty_like(price_paths)
    usd_paths = np.empty_like(price_paths)
    depletion_steps = np.full(n_sims, -1, dtype=int)

    for sim in range(n_sims):
        stack = start_stack
        for step in range(n_steps):
            price = price_paths[sim, step]
            # Inflation-adjusted withdrawal
            wd = withdrawal * (1 + inflation_rate) ** (step * dt)
            if price > 0 and stack > 0:
                btc_needed = wd / price
                stack -= btc_needed
            if stack <= 0 and depletion_steps[sim] == -1:
                depletion_steps[sim] = step
                stack = 0
            btc_paths[sim, step] = stack
            usd_paths[sim, step] = stack * price

    return btc_paths, usd_paths, depletion_steps


# ── Statistical summary ──────────────────────────────────────────────────────

def compute_fan_percentiles(paths, percentiles=(0.05, 0.25, 0.50, 0.75, 0.95)):
    """Compute percentile bands across simulations at each time step.

    Parameters
    ----------
    paths : ndarray (n_sims, n_steps)
    percentiles : tuple of floats (0–1)

    Returns
    -------
    dict mapping percentile → ndarray (n_steps,)
    """
    result = {}
    for p in percentiles:
        result[p] = np.percentile(paths, p * 100, axis=0)
    return result


def depletion_stats(depletion_steps, n_steps, dt, start_yr):
    """Summarize depletion statistics from retirement MC runs.

    Returns
    -------
    dict with keys:
        pct_depleted : float — fraction of sims that depleted
        median_depletion_yr : float or None
        p10_depletion_yr : float or None
        p90_depletion_yr : float or None
    """
    valid = depletion_steps[depletion_steps >= 0]
    n_sims = len(depletion_steps)
    pct_depleted = len(valid) / n_sims if n_sims > 0 else 0.0

    if len(valid) == 0:
        return {"pct_depleted": pct_depleted,
                "median_depletion_yr": None,
                "p10_depletion_yr": None,
                "p90_depletion_yr": None}

    depl_yrs = start_yr + valid * dt
    return {
        "pct_depleted": pct_depleted,
        "median_depletion_yr": float(np.median(depl_yrs)),
        "p10_depletion_yr": float(np.percentile(depl_yrs, 10)),
        "p90_depletion_yr": float(np.percentile(depl_yrs, 90)),
    }
