"""Parametric bubble shape function -- pure math, no state."""
from __future__ import annotations
import numpy as np


def bubble_shape(t, t_rise, r, t_plateau, t_decay, d, plat_pow=0.0, slope_sup=0.0):
    """Compute log10(excess above support) for a single bubble.

    Parameters
    ----------
    t         : array of time values (years since genesis)
    t_rise    : start of the exponential rise
    r         : rise rate (yr-1); price doubles relative to support every log(2)/r years
    t_plateau : end of rise / start of plateau
    t_decay   : end of plateau / start of decay
    d         : decay rate (yr-1)
    plat_pow  : differential power-law exponent during plateau, relative to support.
                price ~ t^(slope_sup + plat_pow) during [t_plateau, t_decay].
                  plat_pow = 0          -> plateau parallels support (log_excess = K)
                  plat_pow = -slope_sup -> price is constant during plateau
                  plat_pow > 0          -> blow-off top (price grows faster than support)
    slope_sup : power-law exponent of the support line. Required because
                the rise/decay phases include a power-law correction term.
                In the notebook this was a closure over the global scope.

    Phases
    ------
    Rise    [t_rise, t_plateau):
        log_excess(t) = r*(t - t_rise) + slope_sup*log10(t_rise / t)

    Plateau [t_plateau, t_decay):
        log_excess(t) = K + plat_pow*log10(t / t_plateau)
        where K = log_excess(t_plateau) = peak of rise phase.

    Decay   [t_decay, inf):
        log_excess(t) = K_end - d*(t - t_decay) + slope_sup*log10(t_decay / t), clipped >= 0
        where K_end = K + plat_pow*log10(t_decay / t_plateau) = log_excess at t_decay.
    """
    t = np.asarray(t, dtype=float)
    result = np.zeros_like(t)
    if t_plateau <= t_rise:
        return result
    # K: log-excess at the start of the plateau (= peak of rise)
    K = r * (t_plateau - t_rise) + slope_sup * np.log10(
        np.maximum(t_rise / t_plateau, 1e-12))
    if K <= 0:
        return result
    # K_end: log-excess at the end of the plateau (= start of decay)
    K_end = (K + plat_pow * np.log10(np.maximum(t_decay / t_plateau, 1e-12))
             if t_decay > t_plateau else K)

    # Rise
    m = (t >= t_rise) & (t < t_plateau)
    if m.any():
        result[m] = np.maximum(
            r * (t[m] - t_rise) + slope_sup * np.log10(
                np.maximum(t_rise / t[m], 1e-12)), 0.0)
    # Plateau (general power law)
    m = (t >= t_plateau) & (t < t_decay)
    if m.any():
        result[m] = np.maximum(
            K + plat_pow * np.log10(np.maximum(t[m] / t_plateau, 1e-12)), 0.0)
    # Decay
    m = t >= t_decay
    if m.any():
        result[m] = np.maximum(
            K_end - d * (t[m] - t_decay) + slope_sup * np.log10(
                np.maximum(t_decay / t[m], 1e-12)), 0.0)
    return result
