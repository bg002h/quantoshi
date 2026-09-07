"""Tab-1 moving-average table — the single source of truth for ``bub-ma``.

The Display card on Tab 1 offers four simple moving averages of the daily
close.  Adding or renaming one has to move four things in lockstep:

* the checklist options in ``layout/bubble.py``,
* the bitmask option order in ``snapshot.py::_CHECKLIST_OPTIONS`` (index
  addressed — reordering it silently rewrites every shipped share link),
* the window length + legend name + dash pattern in ``figures/bubble.py``,
* the default in ``snapshot_defaults.py`` / ``tab_defaults.py``.

They all read :data:`MA_WINDOWS` instead.  Same pattern as ``bub_views.py``:
deliberately import-light (numpy only, no Dash, no layout) so every one of
those modules can import it without a cycle.

**Window lengths are DAYS, not samples of some coarser grid.**  The
conventional "200-week moving average" is a 1400-day arithmetic mean of daily
closes, not a mean of 200 weekly closes.  ``BitcoinPricesDaily.csv`` is
contiguous daily (verified in ``test_bub_moving_avg.py``), so one sample is
one day and a plain window mean is exact.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

import numpy as np

__all__ = ["MAWindow", "MA_WINDOWS", "MA_VALUES", "MA_BY_VALUE", "MA_OPTIONS",
           "rolling_mean"]


class MAWindow(NamedTuple):
    """One selectable moving average.  Frozen by construction (NamedTuple)."""

    value: str    # checklist value / snapshot bitmask token
    label: str    # checklist label (leading space matches the sibling controls)
    days: int     # window length in days == samples of the daily close series
    legend: str   # chart legend name
    dash: str     # plotly line dash pattern — the ONLY per-window encoding


# ── The table ────────────────────────────────────────────────────────────────
# Order is load-bearing: _CHECKLIST_OPTIONS["bub-ma"] is this order, and the
# bitmask in every shipped share link is addressed by index.  APPEND ONLY.
#
# Colour is deliberately shared (colors.MA_LINE_COLOR) and the windows are told
# apart by DASH PATTERN.  The operator has deuteranomaly, and shape survives
# every palette; four hues would not.
MA_WINDOWS: tuple[MAWindow, ...] = (
    MAWindow("ma200w", " 200 week", 1400, "200W MA", "solid"),
    MAWindow("ma52w",  " 52 week",   364, "52W MA",  "dash"),
    MAWindow("ma30d",  " 30 day",     30, "30D MA",  "dot"),
    MAWindow("ma7d",   " 7 day",       7, "7D MA",   "dashdot"),
)

MA_VALUES: tuple[str, ...] = tuple(w.value for w in MA_WINDOWS)
MA_BY_VALUE: dict[str, MAWindow] = {w.value: w for w in MA_WINDOWS}
MA_OPTIONS: list[dict[str, str]] = [{"label": w.label, "value": w.value}
                                    for w in MA_WINDOWS]


def rolling_mean(prices: Sequence[float] | np.ndarray, window: int) -> np.ndarray:
    """Full-window simple moving average of ``prices``.

    Returns a float array the same length as ``prices``.  Position ``i`` holds
    the arithmetic mean of ``prices[i-window+1 : i+1]``; the first
    ``window - 1`` positions are ``NaN`` because their window is not complete.
    A partial-window value is never emitted — a "200 week MA" that starts on
    day 3 of the price history is a different, and misleading, statistic.

    Cumulative-sum implementation: O(n) rather than O(n·window), which matters
    because the 1400-day window over 5.9k daily closes is recomputed on every
    chart build.  ``test_bub_moving_avg.py`` oracle-tests it against an
    explicit ``np.mean`` over the live series.
    """
    a = np.asarray(prices, dtype=float)
    out = np.full(a.shape, np.nan, dtype=float)
    window = int(window)
    if window <= 0 or a.size < window:
        return out
    csum = np.concatenate(([0.0], np.cumsum(a)))
    out[window - 1:] = (csum[window:] - csum[:-window]) / window
    return out
