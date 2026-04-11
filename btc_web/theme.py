"""Quantoshi chart theme — re-exports from colors.py for backward compat.

This module preserves the original constant names so existing importers
(`from theme import PLOT_BG_COLOR`) keep working without modification.
The actual values live in btc_web/colors.py.
"""
from colors import (
    PLOT_BG_COLOR,
    TEXT_COLOR,
    TITLE_COLOR,
    SPINE_COLOR,
    GRID_MAJOR_COLOR,
)

__all__ = [
    "PLOT_BG_COLOR",
    "TEXT_COLOR",
    "TITLE_COLOR",
    "SPINE_COLOR",
    "GRID_MAJOR_COLOR",
]
