"""figures — Plotly chart builders for the Bitcoin Projections web app.

Re-exports the public API so that ``from figures import build_bubble_figure``
continues to work unchanged after the monolith was split into sub-modules.
"""

from figures.common import (
    FREQ_PPY, _FREQ_STEP_DAYS, _LOGO_B64_ALL, _apply_watermark,
    _build_qr_config_text, _build_mc_config_text,
    _apply_config_annotation, _build_thermal_colors, _price_tickvals,
)
from figures.bubble import build_bubble_figure
from figures.heatmap import build_heatmap_figure, build_mc_heatmap_figure
from figures.dca import build_dca_figure
from figures.retire import build_retire_figure
from figures.supercharge import build_supercharge_figure
