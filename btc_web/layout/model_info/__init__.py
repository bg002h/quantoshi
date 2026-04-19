"""Tab 8 — Model Info: detailed documentation of all price models.

Public API:
    _model_info_tab()     — returns the layout Div for the Model Info tab
    _MODEL_INFO_ITEM_IDS  — ordered list of accordion item_id values for /8.N
                            and /mi.N deep-linking

This module was split from a single 2764-line ``model_info.py``. The large
static documentation content (26 AccordionItems) lives in ``_items.py``; the
live coefficient-table helpers live in ``_helpers.py``. This file assembles
the outer layout (accordion, lightbox modal) and re-exports the public API.
"""

from dash import html
import dash_bootstrap_components as dbc

from colors import LIGHTBOX_BG

from ._items import _build_accordion_items


def _model_info_tab():
    return html.Div([
        dbc.Row(
            dbc.Col(
                html.Div([
                    html.H5("Price Models & Simulation Reference",
                            className="mb-3 mt-2"),
                    html.P(
                        "Quantoshi uses several price models and a Monte Carlo simulation engine. "
                        "All models operate in log\u2081\u2080 space where t is years since "
                        "July 25, 2009 \u2014 the statistically optimal time origin for the "
                        "power law fit, confirmed independently by multiple researchers. "
                        "This page documents the mathematics, fitted coefficients, and "
                        "methodology behind each.",
                        className="text-muted mb-4",
                    ),
                    dbc.Accordion(
                        _build_accordion_items(),
                        id="model-info-accordion",
                        start_collapsed=True,
                        flush=True,
                    ),
                ]),
                width={"size": 10, "offset": 1},
            )
        ),
        # Lightbox modal for enlarged images
        dbc.Modal([
            dbc.ModalBody(
                html.Img(id="mi-lightbox-img", style={"width": "100%"}),
                style={"padding": "0", "backgroundColor": LIGHTBOX_BG},
            ),
        ], id="mi-lightbox", size="xl", centered=True, is_open=False),
    ], className="p-3")


# Ordered list of accordion item_id values for /8.N and /mi.N deep linking.
# Must match the order of items returned by _build_accordion_items().
_MODEL_INFO_ITEM_IDS = [
    "mi-bub", "mi-qr", "mi-pl", "mi-lppl", "mi-lp2",
    "mi-lppl-weighting", "mi-linppl", "mi-hybppl", "mi-hybppl-dd",
    "mi-hyb2l", "mi-hyb2c", "mi-hyb2b", "mi-hyb4d", "mi-pca",
    "mi-grdy", "mi-eppl", "mi-exp", "mi-gomp", "mi-bpl",
    "mi-plo", "mi-sexp", "mi-logi",
    "mi-s2f", "mi-mc", "mi-ef", "mi-u1",
    "mi-compare", "mi-regimes", "mi-citadel",
]
