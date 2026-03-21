"""Callbacks package — importing this module registers all Dash callbacks."""
from dash import ctx  # noqa: F401 — re-export for test_web.py mock patching

from callbacks import coerce, mc_helpers, charts, mc_controls, mc_payment
from callbacks import mc_upload, sc_loan, lots, nav, snapshot_cb, ticker

# Re-exports for test_web.py and app.py backward compatibility
from callbacks.coerce import _ci, _cf, _format_lots_for_table
from callbacks.mc_helpers import (_build_mc_params, _mc_setup,
                                  _mc_finalize, _mc_payment_check,
                                  _mc_status, _ghost_match, _unblocked_val)
from callbacks.mc_upload import (_parse_mc_upload, _extract_mc_key_val,
                                 _MC_UPLOAD_FIELDS)
from callbacks.mc_controls import (_mc_years_options, _toggle_dca_sc_body,
                                   _restore_mc)
from callbacks.lots import (_lots_summary, manage_lots, preview_percentile,
                            sync_table_on_load)
from callbacks.charts import (update_bubble, update_heatmap, update_dca,
                               update_retire, update_supercharge,
                               auto_bubble_yrange)
from callbacks.snapshot_cb import (restore_from_url, update_effective_lots,
                                   manage_snapshot, update_snapshot_banner,
                                   restore_my_lots, render_link_history,
                                   clear_history, apply_hm_palette,
                                   generate_share_qr)
from callbacks.nav import (_TAB_CONTROLS, _TAB_TO_PATH, _PATH_TO_TAB,
                            toggle_sc_mode, toggle_sc_display_q,
                            open_faq_item, toggle_share_modal)
from callbacks.sc_loan import update_sc_info
from callbacks.ticker import update_price_ticker
import callbacks.scanner  # noqa: F401
