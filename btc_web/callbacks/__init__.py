"""Callbacks package — importing this module registers all Dash callbacks."""
from dash import ctx  # noqa: F401 — re-export for test_web.py mock patching

from callbacks import coerce, mc_helpers, charts, mc_controls, mc_payment
from callbacks import mc_upload, sc_loan, lots, nav, snapshot_cb, ticker
from callbacks import routing, splash  # noqa: F401

# Re-exports for test_web.py and app.py backward compatibility
from callbacks.coerce import _ci, _cf, _format_lots_for_table
from callbacks.mc_helpers import (_build_mc_params, _mc_setup,
                                  _mc_finalize, _mc_payment_check,
                                  _mc_status, _ghost_match, _unblocked_val)
from callbacks.mc_upload import (_parse_mc_upload, _extract_mc_key_val,
                                 _MC_UPLOAD_FIELDS)
from callbacks.mc_controls import (_mc_years_options, _restore_mc)
from callbacks.lots import (_lots_summary, manage_lots, preview_percentile,
                            sync_table_on_load)
from callbacks.charts import (update_bubble, update_heatmap, update_dca,
                               update_retire, update_supercharge)
from callbacks.snapshot_cb import (restore_from_url, apply_globals,
                                   update_effective_lots,
                                   manage_snapshot, update_snapshot_banner,
                                   render_link_history,
                                   apply_hm_palette,
                                   generate_share_qr)
from callbacks.routing import (_TAB_CONTROLS, _TAB_TO_PATH, _PATH_TO_TAB,
                               open_faq_item)
import callbacks.nav  # noqa: F401 — clientside callbacks registered at import
from callbacks.sc_loan import update_sc_info
from callbacks.ticker import update_price_ticker
import callbacks.citadel_cb  # noqa: F401
import callbacks.citadel_tax_cb  # noqa: F401
import callbacks.scanner  # noqa: F401
import callbacks.user_model  # noqa: F401
import callbacks.citadel_save_cb  # noqa: F401
import callbacks.citadel_scenarios  # noqa: F401
import callbacks.plot_appearance  # noqa: F401
import callbacks.axes_presets  # noqa: F401 — callbacks registered at import
import callbacks.custom_time  # noqa: F401
import callbacks.timemachine  # noqa: F401 — Time Machine callbacks (Task 10)
from callbacks.timemachine import _asof_frame  # noqa: F401 — re-export for tests
import callbacks.leverage_cb  # noqa: F401 — callbacks registered at import
