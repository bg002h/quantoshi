"""btc_core — Bitcoin Projections shared model package.

Split from the original btc_core.py (3267 lines, 37 classes) into thematic
submodules so callers can navigate by topic:

  _helpers      — lazy imports, sigma fitting, resqr, date/price helpers, R2
  _model_data   — ModelData + load_model_data
  _base         — PriceModel Protocol, _FitsBasedModel, QuantileRegressionModel,
                  _CompositeModel, _ShrinkingBandsMixin
  _simple       — BubbleModel, PowerLaw, Exp, Logistic, BrokenPowerLaw,
                  EmpiricalFloor, S2F, UserModel
  _lppl         — 10 LPPL variants + LinPPL
  _hybppl_eppl  — HybPPL family (6 classes) + EPPL family + HybPPLConfig
  _basis        — PCA, Greedy

The public API is preserved exactly: every name that was importable from
btc_core.py remains importable from btc_core.
"""

# ── Phase 2a sys.path bridge ──────────────────────────────────────────────
# btc_core needs `time_basis` (lives in btc_web/) for axis-aware constants
# (T_MIN, T_PER_YEAR, year_to_t). Adding btc_web/ to sys.path here means
# every btc_core submodule can `from time_basis import …` without its own
# path manipulation. This runs once when btc_core is first imported.
#
# TODO(phase2c): move time_basis to a more neutral location (e.g.
# btc_core/time_basis.py) or make btc_core a proper package. Either
# obviates this bridge.
import sys as _sys
from pathlib import Path as _Path
_BTC_WEB = str(_Path(__file__).resolve().parent.parent / "btc_web")
if _BTC_WEB not in _sys.path:
    _sys.path.insert(0, _BTC_WEB)
del _sys, _Path, _BTC_WEB
# ──────────────────────────────────────────────────────────────────────────

# ── helpers + date/price utilities ────────────────────────────────────────────
from btc_core._helpers import (
    _lazy_norm, _lazy_linregress, _lazy_QuantReg,
    _fit_shrinking_sigma, _resqr_price_at,
    _parse_ls, qr_price, yr_to_t, today_t, today_year, fmt_price,
    _find_lot_percentile, leo_weighted_entry, fit_qr_from_csv,
    _find_model_data, _DEFAULT_QS,
    _compute_log_r2, compute_model_r2,
)

# ── ModelData + loader ────────────────────────────────────────────────────────
from btc_core._model_data import ModelData, load_model_data

# ── base types ────────────────────────────────────────────────────────────────
from btc_core._base import (
    _ShrinkingBandsMixin, PriceModel, _FitsBasedModel,
    QuantileRegressionModel, _CompositeModel,
)

# ── concrete models ───────────────────────────────────────────────────────────
from btc_core._simple import (
    BubbleModel, PowerLawModel, OffsetPowerLawModel,
    ExponentialModel, StretchedExponentialModel,
    GompertzModel, LogisticSCurveModel, SaturatingPowerLawModel,
    BrokenPowerLawModel, EmpiricalFloorModel, S2FModel, UserModel,
)
from btc_core._lppl import (
    LPPLModel, LPPL2Model, LPPL3Model,
    LPPLModelW, LPPL2ModelW, LPPL3ModelW,
    LPPL4Model, LPPL4ModelW, LPPL4ModelN13, LPPL4ModelWN13,
    LinPPLModel,
)
from btc_core._hybppl_eppl import (
    HybPPLModel, HybPPLDDModel,
    Hyb2LModel, Hyb2CModel, Hyb2BModel, Hyb4DModel,
    HybPPLConfigModel, _HYBPPL_CONFIG_PARAMS,
    EntropyPPLModel, EPPLConfigModel, _EPPL_CONFIG_PARAMS,
)
from btc_core._basis import PCAModel, GreedyModel
