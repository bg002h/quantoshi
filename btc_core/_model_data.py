"""ModelData container + load_model_data entry point.

ModelData unpickles model_data.pkl and lazy-loads per-frequency bundles.
"""

import pickle

import numpy as np
import pandas as pd

from btc_core._helpers import (
    _find_model_data, _parse_ls, _DEFAULT_QS, yr_to_t,
)


class ModelData:
    def __init__(self, path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        self._path = path
        self.qr_fits       = {float(k): v for k, v in d["qr_fits"].items()}
        self.QR_QUANTILES  = [float(q) for q in d["QR_QUANTILES"]]
        self.ols_intercept = d["ols_intercept"]
        self.ols_slope     = d["ols_slope"]
        self.genesis       = pd.Timestamp(d.get("GENESIS_DATE", "2009-07-25"))
        self.years_plot_bm = np.array(d["years_plot_bm"])
        self.support_bm    = np.array(d["support_plot_bm"])
        self.support_intercept = float(d.get("bm_support_intercept", -1.5594))
        self.support_slope     = float(d.get("bm_support_slope", 5.1248))
        self.comp_by_n     = [np.array(c) for c in d["bm_comp_by_n"]]
        self.bm_r2         = d["bm_r2_comp"]
        self.n_future_max  = d["bm_n_future_max"]
        self.price_dates   = d["price_dates"]
        self.price_years   = np.array(d["price_years"])
        self.price_prices  = np.array(d["price_prices"])
        self.qr_colors     = {float(k): v for k, v in d["qr_colors"].items()} if "qr_colors" in d else {}
        raw_ls = d.get("QR_LINESTYLES", {})
        self.qr_linestyles = {float(k): _parse_ls(v) for k, v in raw_ls.items()}
        # Residual QR sigma bands (optional — present only after build_bm_model
        # runs the resqr fit phase).
        self.resqr_coefs = d.get("resqr_coefs", {})
        self.resqr_models = list(d.get("resqr_models", []))
        self.resqr_knots = tuple(d.get("resqr_knots", ()))
        self.resqr_quantiles = list(d.get("resqr_quantiles", []))
        self.resqr_build_ts = d.get("resqr_build_ts", None)
        # Visual config — .get() fallbacks so lean pkls (missing visual keys) don't crash
        _VIS_STR = {
            "PLOT_BG_COLOR": "#FFFFFF", "TEXT_COLOR": "#222222",
            "TITLE_COLOR": "#1A3060", "SPINE_COLOR": "#888888",
            "GRID_MAJOR_COLOR": "#BBBBBB", "GRID_MINOR_COLOR": "#E8E8E8",
            "DATA_COLOR": "#606060",
            "CAGR_SEG_C_LO": "#2166AC", "CAGR_SEG_C_MID1": "#F7F7F7",
            "CAGR_SEG_C_MID2": "#FF8C00", "CAGR_SEG_C_HI": "#CC1100",
        }
        _VIS_INT = {
            "DATA_PT_SIZE": 16, "DATA_PT_SIZE_ZOOM": 32,
            "ZOOM_YEAR_LO": 2025, "ZOOM_YEAR_HI": 2038,
            "CAGR_GRAD_STEPS": 24, "CAGR_HEATMAP_FONTSIZE": 6,
        }
        _VIS_FLOAT = {
            "ZOOM_PRICE_LO": 40000.0, "ZOOM_PRICE_HI": 1750000.0,
            "CAGR_SEG_B1": 5.0, "CAGR_SEG_B2": 16.0,
        }
        for key, default in _VIS_STR.items():
            setattr(self, key, d.get(key, default))
        for key, default in _VIS_INT.items():
            setattr(self, key, int(d.get(key, default)))
        for key, default in _VIS_FLOAT.items():
            setattr(self, key, float(d.get(key, default)))
        self.TABLE_YEARS = d.get("TABLE_YEARS", list(range(2025, 2041)))
        # Shrinking sigma parameters (fitted by tools/fit_sigma.py)
        self.bm_sigma0_up = d.get("bm_sigma0_up", 0.085)
        self.bm_alpha_up = d.get("bm_alpha_up", 0.132)
        self.bm_sigma0_down = d.get("bm_sigma0_down", 0.075)
        self.bm_alpha_down = d.get("bm_alpha_down", 0.218)

    def update_from_csv(self, csv_path):
        df, qr, ols_int, ols_sl = fit_qr_from_csv(
            csv_path, self.QR_QUANTILES, str(self.genesis.date()))
        self.qr_fits       = qr
        self.ols_intercept = ols_int
        self.ols_slope     = ols_sl
        self.price_years   = df["years"].values
        self.price_prices  = df["price"].values
        self.price_dates   = df["date"].dt.strftime("%Y-%m-%d").tolist()

    def save_user_override(self):
        cfg_dir = Path.home() / ".config" / "btc-projections"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        dst = cfg_dir / "model_data.pkl"
        with open(self._path, "rb") as f:
            d = pickle.load(f)
        d["qr_fits"]       = {str(k): v for k, v in self.qr_fits.items()}
        d["ols_intercept"] = self.ols_intercept
        d["ols_slope"]     = self.ols_slope
        d["price_dates"]   = list(self.price_dates)
        d["price_years"]   = list(self.price_years)
        d["price_prices"]  = list(self.price_prices)
        with open(dst, "wb") as f:
            pickle.dump(d, f, protocol=4)
        return str(dst)


def load_model_data(explicit_path=None):
    """Convenience: find model_data.pkl and return a ModelData instance."""
    path = _find_model_data(explicit_path)
    if path is None:
        raise FileNotFoundError(
            "model_data.pkl not found. Run SP.ipynb export cell first, "
            "or pass an explicit path.")
    return ModelData(path)


# ── R² computation ────────────────────────────────────────────────────────────
