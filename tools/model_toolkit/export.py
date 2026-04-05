# tools/model_toolkit/export.py
"""Assemble and write model pkl files."""
from __future__ import annotations
import os
import pickle


def build_bm_pkl_dict(price_data, support, composite, comp_by_n, qr, sigma,
                       genesis_date="2009-07-25"):
    """17 keys. String keys for qr_fits. float() wrappers on scalars.

    E6: price_dates/years/prices from df_full (date>=fit_min_date).
    """
    return {
        "qr_fits": {str(k): dict(v) for k, v in qr.fits.items()},
        "QR_QUANTILES": list(qr.fits.keys()),
        "ols_intercept": float(qr.ols_intercept),
        "ols_slope": float(qr.ols_slope),
        "GENESIS_DATE": genesis_date,
        "years_plot_bm": list(composite.t_grid),
        "support_plot_bm": list(composite.support_grid),
        "bm_support_intercept": composite.support_intercept,
        "bm_support_slope": composite.support_slope,
        "bm_comp_by_n": comp_by_n,
        "bm_r2_comp": float(composite.r2),
        "bm_n_future_max": len(comp_by_n) - 1,
        "bm_sigma0_up": float(sigma.sigma0_up),
        "bm_sigma0_down": float(sigma.sigma0_down),
        "bm_alpha_up": float(sigma.alpha_up),
        "bm_alpha_down": float(sigma.alpha_down),
        "price_dates": price_data.df_full["date"].dt.strftime("%Y-%m-%d").tolist(),
        "price_years": price_data.df_full["years"].tolist(),
        "price_prices": price_data.df_full["price"].tolist(),
    }


def build_ef_pkl_dict(support, composite, comp_by_n, sigma, fitted,
                       price_years, price_prices, quantiles,
                       genesis_date="2009-07-25"):
    """EF pkl. Different key names from BM."""
    fitted_params = []
    for b in sorted(fitted, key=lambda b: b["t_rise"]):
        fitted_params.append({k: b.get(k, 0.0) for k in
            ["t_rise", "r", "t_plateau", "t_decay", "d", "K",
             "plat_pow", "dur_rise", "dur_plateau"]})
    return {
        "ef_support_slope": support.slope,
        "ef_support_intercept": support.intercept,
        "genesis": genesis_date,
        "years_plot": composite.t_grid.tolist(),
        "support_plot": composite.support_grid.tolist(),
        "comp_by_n": comp_by_n,
        "bm_r2": float(composite.r2),
        "n_future_max": len(comp_by_n) - 1,
        "sigma0_up": float(sigma.sigma0_up),
        "sigma0_down": float(sigma.sigma0_down),
        "alpha_up": float(sigma.alpha_up),
        "alpha_down": float(sigma.alpha_down),
        "price_years": price_years,
        "price_prices": price_prices,
        "QR_QUANTILES": list(quantiles),
        "fitted_bubbles": fitted_params,
    }


def write_pkl(data, path, protocol=4):
    """Write pkl file with specified protocol."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=protocol)
    print(f"Wrote {path}  ({os.path.getsize(path) // 1024} KB, {len(data)} keys)")
