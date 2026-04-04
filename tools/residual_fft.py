"""FFT spectrum analysis of model residuals — log-time, window comparison.

Computes FFT of residuals (log10(price) - log10(model_ref)) for 5 models:
  BM floor, BM composite, LPPL, LPPL2, LPPL4
across 6 combinations (daily/weekly sampling x rectangular/Hann/Blackman-Harris windows).
All panels use log-time FFT.

Grid layout: 6 rows (sampling × window) × 5 columns (model residuals).
Each column shows all window/sampling combos for one model — easy window comparison.

Loads the project's own trusted model_data.pkl (built by tools/build_bm_model.py).
"""

from __future__ import annotations

import os
import sys
import pickle as _pickle  # noqa: S403 — loading our own trusted model file

import numpy as np
from scipy.signal import windows as sp_windows
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Setup paths
PROJ_ROOT = "/scratch/code/bitcoinprojections"
sys.path.insert(0, PROJ_ROOT)
sys.path.insert(0, os.path.join(PROJ_ROOT, "btc_web"))

import btc_core as bc


# Theme colors
FIG_BG = "#1a1a2e"
AX_BG = "#16213e"
TEXT = "#cccccc"
GRID = "#2d3a5c"

ROW_COLORS = {
    "BM floor":     "#E8C860",
    "BM composite": "#DAA520",
    "LPPL":         "#FF6D00",
    "LP2":          "#FF9F40",
    "LP4":          "#FFE0A0",
}

# Known LPPL frequencies (log-time angular frequencies)
LPPL_W1 = 7.38
LPPL_W2 = 20.9


def load_data():
    """Load model_data.pkl and build residual reference series."""
    pkl = os.path.join(PROJ_ROOT, "model_data.pkl")
    with open(pkl, "rb") as f:
        md = _pickle.load(f)

    price_years = np.asarray(md["price_years"], float)
    price_prices = np.asarray(md["price_prices"], float)
    years_plot_bm = np.asarray(md["years_plot_bm"], float)
    support_bm = np.asarray(md["support_plot_bm"], float)
    comp_by_n = md["bm_comp_by_n"]
    # n_future=3 composite
    n_fut = min(3, len(comp_by_n) - 1)
    composite = np.asarray(comp_by_n[n_fut], float)

    # Filter to t >= 1.0 years
    mask = price_years >= 1.0
    t = price_years[mask]
    log_p = np.log10(np.maximum(price_prices[mask], 1e-10))

    # BM floor residual
    sup_at_t = np.interp(t, years_plot_bm, support_bm)
    resid_bm_floor = log_p - np.log10(np.maximum(sup_at_t, 1e-10))

    # BM composite residual
    comp_at_t = np.interp(t, years_plot_bm, composite)
    resid_bm_comp = log_p - np.log10(np.maximum(comp_at_t, 1e-10))

    # LPPL models (use median / z=0 shift)
    quantiles = [0.5]
    lppl = bc.LPPLModel(price_years, price_prices, quantiles)
    lp2 = bc.LPPL2Model(price_years, price_prices, quantiles)
    lp4 = bc.LPPL4Model(price_years, price_prices, quantiles)

    lppl_pred = np.asarray(lppl.price_at(0.5, t), float)
    lp2_pred = np.asarray(lp2.price_at(0.5, t), float)
    lp4_log = np.asarray(lp4._lppl_log10(t), float)
    resid_lppl = log_p - np.log10(np.maximum(lppl_pred, 1e-10))
    resid_lp2 = log_p - np.log10(np.maximum(lp2_pred, 1e-10))
    resid_lp4 = log_p - lp4_log

    return {
        "t": t,
        "residuals": {
            "BM floor":     resid_bm_floor,
            "BM composite": resid_bm_comp,
            "LPPL":         resid_lppl,
            "LP2":          resid_lp2,
            "LP4":          resid_lp4,
        },
    }


def _make_window(name: str, n: int) -> np.ndarray:
    if name == "none":
        return np.ones(n)
    if name == "hann":
        return sp_windows.hann(n)
    if name == "blackmanharris":
        return sp_windows.blackmanharris(n)
    raise ValueError(f"unknown window: {name}")


def compute_fft_log(t, y, samples_per_unit_lnt, window_name="hann"):
    """FFT on uniform ln(t) grid with a chosen window. Returns (omega, power)."""
    t0, t1 = float(t[0]), float(t[-1])
    if t0 <= 0:
        t0 = max(t[t > 0][0], 0.01)
    u0, u1 = np.log(t0), np.log(t1)
    n = int(np.ceil((u1 - u0) * samples_per_unit_lnt))
    if n < 16:
        return np.array([]), np.array([])
    u_uni = np.linspace(u0, u1, n)
    t_uni = np.exp(u_uni)
    y_uni = np.interp(t_uni, t, y)
    y_uni = y_uni - np.mean(y_uni)
    window = _make_window(window_name, n)
    y_win = y_uni * window
    fft = np.fft.rfft(y_win)
    power = np.abs(fft) ** 2
    du = (u1 - u0) / (n - 1)
    freqs_cyc = np.fft.rfftfreq(n, d=du)
    omega = 2.0 * np.pi * freqs_cyc
    return omega, power


def find_top_peaks(freqs, power, top_n=3, min_freq=None):
    """Find top-N peaks excluding DC. Returns list of (freq, power)."""
    if len(freqs) < 3:
        return []
    f = freqs[1:]
    p = power[1:]
    if min_freq is not None:
        m = f >= min_freq
        f = f[m]
        p = p[m]
    if len(p) < 3:
        return []
    peaks = []
    for i in range(1, len(p) - 1):
        if p[i] > p[i - 1] and p[i] > p[i + 1]:
            peaks.append((f[i], p[i]))
    peaks.sort(key=lambda x: x[1], reverse=True)
    return peaks[:top_n]


def plot_panel(ax, freqs, power, color, title, peaks):
    """Plot a single log-time FFT power spectrum panel."""
    if len(freqs) == 0:
        ax.text(0.5, 0.5, "insufficient data", transform=ax.transAxes,
                ha="center", va="center", color=TEXT, fontsize=8)
        ax.set_title(title, color=TEXT, fontsize=9)
        return

    f = freqs[1:]
    p = power[1:]
    p_plot = np.maximum(p, 1e-12)

    ax.semilogy(f, p_plot, color=color, linewidth=1.0, alpha=0.95)
    ax.set_facecolor(AX_BG)

    for w, label in [(LPPL_W1, "w1=7.38"), (LPPL_W2, "w2=20.9")]:
        if f[0] <= w <= f[-1]:
            ax.axvline(w, color="#66ccff", linestyle="--",
                       linewidth=0.8, alpha=0.55)
            ax.text(w, p_plot.max() * 0.5, label,
                    color="#66ccff", fontsize=6.5,
                    rotation=90, va="top", ha="right", alpha=0.8)

    for i, (pf, pp) in enumerate(peaks):
        if f[0] <= pf <= f[-1]:
            ax.plot([pf], [pp], marker="v", color="#ff4466",
                    markersize=5, markeredgecolor="white",
                    markeredgewidth=0.3, zorder=5)
            ax.annotate(f"{pf:.2f}", xy=(pf, pp),
                        xytext=(2, 2), textcoords="offset points",
                        color="#ff8899", fontsize=6.5, alpha=0.9)

    ax.set_title(title, color=TEXT, fontsize=9, pad=3)
    ax.tick_params(colors=TEXT, labelsize=7)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(True, color=GRID, alpha=0.3, linewidth=0.5)


def main():
    print("Loading data...")
    data = load_data()
    t = data["t"]
    residuals = data["residuals"]
    print(f"  t range: {t[0]:.3f} .. {t[-1]:.3f} years (N={len(t)})")

    # 5 columns: one per model residual
    cols = ["BM floor", "BM composite", "LPPL", "LP2", "LP4"]

    # 6 rows: sampling × window combos (top to bottom)
    rows = [
        ("Daily · None",             "daily",  "none"),
        ("Daily · Hann",             "daily",  "hann"),
        ("Daily · Blackman-Harris",  "daily",  "blackmanharris"),
        ("Weekly · None",            "weekly", "none"),
        ("Weekly · Hann",            "weekly", "hann"),
        ("Weekly · Blackman-Harris", "weekly", "blackmanharris"),
    ]

    fig, axes = plt.subplots(6, 5, figsize=(20, 20), facecolor=FIG_BG)

    t0, t1 = float(t[0]), float(t[-1])
    lnt_span = np.log(t1) - np.log(t0)

    for i, (row_label, sampling, window_name) in enumerate(rows):
        fpy = 365.0 if sampling == "daily" else 52.0
        n_target = int((t1 - t0) * fpy)
        samples_per_unit = n_target / lnt_span

        for j, col_name in enumerate(cols):
            ax = axes[i, j]
            y = residuals[col_name]
            color = ROW_COLORS[col_name]

            freqs, power = compute_fft_log(t, y, samples_per_unit, window_name)
            peaks = find_top_peaks(freqs, power, top_n=3, min_freq=1.0)

            # Panel title: model name only (row label is on the left Y axis)
            plot_panel(ax, freqs, power, color, col_name, peaks)

            if len(freqs) > 1:
                ax.set_xlim(0, 45.0)

            # Bottom row: x-axis label
            if i == 5:
                ax.set_xlabel("log-time angular freq  omega",
                              color=TEXT, fontsize=8)

            # Left column: row label (sampling × window)
            if j == 0:
                ax.set_ylabel(row_label, color=TEXT, fontsize=8)

    fig.suptitle("Residual FFT Spectrum — Log-time · Window Comparison",
                 color=TEXT, fontsize=15, y=0.998)
    fig.text(0.5, 0.972,
             "Cols: model residuals  ·  Rows: sampling rate × window function  ·  "
             "Blue dashed = LPPL ω₁=7.38, ω₂=20.9  ·  LP4 captures ω≈7, 9, 13, 21",
             ha="center", color="#99aaff", fontsize=9, style="italic")

    plt.tight_layout(rect=(0, 0, 1, 0.965))

    svg_path = os.path.join(PROJ_ROOT, "residual_fft.svg")
    jpg_path = os.path.join(PROJ_ROOT, "residual_fft.jpg")
    fig.savefig(svg_path, facecolor=FIG_BG, edgecolor="none")
    fig.savefig(jpg_path, facecolor=FIG_BG, edgecolor="none",
                dpi=200, pil_kwargs={"quality": 92})
    print(f"Wrote: {svg_path}")
    print(f"Wrote: {jpg_path}")

    print("\n--- Top peaks (log-time, daily sampling, Hann window) ---")
    n_target = int((t1 - t0) * 365.0)
    samples_per_unit = n_target / lnt_span
    for col_name in cols:
        y = residuals[col_name]
        freqs, power = compute_fft_log(t, y, samples_per_unit, "hann")
        peaks = find_top_peaks(freqs, power, top_n=5, min_freq=1.0)
        peaks_str = ", ".join(f"w={p[0]:.2f}" for p in peaks)
        print(f"  {col_name:14s}: {peaks_str}")


if __name__ == "__main__":
    main()
