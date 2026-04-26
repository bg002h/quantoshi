#!/usr/bin/env python3
"""Render 3 mockup PNGs for the /mcideas brainstorming page.

(a) MC-derived quantile bands
(b) Regime-conditional highlight of existing bands
(c) Spaghetti fan of MC paths

Saves to btc_web/assets/mcideas_{a,b,c}.png. Re-run after changing.
"""
import os
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import LineCollection

ROOT = pathlib.Path(__file__).resolve().parent.parent
ASSETS = ROOT / "btc_web" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)
W, H = 9.0, 5.4   # inches
DPI = 100

BG = "#fafaf6"
TXT = "#2c2c2c"
GRID = "#d8d6cf"
ORANGE = "#C48209"
NAVY = "#1B3352"
GREEN = "#1F6B5C"
RED = "#9B2244"

rng = np.random.default_rng(7)


def _baseline(t):
    """Power-law-ish trend (synthetic)."""
    return 0.5 + 0.85 * np.log(t)


def _setup(title):
    fig, ax = plt.subplots(figsize=(W, H), dpi=DPI)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=TXT, length=0)
    ax.grid(True, color=GRID, lw=0.7)
    ax.set_title(title, color=TXT, fontsize=14, fontweight="bold", loc="left",
                 pad=10)
    ax.set_xlabel("Year", color=TXT)
    ax.set_ylabel("log price", color=TXT)
    return fig, ax


def _historical_scatter(ax, t):
    """Add fake historical price scatter (left half of plot)."""
    hist_mask = t < 16
    th = t[hist_mask]
    yh = _baseline(th) + rng.normal(0, 0.18, size=th.size) \
         + 0.3 * np.sin(0.7 * th)
    ax.scatter(th + 2010, yh, s=10, color=TXT, alpha=0.45, edgecolors="none")
    ax.axvline(2010 + 16, color=ORANGE, ls=":", lw=1.0, alpha=0.6)
    ax.text(2010 + 16, ax.get_ylim()[1] * 0.96, "today",
            color=ORANGE, fontsize=9, ha="left", va="top")


def render_a():
    """MC-derived quantile bands (asymmetric, widening with horizon)."""
    fig, ax = _setup("(A) MC-derived quantile bands")
    t = np.linspace(0.5, 28, 200)
    base = _baseline(t)
    ax.plot(t + 2010, base, color=ORANGE, lw=2.5, label="BM trend (median)")

    # Asymmetric MC-derived bands: upper grows faster than lower
    # (regime-asymmetric — bull dispersion > bear)
    horizon_factor = np.clip((t - 14) / 14, 0, None)
    for q_lo, q_hi, alpha, label in [
        (-1.5, 1.8, 0.10, "5th–95th (MC paths)"),
        (-0.8, 1.0, 0.18, "25th–75th"),
    ]:
        lo = base + q_lo * horizon_factor
        hi = base + q_hi * horizon_factor
        ax.fill_between(t + 2010, lo, hi, color=NAVY, alpha=alpha, lw=0,
                        label=label)
    _historical_scatter(ax, t)
    ax.set_xlim(2010, 2038)
    ax.legend(loc="lower right", facecolor=BG, edgecolor=GRID,
              labelcolor=TXT, framealpha=0.9, fontsize=9)
    fig.tight_layout()
    fig.savefig(ASSETS / "mcideas_a.png", dpi=DPI, facecolor=BG)
    plt.close(fig)


def render_b():
    """Regime-conditional: existing bands, one emphasized by current regime."""
    fig, ax = _setup("(B) Regime-conditional highlight")
    t = np.linspace(0.5, 28, 200)
    base = _baseline(t)

    # Standard bands (analytical resqr) — symmetric, dim
    horizon_factor = np.clip((t - 14) / 14, 0, None)
    for q_lo, q_hi, alpha in [
        (-1.5, 1.5, 0.06),
        (-1.0, 1.0, 0.10),
        (-0.5, 0.5, 0.13),
    ]:
        ax.fill_between(t + 2010, base + q_lo * horizon_factor,
                        base + q_hi * horizon_factor,
                        color=NAVY, alpha=alpha, lw=0)

    ax.plot(t + 2010, base, color=NAVY, lw=1.0, alpha=0.4)

    # Regime-emphasized band: current regime is bin 4 (high-momentum) →
    # emphasize 75th-percentile band
    upper = base + 0.95 * horizon_factor
    ax.plot(t + 2010, upper, color=GREEN, lw=2.5, label="75th (regime 4 mode)")
    ax.fill_between(t + 2010, base + 0.55 * horizon_factor, upper,
                    color=GREEN, alpha=0.20, lw=0)
    # Annotation
    ax.annotate("Currently in regime 4\n(high-momentum)\n→ emphasize 75th band",
                xy=(2031, base[120] + 0.7 * horizon_factor[120]),
                xytext=(2024, base[120] + 1.6),
                color=GREEN, fontsize=9, ha="left",
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.0))

    _historical_scatter(ax, t)
    ax.set_xlim(2010, 2038)
    ax.legend(loc="lower right", facecolor=BG, edgecolor=GRID,
              labelcolor=TXT, framealpha=0.9, fontsize=9)
    fig.tight_layout()
    fig.savefig(ASSETS / "mcideas_b.png", dpi=DPI, facecolor=BG)
    plt.close(fig)


def render_c():
    """Spaghetti fan of MC paths (sample of individual sims)."""
    fig, ax = _setup("(C) Spaghetti fan of MC paths")
    t = np.linspace(0.5, 28, 200)
    base = _baseline(t)
    horizon_factor = np.clip((t - 14) / 14, 0, None)

    n_paths = 80
    paths = []
    for _ in range(n_paths):
        steps = rng.normal(0, 0.04, size=t.size)
        # AR(1)-ish to make it look like Markov
        drift = np.cumsum(steps) * horizon_factor
        path = base + drift
        paths.append(path)
    paths = np.array(paths)
    # Color-grade by terminal value
    finals = paths[:, -1]
    norm = (finals - finals.min()) / max(np.ptp(finals), 1e-6)
    cmap = plt.cm.RdYlGn
    for i in range(n_paths):
        ax.plot(t + 2010, paths[i], color=cmap(norm[i]), lw=0.6, alpha=0.45)

    # Median trace
    ax.plot(t + 2010, base, color=ORANGE, lw=2.5, label="BM trend (median)")

    _historical_scatter(ax, t)
    ax.set_xlim(2010, 2038)
    ax.text(2036, base[-1] + 1.2,
            f"{n_paths} sample paths\n(of N=2,000 sims)",
            color=TXT, fontsize=9, ha="right",
            bbox=dict(facecolor=BG, edgecolor=GRID, boxstyle="round,pad=0.4"))
    ax.legend(loc="lower right", facecolor=BG, edgecolor=GRID,
              labelcolor=TXT, framealpha=0.9, fontsize=9)
    fig.tight_layout()
    fig.savefig(ASSETS / "mcideas_c.png", dpi=DPI, facecolor=BG)
    plt.close(fig)


if __name__ == "__main__":
    render_a()
    render_b()
    render_c()
    for f in ("mcideas_a.png", "mcideas_b.png", "mcideas_c.png"):
        sz = (ASSETS / f).stat().st_size // 1024
        print(f"  {f}: {sz} KB")
