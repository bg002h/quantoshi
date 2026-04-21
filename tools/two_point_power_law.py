#!/usr/bin/env python3
"""Power law through first and last data points, plotted log-log and log-linear."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = Path(__file__).resolve().parent.parent

prices = pd.read_csv(ROOT / "BitcoinPricesDaily.csv")
prices["date"] = pd.to_datetime(prices["Date"], format="%m/%d/%y")
blocks = pd.read_csv(ROOT / "BitcoinBlocksDaily.csv", parse_dates=["date"])
df = (prices.merge(blocks, on="date", how="inner")
           .query("Price > 0").dropna(subset=["Price", "blockheight"])
           .sort_values("date").reset_index(drop=True))

LOG_P = np.log10(df["Price"].values)
BLK   = df["blockheight"].values.astype(np.float64)
DATES = mdates.date2num(df["date"].values.astype("datetime64[ms]").astype(object))

last = df.iloc[-1]
lp2, lb2 = np.log10(last["Price"]), np.log10(last["blockheight"])

LINES = [
    dict(row=df.iloc[0],                                          color="#FFB300"),
    dict(row=df[df["date"] == pd.Timestamp("2010-10-07")].iloc[0], color="#44AAFF"),
]

# extend ~6 years into future
blocks_per_day = np.diff(BLK[-60:]).mean()
date_per_blk   = np.diff(DATES[-60:]).mean() / np.diff(BLK[-60:]).mean()
blk_future     = BLK[-1] + 365 * 6 * blocks_per_day
blk_line       = np.linspace(BLK[0], blk_future, 800)
date_line_full = np.where(
    blk_line <= BLK[-1],
    np.interp(blk_line, BLK, DATES),
    DATES[-1] + (blk_line - BLK[-1]) * date_per_blk,
)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Two-point power laws anchored to last data point", fontsize=11)

# ── log-log ───────────────────────────────────────────────────────────────────
ax = axes[0]
ax.scatter(np.log10(BLK), LOG_P, s=2, c="#888", alpha=0.2, linewidths=0, rasterized=True)

for cfg in LINES:
    row = cfg["row"]; col = cfg["color"]
    lp1, lb1 = np.log10(row["Price"]), np.log10(row["blockheight"])
    b = (lp2 - lp1) / (lb2 - lb1)
    a = lp1 - b * lb1
    below = np.mean(LOG_P < a + b * np.log10(BLK))
    label = f"{row['date'].date()}  b={b:.4f}  Q{below*100:.1f}%"
    ax.plot(np.log10(blk_line), a + b * np.log10(blk_line), color=col, lw=2.0, label=label)
    ax.scatter([lb1, lb2], [lp1, lp2], s=80, color=col, zorder=5,
               edgecolors="white", linewidths=0.8)
    print(f"{row['date'].date()}: log10(p) = {a:.4f} + {b:.4f}*log10(blk)  Q{below*100:.1f}% below")

ax.set_xlabel("log₁₀(block)", fontsize=10)
ax.set_ylabel("log₁₀(price  USD)", fontsize=10)
ax.set_title("log-log", fontsize=10)
ax.legend(fontsize=9, loc="upper left")
ax.grid(True, alpha=0.2)

# ── log-linear ────────────────────────────────────────────────────────────────
ax = axes[1]
ax.scatter(DATES, LOG_P, s=2, c="#888", alpha=0.2, linewidths=0, rasterized=True)

date_idx = {row["date"]: np.interp(row["blockheight"], BLK, DATES) for cfg in LINES for row in [cfg["row"]]}
for cfg in LINES:
    row = cfg["row"]; col = cfg["color"]
    lp1, lb1 = np.log10(row["Price"]), np.log10(row["blockheight"])
    b = (lp2 - lp1) / (lb2 - lb1)
    a = lp1 - b * lb1
    ax.plot(date_line_full, a + b * np.log10(np.maximum(blk_line, 1)),
            color=col, lw=2.0, label=f"{row['date'].date()}  b={b:.4f}")
    ax.scatter([date_idx[row["date"]], DATES[-1]], [lp1, lp2], s=80, color=col,
               zorder=5, edgecolors="white", linewidths=0.8)

ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_major_locator(mdates.YearLocator(4))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
ax.set_xlabel("Date", fontsize=10)
ax.set_ylabel("log₁₀(price  USD)", fontsize=10)
ax.set_title("log-linear", fontsize=10)
ax.legend(fontsize=9, loc="upper left")
ax.grid(True, alpha=0.2)

fig.tight_layout()
out = ROOT / "btc_web" / "assets" / "research" / "two_point_power_law.jpg"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
