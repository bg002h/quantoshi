---
name: Price Scanner tab feature idea
description: New tab or Model Info panel showing percentile-per-model for a user-input price, defaulting to live ticker
type: project
---

Feature idea: "Price Scanner" — user inputs a price (default: live ticker) and date (default: today), sees a table of percentiles across all registered models.

**Prototype output (from this session):**
| Model | Percentile | Median today | Price vs median |
|---|---|---|---|
| QR (raw) | 18.7% | $106,482 | 66.5% |
| Bubble Model | 71.3% | $65,645 | 107.9% |
| Power Law | 26.7% | $108,267 | 65.4% |
| LPPL | 33.7% | $88,372 | 80.2% |
| Exponential | 6.2% | $504,156 | 14.1% |
| BM Empirical Floor | 38.8% | $72,789 | 97.3% |

**Design decisions needed:**
- New tab (after Supercharger) vs panel within Model Info tab
- New tab would shift Stack Tracker→/7, Model Info→/8, FAQ→/9 (breaks deep links)
- Model Info panel avoids renumbering but is less discoverable
- Should include QR (raw) as a row even though it's not a registered PriceModel
- Could add a log-log chart showing all model curves with horizontal line at input price
- Auto-discovers models via PRICE_MODELS iteration

**How to apply:** Use brainstorming skill when starting this feature.
