---
name: Unfairly Cheap Line constants
description: Two-point power law floor for Bitcoin — slope=5.510508, intercept=-1.989444 — only 2 breaches in 16 years
type: reference
---

**Unfairly Cheap Line (UCL):** A unique power law floor defined by exactly two points separated by 7.3 years:
- Anchor 1: Sept 21, 2015 ($229.16, t=6.1574)
- Anchor 2: Jan 1, 2023 ($16,905.05, t=13.4374)

**Constants:** slope = 5.510508, intercept = -1.989444

**Derivation:** Of 5,714 daily prices, only these 2 fall below the line. The feasible (slope, intercept) region for exactly these 2 points below is just 0.0026 wide in slope — effectively a unique line. No other pair of points separated by >1 year can serve as a 2-point floor.

**How to apply:** Constants stored in `_app_ctx.py`. Toggle on bubble chart Display section. Scanner shows distance above UCL.
