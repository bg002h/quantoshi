"""Frozen t₀ preset tables for the Custom Time Axis panel.

Load-bearing properties (enforced by btc_web/test_custom_time.py):
- Calendar presets MUST be < 2016-01-01 (fitting-data floor).
- Block presets MUST be < _BLOCK_CAP (computed at custom_fit import time).
- len(CAL_PRESETS) == 6, len(BLK_PRESETS) == 5. Whitepaper has no block equivalent.
"""
from __future__ import annotations

from datetime import date

# (key, date, display label)
CAL_PRESETS: tuple[tuple[str, date, str], ...] = (
    ("whitepaper", date(2008, 10, 31), "Bitcoin whitepaper (2008-10-31)"),
    ("genesis",    date(2009,  1,  3), "Genesis block (2009-01-03)"),
    ("optimal",    date(2009,  7, 25), "Current optimal (2009-07-25)"),
    ("nls",        date(2009, 10,  5), "New Liberty Standard (2009-10-05)"),
    ("pizza",      date(2010,  5, 22), "Bitcoin Pizza Day (2010-05-22)"),
    ("mtgox",      date(2010,  7, 17), "Mt. Gox launch (2010-07-17)"),
)

# (key, blockheight, display label)
BLK_PRESETS: tuple[tuple[str, int, str], ...] = (
    ("block_0",     0,     "Block 0 (genesis)"),
    ("block_3300",  3300,  "Block 3300 (≈ 2009-07-25)"),
    ("block_32000", 32000, "Block 32000 (≈ first dollar trade)"),
    ("block_67700", 67700, "Block 67700 (≈ Pizza Day)"),
    ("block_70000", 70000, "Block 70000 (≈ Mt. Gox)"),
)

CAL_PRESET_BY_KEY = {k: (d, lbl) for k, d, lbl in CAL_PRESETS}
BLK_PRESET_BY_KEY = {k: (h, lbl) for k, h, lbl in BLK_PRESETS}
