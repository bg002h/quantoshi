# Cosmetic UI Harmonization — Phase 1

**Date:** 2026-03-29
**Scope:** Text labels, spacing values, and hint text only. No structural changes to components, callbacks, or layout hierarchy.

---

## Approach

Hybrid (Approach C): Add shared constants in `layout/common.py` for values that genuinely repeat across multiple tabs. Fix one-off inconsistencies in place. No over-engineering.

## New Constants (`layout/common.py`)

```python
_CB_MARGIN = {"marginRight": "4px"}
_INFL_LABEL = "Inflation rate (0–100% / yr)"
_Q_HINT_BASE = "Lower quantiles = more conservative price paths."
```

## Label Standardization

### Amount / Withdrawal Labels

| Tab | Current | New |
|-----|---------|-----|
| DCA | `"Per-period amount ($)"` | `"Purchase amount ($)"` |
| Retire | `"Withdrawal/period ($)"` | `"Withdrawal amount ($)"` |
| Supercharge | `"Withdrawal/period ($)"` | `"Withdrawal amount ($)"` |
| Citadel | `"Spending amount ($ / month)"` | `"Monthly spending ($)"` |

Rationale: Action-specific labels that describe intent. Citadel keeps "Monthly" qualifier since it has no frequency dropdown (always monthly).

### Inflation Labels

| Tab | Current | New |
|-----|---------|-----|
| DCA | `"Inflation rate (0–100% / yr)"` | No change (already correct) |
| Retire | `"Inflation rate (0–100% / yr)"` | No change |
| Supercharge | `"(0–100% / yr)"` (in shared settings card) | No change (range shown) |
| Citadel | `"Inflation rate (% / yr)"` | `"Inflation rate (0–100% / yr)"` via `_INFL_LABEL` |

## Checkbox Margin Normalization

All `inputStyle={"marginRight": "5px"}` → `_CB_MARGIN` (4px).

Already at 4px (swap to constant): Heatmap, Citadel.
Currently at 5px (change value + use constant): DCA, Retire, Supercharge, Stack, Bubble.

Also applies to `layout/mc_controls.py` if any 5px instances exist.

## Quantile Hint Normalization

Two-part structure: `_Q_HINT_BASE` + tab-specific context note.

| Tab | Full Hint |
|-----|-----------|
| Bubble | `"Lower quantiles = more conservative price paths."` |
| DCA | `"Lower quantiles = more conservative price paths. Lower prices mean more sats per period."` |
| Retire | `"Lower quantiles = more conservative price paths. Lower prices mean faster depletion."` |
| Supercharge | `"Lower quantiles = more conservative price paths. Lower prices mean earlier depletion."` |
| Heatmap | `"Lower quantiles = more conservative price paths. Select quantiles for CAGR projection columns."` |

## Intentionally Unchanged

- **Tax toggle** (`dbc.Switch`): Stays as-is. Intentional visual distinction for master enable/disable vs. chart display toggles (`dcc.Checklist`).
- **Year range control types**: Structural (Phase 2).
- **Collapse mechanism differences**: Structural (Phase 2).
- **Frequency control on Supercharge**: Structural (Phase 2).
- **Export row on Stack Tracker**: New feature (Phase 2).
- **Section header/card wrapper patterns**: Structural (Phase 2).
- **Toggle component types** (Checklist vs Switch): Structural (Phase 2).

## Files Touched

| File | Changes |
|------|---------|
| `layout/common.py` | Add `_CB_MARGIN`, `_INFL_LABEL`, `_Q_HINT_BASE` |
| `layout/sim_tabs.py` | Amount labels, margins → `_CB_MARGIN`, hint text (DCA + Retire) |
| `layout/supercharge.py` | Withdrawal label, margins → `_CB_MARGIN`, hint text |
| `layout/citadel.py` | Inflation label → `_INFL_LABEL`, margins → `_CB_MARGIN` |
| `layout/bubble.py` | Margins → `_CB_MARGIN`, hint text |
| `layout/heatmap.py` | Margins → `_CB_MARGIN`, hint text |
| `layout/stack.py` | Margins → `_CB_MARGIN` |
| `layout/mc_controls.py` | Margins → `_CB_MARGIN` (if any 5px instances) |

## Testing

Run `btc_venv/bin/python3 -m pytest btc_web/test_web.py -v`. Label changes should not break tests (tests reference component IDs, not label text). If any tests assert on label strings, update them to match.

## Future Phases

- **Phase 2 (Structural):** Toggle component unification, year range slider consistency, collapse mechanism standardization, frequency control exposure, section header patterns.
- **Phase 3 (Polish):** Export row on Stack Tracker, MC controls naming clarity, mobile responsive improvements.
