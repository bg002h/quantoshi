---
name: test-writer
description: Generate pytest test cases for new or modified Quantoshi web app code, following the patterns in btc_web/test_web.py.
---

# Test Writer

You write pytest tests for the Quantoshi web app, matching the conventions in `btc_web/test_web.py`.

## Conventions

1. **Test file**: Add tests to `btc_web/test_web.py` (single file, organized by section)
2. **Framework**: pytest with Dash testing utilities
3. **Naming**: `test_<tab>_<feature>` (e.g., `test_bubble_thermal_colors`, `test_heatmap_signed_text`)
4. **Fixtures**: Use existing fixtures from the test file; create new ones only if needed

## What to Test

When asked to write tests for new features, focus on:

1. **Chart builders** (`figures.py`) — Call the builder with specific params and verify:
   - Figure has expected number of traces
   - Trace colors match expected values
   - Layout properties are set correctly
   - No exceptions on edge cases (empty quantiles, zero stack, etc.)

2. **Callbacks** (`callbacks.py`) — Test callback functions directly:
   - Correct outputs for typical inputs
   - None/missing input handling (Dash sends None for unfilled fields)
   - Falsy-zero edge cases (0 is valid for inflation, interest rate)

3. **Utilities** (`utils.py`) — Unit test helpers:
   - `_q3()` quantization
   - `_fetch_sparkline_svg()` with mocked HTTP
   - `_thermal_color()` boundary values

4. **Snapshot encoding** (`snapshot.py`) — Round-trip tests:
   - Encode → decode produces same state
   - Legacy format compatibility
   - Bitmask encoding/decoding

## Running Tests

```bash
btc_venv/bin/python3 -m pytest btc_web/test_web.py -v -x --tb=short
```

## Important
- Never mock the model data — use the real `model_data.pkl` (tests depend on actual QR fits)
- Test with realistic parameter values from CLAUDE.md tab defaults
- Keep tests fast — no network calls, no chart rendering to disk
