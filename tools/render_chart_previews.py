#!/usr/bin/env python3
"""Render static PNG previews of all chart tabs via Plotly + kaleido.

Imports the real app module (which registers all models + caches) and
calls the actual figure builders to generate true-to-life previews.
Saves ~20-100KB PNGs per tab as placeholders shown during initial load.

Run manually or via the daily update pipeline:
    btc_venv/bin/python3 tools/render_chart_previews.py

kaleido is a local dev dependency only — not required on prod.
"""
import os, sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent

# Skip expensive startup paths
os.environ["DEV"] = "1"
os.environ["SKIP_PREWARM"] = "1"

# Must cd into btc_web for app.py's relative imports to work
os.chdir(str(ROOT / "btc_web"))
sys.path.insert(0, str(ROOT / "btc_web"))
sys.path.insert(0, str(ROOT))

ASSETS = ROOT / "btc_web" / "assets"
W, H = 1200, 750


def main():
    print("Rendering chart previews via Plotly + kaleido...")
    # Import app — this registers all models + builds the app context
    import app  # noqa: F401

    from tab_defaults import (bubble_defaults, heatmap_defaults, dca_defaults,
                                retire_defaults, supercharge_defaults, citadel_defaults)
    from utils import (_get_bubble_fig, _get_heatmap_fig, _get_dca_fig,
                       _get_retire_fig, _get_supercharge_fig, _get_citadel_fig)

    charts = [
        ("bubble", _get_bubble_fig, bubble_defaults),
        ("heatmap", _get_heatmap_fig, heatmap_defaults),
        ("dca", _get_dca_fig, dca_defaults),
        ("retire", _get_retire_fig, retire_defaults),
        ("supercharge", _get_supercharge_fig, supercharge_defaults),
        ("citadel", _get_citadel_fig, citadel_defaults),
    ]

    # DCA/Retire/SC default to no models shown (`active_models=[]`) so a
    # naive defaults() call yields a blank "No models selected" placeholder.
    # Force `["bub"]` so the preview shows the canonical median band.
    _MODEL_OVERRIDES = {
        "dca": {"active_models": ["bub"], "show_qr": True},
        "retire": {"active_models": ["bub"], "show_qr": True},
        "supercharge": {"active_models": ["bub"], "show_qr": True},
    }
    for name, fn, defaults in charts:
        try:
            params = dict(defaults())
            params.update(_MODEL_OVERRIDES.get(name, {}))
            result = fn(params)
            fig = result[0] if isinstance(result, tuple) else result
            # Tighter margins for cleaner preview
            fig.update_layout(margin=dict(l=60, r=40, t=40, b=50),
                              width=W, height=H)
            out = ASSETS / f"{name}_preview.png"
            fig.write_image(str(out), width=W, height=H, scale=1)
            # Quantize to 128 colors — shrinks PNG ~4x with imperceptible quality loss
            from PIL import Image as _PILImage
            img = _PILImage.open(out).convert("RGB").quantize(colors=128)
            img.save(out, "PNG", optimize=True)
            size_kb = out.stat().st_size // 1024
            print(f"  {name}: {size_kb} KB")
        except Exception as e:
            print(f"  {name}: SKIPPED — {type(e).__name__}: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
