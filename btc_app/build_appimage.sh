#!/usr/bin/env bash
# build_appimage.sh — Build the Quantoshi Linux x86_64 AppImage
#
# Prerequisites (install once):
#   pip install pyinstaller   (already done via btc_venv)
#   wget https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage
#   chmod +x appimagetool-x86_64.AppImage
#
# Usage (from btc_app/ directory):
#   bash build_appimage.sh
#   JOBS=8 bash build_appimage.sh   # override CPU count
#
# Output:
#   Quantoshi-x86_64.AppImage

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV="$(dirname "$SCRIPT_DIR")/btc_venv"
PYINSTALLER="$VENV/bin/pyinstaller"
APPIMAGE_TOOL="${APPIMAGE_TOOL:-./appimagetool-x86_64.AppImage}"
APPDIR="$SCRIPT_DIR/AppDir"
OUTPUT="Quantoshi-x86_64.AppImage"

# ── CPU parallelism ───────────────────────────────────────────────────────────
# PyInstaller 6.x has no --jobs flag; we speed it up by pre-compiling all
# .py files in the venv to .pyc in parallel so PyInstaller skips recompilation.
_NCPU="$(nproc 2>/dev/null || echo 4)"
JOBS="${JOBS:-$(( _NCPU < 18 ? _NCPU : 18 ))}"

echo "════════════════════════════════════════════════════════"
echo " Quantoshi — AppImage Builder"
echo "════════════════════════════════════════════════════════"
echo " Script dir : $SCRIPT_DIR"
echo " Venv       : $VENV"
echo " CPUs       : $JOBS / $_NCPU available"
echo " Output     : $OUTPUT"
echo

# ── 0. Sanity checks ─────────────────────────────────────────────────────────
if [[ ! -f "$PYINSTALLER" ]]; then
    echo "ERROR: pyinstaller not found at $PYINSTALLER"
    echo "  Run:  python3 -m venv ../btc_venv --system-site-packages"
    echo "        ../btc_venv/bin/pip install pyinstaller statsmodels pandas"
    exit 1
fi

if [[ ! -f "model_data.pkl" ]]; then
    echo "ERROR: model_data.pkl not found."
    echo "  Run SP.ipynb in Jupyter first — the export cell (cell 3) generates it."
    exit 1
fi

if [[ ! -f "$APPIMAGE_TOOL" ]]; then
    echo "appimagetool not found. Attempting download..."
    TOOL_URL="https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage"
    if command -v wget &>/dev/null; then
        wget -q --show-progress -O appimagetool-x86_64.AppImage "$TOOL_URL"
    elif command -v curl &>/dev/null; then
        curl -L --progress-bar -o appimagetool-x86_64.AppImage "$TOOL_URL"
    else
        echo "ERROR: wget/curl not found. Download manually:"
        echo "  $TOOL_URL"
        exit 1
    fi
    chmod +x appimagetool-x86_64.AppImage
    echo "appimagetool downloaded."
fi

# ── 0b. Pre-compile venv packages in parallel ─────────────────────────────────
# Generates .pyc files so PyInstaller finds them cached instead of recompiling.
echo "── Step 0: Pre-compile packages (${JOBS} jobs) ──────────"
"$VENV/bin/python3" -m compileall -j "$JOBS" -q -x '__pycache__' \
    "$VENV/lib" 2>/dev/null || true
echo "Pre-compilation done."

# ── 1. PyInstaller bundle ─────────────────────────────────────────────────────
echo
echo "── Step 1: PyInstaller ──────────────────────────────────"
rm -rf build dist
UPX_BIN="$(command -v upx 2>/dev/null || true)"
UPX_ARGS=()
if [[ -n "$UPX_BIN" ]]; then
    UPX_ARGS=(--upx-dir "$(dirname "$UPX_BIN")")
    echo "UPX found: $UPX_BIN (will compress binaries)"
else
    echo "UPX not found — binaries will not be compressed"
fi
"$PYINSTALLER" btc_projections.spec --noconfirm "${UPX_ARGS[@]}" 2>&1
echo "PyInstaller done. dist/btc_projections/ created."

# ── 2. Assemble AppDir ───────────────────────────────────────────────────────
echo
echo "── Step 2: Assemble AppDir ──────────────────────────────"
rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/bin"
mkdir -p "$APPDIR/usr/share/applications"
mkdir -p "$APPDIR/usr/share/icons/hicolor/256x256/apps"

# Copy PyInstaller output into AppDir (use rsync for faster parallel I/O)
if command -v rsync &>/dev/null; then
    rsync -a --info=progress2 dist/btc_projections/ "$APPDIR/usr/bin/"
else
    cp -r dist/btc_projections/* "$APPDIR/usr/bin/"
fi

# Desktop entry (required by AppImageTool at AppDir root)
cp btc_projections.desktop "$APPDIR/"
cp btc_projections.desktop "$APPDIR/usr/share/applications/"

# Icon (required at AppDir root as btc_projections.png, and in hicolor tree)
cp btc_projections.png "$APPDIR/"
cp btc_projections.png "$APPDIR/usr/share/icons/hicolor/256x256/apps/"

# AppRun launcher (AppImageTool will symlink if not present, but explicit is safer)
cat > "$APPDIR/AppRun" << 'APPRUN'
#!/usr/bin/env bash
# AppRun — entry point for the AppImage
HERE="$(dirname "$(readlink -f "${0}")")"
export PATH="$HERE/usr/bin:$PATH"
export LD_LIBRARY_PATH="$HERE/usr/bin:${LD_LIBRARY_PATH:-}"
# Qt platform plugin path (Qt5 bundled by PyInstaller)
export QT_QPA_PLATFORM_PLUGIN_PATH="$HERE/usr/bin/PyQt5/Qt5/plugins/platforms"
export QT_PLUGIN_PATH="$HERE/usr/bin/PyQt5/Qt5/plugins"
# Matplotlib config (write to writable location)
export MPLCONFIGDIR="${HOME}/.config/btc-projections/mpl"
mkdir -p "$MPLCONFIGDIR"
exec "$HERE/usr/bin/btc_projections" "$@"
APPRUN
chmod +x "$APPDIR/AppRun"

echo "AppDir assembled at: $APPDIR"
du -sh "$APPDIR"

# ── 3. Build AppImage ─────────────────────────────────────────────────────────
echo
echo "── Step 3: AppImage packaging ───────────────────────────"
rm -f "$OUTPUT"
ARCH=x86_64 "$APPIMAGE_TOOL" "$APPDIR" "$OUTPUT" 2>&1
echo
echo "════════════════════════════════════════════════════════"
echo " Done!  AppImage: $SCRIPT_DIR/$OUTPUT"
ls -lh "$OUTPUT"
echo
echo " To run:  ./$OUTPUT"
echo " To update price data: File > Update Price Data..."
echo " To rebuild after new notebook run:"
echo "   1.  jupyter nbconvert --execute SP.ipynb  (from project root)"
echo "   2.  bash btc_app/build_appimage.sh"
echo "════════════════════════════════════════════════════════"
