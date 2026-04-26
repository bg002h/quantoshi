#!/bin/bash
# rebuild_caches.sh — Regenerate MC cache + Citadel band cache on dev, rsync to prod.
#
# These caches are compute-intensive (hours) and too large for git (~1.5 GB).
# Build on dev where CPU is plentiful, then ship to prod via rsync.
#
# When to run:
# - After major model revisions (new model class, parameter-bound changes)
# - After significant price data accumulation (many months of new data)
# - NOT routinely — small LPPL param drift doesn't warrant a rebuild
#
# Usage:
#   tools/rebuild_caches.sh           # build both caches + rsync to prod
#   tools/rebuild_caches.sh --mc      # MC cache only
#   tools/rebuild_caches.sh --citadel # Citadel bands only
#   tools/rebuild_caches.sh --no-deploy  # build only, skip rsync
#
# Prerequisites:
# - btc_venv exists at project root
# - model_data.pkl is current (run update_prices.py first if needed)
# - SSH access to root@89.167.70.45

set -euo pipefail
cd "$(dirname "$0")/.."

BUILD_MC=true
BUILD_CITADEL=true
DEPLOY=true
PROD_HOST="root@89.167.70.45"
PROD_PATH="/opt/quantoshi"

for arg in "$@"; do
    case "$arg" in
        --mc) BUILD_CITADEL=false ;;
        --citadel) BUILD_MC=false ;;
        --no-deploy) DEPLOY=false ;;
        --help|-h)
            grep '^#' "$0" | sed 's/^# *//'
            exit 0 ;;
        *)
            echo "Unknown argument: $arg" >&2
            exit 1 ;;
    esac
done

echo "═══════════════════════════════════════════════════"
echo "Quantoshi cache regeneration"
echo "═══════════════════════════════════════════════════"
echo "MC cache:      $BUILD_MC"
echo "Citadel bands: $BUILD_CITADEL"
echo "Deploy:        $DEPLOY"
echo ""

# ── MC cache (~2-4 hours) ────────────────────────────────────────────────
if [ "$BUILD_MC" = "true" ]; then
    echo "▶ Building MC cache (expect 2-4 hours)..."
    if [ -f btc_web/mc_cache.py ]; then
        # flock prevents two concurrent rebuilds from racing on stash/rename.
        # Auto-creates the lockfile; -n exits immediately if already held.
        LOCKFILE="btc_web/mc_cache/.rebuild.lock"
        mkdir -p "$(dirname "$LOCKFILE")"
        exec 200>"$LOCKFILE"
        if ! flock -n 200; then
            echo "✗ Another rebuild appears to be in progress (lockfile held)."
            echo "  If you're sure no other rebuild is running, delete: $LOCKFILE"
            exit 1
        fi

        PYTHONPATH=".:btc_web" btc_venv/bin/python3 -c "
import _app_ctx
from btc_core import load_model_data
import btc_web.mc_cache as mc

M = load_model_data('model_data.pkl')

mc.stash_stale_files()
try:
    mc.generate_all_caches_parallel(M, n_workers=4)
    mc.commit_stale_files()
except BaseException:
    mc.restore_stale_files()
    raise
"
    else
        echo "MC cache builder not found — skipping."
    fi
fi

# ── Citadel bands (~4 hours) ─────────────────────────────────────────────
if [ "$BUILD_CITADEL" = "true" ]; then
    echo "▶ Building Citadel band cache (expect ~4 hours with 18 workers)..."
    if [ -f tools/generate_citadel_bands.py ]; then
        PYTHONPATH=".:btc_web" btc_venv/bin/python3 tools/generate_citadel_bands.py
    else
        echo "Citadel band generator not found — skipping."
    fi
fi

# ── Deploy via rsync ─────────────────────────────────────────────────────
if [ "$DEPLOY" = "true" ]; then
    echo ""
    echo "▶ Deploying caches to prod ($PROD_HOST)..."
    if [ "$BUILD_MC" = "true" ] && [ -d btc_web/mc_cache ]; then
        echo "  rsync btc_web/mc_cache/ → prod..."
        rsync -avz --progress btc_web/mc_cache/ "$PROD_HOST:$PROD_PATH/btc_web/mc_cache/"
    fi
    if [ "$BUILD_CITADEL" = "true" ] && [ -d btc_web/citadel_band_cache ]; then
        echo "  rsync btc_web/citadel_band_cache/ → prod..."
        rsync -avz --progress btc_web/citadel_band_cache/ "$PROD_HOST:$PROD_PATH/btc_web/citadel_band_cache/"
    fi
    echo ""
    echo "▶ Clearing /dev/shm snapshot + restarting quantoshi on prod..."
    ssh "$PROD_HOST" "rm -f /dev/shm/quantoshi_mc.pkl && redis-cli FLUSHDB && systemctl restart quantoshi"
    echo "✓ Deploy complete."
fi

echo ""
echo "═══════════════════════════════════════════════════"
echo "Done."
echo "═══════════════════════════════════════════════════"
