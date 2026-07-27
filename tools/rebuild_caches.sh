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
# - SSH access to root@89.167.70.45
# - model_data.pkl current — enforced automatically, see "Preflight" below
#
# Env overrides:
#   MAX_MODEL_AGE_DAYS  abort if model_data.pkl is older than this (default 14)
#   MODEL_SRC_WORKTREE  where to pull a fresher model from (default: auto —
#                       newest model_data.pkl across `git worktree list`)

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

# ── Preflight: model_data.pkl must be current ────────────────────────────
# This script is timer-driven (quantoshi-cache-rebuild.timer), so "run
# update_prices.py first" is not a prerequisite a human can be relied on to
# satisfy. Caches are a function of the model: building from a stale
# model_data.pkl and then rsyncing to prod ships bands/paths that disagree
# with the model prod actually serves — silently, since nothing downstream
# revalidates them.
#
# The daily price automation (daily_update.sh) maintains a current model in
# its own worktree, and this checkout may sit on a feature branch whose
# model is months behind. So: adopt the newest model across all worktrees,
# then refuse to build at all if even that one is too old.
MAX_MODEL_AGE_DAYS="${MAX_MODEL_AGE_DAYS:-14}"

echo "▶ Preflight: checking model freshness..."
newest_pkl="model_data.pkl"
newest_wt=""
if [ -n "${MODEL_SRC_WORKTREE:-}" ]; then
    _worktrees="$MODEL_SRC_WORKTREE"
else
    _worktrees="$(git worktree list --porcelain 2>/dev/null | awk '/^worktree /{print $2}' || true)"
fi
for wt in $_worktrees; do
    cand="$wt/model_data.pkl"
    if [ -f "$cand" ] && [ "$cand" -nt "$newest_pkl" ]; then
        newest_pkl="$cand"
        newest_wt="$wt"
    fi
done

if [ -n "$newest_wt" ]; then
    echo "  model_data.pkl here is $(date -r model_data.pkl +%F) — adopting"
    echo "  $(date -r "$newest_pkl" +%F) copy from $newest_wt"
    # -p keeps the source mtime so the age check below reflects when the
    # model was actually built, not when it was copied.
    cp -p "$newest_pkl" model_data.pkl
    if [ -f "$newest_wt/model_data_ef.pkl" ]; then
        cp -p "$newest_wt/model_data_ef.pkl" model_data_ef.pkl
    fi
else
    echo "  model_data.pkl ($(date -r model_data.pkl +%F)) is already the newest available"
fi

model_age_days=$(( ( $(date +%s) - $(stat -c %Y model_data.pkl) ) / 86400 ))
if [ "$model_age_days" -gt "$MAX_MODEL_AGE_DAYS" ]; then
    echo "" >&2
    echo "✗ ABORT: model_data.pkl is ${model_age_days}d old (limit ${MAX_MODEL_AGE_DAYS}d)." >&2
    echo "  Every worktree's model is stale, so this run would spend hours" >&2
    echo "  building caches against an outdated model and then rsync them" >&2
    echo "  to prod. Refresh the model first (daily_update.sh / update_prices.py)," >&2
    echo "  or re-run with MAX_MODEL_AGE_DAYS=<n> to override deliberately." >&2
    exit 1
fi
echo "  ✓ model is ${model_age_days}d old (limit ${MAX_MODEL_AGE_DAYS}d)"
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
    mc.generate_all_caches_parallel(M)  # n_workers defaults to min(cpu_count, 15)
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
    # FLUSHDB wipes the precomputed Citadel entries too — without this, tab 6
    # falls back to live-compute and renders grayed-out until they repopulate.
    echo "▶ Regenerating Citadel figure cache on prod..."
    ssh "$PROD_HOST" "PYTHONPATH=$PROD_PATH:$PROD_PATH/btc_web \
        $PROD_PATH/btc_venv/bin/python3 $PROD_PATH/btc_web/generate_citadel_cache.py"
    echo "✓ Deploy complete."
fi

echo ""
echo "═══════════════════════════════════════════════════"
echo "Done."
echo "═══════════════════════════════════════════════════"
