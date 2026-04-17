#!/bin/bash
# Daily Bitcoin price update — run via systemd timer at 6 AM.
# Updates CSV, re-runs notebook, commits, pushes, and auto-deploys.
# Deploy chain: git pull → redis-cli FLUSHDB → restart → regen Citadel cache.
set -eo pipefail

# Manual lockfile escape — touch /tmp/quantoshi-update.disable to skip the
# next scheduled run (e.g. while debugging an in-flight build issue).
if [[ -f /tmp/quantoshi-update.disable ]]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') — daily update disabled via /tmp/quantoshi-update.disable — exiting"
    exit 0
fi

cd /scratch/code/bitcoinprojections
LOG="/tmp/quantoshi-daily-update.log"
exec >> "$LOG" 2>&1
echo "──────────────────────────────────────────"
echo "$(date '+%Y-%m-%d %H:%M:%S') — Starting daily update"

notify_failure() {
    local msg="$1"
    echo "FAILURE: $msg"
    # systemd journal — visible via: systemctl --user status quantoshi-update
    echo "$msg" | systemd-cat -t quantoshi-update -p err
    # desktop notification (if session is active)
    notify-send -u critical "Quantoshi update failed" "$msg" 2>/dev/null || true
}

# Activate venv
source btc_venv/bin/activate

# Run update (fetches prices, appends CSV, re-runs notebook)
if ! python3 update_prices.py; then
    notify_failure "update_prices.py failed"
    exit 1
fi

# Append new rows to BitcoinBlocksDaily.csv so block-mode axis stays in
# sync on prod (prod has no bitcoind and reads the committed CSV as a
# static file). --append is a no-op if no new price rows were added.
if ! python3 tools/build_block_map.py --append; then
    # Don't fail the whole run — block mode will fall back to "unavailable"
    # via /health graceful degradation, and a manual rerun later will fix it.
    notify_failure "build_block_map.py --append failed (block mode may decay)"
fi

# Rebuild the Custom Time Axis $1M projection table (skipped automatically
# if the existing btc_web/_projection_table.json is <28 days old, so this
# refreshes roughly monthly, not daily).
if ! python3 tools/build_projection_table.py; then
    notify_failure "build_projection_table.py failed (modal will show stale data)"
fi

# Check if there are changes to commit
# btc_core.py was split into btc_core/ package on 2026-04-16 (commit 26af8d8)
# — reference the directory form so git diff/add don't error on a missing path
# and halt the daily deploy under `set -eo pipefail`.
if git diff --quiet BitcoinPricesDaily.csv model_data.pkl btc_core/ BitcoinBlocksDaily.csv btc_web/_projection_table.json 2>/dev/null; then
    echo "No new data — nothing to commit."
    exit 0
fi

# Commit and push
git add BitcoinPricesDaily.csv model_data.pkl btc_core/ BitcoinBlocksDaily.csv btc_web/_projection_table.json
git commit -m "Daily price update $(date '+%Y-%m-%d')"
if ! git push origin master; then
    notify_failure "git push failed"
    exit 1
fi

# Deploy to production: pull, restart, regen Citadel cache.
# NOTE: redis-cli FLUSHDB is intentionally OMITTED here. The daily update
# rebuilds model_data.pkl, which bumps the L0/L1/L2 cache-key fingerprint
# (md5 of model_fp + defaults_hash) — so every cached figure keyed against
# the old pkl automatically misses and is recomputed lazily. Flushing Redis
# wasted ~4s of restart time for zero functional benefit. See
# scripts/quantoshi-restart-full for the FLUSHDB path (monthly LPPL refits).
echo "$(date '+%Y-%m-%d %H:%M:%S') — Deploying to production..."
if ssh root@89.167.70.45 "cd /opt/quantoshi && git pull && systemctl restart quantoshi" 2>&1; then
    echo "Production restarted. Regenerating Citadel cache..."
    ssh root@89.167.70.45 "cd /opt/quantoshi && \
        PYTHONPATH='/opt/quantoshi:/opt/quantoshi/btc_web' \
        btc_venv/bin/python3 btc_web/generate_citadel_cache.py" 2>&1 || true
    echo "$(date '+%Y-%m-%d %H:%M:%S') — Deploy complete."
else
    notify_failure "Production deploy failed"
    exit 1
fi
