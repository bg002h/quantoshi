#!/bin/bash
# Daily Bitcoin price update — runs via user systemd timer at 6 AM.
#
# Architecture (2026-05-20 rewrite):
#   - Operates in a dedicated worktree at DEPLOY_DIR so active dev work in
#     the main checkout is never disrupted.
#   - Auto-detects prod's checked-out branch via ssh, so the script self-heals
#     if you switch prod between feature branches (e.g. time-basis-toggle ↔
#     master) without needing to edit this file.
#   - Commits + pushes automatically. Prompts via kdialog before the
#     ssh-prod deploy step. Approval timeout = APPROVAL_TIMEOUT_SEC.
#   - On decline / timeout: commit remains on origin; user can deploy later
#     via scripts/quantoshi-restart.

set -eo pipefail

DEPLOY_DIR="/scratch/code/bitcoinprojections-deploy"
SOURCE_VENV="/scratch/code/bitcoinprojections/btc_venv"
LOG="/tmp/quantoshi-daily-update.log"
PROD_HOST="root@89.167.70.45"
APPROVAL_TIMEOUT_SEC=82800  # 23 hours (1h buffer before next 6 AM run)

# ── Behavior knobs (flip these two to revert to the conservative posture) ──────
# SETTLE_LAG       : days skipped at the recent end, passed to update_prices.py
#                    --lag. 8 = fully-settled/safe (sources revise up to ~7d);
#                    1 = freshest (yesterday's close), but forward-only appends
#                    lock in less-settled values permanently.
# REQUIRE_APPROVAL : true  = kdialog prompt before deploying to live prod;
#                    false = auto-deploy with no human in the loop.
# Conservative revert: set SETTLE_LAG=8 and REQUIRE_APPROVAL=true.
SETTLE_LAG=1
REQUIRE_APPROVAL=false

# Manual lockfile escape — touch /tmp/quantoshi-update.disable to skip the
# next scheduled run (e.g. while debugging an in-flight build issue).
if [[ -f /tmp/quantoshi-update.disable ]]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') — daily update disabled via /tmp/quantoshi-update.disable — exiting" >> "$LOG"
    exit 0
fi

exec >> "$LOG" 2>&1
echo "──────────────────────────────────────────"
echo "$(date '+%Y-%m-%d %H:%M:%S') — Starting daily update"

notify_failure() {
    local msg="$1"
    echo "FAILURE: $msg"
    echo "$msg" | systemd-cat -t quantoshi-update -p err
    notify-send -u critical "Quantoshi update failed" "$msg" 2>/dev/null || true
}

# --- Step 1: detect prod's checked-out branch (self-healing) ---
PROD_BRANCH=$(ssh -o ConnectTimeout=10 "$PROD_HOST" "cd /opt/quantoshi && git branch --show-current" 2>/dev/null || true)
if [[ -z "$PROD_BRANCH" ]]; then
    notify_failure "couldn't read prod branch via ssh — aborting"
    exit 1
fi
echo "Prod branch: $PROD_BRANCH"

# --- Step 2: sync worktree to prod's branch (detached HEAD) ---
# Use a DETACHED HEAD rather than checking out the branch by name. A git branch
# can be checked out in only ONE worktree at a time, and PROD_BRANCH is normally
# already checked out in the dev main checkout — so `git checkout <branch>` here
# dies with "fatal: '<branch>' is already used by worktree" (exit 128), which
# silently stalled the daily update for days once prod + dev shared a branch.
# Detaching sidesteps the constraint entirely and is robust to any branch
# topology; Step 6 pushes HEAD:PROD_BRANCH to advance the remote branch. `-f`
# discards any leftover uncommitted append from a previous aborted run.
cd "$DEPLOY_DIR"
git fetch origin "$PROD_BRANCH"
git checkout -f --detach "origin/$PROD_BRANCH"
git reset --hard "origin/$PROD_BRANCH"

# --- Step 3: activate shared venv from main checkout ---
source "$SOURCE_VENV/bin/activate"

# --- Step 4: run updaters ---
if ! python3 update_prices.py --lag "$SETTLE_LAG"; then
    notify_failure "update_prices.py failed"
    exit 1
fi

# Append new rows to BitcoinBlocksDaily.csv. Price and block CSVs MUST advance
# together: prod's custom_fit refuses to load when their data-row counts differ
# (it treats a mismatch as corruption). build_block_map reads the LOCAL bitcoind
# RPC, which can hang transiently — so retry, then ABORT before committing if it
# still fails, so a price-only partial update can never ship. The uncommitted
# price append in this worktree is discarded by the next run's `git reset --hard`.
block_ok=false
for attempt in 1 2 3; do
    if python3 tools/build_block_map.py --append; then
        block_ok=true
        break
    fi
    echo "build_block_map --append failed (attempt $attempt/3)"
    if [[ $attempt -lt 3 ]]; then sleep 30; fi
done
if [[ "$block_ok" != true ]]; then
    notify_failure "build_block_map.py --append failed after 3 attempts (bitcoind down?) — aborting before deploy to keep price/block CSVs in sync"
    exit 1
fi

# Custom Time Axis $1M projection table (skipped automatically if existing
# file is <28 days old, so this refreshes roughly monthly).
if ! python3 tools/build_projection_table.py; then
    notify_failure "build_projection_table.py failed (modal will show stale data)"
fi

# --- Step 4b: price/block row-count invariant ---
# prod custom_fit rejects a mismatch as corruption; never commit/deploy one.
# (Belt-and-suspenders: the block-map abort above already prevents the common
# case, but this also catches a partial price append or any other desync.)
price_rows=$(tail -n +2 BitcoinPricesDaily.csv | wc -l)
block_rows=$(tail -n +2 BitcoinBlocksDaily.csv | wc -l)
if [[ "$price_rows" -ne "$block_rows" ]]; then
    notify_failure "price/block row mismatch ($price_rows vs $block_rows) — aborting before commit/deploy"
    exit 1
fi
echo "Invariant OK: price rows = block rows = $price_rows"

# --- Step 5: check for changes ---
# model_data_resqr_diagnostics.json is a build byproduct of build_bm_model.py
# (rebuilt every run alongside model_data.pkl). It MUST be committed with the
# pkl or it drifts stale vs the price history (test_resqr_build.py::
# test_diagnostics_n_samples_matches_price_history fails: n_samples < rows).
WATCHED_PATHS=(BitcoinPricesDaily.csv model_data.pkl model_data_resqr_diagnostics.json btc_core/ BitcoinBlocksDaily.csv btc_web/_projection_table.json)
if git diff --quiet "${WATCHED_PATHS[@]}" 2>/dev/null; then
    echo "No new data — nothing to commit."
    exit 0
fi

# --- Step 6: commit + push (automatic) ---
git add "${WATCHED_PATHS[@]}"
git commit -m "Daily price update $(date '+%Y-%m-%d')"
COMMIT_SHORT=$(git rev-parse --short HEAD)
# Detached HEAD (Step 2) → push the commit onto the remote branch explicitly.
if ! git push origin "HEAD:$PROD_BRANCH"; then
    notify_failure "git push failed"
    exit 1
fi
echo "Commit $COMMIT_SHORT pushed to origin/$PROD_BRANCH"

# --- Step 7: APPROVAL DIALOG before deploying to live prod ---
LAST_PRICE_ROW=$(tail -1 BitcoinPricesDaily.csv)
if [[ "$REQUIRE_APPROVAL" == true ]]; then
    PROMPT="Quantoshi daily update ready.

Branch: $PROD_BRANCH
Commit: $COMMIT_SHORT
New CSV row: $LAST_PRICE_ROW

Deploy live to quantoshi.xyz now?

Yes  →  ssh prod, git pull, restart, regen Citadel cache.
No   →  leave commit on origin; deploy later via scripts/quantoshi-restart."

    # Ensure kdialog has access to user's session bus + display.
    export DISPLAY="${DISPLAY:-:0}"
    export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
    export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=$XDG_RUNTIME_DIR/bus}"

    set +e
    timeout "${APPROVAL_TIMEOUT_SEC}s" kdialog --title "Quantoshi daily deploy" --yesno "$PROMPT"
    KD_RC=$?
    set -e

    case "$KD_RC" in
        0)
            echo "User approved deploy."
            ;;
        1)
            echo "User declined deploy. Commit $COMMIT_SHORT is on origin/$PROD_BRANCH — deploy manually."
            notify-send "Quantoshi" "Daily update commit pushed but NOT deployed (declined). Run scripts/quantoshi-restart when ready."
            exit 0
            ;;
        124)
            echo "Approval timed out after ${APPROVAL_TIMEOUT_SEC}s. Commit $COMMIT_SHORT is on origin/$PROD_BRANCH."
            notify-send -u critical "Quantoshi" "Daily update timed out waiting for approval. Commit on origin/$PROD_BRANCH — deploy manually."
            exit 0
            ;;
        *)
            # kdialog failed (no display? bus down?) — fail-safe: do not deploy.
            notify_failure "kdialog returned $KD_RC — could not prompt. Commit $COMMIT_SHORT pushed but NOT deployed."
            exit 0
            ;;
    esac
else
    echo "Approval gate disabled (REQUIRE_APPROVAL=false) — auto-deploying $COMMIT_SHORT to prod."
fi

# --- Step 8: deploy to prod ---
echo "$(date '+%Y-%m-%d %H:%M:%S') — Deploying to production..."
# redis-cli FLUSHDB intentionally omitted: model_data.pkl rebuild bumps the
# cache fingerprint (md5 of model_fp + defaults_hash), so old entries miss
# naturally. See scripts/quantoshi-restart-full for the FLUSHDB path.
if ssh "$PROD_HOST" "cd /opt/quantoshi && git pull origin '$PROD_BRANCH' && systemctl restart quantoshi"; then
    echo "Production restarted. Regenerating Citadel cache..."
    ssh "$PROD_HOST" "cd /opt/quantoshi && \
        PYTHONPATH='/opt/quantoshi:/opt/quantoshi/btc_web' \
        btc_venv/bin/python3 btc_web/generate_citadel_cache.py" || true
    notify-send "Quantoshi" "Daily update deployed ✓  $LAST_PRICE_ROW"
    echo "$(date '+%Y-%m-%d %H:%M:%S') — Deploy complete."
else
    notify_failure "Production deploy failed"
    exit 1
fi
