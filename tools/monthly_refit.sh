#!/bin/bash
# Monthly PPL refit — runs on DEV via user systemd timer (quantoshi-refit.timer,
# 1st of the month 01:00 local). Moved here from prod on 2026-09-04: prod's
# quantoshi-ppl-refit.timer patched btc_core/*.py in place on the server, which
# left prod's checkout dirty (git was no longer the source of truth and the next
# commit touching those files would have made every `git pull` deploy refuse).
#
# What it does:
#   1. Syncs the shared deploy worktree (same one daily_update.sh uses) to a
#      detached origin/master, so dev work in the main checkout is untouched.
#   2. Runs tools/refit_all_ppl.py there (~60-90 min, patches class attributes
#      in btc_core/*.py) and removes the .bak files the fit tools leave.
#   3. Gates on the model-side tests (run inside the worktree; the untracked
#      MC / Citadel caches are symlinked from the main checkout so the test
#      environment matches). Any failure -> refit discarded, nothing pushed.
#   4. Commits ONLY btc_core/*.py and pushes origin/master.
# Deploy is deliberately left to the next 06:00 daily_update.sh run: its
# WATCHED_PATHS include btc_core/, it restarts prod and regenerates the Citadel
# cache, and its model_data.pkl rebuild (which also re-fits the resqr bands
# against the new parameters) changes the cache fingerprint, so no manual
# `redis-cli FLUSHDB` is needed for a refit that lands this way.
#
# While running, /tmp/quantoshi-update.disable is held so an over-running refit
# can never collide with the daily job in the same worktree (the daily job just
# skips that morning and self-heals the next day).
#
# Usage:  tools/monthly_refit.sh            # real run
#         tools/monthly_refit.sh --dry-run  # print every state-changing step
set -eo pipefail
DEPLOY_DIR="/scratch/code/bitcoinprojections-deploy"
MAIN_DIR="/scratch/code/bitcoinprojections"
SOURCE_VENV="$MAIN_DIR/btc_venv"
LOG="/tmp/quantoshi-monthly-refit.log"
BRANCH="master"
DISABLE_FILE="/tmp/quantoshi-update.disable"
GATE_TESTS=(btc_web/test_models.py btc_web/test_r2.py btc_web/test_model_registration.py
            btc_web/test_patch_class_attrs.py btc_web/test_resqr_bands.py
            btc_web/test_resqr_runtime.py btc_web/test_timemachine_overrides.py
            btc_web/test_core.py)

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true
run() { if $DRY_RUN; then echo "+ $*"; else "$@"; fi; }

if ! $DRY_RUN; then exec >> "$LOG" 2>&1; fi
echo "──────────────────────────────────────────"
echo "$(date '+%Y-%m-%d %H:%M:%S') — Starting monthly PPL refit (dry_run=$DRY_RUN)"

notify_failure() {
    local msg="$1"
    echo "FAILURE: $msg"
    echo "$msg" | systemd-cat -t quantoshi-refit -p err 2>/dev/null || true
    notify-send -u critical "Quantoshi monthly refit failed" "$msg" 2>/dev/null || true
}

# --- Step 0: hold the daily job off the shared worktree while we run ---
if [[ -f "$DISABLE_FILE" ]]; then
    echo "$DISABLE_FILE already present — another job holds the worktree; exiting"
    exit 0
fi
run touch "$DISABLE_FILE"
trap 'run rm -f "$DISABLE_FILE"' EXIT

# --- Step 1: sync worktree to a detached origin/master (as daily_update.sh) ---
cd "$DEPLOY_DIR"
run git fetch origin "$BRANCH"
run git checkout -f --detach "origin/$BRANCH"
run git reset --hard "origin/$BRANCH"
run git clean -fdq btc_core/          # stray .bak from an aborted run

# untracked caches live only in the main checkout; the gate tests need them
for d in btc_web/mc_cache btc_web/citadel_band_cache; do
    if [[ -d "$MAIN_DIR/$d" && ! -e "$DEPLOY_DIR/$d" ]]; then
        run ln -s "$MAIN_DIR/$d" "$DEPLOY_DIR/$d"
    fi
done

# --- Step 2: refit ---
source "$SOURCE_VENV/bin/activate"
if ! run python3 tools/refit_all_ppl.py; then
    notify_failure "refit_all_ppl.py failed — refit discarded"
    run git checkout -- btc_core/
    exit 1
fi
run rm -f btc_core/*.bak

if $DRY_RUN; then
    echo "+ (gate) python3 -m pytest ${GATE_TESTS[*]} -q"
    echo "+ git add btc_core/*.py && git commit -m 'Monthly PPL refit $(date +%F)' && git push origin HEAD:$BRANCH"
    echo "dry run complete"
    exit 0
fi

# --- Step 3: gate on model-side tests, inside the worktree ---
if ! python3 -m pytest "${GATE_TESTS[@]}" -q; then
    notify_failure "model tests failed after refit — refit discarded, nothing pushed"
    git checkout -- btc_core/
    exit 1
fi

# --- Step 4: commit + push only the parameter files ---
if git diff --quiet -- 'btc_core/*.py'; then
    echo "No parameter changes — nothing to commit"
    exit 0
fi
git add 'btc_core/*.py'
git commit -m "Monthly PPL refit $(date '+%Y-%m-%d')"
if ! git push origin "HEAD:$BRANCH"; then
    notify_failure "git push failed — refit committed in $DEPLOY_DIR but not on origin"
    exit 1
fi
echo "$(date '+%Y-%m-%d %H:%M:%S') — refit pushed as $(git rev-parse --short HEAD); the 06:00 daily job deploys it"
