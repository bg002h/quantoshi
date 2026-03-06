#!/bin/bash
# Daily Bitcoin price update — run via cron.
# Updates CSV, re-runs notebook, commits and pushes.
# The production server is NOT restarted automatically.
# Run ~/bin/quantoshi-restart manually when ready.
set -eo pipefail

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

# Check if there are changes to commit
if git diff --quiet BitcoinPricesDaily.csv btc_app/model_data.pkl 2>/dev/null; then
    echo "No new data — nothing to commit."
    exit 0
fi

# Commit and push
git add BitcoinPricesDaily.csv btc_app/model_data.pkl
git commit -m "Daily price update $(date '+%Y-%m-%d')"
if ! git push origin master; then
    notify_failure "git push failed"
    exit 1
fi

echo "$(date '+%Y-%m-%d %H:%M:%S') — Update complete. Run quantoshi-restart to deploy."
