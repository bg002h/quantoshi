#!/bin/bash
# Daily Bitcoin price update — run via cron.
# Updates CSV, re-runs notebook, commits and pushes.
# The production server is NOT restarted automatically.
# Run ~/bin/quantoshi-restart manually when ready.
set -e

cd /scratch/code/bitcoinprojections
LOG="/tmp/quantoshi-daily-update.log"
exec >> "$LOG" 2>&1
echo "──────────────────────────────────────────"
echo "$(date '+%Y-%m-%d %H:%M:%S') — Starting daily update"

# Activate venv
source btc_venv/bin/activate

# Run update (fetches prices, appends CSV, re-runs notebook)
python3 update_prices.py
STATUS=$?

if [ $STATUS -ne 0 ]; then
    echo "update_prices.py failed with exit code $STATUS"
    exit $STATUS
fi

# Check if there are changes to commit
if git diff --quiet BitcoinPricesDaily.csv btc_app/model_data.pkl 2>/dev/null; then
    echo "No new data — nothing to commit."
    exit 0
fi

# Commit and push
git add BitcoinPricesDaily.csv btc_app/model_data.pkl
git commit -m "Daily price update $(date '+%Y-%m-%d')"
git push origin master

echo "$(date '+%Y-%m-%d %H:%M:%S') — Update complete. Run quantoshi-restart to deploy."
