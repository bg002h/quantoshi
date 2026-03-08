#!/usr/bin/env bash
# run_web.sh — Start the Quantoshi web app using the project venv.
# Usage: bash run_web.sh           # gunicorn (multi-user, production)
#        DEV=1 bash run_web.sh     # Dash dev server (single user, no reloader)
#        PORT=8080 bash run_web.sh # override port
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/btc_venv"
PYTHON="$VENV/bin/python3"
GUNICORN="$VENV/bin/gunicorn"

if [[ ! -f "$PYTHON" ]]; then
    echo "ERROR: venv not found at $VENV"
    echo "  Run: python3 -m venv btc_venv --system-site-packages"
    echo "       btc_venv/bin/pip install -r btc_web/requirements.txt"
    exit 1
fi

PORT="${PORT:-8050}"
export PORT
export PYTHONPATH="$SCRIPT_DIR:$SCRIPT_DIR/btc_app:$SCRIPT_DIR/btc_web"

# Kill any existing process on the port, wait until it's actually free
fuser -k -9 "$PORT/tcp" 2>/dev/null
for _i in 1 2 3 4 5; do
    fuser -s "$PORT/tcp" 2>/dev/null || break
    sleep 1
done

if [[ "${DEV:-0}" == "1" ]]; then
    LOG="/tmp/quantoshi_dev.log"
    echo "Starting Dash dev server on port $PORT..."
    echo "Console output → $LOG"
    exec "$PYTHON" "$SCRIPT_DIR/btc_web/app.py" > "$LOG" 2>&1
else
    echo "Starting gunicorn (5 workers) on port $PORT..."
    exec "$GUNICORN" btc_web.app:server \
        --bind "0.0.0.0:$PORT" \
        --workers 5 \
        --timeout 120 \
        --preload \
        --max-requests 1000 \
        --max-requests-jitter 50 \
        --access-logfile - \
        --error-logfile -
fi
