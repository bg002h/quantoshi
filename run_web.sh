#!/usr/bin/env bash
# run_web.sh — Start the Quantoshi web app using the project venv.
# Usage: bash run_web.sh           # gunicorn (multi-user, production)
#        DEV=1 bash run_web.sh     # Dash dev server (single user, hot-reload)
#        PORT=8080 bash run_web.sh  # override port
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

if [[ "${DEV:-0}" == "1" ]]; then
    echo "Starting Dash dev server on port $PORT..."
    exec "$PYTHON" "$SCRIPT_DIR/btc_web/app.py"
else
    echo "Starting gunicorn (4 workers) on port $PORT..."
    exec "$GUNICORN" btc_web.app:server \
        --bind "0.0.0.0:$PORT" \
        --workers 4 \
        --timeout 120 \
        --preload \
        --access-logfile - \
        --error-logfile -
fi
