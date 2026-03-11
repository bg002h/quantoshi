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

# Load BTCPay config if present (not in git)
if [[ -f "$SCRIPT_DIR/.env" ]]; then
    set -a; source "$SCRIPT_DIR/.env"; set +a
fi

# Kill any existing process on the port, wait until it's actually free
PIDFILE="/tmp/quantoshi_${PORT}.pid"
if [[ -f "$PIDFILE" ]]; then
    OLD_PID=$(cat "$PIDFILE" 2>/dev/null)
    if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
        kill "$OLD_PID" 2>/dev/null
        sleep 0.25
        kill -0 "$OLD_PID" 2>/dev/null && kill -9 "$OLD_PID" 2>/dev/null
    fi
    rm -f "$PIDFILE"
fi
# Fallback: kill anything still on the port
fuser -k -9 "$PORT/tcp" 2>/dev/null
for _i in 1 2 3 4 5; do
    fuser -s "$PORT/tcp" 2>/dev/null || break
    sleep 0.25
done

_cleanup() {
    local pid
    pid=$(cat "$PIDFILE" 2>/dev/null)
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null
        sleep 0.1
        kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null
    fi
    rm -f "$PIDFILE"
}
trap _cleanup EXIT INT TERM

if [[ "${DEV:-0}" == "1" ]]; then
    LOG="/tmp/quantoshi_dev.log"
    echo "Starting Dash dev server on port $PORT..."
    echo "Console output → $LOG"
    "$PYTHON" "$SCRIPT_DIR/btc_web/app.py" > "$LOG" 2>&1 &
    echo $! > "$PIDFILE"
    wait $!
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
        --access-logformat '%(t)s "%(r)s" %(s)s %(b)s' \
        --error-logfile -
fi
