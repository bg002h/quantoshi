#!/usr/bin/env bash
# Can quantoshi-health's alarms actually fire?
#
# Every check in that script is a claim, and an alarm nobody has seen fire is
# indistinguishable from one that passes. This serves crafted /health payloads
# from a throwaway HTTP server and asserts each alarm appears — and, just as
# importantly, that it does NOT appear on a healthy payload.
#
# Written 2026-09-12 alongside the data-freshness alarm, which exists because
# prod served nine-day-old prices while every other check reported OK.
set -uo pipefail
cd "$(dirname "$0")/.."
PORT=${PORT:-8099}
DIR=$(mktemp -d); trap 'rm -rf "$DIR"; kill %1 2>/dev/null' EXIT
pass=0; fail=0

serve() {  # $1 = json body
    printf '%s' "$1" > "$DIR/health"
    (cd "$DIR" && python3 -m http.server "$PORT" >/dev/null 2>&1) &
    for _ in $(seq 1 40); do
        curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
        sleep 0.25
    done
    echo "could not start fake health server"; exit 1
}

check() {  # $1 = label, $2 = expect-present ("yes"/"no"), $3 = needle, $4 = json
    serve "$4"
    out=$(QUANTOSHI_HEALTH_URL="http://127.0.0.1:$PORT/health" \
          timeout 60 bash scripts/quantoshi-health --verbose 2>&1)
    kill %1 2>/dev/null; wait %1 2>/dev/null
    if [[ "$2" == "yes" ]]; then
        if grep -qi -- "$3" <<<"$out"; then echo "  PASS  $1"; pass=$((pass+1))
        else echo "  FAIL  $1 — expected /$3/, got:"; sed 's/^/        /' <<<"$out"; fail=$((fail+1)); fi
    else
        if grep -qi -- "$3" <<<"$out"; then
            echo "  FAIL  $1 — /$3/ fired when it should not have:"
            sed 's/^/        /' <<<"$out"; fail=$((fail+1))
        else echo "  PASS  $1"; pass=$((pass+1)); fi
    fi
}

ok='"status":"ok","model":true,"data_last_date":"2026-09-11","data_age_days":1,"data_stale_days":3,"data_stale":false,"cache_age_days":48.0,"cache_warn_days":120,"cache_stale_days":180,"cache_warn":false,"cache_stale":false'

echo "sabotage: quantoshi-health alarms"
check "stale DATA fires"              yes "DATA stale"    "{${ok//\"data_stale\":false/\"data_stale\":true}}"
check "stale DATA names the date"     yes "2026-09-11"    "{${ok//\"data_stale\":false/\"data_stale\":true}}"
check "healthy data does NOT fire"    no  "DATA stale"    "{$ok}"
check "cache warn fires"              yes "Cache aging"   "{${ok//\"cache_warn\":false/\"cache_warn\":true}}"
check "cache stale fires"             yes "Cache STALE"   "{${ok//\"cache_stale\":false/\"cache_stale\":true}}"
check "healthy cache does NOT fire"   no  "Cache aging"   "{$ok}"
# the deprecated-alias fallback: prod not yet redeployed still alarms
old='"status":"ok","model":true,"cache_age_days":200.0,"cache_warn_45d":true,"cache_stale_90d":false'
check "old *_45d key still alarms"    yes "Cache aging"   "{$old}"

echo "  ${pass} passed, ${fail} failed"
[[ $fail -eq 0 ]]
