#!/usr/bin/env bash
# Does the live nginx config still match the copy in this repo?
#
# nginx on the VPS is hand-edited, so the repo copy is a reference that can go
# stale in either direction. This makes noticing that a command rather than a
# discipline. Exits 0 when they agree, 1 when they differ or the host is
# unreachable.
#
#   bash deploy/nginx/check-drift.sh
#
# It does NOT change anything on either side. To push a repo change live, see
# "Applying a change" in the README next to this script.
set -uo pipefail

HOST="${QUANTOSHI_HOST:-root@89.167.70.45}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
rc=0

echo "== sites-enabled/quantoshi =="
if live=$(ssh -o ConnectTimeout=15 "$HOST" "cat /etc/nginx/sites-enabled/quantoshi" 2>/dev/null); then
    if diff -u "$HERE/quantoshi.conf" <(printf '%s\n' "$live") > /tmp/qs-nginx-drift.diff; then
        echo "   in sync"
    else
        echo "   DRIFT (repo < / > live):"
        sed 's/^/   /' /tmp/qs-nginx-drift.diff
        rc=1
    fi
else
    echo "   could not read it from $HOST"
    rc=1
fi

echo "== limit_req_zone lines in nginx.conf =="
if live_zones=$(ssh -o ConnectTimeout=15 "$HOST" "grep -h 'limit_req_zone' /etc/nginx/nginx.conf" 2>/dev/null); then
    # Per LINE: keep only real directives (not the comments that mention
    # limit_req_zone), collapse the VPS's tab indentation, and sort. Doing
    # this with `tr -s '[:space:]'` across the whole file would fold every
    # line into one and match comment prose — an earlier version of this
    # script did exactly that and reported drift that did not exist.
    norm() {
        grep -E '^[[:space:]]*limit_req_zone' \
        | sed 's/[[:space:]]\+/ /g; s/^ //; s/ $//' \
        | sort
    }
    if diff -u <(norm < "$HERE/limit_req_zones.conf") \
               <(printf '%s\n' "$live_zones" | norm) \
               > /tmp/qs-nginx-zones.diff; then
        echo "   in sync"
    else
        echo "   DRIFT:"
        sed 's/^/   /' /tmp/qs-nginx-zones.diff
        rc=1
    fi
else
    echo "   could not read them from $HOST"
    rc=1
fi

echo "== the burst=200 guard on the callback endpoint =="
# The specific line whose loss reintroduces the 429 share-link bug.
if grep -q 'location = /_dash-update-component' "$HERE/quantoshi.conf" \
   && grep -A 12 'location = /_dash-update-component' "$HERE/quantoshi.conf" \
      | grep -q 'burst=200'; then
    echo "   present in the repo copy"
else
    echo "   MISSING from the repo copy — see README, this is the F-429 fix"
    rc=1
fi

exit $rc
