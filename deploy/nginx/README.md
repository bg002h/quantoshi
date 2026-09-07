# nginx config for quantoshi.xyz

**Reference copies, not a deployment mechanism.** nginx on the VPS is still
edited by hand; these files exist so a server rebuild cannot silently lose
config that a bug was fixed in. Check drift with:

```bash
bash deploy/nginx/check-drift.sh
```

| file | lives on the VPS at |
|---|---|
| `quantoshi.conf` | `/etc/nginx/sites-enabled/quantoshi` |
| `limit_req_zones.conf` | four lines inside the `http { }` block of `/etc/nginx/nginx.conf` |

Certificate paths are Certbot-managed and machine-specific; on a fresh host
run Certbot first and let it write those lines, rather than pasting them.

## The one thing that must not be lost

```nginx
location = /_dash-update-component {
    ...
    limit_req zone=quantoshi burst=200 nodelay;
}
```

Added **2026-09-06**. A share-link restore fires ~150 callback POSTs in a few
seconds. Under the general `location /` limit (`10r/s burst=40`) the tail of
that burst was rejected with 429 — and **the Dash renderer retries only 401 and
403** (`MAX_AUTH_RETRIES`, for JWT refresh), so a rejected callback is simply
dead and whatever chart it was building stays blank forever.

That is what "this share link isn't restoring" turned out to be: not a bad
link, not a decode failure, but nginx dropping the callback that draws the
chart. The operator's own failed attempts were in the access log — 216
rejected callbacks at 19:59–20:00 and 48 at 20:19 — before it was diagnosed.

Sustained rate is unchanged at 10r/s once the bucket drains, so abuse is still
capped; only the page-load burst is absorbed. After the fix the same load
produced **zero** 429s.

Note that nginx `location` blocks do **not** inherit `proxy_*` directives from
a sibling `location /`, which is why this block repeats them.

## Trap: `sites-available/quantoshi` is a stale copy

`/etc/nginx/sites-enabled/quantoshi` is a **regular file, not a symlink**, so
the two have drifted. The disabled `sites-available` copy carries
`limit_req zone=dash_cb burst=10` on the callback endpoint, and `dash_cb` is
**2 r/s** — roughly a fifth of what a single page load needs. Enabling it, or
"restoring from sites-available", breaks **every** share link and every chart
that loads late.

Edit `sites-enabled`. This directory mirrors `sites-enabled`.

## Applying a change

```bash
scp deploy/nginx/quantoshi.conf root@89.167.70.45:/etc/nginx/sites-enabled/quantoshi
ssh root@89.167.70.45 "nginx -t && systemctl reload nginx"
```

`nginx -t` before reload, always: a bad config plus a reload takes the site
down, while a failed test changes nothing.
