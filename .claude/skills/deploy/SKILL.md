---
name: deploy
description: Push current branch to master and deploy to the production Hetzner VPS. Runs tests first, then pushes, SSHs to server, pulls, and restarts the service.
disable-model-invocation: true
---

# Deploy to Production

Deploy the Quantoshi web app to the Hetzner VPS at 89.167.70.45.

## Steps

1. **Pre-flight checks** — Run the test suite and syntax-check all web app files:
   ```bash
   btc_venv/bin/python3 -m py_compile btc_web/app.py btc_web/figures.py btc_web/layout.py btc_web/callbacks.py
   ```
   If any fail, stop and report the error.

2. **Confirm with user** — Show `git log --oneline -5` and `git diff --stat master` so the user can review what will be deployed. Ask for explicit confirmation before proceeding.

3. **Push to master** — Merge current branch to master (or push master directly if already on it):
   ```bash
   git push origin master
   ```

4. **Deploy on server** — SSH to the VPS, pull changes, and restart the service:
   ```bash
   ssh root@89.167.70.45 "cd /opt/quantoshi && git pull && systemctl restart quantoshi"
   ```

5. **Verify** — Check the health endpoint:
   ```bash
   curl -s https://quantoshi.xyz/health | python3 -m json.tool
   ```

## Important
- NEVER deploy without user confirmation
- NEVER force-push to master
- If tests fail, fix them before deploying
- The production server runs gunicorn with 4 workers — restarts cause ~2s downtime
