---
name: no_auto_deploy_diagrams
description: When generating visual assets like diagrams, show the user the proposed image and get approval before committing/deploying
type: feedback
---

When generating visual assets (diagrams, charts, images), show the proposed image and wait for explicit approval before committing or deploying. Don't auto-deploy visual work.

**Why:** User found iterative diagram generation frustrating — multiple versions deployed without being good enough. Better to get sign-off first.

**How to apply:** After generating any image asset, show it and ask "Does this look good?" before git add/commit/push/deploy.
