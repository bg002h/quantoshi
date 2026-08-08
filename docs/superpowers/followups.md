# Follow-ups

Items deferred out of a plan, each with an **owning phase** so burndown is a
grep rather than a memory. An item whose owning phase has passed is overdue,
not deferred.

Severity follows the project rule: **Critical** / **Important** block a gate;
**Minor** / **Nit** are recorded and batched.

---

## Open

### F-1 — Greedy Select is silently skipped by every refit run
**Severity:** Important · **Owning phase:** next `btc_core/` param maintenance
· **Found:** 2026-08-07, during spl Phase 1 Task 3 · **Pre-existing, not
introduced by that work**

`tools/refit_all_ppl.py:61` stores a command-line flag *inside the script path*:

```python
("Greedy Select",  "tools/fit_grdy.py --mode=de"),
```

Line 86 then does `os.path.exists(script_path)`, which cannot resolve
`"tools/fit_grdy.py --mode=de"` as a filename. The entry is skipped with no
error. It is the only entry in `SCRIPTS` whose path fails to resolve.

**Consequence:** `grdy` has never been refit by the monthly job. Its class
attributes in `btc_core/_basis.py` are frozen at whatever the last manual
`tools/fit_grdy.py --update` wrote, while every other model's have moved.

**Verify in one command:**
```bash
btc_venv/bin/python3 tools/refit_all_ppl.py --dry-run | grep MISSING
```

**Fix sketch:** carry the args separately — `("Greedy Select",
"tools/fit_grdy.py", ["--mode=de"])` — and join them at the subprocess call,
so the existence check sees a real path. Any other entry that grows a flag
later then cannot reintroduce the bug.

**After fixing:** `grdy`'s parameters will change on the next run. `btc_core/`
param changes invalidate nothing automatically (see the cache-invalidation
table in `CLAUDE.md`), so a `redis-cli FLUSHDB` on prod is required, followed
by `generate_citadel_cache.py`.

---

### F-2 — `logi` sigma bands are fitted against an unmasked residual set
**Severity:** Important · **Owning phase:** next `btc_core/` param maintenance
· **Found:** 2026-08-07, during spl Phase 1 Task 3 · **Pre-existing**

`LogisticSCurveModel.__init__` masks its fit to `price_years >= T_MIN`, but the
`logi` branch in `tools/fit_shrinking_sigma.py:219` applies no mask. The sigma
bands are therefore fitted against a **different residual set** than the model
itself was fitted on — the pre-`T_MIN` residuals, which are large and are
exactly what the constructor excludes, are included in the sigma fit.

**Consequence:** `logi`'s quantile band widths are wrong. Silent — the bands
render, they are simply fitted to the wrong residuals.

**Verify in one command:**
```bash
btc_venv/bin/python3 -c "
import pathlib, inspect, sys; sys.path.insert(0,'.')
from btc_core import LogisticSCurveModel as L
s = pathlib.Path('tools/fit_shrinking_sigma.py').read_text()
i = s.index('model_name == \"logi\"'); j = s.index('elif model_name ==', i+10)
print('ctor masks T_MIN :', 'T_MIN' in inspect.getsource(L.__init__))
print('branch masks     :', 'T_MIN' in s[i:j])"
```

**Fix sketch:** mirror the `spl` branch at `tools/fit_shrinking_sigma.py:225-240`,
which does this correctly: import `T_MIN`, compute residuals from the model's
own `_model_log10`, and `return np.where(t >= T_MIN, resid, np.nan)`, letting
the existing downstream NaN drop (`:252`) handle the excluded window.

**Check the other branches while you are there.** Only `plo`, `sexp` and `spl`
were confirmed to mask. The remaining branches (`pl`, `exp`, `lppl`, `hybppl`,
`hybppl_dd`, `eppl`, `grdy`, `gomp`, `bpl`) were **not** audited — whether each
needs a mask depends on whether its own constructor applies one, so this is a
per-model check, not a blanket edit.

---

### F-3 — Generalise the `SCRIPTS` path assertion once F-1 lands
**Severity:** Minor · **Owning phase:** immediately after F-1 · **Depends on
F-1**

`btc_web/test_spl_registration.py` asserts that the **`spl` entry's** script
path resolves on disk. It was deliberately scoped to that one entry, because
generalising it today would go red on F-1 — and forcing an unrelated fix
through a failing test is the wrong way to raise it.

Once F-1 is fixed, widen the assertion to every entry in `SCRIPTS`. That closes
the whole class of bug: a flag appended to any script path becomes a red test
instead of a model that silently stops being refit.

Note the shape is worth remembering — the substring assertion `"fit_spl.py" in
src` was blind to precisely the bug (F-1) that was live three lines away in the
same file it was testing.

---

### F-4 — MC free-tier badge disagrees with what the server charges
**Severity:** Important · **Owning phase:** next payment/MC touch · **Found:**
2026-08-07, during spl Phase 1 Task 4 review · **Pre-existing**

`btc_web/assets/_mc_cost_consts.js` hardcodes the free-tier model list, and it
has drifted from the server's derived list. Both directions are wrong:

| | |
|---|---|
| UI says **free**, server **charges** | `exp`, `lppl` |
| UI says **paid**, server gives free | `ecfg_1d_1u` |

**Consequence:** a user selecting `exp` or `lppl` for Monte Carlo sees no
payment prompt in the UI's cost display, then hits a charge. The reverse case
is milder — `ecfg_1d_1u` is free but advertised as paid, so it is simply never
chosen for free use.

`spl` is in neither list and is therefore unaffected — it is correctly paid on
both sides. That is why this was not introduced by the spl work.

**Verify in one command:**
```bash
btc_venv/bin/python3 - <<'EOF'
import sys, json, re, pathlib
sys.path[:0] = ['btc_web', '.']
import mc_cache
js = pathlib.Path('btc_web/assets/_mc_cost_consts.js').read_text()
client = set(json.loads(re.search(r'"CACHED_MODEL_KEYS"\s*:\s*(\[[^\]]*\])', js).group(1)))
server = set(mc_cache._derive_cached_model_keys())
print('UI free / server charges:', sorted(client - server))
print('UI paid / server free   :', sorted(server - client))
EOF
```

**Fix sketch:** the JS list must be *generated* from `mc_cache._derive_cached_model_keys()`,
the way `_colors_generated.js` is generated from `colors.py` — not hand-kept.
Any hand-maintained mirror of a server-side list will drift again. If a
generator is too much for now, at minimum add a test asserting the two sets are
equal, so the next drift is a red suite rather than a billing surprise.

---

### F-5 — `/mi` and `/faq` cache for 24h, so card edits are invisible to returning readers
**Severity:** Minor · **Owning phase:** next static-pages touch · **Found:**
2026-08-07, Task 7 · **Pre-existing**

`btc_web/static_pages.py:328` sets `Cache-Control: public, max-age=86400` on
the static `/mi` and `/faq` routes. The Dash equivalents (`/9`, `/10`) send
`no-cache, no-store, must-revalidate`.

**Consequence:** after a deploy that changes a Model Info card, anyone who
loaded `/mi` in the previous 24 hours keeps seeing the old copy — including
missing images for assets added in the same deploy. Observed during Task 7: a
first `/mi.30` load replayed a pre-Task-6 copy with the profile figure absent;
cache-busting rendered both.

The long TTL is deliberate and worth keeping — these pages are rendered once at
import, so they only change on deploy, and the bandwidth saving matters for the
onion service.

**Fix sketch:** keep the TTL but add a content-derived `ETag`
(`md5(_STATIC_MI_HTML)`) and honour `If-None-Match`. Returning readers then pay
one conditional request and get a 304 when nothing changed, but see new content
immediately after a deploy. Alternatively `max-age=0, must-revalidate` plus the
ETag — same freshness, marginally more requests.

**Interim:** a deploy that must be visible immediately can be announced as
"hard-refresh `/mi`", or the Dash route `/9.N` linked instead, which never
caches.

---

## Closed

_none yet_
