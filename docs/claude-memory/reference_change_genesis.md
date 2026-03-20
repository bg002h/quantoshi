---
name: change_genesis_procedure
description: Step-by-step procedure for changing the optimal time origin date across the entire Quantoshi codebase
type: reference
---

# Change Optimal Time Origin Procedure

All models use a single time origin date (currently `2009-07-25`). To change it:

## 1. Run the automated script

```bash
python3 scripts/change_origin.py YYYY-MM-DD         # apply changes
python3 scripts/change_origin.py YYYY-MM-DD --dry-run # preview first
```

This patches ~30 locations across:
- SP.ipynb (cells 0/1/3 + axis labels + comment)
- btc_core.py (6 date strings)
- btc_web/layout/model_info.py (formula sections)
- btc_web/layout/faq.py (FAQ entry)
- btc_web/figures/bubble.py (xlabel)
- docs/architecture.md, docs/user_manual.md, CLAUDE.md

Every replacement is assertion-checked. The script auto-detects the current origin from SP.ipynb.

## 2. Execute notebook to regenerate model_data.pkl

```bash
~/.local/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 SP.ipynb
```

## 3. Verify

```bash
btc_venv/bin/python3 -c "import pickle; d=pickle.load(open('archive/btc_app/model_data.pkl','rb')); print('GENESIS_DATE:', d['GENESIS_DATE'])"
PYTHONPATH=".:archive/btc_app:btc_web" btc_venv/bin/python3 -m pytest btc_web/test_web.py -q --tb=short
```

## 4. LPPL constants (if needed)

LPPL constants in btc_core.py are hardcoded and were fit with genesis=2009-07-25. If the new origin is significantly different, refit via differential evolution (`/tmp/fit_lppl2.py` Strategy 3).

## 5. Rebuild MC cache

Build locally on desktop (~10 min with 24 cores), scp to server. Do NOT build on VPS (8+ hours on 2 cores).

## 6. Deploy

```bash
git add -A && git commit -m "Change optimal time origin to YYYY-MM-DD"
git push origin master
ssh root@89.167.70.45 "cd /opt/quantoshi && git pull && systemctl restart quantoshi"
```

## Notes

- The term "economic genesis" was retired — the date is a statistical optimum, not an economic event
- OGPL (Optimal Genesis Power Law) model was removed — it was redundant
- Test suite has genesis-dependent assertions (TestYrToT) — may need tolerance updates
- `--dry-run` is strongly recommended before applying changes
