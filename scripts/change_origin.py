#!/usr/bin/env python3
"""Change the optimal time origin date across the entire Quantoshi codebase.

Usage:
    python3 scripts/change_origin.py 2009-08-15          # change to new date
    python3 scripts/change_origin.py 2009-08-15 --dry-run # preview changes

Patches ~20 locations across SP.ipynb, btc_core.py, web app layout,
documentation, and chart labels. After running, you must manually:
  1. Execute SP.ipynb to regenerate model_data.pkl
  2. Rebuild MC cache
  3. Run tests
  4. Deploy

See: memory reference 'change_genesis_procedure' or CLAUDE.md for full procedure.
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

# ── Helpers ──────────────────────────────────────────────────────────────────

def replace_checked(src, old, new, expected_count=1, label=""):
    """Replace with assertion on occurrence count."""
    actual = src.count(old)
    if actual != expected_count:
        print(f"  ERROR: {label}: expected {expected_count} occurrences of")
        print(f"         {old!r}")
        print(f"         found {actual}")
        sys.exit(1)
    return src.replace(old, new)


def month_abbr(dt):
    """Return 3-letter month abbreviation: Jan, Feb, ..."""
    return dt.strftime("%b")


def long_date(dt):
    """Return 'Month D, YYYY' format: July 25, 2009."""
    return dt.strftime("%B %d, %Y").replace(" 0", " ")  # strip leading zero from day


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Change optimal time origin date across Quantoshi codebase")
    parser.add_argument("new_date", help="New origin date in YYYY-MM-DD format")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without modifying files")
    args = parser.parse_args()

    new = datetime.strptime(args.new_date, "%Y-%m-%d")
    new_str = new.strftime("%Y-%m-%d")
    new_mon_yr = f"{month_abbr(new)} {new.year}"        # "Jul 2009"
    new_long = long_date(new)                             # "July 25, 2009"
    new_endash = f"{new.year}\u2013{new.month:02d}\u2013{new.day:02d}"  # "2009–07–25"
    dry = args.dry_run

    # Auto-detect old date from SP.ipynb
    root = Path(__file__).resolve().parent.parent
    nb_path = root / "SP.ipynb"

    with open(nb_path) as f:
        nb = json.load(f)

    src0 = "".join(nb["cells"][0]["source"])
    m = re.search(r"genesis\s*=\s*pd\.to_datetime\('(\d{4}-\d{2}-\d{2})'\)", src0)
    if not m:
        print("ERROR: Could not find genesis date in SP.ipynb Cell 0")
        sys.exit(1)

    old = datetime.strptime(m.group(1), "%Y-%m-%d")
    old_str = old.strftime("%Y-%m-%d")
    old_mon_yr = f"{month_abbr(old)} {old.year}"
    old_long = long_date(old)
    old_endash = f"{old.year}\u2013{old.month:02d}\u2013{old.day:02d}"

    if old_str == new_str:
        print(f"Origin is already {old_str}, nothing to do.")
        sys.exit(0)

    print(f"Changing origin: {old_str} → {new_str}")
    if dry:
        print("(DRY RUN — no files will be modified)\n")
    else:
        print()

    changes = 0

    # ── SP.ipynb ─────────────────────────────────────────────────────────

    # Cell 0: genesis variable
    src0 = replace_checked(src0,
        f"genesis     = pd.to_datetime('{old_str}')",
        f"genesis     = pd.to_datetime('{new_str}')",
        1, "Cell 0 genesis variable")
    changes += 1

    # Cell 0: comment (may say "economic genesis" or "optimal time origin")
    comment_old = f"(economic genesis {old_str})"
    comment_new = f"(optimal time origin {new_str})"
    if comment_old not in src0:
        comment_old = f"(optimal time origin {old_str})"
    src0 = replace_checked(src0, comment_old, comment_new, 1, "Cell 0 comment")
    changes += 1

    # Cell 0: xlabels (5 instances)
    src0 = replace_checked(src0,
        f"Years since economic genesis ({old_mon_yr})",
        f"Years since economic genesis ({new_mon_yr})",
        5, "Cell 0 xlabels")
    changes += 5

    # Cell 1: GENESIS_DATE
    src1 = "".join(nb["cells"][1]["source"])
    src1 = replace_checked(src1,
        f"GENESIS_DATE = pd.Timestamp('{old_str}')",
        f"GENESIS_DATE = pd.Timestamp('{new_str}')",
        1, "Cell 1 GENESIS_DATE")
    changes += 1

    # Cell 1: xlabels with month abbreviation (2 instances)
    src1 = replace_checked(src1,
        f"Years since economic genesis ({old_mon_yr})",
        f"Years since economic genesis ({new_mon_yr})",
        2, "Cell 1 xlabels (month)")
    changes += 2

    # Cell 1: xlabels with en-dash date (3 instances)
    src1 = replace_checked(src1,
        f"Years since economic genesis ({old_endash})",
        f"Years since economic genesis ({new_endash})",
        3, "Cell 1 xlabels (en-dash)")
    changes += 3

    # Cell 3: GENESIS_DATE in export dict
    src3 = "".join(nb["cells"][3]["source"])
    src3 = replace_checked(src3,
        f"'GENESIS_DATE':    '{old_str}'",
        f"'GENESIS_DATE':    '{new_str}'",
        1, "Cell 3 GENESIS_DATE")
    changes += 1

    # Cell 3: comment
    src3 = replace_checked(src3,
        f"GENESIS_DATE {old_str}",
        f"GENESIS_DATE {new_str}",
        1, "Cell 3 comment")
    changes += 1

    if not dry:
        nb["cells"][0]["source"] = src0
        nb["cells"][1]["source"] = src1
        nb["cells"][3]["source"] = src3
        with open(nb_path, "w") as f:
            json.dump(nb, f, indent=1)
    print(f"  SP.ipynb: {changes} replacements")

    # ── btc_core.py ──────────────────────────────────────────────────────

    btc_core = root / "archive" / "btc_app" / "btc_core.py"
    src = btc_core.read_text()
    count = src.count(old_str)
    if count == 0:
        print(f"  ERROR: btc_core.py: no occurrences of {old_str}")
        sys.exit(1)
    src = src.replace(old_str, new_str)
    if not dry:
        btc_core.write_text(src)
    print(f"  btc_core.py: {count} replacements")
    changes += count

    # ── model_info.py ────────────────────────────────────────────────────

    mi = root / "btc_web" / "layout" / "model_info.py"
    src = mi.read_text()
    count = src.count(old_str)
    src = src.replace(old_str, new_str)
    if not dry:
        mi.write_text(src)
    print(f"  model_info.py: {count} replacements")
    changes += count

    # ── faq.py ───────────────────────────────────────────────────────────

    faq = root / "btc_web" / "layout" / "faq.py"
    src = faq.read_text()
    # "July 25, 2009" → new long date
    count = src.count(old_long)
    src = src.replace(old_long, new_long)
    if not dry:
        faq.write_text(src)
    print(f"  faq.py: {count} replacements ({old_long} → {new_long})")
    changes += count

    # ── bubble.py xlabel ─────────────────────────────────────────────────

    bub = root / "btc_web" / "figures" / "bubble.py"
    src = bub.read_text()
    src = replace_checked(src,
        f"Years since genesis ({old_str})",
        f"Years since genesis ({new_str})",
        1, "bubble.py xlabel")
    if not dry:
        bub.write_text(src)
    print(f"  bubble.py: 1 replacement")
    changes += 1

    # ── architecture.md ──────────────────────────────────────────────────

    arch = root / "docs" / "architecture.md"
    src = arch.read_text()
    count = src.count(old_str) + src.count(old_long)
    src = src.replace(old_str, new_str)
    src = src.replace(old_long, new_long)
    if not dry:
        arch.write_text(src)
    print(f"  architecture.md: {count} replacements")
    changes += count

    # ── user_manual.md ───────────────────────────────────────────────────

    um = root / "docs" / "user_manual.md"
    src = um.read_text()
    count = src.count(old_long)
    src = src.replace(old_long, new_long)
    if not dry:
        um.write_text(src)
    print(f"  user_manual.md: {count} replacements ({old_long} → {new_long})")
    changes += count

    # ── CLAUDE.md ────────────────────────────────────────────────────────

    claude = root / "CLAUDE.md"
    src = claude.read_text()
    count = src.count(old_str)
    src = src.replace(old_str, new_str)
    if not dry:
        claude.write_text(src)
    print(f"  CLAUDE.md: {count} replacements")
    changes += count

    # ── Summary ──────────────────────────────────────────────────────────

    print(f"\n{'DRY RUN: ' if dry else ''}Total: {changes} replacements across all files")

    if not dry:
        print(f"""
=== AUTOMATED PATCHES COMPLETE ===

Now run these manual steps:

1. Execute notebook to regenerate model_data.pkl:
   ~/.local/bin/jupyter nbconvert \\
       --to notebook --execute --inplace \\
       --ExecutePreprocessor.timeout=600 SP.ipynb

2. Verify pkl:
   btc_venv/bin/python3 -c "import pickle; d=pickle.load(open('archive/btc_app/model_data.pkl','rb')); print('GENESIS_DATE:', d['GENESIS_DATE'])"

3. Run tests:
   PYTHONPATH=".:archive/btc_app:btc_web" btc_venv/bin/python3 -m pytest btc_web/test_web.py -q --tb=short

4. LPPL constants in btc_core.py may need refitting if origin changed significantly.
   Current constants were fit with genesis=2009-07-25.

5. Rebuild MC cache on desktop (10 min), scp to server.
   Do NOT rebuild on VPS (8+ hours on 2 cores).

6. Commit, push, deploy:
   git add -A && git commit -m "Change optimal time origin to {new_str}"
   git push origin master
   ssh root@89.167.70.45 "cd /opt/quantoshi && git pull && systemctl restart quantoshi"
""")


if __name__ == "__main__":
    main()
