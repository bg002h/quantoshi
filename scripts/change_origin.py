#!/usr/bin/env python3
"""Change the optimal time origin date across the entire Quantoshi codebase.

Usage:
    python3 scripts/change_origin.py 2009-08-15          # change to new date
    python3 scripts/change_origin.py 2009-08-15 --dry-run # preview changes

Source of truth: `btc_core/_helpers.py` holds the genesis date literal
(post-26af8d8 `btc_core.py` → `btc_core/` package split). The script
auto-detects the current origin from that file and patches every
occurrence across the repo.

SP.ipynb was retired to `debris/` in 2026-03-30 when the model build was
moved to `tools/build_bm_model.py` + `tools/model_toolkit/`. The notebook
is no longer source of truth and is NOT patched by this script — if you
want to keep a legacy reference copy in sync, do it by hand.

After running, you must manually:
  1. Execute `btc_venv/bin/python3 tools/build_bm_model.py` to regenerate
     model_data.pkl (and optionally `tools/build_ef_model.py` for EF).
  2. Rebuild MC cache via `bash tools/rebuild_caches.sh`.
  3. Run tests (`btc_venv/bin/python3 -m pytest btc_web/`).
  4. Deploy.

See: memory reference 'change_genesis_procedure' or CLAUDE.md for full procedure.
"""

import argparse
import json
import re
import sys
from datetime import datetime, timedelta
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

    # Auto-detect old date from btc_core/_helpers.py (source of truth
    # post-26af8d8 package split; SP.ipynb was retired to debris/).
    root = Path(__file__).resolve().parent.parent
    helpers_path = root / "btc_core" / "_helpers.py"
    helpers_src = helpers_path.read_text()
    m = re.search(
        r"genesis\s*=\s*pd\.Timestamp\(\"(\d{4}-\d{2}-\d{2})\"\)",
        helpers_src,
    )
    if not m:
        print("ERROR: Could not find genesis date in btc_core/_helpers.py")
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

    # ── btc_core/_helpers.py ────────────────────────────────────────────
    # (post-26af8d8 split: yr_to_t/today_t live in the helpers submodule)

    btc_core = root / "btc_core" / "_helpers.py"
    src = btc_core.read_text()
    count = src.count(old_str)
    if count == 0:
        print(f"  ERROR: btc_core/_helpers.py: no occurrences of {old_str}")
        sys.exit(1)
    src = src.replace(old_str, new_str)
    if not dry:
        btc_core.write_text(src)
    print(f"  btc_core/_helpers.py: {count} replacements")
    changes += count

    # ── layout/model_info/_items.py ──────────────────────────────────────
    # (post-refactor: model_info.py was split into a package on 2026-04-16.
    #  The genesis-date string literals now live in _items.py.)

    mi = root / "btc_web" / "layout" / "model_info" / "_items.py"
    src = mi.read_text()
    count = src.count(old_str)
    src = src.replace(old_str, new_str)
    if not dry:
        mi.write_text(src)
    print(f"  layout/model_info/_items.py: {count} replacements")
    changes += count

    # ── faq.py ───────────────────────────────────────────────────────────

    faq = root / "btc_web" / "layout" / "faq.py"
    src = faq.read_text()
    # "July 25, 2009" → new long date
    count = src.count(old_long)
    src = src.replace(old_long, new_long)
    # Toggle July 25 analysis block (images + text for 4b, 5, 6)
    begin_marker = "# ── BEGIN JUL25 ANALYSIS (change_origin.py toggles this block) ──"
    end_marker = "# ── END JUL25 ANALYSIS ──"
    jul25_active = new_str == "2009-07-25"

    if begin_marker in src and end_marker in src:
        begin_idx = src.index(begin_marker)
        end_idx = src.index(end_marker) + len(end_marker)
        block = src[begin_idx:end_idx]

        # Check if block is currently commented out
        is_commented = "# DISABLED:" in block

        if jul25_active and is_commented:
            # Uncomment: remove "# DISABLED: " prefix from each line in the block
            lines = block.split("\n")
            restored = []
            for line in lines:
                if line.strip().startswith("# DISABLED: "):
                    restored.append(line.replace("# DISABLED: ", "", 1))
                else:
                    restored.append(line)
            src = src[:begin_idx] + "\n".join(restored) + src[end_idx:]
            print(f"  faq.py: RESTORED Jul 25 analysis block (3 charts + text)")
            changes += 1
        elif not jul25_active and not is_commented:
            # Comment out: prefix each content line with "# DISABLED: "
            lines = block.split("\n")
            disabled = []
            for line in lines:
                if line.strip() in (begin_marker.strip(), end_marker.strip()):
                    disabled.append(line)  # keep markers
                elif line.strip():
                    disabled.append(line.replace(line.lstrip(), "# DISABLED: " + line.lstrip(), 1))
                else:
                    disabled.append(line)
            src = src[:begin_idx] + "\n".join(disabled) + src[end_idx:]
            print(f"  faq.py: DISABLED Jul 25 analysis block (genesis != 2009-07-25)")
            changes += 1
        else:
            state = "active" if not is_commented else "disabled"
            print(f"  faq.py: Jul 25 analysis block already {state}")

    if not dry:
        faq.write_text(src)
    print(f"  faq.py: {count} date replacements ({old_long} → {new_long})")
    changes += count

    # ── bubble.py xlabel ─────────────────────────────────────────────────
    # (The explicit "Years since genesis (YYYY-MM-DD)" xlabel was removed
    # from the bubble figure when chart labels were harmonised. Keep the
    # patch as best-effort — if the target string isn't present, skip
    # silently rather than aborting the whole script.)

    bub = root / "btc_web" / "figures" / "bubble.py"
    if bub.exists():
        src = bub.read_text()
        needle = f"Years since genesis ({old_str})"
        if needle in src:
            src = src.replace(needle, f"Years since genesis ({new_str})", 1)
            if not dry:
                bub.write_text(src)
            print("  bubble.py: 1 replacement")
            changes += 1
        else:
            print("  bubble.py: no xlabel literal — skipped (harmonised away)")

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

2. Clear notebook outputs (keeps file small for git/context):
   ~/.local/bin/jupyter nbconvert \\
       --to notebook --inplace \\
       --ClearOutputPreprocessor.enabled=True SP.ipynb

3. Verify pkl:
   btc_venv/bin/python3 -c "import pickle; d=pickle.load(open('model_data.pkl','rb')); print('GENESIS_DATE:', d['GENESIS_DATE'])"

4. Run tests:
   PYTHONPATH=".:btc_web" btc_venv/bin/python3 -m pytest btc_web/test_web.py -q --tb=short

5. LPPL constants in btc_core/_lppl.py may need refitting if origin changed significantly.
   Current constants were fit with genesis=2009-07-25.

6. Rebuild MC cache on desktop (10 min), scp to server.
   Do NOT rebuild on VPS (8+ hours on 2 cores).

7. Commit, push, deploy:
   git add -A && git commit -m "Change optimal time origin to {new_str}"
   git push origin master
   ssh root@89.167.70.45 "cd /opt/quantoshi && git pull && systemctl restart quantoshi"
""")


if __name__ == "__main__":
    main()
