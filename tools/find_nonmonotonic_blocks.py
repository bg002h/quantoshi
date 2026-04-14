#!/usr/bin/env python3
"""Find a non-monotonic block-timestamp pair in early Bitcoin history.

Output is used as a test fixture for tools/build_block_map.py's running-max
algorithm. Runs once against local bitcoind; the discovered pair (height +
both timestamps) gets hardcoded into btc_web/test_block_map_cli.py.
"""
import json
import subprocess

START, END = 30000, 80000


def rpc(method, *params):
    out = subprocess.run(
        ["bitcoin-cli", method] + [str(p) for p in params],
        capture_output=True, text=True, check=True)
    try:
        return json.loads(out.stdout)
    except json.JSONDecodeError:
        return out.stdout.strip()


def main():
    prev_time = None
    prev_h = None
    for h in range(START, END + 1):
        hsh = rpc("getblockhash", h)
        hdr = rpc("getblockheader", hsh)
        t = hdr["time"]
        if prev_time is not None and t < prev_time:
            print("Non-monotonic pair found:")
            print(f"  block {prev_h}: time={prev_time}")
            print(f"  block {h}:     time={t}")
            print(f"  delta = {t - prev_time} seconds")
            return
        prev_time = t
        prev_h = h
    print(f"No non-monotonic pair in [{START}, {END}]")


if __name__ == "__main__":
    main()
