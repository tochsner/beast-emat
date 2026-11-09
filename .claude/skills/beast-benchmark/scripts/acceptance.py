"""Prints the acceptance rate and tuning of every operator from a BEAST state file.

Usage: python3 acceptance.py <runs/<variant>_<dataset>-<seed>/state>

The state file is written every storeEvery samples, so it also works for chains that were
capped or are still running, where stdout.txt has no operator table.
"""

import json
import sys

for line in open(sys.argv[1], errors="ignore"):
    line = line.strip().rstrip(",")
    if not line.startswith('{"id"'):
        continue
    try:
        op = json.loads(line)
    except json.JSONDecodeError:
        continue

    accept, reject = op["accept"], op["reject"]
    total = accept + reject
    rate = accept / total if total else float("nan")
    print(f"{op['id']:70s} tuning={op['p']!s:>12} accept={accept:>9} reject={reject:>9} rate={rate:.3f}")
