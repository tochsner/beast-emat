"""Summarises the finished and running chains of a benchmark directory: ESS per million samples after 10% burn-in.

Usage, from the python/ directory of the repo: uv run python <skill>/scripts/analyse.py <benchmark dir> [baseline variant]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.utils.ess import compute_ess  # noqa: E402

BURN_IN = 0.1

bench = Path(sys.argv[1])
baseline = sys.argv[2] if len(sys.argv) > 2 else None

rows = []
for run in sorted((bench / "runs").iterdir()):
    logs = list((run / "data" / "out").glob("*.log"))
    if not logs:
        continue
    df = pd.read_csv(logs[0], sep="\t", comment="#")
    if len(df) < 100:
        print(f"{run.name}: too short ({len(df)} rows)")
        continue

    post = df.iloc[int(len(df) * BURN_IN):]
    millions = (df.iloc[-1, 0] - post.iloc[0, 0]) / 1e6
    ess = {c: compute_ess(post[c].to_numpy(float)) for c in post.columns[1:] if post[c].std() > 0}
    slowest = min(ess, key=ess.get)
    stdout = (run / "stdout.txt").read_text(errors="ignore")
    variant, rest = run.name.split("_", 1)
    rows.append(dict(
        variant=variant,
        run=rest,
        finished=(run / "timing.txt").exists(),
        converged="All logged parameters reached" in stdout,
        samples_M=df.iloc[-1, 0] / 1e6,
        min_ess_per_M=ess[slowest] / millions,
        slowest=slowest,
        posterior_ess_per_M=ess.get("posterior", np.nan) / millions,
    ))

if not rows:
    sys.exit("no run has logged enough samples yet")

runs = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(runs.round(2).to_string(index=False))

geometric_mean = lambda x: float(np.exp(np.log(x).mean()))
summary = runs.groupby("variant").agg(
    n=("run", "count"),
    converged=("converged", "sum"),
    min_ess_per_M=("min_ess_per_M", geometric_mean),
    posterior_ess_per_M=("posterior_ess_per_M", geometric_mean),
)
if baseline in summary.index:
    summary["relative"] = summary.min_ess_per_M / summary.loc[baseline, "min_ess_per_M"]
print()
print(summary.sort_values("min_ess_per_M", ascending=False).round(2).to_string())
