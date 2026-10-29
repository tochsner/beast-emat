---
name: beast-benchmark
description: How to benchmark BEAST operators or model variants of this repo locally (variant XMLs, parallel capped runs, ESS per sample analysis), and how to register new BEAST classes. Use when asked to benchmark, compare or test new operators, distributions or XML settings on datasets such as zika, rsva or sars-757.
---

# Running BEAST benchmarks

## Registering a new class

A class used as `spec="emat...."` in an XML must be registered as a BEAST service:

1. `src/main/java/module-info.java`: import the class and add it to `provides beast.base.core.BEASTInterface with ...`. A new package also needs an `exports` line.
2. `version.xml`: add `<provider classname="..."/>`. The local runs below use only the module path and `module-info.java`; `version.xml` matters for the installed package.

Before benchmarking a new MCMC move, check that it targets the right distribution. For example, `src/test/java/emat/operators/nodetimes/RatioScaleOperatorTest.java` compares the chain to an exact posterior. Also run a negative control (drop a Hastings term and check that the test fails), which shows that the test can detect a wrong move.

## Setting up a benchmark directory

Keep everything in the scratchpad, one directory per benchmark:

```
bench/
  classes/      snapshot of target/classes, so that later edits don't change running chains
  deps.txt      module path of the dependencies
  xmls/         <variant>_<dataset>.xml
  run.sh        copy of scripts/run.sh of this skill
  jobs.txt      one "<variant> <dataset> <seed> <cap seconds>" per line
  runs/         one directory per chain, created by run.sh
```

```bash
mvn -q -o compile
mvn -q -o dependency:build-classpath -Dmdep.outputFile=$B/deps.txt
cp -R target/classes $B/classes
cp .claude/skills/beast-benchmark/scripts/run.sh $B/
```

**Base XMLs.** `data/xmls/<model>_<dataset>.xml` may be older than the templates in `python/src/data_pipeline/templates/`. To get XMLs from the current templates, render them:

```bash
cd python && uv run python -m src.data_pipeline.cli ../data/fasta/zika.fasta --id zika --models coal_emat --out-dir $B/base --force
```

The models are `bd_emat`, `bsc_emat`, `coal_emat` and `sbd_emat`. Datasets whose dates come from a metadata table also need `--metadata`, `--id-column`, `--date-column` and `--id-field`; see `python/src/data_pipeline/cli.py`. sars-757 and zika use `data/fasta/<id>.fasta`.

**Variant XMLs.** Write a small Python script that swaps the operator element in the base XML with a regex, for example `<operator id="RatioScaleOperator\.[^"]*"[^>]*/>`. Keep the id and the weight, and assert that the replacement actually changed something. The object ids follow the dataset id: `Tree.t:<id>`, `mutations.<id>`, `geneticPrior.<id>`, `ePopSize.t:<id>`.

## Running

- **Pilot first.** Run each new variant for 30–60 s (`./run.sh <variant> <dataset> 99 45`). Check that `stdout.txt` has no exception and that the samples per second look sane.
- **Full runs.** Use at least 3 seeds per variant and shuffle the job order, then run the jobs in the background:
  ```bash
  cd $B && xargs -P 8 -L 1 ./run.sh < jobs.txt > xargs.log 2>&1
  ```
  The templates use `mcmc.ESSStoppingMCMC` (from beast-adaptive-operators), which stops once every logged column has ESS 200 and prints "All logged parameters reached". The cap in `jobs.txt` is a fallback. In the earlier runs, 20–30 min was enough for sars-757 and 12–20 min for zika.
- **Stopping.** Kill only the processes of this benchmark, matching on its directory:
  ```bash
  pkill -f "xargs -P 8 -L 1 ./run.sh"; pkill -f "$B/run.sh"; pkill -f "$B/classes"
  ```
  The user often has their own BEAST runs open, for example from IntelliJ. Check with `pgrep -fl BeastMain` and never kill those.

## Analysing

From `python/`:

```bash
uv run python ../.claude/skills/beast-benchmark/scripts/analyse.py $B <baseline variant>
```

It also works on running chains, which gives preliminary results. It drops 10% as burn-in.

- **Main metric:** ESS per million samples of the slowest logged column, as the geometric mean over seeds and relative to the baseline. Wall time depends on how many other chains or user runs share the machine, so ESS per sample is the fairer comparison. Report wall time to ESS 200, or samples per minute, separately, for variants that cost more per sample.
- **Also report** which column is slowest (often `prior`, `Tree.treeLength`, `CoalescentExponential` or `mutations.count`) and how far each run got. Short runs still contain part of the burn-in transient.
- **Tuned operator state.** At the end of each run, `stdout.txt` holds the operator table, with acceptance rates and tuned parameters. It shows, for example, whether a tuned step got stuck at a bound.
- **One seed is noise.** The spread between seeds is often 1.5–2×.

## Pitfalls

- macOS has no `timeout`; `run.sh` uses a background `sleep`+`kill` instead.
- Background-task exit code 144 after `pkill` is expected.
- Copying `target/classes` before every benchmark keeps runs reproducible. Otherwise, rebuilding mid-benchmark mixes code versions.
