"""Benchmarks the tree operators of an EMAT analysis.

Derives a set of configurations from a base XML, runs each of them with several seeds
through emat.benchmark.TimedMCMC, and summarises the mixing and the cost of every
operator. The configurations are a leave-one-out analysis, which removes one tree
operator at a time, and a grid over the parameters of the mutation-directed SPR.

Usage, from the python directory after `mvn compile` in the repository root:

    uv run python -m src.benchmarks.tree_operator_benchmark run
    uv run python -m src.benchmarks.tree_operator_benchmark analyse
"""

import argparse
import copy
import json
import re
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from nexwick_py import parse_nexus_file

from src.utils.ess import compute_ess
from src.utils.tree_ess import compute_tree_ess, compute_tree_pseudo_ess

REPOSITORY_PATH = Path(__file__).resolve().parents[3]
TESTS_PATH = REPOSITORY_PATH / "tests"
CLASSES_PATH = REPOSITORY_PATH / "target" / "classes"



@dataclass
class Dataset:
    """A base analysis: its XML, a tree log of its posterior to start from, and the values at which its parameters start."""

    name: str
    partition: str
    base_xml_path: Path
    start_path: Path
    has_dated_tips: bool = False
    start_values: dict[str, str] = field(default_factory=dict)
    # the topology ESS compares all pairs of logged trees, which is slow for large trees
    num_tree_samples: int = 3_000
    # whether the experiments start from the state of a burn-in chain, which itself starts from the start path
    needs_burnin: bool = False

    @property
    def tree_id(self) -> str:
        return f"Tree.t:{self.partition}"

    @property
    def spr_operator_id(self) -> str:
        return f"MutationDirectedSprOperator.{self.partition}"


DATASETS = {
    "hcv": Dataset("hcv", "hcv", TESTS_PATH / "hcv_emat.xml", TESTS_PATH / "hcv_emat_baseline-hcv-1.trees"),
    "zika": Dataset(
        "zika", "zika", TESTS_PATH / "zika_emat.xml", TESTS_PATH / "zika_beauti-zika.trees", has_dated_tips=True,
        # posterior values of the run that the start tree is taken from, as the tree does not fit the initial clock rate
        start_values={"clockRate.c:zika": "7.5E-4", "kappa.s:zika": "16.75", "ePopSize.t:zika": "211.0", "growthRate.t:zika": "1.48"},
    ),
    "sars": Dataset(
        "sars", "ma_sars_cov_2", TESTS_PATH / "sars_emat.xml", TESTS_PATH / "sars_beauti.xml.state", has_dated_tips=True,
        start_values={
            "clockRate.c:ma_sars_cov_2": "8.5E-4", "kappa.s:ma_sars_cov_2": "6.17",
            "ePopSize.t:ma_sars_cov_2": "4.35", "growthRate.t:ma_sars_cov_2": "7.14",
        },
        num_tree_samples=1_500,
        # the run that the start tree is taken from had not converged
        needs_burnin=True,
    ),
}
DATASET = DATASETS["hcv"]

# the output directory per experiment, relative to the benchmark directory
OUTPUT_NAMES = {"operators": "{dataset}", "window": "{dataset}_window", "burnin": "{dataset}_burnin"}
OUTPUT_PATH = TESTS_PATH / "operator_benchmark" / "hcv"

# the operators that change the tree, which are left out one at a time
TREE_OPERATOR_IDS = [
    "IntervalScaleOperator.hcv",
    "RootScaleOperator.hcv",
    "InnerNodeResampleOperator.hcv",
    "MutationDirectedSprOperator.hcv",
    "WilsonBalding.hcv",
]

WINDOW_PROBABILITIES = [0.0, 0.1, 0.3, 0.5, 0.7]
WINDOW_FULL_EXPLORATION_PROBABILITIES = [0.01, 0.5]

ANNEALINGS = [0.4, 0.6, 0.8, 1.0]
FULL_EXPLORATION_PROBABILITIES = [0.01, 0.1, 0.5, 1.0]

CHAIN_LENGTH = 20_000_000
BURNIN_FRACTION = 0.2
NUM_TRACE_SAMPLES = 15_000
SEEDS = [1, 2, 3, 4]
BURNIN_SEED = 1

# the traces whose ESS is reported, by their column in the trace log
TRACE_COLUMNS = ["posterior", "prior", "geneticPrior", "Tree.height", "Tree.treeLength"]


@dataclass
class Configuration:
    """A variant of the base analysis: the operators to remove and the attributes to set on operators."""

    name: str
    experiment: str
    removed_operators: list[str] = field(default_factory=list)
    operator_attributes: dict[str, dict[str, str]] = field(default_factory=dict)
    labels: dict = field(default_factory=dict)


def get_grid_name(annealing: float, full_exploration_probability: float) -> str:
    """Returns the name of the configuration with the given mdSPR parameters."""
    return f"mdspr_a{annealing:g}_f{full_exploration_probability:g}".replace(".", "p")


def get_configurations() -> list[Configuration]:
    """Returns the leave-one-out configurations followed by the grid over the mdSPR parameters."""
    configurations = [Configuration("baseline", "leave_one_out", labels={"removed": "none"})]

    for operator_id in TREE_OPERATOR_IDS:
        short_name = operator_id.split(".")[0]
        configurations.append(Configuration(
            f"without_{short_name}", "leave_one_out",
            removed_operators=[operator_id], labels={"removed": short_name},
        ))

    for annealing in ANNEALINGS:
        for full_exploration_probability in FULL_EXPLORATION_PROBABILITIES:
            configurations.append(Configuration(
                get_grid_name(annealing, full_exploration_probability), "mdspr_grid",
                operator_attributes={"MutationDirectedSprOperator.hcv": {
                    "annealing": str(annealing),
                    "fullExplorationProbability": str(full_exploration_probability),
                }},
                labels={"annealing": annealing, "fullExplorationProbability": full_exploration_probability},
            ))

    # the mix suggested by the two experiments above: less Wilson-Balding and node displacement, more interval scaling and mdSPR
    configurations.append(Configuration(
        "proposed_mix", "proposal",
        operator_attributes={
            "IntervalScaleOperator.hcv": {"weight": "4.0"},
            "InnerNodeResampleOperator.hcv": {"weight": "30.0"},
            "MutationDirectedSprOperator.hcv": {"weight": "20.0"},
            "WilsonBalding.hcv": {"weight": "2.0"},
        },
        labels={"mix": "proposed"},
    ))

    return configurations


def get_window_configurations() -> list[Configuration]:
    """Returns the grid over the window and the full exploration probability of mdSPR, and full explorations only as a reference."""
    grid = [
        (window_probability, full_exploration_probability)
        for full_exploration_probability in WINDOW_FULL_EXPLORATION_PROBABILITIES
        for window_probability in WINDOW_PROBABILITIES
    ]

    configurations = []
    for window_probability, full_exploration_probability in grid + [(0.0, 1.0)]:
        configurations.append(Configuration(
            f"window_q{window_probability:g}_p{full_exploration_probability:g}".replace(".", "p"), "window_grid",
            operator_attributes={DATASET.spr_operator_id: {
                "windowProbability": str(window_probability),
                "fullExplorationProbability": str(full_exploration_probability),
            }},
            labels={"windowProbability": window_probability, "fullExplorationProbability": full_exploration_probability},
        ))
    return configurations


def get_burnin_configurations() -> list[Configuration]:
    """Returns the unchanged base analysis, whose final state the other experiments start from."""
    return [Configuration("burnin", "burnin")]


CONFIGURATIONS = {"operators": get_configurations, "window": get_window_configurations, "burnin": get_burnin_configurations}


# XML generation


def read_start_tree(trees_path: Path) -> str:
    """Returns the last tree of a BEAST tree log as a Newick string with the taxon labels."""
    labels = {}
    last_tree = None
    in_translate = False

    with open(trees_path) as file:
        for line in file:
            stripped = line.strip()
            if stripped.lower().startswith("translate"):
                in_translate = True
            elif in_translate:
                match = re.match(r"(\d+)\s+([^,;]+)[,;]?", stripped)
                if match:
                    labels[match.group(1)] = match.group(2)
                if stripped.endswith(";") or stripped == ";":
                    in_translate = False
            elif stripped.startswith("tree "):
                last_tree = stripped.split("=", 1)[1].strip()

    if last_tree is None:
        raise ValueError(f"no tree found in {trees_path}")

    # drop the metadata comments, then replace the tip numbers, which directly follow a bracket or a comma
    last_tree = re.sub(r"\[[^\]]*\]", "", last_tree)
    return re.sub(r"(?<=[(,])(\d+)(?=:)", lambda match: labels[match.group(1)], last_tree)


def read_start_tree_from_state(state_path: Path, tree_id: str, base_xml_path: Path) -> str:
    """Returns the tree of a BEAST state file as a Newick string with the taxon labels.

    The state file numbers the tips by the order of the sequences in the XML and labels
    the internal nodes with their numbers, which are dropped.
    """
    state = ET.parse(state_path).getroot()
    newick = next(node.text for node in state.findall("statenode") if node.get("id") == tree_id).strip()

    taxa = [sequence.get("taxon") for sequence in ET.parse(base_xml_path).getroot().iter("sequence")]

    newick = re.sub(r"\)\d+", ")", newick)
    newick = re.sub(r"(?<=[(,])(\d+)(?=:)", lambda match: taxa[int(match.group(1))], newick)
    return newick if newick.endswith(";") else newick + ";"


def read_start_values_from_state(state_path: Path) -> dict[str, str]:
    """Returns the values of the scalar parameters of a BEAST state file by their ids."""
    values = {}
    for node in ET.parse(state_path).getroot().findall("statenode"):
        match = re.fullmatch(rf"{re.escape(node.get('id'))}: (\S+)", (node.text or "").strip())
        if match:
            values[node.get("id")] = match.group(1)
    return values


def build_xml(
        configuration: Configuration, seed: int, start_tree: str, start_values: dict[str, str], chain_length: int,
) -> ET.ElementTree:
    """Builds the XML of a configuration: a timed chain that starts at the given tree."""
    document = ET.parse(OUTPUT_PATH / "base.xml")
    run = document.getroot().find("run")
    run_name = get_run_name(configuration, seed)

    run.set("spec", "emat.benchmark.TimedMCMC")
    run.set("chainLength", str(chain_length))
    run.set("storeEvery", str(chain_length))
    run.set("timingFile", f"{run_name}_timing.csv")
    run.set("timingBurnin", str(int(chain_length * BURNIN_FRACTION)))
    run.set("tree", f"@{DATASET.tree_id}")
    run.find("state").set("storeEvery", str(chain_length))

    # start at a tree of the posterior, so that the burn-in is not dominated by the random tree
    for index, child in enumerate(list(run)):
        if child.tag == "init" and child.get("spec") == "RandomTree":
            run.remove(child)
            run.insert(index, ET.Element("init", {
                "id": "StartTree",
                "spec": "beast.base.evolution.tree.TreeParser",
                "initial": f"@{DATASET.tree_id}",
                "taxa": f"@{DATASET.partition}",
                "IsLabelledNewick": "true",
                # the tips of the start tree are already at the heights of their dates
                "adjustTipHeights": str(not DATASET.has_dated_tips).lower(),
                "newick": start_tree,
            }))

    for state_node in run.find("state").findall("stateNode"):
        if state_node.get("id") in start_values:
            state_node.set("value", start_values[state_node.get("id")])

    operators = {operator.get("id"): operator for operator in run.findall("operator")}

    for operator_id in configuration.removed_operators:
        run.remove(operators[operator_id])
    for operator_id, attributes in configuration.operator_attributes.items():
        for key, value in attributes.items():
            operators[operator_id].set(key, value)

    for logger in run.findall("logger"):
        if logger.get("id") == "tracelog":
            logger.set("fileName", f"{run_name}.log")
            logger.set("logEvery", str(chain_length // NUM_TRACE_SAMPLES))
        elif logger.get("mode") == "tree":
            logger.set("fileName", f"{run_name}.trees")
            logger.set("logEvery", str(chain_length // DATASET.num_tree_samples))
        else:
            logger.set("logEvery", str(chain_length // 10))

    return document


def get_run_name(configuration: Configuration, seed: int) -> str:
    return f"{configuration.name}_s{seed}"


# running


def get_module_path() -> str:
    """Returns the module path with the compiled package and its dependencies, as resolved by Maven."""
    classpath_path = OUTPUT_PATH / "classpath.txt"
    if not classpath_path.exists():
        subprocess.run(
            ["mvn", "-q", "dependency:build-classpath", f"-Dmdep.outputFile={classpath_path}"],
            cwd=REPOSITORY_PATH, check=True,
        )
    return f"{CLASSES_PATH}:{classpath_path.read_text().strip()}"


def run_configuration(
        configuration: Configuration, seed: int, start_tree: str, start_values: dict[str, str], chain_length: int, module_path: str,
) -> None:
    """Runs one seed of a configuration, unless its timings were already written."""
    run_name = get_run_name(configuration, seed)
    if (OUTPUT_PATH / f"{run_name}_timing.csv").exists():
        return

    xml_path = OUTPUT_PATH / f"{run_name}.xml"
    build_xml(configuration, seed, start_tree, start_values, chain_length).write(xml_path, encoding="UTF-8", xml_declaration=True)

    start = time.time()
    with open(OUTPUT_PATH / f"{run_name}.out", "w") as output:
        result = subprocess.run(
            [
                "java", "--module-path", module_path, f"-DBEAST_PACKAGE_PATH={CLASSES_PATH}",
                "-m", "beast.base/beast.base.minimal.BeastMain",
                "-seed", str(seed), "-overwrite", xml_path.name,
            ],
            cwd=OUTPUT_PATH, stdout=output, stderr=subprocess.STDOUT,
        )

    status = "done" if result.returncode == 0 else f"FAILED ({result.returncode})"
    print(f"{run_name}: {status} after {time.time() - start:.0f}s", flush=True)


def run(args: argparse.Namespace) -> None:
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    module_path = get_module_path()

    # all runs are derived from the same copy of the base XML, even if the base XML is edited while they run
    base_xml_path = OUTPUT_PATH / "base.xml"
    if not base_xml_path.exists():
        shutil.copyfile(DATASET.base_xml_path, base_xml_path)

    # the experiments of a dataset with a burn-in start from the final state of its burn-in chain

    start_path = DATASET.start_path
    start_values = DATASET.start_values
    if DATASET.needs_burnin and args.experiment != "burnin":
        burnin_name = OUTPUT_NAMES["burnin"].format(dataset=args.dataset)
        start_path = TESTS_PATH / "operator_benchmark" / burnin_name / f"burnin_s{BURNIN_SEED}.xml.state"
        start_values = read_start_values_from_state(start_path)

    start_tree_path = OUTPUT_PATH / "start_tree.newick"
    if not start_tree_path.exists():
        if start_path.suffix == ".state":
            start_tree_path.write_text(read_start_tree_from_state(start_path, DATASET.tree_id, DATASET.base_xml_path))
        else:
            start_tree_path.write_text(read_start_tree(start_path))
    start_tree = start_tree_path.read_text()

    configurations = [
        configuration for configuration in CONFIGURATIONS[args.experiment]()
        if args.only is None or re.search(args.only, configuration.name)
    ]

    # the seeds are the outer loop, so that an interrupted benchmark covers all configurations
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for seed in args.seeds:
            for configuration in configurations:
                executor.submit(run_configuration, configuration, seed, start_tree, start_values, args.chain_length, module_path)


# analysis


def read_timings(path: Path) -> tuple[pd.DataFrame, dict[str, float]]:
    """Reads the per-operator measurements of a run and the totals from the header."""
    totals = {}
    with open(path) as file:
        for line in file:
            if not line.startswith("#"):
                break
            key, value = line[1:].strip().split("=")
            totals[key] = float(value)

    return pd.read_csv(path, comment="#"), totals


def analyse_run(configuration: Configuration, seed: int, output_path: Path) -> dict | None:
    """Computes the ESS values and the operator measurements of one run after the burn-in."""
    run_name = get_run_name(configuration, seed)
    timing_path = output_path / f"{run_name}_timing.csv"
    if not timing_path.exists():
        return None

    timings_df, totals = read_timings(timing_path)
    hours = totals["total_ns"] / 3.6e12

    trace_df = pd.read_csv(output_path / f"{run_name}.log", sep="\t", comment="#")
    chain_length = trace_df["Sample"].iloc[-1]
    burnin = chain_length * BURNIN_FRACTION
    trace_df = trace_df[trace_df["Sample"] >= burnin]

    ess = {column: compute_ess(trace_df[column].to_numpy()) for column in TRACE_COLUMNS}

    trees, _ = parse_nexus_file(str(output_path / f"{run_name}.trees"))
    trees = trees[int(len(trees) * BURNIN_FRACTION):]
    ess["topology"] = compute_tree_ess(trees, cap_at_n=True)
    ess["topology_rooted"] = compute_tree_ess(trees, rooted=True, cap_at_n=True)
    ess["topology_pseudo_median"] = compute_tree_pseudo_ess(trees, seed=0)["median"]

    operators = []
    for row in timings_df.itertuples():
        if row.proposals == 0:
            continue
        operators.append({
            "operator": row.operator.split(".")[0],
            "share": row.proposals / totals["steps"],
            "acceptance": row.accepted / row.proposals,
            "direct_rejects": row.direct_rejects / row.proposals,
            "topology_changes": row.topology_changes / row.proposals,
            "proposal_us": row.proposal_ns / row.proposals / 1e3,
            # only the proposals that were not rejected directly are evaluated
            "evaluation_us": row.evaluation_ns / max(row.proposals - row.direct_rejects, 1) / 1e3,
            "resolution_us": row.resolution_ns / row.proposals / 1e3,
            "step_us": (row.proposal_ns + row.evaluation_ns + row.resolution_ns) / row.proposals / 1e3,
            "time_share": (row.proposal_ns + row.evaluation_ns + row.resolution_ns) / totals["total_ns"],
        })

    num_steps = int(totals["steps"])
    return {
        "configuration": configuration.name,
        "experiment": configuration.experiment,
        "labels": configuration.labels,
        "seed": seed,
        "steps": num_steps,
        "hours": hours,
        "step_us": totals["total_ns"] / num_steps / 1e3,
        "num_trace_samples": len(trace_df),
        "num_tree_samples": len(trees),
        "ess": ess,
        "ess_per_hour": {key: value / hours for key, value in ess.items()},
        "ess_per_million_steps": {key: value / num_steps * 1e6 for key, value in ess.items()},
        "topology_changes_per_second": timings_df["topology_changes"].sum() / (totals["total_ns"] / 1e9),
        "mdspr_local_regions": totals.get("mdspr_local_regions"),
        "operators": operators,
    }


def analyse(args: argparse.Namespace) -> None:
    configurations = CONFIGURATIONS[args.experiment]()

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(analyse_run, configuration, seed, OUTPUT_PATH)
            for configuration in configurations for seed in args.seeds
        ]
        results = [future.result() for future in futures]
    results = [result for result in results if result is not None]

    results_path = OUTPUT_PATH / "results.json"
    results_path.write_text(json.dumps(results, indent=1))
    print(f"wrote {len(results)} runs to {results_path}")

    # print the mean over the seeds of the main measures per configuration

    rows = []
    for result in results:
        rows.append({
            "configuration": result["configuration"],
            "step_us": result["step_us"],
            **{f"{key}/h": value for key, value in result["ess_per_hour"].items()},
            "topology_changes/s": result["topology_changes_per_second"],
        })
    summary_df = pd.DataFrame(rows).groupby("configuration", sort=False).mean()
    print(summary_df.round(1).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="generate the XMLs and run the missing configurations")
    run_parser.add_argument("--workers", type=int, default=4, help="the number of chains that run at the same time")
    run_parser.add_argument("--chain-length", type=int, default=CHAIN_LENGTH)
    run_parser.add_argument("--only", help="a regular expression that the names of the configurations to run must match")
    run_parser.set_defaults(function=run)

    analyse_parser = subparsers.add_parser("analyse", help="summarise the finished runs into results.json")
    analyse_parser.add_argument("--workers", type=int, default=8)
    analyse_parser.set_defaults(function=analyse)

    for subparser in (run_parser, analyse_parser):
        subparser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
        subparser.add_argument("--experiment", choices=list(CONFIGURATIONS), default="operators")
        subparser.add_argument("--dataset", choices=list(DATASETS), default="hcv")

    args = parser.parse_args()

    global DATASET, OUTPUT_PATH
    DATASET = DATASETS[args.dataset]
    OUTPUT_PATH = TESTS_PATH / "operator_benchmark" / OUTPUT_NAMES[args.experiment].format(dataset=args.dataset)
    args.function(args)


if __name__ == "__main__":
    main()
