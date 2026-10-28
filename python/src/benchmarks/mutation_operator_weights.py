"""Measures how fast each mutation operator mixes the history on a fixed posterior tree.

Takes one posterior state (tree and parameters) per dataset from tests/fixed_tree_gibbs,
starts from an exact draw of the history, and runs a chain with a single
mutation operator, or a given mix of them, while the tree and the parameters stay fixed.
The trace holds the genetic prior, the number of mutations, and the distances of the
node states at the root, the other inner nodes and the tips to an independent exact
draw. Their effective sample sizes per proposal give the weight that a set of local
operators needs to mix the history as fast as the global SiteHistoryGibbsOperator.

Usage, from the python directory after `mvn compile` in the repository root:

    uv run python -m src.benchmarks.mutation_operator_weights run
    uv run python -m src.benchmarks.mutation_operator_weights analyse

The baseline configuration keeps the mutation operators of the analysis with their weights.
Further mixes are given as name=operator:weight,operator:weight, e.g. --mix mine=branch:2,missing:1.
"""

import argparse
import copy
import subprocess
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPOSITORY_PATH = Path(__file__).resolve().parents[3]
CLASSES_PATH = REPOSITORY_PATH / "target" / "classes"
INPUT_PATH = REPOSITORY_PATH / "tests" / "fixed_tree_gibbs"
OUTPUT_PATH = REPOSITORY_PATH / "tests" / "mutation_operator_weights"

DATASETS = ["rsva", "rsvb", "zika", "sars", "hcv"]
STATE_NR = 2070000

# a dataset with this suffix is the base dataset without the sites at which every tip is missing
OBSERVED_SUFFIX = "_observed"
MISSING_CHARACTERS = set("N?-")

CHAIN_LENGTH = 2_000_000
LOG_EVERY = 1000

# the share of the logged samples discarded at the start
BURNIN_FRACTION = 0.1

STATISTICS = ["geneticPrior", "mutationCount", "rootDistance", "innerDistance", "tipDistance"]

# the mutation operators by name, with their spec and the attributes of the base XML that they keep
OPERATOR_SPECS = {
    "global": "emat.operators.mutations.SiteHistoryGibbsOperator",
    "branch": "emat.operators.mutations.BranchReformOperator",
    "missing": "emat.operators.mutations.MissingCladeGibbsOperator",
    "time": "emat.operators.mutations.MutationTimeOperator",
}

# the operators that do not take the genetic prior
OPERATORS_WITHOUT_PRIOR = {"time"}



def get_state_path(dataset: str) -> Path:
    """Returns the posterior state of the base dataset."""
    return INPUT_PATH / dataset.removesuffix(OBSERVED_SUFFIX) / f"state_{STATE_NR}.xml"


def get_partition(dataset: str) -> str:
    """Returns the id suffix of the dataset, read from the id of its mutations."""
    root = ET.parse(get_state_path(dataset)).getroot()
    mutations = next(node for node in root.iter() if (node.get("id") or "").startswith("mutations."))
    return mutations.get("id").removeprefix("mutations.")


def build_xml(dataset: str, weights: dict[str, float], chain_length: int) -> ET.ElementTree:
    """Builds a chain on the fixed tree of the dataset with the given mutation operators and weights."""
    tree = ET.parse(get_state_path(dataset))
    run = tree.getroot().find("run")
    partition = get_partition(dataset)
    if dataset.endswith(OBSERVED_SUFFIX):
        remove_missing_sites(tree)

    # keep the state, the posterior and the initialisers of the analysis, as a plain chain

    old_operators = {node.get("spec"): node for node in run.findall("operator")}
    for attribute in list(run.attrib):
        if attribute not in ("id",):
            del run.attrib[attribute]
    run.set("spec", "MCMC")
    run.set("chainLength", str(chain_length))
    for child in list(run):
        if child.tag in ("operator", "logger"):
            run.remove(child)

    # the history starts from an exact draw, after the parsimony initialiser has run

    inits = run.findall("init")
    exact_init = ET.Element("init", {
        "id": "ExactHistoryInitialiser", "spec": "emat.benchmark.ExactHistoryInitialiser",
        "mutations": f"@mutations.{partition}", "geneticPrior": f"@geneticPrior.{partition}",
    })
    run.insert(list(run).index(inits[-1]) + 1, exact_init)

    for name, weight in weights.items():
        spec = OPERATOR_SPECS[name]
        if spec in old_operators:
            operator = copy.deepcopy(old_operators[spec])
        else:
            operator = ET.Element("operator", {"spec": spec, "mutations": f"@mutations.{partition}"})
            if name not in OPERATORS_WITHOUT_PRIOR:
                operator.set("geneticPrior", f"@geneticPrior.{partition}")
        operator.set("id", f"{name}Operator")
        operator.set("weight", str(weight))
        run.append(operator)

    logger = ET.SubElement(run, "logger", {"id": "tracelog", "spec": "Logger", "fileName": "$(filebase).log", "logEvery": str(LOG_EVERY)})
    ET.SubElement(logger, "log", {"idref": f"geneticPrior.{partition}"})
    ET.SubElement(logger, "log", {"idref": f"mutations.{partition}"})
    ET.SubElement(logger, "log", {"id": "HistoryDistance", "spec": "emat.benchmark.HistoryDistanceLogger",
                                  "mutations": f"@mutations.{partition}", "geneticPrior": f"@geneticPrior.{partition}"})
    return tree


def remove_missing_sites(tree: ET.ElementTree) -> None:
    """Removes the sites at which every tip is missing from all sequences of the alignment."""
    sequences = list(tree.getroot().iter("sequence"))
    values = [sequence.get("value").upper() for sequence in sequences]
    kept_sites = [site for site in range(len(values[0])) if any(value[site] not in MISSING_CHARACTERS for value in values)]
    for sequence, value in zip(sequences, values):
        sequence.set("value", "".join(value[site] for site in kept_sites))
    print(f"kept {len(kept_sites)} of {len(values[0])} sites")


def get_configurations(dataset: str, mixes: dict[str, dict[str, float]] | None = None) -> dict[str, dict[str, float]]:
    """
    Returns the operator mixes to run: every operator alone, the mutation operators of the
    analysis, and the given mixes.
    """
    configurations = {name: {name: 1.0} for name in OPERATOR_SPECS}
    configurations["baseline"] = get_baseline_weights(dataset)
    configurations.update(mixes or {})
    return configurations


def get_baseline_weights(dataset: str) -> dict[str, float]:
    """Returns the weights of the mutation operators in the analysis of the dataset."""
    root = ET.parse(get_state_path(dataset)).getroot()
    names = {spec: name for name, spec in OPERATOR_SPECS.items()}
    return {names[node.get("spec")]: float(node.get("weight")) for node in root.iter("operator") if node.get("spec") in names}


def parse_mixes(values: list[str] | None) -> dict[str, dict[str, float]]:
    """Parses mixes given as name=operator:weight,operator:weight."""
    mixes = {}
    for value in values or []:
        name, weights = value.split("=")
        mixes[name] = {operator: float(weight) for operator, weight in (entry.split(":") for entry in weights.split(","))}
    return mixes


def run(args: argparse.Namespace) -> None:
    module_path = get_module_path()

    jobs = []
    for dataset in args.datasets:
        output_path = OUTPUT_PATH / dataset
        output_path.mkdir(parents=True, exist_ok=True)
        for name, weights in get_configurations(dataset, parse_mixes(args.mix)).items():
            if args.configurations and name not in args.configurations:
                continue
            xml_path = output_path / f"{name}.xml"
            build_xml(dataset, weights, args.chain_length).write(xml_path)
            jobs.append(xml_path)

    def run_timed(xml_path: Path) -> None:
        start = time.time()
        run_beast(xml_path, 1, module_path)
        xml_path.with_suffix(".time").write_text(f"{time.time() - start:.1f}\n")

    with ThreadPoolExecutor(args.workers) as executor:
        list(executor.map(run_timed, jobs))


def get_module_path() -> str:
    """Returns the module path with the compiled package and its dependencies, as resolved by Maven."""
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    classpath_path = OUTPUT_PATH / "classpath.txt"
    if not classpath_path.exists():
        subprocess.run(
            ["mvn", "-q", "dependency:build-classpath", f"-Dmdep.outputFile={classpath_path}"],
            cwd=REPOSITORY_PATH, check=True,
        )
    return f"{CLASSES_PATH}:{classpath_path.read_text().strip()}"


def run_beast(xml_path: Path, seed: int, module_path: str) -> None:
    with open(xml_path.with_suffix(".out"), "w") as output:
        result = subprocess.run(
            [
                "java", "--module-path", module_path, f"-DBEAST_PACKAGE_PATH={CLASSES_PATH}",
                "-m", "beast.base/beast.base.minimal.BeastMain",
                "-seed", str(seed), "-overwrite", xml_path.name,
            ],
            cwd=xml_path.parent, stdout=output, stderr=subprocess.STDOUT,
        )

    status = "done" if result.returncode == 0 else f"FAILED ({result.returncode})"
    print(f"{xml_path.stem}: {status}", flush=True)


def compute_ess(values: np.ndarray) -> float:
    """Computes the effective sample size by Geyer's initial positive sequence of autocorrelations."""
    values = values - values.mean()
    variance = values.var()
    if variance == 0.0:
        return np.nan

    num_values = len(values)
    spectrum = np.fft.rfft(values, 2 * num_values)
    autocorrelations = np.fft.irfft(spectrum * np.conj(spectrum))[:num_values] / (variance * num_values)

    # sum the pairs of autocorrelations while they stay positive
    tau = -1.0
    for lag in range(0, num_values - 1, 2):
        pair = autocorrelations[lag] + autocorrelations[lag + 1]
        if pair <= 0.0:
            break
        tau += 2.0 * pair
    return num_values / max(tau, 1.0)


def read_run(dataset: str, name: str) -> dict[str, float] | None:
    """Returns the proposals per effective sample of every statistic and the time per proposal of a run."""
    log_path = OUTPUT_PATH / dataset / f"{name}.log"
    if not log_path.exists():
        return None

    trace = pd.read_csv(log_path, sep="\t", comment="#", on_bad_lines="skip")
    trace = trace.rename(columns=lambda column: "geneticPrior" if column.startswith("geneticPrior")
                         else "mutationCount" if column.startswith("mutations") else column)
    num_proposals = trace["Sample"].iloc[-1]
    if num_proposals == 0:
        return None

    total_proposals = num_proposals
    trace = trace.iloc[int(BURNIN_FRACTION * len(trace)):]
    num_proposals = trace["Sample"].iloc[-1] - trace["Sample"].iloc[0]

    result = {}
    for statistic in STATISTICS:
        ess = compute_ess(trace[statistic].to_numpy(dtype=float))
        result[statistic] = num_proposals / ess if ess == ess else np.nan
        result[f"sd_{statistic}"] = trace[statistic].std()

    output = (OUTPUT_PATH / dataset / f"{name}.out").read_text()
    seconds = [float(line.split()[-2]) for line in output.splitlines() if line.startswith("Total calculation time")]
    result["us_per_proposal"] = 1e6 * seconds[0] / total_proposals if seconds else np.nan
    result["num_proposals"] = num_proposals
    for statistic in STATISTICS:
        result[f"ess_per_minute_{statistic}"] = 60e6 / (result[statistic] * result["us_per_proposal"])
    return result


def get_global_weight(dataset: str) -> float:
    """Returns the weight of the global SiteHistoryGibbsOperator in the base XML of the dataset."""
    root = ET.parse(get_state_path(dataset)).getroot()
    return float(next(node.get("weight") for node in root.iter("operator") if node.get("spec") == OPERATOR_SPECS["global"]))


def analyse(args: argparse.Namespace) -> None:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)

    for dataset in args.datasets:
        rows = {name: read_run(dataset, name) for name in get_configurations(dataset, parse_mixes(args.mix))}
        table = pd.DataFrame({name: row for name, row in rows.items() if row is not None}).T
        if table.empty:
            continue

        print(f"\n== {dataset} (global weight {get_global_weight(dataset):g}) ==")
        print("proposals per effective sample (NaN: the statistic does not move):")
        print(table[STATISTICS + ["us_per_proposal", "num_proposals"]].to_string(float_format=lambda value: f"{value:.4g}"))
        print("effective samples per minute:")
        print(table[[f"ess_per_minute_{statistic}" for statistic in STATISTICS]].to_string(float_format=lambda value: f"{value:.4g}"))
        print("posterior sd of the statistics:")
        print(table[[f"sd_{statistic}" for statistic in STATISTICS]].to_string(float_format=lambda value: f"{value:.4g}"))

        # the weight a local operator alone needs to give every statistic as many effective samples as the global operator

        if "global" in table.index:
            weights = table.loc[[name for name in table.index if name != "global"], STATISTICS] / table.loc["global", STATISTICS]
            print("weight relative to the global operator to match its rate per statistic:")
            print(weights.to_string(float_format=lambda value: f"{value:.3g}"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", type=lambda value: value.split(","), default=list(DATASETS), help="comma-separated datasets")
    subparsers = parser.add_subparsers(required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--workers", type=int, default=5)
    run_parser.add_argument("--chain-length", type=int, default=CHAIN_LENGTH)
    run_parser.add_argument("--configurations", nargs="+")
    run_parser.add_argument("--mix", action="append", help="a mix as name=operator:weight,operator:weight")
    run_parser.set_defaults(function=run)

    analyse_parser = subparsers.add_parser("analyse")
    analyse_parser.add_argument("--mix", action="append", help="a mix as name=operator:weight,operator:weight")
    analyse_parser.set_defaults(function=analyse)

    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
