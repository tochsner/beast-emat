"""Computes the running time until all parameters of each run reach an ESS of 200 (ignoring the tree topology).

The logs are named <method>_<experiment>-<seed>.log and grouped by method. Every seed of a run is analysed separately.
Runs that have not converged yet are extrapolated from the ESS per minute of their slowest parameter. The plots of each
method show the median over the seeds, with the spread over the seeds as uncertainty. The plots are stored in the plots
directory at the repository root, prefixed with the method.

The analysis of each log and XML is cached and only redone once the file is modified or the analysis settings change,
so that rerunning the script mostly just redraws the plots. Deleting the cache directory forces a full reanalysis.
"""

import math
import pickle
import re
import warnings
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TypeVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text

from src.data_pipeline.summary import SEED, compute_mean_pairwise_distance
from src.utils.ess import compute_ess

# the run outputs live in the data directory at the repository root
REPOSITORY_PATH = Path(__file__).resolve().parents[3]
OUT_PATH = REPOSITORY_PATH / "data" / "out"
XMLS_PATH = REPOSITORY_PATH / "data" / "xmls"
PLOTS_PATH = REPOSITORY_PATH / "plots"
CACHE_PATH = REPOSITORY_PATH / "data" / "cache" / "convergence_time"

LOG_PATTERN = re.compile(r"(?P<method>.+)_(?P<experiment>[^_]+)-(?P<seed>\d+)\.log")
SEQUENCE_PATTERN = re.compile(r'<sequence\b[^>]*\bvalue="([^"]*)"')
TARGET_ESS = 200
BURNIN_FRACTION = 0.1
NUM_POINTS = 20
INCLUDED_METHODS = {"coal_emat", "sbd_emat", "coal_emat_nofilter"}
EXCLUDED_EXPERIMENTS = {"sars20000", "wnv", "ncbisc2"}

T = TypeVar("T")


def set_plot_theme() -> None:
    """Sets the seaborn theme shared by all plots."""
    sns.set_theme(
        style="whitegrid",
        context="notebook",
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#bbbbbb",
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "grid.color": "#e8e8e8",
            "figure.dpi": 120,
        },
    )


def find_log_paths() -> dict[str, dict[tuple[str, int], Path]]:
    """Returns the log file of each experiment and seed, grouped by method."""
    log_paths = {}
    for path in OUT_PATH.glob("*.log"):
        match = LOG_PATTERN.fullmatch(path.name)
        if (
            match
            and match.group("method") in INCLUDED_METHODS
            and match.group("experiment") not in EXCLUDED_EXPERIMENTS
        ):
            method = match.group("method")
            key = (match.group("experiment"), int(match.group("seed")))
            log_paths.setdefault(method, {})[key] = path

    return {
        method: dict(sorted(paths.items()))
        for method, paths in sorted(log_paths.items())
    }


def get_cached(path: Path, analyse: Callable[[Path], T]) -> T:
    """Returns the result of `analyse` on the file, reusing the cached result while the file and the analysis settings are unchanged."""
    stat = path.stat()
    key = (stat.st_mtime_ns, stat.st_size, BURNIN_FRACTION, NUM_POINTS, SEED)
    cache_path = CACHE_PATH / f"{analyse.__name__}_{path.name}.pkl"

    # a missing, outdated or unreadable cache entry is recomputed
    try:
        with open(cache_path, "rb") as file:
            cached = pickle.load(file)
        if cached["key"] == key:
            return cached["result"]
    except (OSError, EOFError, KeyError, pickle.UnpicklingError):
        pass

    result = analyse(path)
    CACHE_PATH.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as file:
        pickle.dump({"key": key, "result": result}, file)
    return result


def analyse_log(log_path: Path) -> dict | None:
    """Returns the ESS trajectory and the posterior mean number of mutations of a log, or None if the log is not usable."""
    log_df = load_log(log_path)
    if log_df is None:
        return None

    trajectory = get_ess_trajectory(log_df)
    if trajectory is None:
        return None

    return {"trajectory": trajectory, "num_mutations": get_mean_num_mutations(log_df)}


def get_job_start(log_path: Path) -> float | None:
    """Returns the timestamp of the `# Started job at` comment of the log file, or None if it has none."""
    pattern = re.compile(r"^# Started job at (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
    with open(log_path) as file:
        for line in file:
            match = pattern.match(line)
            if match:
                return datetime.strptime(
                    match.group(1), "%Y-%m-%d %H:%M:%S"
                ).timestamp()
    return None


def get_seconds_per_sample(log_path: Path, last_sample: int) -> float:
    """Returns the wall-clock seconds per MCMC sample, estimated from the time between the start of the job and the last sample."""
    # the job starts at the started-job comment if the log has one, otherwise at the creation of the log file
    stat = log_path.stat()
    start_timestamp = get_job_start(log_path)
    if start_timestamp is None:
        # st_birthtime is not available on all platforms (e.g. Linux), where the oldest known timestamp is used instead
        start_timestamp = getattr(
            stat, "st_birthtime", min(stat.st_ctime, stat.st_mtime)
        )

    # the log file is last modified when the last sample was logged
    elapsed_seconds = max(stat.st_mtime - start_timestamp, 1.0)
    print(
        f"{log_path.name}: estimated {elapsed_seconds / last_sample * 1e6:.0f}s per million samples from the file timestamps"
    )
    return elapsed_seconds / last_sample


def get_alignment_stats(xml_path: Path) -> dict[str, float]:
    """Returns the number of sequences and the nucleotide diversity (mean pairwise p-distance) of the alignment in a BEAST XML."""
    encoded = read_alignment(xml_path)
    diversity, _ = compute_mean_pairwise_distance(encoded, np.random.default_rng(SEED))
    return {"num_sequences": len(encoded), "diversity": diversity}


def read_alignment(xml_path: Path) -> list[np.ndarray]:
    """Returns the upper-case sequences of the alignment in a BEAST XML, encoded as bytes."""
    # the alignment is the first data block of the XML, with one sequence per line
    encoded = []
    with open(xml_path) as file:
        for line in file:
            match = SEQUENCE_PATTERN.search(line)
            if match:
                encoded.append(
                    np.frombuffer(
                        match.group(1).upper().encode("ascii"), dtype=np.uint8
                    )
                )
            if "</data>" in line:
                break

    if not encoded:
        raise ValueError(f"no sequences found in {xml_path}")
    return encoded


def load_log(log_path: Path) -> pd.DataFrame | None:
    """Reads an MCMC log, or returns None if it is empty or unreadable."""
    # Check if the file is empty and skip if so
    if log_path.stat().st_size == 0:
        print(f"Skipping empty log file: {log_path.name}")
        return None

    # corrupted lines (e.g. partially written or interleaved ones) are dropped, whether they have too many fields,
    # too few fields or values that are not numbers
    try:
        log_df = pd.read_csv(
            log_path, comment="#", delimiter="\t", on_bad_lines="skip"
        )
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as error:
        print(f"Skipping unreadable log file: {log_path.name} ({error})")
        return None
    log_df = log_df.apply(pd.to_numeric, errors="coerce").dropna()

    # two merged lines can still have the right number of fields, but then their sample exceeds the next one
    if "Sample" in log_df.columns:
        next_sample = log_df["Sample"].shift(-1, fill_value=np.inf)
        log_df = log_df[log_df["Sample"] < next_sample]

    # Also skip if the log_df is empty after reading (empty header file, etc)
    if (
        log_df.empty
        or "Sample" not in log_df.columns
        or log_df["Sample"].iloc[-1] <= 0
    ):
        print(f"Skipping log file (no data after parsing): {log_path.name}")
        return None

    seconds_per_sample = get_seconds_per_sample(log_path, log_df["Sample"].iloc[-1])
    log_df["minutes"] = log_df["Sample"] * seconds_per_sample / 60

    return log_df


def get_mean_num_mutations(log_df: pd.DataFrame) -> float:
    """Returns the posterior mean number of mutations after the burn-in, or NaN if it is not logged."""
    if "mutations.count" not in log_df.columns:
        return float("nan")
    return float(log_df["mutations.count"].iloc[int(len(log_df) * BURNIN_FRACTION) :].mean())


def get_ess_trajectory(log_df: pd.DataFrame) -> pd.DataFrame | None:
    """Computes the ESS of each parameter at evenly spaced points in time, as if the run were stopped there.

    Returns None if the log is too short or has no varying parameters.
    """
    # constant and non-numeric columns are left out, as their ESS is undefined
    columns = [
        column
        for column in log_df.columns
        if column not in ["Sample", "minutes", "mutations.excessCount"]
        and pd.api.types.is_numeric_dtype(log_df[column])
        and log_df[column].std() > 0
    ]
    if len(log_df) < 20 or not columns:
        print("Skipping run with too few samples or no varying parameters")
        return None

    values = log_df[columns].to_numpy(dtype=float)
    minutes = log_df["minutes"].to_numpy()

    # at least 20 states are needed for a meaningful ESS estimate
    end_indices = np.unique(np.linspace(20, len(log_df), NUM_POINTS).astype(int))

    rows = []
    for end in end_indices:
        start = int(end * BURNIN_FRACTION)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ess = [compute_ess(values[start:end, i]) for i in range(len(columns))]
        rows.append({"minutes": minutes[end - 1], **dict(zip(columns, ess))})

    return pd.DataFrame(rows).set_index("minutes")


def get_convergence_df(
    ess_trajectories: dict[tuple[str, int], pd.DataFrame],
) -> pd.DataFrame:
    """Returns the time at which each parameter of each seed reaches the target ESS, projected if it has not yet."""
    convergence_rows = []

    for (experiment, seed), trajectory in ess_trajectories.items():
        end_minutes = trajectory.index[-1]

        for column in trajectory.columns:
            below_target = (trajectory[column] < TARGET_ESS).to_numpy()
            final_ess = trajectory[column].iloc[-1]

            if not below_target.any():
                minutes, projected = trajectory.index[0], False
            elif not below_target[-1]:
                # the first point after the last drop below the target
                minutes, projected = (
                    trajectory.index[len(below_target) - np.argmax(below_target[::-1])],
                    False,
                )
            else:
                minutes, projected = TARGET_ESS / (final_ess / end_minutes), True

            convergence_rows.append(
                {
                    "experiment": experiment,
                    "seed": seed,
                    "parameter": column,
                    "minutes": minutes,
                    "projected": projected,
                    "ess_per_minute": final_ess / end_minutes,
                }
            )

    return pd.DataFrame(convergence_rows)


def get_summary_df(
    convergence_df: pd.DataFrame, experiment_stats: dict[str, dict[str, float]]
) -> pd.DataFrame:
    """Returns the median running time of each run over its seeds, with the range over the seeds.

    The status is projected if any seed of the run has not converged yet.
    """
    # a seed converges once its slowest-mixing parameter stays above the target ESS
    slowest_df = (
        convergence_df.sort_values("minutes")
        .groupby(["experiment", "seed"])
        .last()
        .reset_index()
    )

    summary_df = (
        slowest_df.groupby("experiment")
        .agg(
            num_seeds=("seed", "count"),
            minutes=("minutes", "median"),
            min_minutes=("minutes", "min"),
            max_minutes=("minutes", "max"),
            projected=("projected", "any"),
            slowest_parameter=(
                "parameter",
                lambda parameters: parameters.mode().iloc[0],
            ),
        )
        .reset_index()
    )
    stats_df = pd.DataFrame.from_dict(experiment_stats, orient="index")
    summary_df = summary_df.join(stats_df, on="experiment")
    summary_df["status"] = np.where(summary_df["projected"], "projected", "observed")

    return summary_df


def format_minutes(minutes: float) -> str:
    """Formats a running time in minutes as a short string, e.g. `45min` or `2h13min`."""
    minutes = max(0, round(minutes))

    if minutes < 60:
        return f"{minutes}min"

    return f"{minutes // 60}h{minutes % 60}min"


def plot_convergence_time(
    method: str, summary_df: pd.DataFrame, experiment_colors: dict[str, tuple]
) -> None:
    """Plots the median running time of each run against its number of sequences, with the range over its seeds."""
    plot_running_time_against(
        method,
        summary_df,
        experiment_colors,
        x_column="num_sequences",
        x_label="number of sequences",
        x_format="d",
        file_suffix="convergence_time",
    )


def plot_diversity_convergence_time(
    method: str, summary_df: pd.DataFrame, experiment_colors: dict[str, tuple]
) -> None:
    """Plots the median running time of each run against its nucleotide diversity, with the range over its seeds."""
    plot_running_time_against(
        method,
        summary_df,
        experiment_colors,
        x_column="diversity",
        x_label="nucleotide diversity (π)",
        x_format=".4f",
        file_suffix="diversity_convergence_time",
    )


def plot_mutations_convergence_time(
    method: str, summary_df: pd.DataFrame, experiment_colors: dict[str, tuple]
) -> None:
    """Plots the median running time of each run against its posterior mean number of mutations, with the range over its seeds."""
    plot_running_time_against(
        method,
        summary_df,
        experiment_colors,
        x_column="num_mutations",
        x_label="number of mutations (posterior mean)",
        x_format=".0f",
        file_suffix="mutations_convergence_time",
    )


def plot_running_time_against(
    method: str,
    summary_df: pd.DataFrame,
    experiment_colors: dict[str, tuple],
    x_column: str,
    x_label: str,
    x_format: str,
    file_suffix: str,
) -> None:
    """Plots the median running time of each run against a column of the summary, with the range over its seeds."""
    # runs that do not log the column cannot be placed on the plot
    summary_df = summary_df.dropna(subset=[x_column])
    if summary_df.empty:
        print(f"{method}: no runs with {x_column}, skipping the {file_suffix} plot")
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    for _, row in summary_df.iterrows():
        ax.errorbar(
            row[x_column],
            row["minutes"],
            yerr=[
                [row["minutes"] - row["min_minutes"]],
                [row["max_minutes"] - row["minutes"]],
            ],
            fmt="X" if row["status"] == "projected" else "o",
            color=experiment_colors[row["experiment"]],
            markersize=10.5,
            markeredgecolor="white",
            markeredgewidth=1.5,
            elinewidth=1.2,
            capsize=3,
        )

    # the axes start slightly below zero, so that points close to zero are not cut off
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlim(
        -0.03 * summary_df[x_column].max(),
        summary_df[x_column].max() * 1.25,
    )
    ax.set_ylim(
        -0.03 * summary_df["min_minutes"].max(), summary_df["max_minutes"].max() * 1.2
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel("running time (min)")
    ax.set_title(f"{method}: running time until ESS ≥ 200 on all parameters")

    fig.tight_layout()

    # labels are pushed apart iteratively until they overlap neither each other nor the points
    texts = [
        ax.text(
            row[x_column],
            row["minutes"],
            f"{row['experiment']} ({row[x_column]:{x_format}}, {format_minutes(row['minutes'])})",
            fontsize=10,
            color="#444444",
        )
        for _, row in summary_df.iterrows()
    ]
    adjust_text(
        texts,
        x=summary_df[x_column],
        y=summary_df["minutes"],
        ax=ax,
        arrowprops={
            "arrowstyle": "-",
            "color": "#bbbbbb",
            "linewidth": 0.8,
            "shrinkB": 6,
        },
        expand=(0.5, 1.5),
    )

    fig.savefig(
        PLOTS_PATH / f"{method}_{file_suffix}.png", dpi=200, bbox_inches="tight"
    )
    plt.close(fig)


def plot_min_ess_trajectories(
    method: str,
    ess_trajectories: dict[tuple[str, int], pd.DataFrame],
    experiment_colors: dict[str, tuple],
) -> None:
    """Plots the ESS of the slowest parameter over running time, with one line per seed."""
    experiments = list(experiment_colors)
    ncols = 3
    nrows = math.ceil(len(experiments) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.64 * ncols, 2.8 * nrows), squeeze=False
    )
    axes_by_experiment = dict(zip(experiments, axes.flat))

    for (experiment, _), trajectory in ess_trajectories.items():
        axes_by_experiment[experiment].plot(
            trajectory.index,
            trajectory.min(axis=1),
            color=experiment_colors[experiment],
            linewidth=2,
            alpha=0.6,
        )

    for experiment, ax in axes_by_experiment.items():
        ax.axhline(TARGET_ESS, color="gray", linestyle="--", linewidth=1)
        ax.set_title(experiment, fontweight="bold")
        ax.set_xlabel("running time (min)")
        ax.set_ylabel("minimum ESS")

    for ax in axes.flat[len(experiments) :]:
        ax.set_visible(False)

    fig.suptitle(
        f"{method}: minimum ESS over running time", x=0.01, ha="left", fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(
        PLOTS_PATH / f"{method}_ess_trajectories.png", dpi=200, bbox_inches="tight"
    )
    plt.close(fig)


def plot_relative_convergence_time(
    method: str, convergence_df: pd.DataFrame, experiments: list[str]
) -> None:
    """Plots the mean time until each parameter reaches the target ESS, relative to the slowest parameter of each seed."""
    relative_df = convergence_df.assign(
        relative=convergence_df["minutes"]
        / convergence_df.groupby(["experiment", "seed"])["minutes"].transform("max"),
    )
    stats_df = relative_df.groupby(["parameter", "experiment"])["relative"].agg(
        ["mean", "std"]
    )
    mean_df = stats_df["mean"].unstack("experiment")[experiments]
    std_df = stats_df["std"].unstack("experiment")[experiments].fillna(0)

    # parameters shared by more experiments are listed first
    order_df = pd.DataFrame(
        {"count": mean_df.notna().sum(axis=1), "mean": mean_df.mean(axis=1)}
    )
    order = order_df.sort_values(["count", "mean"], ascending=False).index
    mean_df = mean_df.loc[order]
    std_df = std_df.loc[order]

    # each cell shows the mean and the standard deviation over the seeds
    annot_df = pd.DataFrame(
        [
            [
                f"{mean:.2f}±{std:.2f}" if pd.notna(mean) else ""
                for mean, std in zip(mean_row, std_row)
            ]
            for mean_row, std_row in zip(mean_df.to_numpy(), std_df.to_numpy())
        ],
        index=mean_df.index,
        columns=mean_df.columns,
    )

    fig, ax = plt.subplots(figsize=(10, 0.5 * len(mean_df) + 1))

    sns.heatmap(
        mean_df,
        cmap="Blues",
        vmin=0,
        vmax=1,
        annot=annot_df,
        fmt="",
        annot_kws={"fontsize": 7},
        linewidths=1,
        linecolor="white",
        ax=ax,
        cbar_kws={"label": "relative time until ESS ≥ 200 (mean ± sd over seeds)"},
    )
    ax.grid(False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(f"{method}: time until ESS ≥ 200 per parameter", loc="left")

    fig.tight_layout()
    fig.savefig(
        PLOTS_PATH / f"{method}_relative_convergence_time.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def analyse_method(method: str, log_paths: dict[tuple[str, int], Path]) -> None:
    """Analyses all seeds of all experiments of a method and plots the results, prefixed with the method."""
    log_results = {key: get_cached(log_path, analyse_log) for key, log_path in log_paths.items()}
    log_results = {key: result for key, result in log_results.items() if result is not None}
    if not log_results:
        print(f"\n{method}: no usable runs, skipping")
        return

    # experiments without a readable XML cannot be placed on the plots
    alignment_stats = {}
    for experiment in sorted({experiment for experiment, _ in log_results}):
        try:
            alignment_stats[experiment] = get_cached(XMLS_PATH / f"{method}_{experiment}.xml", get_alignment_stats)
        except (ValueError, OSError) as error:
            print(f"Skipping experiment {method}_{experiment}: {error}")
    log_results = {key: result for key, result in log_results.items() if key[0] in alignment_stats}
    experiments = sorted(alignment_stats)

    # each experiment keeps the same color in all plots of the method
    # (the palette is sized to the experiments, as zip would silently drop experiments beyond the palette length)
    palette = (
        sns.color_palette("deep")
        if len(experiments) <= 10
        else sns.color_palette("husl", len(experiments))
    )
    experiment_colors = dict(zip(experiments, palette))

    ess_trajectories = {key: result["trajectory"] for key, result in log_results.items()}

    # the number of mutations is averaged over the seeds that log it
    experiment_stats = {}
    for experiment in experiments:
        num_mutations = [
            result["num_mutations"]
            for key, result in log_results.items()
            if key[0] == experiment and not np.isnan(result["num_mutations"])
        ]
        experiment_stats[experiment] = {
            **alignment_stats[experiment],
            "num_mutations": float(np.mean(num_mutations)) if num_mutations else float("nan"),
        }

    convergence_df = get_convergence_df(ess_trajectories)
    summary_df = get_summary_df(convergence_df, experiment_stats)

    print(f"\n{method}")
    print(
        summary_df[
            [
                "experiment",
                "num_sequences",
                "diversity",
                "num_mutations",
                "num_seeds",
                "minutes",
                "min_minutes",
                "max_minutes",
                "status",
                "slowest_parameter",
            ]
        ].to_string(index=False)
    )

    plot_convergence_time(method, summary_df, experiment_colors)
    plot_diversity_convergence_time(method, summary_df, experiment_colors)
    plot_mutations_convergence_time(method, summary_df, experiment_colors)
    plot_min_ess_trajectories(method, ess_trajectories, experiment_colors)
    plot_relative_convergence_time(method, convergence_df, experiments)


def main() -> None:
    set_plot_theme()
    PLOTS_PATH.mkdir(parents=True, exist_ok=True)

    for method, log_paths in find_log_paths().items():
        analyse_method(method, log_paths)


if __name__ == "__main__":
    main()
