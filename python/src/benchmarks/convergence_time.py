"""Computes the running time until all parameters of each run reach an ESS of 200 (ignoring the tree topology).

Every seed of a run is analysed separately. Runs that have not converged yet are extrapolated from the ESS per minute
of their slowest parameter. The plots show the mean over the seeds, with the spread over the seeds as uncertainty.
The plots are stored in the plots directory at the repository root.
"""

import math
import re
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text

from src.utils.ess import compute_ess

# the run outputs live in the data directory at the repository root
REPOSITORY_PATH = Path(__file__).resolve().parents[3]
OUT_PATH = REPOSITORY_PATH / "data" / "out"
PLOTS_PATH = REPOSITORY_PATH / "plots"

METHOD = "emat"
TARGET_ESS = 200
BURNIN_FRACTION = 0.1
NUM_POINTS = 200

EXPERIMENTS = [
    "hcv", "zika", "sars", "ebola", "sars5000", "sars10000",
    "bd_hcv", "bd_zika", "bd_sars", "bd_ebola", "bd_sars5000", "bd_sars10000",
]


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


def find_log_paths(experiment: str) -> dict[int, Path]:
    """Returns the log file of each seed of the run, keyed by seed."""
    pattern = re.compile(rf"{METHOD}_{experiment}-(\d+)\.log")

    log_paths = {}
    for path in OUT_PATH.glob(f"{METHOD}_{experiment}-*.log"):
        match = pattern.fullmatch(path.name)
        if match:
            log_paths[int(match.group(1))] = path

    return dict(sorted(log_paths.items()))


def get_job_start(log_path: Path) -> float | None:
    """Returns the timestamp of the `# Started job at` comment of the log file, or None if it has none."""
    pattern = re.compile(r"^# Started job at (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
    with open(log_path) as file:
        for line in file:
            match = pattern.match(line)
            if match:
                return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S").timestamp()
    return None


def get_seconds_per_sample(experiment: str, log_path: Path, last_sample: int) -> float:
    """Returns the wall-clock seconds per MCMC sample, estimated from the time between the start of the job and the last sample."""
    # the job starts at the started-job comment if the log has one, otherwise at the creation of the log file
    stat = log_path.stat()
    start_timestamp = get_job_start(log_path)
    if start_timestamp is None:
        start_timestamp = stat.st_birthtime

    # the log file is last modified when the last sample was logged
    elapsed_seconds = stat.st_mtime - start_timestamp
    print(f"{experiment}: no timing known, estimated {elapsed_seconds / last_sample * 1e6:.0f}s per million samples from the file timestamps")
    return elapsed_seconds / last_sample


def get_num_sequences(experiment: str) -> int:
    """Returns the number of sequences, read from the header of a tree file of the run."""
    tree_path = next(OUT_PATH.glob(f"{METHOD}_{experiment}-*.trees"))
    with open(tree_path) as file:
        for line in file:
            match = re.search(r"ntax\s*=\s*(\d+)", line, re.IGNORECASE)
            if match:
                return int(match.group(1))
    raise ValueError(f"no ntax found in {tree_path}")


def load_logs() -> tuple[dict[tuple[str, int], pd.DataFrame], dict[str, int]]:
    """Reads the MCMC log of each seed of each run, keyed by experiment and seed, and the number of sequences of each run."""
    logs = {}
    num_sequences = {}

    for experiment in EXPERIMENTS:
        log_paths = find_log_paths(experiment)
        if not log_paths:
            print(f"{experiment}: no log found")
            continue

        for seed, log_path in log_paths.items():
            # a run that is still in progress may have a partially written last line
            log_df = pd.read_csv(log_path, comment="#", delimiter="\t").dropna()

            seconds_per_sample = get_seconds_per_sample(experiment, log_path, log_df["Sample"].iloc[-1])
            log_df["minutes"] = log_df["Sample"] * seconds_per_sample / 60

            logs[(experiment, seed)] = log_df

        num_sequences[experiment] = get_num_sequences(experiment)

    return logs, num_sequences


def get_ess_trajectory(log_df: pd.DataFrame) -> pd.DataFrame:
    """Computes the ESS of each parameter at evenly spaced points in time, as if the run were stopped there."""
    # constant columns are left out, as their ESS is undefined
    columns = [column for column in log_df.columns if column not in ["Sample", "minutes", "mutations.excessCount"] and log_df[column].std() > 0]
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


def get_convergence_df(ess_trajectories: dict[tuple[str, int], pd.DataFrame]) -> pd.DataFrame:
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
                minutes, projected = trajectory.index[len(below_target) - np.argmax(below_target[::-1])], False
            else:
                minutes, projected = TARGET_ESS / (final_ess / end_minutes), True

            convergence_rows.append({
                "experiment": experiment,
                "seed": seed,
                "parameter": column,
                "minutes": minutes,
                "projected": projected,
                "ess_per_minute": final_ess / end_minutes,
            })

    return pd.DataFrame(convergence_rows)


def get_summary_df(convergence_df: pd.DataFrame, experiments: list[str], num_sequences: dict[str, int]) -> pd.DataFrame:
    """Returns the running time of each run, averaged over its seeds, with the range over the seeds.

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
            minutes=("minutes", "mean"),
            min_minutes=("minutes", "min"),
            max_minutes=("minutes", "max"),
            projected=("projected", "any"),
            slowest_parameter=("parameter", lambda parameters: parameters.mode().iloc[0]),
        )
        .reindex(experiments)
        .reset_index()
    )
    summary_df["num_sequences"] = summary_df["experiment"].map(num_sequences)
    summary_df["status"] = np.where(summary_df["projected"], "projected", "observed")

    return summary_df


def format_minutes(minutes: float) -> str:
    """Formats a running time in minutes as a short string, e.g. `45min` or `2h13min`."""
    minutes = max(0, round(minutes))

    if minutes < 60:
        return f"{minutes}min"

    return f"{minutes // 60}h{minutes % 60}min"


def plot_convergence_time(summary_df: pd.DataFrame, experiment_colors: dict[str, tuple]) -> None:
    """Plots the mean running time of each run against its number of sequences, with the range over its seeds."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for _, row in summary_df.iterrows():
        ax.errorbar(
            row["num_sequences"], row["minutes"],
            yerr=[[row["minutes"] - row["min_minutes"]], [row["max_minutes"] - row["minutes"]]],
            fmt="X" if row["status"] == "projected" else "o", color=experiment_colors[row["experiment"]],
            markersize=10.5, markeredgecolor="white", markeredgewidth=1.5, elinewidth=1.2, capsize=3,
        )

    # the axes start slightly below zero, so that points close to zero are not cut off
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlim(-0.03 * summary_df["num_sequences"].max(), summary_df["num_sequences"].max() * 1.25)
    ax.set_ylim(-0.03 * summary_df["min_minutes"].max(), summary_df["max_minutes"].max() * 1.2)
    ax.set_xlabel("number of sequences")
    ax.set_ylabel("running time (min)")
    ax.set_title("Running time until ESS ≥ 200 on all parameters")

    fig.tight_layout()

    # labels are pushed apart iteratively until they overlap neither each other nor the points
    texts = [
        ax.text(
            row["num_sequences"], row["minutes"],
            f"{row['experiment']} ({row['num_sequences']}, {format_minutes(row['minutes'])})",
            fontsize=10, color="#444444",
        )
        for _, row in summary_df.iterrows()
    ]
    adjust_text(
        texts, x=summary_df["num_sequences"], y=summary_df["minutes"], ax=ax,
        arrowprops={"arrowstyle": "-", "color": "#bbbbbb", "linewidth": 0.8, "shrinkB": 6},
        expand=(1.3, 1.6),
    )

    fig.savefig(PLOTS_PATH / "convergence_time.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_min_ess_trajectories(ess_trajectories: dict[tuple[str, int], pd.DataFrame], experiments: list[str], experiment_colors: dict[str, tuple]) -> None:
    """Plots the ESS of the slowest parameter over running time, with one line per seed."""
    ncols = 3
    nrows = math.ceil(len(experiments) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.64 * ncols, 2.8 * nrows), squeeze=False)
    axes_by_experiment = dict(zip(experiments, axes.flat))

    for (experiment, _), trajectory in ess_trajectories.items():
        axes_by_experiment[experiment].plot(
            trajectory.index, trajectory.min(axis=1), color=experiment_colors[experiment], linewidth=2, alpha=0.6,
        )

    for experiment, ax in axes_by_experiment.items():
        ax.axhline(TARGET_ESS, color="gray", linestyle="--", linewidth=1)
        ax.set_title(experiment, fontweight="bold")
        ax.set_xlabel("running time (min)")
        ax.set_ylabel("minimum ESS")

    for ax in axes.flat[len(experiments):]:
        ax.set_visible(False)

    fig.tight_layout()
    fig.savefig(PLOTS_PATH / "ess_trajectories.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_relative_convergence_time(convergence_df: pd.DataFrame, experiments: list[str]) -> None:
    """Plots the mean time until each parameter reaches the target ESS, relative to the slowest parameter of each seed."""
    relative_df = convergence_df.assign(
        relative=convergence_df["minutes"] / convergence_df.groupby(["experiment", "seed"])["minutes"].transform("max"),
    )
    stats_df = relative_df.groupby(["parameter", "experiment"])["relative"].agg(["mean", "std"])
    mean_df = stats_df["mean"].unstack("experiment")[experiments]
    std_df = stats_df["std"].unstack("experiment")[experiments].fillna(0)

    # parameters shared by more experiments are listed first
    order_df = pd.DataFrame({"count": mean_df.notna().sum(axis=1), "mean": mean_df.mean(axis=1)})
    order = order_df.sort_values(["count", "mean"], ascending=False).index
    mean_df = mean_df.loc[order]
    std_df = std_df.loc[order]

    # each cell shows the mean and the standard deviation over the seeds
    annot_df = pd.DataFrame(
        [
            [f"{mean:.2f}±{std:.2f}" if pd.notna(mean) else "" for mean, std in zip(mean_row, std_row)]
            for mean_row, std_row in zip(mean_df.to_numpy(), std_df.to_numpy())
        ],
        index=mean_df.index,
        columns=mean_df.columns,
    )

    fig, ax = plt.subplots(figsize=(6, 0.28 * len(mean_df) + 1))

    sns.heatmap(
        mean_df, cmap="Blues", vmin=0, vmax=1, annot=annot_df, fmt="", annot_kws={"fontsize": 7},
        linewidths=1, linecolor="white", ax=ax, cbar_kws={"label": "relative time until ESS ≥ 200 (mean ± sd over seeds)"},
    )
    ax.grid(False)
    ax.set_xlabel("")
    ax.set_ylabel("")

    fig.tight_layout()
    fig.savefig(PLOTS_PATH / "relative_convergence_time.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    set_plot_theme()
    PLOTS_PATH.mkdir(parents=True, exist_ok=True)

    # each experiment keeps the same color in all plots
    experiment_colors = dict(zip(EXPERIMENTS, sns.color_palette("deep")))

    logs, num_sequences = load_logs()
    experiments = [experiment for experiment in EXPERIMENTS if experiment in num_sequences]

    ess_trajectories = {key: get_ess_trajectory(log_df) for key, log_df in logs.items()}
    convergence_df = get_convergence_df(ess_trajectories)
    summary_df = get_summary_df(convergence_df, experiments, num_sequences)

    print(summary_df[["experiment", "num_sequences", "num_seeds", "minutes", "min_minutes", "max_minutes", "status", "slowest_parameter"]].to_string(index=False))

    plot_convergence_time(summary_df, experiment_colors)
    plot_min_ess_trajectories(ess_trajectories, experiments, experiment_colors)
    plot_relative_convergence_time(convergence_df, experiments)


if __name__ == "__main__":
    main()
