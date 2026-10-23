"""Summarises the FASTA datasets in `data/fasta/` as a markdown table."""

import argparse
import itertools
from pathlib import Path

import numpy as np

from src.data_pipeline.fasta import iterate_records

REPO_ROOT = Path(__file__).resolve().parents[3]

# pairs compared per dataset before switching to random sampling of pairs
MAX_PAIRS = 50_000
SEED = 0

NUCLEOTIDE_TABLE = np.zeros(256, dtype=bool)
NUCLEOTIDE_TABLE[list(b"ACGT")] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta-dir", type=Path, default=REPO_ROOT / "data" / "fasta", help="directory with the FASTA files")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "summary.md", help="markdown output file")
    args = parser.parse_args()

    rng = np.random.default_rng(SEED)
    rows = []
    for fasta_path in sorted(args.fasta_dir.glob("*.fasta")):
        print(f"summarising {fasta_path.name}", flush=True)
        rows.append(summarise_dataset(fasta_path, rng))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(format_table(rows))
    print(f"wrote {args.out}")


def summarise_dataset(path, rng):
    """Returns the summary row of a FASTA file: name, sequence count, length range and mean pairwise distance."""
    sequences = [sequence.upper() for _, sequence in iterate_records(path)]
    if not sequences:
        raise ValueError(f"no sequences in {path}")

    lengths = [len(sequence) for sequence in sequences]
    row = {
        "name": path.stem,
        "sequences": len(sequences),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "mean_distance": None,
        "pair_count": None,
    }

    # distances between sites only make sense for aligned sequences of equal length
    if row["min_length"] == row["max_length"]:
        encoded = [np.frombuffer(sequence.encode("ascii"), dtype=np.uint8) for sequence in sequences]
        row["mean_distance"], row["pair_count"] = compute_mean_pairwise_distance(encoded, rng)
    return row


def compute_mean_pairwise_distance(encoded, rng):
    """Returns the mean p-distance over pairs of sequences and the number of pairs compared.

    The p-distance counts the differing sites among the sites where both sequences have a
    nucleotide, so gaps and ambiguous bases are ignored. All pairs are compared unless there
    are more than `MAX_PAIRS`, in which case a random sample of pairs is used.
    """
    count = len(encoded)
    pairs = sample_pairs(count, rng)

    distances = []
    for i, j in pairs:
        a, b = encoded[i], encoded[j]
        valid = NUCLEOTIDE_TABLE[a] & NUCLEOTIDE_TABLE[b]
        valid_count = np.count_nonzero(valid)
        if valid_count == 0:
            continue
        distances.append(np.count_nonzero((a != b) & valid) / valid_count)

    if not distances:
        return float("nan"), 0
    return float(np.mean(distances)), len(distances)


def sample_pairs(count, rng):
    """Yields index pairs (i, j) with i != j, all of them if there are few enough, else a random sample."""
    if count * (count - 1) // 2 <= MAX_PAIRS:
        return itertools.combinations(range(count), 2)

    first = rng.integers(0, count, size=MAX_PAIRS)
    second = rng.integers(0, count - 1, size=MAX_PAIRS)
    second += second >= first
    return zip(first.tolist(), second.tolist())


def format_table(rows):
    """Returns the summary rows as a markdown table."""
    lines = [
        "| Dataset | Sequences | Min length | Max length | Mean pairwise distance | Pairs compared |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        if row["mean_distance"] is None:
            distance, pairs = "n/a", "n/a"
        else:
            distance, pairs = f"{row['mean_distance']:.4f}", row["pair_count"]
        lines.append(
            f"| {row['name']} | {row['sequences']} | {row['min_length']} | {row['max_length']} "
            f"| {distance} | {pairs} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
