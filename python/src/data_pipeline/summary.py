"""Summarises the FASTA datasets in `data/fasta/` as a markdown table."""

import argparse
import itertools
from pathlib import Path

import numpy as np

from src.data_pipeline.fasta import iterate_records

REPO_ROOT = Path(__file__).resolve().parents[3]


def character_table(characters):
    """Returns a boolean lookup table over byte values that is true for the given characters."""
    table = np.zeros(256, dtype=bool)
    table[list(characters)] = True
    return table


# pairs compared per dataset before switching to random sampling of pairs
MAX_PAIRS = 50_000
SEED = 0

NUCLEOTIDE_TABLE = character_table(b"ACGT")
# gaps and unknown bases count as missing, the other IUPAC codes as ambiguous
MISSING_TABLE = character_table(b"-.?NX")
AMBIGUOUS_TABLE = character_table(b"RYSWKMBDHV")


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
    """Returns the summary row of a FASTA file: name, sequence count, length range, missing and ambiguous fractions, mean pairwise distance and fully constant sites."""
    sequences = [sequence.upper() for _, sequence in iterate_records(path)]
    if not sequences:
        raise ValueError(f"no sequences in {path}")

    lengths = [len(sequence) for sequence in sequences]
    encoded = [np.frombuffer(sequence.encode("ascii"), dtype=np.uint8) for sequence in sequences]
    row = {
        "name": path.stem,
        "sequences": len(sequences),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "missing_fraction": compute_character_fraction(encoded, MISSING_TABLE),
        "ambiguous_fraction": compute_character_fraction(encoded, AMBIGUOUS_TABLE),
        "mean_distance": None,
        "pair_count": None,
        "constant_sites": None,
    }

    # distances between sites only make sense for aligned sequences of equal length
    if row["min_length"] == row["max_length"]:
        row["mean_distance"], row["pair_count"] = compute_mean_pairwise_distance(encoded, rng)
        row["constant_sites"] = count_fully_constant_sites(encoded)
    return row


def compute_character_fraction(encoded, table):
    """Returns the fraction of all characters over all sequences that are marked in the lookup `table`."""
    total = sum(len(sequence) for sequence in encoded)
    marked = sum(np.count_nonzero(table[sequence]) for sequence in encoded)
    return marked / total


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


def count_fully_constant_sites(encoded):
    """Returns the number of sites where every sequence has the same nucleotide, without any gaps or ambiguous bases.

    The sequences must be aligned, so they all have the same length.
    """
    matrix = np.stack(encoded)
    nucleotides = NUCLEOTIDE_TABLE[matrix].all(axis=0)
    identical = (matrix == matrix[0]).all(axis=0)
    return int(np.count_nonzero(nucleotides & identical))


def format_table(rows):
    """Returns the summary rows as a markdown table."""
    lines = [
        "| Dataset | Sequences | Min length | Max length | Missing fraction | Ambiguous fraction | Mean pairwise distance | Pairs compared | Fully constant sites |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        if row["mean_distance"] is None:
            distance, pairs = "n/a", "n/a"
        else:
            distance, pairs = f"{row['mean_distance']:.4f}", row["pair_count"]
        constant = "n/a" if row["constant_sites"] is None else row["constant_sites"]
        lines.append(
            f"| {row['name']} | {row['sequences']} | {row['min_length']} | {row['max_length']} "
            f"| {row['missing_fraction']:.4f} | {row['ambiguous_fraction']:.4f} | {distance} | {pairs} | {constant} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
