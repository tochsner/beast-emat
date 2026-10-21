"""Computes the parsimony ESS of sampled trees when each tip is removed offline.

For every sampled tree, the Fitch parsimony score of the tree is computed once, together
with the score of the tree after removing each single tip. The ESS of every resulting trace
is compared to the ESS of the trace on the full tree.
"""

import argparse
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from nexwick_py import CompactTree, LeafLabelMap, parse_nexus_file
from tqdm import tqdm

from src.utils.ess import compute_ess

# bit masks over the states a, c, g, t; everything else is compatible with all states
STATE_MASKS = {"a": 1, "c": 2, "g": 4, "t": 8}
ALL_STATES = 15


def load_alignment(xml_path: Path) -> dict[str, str]:
    """Reads the sequences of a BEAST xml file by taxon."""
    sequences = {}
    pattern = re.compile(r'<sequence [^>]*?taxon="([^"]*)"[^>]*?value="([^"]*)"')
    with open(xml_path) as xml:
        for match in pattern.finditer(xml.read()):
            sequences[match.group(1)] = match.group(2)
    return sequences


def build_patterns(sequences: dict[str, str], taxa: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Returns the state masks per taxon and pattern, and the number of sites per pattern.

    Sites that cannot change the parsimony score of any subset of tips are dropped, i.e. sites with
    at most one distinct unambiguous state.
    """
    lookup = np.full(256, ALL_STATES, dtype=np.uint8)
    for state, mask in STATE_MASKS.items():
        lookup[ord(state)] = mask
        lookup[ord(state.upper())] = mask

    masks = np.stack([lookup[np.frombuffer(sequences[taxon].encode(), dtype=np.uint8)] for taxon in taxa])

    # a site is informative if at least two different unambiguous states are present
    is_unambiguous = masks != ALL_STATES
    present = np.zeros(masks.shape[1], dtype=np.uint8)
    for mask in STATE_MASKS.values():
        present += np.any(is_unambiguous & (masks == mask), axis=0)
    masks = masks[:, present >= 2]

    patterns, counts = np.unique(masks, axis=1, return_counts=True)
    return patterns, counts


def get_vertices(tree: CompactTree) -> list:
    """Returns the vertices of the tree indexed by their vertex index."""
    vertices = [None] * tree.num_vertices()
    for vertex in tree.vertices():
        vertices[vertex.index] = vertex
    return vertices


def combine(left: np.ndarray, right: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, int]:
    """Combines two Fitch state sets and returns the combined set and its cost."""
    intersection = left & right
    is_disjoint = intersection == 0
    return np.where(is_disjoint, left | right, intersection), int(weights[is_disjoint].sum())


def compute_scores(tree: CompactTree, tip_sets: np.ndarray, weights: np.ndarray) -> tuple[int, np.ndarray]:
    """Returns the Fitch score of the tree and the score after removing each single tip.

    Fitch parsimony is independent of the root, so the tree without a tip scores the same as the
    sibling's subtree joined to the rest of the tree, which is combined from a down pass and an
    up pass.
    """
    vertices = get_vertices(tree)
    root = tree.root_index()

    # post order, children before parents
    order = []
    stack = [root]
    while stack:
        index = stack.pop()
        order.append(index)
        if vertices[index].children is not None:
            stack.extend(vertices[index].children)
    order.reverse()

    down_sets = [None] * len(vertices)
    down_scores = [0] * len(vertices)
    for index in order:
        children = vertices[index].children
        if children is None:
            down_sets[index] = tip_sets[vertices[index].label_index]
            continue
        left, right = children
        down_sets[index], cost = combine(down_sets[left], down_sets[right], weights)
        down_scores[index] = down_scores[left] + down_scores[right] + cost

    # up pass, the set and score of the rest of the tree seen from the branch above a vertex
    up_sets = [None] * len(vertices)
    up_scores = [0] * len(vertices)
    removal_scores = np.zeros(tip_sets.shape[0], dtype=np.int64)
    for index in reversed(order):
        children = vertices[index].children
        if children is None:
            continue

        for child, sibling in (children, children[::-1]):
            if index == root:
                up_sets[child] = down_sets[sibling]
                up_scores[child] = down_scores[sibling]
                removed_score = down_scores[sibling]
            else:
                up_sets[child], cost = combine(up_sets[index], down_sets[sibling], weights)
                up_scores[child] = up_scores[index] + down_scores[sibling] + cost
                _, join_cost = combine(down_sets[sibling], up_sets[index], weights)
                removed_score = down_scores[sibling] + up_scores[index] + join_cost

            if vertices[child].children is None:
                removal_scores[vertices[child].label_index] = removed_score

    return down_scores[root], removal_scores


def check_removal(tree: CompactTree, tip_sets: np.ndarray, weights: np.ndarray, tip: int) -> None:
    """Checks the score after removing a tip against a plain Fitch pass without that tip."""
    vertices = get_vertices(tree)
    removed_leaf = next(v.index for v in vertices if v.label_index == tip)

    def fitch(index: int) -> tuple[np.ndarray | None, int]:
        if index == removed_leaf:
            return None, 0
        children = vertices[index].children
        if children is None:
            return tip_sets[vertices[index].label_index], 0
        (left_set, left_score), (right_set, right_score) = (fitch(child) for child in children)
        if left_set is None:
            return right_set, right_score
        if right_set is None:
            return left_set, left_score
        merged, cost = combine(left_set, right_set, weights)
        return merged, left_score + right_score + cost

    expected = fitch(tree.root_index())[1]
    actual = compute_scores(tree, tip_sets, weights)[1][tip]
    if expected != actual:
        raise AssertionError(f"Removal score of tip {tip} is {actual}, expected {expected}.")


def parse_trees(trees_path: Path) -> tuple[list[CompactTree], LeafLabelMap]:
    """Parses a trees file, closing the trees block if the run was stopped before it was written."""
    with open(trees_path, "rb") as trees_file:
        trees_file.seek(-64, 2)
        is_closed = b"End;" in trees_file.read()

    if is_closed:
        return parse_nexus_file(trees_path)

    with tempfile.TemporaryDirectory() as directory:
        closed_path = Path(directory) / trees_path.name
        shutil.copyfile(trees_path, closed_path)
        with open(closed_path, "ab") as closed_file:
            closed_file.write(b"\nEnd;\n")
        return parse_nexus_file(closed_path)


def compute_traces(trees_path: Path, tip_sets: np.ndarray, weights: np.ndarray, taxa: list[str],
                   thinning: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns the full parsimony trace and the trace per removed tip, thinned and without burn-in."""
    trees, label_map = parse_trees(trees_path)
    labels = label_map.labels()

    # reorder the tip sets to the label indices of the tree file
    tip_sets = tip_sets[[taxa.index(label) for label in labels]]

    burn_in = len(trees) // 10
    selected = trees[burn_in::thinning]

    check_removal(selected[0], tip_sets, weights, tip=0)
    check_removal(selected[-1], tip_sets, weights, tip=len(labels) - 1)

    full_trace = np.zeros(len(selected))
    removal_traces = np.zeros((len(selected), len(labels)))
    for i, tree in enumerate(tqdm(selected, desc=trees_path.name)):
        full_trace[i], removal_traces[i] = compute_scores(tree, tip_sets, weights)

    # tips are reported by name in the order of the tree file labels
    return full_trace, removal_traces[:, [labels.index(taxon) for taxon in taxa]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests", type=Path, default=Path("../tests"))
    parser.add_argument("--name", default="sars_emat")
    parser.add_argument("--tree", default="ma_sars_cov_2")
    parser.add_argument("--seeds", type=int, nargs="+", default=[22, 32, 42])
    parser.add_argument("--thinning", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("tip_removal_parsimony_ess.csv"))
    args = parser.parse_args()

    sequences = load_alignment(args.tests / f"{args.name}.xml")
    taxa = sorted(sequences)
    patterns, weights = build_patterns(sequences, taxa)
    tip_sets = patterns
    print(f"{len(taxa)} taxa, {tip_sets.shape[1]} informative patterns")

    rows = []
    for seed in args.seeds:
        trees_path = args.tests / f"{args.name}-{args.tree}-{seed}.trees"
        full_trace, removal_traces = compute_traces(trees_path, tip_sets, weights, taxa, args.thinning)
        full_ess = compute_ess(full_trace)
        for tip, taxon in enumerate(taxa):
            rows.append({
                "seed": seed,
                "taxon": taxon,
                "full_ess": full_ess,
                "removed_ess": compute_ess(removal_traces[:, tip]),
                "full_mean": full_trace.mean(),
                "removed_mean": removal_traces[:, tip].mean(),
                "samples": len(full_trace),
            })

    result = pd.DataFrame(rows)
    result["ess_change"] = result["removed_ess"] - result["full_ess"]
    result.to_csv(args.output, index=False)
    print(result.groupby("seed")[["full_ess", "removed_ess", "ess_change"]].describe().T.to_string())


if __name__ == "__main__":
    main()
