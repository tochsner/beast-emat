"""TreeTracer-compatible topology ESS for already parsed Nexwick trees.

Based on the estimators in beast-dev/treetracer at commit
baffaba0392a2d21398e46bf3d4ea42593d57220. Trees must be in MCMC order
and share a Nexwick label-index mapping (as returned by one parse call).
No file reading, burn-in removal, thinning, or branch-length analysis occurs.
"""

from collections.abc import Callable, Iterable
from operator import index

import numpy as np
from nexwick_py import CompactTree
from scipy.stats import norm, rankdata

from src.utils.rf_distance import _clade_masks


def _positive_integer(value: int, name: str) -> int:
    """Validate an integer tuning parameter."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    try:
        value = index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer") from None
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _tree_splits(trees: Iterable[CompactTree], rooted: bool) -> list[set[int]]:
    """Extract each topology once using the shared label-index mapping."""
    splits = []
    leaves = None
    for tree in trees:
        clades, full_mask = _clade_masks(tree)
        if leaves is not None and leaves != full_mask:
            raise ValueError("tree ESS requires trees with the same leaf labels")
        leaves = full_mask
        if not rooted:
            clades = {
                min(mask, full_mask ^ mask)
                for mask in clades
                if mask.bit_count() > 1 and (full_mask ^ mask).bit_count() > 1
            }
        splits.append(clades)
    if not splits:
        raise ValueError("tree ESS requires at least one tree")
    return splits


def _ess_from_lags(
    correlation: Callable[[int], float], n: int, max_lag: int, cap_at_n: bool = False
) -> float:
    """Reduce complete lag pairs with TreeTracer's positive/monotone rule."""
    previous = float("inf")
    total = 0.0
    for lag in range(0, max_lag - 1, 2):
        pair = (1.0 if lag == 0 else correlation(lag)) + correlation(lag + 1)
        if not np.isfinite(pair):
            raise ValueError("lag correlations must be finite")
        if pair < 0:
            break
        previous = min(previous, pair)
        total += previous
    tau = 2.0 * total - 1.0
    if tau < 0:
        tau = 1.0
    ess = float(n / tau) if tau else float("inf")
    return min(float(n), ess) if cap_at_n else ess


def compute_tree_ess(
    trees: Iterable[CompactTree],
    *,
    rooted: bool = False,
    min_samples: int = 5,
    cap_at_n: bool = False,
) -> float:
    """Compute TreeTracer's Fréchet-correlation ESS (treeess lower bound).

    Pass a single ordered chain of CompactTree objects after any desired
    burn-in. Label indices must refer to the same taxa in every tree; do not
    combine independently parsed files with different label mappings.
    Unrooted RF is the TreeTracer default; rooted=True compares rooted clades.
    Branch lengths are ignored. Identical topologies return 1.0, including a
    singleton. Other chains need at least min_samples + 2 trees. An empty
    chain raises ValueError. ESS is uncapped unless cap_at_n=True.

    Each unordered RF distance is evaluated once. Only endpoint and lag sums
    of squared distances are retained: O(n) numeric storage rather than an
    O(n**2) distance matrix, in addition to cached topology masks. Exact
    Fréchet ESS still needs O(n**2) pairwise distance work.
    """
    min_samples = _positive_integer(min_samples, "min_samples")
    splits = _tree_splits(trees, rooted)
    n = len(splits)
    if all(topology == splits[0] for topology in splits[1:]):
        return 1.0
    if n < min_samples + 2:
        raise ValueError("chain is too short: at least min_samples + 2 trees required")

    right = np.zeros(n)
    left = np.zeros(n)
    lag_sums = np.zeros(n - min_samples)
    for i, topology in enumerate(splits[:-1]):
        squared = np.fromiter(
            (len(topology ^ other) ** 2 for other in splits[i + 1 :]),
            dtype=float,
            count=n - i - 1,
        )
        left[i] = squared.sum()
        right[i + 1 :] += squared
        count = min(len(squared), len(lag_sums) - 1)
        lag_sums[1 : count + 1] += squared[:count]
    front = np.cumsum(right)
    back = np.cumsum(left[::-1])[::-1]

    def correlation(lag: int) -> float:
        retained = n - lag
        denominator = retained * (retained - 1)
        v_front = front[retained - 1] / denominator
        v_back = back[lag] / denominator
        if v_front == 0 or v_back == 0:
            return 1.0
        displacement = lag_sums[lag] / retained
        return float((v_front + v_back - displacement) / (2 * np.sqrt(v_front * v_back)))

    return _ess_from_lags(correlation, n, n - min_samples, cap_at_n)


def _rank_split_ess(trace: np.ndarray) -> float:
    """Compute rank-normalized split-chain variogram ESS as in TreeTracer."""
    ranks = rankdata(trace, method="average")
    normalized = norm.ppf((ranks - 0.375) / (len(trace) + 0.25))
    half = len(trace) // 2
    chains = normalized[: 2 * half].reshape(2, half)
    means = chains.mean(axis=1)
    within = ((chains - means[:, None]) ** 2).sum() / (2 * (half - 1))
    between = ((means - chains.mean()) ** 2).sum()
    variance = within * (half - 1) / half + between
    if variance == 0:
        return float(2 * half)

    def correlation(lag: int) -> float:
        differences = chains[:, lag:] - chains[:, :-lag]
        return float(1 - np.mean(differences**2) / (2 * variance))

    return _ess_from_lags(correlation, 2 * half, half)


def compute_tree_pseudo_ess(
    trees: Iterable[CompactTree],
    *,
    rooted: bool = False,
    n_refs: int = 100,
    seed: int | None = None,
) -> dict:
    """Compute TreeTracer's reference-tree pseudo-ESS distribution.

    Accept the same parsed, ordered trees and shared label mapping as
    compute_tree_ess. Sample up to n_refs reference indices without replacement.
    Compute only those RF traces, rank-normalize them, and use TreeTracer's
    split-chain ESS estimator (not the scalar Tracer helper compute_ess).
    Return ess_values, ref_indices, min, median, max, and n_refs_used.
    A nonempty chain shorter than four returns empty arrays and NaN summaries.
    Odd chains drop the final sample after rank normalization for split ESS.
    A constant trace returns the retained even sample count, matching upstream;
    use Fréchet ESS when identical topologies should give ESS=1.

    Distance work is O(n * min(n_refs, n)); only one numeric trace is stored
    at a time. seed makes reference selection reproducible.
    """
    n_refs = _positive_integer(n_refs, "n_refs")
    splits = _tree_splits(trees, rooted)
    n = len(splits)
    refs = (
        np.random.default_rng(seed).choice(n, size=min(n_refs, n), replace=False)
        if n >= 4 else np.array([], dtype=int)
    )
    values = np.empty(len(refs))
    for i, ref in enumerate(refs):
        trace = np.fromiter((len(s ^ splits[ref]) for s in splits), dtype=float, count=n)
        values[i] = _rank_split_ess(trace)
    valid = values[~np.isnan(values)]
    return {
        "ess_values": values,
        "ref_indices": refs,
        "min": float(valid.min()) if valid.size else float("nan"),
        "median": float(np.median(valid)) if valid.size else float("nan"),
        "max": float(valid.max()) if valid.size else float("nan"),
        "n_refs_used": len(refs),
    }
