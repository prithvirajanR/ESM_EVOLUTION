"""Small mathematical helpers used by the pre-correction REVO controller.

REVO keeps several live sequence walkers. This file measures how different the
walkers are, turns their distances and statistical weights into one variation
score, and chooses a merge-and-clone operation expected to increase that score.
It never receives the target amino-acid sequence itself.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class RevoOperation:
    """Describe which walker to clone and which nearby pair to merge."""
    clone: int
    merge_a: int
    merge_b: int
    expected_gain: float


def normalized_sequence_distances(sequences: Sequence[str]) -> np.ndarray:
    """Return pairwise mismatch fractions among equal-length live sequences."""
    n = len(sequences)
    matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        if len(sequences[i]) == 0:
            continue
        for j in range(i + 1, n):
            if len(sequences[i]) != len(sequences[j]):
                raise ValueError("All walker sequences must have equal length")
            value = sum(a != b for a, b in zip(sequences[i], sequences[j])) / len(
                sequences[i]
            )
            matrix[i, j] = matrix[j, i] = value
    return matrix


def cosine_distances(vectors: np.ndarray) -> np.ndarray:
    """Return pairwise cosine distance for one vector per walker."""
    vectors = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / np.maximum(norms, 1e-12)
    similarities = np.clip(normalized @ normalized.T, -1.0, 1.0)
    distances = 1.0 - similarities
    np.fill_diagonal(distances, 0.0)
    return distances


def standardized_history_distances(features: np.ndarray) -> np.ndarray:
    """Return Euclidean distances after robust feature-wise scaling."""
    features = np.asarray(features, dtype=float)
    if features.ndim != 2:
        raise ValueError("History features must be a two-dimensional array")
    median = np.median(features, axis=0)
    mad = np.median(np.abs(features - median), axis=0)
    scale = np.where(mad > 1e-12, 1.4826 * mad, 1.0)
    z = (features - median) / scale
    diffs = z[:, None, :] - z[None, :, :]
    distances = np.sqrt(np.sum(diffs * diffs, axis=2))
    np.fill_diagonal(distances, 0.0)
    return distances


def _scale_distance_matrix(matrix: np.ndarray) -> np.ndarray:
    """Put one distance component on a comparable median-based scale."""
    nonzero = matrix[matrix > 1e-12]
    if nonzero.size == 0:
        return np.zeros_like(matrix, dtype=float)
    return matrix / float(np.median(nonzero))


def combined_walker_distances(
    sequences: Sequence[str],
    pooled_embeddings: np.ndarray,
    history_features: np.ndarray,
    component_weights: tuple[float, float, float] = (0.35, 0.40, 0.25),
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Combine sequence, embedding, and trajectory-history dissimilarities."""
    if not np.isclose(sum(component_weights), 1.0):
        raise ValueError("Distance component weights must sum to one")
    components = {
        "sequence": _scale_distance_matrix(normalized_sequence_distances(sequences)),
        "embedding": _scale_distance_matrix(cosine_distances(pooled_embeddings)),
        "history": _scale_distance_matrix(
            standardized_history_distances(history_features)
        ),
    }
    total = sum(
        weight * components[name]
        for weight, name in zip(component_weights, ("sequence", "embedding", "history"))
    )
    np.fill_diagonal(total, 0.0)
    return total, components


def update_distance_scale(
    distances: np.ndarray, previous: float | None, ema: float = 0.20
) -> float:
    """Update the REVO distance scale from the median nonzero separation."""
    nonzero = distances[distances > 1e-12]
    observed = float(np.median(nonzero)) if nonzero.size else 1.0
    if previous is None or not np.isfinite(previous) or previous <= 0:
        return max(observed, 1e-12)
    return max((1.0 - ema) * previous + ema * observed, 1e-12)


def variation(
    weights: np.ndarray,
    distances: np.ndarray,
    distance_scale: float,
    alpha: float,
    weight_floor: float,
) -> float:
    """Compute the REVO ensemble variation objective."""
    weights = np.asarray(weights, dtype=float)
    phi = np.log1p(weights / max(weight_floor, 1e-15))
    scaled = np.power(distances / max(distance_scale, 1e-12), alpha)
    return float(np.sum(np.triu(phi[:, None] * phi[None, :] * scaled, k=1)))


def _operation_outcome(
    weights: np.ndarray,
    distances: np.ndarray,
    ancestors: Sequence[str],
    clone: int,
    survivor: int,
    removed: int,
) -> tuple[np.ndarray, np.ndarray, list[str], list[int]]:
    """Build the virtual ensemble produced by one merge and one clone.

    The clone's weight is split equally between the original and copied slots.
    The surviving member of the merged pair receives both merge weights. This
    keeps the total statistical weight unchanged.
    """
    new_weights = np.asarray(weights, dtype=float).copy()
    clone_weight = new_weights[clone]
    merged_weight = new_weights[survivor] + new_weights[removed]
    new_weights[clone] = clone_weight / 2.0
    new_weights[survivor] = merged_weight
    new_weights[removed] = clone_weight / 2.0

    source_indices = list(range(len(weights)))
    source_indices[removed] = clone
    new_distances = distances[np.ix_(source_indices, source_indices)]
    new_ancestors = list(ancestors)
    new_ancestors[removed] = ancestors[clone]
    return new_weights, new_distances, new_ancestors, source_indices


def _lineage_fraction(ancestors: Sequence[str]) -> float:
    """Estimate lineage occupancy by counting walker slots in this old version."""
    counts = Counter(ancestors)
    return max(counts.values(), default=0) / max(len(ancestors), 1)


def select_revo_operation(
    weights: np.ndarray,
    distances: np.ndarray,
    ancestors: Sequence[str],
    distance_scale: float,
    alpha: float = 4.0,
    weight_floor: float | None = None,
    merge_weight_cap: float | None = None,
    merge_distance_quantile: float = 0.30,
    max_ancestor_fraction: float = 0.50,
    excluded_clones: set[int] | None = None,
) -> RevoOperation | None:
    """Choose the legal clone/merge operation with the best expected gain.

    Both possible merge survivors are considered and averaged according to their
    statistical weights. This historical version chooses on expected variation;
    it does not recheck the variation of the random outcome eventually drawn.
    """
    n = len(weights)
    weights = np.asarray(weights, dtype=float)
    weight_floor = weight_floor if weight_floor is not None else 1e-4 / n
    merge_weight_cap = (
        merge_weight_cap if merge_weight_cap is not None else min(0.50, 2.5 / n)
    )
    excluded_clones = excluded_clones or set()
    nonzero = distances[distances > 1e-12]
    if nonzero.size == 0:
        return None
    merge_cutoff = float(np.quantile(nonzero, merge_distance_quantile))
    current = variation(weights, distances, distance_scale, alpha, weight_floor)
    best = None

    for clone in range(n):
        if clone in excluded_clones or weights[clone] < 2.0 * weight_floor:
            continue
        for merge_a in range(n):
            for merge_b in range(merge_a + 1, n):
                if clone in (merge_a, merge_b):
                    continue
                if weights[merge_a] + weights[merge_b] > merge_weight_cap + 1e-12:
                    continue
                if distances[merge_a, merge_b] > merge_cutoff + 1e-12:
                    continue

                p_a = weights[merge_a] / (weights[merge_a] + weights[merge_b])
                outcomes = []
                legal = True
                for probability, survivor, removed in (
                    (p_a, merge_a, merge_b),
                    (1.0 - p_a, merge_b, merge_a),
                ):
                    new_w, new_d, new_a, _ = _operation_outcome(
                        weights,
                        distances,
                        ancestors,
                        clone,
                        survivor,
                        removed,
                    )
                    if _lineage_fraction(new_a) > max_ancestor_fraction + 1e-12:
                        legal = False
                        break
                    outcomes.append(
                        probability
                        * variation(new_w, new_d, distance_scale, alpha, weight_floor)
                    )
                if not legal:
                    continue
                gain = float(sum(outcomes) - current)
                if gain > 1e-12 and (best is None or gain > best.expected_gain):
                    best = RevoOperation(clone, merge_a, merge_b, gain)
    return best


def execute_revo_operation(
    weights: np.ndarray,
    distances: np.ndarray,
    ancestors: Sequence[str],
    operation: RevoOperation,
    rng,
) -> tuple[np.ndarray, np.ndarray, list[str], list[int], int, int]:
    """Draw the merge survivor and return the resulting weighted walkers."""
    a, b = operation.merge_a, operation.merge_b
    p_a = weights[a] / (weights[a] + weights[b])
    survivor, removed = (a, b) if rng.random() < p_a else (b, a)
    new_w, new_d, new_a, sources = _operation_outcome(
        weights, distances, ancestors, operation.clone, survivor, removed
    )
    return new_w, new_d, new_a, sources, survivor, removed
