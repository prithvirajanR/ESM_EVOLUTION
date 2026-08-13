#!/usr/bin/env python3
"""Final corrected adaptive independent-walker engine.

This is the production path distilled from the verified research engine used
for the successful independent-walker runs. It intentionally contains no
REVO resampling, gateway controller, rescue pulse, neutral drift, plotting, or
alternative experiment modes.

The optimization signal is target ESM2 embedding similarity (PRCS) plus the
same dynamically recomputed epistasis term used in the verified runs. Target
amino-acid identities are not read by this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
import math
import pickle
import random
from pathlib import Path
from typing import Sequence
import warnings

import esm
import numpy as np
from scipy.spatial.distance import jensenshannon
import torch
import torch.nn.functional as F


warnings.filterwarnings("ignore", category=FutureWarning)

ESM2_MODEL = "esm2_t33_650M_UR50D"
REPR_LAYER = 33
AMINO_ACIDS = tuple("ACDEFGHIKLMNPQRSTVWY")

# These are the settings active in the verified corrected independent runs.
INITIAL_TEMPERATURE = 0.30
FINAL_TEMPERATURE = 0.005
PROPOSALS_PER_STEP = 40

BIG_PROTEIN_MIN_LENGTH = 300
BIG_HEATING_STEPS = 1000
BIG_HEATING_K_MIN = 5
BIG_HEATING_K_MAX = 10
BIG_HEATING_PROBABILITY = 0.70

ADAPTIVE_WINDOW = 400
ADAPTIVE_PRCS_EPSILON = 0.005
ADAPTIVE_STAGE1_THRESHOLD = 0.50
ADAPTIVE_STAGE2_THRESHOLD = 0.60
ADAPTIVE_STAGE3_THRESHOLD = 0.75
ADAPTIVE_PLATEAU_THRESHOLD = 0.35
ADAPTIVE_STAGE1_K_CAP = 4
ADAPTIVE_STAGE2_K_CAP = 2
ADAPTIVE_STAGE3_K_CAP = 1
ADAPTIVE_STAGE1_COUPLED_CAP = 0.40
ADAPTIVE_STAGE2_COUPLED_CAP = 0.25
ADAPTIVE_STAGE3_COUPLED_CAP = 0.10
P_GUIDED_ADAPTIVE = 0.85
P_GUIDED_HEATING = 0.50
PRCS_WORST_FRACTION = 0.30

COUPLING_RECOMPUTE_EVERY = 150
COUPLING_THRESHOLD = 0.01
COUPLING_ROBUST_Z = 2.5
COUPLING_GROUP_MIN = 2
COUPLING_GROUP_MAX = 4

LAMBDA_MIN = 0.10
LAMBDA_CAP = 0.50
LAMBDA_COUPLING_SCALE = 5.0
LAMBDA_RAMP_FRACTION = 0.40

PLL_DIAGNOSTIC_EVERY = 500
LOG_EVERY = 50
CHECKPOINT_EVERY = 200


@dataclass(frozen=True)
class WalkConfig:
    """The single supported policy: corrected adaptive independent search."""

    length: int
    max_steps: int
    k_single: int
    p_coupled: float
    n_groups: int
    group_size_min: int
    group_size_max: int
    cooling_rate: float
    heating_steps: int
    heating_k_min: int
    heating_k_max: int
    heating_probability: float
    proposals_per_step: int = PROPOSALS_PER_STEP

    @classmethod
    def for_sequence(cls, length: int, max_steps: int) -> "WalkConfig":
        if length < 2:
            raise ValueError("Sequence length must be at least two residues")
        if length > 1022:
            raise ValueError(
                "This finalized single-protein engine supports at most 1022 residues. "
                "Long-sequence chunking belongs to the separate strain handoff work."
            )
        if max_steps < 1:
            raise ValueError("max_steps must be positive")

        group_min = min(COUPLING_GROUP_MIN, length)
        group_max = max(group_min, min(COUPLING_GROUP_MAX, length))
        average_group = (group_min + group_max) / 2.0
        n_groups = max(15, int(0.7 * length / average_group))
        p_coupled = min(0.80, 0.30 + length / 1000.0)
        cooling_rate = (FINAL_TEMPERATURE / INITIAL_TEMPERATURE) ** (1.0 / max_steps)

        is_big = length >= BIG_PROTEIN_MIN_LENGTH
        heating_steps = min(BIG_HEATING_STEPS, max_steps // 4) if is_big else 0
        heating_k_min = min(BIG_HEATING_K_MIN, length) if is_big else 0
        heating_k_max = min(BIG_HEATING_K_MAX, length) if is_big else 0
        return cls(
            length=length,
            max_steps=max_steps,
            k_single=max(1, round(length / 50)),
            p_coupled=p_coupled,
            n_groups=n_groups,
            group_size_min=group_min,
            group_size_max=group_max,
            cooling_rate=cooling_rate,
            heating_steps=heating_steps,
            heating_k_min=heating_k_min,
            heating_k_max=heating_k_max,
            heating_probability=BIG_HEATING_PROBABILITY if is_big else 0.0,
        )

    def as_dict(self) -> dict:
        return asdict(self)


def get_coupling_sample_pairs(length: int) -> int:
    """Match the verified engine's practical coupling-map sample count."""

    return min(4000, max(1000, length * 6))


def random_sequence(length: int, rng: random.Random) -> str:
    return "".join(rng.choice(AMINO_ACIDS) for _ in range(length))


def mutate_positions(sequence: str, positions: Sequence[int], amino_acids: Sequence[str]) -> str:
    values = list(sequence)
    for position, amino_acid in zip(positions, amino_acids):
        values[position] = amino_acid
    return "".join(values)


def write_sequence_fasta(path: Path, name: str, sequence: str) -> None:
    with path.open("w") as handle:
        handle.write(f">{name}\n")
        for start in range(0, len(sequence), 60):
            handle.write(sequence[start : start + 60] + "\n")


def load_esm2(device: str):
    print("Loading ESM2...")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.eval().to(torch.device(device))
    batch_converter = alphabet.get_batch_converter()
    print(f"  Model loaded on {device}")
    return model, alphabet, batch_converter


@torch.no_grad()
def get_per_residue_embedding(sequence, model, alphabet, batch_converter, device):
    if len(sequence) > 1022:
        raise ValueError("ESM2 input exceeds 1022 amino acids")
    _, _, tokens = batch_converter([("sequence", sequence)])
    tokens = tokens.to(device)
    output = model(tokens, repr_layers=[REPR_LAYER], return_contacts=False)
    return output["representations"][REPR_LAYER][0, 1 : len(sequence) + 1]


@torch.no_grad()
def get_per_residue_embeddings_batch(
    sequences, model, alphabet, batch_converter, device, max_batch=None
):
    if not sequences:
        return []
    if any(len(sequence) != len(sequences[0]) for sequence in sequences):
        raise ValueError("Candidate batches must contain equal-length sequences")
    if len(sequences[0]) > 1022:
        raise ValueError("ESM2 input exceeds 1022 amino acids")

    if max_batch is None:
        length = len(sequences[0])
        if length < 150:
            max_batch = 20
        elif length < 300:
            max_batch = 12
        elif length < 500:
            max_batch = 8
        else:
            max_batch = 4

    embeddings = []
    for start in range(0, len(sequences), max_batch):
        batch = sequences[start : start + max_batch]
        data = [(f"candidate_{index}", sequence) for index, sequence in enumerate(batch)]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(device)
        output = model(tokens, repr_layers=[REPR_LAYER], return_contacts=False)
        representations = output["representations"][REPR_LAYER]
        for row, sequence in enumerate(batch):
            embeddings.append(representations[row, 1 : len(sequence) + 1].clone())
        del representations, output, tokens
    return embeddings


def score_prcs(candidate_embedding, target_embedding):
    per_position = F.cosine_similarity(candidate_embedding, target_embedding, dim=1)
    return per_position.mean().item(), per_position.cpu().numpy()


def score_fixed_diagnostic(per_position_prcs):
    """Map-independent logging score; it does not control acceptance."""

    values = np.asarray(per_position_prcs, dtype=float)
    if values.size == 0:
        return float("-inf"), float("nan"), float("nan")
    lower_quartile = float(np.quantile(values, 0.25))
    window = max(1, min(values.size, max(8, values.size // 12)))
    kernel = np.ones(window, dtype=float) / window
    weakest_window = float(np.convolve(values, kernel, mode="valid").min())
    diagnostic = 0.70 * float(values.mean()) + 0.20 * lower_quartile + 0.10 * weakest_window
    return diagnostic, lower_quartile, weakest_window


def score_epistatic(candidate_embedding, target_embedding, groups):
    if not groups:
        return 0.0
    values = []
    for group in groups:
        for left, right in combinations(group, 2):
            candidate_difference = candidate_embedding[left] - candidate_embedding[right]
            target_difference = target_embedding[left] - target_embedding[right]
            value = F.cosine_similarity(
                candidate_difference.unsqueeze(0), target_difference.unsqueeze(0)
            ).item()
            values.append(value)
    return float(np.mean(values)) if values else 0.0


def compute_combined_score(candidate_embedding, target_embedding, groups, coupling_lambda):
    prcs, per_position = score_prcs(candidate_embedding, target_embedding)
    epistasis = (
        score_epistatic(candidate_embedding, target_embedding, groups)
        if coupling_lambda > 0.01
        else 0.0
    )
    return prcs + coupling_lambda * epistasis, prcs, epistasis, per_position


@torch.no_grad()
def score_true_pll(
    sequence, model, alphabet, batch_converter, device, position_batch_size=16
):
    """Masked pseudo-log-likelihood for logging only."""

    _, _, tokens = batch_converter([("sequence", sequence)])
    tokens = tokens.to(device)
    values = []
    for start in range(0, len(sequence), position_batch_size):
        end = min(start + position_batch_size, len(sequence))
        batch = tokens.repeat(end - start, 1)
        rows = torch.arange(end - start, device=device)
        token_positions = torch.arange(start, end, device=device) + 1
        true_tokens = batch[rows, token_positions].clone()
        batch[rows, token_positions] = alphabet.mask_idx
        output = model(batch, repr_layers=[], return_contacts=False)
        log_probabilities = torch.log_softmax(output["logits"], dim=-1)
        values.extend(log_probabilities[rows, token_positions, true_tokens].cpu().tolist())
        del output, log_probabilities, batch
    return float(np.mean(values))


@torch.no_grad()
def compute_single_masked_marginals(
    sequence, model, alphabet, batch_converter, device, positions
):
    _, _, tokens = batch_converter([("sequence", sequence)])
    tokens = tokens.to(device)
    amino_acid_indices = [alphabet.get_idx(amino_acid) for amino_acid in AMINO_ACIDS]
    marginals = {}
    for position in positions:
        masked = tokens.clone()
        masked[0, position + 1] = alphabet.mask_idx
        output = model(masked, repr_layers=[], return_contacts=False)
        probabilities = torch.softmax(output["logits"][0, position + 1], dim=-1)
        marginals[position] = probabilities[amino_acid_indices].cpu().numpy()
    return marginals


@torch.no_grad()
def compute_double_masked_marginals(
    sequence, model, alphabet, batch_converter, device, left, right
):
    _, _, tokens = batch_converter([("sequence", sequence)])
    tokens = tokens.to(device)
    amino_acid_indices = [alphabet.get_idx(amino_acid) for amino_acid in AMINO_ACIDS]
    masked = tokens.clone()
    masked[0, left + 1] = alphabet.mask_idx
    masked[0, right + 1] = alphabet.mask_idx
    output = model(masked, repr_layers=[], return_contacts=False)
    left_probabilities = torch.softmax(output["logits"][0, left + 1], dim=-1)
    right_probabilities = torch.softmax(output["logits"][0, right + 1], dim=-1)
    return (
        left_probabilities[amino_acid_indices].cpu().numpy(),
        right_probabilities[amino_acid_indices].cpu().numpy(),
    )


@torch.no_grad()
def compute_epistatic_coupling_map(
    sequence, model, alphabet, batch_converter, device, sample_count, rng
):
    all_pairs = list(combinations(range(len(sequence)), 2))
    sampled_pairs = rng.sample(all_pairs, min(sample_count, len(all_pairs)))
    unique_positions = sorted({position for pair in sampled_pairs for position in pair})
    single_marginals = compute_single_masked_marginals(
        sequence, model, alphabet, batch_converter, device, unique_positions
    )
    scores = []
    for left, right in sampled_pairs:
        double_left, double_right = compute_double_masked_marginals(
            sequence, model, alphabet, batch_converter, device, left, right
        )
        coupling = (
            jensenshannon(single_marginals[left], double_left)
            + jensenshannon(single_marginals[right], double_right)
        ) / 2
        if not np.isnan(coupling):
            scores.append((left, right, coupling))
    scores.sort(key=lambda item: item[2], reverse=True)
    return scores


def build_coupling_groups(
    coupling_scores,
    length,
    group_count,
    group_size_min,
    group_size_max,
    threshold,
):
    if not coupling_scores:
        return [], threshold
    strengths = np.asarray([strength for _, _, strength in coupling_scores], dtype=float)
    median = float(np.median(strengths))
    mad = float(np.median(np.abs(strengths - median)))
    robust_cutoff = median + COUPLING_ROBUST_Z * 1.4826 * mad
    effective_threshold = max(threshold, robust_cutoff)

    adjacency = {position: [] for position in range(length)}
    for left, right, strength in coupling_scores:
        if strength >= effective_threshold:
            adjacency[left].append((right, strength))
            adjacency[right].append((left, strength))
    counts = {position: len(neighbors) for position, neighbors in adjacency.items()}
    supported_neighbors = {
        position: {neighbor for neighbor, _ in neighbors}
        for position, neighbors in adjacency.items()
    }

    used = set()
    groups = []
    for seed_position in sorted(range(length), key=lambda value: counts[value], reverse=True):
        if seed_position in used:
            continue
        if len(groups) >= group_count or counts[seed_position] == 0:
            break
        group = [seed_position]
        used.add(seed_position)
        neighbors = sorted(adjacency[seed_position], key=lambda item: item[1], reverse=True)
        for neighbor, _ in neighbors:
            fully_supported = all(
                neighbor in supported_neighbors[member] for member in group
            )
            if neighbor not in used and fully_supported and len(group) < group_size_max:
                group.append(neighbor)
                used.add(neighbor)
        if len(group) >= group_size_min:
            groups.append(group)
    return groups, effective_threshold


def coupling_support_set(coupling_scores, threshold):
    return {
        (min(left, right), max(left, right))
        for left, right, strength in coupling_scores
        if strength >= threshold
    }


def set_jaccard(left, right):
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def propose_single_guided(
    sequence, mutation_count, per_position_prcs, rng, guided_probability=P_GUIDED_ADAPTIVE
):
    length = len(sequence)
    mutation_count = min(max(1, mutation_count), length)
    worst_count = max(mutation_count, int(PRCS_WORST_FRACTION * length))
    worst_positions = np.argsort(per_position_prcs)[:worst_count].tolist()
    selected = []
    used = set()
    for _ in range(mutation_count):
        guided_available = [position for position in worst_positions if position not in used]
        all_available = [position for position in range(length) if position not in used]
        if rng.random() < guided_probability and guided_available:
            position = rng.choice(guided_available)
        else:
            position = rng.choice(all_available)
        selected.append(position)
        used.add(position)
    replacements = [
        rng.choice([amino_acid for amino_acid in AMINO_ACIDS if amino_acid != sequence[position]])
        for position in selected
    ]
    return mutate_positions(sequence, selected, replacements), selected


def propose_coupled(sequence, groups, rng):
    if not groups:
        return propose_single_guided(sequence, 1, np.zeros(len(sequence)), rng, 0.0)
    positions = list(rng.choice(groups))
    replacements = [
        rng.choice([amino_acid for amino_acid in AMINO_ACIDS if amino_acid != sequence[position]])
        for position in positions
    ]
    return mutate_positions(sequence, positions, replacements), positions


def compute_adaptive_lambda_max(coupling_scores, threshold):
    supported = [strength for _, _, strength in coupling_scores if strength >= threshold]
    if not supported:
        return 0.0, 0.0
    mean_coupling = float(np.mean(supported))
    lambda_max = float(
        np.clip(mean_coupling * LAMBDA_COUPLING_SCALE, LAMBDA_MIN, LAMBDA_CAP)
    )
    return lambda_max, mean_coupling


def compute_lambda(current_prcs, start_prcs, lambda_max):
    gap = 1.0 - start_prcs
    if gap <= 0:
        return lambda_max
    progress = (current_prcs - start_prcs) / gap
    ramp = min(1.0, max(0.0, progress / LAMBDA_RAMP_FRACTION))
    return lambda_max * ramp


def choose_adaptive_refinement(
    trajectory, current_prcs, start_prcs, k_single, p_coupled
):
    gap = max(1e-6, 1.0 - start_prcs)
    progress = max(0.0, min(1.0, (current_prcs - start_prcs) / gap))
    plateau = False
    if len(trajectory) >= ADAPTIVE_WINDOW * 2:
        earlier_best = max(row["prcs"] for row in trajectory[:-ADAPTIVE_WINDOW])
        recent_best = max(row["prcs"] for row in trajectory[-ADAPTIVE_WINDOW:])
        plateau = recent_best - earlier_best < ADAPTIVE_PRCS_EPSILON

    stage = 0
    if progress >= ADAPTIVE_STAGE1_THRESHOLD:
        stage = 1
    if progress >= ADAPTIVE_STAGE2_THRESHOLD:
        stage = 2
    if progress >= ADAPTIVE_STAGE3_THRESHOLD:
        stage = 3
    if plateau and progress >= ADAPTIVE_PLATEAU_THRESHOLD:
        stage = min(3, max(1, stage + 1))

    if stage == 3:
        return (
            min(k_single, ADAPTIVE_STAGE3_K_CAP),
            min(p_coupled, ADAPTIVE_STAGE3_COUPLED_CAP),
            stage,
            progress,
            plateau,
        )
    if stage == 2:
        return (
            min(k_single, ADAPTIVE_STAGE2_K_CAP),
            min(p_coupled, ADAPTIVE_STAGE2_COUPLED_CAP),
            stage,
            progress,
            plateau,
        )
    if stage == 1:
        return (
            min(k_single, ADAPTIVE_STAGE1_K_CAP),
            min(p_coupled, ADAPTIVE_STAGE1_COUPLED_CAP),
            stage,
            progress,
            plateau,
        )
    return k_single, p_coupled, stage, progress, plateau


def save_checkpoint(path: Path, state: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(state, handle)
    temporary.replace(path)


def load_checkpoint(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)
