#!/usr/bin/env python3
"""Run the pre-correction REVO sequence-search controller.

The script keeps several evolving protein sequences, called walkers. Each walker
is advanced by the adaptive ESM2 mutation engine for a short segment. REVO then
compares the walkers, merges two similar low-impact walkers, and clones a more
useful distinct walker into the freed slot. Statistical weights are moved during
that operation so the total weight remains one.

This is the readable REVO-only copy of the historical implementation. Comments
explain the existing behavior; they do not include the later audit corrections.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from revo_core import (
    combined_walker_distances,
    execute_revo_operation,
    select_revo_operation,
    update_distance_scale,
    variation,
)


def load_engine(path: Path):
    """Load the adaptive mutation engine from the exact file supplied by the run."""
    spec = importlib.util.spec_from_file_location("universal_ignition_engine", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import engine from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args():
    """Read the protein, target artifact, compute budget, and output settings."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=("revo",), default="revo")
    parser.add_argument("--protein", required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--engine-path", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--walkers", type=int, default=4)
    parser.add_argument("--total-steps", type=int, default=3000)
    parser.add_argument("--segment-steps", type=int, default=20)
    parser.add_argument("--max-resample-ops", type=int, default=6)
    parser.add_argument("--proposals-per-step", type=int, default=40)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-ancestor-fraction", type=float, default=0.50)
    return parser.parse_args()


def load_torch_artifact(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def atomic_pickle(path: Path, payload) -> None:
    """Write a checkpoint through a temporary file to avoid half-written files."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle)
    temporary.replace(path)


def checkpoint_path(outdir: Path, protein: str, walker: int) -> Path:
    return outdir / "walkers" / f"walker_{walker:02d}" / protein / "checkpoint_run0.pkl"


def load_walker_states(outdir: Path, protein: str, walkers: int) -> list[dict]:
    states = []
    for walker in range(walkers):
        path = checkpoint_path(outdir, protein, walker)
        if not path.exists():
            raise FileNotFoundError(f"Missing walker checkpoint: {path}")
        with path.open("rb") as handle:
            states.append(pickle.load(handle))
    return states


def history_features(state: dict, window: int = 50) -> np.ndarray:
    """Summarize how a walker behaved recently for the REVO distance calculation."""
    trajectory = state.get("trajectory", [])
    recent = trajectory[-window:]
    if recent:
        scores = np.asarray([row["prcs"] for row in recent], dtype=float)
        x = np.arange(len(scores), dtype=float)
        slope = float(np.polyfit(x, scores, 1)[0]) if len(scores) > 1 else 0.0
        accepted = float(np.mean([bool(row.get("accepted", False)) for row in recent]))
        coupled = float(
            np.mean([bool(row.get("candidate_coupled", False)) for row in recent])
        )
        diagnostic = float(recent[-1].get("diagnostic_score", scores[-1]))
        stage = float(recent[-1].get("adaptive_stage", 0.0))
    else:
        slope = accepted = coupled = stage = 0.0
        diagnostic = float(state["current_prcs"])
    return np.asarray(
        [
            state["current_prcs"],
            diagnostic,
            slope,
            accepted,
            coupled,
            stage,
            state["temperature"],
            state["lambda_max"],
        ],
        dtype=float,
    )


def diversify_clone_rng(state: dict, controller_rng: random.Random) -> dict:
    """Give a cloned walker a fresh random stream so both copies can diverge."""
    cloned = copy.deepcopy(state)
    branch_rng = random.Random(controller_rng.getrandbits(63))
    cloned["rng_state"] = branch_rng.getstate()
    return cloned


def write_states_after_operation(
    paths: list[Path],
    states: list[dict],
    source_indices: list[int],
    removed: int,
    controller_rng: random.Random,
) -> list[dict]:
    """Apply one clone/merge operation to the walker checkpoint files."""
    originals = [copy.deepcopy(state) for state in states]
    updated = []
    for destination, source in enumerate(source_indices):
        state = copy.deepcopy(originals[source])
        if destination == removed:
            state = diversify_clone_rng(state, controller_rng)
        updated.append(state)
    for path, state in zip(paths, updated):
        atomic_pickle(path, state)
    return updated


def save_shadow_state(
    outdir: Path, segment: int, states: list[dict], weights: np.ndarray
) -> None:
    """Keep a safety copy of the walker that has seen the highest PRCS."""
    best_index = max(range(len(states)), key=lambda i: states[i]["best_prcs"])
    archive_dir = outdir / "shadow_archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "segment": segment,
        "walker": best_index,
        "statistical_weight": float(weights[best_index]),
        "state": states[best_index],
    }
    atomic_pickle(archive_dir / f"segment_{segment:04d}.pkl", payload)


def configure_engine(engine, total_steps: int, proposals_per_step: int) -> None:
    """Freeze the inner engine settings used by this REVO experiment."""
    engine.EXPERIMENT_MODE = "adaptive_refine"
    engine.MAX_STEPS_OVERRIDE = total_steps
    engine.PROPOSALS_PER_STEP = proposals_per_step
    engine.SAVE_SEQUENCE_TRAJECTORY = True
    engine.RESCUE_ENABLED = False
    engine.NEUTRAL_REACHABILITY_ENABLED = False
    engine.LIVE_DISTANCE_AUDIT_ENABLED = False


def pooled_embeddings(sequences, engine, model, alphabet, converter, device):
    """Represent each whole live sequence with one mean ESM2 embedding vector."""
    embeddings = engine.get_per_residue_embeddings_batch(
        sequences, model, alphabet, converter, device
    )
    return np.stack([embedding.mean(dim=0).detach().cpu().numpy() for embedding in embeddings])


def main():
    """Alternate adaptive sequence evolution with REVO resampling."""
    args = parse_args()
    if args.total_steps <= 0 or args.segment_steps <= 0:
        raise ValueError("Step counts must be positive")
    if args.total_steps % args.segment_steps != 0:
        raise ValueError("total-steps must be divisible by segment-steps")
    if args.walkers < 2:
        raise ValueError("At least two walkers are required")

    # Load only the precomputed target embedding. The live controller receives an
    # all-X placeholder instead of the target amino-acid letters, so Hamming
    # distance cannot guide the search.
    args.outdir.mkdir(parents=True, exist_ok=True)
    engine = load_engine(args.engine_path.resolve())
    configure_engine(engine, args.total_steps, args.proposals_per_step)
    artifact_path = args.artifact_dir / f"{args.protein}.target_embedding.pt"
    artifact = load_torch_artifact(artifact_path)
    length = int(artifact["length"])
    target_embedding = artifact["target_embedding"].to(args.device)
    placeholder = "X" * length

    model, alphabet, batch_converter = engine.load_esm2(args.device)
    cfg_template = engine.get_protein_config(length)
    cfg_template["cooling_rate"] = (0.005 / engine.INITIAL_TEMP) ** (
        1.0 / args.total_steps
    )

    # Start every walker from a different reproducible random sequence. The seed
    # spacing keeps their random streams separate while allowing exact reruns.
    state_path = args.outdir / "outer_controller_state.pkl"
    controller_rng = random.Random(args.seed + 900_001)
    initial_sequences = []
    for walker in range(args.walkers):
        sequence_rng = random.Random(args.seed + walker * 1009)
        initial_sequences.append(engine.random_sequence(length, sequence_rng))

    initial_manifest = pd.DataFrame(
        [
            {
                "walker": walker,
                "sequence_seed": args.seed + walker * 1009,
                "sequence_sha256": hashlib.sha256(sequence.encode("ascii")).hexdigest(),
                "sequence": sequence,
            }
            for walker, sequence in enumerate(initial_sequences)
        ]
    )
    initial_manifest_path = args.outdir / "initial_walkers.csv"
    if initial_manifest_path.exists():
        existing_manifest = pd.read_csv(initial_manifest_path)
        if not existing_manifest.equals(initial_manifest):
            raise RuntimeError("Existing initial walker manifest does not match this run")
    else:
        initial_manifest.to_csv(initial_manifest_path, index=False)

    # Continue from the outer-controller checkpoint when a previous run exists.
    # Otherwise assign equal statistical weight to all starting walkers.
    if state_path.exists():
        with state_path.open("rb") as handle:
            outer_state = pickle.load(handle)
        weights = np.asarray(outer_state["weights"], dtype=float)
        ancestors = list(outer_state["ancestors"])
        distance_scale = outer_state["distance_scale"]
        controller_rng.setstate(outer_state["controller_rng_state"])
        start_segment = int(outer_state["completed_segment"]) + 1
        print(f"Resuming outer controller at segment {start_segment}")
    else:
        existing = list((args.outdir / "walkers").glob("**/checkpoint_run0.pkl"))
        if existing:
            raise RuntimeError(
                "Walker checkpoints exist without outer_controller_state.pkl; "
                "use a clean output directory"
            )
        weights = np.full(args.walkers, 1.0 / args.walkers, dtype=float)
        ancestors = [f"ancestor_{walker:02d}" for walker in range(args.walkers)]
        distance_scale = None
        start_segment = 0

    segments = args.total_steps // args.segment_steps
    segment_rows = []
    event_rows = []
    metrics_path = args.outdir / "segment_metrics.csv"
    events_path = args.outdir / "resampling_events.csv"
    if metrics_path.exists():
        segment_rows = pd.read_csv(metrics_path).to_dict("records")
    if events_path.exists() and events_path.stat().st_size > 1:
        event_rows = pd.read_csv(events_path).to_dict("records")

    # One outer segment has two stages:
    #   1. advance every walker with the adaptive ESM2 mutation engine;
    #   2. compare the resulting walkers and optionally perform REVO resampling.
    for segment in range(start_segment, segments):
        cumulative_steps = (segment + 1) * args.segment_steps
        print(
            f"\n=== {args.strategy} {args.protein} seed={args.seed} "
            f"segment={segment + 1}/{segments} steps={cumulative_steps} ==="
        )
        # Each walker evolves for the same number of additional mutation steps.
        for walker in range(args.walkers):
            walker_dir = args.outdir / "walkers" / f"walker_{walker:02d}"
            walker_dir.mkdir(parents=True, exist_ok=True)
            # The frozen engine writes to <checkpoint_dir>/<protein>/... and
            # intentionally does not create that protein-level directory.
            (walker_dir / args.protein).mkdir(parents=True, exist_ok=True)
            cfg = dict(cfg_template)
            cfg["max_steps"] = cumulative_steps
            walker_rng = random.Random(args.seed + walker * 1009 + 17)
            engine.run_walk(
                placeholder,
                args.protein,
                0,
                cfg,
                target_embedding,
                model,
                alphabet,
                batch_converter,
                args.device,
                walker_rng,
                checkpoint_dir=walker_dir,
                start_sequence=initial_sequences[walker],
            )

        # REVO measures separation in three complementary ways: sequence
        # mismatch, pooled ESM2 embedding distance, and recent search behavior.
        states = load_walker_states(args.outdir, args.protein, args.walkers)
        sequences = [state["current_seq"] for state in states]
        histories = np.stack([history_features(state) for state in states])
        pooled = pooled_embeddings(
            sequences, engine, model, alphabet, batch_converter, args.device
        )
        distances, components = combined_walker_distances(sequences, pooled, histories)
        distance_scale = update_distance_scale(distances, distance_scale)
        weight_floor = 1e-4 / args.walkers
        variation_before = variation(weights, distances, distance_scale, 4.0, weight_floor)
        ess_before = 1.0 / float(np.sum(weights * weights))
        operations = 0

        # Resampling tries to increase ensemble variation. It merges two nearby
        # walkers and splits the selected clone's weight between two copies. The
        # final segment is left untouched so its recorded states are the actual
        # endpoints of the mutation run.
        if segment + 1 < segments:
            excluded_clones = set()
            max_ops = min(args.max_resample_ops, max(1, args.walkers // 2))
            if ess_before < args.walkers / 3.0:
                max_ops = min(max_ops, 1)
            paths = [checkpoint_path(args.outdir, args.protein, i) for i in range(args.walkers)]

            for operation_index in range(max_ops):
                operation = select_revo_operation(
                    weights,
                    distances,
                    ancestors,
                    distance_scale,
                    alpha=4.0,
                    weight_floor=weight_floor,
                    max_ancestor_fraction=args.max_ancestor_fraction,
                    excluded_clones=excluded_clones,
                )
                if operation is None:
                    break
                # The merge survivor is drawn in proportion to the two walkers'
                # weights. The freed slot is replaced with a clone whose random
                # state was diversified above.
                old_weights = weights.copy()
                (
                    weights,
                    distances,
                    ancestors,
                    source_indices,
                    survivor,
                    removed,
                ) = execute_revo_operation(
                    weights, distances, ancestors, operation, controller_rng
                )
                states = write_states_after_operation(
                    paths, states, source_indices, removed, controller_rng
                )
                sequences = [sequences[source] for source in source_indices]
                pooled = pooled[source_indices]
                histories = histories[source_indices]
                components = {
                    name: matrix[np.ix_(source_indices, source_indices)]
                    for name, matrix in components.items()
                }
                excluded_clones.update({operation.clone, removed})
                operations += 1
                event_rows.append(
                    {
                        "segment": segment,
                        "operation": operation_index,
                        "clone_slot": operation.clone,
                        "merge_a": operation.merge_a,
                        "merge_b": operation.merge_b,
                        "survivor": survivor,
                        "replaced_slot": removed,
                        "expected_variation_gain": operation.expected_gain,
                        "weight_sum_before": float(old_weights.sum()),
                        "weight_sum_after": float(weights.sum()),
                    }
                )
                if not np.isclose(weights.sum(), 1.0, atol=1e-10):
                    raise RuntimeError("REVO statistical weights no longer sum to one")

        # Record enough information to reconstruct the search and verify that
        # statistical weights remain normalized after every outer segment.
        variation_after = variation(weights, distances, distance_scale, 4.0, weight_floor)
        ess_after = 1.0 / float(np.sum(weights * weights))
        current_scores = np.asarray([state["current_prcs"] for state in states])
        best_scores = np.asarray([state["best_prcs"] for state in states])
        segment_rows.append(
            {
                "strategy": args.strategy,
                "protein": args.protein,
                "seed": args.seed,
                "segment": segment,
                "evolution_steps_per_walker": cumulative_steps,
                "total_candidate_evaluations": (
                    cumulative_steps * args.proposals_per_step * args.walkers
                ),
                "weighted_current_prcs": float(np.dot(weights, current_scores)),
                "best_current_prcs": float(current_scores.max()),
                "best_seen_prcs": float(best_scores.max()),
                "effective_sample_size_before": ess_before,
                "effective_sample_size_after": ess_after,
                "variation_before": variation_before,
                "variation_after": variation_after,
                "distance_scale": distance_scale,
                "mean_sequence_distance": float(
                    components["sequence"][np.triu_indices(args.walkers, 1)].mean()
                ),
                "mean_embedding_distance": float(
                    components["embedding"][np.triu_indices(args.walkers, 1)].mean()
                ),
                "mean_history_distance": float(
                    components["history"][np.triu_indices(args.walkers, 1)].mean()
                ),
                "resampling_operations": operations,
                "weight_sum": float(weights.sum()),
            }
        )
        pd.DataFrame(segment_rows).to_csv(metrics_path, index=False)
        pd.DataFrame(
            event_rows,
            columns=[
                "segment",
                "operation",
                "clone_slot",
                "merge_a",
                "merge_b",
                "survivor",
                "replaced_slot",
                "expected_variation_gain",
                "weight_sum_before",
                "weight_sum_after",
            ],
        ).to_csv(events_path, index=False)
        save_shadow_state(args.outdir, segment, states, weights)
        atomic_pickle(
            state_path,
            {
                "completed_segment": segment,
                "weights": weights,
                "ancestors": ancestors,
                "distance_scale": distance_scale,
                "controller_rng_state": controller_rng.getstate(),
                "strategy": args.strategy,
                "protein": args.protein,
                "seed": args.seed,
            },
        )

    # Export one complete genealogical trajectory and one final FASTA per walker,
    # followed by a compact run summary for analysis.
    final_states = load_walker_states(args.outdir, args.protein, args.walkers)
    trajectory_dir = args.outdir / "walker_trajectories"
    sequence_dir = args.outdir / "final_sequences"
    trajectory_dir.mkdir(exist_ok=True)
    sequence_dir.mkdir(exist_ok=True)
    summaries = []
    for walker, state in enumerate(final_states):
        pd.DataFrame(state["trajectory"]).to_csv(
            trajectory_dir / f"walker_{walker:02d}.csv", index=False
        )
        engine.write_sequence_fasta(
            sequence_dir / f"walker_{walker:02d}_final.fasta",
            f"{args.protein}_{args.strategy}_seed{args.seed}_walker{walker}",
            state["current_seq"],
        )
        summaries.append(
            {
                "walker": walker,
                "ancestor": ancestors[walker],
                "statistical_weight": float(weights[walker]),
                "start_prcs": state["start_prcs"],
                "final_prcs": state["current_prcs"],
                "best_prcs": state["best_prcs"],
                "final_diagnostic": state["trajectory"][-1]["diagnostic_score"],
            }
        )
    pd.DataFrame(summaries).to_csv(args.outdir / "final_walker_summary.csv", index=False)
    metadata = {
        "strategy": args.strategy,
        "protein": args.protein,
        "seed": args.seed,
        "walkers": args.walkers,
        "total_steps_per_walker": args.total_steps,
        "segment_steps": args.segment_steps,
        "proposals_per_step": args.proposals_per_step,
        "total_candidate_evaluations": (
            args.total_steps * args.proposals_per_step * args.walkers
        ),
        "target_artifact": artifact_path.name,
        "target_length": length,
        "target_model": artifact["model"],
        "initial_walker_manifest": initial_manifest_path.name,
        "live_identity_audit": False,
    }
    (args.outdir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"\nREVO experiment complete: {args.outdir}")


if __name__ == "__main__":
    main()
