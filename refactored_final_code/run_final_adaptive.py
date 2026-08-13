#!/usr/bin/env python3
"""Run only the final corrected adaptive independent-walker algorithm."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

import pandas as pd
import torch

import adaptive_walk
import corrected_adaptive_engine as core


def load_artifact(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Final corrected adaptive search with independent walkers"
    )
    parser.add_argument("--protein", required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--walkers", type=int, default=4)
    parser.add_argument("--total-steps", type=int, default=3000)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def validate_artifact(artifact: dict, protein: str) -> None:
    required = {
        "protein",
        "length",
        "target_embedding",
        "target_sha256",
        "model",
        "repr_layer",
    }
    missing = required - set(artifact)
    if missing:
        raise ValueError(f"Target artifact is missing fields: {sorted(missing)}")
    if artifact["protein"] != protein:
        raise ValueError("Target artifact protein does not match --protein")
    if artifact["model"] != core.ESM2_MODEL or artifact["repr_layer"] != core.REPR_LAYER:
        raise ValueError("Target artifact was built with a different ESM model")
    if tuple(artifact["target_embedding"].shape)[:1] != (int(artifact["length"]),):
        raise ValueError("Target artifact embedding shape is inconsistent")


def main() -> None:
    args = parse_args()
    if args.walkers < 1:
        raise ValueError("--walkers must be positive")
    if args.total_steps < 1:
        raise ValueError("--total-steps must be positive")

    artifact_path = args.artifact_dir / f"{args.protein}.target_embedding.pt"
    artifact = load_artifact(artifact_path)
    validate_artifact(artifact, args.protein)
    length = int(artifact["length"])
    config = core.WalkConfig.for_sequence(length, args.total_steps)

    args.outdir.mkdir(parents=True, exist_ok=True)
    trajectory_dir = args.outdir / "walker_trajectories"
    sequence_dir = args.outdir / "final_sequences"
    checkpoint_dir = args.outdir / "checkpoints"
    trajectory_dir.mkdir(exist_ok=True)
    sequence_dir.mkdir(exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)

    initial_rows = []
    starts = []
    for walker_id in range(args.walkers):
        sequence_seed = args.seed + walker_id * 1009
        sequence_rng = random.Random(sequence_seed)
        sequence = core.random_sequence(length, sequence_rng)
        starts.append(sequence)
        initial_rows.append(
            {
                "walker": walker_id,
                "sequence_seed": sequence_seed,
                "sequence_sha256": hashlib.sha256(
                    sequence.encode("ascii")
                ).hexdigest(),
                "sequence": sequence,
            }
        )

    initial_path = args.outdir / "initial_walkers.csv"
    initial_frame = pd.DataFrame(initial_rows)
    if initial_path.exists():
        existing = pd.read_csv(initial_path)
        if not existing.equals(initial_frame):
            raise RuntimeError("Existing initial walker manifest does not match this run")
    else:
        initial_frame.to_csv(initial_path, index=False)

    print(f"Loading target artifact: {artifact_path}")
    model, alphabet, batch_converter = core.load_esm2(args.device)
    target_embedding = artifact["target_embedding"].to(args.device)
    summaries = []

    for walker_id, start_sequence in enumerate(starts):
        walk_seed = args.seed + walker_id * 1009 + 17
        result = adaptive_walk.run_walk(
            protein_name=args.protein,
            walker_id=walker_id,
            start_sequence=start_sequence,
            target_embedding=target_embedding,
            model=model,
            alphabet=alphabet,
            batch_converter=batch_converter,
            device=args.device,
            rng=random.Random(walk_seed),
            config=config,
            checkpoint_path=checkpoint_dir / f"walker_{walker_id:02d}.pkl",
        )
        pd.DataFrame(result["trajectory"]).to_csv(
            trajectory_dir / f"walker_{walker_id:02d}.csv", index=False
        )
        core.write_sequence_fasta(
            sequence_dir / f"walker_{walker_id:02d}_final.fasta",
            f"{args.protein}_seed{args.seed}_walker{walker_id}",
            result["final_sequence"],
        )
        core.write_sequence_fasta(
            sequence_dir / f"walker_{walker_id:02d}_best.fasta",
            f"{args.protein}_seed{args.seed}_walker{walker_id}_best",
            result["best_sequence"],
        )
        summaries.append(
            {
                "walker": walker_id,
                "walk_seed": walk_seed,
                "start_prcs": result["start_prcs"],
                "final_prcs": result["final_prcs"],
                "best_prcs": result["best_prcs"],
                "final_combined": result["final_combined"],
                "best_combined": result["best_combined"],
                "acceptance_rate_single": result["acceptance_rate_single"],
                "acceptance_rate_coupled": result["acceptance_rate_coupled"],
            }
        )
        pd.DataFrame(summaries).to_csv(
            args.outdir / "final_walker_summary.csv", index=False
        )

    metadata = {
        "algorithm": "final_corrected_adaptive_independent",
        "protein": args.protein,
        "seed": args.seed,
        "walkers": args.walkers,
        "total_steps_per_walker": args.total_steps,
        "proposals_per_step": core.PROPOSALS_PER_STEP,
        "total_candidate_evaluations": (
            args.walkers * args.total_steps * core.PROPOSALS_PER_STEP
        ),
        "target_artifact": artifact_path.name,
        "target_length": length,
        "target_model": artifact["model"],
        "target_sha256": artifact["target_sha256"],
        "live_target_sequence_access": False,
        "config": config.as_dict(),
    }
    (args.outdir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(f"Completed final corrected adaptive run: {args.outdir}")


if __name__ == "__main__":
    main()
