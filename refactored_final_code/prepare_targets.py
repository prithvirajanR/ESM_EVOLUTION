#!/usr/bin/env python3
"""Build target-embedding artifacts before optimization begins."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd
import torch

import corrected_adaptive_engine as core


def read_fasta(path: Path) -> tuple[str, str]:
    lines = path.read_text().strip().splitlines()
    if not lines or not lines[0].startswith(">"):
        raise ValueError(f"Invalid FASTA: {path}")
    name = lines[0][1:].strip().split()[0]
    sequence = "".join(line.strip() for line in lines[1:] if not line.startswith(">"))
    sequence = sequence.replace(" ", "").upper()
    invalid = sorted(set(sequence) - set(core.AMINO_ACIDS))
    if invalid:
        raise ValueError(f"Unsupported amino acids in {path}: {invalid}")
    core.WalkConfig.for_sequence(len(sequence), 1)
    return name, sequence


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument(
        "--proteins",
        required=True,
        help="Comma-separated FASTA basenames without .fasta",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    proteins = [value.strip() for value in args.proteins.split(",") if value.strip()]
    if not proteins:
        raise ValueError("No proteins were provided")
    args.outdir.mkdir(parents=True, exist_ok=True)
    model, alphabet, batch_converter = core.load_esm2(args.device)
    rows = []

    for protein in proteins:
        fasta = args.fasta_dir / f"{protein}.fasta"
        if not fasta.exists():
            raise FileNotFoundError(f"Missing target FASTA: {fasta}")
        _, sequence = read_fasta(fasta)
        embedding = core.get_per_residue_embedding(
            sequence, model, alphabet, batch_converter, args.device
        ).detach().cpu()
        digest = hashlib.sha256(sequence.encode("ascii")).hexdigest()
        artifact = {
            "protein": protein,
            "length": len(sequence),
            "target_embedding": embedding,
            "target_sha256": digest,
            "model": core.ESM2_MODEL,
            "repr_layer": core.REPR_LAYER,
        }
        output = args.outdir / f"{protein}.target_embedding.pt"
        torch.save(artifact, output)
        rows.append(
            {
                "protein": protein,
                "length": len(sequence),
                "target_sha256": digest,
                "model": core.ESM2_MODEL,
                "repr_layer": core.REPR_LAYER,
                "artifact": output.name,
            }
        )
        print(f"Wrote {output} (length={len(sequence)})")

    pd.DataFrame(rows).to_csv(
        args.outdir / "artifact_manifest.tsv", sep="\t", index=False
    )


if __name__ == "__main__":
    main()
