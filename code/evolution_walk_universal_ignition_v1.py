"""
Multi-protein ESM2 evolution walk with full sequence trajectory logging.

This script starts from random protein sequences and tries to move them back
toward each wild-type sequence. The main score is PRCS, and the epistasis term
comes from a simple double-masking check:

    coupling(i, j) = JSD( p(aa_i | seq_{-i}), p(aa_i | seq_{-i,-j}) )

In plain English: if hiding position j changes what ESM2 predicts at position i,
we treat those positions as coupled and sometimes mutate them together.

This copy keeps the adaptive PRCS-only behavior, but also saves the exact
current amino-acid sequence at every step. That makes later AF3 diagnostics
possible without rerunning the walk just to recover milestone sequences.

Rescue-pulse variant:
when an adaptive-refine run plateaus, this copy can deliberately perturb a
small set of high-coupling positions before normal proposal/acceptance resumes.
The underlying adaptive PRCS/epistasis objective is unchanged; only the rescue
pulse size/timing is exposed as an ablation knob.

When this engine is called by the REVO wrapper, each walker runs this mutation
logic for one short segment and saves an exact checkpoint. The wrapper then uses
those checkpoints for its merge-and-clone step before the next segment begins.
"""

import os
import warnings
import sys
from pathlib import Path
import random
import math
import time
import pickle
from typing import List, Tuple
from itertools import combinations

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore", message=".*libomp.*")
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import esm

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon


FASTA_DIR = Path(os.environ.get("FASTA_DIR", Path(__file__).resolve().parent / "target_fastas"))
OUTDIR = os.environ.get("OUTDIR", "evolution_walk_adaptive_refine_prcs_only_save_sequences")
PROTEIN_FILTER = os.environ.get("PROTEIN_FILTER", "").strip()
EXPERIMENT_MODE = os.environ.get("EXPERIMENT_MODE", "adaptive_refine").strip().lower()

ESM2_MODEL = "esm2_t33_650M_UR50D"
REPR_LAYER = 33
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = int(os.environ.get("SEED", "42"))

RUNS_PER_PROTEIN = int(os.environ.get("RUNS_PER_PROTEIN", "1"))

# Big-protein test knobs. These are deliberately separate from the baseline
# replication script, so we can compare the experiment against the original.
BIG_PROTEIN_MIN_LENGTH = int(os.environ.get("BIG_PROTEIN_MIN_LENGTH", "300"))
BIG_STEPS_PER_RESIDUE = int(os.environ.get("BIG_STEPS_PER_RESIDUE", "25"))
BIG_GROUP_SIZE_MIN = int(os.environ.get("BIG_GROUP_SIZE_MIN", "8"))
BIG_GROUP_SIZE_MAX = int(os.environ.get("BIG_GROUP_SIZE_MAX", "14"))
BIG_HEATING_STEPS = int(os.environ.get("BIG_HEATING_STEPS", "1000"))
BIG_HEATING_K_MIN = int(os.environ.get("BIG_HEATING_K_MIN", "5"))
BIG_HEATING_K_MAX = int(os.environ.get("BIG_HEATING_K_MAX", "10"))
BIG_HEATING_PROBABILITY = float(os.environ.get("BIG_HEATING_PROBABILITY", "0.7"))
ADAPTIVE_WINDOW = int(os.environ.get("ADAPTIVE_WINDOW", "400"))
ADAPTIVE_PRCS_EPS = float(os.environ.get("ADAPTIVE_PRCS_EPS", "0.005"))
ADAPTIVE_STAGE1_THRESH = float(os.environ.get("ADAPTIVE_STAGE1_THRESH", "0.50"))
ADAPTIVE_STAGE2_THRESH = float(os.environ.get("ADAPTIVE_STAGE2_THRESH", "0.60"))
ADAPTIVE_STAGE3_THRESH = float(os.environ.get("ADAPTIVE_STAGE3_THRESH", "0.75"))
ADAPTIVE_PLATEAU_THRESH = float(os.environ.get("ADAPTIVE_PLATEAU_THRESH", "0.35"))
SAVE_SEQUENCE_TRAJECTORY = os.environ.get("SAVE_SEQUENCE_TRAJECTORY", "1") == "1"
MAX_STEPS_OVERRIDE = int(os.environ.get("MAX_STEPS_OVERRIDE", "0"))
# Live controller runs can disable sequence-identity diagnostics completely.
# The target sequence may then be replaced by a same-length placeholder while
# the optimizer continues to use only the precomputed target embedding.
LIVE_DISTANCE_AUDIT_ENABLED = os.environ.get("LIVE_DISTANCE_AUDIT_ENABLED", "1") == "1"

RESCUE_ENABLED = os.environ.get("RESCUE_ENABLED", "0") == "1"
RESCUE_PULSE_SIZE = int(os.environ.get("RESCUE_PULSE_SIZE", "10"))
RESCUE_MIN_STEP = int(os.environ.get("RESCUE_MIN_STEP", "500"))
RESCUE_COOLDOWN = int(os.environ.get("RESCUE_COOLDOWN", "500"))
RESCUE_TOP_PAIR_MULTIPLIER = int(os.environ.get("RESCUE_TOP_PAIR_MULTIPLIER", "4"))
RESCUE_FORCE_ACCEPT = os.environ.get("RESCUE_FORCE_ACCEPT", "0") == "1"


INITIAL_TEMP = float(os.environ.get("INITIAL_TEMP", "0.3"))
COOLING_RATE = float(os.environ.get("COOLING_RATE", "0.997"))

LAMBDA_MIN = float(os.environ.get("LAMBDA_MIN", "0.1"))
LAMBDA_CAP = float(os.environ.get("LAMBDA_CAP", "0.5"))
LAMBDA_COUPLING_SCALE = float(os.environ.get("LAMBDA_COUPLING_SCALE", "5.0"))
LAMBDA_RAMP_FRAC = float(os.environ.get("LAMBDA_RAMP_FRAC", "0.4"))


PROPOSALS_PER_STEP = int(os.environ.get("PROPOSALS_PER_STEP", "40"))

COUPLING_RECOMPUTE_EVERY = int(os.environ.get("COUPLING_RECOMPUTE_EVERY", "150"))
COUPLING_ENABLED = os.environ.get("COUPLING_ENABLED", "1") == "1"

COUPLING_THRESHOLD = float(os.environ.get("COUPLING_THRESHOLD", "0.01"))
COUPLING_ROBUST_Z = float(os.environ.get("COUPLING_ROBUST_Z", "2.5"))
COUPLING_GROUP_MIN = int(os.environ.get("COUPLING_GROUP_MIN", "2"))
COUPLING_GROUP_MAX = int(os.environ.get("COUPLING_GROUP_MAX", "4"))

COUPLING_MIN_FRAC = float(os.environ.get("COUPLING_MIN_FRAC", "0.20"))
P_COUPLED_OVERRIDE = float(os.environ.get("P_COUPLED_OVERRIDE", "-1"))
K_SINGLE_OVERRIDE = int(os.environ.get("K_SINGLE_OVERRIDE", "0"))

ADAPTIVE_STAGE1_K_CAP = int(os.environ.get("ADAPTIVE_STAGE1_K_CAP", "4"))
ADAPTIVE_STAGE2_K_CAP = int(os.environ.get("ADAPTIVE_STAGE2_K_CAP", "2"))
ADAPTIVE_STAGE3_K_CAP = int(os.environ.get("ADAPTIVE_STAGE3_K_CAP", "1"))
ADAPTIVE_STAGE1_P_COUPLED_CAP = float(os.environ.get("ADAPTIVE_STAGE1_P_COUPLED_CAP", "0.40"))
ADAPTIVE_STAGE2_P_COUPLED_CAP = float(os.environ.get("ADAPTIVE_STAGE2_P_COUPLED_CAP", "0.25"))
ADAPTIVE_STAGE3_P_COUPLED_CAP = float(os.environ.get("ADAPTIVE_STAGE3_P_COUPLED_CAP", "0.10"))
P_GUIDED_ADAPTIVE = float(os.environ.get("P_GUIDED_ADAPTIVE", "0.85"))
P_GUIDED_BASELINE = float(os.environ.get("P_GUIDED_BASELINE", "0.70"))
P_GUIDED_HEATING = float(os.environ.get("P_GUIDED_HEATING", "0.50"))
PRCS_WORST_FRAC = float(os.environ.get("PRCS_WORST_FRAC", "0.30"))

NEUTRAL_REACHABILITY_ENABLED = os.environ.get("NEUTRAL_REACHABILITY_ENABLED", "0") == "1"
NEUTRAL_RERANK_TOP_N = int(os.environ.get("NEUTRAL_RERANK_TOP_N", "5"))
NEUTRAL_SELF_SUPPORT_WEIGHT = float(os.environ.get("NEUTRAL_SELF_SUPPORT_WEIGHT", "1.0"))
NEUTRAL_REGION_BALANCE_WEIGHT = float(os.environ.get("NEUTRAL_REGION_BALANCE_WEIGHT", "0.5"))
NEUTRAL_DISRUPTION_WEIGHT = float(os.environ.get("NEUTRAL_DISRUPTION_WEIGHT", "0.10"))
NEUTRAL_BETA = float(os.environ.get("NEUTRAL_BETA", "0.12"))
NEUTRAL_REGION_COUNT = int(os.environ.get("NEUTRAL_REGION_COUNT", "6"))


def get_coupling_sample_pairs(L):
    """Pick a practical number of residue pairs to test for coupling."""
    return min(4000, max(1000, L * 6))


PLL_DIAGNOSTIC_EVERY = 500
LOG_EVERY = 50
CHECKPOINT_EVERY = 200

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


def plot_walk(trajectories, protein_name, L, outdir):
    """Make the detailed per-protein plot for checking how each run behaved."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"{protein_name} (L={L}) — Evolution Walk v10b-v3",
        fontsize=14,
        fontweight="bold",
    )

    colors = plt.cm.Set1(np.linspace(0, 1, len(trajectories)))

    for idx, (traj, color) in enumerate(zip(trajectories, colors)):
        df = pd.DataFrame(traj)
        steps = df["step"]

        axes[0, 0].plot(steps, df["prcs"], color=color, alpha=0.8, label=f"Run {idx}")
        axes[0, 0].set_ylabel("PRCS")
        axes[0, 0].set_title("PRCS Trajectory")
        axes[0, 0].legend(fontsize=8)

        axes[0, 1].plot(steps, df["hamming"], color=color, alpha=0.8)
        axes[0, 1].plot(steps, df["best_hamming"], color=color, alpha=0.4, linestyle="--")
        axes[0, 1].set_ylabel("Hamming to WT")
        axes[0, 1].set_title("Hamming Distance (solid=current, dash=best)")

        axes[0, 2].plot(steps, df["combined"], color=color, alpha=0.8)
        axes[0, 2].set_ylabel("Combined Score")
        axes[0, 2].set_title("Combined (PRCS + λ·Epi)")

        axes[1, 0].plot(steps, df["lambda"], color=color, alpha=0.8)
        if "lambda_max" in df.columns:
            axes[1, 0].plot(steps, df["lambda_max"], color=color, alpha=0.3, linestyle="--")
        axes[1, 0].set_ylabel("λ")
        axes[1, 0].set_title("Lambda (solid=current, dash=λ_max)")
        axes[1, 0].set_xlabel("Step")

        axes[1, 1].plot(steps, df["epi"], color=color, alpha=0.8)
        axes[1, 1].set_ylabel("Epistatic Score")
        axes[1, 1].set_title("Epistatic Component")
        axes[1, 1].set_xlabel("Step")

        axes[1, 2].plot(steps, df["temperature"], color=color, alpha=0.8)
        axes[1, 2].set_ylabel("Temperature")
        axes[1, 2].set_title("Annealing Schedule")
        axes[1, 2].set_xlabel("Step")
        axes[1, 2].set_yscale("log")

    plt.tight_layout()
    path = outdir / "plots_walk.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")
    return path


def plot_proposals(trajectories, protein_name, L, outdir):
    """Plot whether proposed mutations are getting accepted or rejected."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"{protein_name} — Proposal Analysis", fontsize=13)

    colors = plt.cm.Set1(np.linspace(0, 1, len(trajectories)))

    for idx, (traj, color) in enumerate(zip(trajectories, colors)):
        df = pd.DataFrame(traj)

        window = 50
        if len(df) > window:
            rolling_acc = df["accepted"].rolling(window).mean() * 100
            axes[0].plot(df["step"], rolling_acc, color=color, alpha=0.8, label=f"Run {idx}")
        axes[0].set_ylabel("Acceptance Rate (%)")
        axes[0].set_title(f"Rolling Acceptance (window={window})")
        axes[0].legend(fontsize=8)

        deltas = df["delta"].dropna()
        axes[1].hist(
            deltas[deltas > 0],
            bins=50,
            alpha=0.4,
            color="green",
            label="Uphill" if idx == 0 else None,
        )
        axes[1].hist(
            deltas[deltas <= 0],
            bins=50,
            alpha=0.4,
            color="red",
            label="Downhill" if idx == 0 else None,
        )
        axes[1].set_title("Score Delta Distribution")
        axes[1].legend(fontsize=8)

    all_coupled_acc = []
    all_single_acc = []
    for traj in trajectories:
        df = pd.DataFrame(traj)
        coupled = df[df["coupled"]]
        single = df[~df["coupled"]]
        if len(coupled) > 0:
            all_coupled_acc.append(coupled["accepted"].mean() * 100)
        if len(single) > 0:
            all_single_acc.append(single["accepted"].mean() * 100)

    labels = ["Single", "Coupled"]
    means = [
        np.mean(all_single_acc) if all_single_acc else 0,
        np.mean(all_coupled_acc) if all_coupled_acc else 0,
    ]
    bars = axes[2].bar(labels, means, color=["steelblue", "coral"])
    axes[2].set_ylabel("Acceptance Rate (%)")
    axes[2].set_title("Single vs Coupled Proposals")
    for bar, val in zip(bars, means):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}%",
            ha="center",
            fontsize=10,
        )

    plt.tight_layout()
    path = outdir / "plots_proposals.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_cross_protein(all_results, outdir):
    """Make the final comparison plot across all proteins."""
    if not all_results:
        return

    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    fig.suptitle("Cross-Protein Comparison — v10b-v3", fontsize=14, fontweight="bold")

    proteins = []
    lengths = []
    best_hammings = []
    pct_recovered = []
    best_prcs_vals = []
    coupled_advantages = []

    for res in all_results:
        proteins.append(res["protein"])
        lengths.append(res["length"])
        best_hammings.append(res["best_hamming"])
        pct_recovered.append(100 * (1 - res["best_hamming"] / res["length"]))
        best_prcs_vals.append(res["best_prcs"])
        sa = res.get("accept_rate_single", 0)
        ca = res.get("accept_rate_coupled", 0)
        coupled_advantages.append(ca / sa if sa > 0 else 0)

    x = range(len(proteins))

    axes[0].bar(x, best_hammings, color="steelblue")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(proteins, rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("Best Hamming")
    axes[0].set_title("Best Hamming Distance")

    axes[1].bar(x, pct_recovered, color="coral")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(proteins, rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("% Residues Recovered")
    axes[1].set_title("Sequence Recovery")
    axes[1].axhline(y=82, color="green", linestyle="--", alpha=0.5, label="Insulin v10b (82%)")
    axes[1].legend(fontsize=8)

    axes[2].bar(x, best_prcs_vals, color="mediumpurple")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(proteins, rotation=45, ha="right", fontsize=8)
    axes[2].set_ylabel("Best PRCS")
    axes[2].set_title("Best PRCS Achieved")

    axes[3].scatter(lengths, pct_recovered, s=100, c="steelblue", edgecolors="black")
    for i, name in enumerate(proteins):
        axes[3].annotate(
            name, (lengths[i], pct_recovered[i]), fontsize=7, ha="center", va="bottom"
        )
    axes[3].set_xlabel("Protein Length")
    axes[3].set_ylabel("% Residues Recovered")
    axes[3].set_title("Length vs Recovery")

    plt.tight_layout()
    path = Path(outdir) / "plots_cross_protein.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def display_name(name):
    """Clean up FASTA filenames so the legend looks like the report figure."""
    labels = {
        "cycs": "CYCS",
        "gapdh": "GAPDH",
        "growth_hormone": "Growth_hormone",
        "human_insulin": "human_insulin",
        "ighg1": "IGHG1",
        "myc_proto_oncogene": "MYC_proto_oncogene",
        "pro_glucagon_preprotein": "pro_glucagon_preprotein",
        "tp53_isoform_a": "TP53_isoform_a",
    }
    return labels.get(name, name)


def best_trajectory(trajectories):
    """Use the best embedding-scored run for the report trajectory."""
    if not trajectories:
        return None

    def key(traj):
        """Never use Hamming to choose which run is presented as best."""
        df = pd.DataFrame(traj)
        if "diagnostic_score" in df.columns:
            return float(df["diagnostic_score"].max())
        return float(df["prcs"].max())

    return max(trajectories, key=key)


def plot_recovery_over_steps(all_protein_results, outdir):
    """Recreate the two-panel PRCS/Hamming plot from the report."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    for protein_name, result in all_protein_results.items():
        traj = best_trajectory(result.get("trajectories", []))
        if not traj:
            continue

        df = pd.DataFrame(traj).sort_values("step")
        label = display_name(protein_name)

        axes[0].plot(df["step"], df["prcs"], label=label, linewidth=2)
        axes[1].plot(df["step"], df["hamming"], label=label, linewidth=2)

    axes[0].set_title("Protein Sequence Recovery (PRCS Score) over Steps")
    axes[0].set_ylabel("PRCS Score")
    axes[0].grid(True, alpha=0.35)
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)

    axes[1].set_title("Hamming Distance over Steps")
    axes[1].set_xlabel("Simulation Steps")
    axes[1].set_ylabel("Hamming Distance")
    axes[1].grid(True, alpha=0.35)
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)

    plt.tight_layout()
    path = Path(outdir) / "protein_sequence_recovery_over_steps.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")
    return path


def get_protein_config(L: int) -> dict:
    """Set walk parameters based on protein length."""
    cfg = {}
    valid_modes = {
        "baseline",
        "more_steps",
        "larger_groups",
        "heating",
        "combined",
        "adaptive_refine",
    }
    if EXPERIMENT_MODE not in valid_modes:
        raise ValueError(
            f"Unknown EXPERIMENT_MODE={EXPERIMENT_MODE!r}; "
            f"use one of {sorted(valid_modes)}"
        )

    cfg["k_single"] = max(1, round(L / 50))
    if K_SINGLE_OVERRIDE > 0:
        cfg["k_single"] = min(L, K_SINGLE_OVERRIDE)

    # Small groups keep a coupled proposal interpretable. Larger groups made a
    # weak or noisy edge capable of randomizing 5-11 unrelated residues.
    cfg["group_size_min"] = max(2, min(COUPLING_GROUP_MIN, L))
    cfg["group_size_max"] = max(cfg["group_size_min"], min(COUPLING_GROUP_MAX, L))
    cfg["is_big_protein_experiment"] = L >= BIG_PROTEIN_MIN_LENGTH
    cfg["experiment_mode"] = EXPERIMENT_MODE
    cfg["adaptive_refine"] = EXPERIMENT_MODE == "adaptive_refine"

    if cfg["is_big_protein_experiment"] and EXPERIMENT_MODE in {"larger_groups", "combined"}:
        cfg["group_size_min"] = max(cfg["group_size_min"], BIG_GROUP_SIZE_MIN)
        cfg["group_size_max"] = max(cfg["group_size_max"], BIG_GROUP_SIZE_MAX)

    avg_group = (cfg["group_size_min"] + cfg["group_size_max"]) / 2
    cfg["n_groups"] = max(15, int(0.7 * L / avg_group))

    cfg["p_coupled"] = min(0.8, 0.3 + L / 1000)
    if P_COUPLED_OVERRIDE >= 0:
        cfg["p_coupled"] = min(1.0, max(0.0, P_COUPLED_OVERRIDE))

    cfg["max_steps"] = max(2000, 10 * L)
    if cfg["is_big_protein_experiment"] and EXPERIMENT_MODE in {
        "more_steps",
        "combined",
        "adaptive_refine",
    }:
        cfg["max_steps"] = max(cfg["max_steps"], BIG_STEPS_PER_RESIDUE * L)

    if MAX_STEPS_OVERRIDE > 0:
        cfg["max_steps"] = MAX_STEPS_OVERRIDE

    cfg["cooling_rate"] = (0.005 / INITIAL_TEMP) ** (1.0 / cfg["max_steps"])

    cfg["prcs_window"] = max(20, min(50, L // 4))

    cfg["heating_steps"] = 0
    cfg["heating_k_min"] = 0
    cfg["heating_k_max"] = 0
    cfg["heating_probability"] = 0.0
    if cfg["is_big_protein_experiment"] and EXPERIMENT_MODE in {
        "heating",
        "combined",
        "adaptive_refine",
    }:
        cfg["heating_steps"] = min(BIG_HEATING_STEPS, cfg["max_steps"] // 4)
        cfg["heating_k_min"] = min(BIG_HEATING_K_MIN, L)
        cfg["heating_k_max"] = min(BIG_HEATING_K_MAX, L)
        cfg["heating_probability"] = BIG_HEATING_PROBABILITY

    return cfg


def read_fasta_one(path: str) -> Tuple[str, str]:
    """Read one FASTA file and clean the sequence into one uppercase string."""
    lines = Path(path).read_text().strip().splitlines()
    name = lines[0].lstrip(">").strip().split()[0]
    seq = "".join(ln.strip() for ln in lines[1:] if not ln.startswith(">"))
    seq = seq.replace(" ", "").upper()
    return name, seq


def write_sequence_fasta(path: Path, name: str, seq: str) -> None:
    """Write one protein sequence as a simple wrapped FASTA file."""
    with open(path, "w") as f:
        f.write(f">{name}\n")
        for i in range(0, len(seq), 60):
            f.write(seq[i : i + 60] + "\n")


def hamming(a: str, b: str) -> int:
    """Count how many aligned positions are different."""
    return sum(x != y for x, y in zip(a, b))


def analysis_hamming(a: str, b: str) -> int:
    """Return an analysis-only sequence distance, or -1 in live-firewall mode."""
    if not LIVE_DISTANCE_AUDIT_ENABLED:
        return -1
    return hamming(a, b)


def mutate_positions(seq: str, positions: List[int], aas: List[str]) -> str:
    """Make a mutated copy of the sequence at the requested positions."""
    s = list(seq)
    for pos, aa in zip(positions, aas):
        s[pos] = aa
    return "".join(s)


def random_sequence(length: int, rng: random.Random) -> str:
    """Make a random amino-acid sequence for the starting point."""
    return "".join(rng.choice(AMINO_ACIDS) for _ in range(length))


def load_esm2(device: str):
    """Load ESM2 once so the rest of the run can reuse it."""
    print("Loading ESM2...")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.eval().to(torch.device(device))
    batch_converter = alphabet.get_batch_converter()
    print(f"  Model loaded on {device}")
    return model, alphabet, batch_converter


@torch.no_grad()
def get_per_residue_embedding(seq, model, alphabet, batch_converter, device):
    """Run one sequence through ESM2 and return residue-level embeddings."""
    data = [("seq", seq)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)
    out = model(tokens, repr_layers=[REPR_LAYER], return_contacts=False)
    return out["representations"][REPR_LAYER][0, 1:-1, :]


@torch.no_grad()
def get_per_residue_embeddings_batch(
    seqs, model, alphabet, batch_converter, device, max_batch=None
):
    """Embed candidate sequences in batches small enough for the GPU/CPU."""
    if max_batch is None:
        L = len(seqs[0]) if seqs else 100
        if L < 150:
            max_batch = 20
        elif L < 300:
            max_batch = 12
        elif L < 500:
            max_batch = 8
        else:
            max_batch = 4

    all_embeddings = []

    for start in range(0, len(seqs), max_batch):
        batch_seqs = seqs[start : start + max_batch]
        data = [(f"s{i}", s) for i, s in enumerate(batch_seqs)]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(device)
        out = model(tokens, repr_layers=[REPR_LAYER], return_contacts=False)
        reps = out["representations"][REPR_LAYER]

        for i in range(len(batch_seqs)):
            all_embeddings.append(reps[i, 1:-1, :].clone())

        del reps, out, tokens

    return all_embeddings


def score_prcs(emb_candidate, emb_wt):
    """Score how close a candidate embedding is to the wild-type embedding."""
    per_pos = F.cosine_similarity(emb_candidate, emb_wt, dim=1)  # (L,)
    return per_pos.mean().item(), per_pos.cpu().numpy()


def score_fixed_diagnostic(per_pos_prcs):
    """Return a fixed diagnostic score for comparing states across policies.

    Mean PRCS alone can hide a badly mismatched region in a long protein. This
    score therefore combines global mean similarity with the lower quartile
    and the weakest local window. It uses target embeddings, never target
    amino-acid identities, and its definition does not change during a run.
    """
    values = np.asarray(per_pos_prcs, dtype=float)
    if values.size == 0:
        return float("-inf"), float("nan"), float("nan")
    lower_quartile = float(np.quantile(values, 0.25))
    window = max(1, min(values.size, max(8, values.size // 12)))
    kernel = np.ones(window, dtype=float) / window
    weakest_window = float(np.convolve(values, kernel, mode="valid").min())
    diagnostic = 0.70 * float(values.mean()) + 0.20 * lower_quartile + 0.10 * weakest_window
    return diagnostic, lower_quartile, weakest_window


def score_epistatic(emb_candidate, emb_wt, groups):
    """Check whether coupled residue groups still look wild-type-like in embedding space."""
    if not groups:
        return 0.0

    all_cosines = []
    for group in groups:
        for i, j in combinations(group, 2):
            diff_cand = emb_candidate[i] - emb_candidate[j]
            diff_wt = emb_wt[i] - emb_wt[j]
            cos = F.cosine_similarity(diff_cand.unsqueeze(0), diff_wt.unsqueeze(0)).item()
            all_cosines.append(cos)

    return np.mean(all_cosines) if all_cosines else 0.0


def compute_combined_score(emb_candidate, emb_wt, groups, lam):
    """Combine the PRCS score with the epistasis bonus used for accepting moves."""
    prcs, per_pos = score_prcs(emb_candidate, emb_wt)
    epi = score_epistatic(emb_candidate, emb_wt, groups) if lam > 0.01 else 0.0
    combined = prcs + lam * epi
    return combined, prcs, epi, per_pos


@torch.no_grad()
def score_true_pll(seq, model, alphabet, batch_converter, device, pos_batch_size=16):
    """Calculate masked PLL for logging only; the walk does not optimize this."""
    data = [("seq", seq)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)
    L = len(seq)

    log_probs = []
    for start in range(0, L, pos_batch_size):
        end = min(start + pos_batch_size, L)
        batch_tokens = tokens.repeat(end - start, 1)
        for idx, pos in enumerate(range(start, end)):
            batch_tokens[idx, pos + 1] = alphabet.mask_idx

        out = model(batch_tokens, repr_layers=[], return_contacts=False)
        logits = out["logits"]

        for idx, pos in enumerate(range(start, end)):
            log_p = torch.log_softmax(logits[idx, pos + 1], dim=-1)
            true_aa = tokens[0, pos + 1].item()
            log_probs.append(log_p[true_aa].item())

    return np.mean(log_probs)


@torch.no_grad()
def score_masked_self_support(
    seq, positions, model, alphabet, batch_converter, device, pos_batch_size=16
):
    """Score the candidate's own residues in its own masked ESM context."""
    positions = sorted(set(int(position) for position in positions))
    if not positions:
        return 0.0
    _, _, tokens = batch_converter([("seq", seq)])
    tokens = tokens.to(device)
    normalized_log_probs = []
    for start in range(0, len(positions), pos_batch_size):
        batch_positions = positions[start : start + pos_batch_size]
        batch_tokens = tokens.repeat(len(batch_positions), 1)
        for index, position in enumerate(batch_positions):
            batch_tokens[index, position + 1] = alphabet.mask_idx
        logits = model(batch_tokens, repr_layers=[], return_contacts=False)["logits"]
        for index, position in enumerate(batch_positions):
            log_probability = torch.log_softmax(logits[index, position + 1], dim=-1)
            current_aa = tokens[0, position + 1].item()
            normalized = (log_probability[current_aa].item() + 12.0) / 12.0
            normalized_log_probs.append(max(0.0, min(1.0, normalized)))
    return float(np.mean(normalized_log_probs))


def regional_balance_score(per_pos_prcs, n_regions=NEUTRAL_REGION_COUNT):
    """Measure the weakest broad target-embedding region."""
    values = np.asarray(per_pos_prcs, dtype=float)
    if values.size == 0:
        return 0.0
    n_regions = max(1, min(int(n_regions), values.size))
    means = [float(chunk.mean()) for chunk in np.array_split(values, n_regions) if len(chunk)]
    return min(means) if means else 0.0


def compute_neutral_reachability_delta(
    candidate_seq,
    current_seq,
    positions,
    candidate_per_pos_prcs,
    current_per_pos_prcs,
    model,
    alphabet,
    batch_converter,
    device,
):
    """Score a clean bridge move without target residue identities or Hamming."""
    positions = sorted(set(int(position) for position in positions))
    if not positions:
        return {
            "reachability_delta": 0.0,
            "self_support_delta": 0.0,
            "region_balance_delta": 0.0,
            "disruption_penalty": 0.0,
        }
    candidate_support = score_masked_self_support(
        candidate_seq, positions, model, alphabet, batch_converter, device
    )
    current_support = score_masked_self_support(
        current_seq, positions, model, alphabet, batch_converter, device
    )
    self_support_delta = candidate_support - current_support
    region_balance_delta = regional_balance_score(
        candidate_per_pos_prcs
    ) - regional_balance_score(current_per_pos_prcs)
    disruption_penalty = len(positions) / max(1, len(current_seq))
    reachability_delta = (
        NEUTRAL_SELF_SUPPORT_WEIGHT * self_support_delta
        + NEUTRAL_REGION_BALANCE_WEIGHT * region_balance_delta
        - NEUTRAL_DISRUPTION_WEIGHT * disruption_penalty
    )
    return {
        "reachability_delta": float(reachability_delta),
        "self_support_delta": float(self_support_delta),
        "region_balance_delta": float(region_balance_delta),
        "disruption_penalty": float(disruption_penalty),
    }


@torch.no_grad()
def compute_single_masked_marginals(seq, model, alphabet, batch_converter, device, positions):
    """Get ESM2's amino-acid probabilities when one position is masked at a time."""
    data = [("seq", seq)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)

    aa_indices = [alphabet.get_idx(aa) for aa in AMINO_ACIDS]
    marginals = {}

    for pos in positions:
        masked = tokens.clone()
        masked[0, pos + 1] = alphabet.mask_idx
        out = model(masked, repr_layers=[], return_contacts=False)
        logits = out["logits"][0, pos + 1]
        probs = torch.softmax(logits, dim=-1)
        marginals[pos] = probs[aa_indices].cpu().numpy()

    return marginals


@torch.no_grad()
def compute_double_masked_marginals(seq, model, alphabet, batch_converter, device, pos_i, pos_j):
    """Get amino-acid probabilities after masking a pair of positions together."""
    data = [("seq", seq)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)

    aa_indices = [alphabet.get_idx(aa) for aa in AMINO_ACIDS]

    masked = tokens.clone()
    masked[0, pos_i + 1] = alphabet.mask_idx
    masked[0, pos_j + 1] = alphabet.mask_idx
    out = model(masked, repr_layers=[], return_contacts=False)

    logits_i = out["logits"][0, pos_i + 1]
    probs_i = torch.softmax(logits_i, dim=-1)

    logits_j = out["logits"][0, pos_j + 1]
    probs_j = torch.softmax(logits_j, dim=-1)

    return (probs_i[aa_indices].cpu().numpy(), probs_j[aa_indices].cpu().numpy())


@torch.no_grad()
def compute_epistatic_coupling_map(
    seq, model, alphabet, batch_converter, device, n_sample_pairs, rng
):
    """Estimate which residue pairs affect each other under double masking."""
    L = len(seq)

    all_pairs = list(combinations(range(L), 2))
    n_pairs = min(n_sample_pairs, len(all_pairs))
    sampled_pairs = rng.sample(all_pairs, n_pairs)

    unique_positions = set()
    for i, j in sampled_pairs:
        unique_positions.add(i)
        unique_positions.add(j)

    single_marginals = compute_single_masked_marginals(
        seq, model, alphabet, batch_converter, device, sorted(unique_positions)
    )

    coupling_scores = []
    for i, j in sampled_pairs:
        double_i, double_j = compute_double_masked_marginals(
            seq, model, alphabet, batch_converter, device, i, j
        )

        jsd_i = jensenshannon(single_marginals[i], double_i)
        jsd_j = jensenshannon(single_marginals[j], double_j)

        coupling = (jsd_i + jsd_j) / 2

        if not np.isnan(coupling):
            coupling_scores.append((i, j, coupling))

    coupling_scores.sort(key=lambda x: x[2], reverse=True)

    return coupling_scores


def build_coupling_groups(
    coupling_scores, L, n_groups, group_size_min, group_size_max, threshold, rng
):
    """Turn strong pair couplings into small groups that can be mutated together."""
    if not coupling_scores:
        return [], threshold

    strengths = np.asarray([s for _, _, s in coupling_scores], dtype=float)
    median = float(np.median(strengths))
    mad = float(np.median(np.abs(strengths - median)))
    robust_cutoff = median + COUPLING_ROBUST_Z * 1.4826 * mad
    effective_threshold = max(threshold, robust_cutoff)

    adjacency = {i: [] for i in range(L)}
    for i, j, strength in coupling_scores:
        if strength >= effective_threshold:
            adjacency[i].append((j, strength))
            adjacency[j].append((i, strength))

    coupling_count = {i: len(neighbors) for i, neighbors in adjacency.items()}
    supported_neighbors = {
        position: {neighbor for neighbor, _ in neighbors}
        for position, neighbors in adjacency.items()
    }

    used = set()
    groups = []
    sorted_positions = sorted(range(L), key=lambda x: coupling_count[x], reverse=True)

    for seed_pos in sorted_positions:
        if seed_pos in used:
            continue
        if len(groups) >= n_groups:
            break
        if coupling_count[seed_pos] == 0:
            break

        group = [seed_pos]
        used.add(seed_pos)

        neighbors = sorted(adjacency[seed_pos], key=lambda x: x[1], reverse=True)
        for nb, _ in neighbors:
            fully_supported = all(nb in supported_neighbors[member] for member in group)
            if nb not in used and fully_supported and len(group) < group_size_max:
                group.append(nb)
                used.add(nb)

        if len(group) >= group_size_min:
            groups.append(group)

    return groups, effective_threshold


def coupling_support_set(coupling_scores, threshold):
    """Return supported undirected edges for stability diagnostics."""
    return {
        (min(i, j), max(i, j))
        for i, j, strength in coupling_scores
        if strength >= threshold
    }


def set_jaccard(left, right):
    """Measure coupling-map stability without using protein annotations."""
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def propose_single(seq, k_mutations, rng):
    """Propose a plain random mutation at k positions."""
    L = len(seq)
    positions = rng.sample(range(L), k_mutations)
    aas = []
    for pos in positions:
        candidates = [aa for aa in AMINO_ACIDS if aa != seq[pos]]
        aas.append(rng.choice(candidates))
    return mutate_positions(seq, positions, aas), positions


def propose_single_guided(seq, k_mutations, per_pos_prcs, rng, p_guided=0.7):
    """Usually mutate weak PRCS positions, with some randomness kept in."""
    L = len(seq)

    n_worst = max(k_mutations, int(PRCS_WORST_FRAC * L))
    worst_indices = np.argsort(per_pos_prcs)[:n_worst].tolist()

    positions = []
    used = set()
    for _ in range(k_mutations):
        if rng.random() < p_guided and worst_indices:
            available = [p for p in worst_indices if p not in used]
            if available:
                pos = rng.choice(available)
            else:
                pos = rng.choice([p for p in range(L) if p not in used])
        else:
            pos = rng.choice([p for p in range(L) if p not in used])
        positions.append(pos)
        used.add(pos)

    aas = []
    for pos in positions:
        candidates = [aa for aa in AMINO_ACIDS if aa != seq[pos]]
        aas.append(rng.choice(candidates))
    return mutate_positions(seq, positions, aas), positions


def propose_coupled(seq, groups, group_size_min, rng):
    """Propose mutating one coupled group in a single move."""
    if not groups:
        return propose_single(seq, 1, rng)

    group = rng.choice(groups)
    positions = list(group)
    aas = []
    for pos in positions:
        candidates = [aa for aa in AMINO_ACIDS if aa != seq[pos]]
        aas.append(rng.choice(candidates))
    return mutate_positions(seq, positions, aas), positions


def select_rescue_positions(coupling_scores, L, pulse_size, rng):
    """Pick high-coupling positions to perturb during a rescue pulse.

    We rank positions by their participation in the strongest current
    double-masking coupling pairs. This keeps the rescue tied to the same
    epistasis signal used elsewhere in the walk, while recomputing maps during
    the run allows different pulses to hit different positions.
    """
    pulse_size = min(max(1, pulse_size), L)
    if not coupling_scores:
        return rng.sample(range(L), pulse_size)

    top_n = max(pulse_size, min(len(coupling_scores), RESCUE_TOP_PAIR_MULTIPLIER * pulse_size))
    top_pairs = sorted(coupling_scores, key=lambda x: x[2], reverse=True)[:top_n]

    position_scores = {i: 0.0 for i in range(L)}
    position_counts = {i: 0 for i in range(L)}
    for i, j, score in top_pairs:
        position_scores[i] += float(score)
        position_scores[j] += float(score)
        position_counts[i] += 1
        position_counts[j] += 1

    ranked = sorted(
        range(L),
        key=lambda p: (position_scores[p], position_counts[p], rng.random()),
        reverse=True,
    )
    selected = ranked[:pulse_size]
    if len(selected) < pulse_size:
        remaining = [p for p in range(L) if p not in selected]
        selected.extend(rng.sample(remaining, pulse_size - len(selected)))
    return selected


def propose_rescue_pulse(seq, coupling_scores, pulse_size, rng):
    """Force-randomize high-coupling positions without using WT residues."""
    positions = select_rescue_positions(coupling_scores, len(seq), pulse_size, rng)
    aas = []
    for pos in positions:
        candidates = [aa for aa in AMINO_ACIDS if aa != seq[pos]]
        aas.append(rng.choice(candidates))
    return mutate_positions(seq, positions, aas), positions


def compute_adaptive_lambda_max(coupling_scores, threshold):
    """Choose the strongest epistasis weight this protein is allowed to use."""
    above = [s for _, _, s in coupling_scores if s >= threshold]
    if not above:
        return 0.0, 0.0

    mean_coupling = np.mean(above)
    lambda_max = np.clip(mean_coupling * LAMBDA_COUPLING_SCALE, LAMBDA_MIN, LAMBDA_CAP)
    return lambda_max, mean_coupling


def compute_lambda(current_prcs, start_prcs, lambda_max, ramp_frac=LAMBDA_RAMP_FRAC):
    """Ramp epistasis in gradually instead of trusting it from step zero."""
    gap = 1.0 - start_prcs
    if gap <= 0:
        return lambda_max

    progress = (current_prcs - start_prcs) / gap  # 0 to 1
    ramp = min(1.0, max(0.0, progress / ramp_frac))
    return lambda_max * ramp


def choose_adaptive_refinement(trajectory, current_prcs, start_prcs, k_single, p_coupled):
    """Shrink moves after PRCS progress or a PRCS-only plateau."""
    gap = max(1e-6, 1.0 - start_prcs)
    progress = max(0.0, min(1.0, (current_prcs - start_prcs) / gap))

    plateau = False
    if len(trajectory) >= ADAPTIVE_WINDOW * 2:
        earlier = trajectory[:-ADAPTIVE_WINDOW]
        recent = trajectory[-ADAPTIVE_WINDOW:]
        earlier_best_prcs = max(row["prcs"] for row in earlier)
        recent_best_prcs = max(row["prcs"] for row in recent)

        prcs_gain = recent_best_prcs - earlier_best_prcs
        plateau = prcs_gain < ADAPTIVE_PRCS_EPS

    refine_stage = 0
    if progress >= ADAPTIVE_STAGE1_THRESH:
        refine_stage = 1
    if progress >= ADAPTIVE_STAGE2_THRESH:
        refine_stage = 2
    if progress >= ADAPTIVE_STAGE3_THRESH:
        refine_stage = 3

    # If a protein stalls after making some progress, become one notch more careful.
    if plateau and progress >= ADAPTIVE_PLATEAU_THRESH:
        refine_stage = min(3, max(1, refine_stage + 1))

    if refine_stage >= 3:
        return min(k_single, ADAPTIVE_STAGE3_K_CAP), min(p_coupled, ADAPTIVE_STAGE3_P_COUPLED_CAP), refine_stage, progress, plateau
    if refine_stage == 2:
        return min(k_single, ADAPTIVE_STAGE2_K_CAP), min(p_coupled, ADAPTIVE_STAGE2_P_COUPLED_CAP), refine_stage, progress, plateau
    if refine_stage == 1:
        return min(k_single, ADAPTIVE_STAGE1_K_CAP), min(p_coupled, ADAPTIVE_STAGE1_P_COUPLED_CAP), refine_stage, progress, plateau
    return k_single, p_coupled, refine_stage, progress, plateau


def save_checkpoint(path, state):
    """Save enough state to continue a long walk later."""
    with open(path, "wb") as f:
        pickle.dump(state, f)


def load_checkpoint(path):
    """Load a saved walk state from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)


def get_checkpoint_path(outdir, protein_name, run_id):
    """Keep checkpoint filenames consistent across runs."""
    return Path(outdir) / protein_name / f"checkpoint_run{run_id}.pkl"


def run_walk(
    wt: str,
    protein_name: str,
    run_id: int,
    cfg: dict,
    emb_wt,
    model,
    alphabet,
    batch_converter,
    device,
    rng: random.Random,
    checkpoint_dir=None,
    start_sequence=None,
) -> dict:
    """Run one full walk for a single protein/run seed."""

    L = len(wt)
    max_steps = cfg["max_steps"]
    k_single = cfg["k_single"]
    p_coupled = cfg["p_coupled"]
    n_groups = cfg["n_groups"]
    gsmin = cfg["group_size_min"]
    gsmax = cfg["group_size_max"]
    cooling = cfg["cooling_rate"]
    adaptive_refine = cfg.get("adaptive_refine", False)
    heating_steps = cfg.get("heating_steps", 0)
    heating_k_min = cfg.get("heating_k_min", 0)
    heating_k_max = cfg.get("heating_k_max", 0)
    heating_probability = cfg.get("heating_probability", 0.0)

    start_step = 0
    ckpt_path = None
    if checkpoint_dir:
        ckpt_path = get_checkpoint_path(checkpoint_dir, protein_name, run_id)
        if ckpt_path.exists():
            print(f"  [Run {run_id}] RESUMING from checkpoint {ckpt_path}")
            ckpt = load_checkpoint(ckpt_path)
            current_seq = ckpt["current_seq"]
            current_prcs = ckpt["current_prcs"]
            current_combined = ckpt["current_combined"]
            current_hamming = ckpt["current_hamming"]
            start_prcs = ckpt["start_prcs"]
            start_hamming = ckpt.get("start_hamming", current_hamming)
            start_seq = ckpt.get("start_seq", "")
            best_seq = ckpt["best_seq"]
            best_prcs = ckpt["best_prcs"]
            best_hamming = ckpt["best_hamming"]
            best_combined = ckpt["best_combined"]
            temperature = ckpt["temperature"]
            lambda_max = ckpt["lambda_max"]
            groups = ckpt["groups"]
            trajectory = ckpt["trajectory"]
            n_accepted_single = ckpt["n_accepted_single"]
            n_proposed_single = ckpt["n_proposed_single"]
            n_accepted_coupled = ckpt["n_accepted_coupled"]
            n_proposed_coupled = ckpt["n_proposed_coupled"]
            rescue_count = ckpt.get("rescue_count", 0)
            last_rescue_step = ckpt.get("last_rescue_step", -RESCUE_COOLDOWN)
            rng.setstate(ckpt["rng_state"])
            start_step = ckpt["step"] + 1

            current_emb = get_per_residue_embedding(
                current_seq, model, alphabet, batch_converter, device
            )
            _, current_per_pos_prcs = score_prcs(current_emb, emb_wt)
            n_sample_pairs = get_coupling_sample_pairs(L)
            effective_threshold = ckpt.get("effective_threshold", COUPLING_THRESHOLD)
            coupling_scores = ckpt.get("coupling_scores", [])
            if not COUPLING_ENABLED:
                coupling_scores = []
                groups = []
                lambda_max = 0.0
            supported_pairs = coupling_support_set(coupling_scores, effective_threshold)
            coupling_map_jaccard = float(ckpt.get("coupling_map_jaccard", np.nan))

            print(
                f"  [Run {run_id}] Resumed at step {start_step}  "
                f"PRCS={current_prcs:.4f}  Hamming={current_hamming}  "
                f"Best Hamming={best_hamming}  λ_max={lambda_max:.3f}"
            )

            goto_loop = True
        else:
            goto_loop = False
    else:
        goto_loop = False

    if not goto_loop:
        print(
            f"  [Run {run_id}] L={L}  k_single={k_single}  "
            f"group_size={gsmin}-{gsmax}  p_coupled={p_coupled:.2f}  "
            f"steps={max_steps}"
        )
        if heating_steps:
            print(
                f"  [Run {run_id}] Big-protein heating: first {heating_steps} steps, "
                f"{heating_probability:.0%} coarse proposals, "
                f"k={heating_k_min}-{heating_k_max}"
            )

        if start_sequence is not None:
            if len(start_sequence) != L:
                raise ValueError(
                    f"start_sequence length {len(start_sequence)} does not match WT length {L}"
                )
            current_seq = start_sequence
            print(f"  [Run {run_id}] Starting from provided branch sequence")
        else:
            current_seq = random_sequence(L, rng)
        start_seq = current_seq
        current_emb = get_per_residue_embedding(
            current_seq, model, alphabet, batch_converter, device
        )

        start_prcs, start_perpos = score_prcs(current_emb, emb_wt)
        current_prcs = start_prcs
        current_per_pos_prcs = start_perpos  # for guided proposals
        current_combined = start_prcs  # λ=0 at start
        current_hamming = analysis_hamming(current_seq, wt)
        start_hamming = current_hamming

        print(f"  [Run {run_id}] Start PRCS={start_prcs:.4f}  Hamming={current_hamming}")

        n_sample_pairs = get_coupling_sample_pairs(L)
        if COUPLING_ENABLED:
            print(
                f"  [Run {run_id}] Computing initial coupling map "
                f"(double-masking, {n_sample_pairs} pairs)..."
            )
            coupling_scores = compute_epistatic_coupling_map(
                current_seq, model, alphabet, batch_converter, device, n_sample_pairs, rng
            )
        else:
            print(f"  [Run {run_id}] Coupling map disabled for this policy")
            coupling_scores = []
        n_above_thresh = sum(1 for _, _, s in coupling_scores if s >= COUPLING_THRESHOLD)
        print(
            f"  [Run {run_id}] {n_above_thresh}/{len(coupling_scores)} pairs "
            f"above coupling threshold {COUPLING_THRESHOLD}"
        )
        groups, effective_threshold = build_coupling_groups(
            coupling_scores, L, n_groups, gsmin, gsmax, COUPLING_THRESHOLD, rng
        )
        group_sizes = [len(g) for g in groups]
        print(
            f"  [Run {run_id}] {len(groups)} coupled groups, "
            f"sizes: {group_sizes[:10]}{'...' if len(group_sizes) > 10 else ''}"
        )
        if effective_threshold != COUPLING_THRESHOLD:
            print(
                f"  [Run {run_id}] Threshold adapted: {COUPLING_THRESHOLD} → "
                f"{effective_threshold:.4f}"
            )

        lambda_max, mean_coupling = compute_adaptive_lambda_max(
            coupling_scores, effective_threshold
        )
        supported_pairs = coupling_support_set(coupling_scores, effective_threshold)
        coupling_map_jaccard = np.nan
        print(
            f"  [Run {run_id}] Adaptive λ_max={lambda_max:.3f}  "
            f"(mean coupling={mean_coupling:.4f})"
        )

        temperature = INITIAL_TEMP
        best_seq = current_seq
        best_prcs = current_prcs
        best_hamming = current_hamming
        best_combined = current_combined

        trajectory = []
        n_accepted_single = 0
        n_proposed_single = 0
        n_accepted_coupled = 0
        n_proposed_coupled = 0
        rescue_count = 0
        last_rescue_step = -RESCUE_COOLDOWN

    def checkpoint_state(step_value):
        """Capture the complete mutable walk state for exact continuation."""
        return {
            "step": step_value,
            "current_seq": current_seq,
            "current_prcs": current_prcs,
            "current_combined": current_combined,
            "current_hamming": current_hamming,
            "start_prcs": start_prcs,
            "start_hamming": start_hamming,
            "start_seq": start_seq,
            "best_seq": best_seq,
            "best_prcs": best_prcs,
            "best_hamming": best_hamming,
            "best_combined": best_combined,
            "temperature": temperature,
            "lambda_max": lambda_max,
            "effective_threshold": effective_threshold,
            "groups": groups,
            "coupling_scores": coupling_scores,
            "coupling_map_jaccard": coupling_map_jaccard,
            "trajectory": trajectory,
            "n_accepted_single": n_accepted_single,
            "n_proposed_single": n_proposed_single,
            "n_accepted_coupled": n_accepted_coupled,
            "n_proposed_coupled": n_proposed_coupled,
            "rescue_count": rescue_count,
            "last_rescue_step": last_rescue_step,
            "rng_state": rng.getstate(),
        }

    for step in range(start_step, max_steps):
        if COUPLING_ENABLED and step > 0 and step % COUPLING_RECOMPUTE_EVERY == 0:
            print(f"  [Run {run_id}] Step {step}: recomputing coupling map (double-masking)...")
            previous_supported_pairs = supported_pairs
            coupling_scores = compute_epistatic_coupling_map(
                current_seq, model, alphabet, batch_converter, device, n_sample_pairs, rng
            )
            groups, effective_threshold = build_coupling_groups(
                coupling_scores, L, n_groups, gsmin, gsmax, COUPLING_THRESHOLD, rng
            )

            lambda_max, mean_coupling = compute_adaptive_lambda_max(
                coupling_scores, effective_threshold
            )
            supported_pairs = coupling_support_set(coupling_scores, effective_threshold)
            coupling_map_jaccard = set_jaccard(previous_supported_pairs, supported_pairs)

        # Candidate and current scores must use the same groups and the same
        # lambda. The historical implementation compared a candidate scored
        # with a new lambda/map against a stale current score.
        lam = compute_lambda(current_prcs, start_prcs, lambda_max)
        current_combined, current_prcs, current_epi, current_per_pos_prcs = (
            compute_combined_score(current_emb, emb_wt, groups, lam)
        )

        cand_seqs = []
        cand_coupled = []
        cand_positions = []
        adaptive_stage = 0
        adaptive_progress = 0.0
        adaptive_plateau = False
        effective_k_single = k_single
        effective_p_coupled = p_coupled
        if adaptive_refine:
            (
                effective_k_single,
                effective_p_coupled,
                adaptive_stage,
                adaptive_progress,
                adaptive_plateau,
            ) = choose_adaptive_refinement(
                trajectory, current_prcs, start_prcs, k_single, p_coupled
            )

        rescue_event = False
        rescue_positions = []
        rescue_prcs_before = None
        rescue_prcs_after = None
        rescue_combined_after = None
        rescue_accepted = False
        if (
            RESCUE_ENABLED
            and adaptive_refine
            and adaptive_plateau
            and step >= RESCUE_MIN_STEP
            and (step - last_rescue_step) >= RESCUE_COOLDOWN
            and RESCUE_PULSE_SIZE > 0
        ):
            rescue_prcs_before = current_prcs
            rescue_seq, rescue_positions = propose_rescue_pulse(
                current_seq, coupling_scores, RESCUE_PULSE_SIZE, rng
            )
            rescue_emb = get_per_residue_embedding(
                rescue_seq, model, alphabet, batch_converter, device
            )
            rescue_combined_after, rescue_prcs_after, rescue_epi_after, rescue_per_pos = (
                compute_combined_score(rescue_emb, emb_wt, groups, lam)
            )
            rescue_event = True
            rescue_count += 1
            last_rescue_step = step

            rescue_delta = rescue_combined_after - current_combined
            rescue_accepted = RESCUE_FORCE_ACCEPT or rescue_delta > 0
            if not rescue_accepted and temperature > 1e-10:
                rescue_accepted = rng.random() < math.exp(rescue_delta / temperature)

            if rescue_accepted:
                current_seq = rescue_seq
                current_emb = rescue_emb
                current_prcs = rescue_prcs_after
                current_epi = rescue_epi_after
                current_combined = rescue_combined_after
                current_hamming = analysis_hamming(current_seq, wt)
                current_per_pos_prcs = rescue_per_pos

                if current_prcs > best_prcs:
                    best_prcs = current_prcs
                    best_seq = current_seq
                    best_hamming = current_hamming
                    best_combined = current_combined

            print(
                f"  [Run {run_id}] Step {step}: rescue pulse "
                f"size={len(rescue_positions)}  "
                f"PRCS {rescue_prcs_before:.4f}->{rescue_prcs_after:.4f}  "
                f"accepted={rescue_accepted}  Hamming={current_hamming}"
            )

        for _ in range(PROPOSALS_PER_STEP):
            heating_active = heating_steps and step < heating_steps
            use_heating = heating_active and rng.random() < heating_probability

            if use_heating:
                k_heat = rng.randint(heating_k_min, heating_k_max)
                if adaptive_refine:
                    cand_seq, cand_pos = propose_single_guided(
                        current_seq, k_heat, current_per_pos_prcs, rng, p_guided=P_GUIDED_HEATING
                    )
                else:
                    cand_seq, cand_pos = propose_single(current_seq, k_heat, rng)
                is_coupled = False
            elif rng.random() < effective_p_coupled and groups:
                cand_seq, cand_pos = propose_coupled(current_seq, groups, gsmin, rng)
                is_coupled = True
            else:
                p_guided = P_GUIDED_ADAPTIVE if adaptive_refine else P_GUIDED_BASELINE
                cand_seq, cand_pos = propose_single_guided(
                    current_seq, effective_k_single, current_per_pos_prcs, rng, p_guided=p_guided
                )
                is_coupled = False
            cand_seqs.append(cand_seq)
            cand_coupled.append(is_coupled)
            cand_positions.append(cand_pos)

        cand_embeddings = get_per_residue_embeddings_batch(
            cand_seqs, model, alphabet, batch_converter, device
        )

        candidate_records = []

        for idx in range(len(cand_seqs)):
            cand_emb = cand_embeddings[idx]
            is_coupled = cand_coupled[idx]

            comb, prcs, epi, per_pos = compute_combined_score(cand_emb, emb_wt, groups, lam)

            candidate_records.append(
                {
                    "seq": cand_seqs[idx],
                    "emb": cand_emb,
                    "positions": cand_positions[idx],
                    "is_coupled": is_coupled,
                    "base_score": comb,
                    "selection_score": comb,
                    "prcs": prcs,
                    "epi": epi,
                    "per_pos": per_pos,
                    "neutral": None,
                }
            )

            if is_coupled:
                n_proposed_coupled += 1
            else:
                n_proposed_single += 1

        if NEUTRAL_REACHABILITY_ENABLED and NEUTRAL_RERANK_TOP_N > 0:
            top_records = sorted(
                candidate_records, key=lambda record: record["base_score"], reverse=True
            )[: min(NEUTRAL_RERANK_TOP_N, len(candidate_records))]
            for record in top_records:
                neutral = compute_neutral_reachability_delta(
                    record["seq"],
                    current_seq,
                    record["positions"],
                    record["per_pos"],
                    current_per_pos_prcs,
                    model,
                    alphabet,
                    batch_converter,
                    device,
                )
                record["neutral"] = neutral
                record["selection_score"] = (
                    record["base_score"] + NEUTRAL_BETA * neutral["reachability_delta"]
                )

        best_record = max(candidate_records, key=lambda record: record["selection_score"])
        best_cand = (best_record["seq"], best_record["emb"])
        best_cand_selection_score = best_record["selection_score"]
        best_cand_base_score = best_record["base_score"]
        best_cand_prcs = best_record["prcs"]
        best_cand_epi = best_record["epi"]
        best_cand_is_coupled = best_record["is_coupled"]
        best_cand_per_pos = best_record["per_pos"]
        best_neutral = best_record["neutral"] or {
            "reachability_delta": 0.0,
            "self_support_delta": 0.0,
            "region_balance_delta": 0.0,
            "disruption_penalty": 0.0,
        }

        delta = best_cand_selection_score - current_combined
        base_delta = best_cand_base_score - current_combined
        if delta > 0:
            accept = True
        elif temperature > 1e-10:
            accept = rng.random() < math.exp(delta / temperature)
        else:
            accept = False

        if accept:
            current_seq, current_emb = best_cand
            current_prcs = best_cand_prcs
            current_epi = best_cand_epi
            # The neutral term is a transition bonus relative to the previous
            # state. It must not accumulate into the next state's base score.
            current_combined = best_cand_base_score
            current_hamming = analysis_hamming(current_seq, wt)
            current_per_pos_prcs = best_cand_per_pos

            if best_cand_is_coupled:
                n_accepted_coupled += 1
            else:
                n_accepted_single += 1

            if current_prcs > best_prcs:
                best_prcs = current_prcs
                best_seq = current_seq
                best_hamming = current_hamming
                best_combined = current_combined

        temperature *= cooling

        diagnostic_score, prcs_lower_quartile, prcs_weakest_window = (
            score_fixed_diagnostic(current_per_pos_prcs)
        )

        row = {
            "step": step,
            "prcs": current_prcs,
            "epi": current_epi,
            "candidate_epi": best_cand_epi,
            "combined": current_combined,
            "diagnostic_score": diagnostic_score,
            "prcs_lower_quartile": prcs_lower_quartile,
            "prcs_weakest_window": prcs_weakest_window,
            "hamming": current_hamming,
            "best_hamming": best_hamming,
            "best_prcs": best_prcs,
            "lambda": lam,
            "lambda_max": lambda_max,
            "coupling_pairs_tested": len(coupling_scores),
            "coupling_pairs_supported": len(supported_pairs),
            "coupling_support_fraction": (
                len(supported_pairs) / len(coupling_scores) if coupling_scores else 0.0
            ),
            "coupling_group_count": len(groups),
            "coupling_map_jaccard": coupling_map_jaccard,
            "temperature": temperature,
            "accepted": accept,
            "coupled": bool(accept and best_cand_is_coupled),
            "candidate_coupled": best_cand_is_coupled,
            "heating_active": bool(heating_steps and step < heating_steps),
            "k_single_effective": effective_k_single,
            "effective_p_coupled": effective_p_coupled,
            "adaptive_stage": adaptive_stage,
            "adaptive_progress": adaptive_progress,
            "adaptive_plateau": adaptive_plateau,
            "delta": delta,
            "base_delta": base_delta,
            "neutral_enabled": NEUTRAL_REACHABILITY_ENABLED,
            "neutral_beta": NEUTRAL_BETA if best_record["neutral"] is not None else 0.0,
            "reachability_delta": best_neutral["reachability_delta"],
            "self_support_delta": best_neutral["self_support_delta"],
            "region_balance_delta": best_neutral["region_balance_delta"],
            "disruption_penalty": best_neutral["disruption_penalty"],
            "rescue_event": rescue_event,
            "rescue_accepted": rescue_accepted,
            "rescue_pulse_size": len(rescue_positions) if rescue_event else 0,
            "rescue_positions": ";".join(str(p) for p in rescue_positions),
            "rescue_prcs_before": rescue_prcs_before,
            "rescue_prcs_after": rescue_prcs_after,
            "rescue_combined_after": rescue_combined_after,
        }
        if SAVE_SEQUENCE_TRAJECTORY:
            row["sequence"] = current_seq
        trajectory.append(row)

        if step > 0 and step % LOG_EVERY == 0:
            msg = (
                f"  [Run {run_id}] Step {step}  "
                f"PRCS={current_prcs:.4f}  Epi={current_epi:.4f}  "
                f"Combined={current_combined:.4f}  "
                f"Hamming={current_hamming}  "
                f"k={effective_k_single}  p_c={effective_p_coupled:.2f}  "
                f"stage={adaptive_stage}  prog={adaptive_progress:.2f}  "
                f"lambda={lam:.3f}(max={lambda_max:.3f})  T={temperature:.4f}"
            )
            print(msg)

        if step > 0 and step % PLL_DIAGNOSTIC_EVERY == 0:
            pll = score_true_pll(current_seq, model, alphabet, batch_converter, device)
            print(f"  [Run {run_id}] Step {step} PLL diagnostic: {pll:.4f}")
            trajectory[-1]["pll"] = pll

        if ckpt_path and step > 0 and step % CHECKPOINT_EVERY == 0:
            save_checkpoint(ckpt_path, checkpoint_state(step))
            print(f"  [Run {run_id}] Checkpoint saved at step {step}")

    acc_single = n_accepted_single / n_proposed_single * 100 if n_proposed_single > 0 else 0
    acc_coupled = n_accepted_coupled / n_proposed_coupled * 100 if n_proposed_coupled > 0 else 0

    final_hamming = analysis_hamming(current_seq, wt)

    # Segment-based outer controllers need an exact continuation point even
    # when a segment ends between the periodic checkpoint intervals.
    if ckpt_path and max_steps > 0:
        save_checkpoint(ckpt_path, checkpoint_state(max_steps - 1))
        print(f"  [Run {run_id}] Segment checkpoint saved at step {max_steps - 1}")

    print(f"  [Run {run_id}] Finished after {max_steps} steps")
    print(f"    Final PRCS={current_prcs:.4f}  Best PRCS={best_prcs:.4f}")
    print(f"    Final Hamming={final_hamming}  Best Hamming={best_hamming}")
    print(f"    Accept rate: single={acc_single:.1f}%  coupled={acc_coupled:.1f}%")
    print(f"    Final λ={lam:.3f}  λ_max={lambda_max:.3f}")

    summary = {
        "protein": protein_name,
        "length": L,
        "run": run_id,
        "experiment_mode": EXPERIMENT_MODE,
        "start_prcs": start_prcs,
        "start_hamming": start_hamming,
        "best_prcs": best_prcs,
        "best_hamming": best_hamming,
        "best_combined": best_combined,
        "final_prcs": current_prcs,
        "final_hamming": final_hamming,
        "final_combined": current_combined,
        "final_lambda": lam,
        "final_lambda_max": lambda_max,
        "accept_rate_single": acc_single,
        "accept_rate_coupled": acc_coupled,
        "steps": max_steps,
        "k_single": k_single,
        "group_size_min": gsmin,
        "group_size_max": gsmax,
        "heating_steps": heating_steps,
        "heating_k_min": heating_k_min,
        "heating_k_max": heating_k_max,
        "heating_probability": heating_probability,
        "p_coupled": p_coupled,
        "n_groups": len(groups),
        "rescue_enabled": RESCUE_ENABLED,
        "rescue_pulse_size": RESCUE_PULSE_SIZE,
        "rescue_min_step": RESCUE_MIN_STEP,
        "rescue_cooldown": RESCUE_COOLDOWN,
        "rescue_count": rescue_count,
    }

    return {
        "trajectory": trajectory,
        "summary": summary,
        "start_seq": start_seq,
        "best_seq": best_seq,
        "final_seq": current_seq,
    }


def main():
    """Coordinate the full experiment across FASTA files and runs."""
    top_outdir = Path(OUTDIR)
    top_outdir.mkdir(exist_ok=True, parents=True)

    fasta_dir = Path(FASTA_DIR)
    if not fasta_dir.exists():
        print(f"ERROR: FASTA directory '{FASTA_DIR}' not found.")
        print("Create it and place .fasta files inside, or edit FASTA_DIR.")
        sys.exit(1)

    fasta_files = sorted(fasta_dir.glob("*.fasta"))
    if not fasta_files:
        fasta_files = sorted(fasta_dir.glob("*.fa"))
    if not fasta_files:
        print(f"ERROR: No .fasta or .fa files found in '{FASTA_DIR}'")
        sys.exit(1)

    # Load protein info
    proteins = []
    for fp in fasta_files:
        name, seq = read_fasta_one(str(fp))
        # Use filename stem as protein name for consistency
        name = fp.stem
        proteins.append({"name": name, "seq": seq, "length": len(seq), "path": str(fp)})

    if PROTEIN_FILTER:
        requested = {name.strip().lower() for name in PROTEIN_FILTER.split(",") if name.strip()}
        proteins = [p for p in proteins if p["name"].lower() in requested]
        if not proteins:
            print(f"ERROR: PROTEIN_FILTER matched no FASTA files: {PROTEIN_FILTER}")
            sys.exit(1)

    print("=" * 70)
    print("  EVOLUTION WALK V4 — ADAPTIVE REFINE PRCS-ONLY")
    print(f"  FASTA directory: {FASTA_DIR}")
    if PROTEIN_FILTER:
        print(f"  Protein filter: {PROTEIN_FILTER}")
    print(f"  Experiment mode: {EXPERIMENT_MODE}")
    print(
        f"  Rescue pulses: enabled={RESCUE_ENABLED}  "
        f"size={RESCUE_PULSE_SIZE}  min_step={RESCUE_MIN_STEP}  "
        f"cooldown={RESCUE_COOLDOWN}"
    )
    print(f"  Proteins: {len(proteins)}")
    print(f"  Runs per protein: {RUNS_PER_PROTEIN}")
    print(f"  Seed base: {SEED}")
    print(
        "  Available big-protein knobs: "
        f"L>={BIG_PROTEIN_MIN_LENGTH}, steps=max(baseline,{BIG_STEPS_PER_RESIDUE}*L), "
        f"max_steps_override={MAX_STEPS_OVERRIDE}, "
        f"groups>={BIG_GROUP_SIZE_MIN}-{BIG_GROUP_SIZE_MAX}, "
        f"heating={BIG_HEATING_STEPS} steps"
    )
    print(
        f"  Coupling: double-masking (length-adaptive pairs, "
        f"threshold={COUPLING_THRESHOLD}, adaptive λ_max)"
    )
    print(
        "  ML schedule knobs: "
        f"window={ADAPTIVE_WINDOW}, eps={ADAPTIVE_PRCS_EPS}, "
        f"stage_thresh=({ADAPTIVE_STAGE1_THRESH}, {ADAPTIVE_STAGE2_THRESH}, {ADAPTIVE_STAGE3_THRESH}), "
        f"k_caps=({ADAPTIVE_STAGE1_K_CAP}, {ADAPTIVE_STAGE2_K_CAP}, {ADAPTIVE_STAGE3_K_CAP}), "
        f"p_coupled_caps=({ADAPTIVE_STAGE1_P_COUPLED_CAP}, {ADAPTIVE_STAGE2_P_COUPLED_CAP}, {ADAPTIVE_STAGE3_P_COUPLED_CAP}), "
        f"p_guided={P_GUIDED_ADAPTIVE}, worst_frac={PRCS_WORST_FRAC}, "
        f"lambda_cap={LAMBDA_CAP}, lambda_ramp={LAMBDA_RAMP_FRAC}, "
        f"proposals={PROPOSALS_PER_STEP}, recompute_every={COUPLING_RECOMPUTE_EVERY}, "
        f"p_coupled_override={P_COUPLED_OVERRIDE}, k_single_override={K_SINGLE_OVERRIDE}"
    )
    print("=" * 70)
    for p in proteins:
        cfg = get_protein_config(p["length"])
        n_pairs = get_coupling_sample_pairs(p["length"])
        print(
            f"  {p['name']:40s} {p['length']:4d} residues  "
            f"k={cfg['k_single']}  steps={cfg['max_steps']}  "
            f"group={cfg['group_size_min']}-{cfg['group_size_max']}  "
            f"p_coupled={cfg['p_coupled']:.2f}  pairs={n_pairs}"
        )
    print()

    model, alphabet, batch_converter = load_esm2(DEVICE)

    all_protein_results = {}
    for protein in proteins:
        all_protein_results[protein["name"]] = {
            "trajectories": [],
            "summaries": [],
            "best_hamming": protein["length"] + 1,
            "best_seq": None,
            "best_prcs": -999,
            "completed_runs": set(),
        }

    n_resumed = 0
    for protein in proteins:
        pname = protein["name"]
        prot_outdir = top_outdir / pname
        pr = all_protein_results[pname]

        for run_id in range(RUNS_PER_PROTEIN):
            traj_path = prot_outdir / f"trajectory_run{run_id}.csv"
            if traj_path.exists():
                try:
                    df = pd.read_csv(traj_path)
                    traj = df.to_dict("records")
                    pr["trajectories"].append(traj)
                    pr["completed_runs"].add(run_id)

                    best_h = int(df["best_hamming"].min())
                    best_p = float(df["best_prcs"].max())
                    final_h = int(df["hamming"].iloc[-1])
                    final_p = float(df["prcs"].iloc[-1])
                    final_c = float(df["combined"].iloc[-1])
                    final_lam = float(df["lambda"].iloc[-1])
                    final_lam_max = (
                        float(df["lambda_max"].iloc[-1]) if "lambda_max" in df.columns else 0.0
                    )

                    summary = {
                        "protein": pname,
                        "length": protein["length"],
                        "run": run_id,
                        "start_prcs": float(df["prcs"].iloc[0]),
                        "start_hamming": int(df["hamming"].iloc[0]),
                        "best_prcs": best_p,
                        "best_hamming": best_h,
                        "best_combined": float(df["combined"].max()),
                        "final_prcs": final_p,
                        "final_hamming": final_h,
                        "final_combined": final_c,
                        "final_lambda": final_lam,
                        "final_lambda_max": final_lam_max,
                        "accept_rate_single": float(
                            df.loc[~df["coupled"], "accepted"].mean() * 100
                        )
                        if (~df["coupled"]).any()
                        else 0,
                        "accept_rate_coupled": float(
                            df.loc[df["coupled"], "accepted"].mean() * 100
                        )
                        if df["coupled"].any()
                        else 0,
                        "steps": len(df),
                        "k_single": get_protein_config(protein["length"])["k_single"],
                        "p_coupled": get_protein_config(protein["length"])["p_coupled"],
                        "n_groups": 0,
                    }
                    pr["summaries"].append(summary)

                    pr["best_hamming"] = min(pr["best_hamming"], best_h)
                    if best_p > pr["best_prcs"]:
                        pr["best_prcs"] = best_p

                    n_resumed += 1
                    print(
                        f"  Resumed {pname} run {run_id}: best Hamming={best_h}, PRCS={best_p:.4f}"
                    )
                except Exception as e:
                    print(f"  Warning: could not load {traj_path}: {e}")

        fasta_path = prot_outdir / "best_sequence.fasta"
        if fasta_path.exists() and pr["best_seq"] is None:
            try:
                _, seq = read_fasta_one(str(fasta_path))
                pr["best_seq"] = seq
            except Exception:
                pass

    if n_resumed > 0:
        print(f"\n  Resumed {n_resumed} completed run(s) from previous session")

    for run_id in range(RUNS_PER_PROTEIN):
        print()
        print("=" * 70)
        print(f"  TRANCHE {run_id + 1}/{RUNS_PER_PROTEIN}")
        print("=" * 70)

        for pidx, protein in enumerate(proteins):
            pname = protein["name"]
            wt = protein["seq"]
            L = protein["length"]
            cfg = get_protein_config(L)
            pr = all_protein_results[pname]

            prot_outdir = top_outdir / pname
            prot_outdir.mkdir(exist_ok=True, parents=True)

            if run_id in pr["completed_runs"]:
                print(f"\n  [{pname}] Run {run_id} already completed, skipping")
                continue

            print()
            print("#" * 70)
            print(
                f"  PROTEIN {pidx + 1}/{len(proteins)}: {pname}  "
                f"[RUN {run_id + 1}/{RUNS_PER_PROTEIN}]"
            )
            print(
                f"  Length: {L}  Steps: {cfg['max_steps']}  "
                f"k_single: {cfg['k_single']}  p_coupled: {cfg['p_coupled']:.2f}"
            )
            print(
                f"  Groups: {cfg['n_groups']} of size "
                f"{cfg['group_size_min']}-{cfg['group_size_max']}"
            )
            print("#" * 70)

            print("  Computing WT reference embedding...")
            emb_wt = get_per_residue_embedding(wt, model, alphabet, batch_converter, DEVICE)

            if run_id == 0:
                test_rng = random.Random(0)
                test_seq = random_sequence(L, test_rng)
                test_emb = get_per_residue_embedding(
                    test_seq, model, alphabet, batch_converter, DEVICE
                )
                rand_prcs, _ = score_prcs(test_emb, emb_wt)
                rand_hamming = analysis_hamming(test_seq, wt)
                print("  WT PRCS: 1.000000")
                print(f"  Random PRCS: {rand_prcs:.4f}  Random Hamming: {rand_hamming}")

            print()
            print(f"  {'=' * 55}")
            print(f"    RUN {run_id + 1}/{RUNS_PER_PROTEIN}  [{pname}]")
            print(f"  {'=' * 55}")

            run_rng = random.Random(SEED + pidx * 100 + run_id)
            t0 = time.time()

            result = run_walk(
                wt,
                pname,
                run_id,
                cfg,
                emb_wt,
                model,
                alphabet,
                batch_converter,
                DEVICE,
                run_rng,
                checkpoint_dir=top_outdir,
            )

            dt = time.time() - t0
            s = result["summary"]
            print(
                f"    Run {run_id} done in {dt / 60:.1f} min  "
                f"PRCS={s['best_prcs']:.4f}  "
                f"Best Hamming={s['best_hamming']}  "
                f"Final Hamming={s['final_hamming']}"
            )

            ckpt_path = get_checkpoint_path(top_outdir, pname, run_id)
            if ckpt_path.exists():
                ckpt_path.unlink()
                print("    Checkpoint removed (run completed)")

            pr = all_protein_results[pname]
            pr["trajectories"].append(result["trajectory"])
            pr["summaries"].append(s)
            pr["completed_runs"].add(run_id)

            pr["best_hamming"] = min(pr["best_hamming"], s["best_hamming"])
            if s["best_prcs"] > pr["best_prcs"]:
                pr["best_seq"] = result["best_seq"]
                pr["best_prcs"] = s["best_prcs"]

            df_traj = pd.DataFrame(result["trajectory"])
            if "sequence" in df_traj.columns:
                df_traj.to_csv(
                    prot_outdir / f"sequence_trajectory_run{run_id}.csv.gz",
                    index=False,
                    compression="gzip",
                )
                df_traj.drop(columns=["sequence"]).to_csv(
                    prot_outdir / f"trajectory_run{run_id}.csv", index=False
                )
            else:
                df_traj.to_csv(prot_outdir / f"trajectory_run{run_id}.csv", index=False)

            if SAVE_SEQUENCE_TRAJECTORY:
                write_sequence_fasta(
                    prot_outdir / f"start_sequence_run{run_id}.fasta",
                    f"{pname}_run{run_id}_start_H{s['start_hamming']}",
                    result["start_seq"],
                )
                write_sequence_fasta(
                    prot_outdir / f"final_sequence_run{run_id}.fasta",
                    f"{pname}_run{run_id}_final_H{s['final_hamming']}",
                    result["final_seq"],
                )

            if pr["best_seq"]:
                write_sequence_fasta(
                    prot_outdir / "best_sequence.fasta",
                    f"{pname}_best_PRCS_{pr['best_prcs']:.6f}",
                    pr["best_seq"],
                )

            # Print running summary for this protein
            print(f"\n  [{pname}] After {run_id + 1} run(s):")
            for rs in pr["summaries"]:
                print(
                    f"    Run {rs['run']}: PRCS={rs['best_prcs']:.4f}  "
                    f"Hamming={rs['final_hamming']} "
                    f"(best={rs['best_hamming']})"
                )
            print(f"    Best overall Hamming: {pr['best_hamming']}")

        print()
        print("=" * 70)
        print(f"  TRANCHE {run_id + 1} COMPLETE — CROSS-PROTEIN SUMMARY")
        print("=" * 70)
        print(
            f"  {'Protein':<25s} {'L':>4s} {'Best H':>7s} {'%Rec':>6s} {'PRCS':>6s} {'Runs':>5s}"
        )
        print(f"  {'-' * 25} {'-' * 4} {'-' * 7} {'-' * 6} {'-' * 6} {'-' * 5}")
        for protein in proteins:
            pn = protein["name"]
            pr = all_protein_results[pn]
            if pr["summaries"]:
                pct = 100 * (1 - pr["best_hamming"] / protein["length"])
                print(
                    f"  {pn:<25s} {protein['length']:4d} "
                    f"{pr['best_hamming']:7d} {pct:5.1f}% "
                    f"{pr['best_prcs']:6.3f} "
                    f"{len(pr['summaries']):5d}"
                )

    print()
    print("=" * 70)
    print("  GENERATING FINAL PLOTS AND SUMMARIES")
    print("=" * 70)

    all_cross_results = []

    for protein in proteins:
        pname = protein["name"]
        L = protein["length"]
        cfg = get_protein_config(L)
        pr = all_protein_results[pname]
        prot_outdir = top_outdir / pname

        if not pr["summaries"]:
            continue

        df_summ = pd.DataFrame(pr["summaries"])
        df_summ.to_csv(prot_outdir / "summary.csv", index=False)

        print(f"\n  [{pname}] FINAL SUMMARY:")
        for s in pr["summaries"]:
            print(
                f"    Run {s['run']}: PRCS={s['best_prcs']:.4f}  "
                f"Hamming={s['final_hamming']} (best={s['best_hamming']})"
            )
        print(
            f"    Mean best Hamming: {np.mean([s['best_hamming'] for s in pr['summaries']]):.1f}"
        )
        print(f"    Best overall Hamming: {pr['best_hamming']}")

        # Plots
        plot_walk(pr["trajectories"], pname, L, prot_outdir)
        plot_proposals(pr["trajectories"], pname, L, prot_outdir)

        all_cross_results.append(
            {
                "protein": pname,
                "length": L,
                "best_hamming": pr["best_hamming"],
                "best_prcs": pr["best_prcs"],
                "mean_best_hamming": np.mean([s["best_hamming"] for s in pr["summaries"]]),
                "accept_rate_single": np.mean([s["accept_rate_single"] for s in pr["summaries"]]),
                "accept_rate_coupled": np.mean(
                    [s["accept_rate_coupled"] for s in pr["summaries"]]
                ),
                "k_single": cfg["k_single"],
                "p_coupled": cfg["p_coupled"],
                "steps": cfg["max_steps"],
            }
        )

    print("\n" + "=" * 70)
    print("  CROSS-PROTEIN SUMMARY")
    print("=" * 70)
    print(
        f"  {'Protein':<25s} {'L':>4s} {'Best H':>7s} {'%Rec':>6s} "
        f"{'PRCS':>6s} {'k':>3s} {'Steps':>6s}"
    )
    print(f"  {'-' * 25} {'-' * 4} {'-' * 7} {'-' * 6} {'-' * 6} {'-' * 3} {'-' * 6}")
    for r in all_cross_results:
        pct = 100 * (1 - r["best_hamming"] / r["length"])
        print(
            f"  {r['protein']:<25s} {r['length']:4d} "
            f"{r['best_hamming']:7d} {pct:5.1f}% "
            f"{r['best_prcs']:6.3f} {r['k_single']:3d} "
            f"{r['steps']:6d}"
        )

    df_cross = pd.DataFrame(all_cross_results)
    df_cross.to_csv(top_outdir / "cross_protein_summary.csv", index=False)

    plot_cross_protein(all_cross_results, top_outdir)
    plot_recovery_over_steps(all_protein_results, top_outdir)

    print(f"\nAll results saved to {top_outdir}/")
    print("Done.")


if __name__ == "__main__":
    main()
