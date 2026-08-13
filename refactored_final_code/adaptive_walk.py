#!/usr/bin/env python3
"""Execution loop for the final corrected adaptive independent walker."""

from __future__ import annotations

import math
from pathlib import Path
import random

import numpy as np

import corrected_adaptive_engine as core


def run_walk(
    *,
    protein_name: str,
    walker_id: int,
    start_sequence: str,
    target_embedding,
    model,
    alphabet,
    batch_converter,
    device: str,
    rng: random.Random,
    config: core.WalkConfig,
    checkpoint_path: Path | None = None,
) -> dict:
    """Run one walker and resume it exactly when a matching checkpoint exists."""

    length = len(start_sequence)
    if length != config.length:
        raise ValueError("Start sequence and configuration lengths do not match")
    if tuple(target_embedding.shape)[:1] != (length,):
        raise ValueError("Target embedding length does not match the start sequence")
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    if checkpoint_path is not None and checkpoint_path.exists():
        state = core.load_checkpoint(checkpoint_path)
        if state.get("protein_name") != protein_name or state.get("walker_id") != walker_id:
            raise RuntimeError("Checkpoint identity does not match this walk")
        if state.get("config") != config.as_dict():
            raise RuntimeError("Checkpoint configuration does not match this walk")
        current_sequence = state["current_sequence"]
        start_sequence = state["start_sequence"]
        start_prcs = float(state["start_prcs"])
        current_prcs = float(state["current_prcs"])
        current_combined = float(state["current_combined"])
        best_sequence = state["best_sequence"]
        best_prcs = float(state["best_prcs"])
        best_combined = float(state["best_combined"])
        temperature = float(state["temperature"])
        lambda_max = float(state["lambda_max"])
        effective_threshold = float(state["effective_threshold"])
        groups = state["groups"]
        coupling_scores = state["coupling_scores"]
        coupling_map_jaccard = float(state.get("coupling_map_jaccard", np.nan))
        trajectory = state["trajectory"]
        accepted_single = int(state["accepted_single"])
        proposed_single = int(state["proposed_single"])
        accepted_coupled = int(state["accepted_coupled"])
        proposed_coupled = int(state["proposed_coupled"])
        rng.setstate(state["rng_state"])
        first_step = int(state["step"]) + 1
        current_embedding = core.get_per_residue_embedding(
            current_sequence, model, alphabet, batch_converter, device
        )
        _, current_per_position = core.score_prcs(current_embedding, target_embedding)
        supported_pairs = core.coupling_support_set(
            coupling_scores, effective_threshold
        )
        print(
            f"  [Walker {walker_id}] Resuming at step {first_step}; "
            f"PRCS={current_prcs:.4f}, best={best_prcs:.4f}"
        )
    else:
        current_sequence = start_sequence
        current_embedding = core.get_per_residue_embedding(
            current_sequence, model, alphabet, batch_converter, device
        )
        start_prcs, current_per_position = core.score_prcs(
            current_embedding, target_embedding
        )
        current_prcs = start_prcs
        current_combined = start_prcs
        print(
            f"  [Walker {walker_id}] {protein_name}: L={length}, "
            f"steps={config.max_steps}, start PRCS={start_prcs:.4f}"
        )
        if config.heating_steps:
            print(
                f"  [Walker {walker_id}] Heating for {config.heating_steps} steps; "
                f"k={config.heating_k_min}-{config.heating_k_max}, "
                f"probability={config.heating_probability:.2f}"
            )
        sample_count = core.get_coupling_sample_pairs(length)
        print(
            f"  [Walker {walker_id}] Computing initial coupling map "
            f"({sample_count} sampled pairs)"
        )
        coupling_scores = core.compute_epistatic_coupling_map(
            current_sequence,
            model,
            alphabet,
            batch_converter,
            device,
            sample_count,
            rng,
        )
        groups, effective_threshold = core.build_coupling_groups(
            coupling_scores,
            length,
            config.n_groups,
            config.group_size_min,
            config.group_size_max,
            core.COUPLING_THRESHOLD,
        )
        lambda_max, mean_coupling = core.compute_adaptive_lambda_max(
            coupling_scores, effective_threshold
        )
        supported_pairs = core.coupling_support_set(
            coupling_scores, effective_threshold
        )
        coupling_map_jaccard = np.nan
        print(
            f"  [Walker {walker_id}] groups={len(groups)}, "
            f"threshold={effective_threshold:.4f}, "
            f"lambda_max={lambda_max:.3f}, mean coupling={mean_coupling:.4f}"
        )
        temperature = core.INITIAL_TEMPERATURE
        best_sequence = current_sequence
        best_prcs = current_prcs
        best_combined = current_combined
        trajectory = []
        accepted_single = proposed_single = 0
        accepted_coupled = proposed_coupled = 0
        first_step = 0

    sample_count = core.get_coupling_sample_pairs(length)

    def checkpoint_state(step: int) -> dict:
        return {
            "schema_version": 1,
            "protein_name": protein_name,
            "walker_id": walker_id,
            "config": config.as_dict(),
            "step": step,
            "start_sequence": start_sequence,
            "start_prcs": start_prcs,
            "current_sequence": current_sequence,
            "current_prcs": current_prcs,
            "current_combined": current_combined,
            "best_sequence": best_sequence,
            "best_prcs": best_prcs,
            "best_combined": best_combined,
            "temperature": temperature,
            "lambda_max": lambda_max,
            "effective_threshold": effective_threshold,
            "groups": groups,
            "coupling_scores": coupling_scores,
            "coupling_map_jaccard": coupling_map_jaccard,
            "trajectory": trajectory,
            "accepted_single": accepted_single,
            "proposed_single": proposed_single,
            "accepted_coupled": accepted_coupled,
            "proposed_coupled": proposed_coupled,
            "rng_state": rng.getstate(),
        }

    for step in range(first_step, config.max_steps):
        if step > 0 and step % core.COUPLING_RECOMPUTE_EVERY == 0:
            print(f"  [Walker {walker_id}] Step {step}: recomputing coupling map")
            previous_supported_pairs = supported_pairs
            coupling_scores = core.compute_epistatic_coupling_map(
                current_sequence,
                model,
                alphabet,
                batch_converter,
                device,
                sample_count,
                rng,
            )
            groups, effective_threshold = core.build_coupling_groups(
                coupling_scores,
                length,
                config.n_groups,
                config.group_size_min,
                config.group_size_max,
                core.COUPLING_THRESHOLD,
            )
            lambda_max, _ = core.compute_adaptive_lambda_max(
                coupling_scores, effective_threshold
            )
            supported_pairs = core.coupling_support_set(
                coupling_scores, effective_threshold
            )
            coupling_map_jaccard = core.set_jaccard(
                previous_supported_pairs, supported_pairs
            )

        # The parent and all candidates are evaluated with this same map/lambda.
        coupling_lambda = core.compute_lambda(current_prcs, start_prcs, lambda_max)
        current_combined, current_prcs, current_epistasis, current_per_position = (
            core.compute_combined_score(
                current_embedding, target_embedding, groups, coupling_lambda
            )
        )
        effective_k, effective_p_coupled, adaptive_stage, adaptive_progress, adaptive_plateau = (
            core.choose_adaptive_refinement(
                trajectory,
                current_prcs,
                start_prcs,
                config.k_single,
                config.p_coupled,
            )
        )

        candidate_sequences = []
        candidate_coupled = []
        candidate_positions = []
        for _ in range(config.proposals_per_step):
            heating_active = bool(config.heating_steps and step < config.heating_steps)
            use_heating = heating_active and rng.random() < config.heating_probability
            if use_heating:
                heating_k = rng.randint(config.heating_k_min, config.heating_k_max)
                candidate, positions = core.propose_single_guided(
                    current_sequence,
                    heating_k,
                    current_per_position,
                    rng,
                    guided_probability=core.P_GUIDED_HEATING,
                )
                is_coupled = False
            elif rng.random() < effective_p_coupled and groups:
                candidate, positions = core.propose_coupled(
                    current_sequence, groups, rng
                )
                is_coupled = True
            else:
                candidate, positions = core.propose_single_guided(
                    current_sequence,
                    effective_k,
                    current_per_position,
                    rng,
                    guided_probability=core.P_GUIDED_ADAPTIVE,
                )
                is_coupled = False
            candidate_sequences.append(candidate)
            candidate_coupled.append(is_coupled)
            candidate_positions.append(positions)

        candidate_embeddings = core.get_per_residue_embeddings_batch(
            candidate_sequences, model, alphabet, batch_converter, device
        )
        records = []
        for candidate, embedding, positions, is_coupled in zip(
            candidate_sequences,
            candidate_embeddings,
            candidate_positions,
            candidate_coupled,
        ):
            combined, prcs, epistasis, per_position = core.compute_combined_score(
                embedding, target_embedding, groups, coupling_lambda
            )
            records.append(
                {
                    "sequence": candidate,
                    "embedding": embedding,
                    "positions": positions,
                    "coupled": is_coupled,
                    "combined": combined,
                    "prcs": prcs,
                    "epistasis": epistasis,
                    "per_position": per_position,
                }
            )
            if is_coupled:
                proposed_coupled += 1
            else:
                proposed_single += 1

        best_candidate = max(records, key=lambda record: record["combined"])
        delta = best_candidate["combined"] - current_combined
        accepted = delta > 0 or (
            temperature > 1e-10 and rng.random() < math.exp(delta / temperature)
        )
        if accepted:
            current_sequence = best_candidate["sequence"]
            current_embedding = best_candidate["embedding"]
            current_prcs = best_candidate["prcs"]
            current_epistasis = best_candidate["epistasis"]
            current_combined = best_candidate["combined"]
            current_per_position = best_candidate["per_position"]
            if best_candidate["coupled"]:
                accepted_coupled += 1
            else:
                accepted_single += 1
            if current_prcs > best_prcs:
                best_sequence = current_sequence
                best_prcs = current_prcs
                best_combined = current_combined

        temperature *= config.cooling_rate
        diagnostic, lower_quartile, weakest_window = core.score_fixed_diagnostic(
            current_per_position
        )
        trajectory.append(
            {
                "step": step,
                "prcs": current_prcs,
                "epistasis": current_epistasis,
                "candidate_epistasis": best_candidate["epistasis"],
                "combined": current_combined,
                "diagnostic_score": diagnostic,
                "prcs_lower_quartile": lower_quartile,
                "prcs_weakest_window": weakest_window,
                "best_prcs": best_prcs,
                "lambda": coupling_lambda,
                "lambda_max": lambda_max,
                "coupling_pairs_tested": len(coupling_scores),
                "coupling_pairs_supported": len(supported_pairs),
                "coupling_group_count": len(groups),
                "coupling_map_jaccard": coupling_map_jaccard,
                "temperature": temperature,
                "accepted": accepted,
                "candidate_coupled": best_candidate["coupled"],
                "heating_active": bool(
                    config.heating_steps and step < config.heating_steps
                ),
                "k_single_effective": effective_k,
                "effective_p_coupled": effective_p_coupled,
                "adaptive_stage": adaptive_stage,
                "adaptive_progress": adaptive_progress,
                "adaptive_plateau": adaptive_plateau,
                "delta": delta,
                "mutated_positions": ";".join(
                    str(position) for position in best_candidate["positions"]
                ),
                "sequence": current_sequence,
            }
        )

        if step > 0 and step % core.LOG_EVERY == 0:
            print(
                f"  [Walker {walker_id}] Step {step}: PRCS={current_prcs:.4f}, "
                f"epi={current_epistasis:.4f}, combined={current_combined:.4f}, "
                f"k={effective_k}, p_c={effective_p_coupled:.2f}, "
                f"stage={adaptive_stage}, progress={adaptive_progress:.2f}, "
                f"lambda={coupling_lambda:.3f}/{lambda_max:.3f}, T={temperature:.4f}"
            )
        if step > 0 and step % core.PLL_DIAGNOSTIC_EVERY == 0:
            pll = core.score_true_pll(
                current_sequence, model, alphabet, batch_converter, device
            )
            trajectory[-1]["pll"] = pll
            print(f"  [Walker {walker_id}] Step {step}: PLL={pll:.4f}")
        if (
            checkpoint_path is not None
            and step > 0
            and step % core.CHECKPOINT_EVERY == 0
        ):
            core.save_checkpoint(checkpoint_path, checkpoint_state(step))

    if checkpoint_path is not None and config.max_steps > 0:
        core.save_checkpoint(checkpoint_path, checkpoint_state(config.max_steps - 1))
    single_rate = 100.0 * accepted_single / proposed_single if proposed_single else 0.0
    coupled_rate = (
        100.0 * accepted_coupled / proposed_coupled if proposed_coupled else 0.0
    )
    print(
        f"  [Walker {walker_id}] Finished: final PRCS={current_prcs:.4f}, "
        f"best PRCS={best_prcs:.4f}, single acceptance={single_rate:.1f}%, "
        f"coupled acceptance={coupled_rate:.1f}%"
    )
    return {
        "protein_name": protein_name,
        "walker_id": walker_id,
        "start_sequence": start_sequence,
        "start_prcs": start_prcs,
        "final_sequence": current_sequence,
        "final_prcs": current_prcs,
        "final_combined": current_combined,
        "best_sequence": best_sequence,
        "best_prcs": best_prcs,
        "best_combined": best_combined,
        "trajectory": trajectory,
        "acceptance_rate_single": single_rate,
        "acceptance_rate_coupled": coupled_rate,
    }
