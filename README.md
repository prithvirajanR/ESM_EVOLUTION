# ESM_EVOLUTION

This repository contains the commented, pre-correction REVO sequence-search implementation and the complete trajectory handoff archive used in the ESM evolution experiments.

## Contents

- `code/revo_weighted_ensemble_v1.py`: weighted-ensemble controller and REVO resampling loop.
- `code/revo_core.py`: walker state, variation calculation, cloning, merging, and weight handling.
- `code/evolution_walk_universal_ignition_v1.py`: adaptive ESM-guided sequence-search engine used by each walker.
- `TRAJECTORY_HANDOFF_ALL_96.zip`: organized handoff containing 96 trajectories across four proteins and three experimental waves, with the sequence and recorded metrics at every evolution step.

The trajectory archive intentionally uses neutral trajectory labels rather than controller names. Each trajectory spans 3,000 evolution steps. The handoff marks a trajectory as successful when PRCS reaches at least 0.90 and remains at or above that threshold for at least 80% of the following 100 steps.

## Running the REVO controller

The controller expects precomputed target artifacts and launches the sequence-search engine as a subprocess. A representative invocation is:

```bash
python code/revo_weighted_ensemble_v1.py \
  --strategy revo \
  --protein <protein_name> \
  --artifact-dir <target_artifact_directory> \
  --engine-path code/evolution_walk_universal_ignition_v1.py \
  --outdir <output_directory> \
  --seed <seed>
```

Use `python code/revo_weighted_ensemble_v1.py --help` for the full set of parameters.

## Scientific scope

This is target-embedding-guided sequence search. PRCS measures similarity to a target ESM representation and is not independent evidence of biological function. Some trajectories share ancestry through REVO cloning and should not be treated as statistically independent replicates. Structural or experimental validation is required before making functional claims.

## Provenance

The code in `code/` is the preserved pre-correction implementation with explanatory comments added for readability. No algorithmic corrections from the later audit are included in this snapshot.
