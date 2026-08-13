# Quick Start

This folder contains the final corrected adaptive independent-walker algorithm.

## What the four Python files do

- `prepare_targets.py` reads the target FASTA files and creates their ESM2 target embeddings. Run this once before evolution.
- `run_final_adaptive.py` is the main file you run. It creates random starting sequences, launches the independent walkers, and saves their results.
- `adaptive_walk.py` runs the evolutionary loop for one walker. It proposes mutations, scores them, adapts the mutation size, and updates coupling information.
- `corrected_adaptive_engine.py` contains the shared calculations and settings, including ESM loading, PRCS, epistasis, mutation generation, temperature, coupling groups, and checkpoints.

## Recommended folder structure

Keep the FASTA files, generated target artifacts, and results beside the four Python files:

```text
refactored_final_code/
|-- adaptive_walk.py
|-- corrected_adaptive_engine.py
|-- prepare_targets.py
|-- run_final_adaptive.py
|-- QUICK_START.md
|-- target_fastas/
|   |-- gapdh.fasta
|   |-- tp53.fasta
|   `-- human_insulin.fasta
|-- target_artifacts/
|   |-- gapdh.target_embedding.pt
|   |-- tp53.target_embedding.pt
|   |-- human_insulin.target_embedding.pt
|   `-- artifact_manifest.tsv
`-- Results/
    |-- gapdh_seed81001/
    |-- tp53_seed81001/
    `-- human_insulin_seed81001/
```

Create `target_fastas/` yourself and place the target FASTA files there. The target-artifact and result directories are created by the scripts.

The FASTA directory can be stored somewhere else, but keeping this layout makes the commands easier to follow.

## Install the dependencies

The environment needs PyTorch, FAIR ESM, NumPy, pandas, and SciPy:

```bash
pip install torch fair-esm numpy pandas scipy
```

A GPU is strongly recommended.

## Prepare the targets

First, enter this folder:

```bash
cd refactored_final_code
```

Prepare one or more target embeddings:

```bash
python prepare_targets.py \
  --fasta-dir target_fastas \
  --outdir target_artifacts \
  --proteins gapdh,tp53,human_insulin \
  --device cuda
```

The protein names must match the FASTA filenames exactly. For example, `--proteins gapdh` expects `target_fastas/gapdh.fasta` and produces `target_artifacts/gapdh.target_embedding.pt`.

## Run evolution

Run one protein at a time:

```bash
python run_final_adaptive.py \
  --protein gapdh \
  --artifact-dir target_artifacts \
  --outdir Results/gapdh_seed81001 \
  --seed 81001 \
  --walkers 4 \
  --total-steps 3500 \
  --device cuda
```

Each walker starts from a different random sequence derived from the given seed.

## Results

Each run directory contains:

- `walker_trajectories/`: the saved trajectory for each walker
- `final_sequences/`: the best and final FASTA sequences
- `checkpoints/`: saved progress for resuming interrupted runs
- `initial_walkers.csv`: the random starting sequences
- `final_walker_summary.csv`: the main result summary
- `run_metadata.json`: the settings used for the run

If a run stops, use the same command again with the same seed and output directory. The walkers will continue from their checkpoints.
