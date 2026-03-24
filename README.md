# Centrality-Based Pruning for Efficient Echo State Networks

Codebase for the preprint:
**Centrality-Based Pruning for Efficient Echo State Networks**  
Sudip Laudari (Independent Researcher)
arXiv: https://arxiv.org/abs/2603.20684

## Overview

This repository implements a centrality-driven pruning framework for Echo State Networks (ESNs), where the reservoir is interpreted as a weighted directed graph and less important nodes are iteratively removed.

The pipeline supports:
- ESN reservoir initialization with configurable spectral radius
- Graph-centrality and signed-connectivity pruning scores
- Dynamic train split setup (`10% init / 80% train / 10% test`)
- Multi-reservoir experiments (`[100, 200, ..., 900]` by default)
- Aggregated CSV result export and plot generation
- Structured logging to `log/log.log`

## Core Files

- `train.py`  
  Main experiment runner with all top-level variables and runtime controls.
- `core_logic.py`  
  ESN dynamics, pruning loop, score definitions, and ESN property logging.
- `betweenness_model.py`  
  Betweenness centrality implementation.
- `closeness_model.py`  
  Closeness centrality implementation.

## Model Set

The code supports 10 pruning models:
- Degree-based: `indeg`, `outdeg`, `total`
- Centrality-based: `between`, `close`
- Paper models: `c1`, `c2`, `c3`
- Additional variants: `c4`, `c5`

Aliases:
- `bet`, `cb` -> `between`
- `closeness`, `cc` -> `close`

## Data Format

Input supports both:
- `.npy`
- `.csv`

Training expects a 1D target series (the loader reduces data to one signal and normalizes it to `[0, 1]`).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Example run (recommended defaults):

```bash
python3 train.py \
  --data-path "data/electric.csv" \
  --models "indeg,outdeg,total,between,close,c1,c2,c3,c4,c5" \
  --reservoir-sizes "100,200,300,400,500,600,700,800,900" \
  --prune-percent 0.40 \
  --runs 10 \
  --seed-start 0 \
  --tau 17 \
  --spectral-radius 0.95 \
  --output-dir outputs \
  --log-path log/log.log
```

## Outputs

Each execution creates a unique run directory:
- `outputs/run_YYYYMMDD_HHMMSS/`

Inside each run directory:
- `all_results.csv` (aggregated results across models/seeds/reservoir sizes)
- Error and prediction plots per run

Logging:
- `log/log.log`

## Reproducibility Notes

- Seed range is controlled in `train.py` (`--seed-start`, `--runs`).
- Effective ESN configuration is centralized in `train.py` (`BASE_ESN_CONFIG`).
- Split ratios are configurable via `--init-ratio`, `--train-ratio`, `--test-ratio`.

## Citation

If you use this code, please cite the corresponding preprint:
**Centrality-Based Pruning for Efficient Echo State Networks** (March 24, 2026).
**https://arxiv.org/abs/2603.20684**
