# Model Weight Stealing

This repository contains reproducible experiments on **model extraction / model stealing** for NLP systems, developed for **DSC 261**.

The project includes:

- Knockoff-style functionality stealing (classification)
- DisGUIDE-style disagreement-guided active querying (classification)
- Logit-based reconstruction (causal LMs), inspired by Finlayson et al. and Carlini et al.

## Environment

**Python:** 3.12+ (see `pyproject.toml`).

### Install (recommended: `uv`)

```bash
uv sync
```

### Install (pip fallback)

```bash
python -m pip install -U pip
python -m pip install -U torch transformers datasets scikit-learn joblib accelerate huggingface-hub
```

## Reproducing Results (Quick Map)

### 1) Knockoff / classification stealing (final project)

Final-project knockoff experiments are implemented in the notebook:

- `knockoff/Teacher_model_accuracy.ipynb`

It trains a DistilBERT teacher on Yelp, simulates a **50k-query** extraction dataset, and evaluates students of increasing capacity (TF-IDF linear, TF-IDF MLP, DistilBERT student).

Expected headline metrics (from our runs):

- Teacher test accuracy: **0.654**
- Best student fidelity (Acc vs Teacher): **0.8063** (DistilBERT KD)

See `knockoff/README.md` for exact steps and the full table.

### 2) DisGUIDE / active querying (live teacher)

Implementation:

- `disguide/disguide.py`

Expected headline metric (1,000-query budget):

- Student accuracy vs teacher: **0.825**

See `disguide/README.md`.

### 3) Logit-based reconstruction (causal LMs)

Implementation:

- `logit_reconstruction/run_experiments.py`

Outputs:

- `logit_reconstruction/experiments/all_results.json` (Finlayson-style)
- `logit_reconstruction/experiments/carlini_results.json` (Carlini-style)
- `logit_reconstruction/experiments/*_metrics.txt` (per-run summaries)

See `logit_reconstruction/README.md`.

## Legacy baseline (midterm)

The earlier baseline used a pretrained teacher (`nlptown/bert-base-multilingual-uncased-sentiment`) and a pushed dataset (`LightFury9/yelp-5star-probs`) with deterministic sklearn scripts in `knockoff/`. These scripts are still available, but the final project report uses `knockoff/Teacher_model_accuracy.ipynb`.

## Notes

- Some experiments benefit significantly from a GPU (teacher/LM inference).
