# Knockoff (Classification Stealing)

This folder contains our **classification model stealing** experiments.

## Final Project (Notebook)

The final-project knockoff pipeline is implemented in:

- `Teacher_model_accuracy.ipynb`

It performs:

1. fine-tuning a **DistilBERT teacher** on Yelp (balanced 150k subset),
2. simulating black-box API access by querying **50k** examples (saving logits/probs/preds),
3. training students of increasing capacity (TF--IDF linear, TF--IDF MLP, DistilBERT student),
4. reporting **fidelity** (Acc vs Teacher) and **task accuracy** (Acc vs True).

### How to run

Option A (recommended): run in Colab (GPU) and execute all cells.

Option B (local): open in Jupyter and run all cells. A GPU is strongly recommended.

Notes for reproducibility:

- The notebook contains a few Google Drive paths (Colab-style). If you run locally, change those paths to a local folder (e.g., `./outputs/knockoff_notebook/`).

### Expected headline results

From our reported runs:

- Teacher test accuracy: **0.654**
- Best student fidelity (Acc vs Teacher): **0.8063** (DistilBERT student via KD)

## Legacy Baseline (Deterministic sklearn scripts)

We keep the earlier deterministic baselines used in the midterm update. These scripts assume a dataset split that contains teacher outputs (e.g., `LightFury9/yelp-5star-probs`).

### Soft-label stealing (KD)

```bash
uv run python knockoff/steal_kd_sklearn_deterministic.py \
  --dataset_hub LightFury9/yelp-5star-probs \
  --split test \
  --val_size 0.2 \
  --random_state 42 \
  --max_features 200000 \
  --save_model student_ridge.joblib \
  --save_manifest manifest.json
```

Outputs are written to `outputs/`:

- `outputs/student_ridge.joblib`
- `outputs/manifest.json`

### Hard-label stealing

```bash
uv run python knockoff/steal_labels_sklearn_deterministic.py \
  --dataset_hub LightFury9/yelp-5star-probs \
  --split test \
  --val_size 0.2 \
  --random_state 42 \
  --max_features 200000 \
  --model_type logreg \
  --C 1.0 \
  --save_model student_labels_logreg.joblib \
  --save_manifest manifest_labels.json
```

Outputs are written to `outputs/`:

- `outputs/student_labels_logreg.joblib`
- `outputs/manifest_labels.json`
