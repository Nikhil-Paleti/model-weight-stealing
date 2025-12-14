# Logit-Based Reconstruction Experiments

This directory contains scripts and results for logit-based model stealing experiments inspired by:

- Finlayson et al. (COLM 2024): _Logits of API-Protected LLMs Leak Proprietary Information_.
- Carlini et al. (ICML 2024): _Stealing Part of a Production Language Model_.

We work with local copies of several open models and simulate different attacker capabilities.

## 1. Finlayson-Style: Scaling with Samples and Model Size (Experiment 1)

**Goal.** Show that next-token logits are sufficient to reconstruct the final projection matrix, and that reconstruction quality improves with more queries and across models.

**Setup.**

- Models: `gpt2`, `gpt2-medium`, `distilgpt2`, `Qwen/Qwen2.5-0.5B`.
- Dataset: `HuggingFaceFW/fineweb` (`sample-10BT`).
- Samples per model: `N ∈ {100, 500, 800, 1000, 1200}`.
- For each (model, N): collect logits and true `W` into `experiments/logits_<model>_<N>.npz`, then run SVD-based reconstruction.

**How to run.**

```bash
uv run python logit_reconstruction/run_experiments.py --experiment 1
```

**Key outputs (in `logit_reconstruction/experiments/`):**

- `all_results.json`: aggregated RMS and baseline per model and sample size.
- `rms_vs_samples.png`: log-scale plot of reconstruction error vs number of samples.
- `<model>_<N>_metrics.txt` and `<model>_<N>_spectrum.png`: detailed error metrics and singular value spectra.

**Qualitative result.**

For all four models, RMS reconstruction error decreases sharply as N grows and eventually becomes several orders of magnitude smaller than a random-baseline RMS. For example, for GPT-2 the RMS drops from ≈0.11 at N=100 to ≈7.7×10⁻⁴ at N=1200, while the baseline remains ≈0.14. This empirically confirms that logits alone leak the output projection.

## 2. Finlayson-Style: Effect of Logit Precision / Rounding (Experiment 2)

**Goal.** Test how robust the attack is when the API rounds logits to a limited decimal precision.

**Setup.**

- Models: `gpt2`, `gpt2-medium`, `distilgpt2`.
- Dataset and N: reuse the N=1000 `.npz` files from Experiment 1.
- Simulated API precisions: `Full`, `5`, `3`, `1` decimal places.

**How to run.**

```bash
uv run python logit_reconstruction/run_experiments.py --experiment 2
```

**Key outputs:**

- `rms_vs_precision_gpt2.png`, `rms_vs_precision_gpt2-medium.png`, `rms_vs_precision_distilgpt2.png`.
- `precision_*_metrics.txt`, `precision_*_spectrum.png` per model/precision.

**Qualitative result.**

Rounding logits to 5 or 3 decimal places has little effect on reconstruction quality; even with only 1 decimal place the attack still substantially outperforms a random baseline. Simple numeric rounding is therefore not an effective defense.

## 3. Carlini-Style: Stealing an Internal Linear Layer (Experiment 3)

**Goal.** Emulate "stealing part of a production model" by reconstructing an internal linear layer using black-box access to its inputs and (possibly restricted) outputs.

**Setup.**

- Models: `gpt2`, `gpt2-medium`, `distilgpt2`, `Qwen/Qwen2.5-0.5B`.
- Dataset: `Muennighoff/natural-instructions` (as referenced in the report).
- Target layer:
  - Prefer `transformer.h[0].mlp.c_fc` for GPT-2-style models.
  - Fallback to the last `nn.Linear` for models without that structure (e.g., Qwen).
- We collect ≈2000 hidden states at the input of that layer and the corresponding outputs, then solve a linear regression to estimate `W_hat` under three visibility patterns:
  - `k = Full`: full output vector.
  - `k = 256`: only top-256 coordinates by magnitude are visible.
  - `k = 64`: only top-64 coordinates are visible.

**How to run.**

```bash
uv run python logit_reconstruction/run_experiments.py --experiment 3
```

**Key outputs:**

- `carlini_results.json`: RMS between reconstructed and true weights for each model and k.
- `carlini_rms_vs_k_<model>.png`: RMS vs visible output coordinates for each model.

**Qualitative result.**

For GPT-2 family models, reducing k (revealing fewer output coordinates) increases reconstruction error, showing that API constraints on what part of the layer output is exposed make layer stealing harder but still feasible. For Qwen-0.5B, the RMS profile behaves differently, suggesting that model- and layer-specific factors significantly influence how much information partial outputs leak.

## Running All Experiments

You can run any combination of experiments from the repository root:

```bash
# Experiments 1 and 2 (Finlayson-style)
uv run python logit_reconstruction/run_experiments.py

# Only precision experiment
uv run python logit_reconstruction/run_experiments.py --experiment 2

# Carlini-style internal-layer stealing
uv run python logit_reconstruction/run_experiments.py --experiment 3
```

## Tabular Results (for reports)

All numeric results needed for tables are written to:

- `logit_reconstruction/experiments/all_results.json` (Experiment 1)
- `logit_reconstruction/experiments/carlini_results.json` (Experiment 3)
- `logit_reconstruction/experiments/*_metrics.txt` (per-model/per-N summaries)
