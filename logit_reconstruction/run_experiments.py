import os
import argparse
import matplotlib.pyplot as plt
import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from reconstruct import reconstruct


def run_experiment_1(base_output_dir: str):
    """Finlayson-style: scaling with samples and models using full logits."""

    models = ["gpt2", "gpt2-medium", "distilgpt2", "Qwen/Qwen2.5-0.5B"]
    sample_sizes = [100, 500, 800, 1000, 1200]
    dataset_name = "HuggingFaceFW/fineweb"
    subset = "sample-10BT"

    os.makedirs(base_output_dir, exist_ok=True)
    results = {}  # model -> {samples -> rms}

    print("=== Experiment 1: Scaling with Model Size & Samples ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    for model_name in models:
        print(f"\n=== Running experiments for model: {model_name} ===")
        results[model_name] = {"samples": [], "rms": [], "baseline": []}
        model_safe_name = model_name.replace("/", "_")

        # --- Load model once per model ---
        print(f"Loading model {model_name}...")
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model.eval()
        except Exception as e:
            print(f"Failed to load model {model_name}: {e}")
            continue

        # True output weights
        if hasattr(model, "lm_head"):
            W_true = model.lm_head.weight.detach().cpu().numpy()
        else:
            W_true = list(model.parameters())[-1].detach().cpu().numpy()

        # Load dataset (streaming)
        print(f"Loading dataset {dataset_name}...")
        ds = load_dataset(dataset_name, name=subset,
                          split="train", streaming=True)

        # Collect max samples once
        max_n = max(sample_sizes)
        print(f"Collecting {max_n} logits (max required)...")
        logits_list = []
        count = 0
        with torch.no_grad():
            for sample in tqdm(ds, total=max_n):
                text = sample["text"]
                if len(text) < 50:
                    continue

                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=64,
                ).to(device)
                outputs = model(inputs.input_ids)
                next_token_logits = outputs.logits[0, -1, :].cpu().numpy()

                logits_list.append(next_token_logits)
                count += 1
                if count >= max_n:
                    break

        L_full = np.array(logits_list)

        # Loop through sample sizes using slices
        for n_samples in sample_sizes:
            print(f"\n--- Samples: {n_samples} ---")
            L_slice = L_full[:n_samples]

            data_file = os.path.join(
                base_output_dir, f"logits_{model_safe_name}_{n_samples}.npz"
            )
            np.savez(data_file, logits=L_slice, W_true=W_true)

            metrics = reconstruct(
                data_file,
                base_output_dir,
                plot_prefix=f"{model_safe_name}_{n_samples}",
            )

            if metrics:
                results[model_name]["samples"].append(n_samples)
                results[model_name]["rms"].append(metrics["rms"])
                results[model_name]["baseline"].append(metrics["rms_baseline"])

        del model
        torch.cuda.empty_cache()

    # Save aggregated results
    with open(os.path.join(base_output_dir, "all_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Plot RMS vs Samples
    plt.figure(figsize=(10, 6))
    for model_name, data in results.items():
        if not data["samples"]:
            continue
        plt.plot(
            data["samples"],
            data["rms"],
            marker="o",
            label=f"{model_name} (Attack)",
        )
        plt.plot(
            data["samples"],
            data["baseline"],
            linestyle="--",
            alpha=0.5,
            label=f"{model_name} (Baseline)",
        )

    plt.title("Reconstruction Error vs Number of Samples")
    plt.xlabel("Number of Samples (N)")
    plt.ylabel("RMS Error")
    plt.yscale("log")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(base_output_dir, "rms_vs_samples.png"))
    print(
        f"\nSaved aggregated plot to {os.path.join(base_output_dir, 'rms_vs_samples.png')}"
    )


def run_experiment_2(base_output_dir: str):
    """Finlayson-style: effect of logit precision (rounding) across models."""

    # Reuse data from Experiment 1 for multiple models if available
    models = ["gpt2", "gpt2-medium", "distilgpt2"]
    precision_samples = 1000
    decimals_list = [None, 5, 3, 1]

    os.makedirs(base_output_dir, exist_ok=True)

    print("\n=== Experiment 2: Effect of API Precision (Rounding) ===")

    precision_results = {}

    for model_name in models:
        model_key = model_name.replace("/", "_")
        data_file = os.path.join(
            base_output_dir, f"logits_{model_key}_{precision_samples}.npz"
        )

        if not os.path.exists(data_file):
            print(
                f"Skipping {model_name}: data file {data_file} not found. Run Experiment 1 first."
            )
            continue

        precision_results[model_name] = {"decimals": [], "rms": []}

        for dec in decimals_list:
            label = "Full" if dec is None else f"{dec} decimals"
            print(f"\n[{model_name}] Precision: {label}")
            metrics = reconstruct(
                data_file,
                base_output_dir,
                plot_prefix=f"precision_{model_key}_{label}",
                decimals=dec,
            )
            if metrics:
                precision_results[model_name]["decimals"].append(
                    str(dec) if dec is not None else "Full"
                )
                precision_results[model_name]["rms"].append(metrics["rms"])

    # Plot: one bar group per model
    for model_name, data in precision_results.items():
        if not data["decimals"]:
            continue
        plt.figure(figsize=(8, 5))
        plt.bar(data["decimals"], data["rms"])
        plt.title(f"Reconstruction Error vs API Precision ({model_name})")
        plt.xlabel("Logit Precision (Decimal Places)")
        plt.ylabel("RMS Error")
        plt.yscale("log")
        plt.grid(True, axis="y")
        out_path = os.path.join(
            base_output_dir,
            f"rms_vs_precision_{model_name.replace('/', '_')}.png",
        )
        plt.savefig(out_path)
        print(f"Saved precision plot for {model_name} to {out_path}")


def run_experiment_3(base_output_dir: str):
    """Carlini-style: steal an internal linear layer via black-box access.

    We pick a transformer block MLP down-project layer (c_fc) and treat
    it as a black-box linear map W: h -> W h for randomly sampled h from
    the model's own hidden states on real data.

    We then reconstruct W via linear regression, under different output
    visibility patterns (k):

      - k = None   : full output vector (idealized attacker)
      - k = 256    : only top-256 coordinates by magnitude
      - k = 64     : only top-64 coordinates (stronger restriction)

    This mirrors the "stealing part of a production model" idea from
    Carlini et al. on a simplified, local setting.
    """

    models = ["gpt2", "gpt2-medium", "distilgpt2", "Qwen/Qwen2.5-0.5B"]
    n_samples = 2000
    ks = [None, 256, 64]
    # Use the dataset referenced in the report for this experiment
    dataset_name = "Muennighoff/natural-instructions"
    subset = None

    os.makedirs(base_output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=== Experiment 3: Carlini-style partial layer stealing ===")
    print(f"Using device: {device}")

    results = {}

    for model_name in models:
        print(f"\n=== Model (Carlini-style): {model_name} ===")
        model_key = model_name.replace("/", "_")
        results[model_name] = {"k": [], "rms": []}

        # Load model + tokenizer
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model.eval()
        except Exception as e:
            print(f"Failed to load model {model_name}: {e}")
            continue

        # Identify a linear sub-layer to steal.
        # For GPT-2-like models with a transformer.h stack, use first block's c_fc.
        linear_layer = None
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            try:
                linear_layer = model.transformer.h[0].mlp.c_fc
            except Exception:
                linear_layer = None

        # Fallback: last linear layer in the model parameters
        if linear_layer is None:
            print(
                "Could not find transformer.h[0].mlp.c_fc; using last linear layer instead.")
            for m in reversed(list(model.modules())):
                if isinstance(m, torch.nn.Linear):
                    linear_layer = m
                    break

        if linear_layer is None:
            print("No linear layer found to steal; skipping model.")
            del model
            torch.cuda.empty_cache()
            continue

        # True weights of the target layer. For standard nn.Linear this is
        # (out_dim, in_dim); for GPT-2 Conv1D-style layers, it's often stored
        # transposed. We'll infer the correct orientation from H and Y shapes
        # later when computing RMS, so here we just keep the raw tensor.
        W_true_raw = linear_layer.weight.detach().cpu().numpy()

        # Collect (h, W h) pairs via black-box access on real data
        print(f"Collecting {n_samples} hidden states and layer outputs...")
        if subset is not None:
            ds = load_dataset(dataset_name, name=subset,
                              split="train", streaming=True)
        else:
            ds = load_dataset(dataset_name, split="train", streaming=True)

        H_list = []
        Y_full_list = []
        count = 0

        # We will hook the layer input and output to capture (h, y).
        stored_pair = []

        def hook_fn(module, inp, out):
            # inp is a tuple; take the first element
            h = inp[0].detach()
            y = out.detach()
            # Flatten batch and sequence into one dimension
            h_flat = h.reshape(-1, h.shape[-1])
            y_flat = y.reshape(-1, y.shape[-1])
            stored_pair.append((h_flat.cpu(), y_flat.cpu()))

        hook = linear_layer.register_forward_hook(hook_fn)

        with torch.no_grad():
            for sample in tqdm(ds, total=n_samples):
                # natural-instructions has nested fields; fall back gracefully
                if "text" in sample:
                    text = sample["text"]
                elif "instruction" in sample and "input" in sample:
                    text = str(sample["instruction"]) + \
                        "\n" + str(sample["input"])
                else:
                    # Fallback: join all string-like fields
                    text_fields = [
                        str(v)
                        for v in sample.values()
                        if isinstance(v, str)
                    ]
                    text = " \n ".join(text_fields)
                if len(text) < 50:
                    continue

                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=64,
                ).to(device)

                stored_pair.clear()
                _ = model(**inputs)

                if not stored_pair:
                    continue

                # shapes: (B*T, d_in), (B*T, d_out)
                h_batch, y_batch = stored_pair[0]

                H_list.append(h_batch)
                Y_full_list.append(y_batch)
                count += h_batch.shape[0]

                if count >= n_samples:
                    break

        hook.remove()

        if not H_list:
            print("No hidden states collected; skipping model.")
            del model
            torch.cuda.empty_cache()
            continue

        H = torch.cat(H_list, dim=0).numpy()[:n_samples]
        Y_full = torch.cat(Y_full_list, dim=0).numpy()[:n_samples]

        # Derive canonical shapes
        d_in = H.shape[1]
        d_out = Y_full.shape[1]

        # For each k, restrict outputs and reconstruct W via regression
        for k in ks:
            print(
                f"  Solving regression for k={k if k is not None else 'Full'}...")

            if k is None:
                # Full output, standard least squares: Y ≈ H W^T
                # We solve for W^T using normal equations.
                # W_hat^T = (H^T H)^-1 H^T Y
                HTH = H.T @ H
                HTY = H.T @ Y_full
                W_hat_T = np.linalg.lstsq(HTH, HTY, rcond=None)[0]
                W_hat = W_hat_T.T
            else:
                # Only top-k coordinates of Y_full by magnitude, per row
                Y_k = np.zeros_like(Y_full)
                # indices of top-k per sample
                topk_idx = np.argpartition(-np.abs(Y_full), k, axis=1)[:, :k]
                rows = np.arange(Y_full.shape[0])[:, None]
                Y_k[rows, topk_idx] = Y_full[rows, topk_idx]

                HTH = H.T @ H
                HTYk = H.T @ Y_k
                W_hat_T = np.linalg.lstsq(HTH, HTYk, rcond=None)[0]
                W_hat = W_hat_T.T

            # Compute RMS error between W_hat and W_true, being robust
            # to possible transposes/reshapes in how W_true is stored.
            W_true = W_true_raw
            if W_true.shape != W_hat.shape:
                # Try transpose
                if W_true.T.shape == W_hat.shape:
                    W_true = W_true.T
                # Try reshaping to match (d_out, d_in)
                elif W_true.size == d_in * d_out:
                    W_true = W_true.reshape(d_out, d_in)
                else:
                    print(
                        f"Warning: cannot align W_true shape {W_true.shape} with W_hat {W_hat.shape}; skipping RMS for this model/k."
                    )
                    continue

            diff = W_hat - W_true
            rms = float(np.sqrt(np.mean(diff ** 2)))
            results[model_name]["k"].append("Full" if k is None else str(k))
            results[model_name]["rms"].append(rms)
            print(f"    RMS(W_hat, W_true) = {rms:.4e}")

        del model
        torch.cuda.empty_cache()

    # Save results and plots
    out_json = os.path.join(base_output_dir, "carlini_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved Carlini-style results to {out_json}")

    for model_name, data in results.items():
        if not data["k"]:
            continue
        plt.figure(figsize=(8, 5))
        plt.bar(data["k"], data["rms"])
        plt.title(f"Carlini-style Layer Stealing RMS vs k ({model_name})")
        plt.xlabel("Visible output coordinates (k)")
        plt.ylabel("RMS(W_hat, W_true)")
        plt.yscale("log")
        plt.grid(True, axis="y")
        out_path = os.path.join(
            base_output_dir,
            f"carlini_rms_vs_k_{model_name.replace('/', '_')}.png",
        )
        plt.savefig(out_path)
        print(f"Saved Carlini-style plot for {model_name} to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run logit reconstruction experiments (Finlayson / Carlini style)."
    )
    parser.add_argument(
        "--experiment",
        type=int,
        choices=[1, 2, 3],
        nargs="+",
        default=[1, 2],
        help="Which experiments to run: 1=Scaling, 2=Precision, 3=Carlini-style (TODO)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="logit_reconstruction/experiments",
        help="Base output directory for experiment artifacts.",
    )

    args = parser.parse_args()

    if 1 in args.experiment:
        run_experiment_1(args.output_dir)

    if 2 in args.experiment:
        run_experiment_2(args.output_dir)

    if 3 in args.experiment:
        run_experiment_3(args.output_dir)


if __name__ == "__main__":
    main()
