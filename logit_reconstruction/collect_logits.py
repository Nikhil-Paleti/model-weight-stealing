import argparse
import os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm


def collect(model_name, dataset_name, subset, num_samples, output_path):
    print(f"Loading model: {model_name}")
    # Use CPU or GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()

    # Get true weights for verification later
    # GPT-2 lm_head is usually tied to wte, but we can access it directly.
    # For gpt2, model.lm_head.weight is the projection matrix (Vocab, Hidden)
    if hasattr(model, "lm_head"):
        W_true = model.lm_head.weight.detach().cpu().numpy()  # Shape: (V, d) or (V, H)
    else:
        # Fallback for other architectures if needed, though most CausalLM have lm_head
        print("Warning: Could not find lm_head. Using last layer weights if possible.")
        W_true = list(model.parameters())[-1].detach().cpu().numpy()

    print(f"True weights shape: {W_true.shape}")

    print(f"Loading dataset: {dataset_name}")
    # Streaming to avoid full download
    ds = load_dataset(dataset_name, name=subset, split="train", streaming=True)

    logits_list = []

    print(f"Collecting {num_samples} logits...")
    count = 0
    with torch.no_grad():
        for sample in tqdm(ds, total=num_samples):
            text = sample['text']
            if len(text) < 50:
                continue  # Skip very short texts

            # We just need a prompt to get the NEXT token logits.
            # We can take a chunk of text.
            inputs = tokenizer(text, return_tensors="pt",
                               truncation=True, max_length=64).to(device)
            input_ids = inputs.input_ids

            # Forward pass
            outputs = model(input_ids)
            # Get logits of the LAST token in the sequence (prediction for next token)
            next_token_logits = outputs.logits[0, -1, :].cpu().numpy()

            logits_list.append(next_token_logits)
            count += 1

            if count >= num_samples:
                break

    L = np.array(logits_list)
    print(f"Collected logits matrix shape: {L.shape}")

    np.savez(output_path, logits=L, W_true=W_true)
    print(f"Saved logits and true weights to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect logits from GPT-2 for reconstruction attack.")
    parser.add_argument("--model", type=str,
                        default="gpt2", help="Target model")
    parser.add_argument("--dataset", type=str,
                        default="HuggingFaceFW/fineweb", help="Dataset for prompts")
    parser.add_argument("--subset", type=str,
                        default="sample-10BT", help="Dataset subset")
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="Number of logit vectors to collect")
    parser.add_argument("--output_dir", type=str,
                        default="logit_reconstruction/data", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "logits_data.npz")

    collect(args.model, args.dataset, args.subset,
            args.num_samples, output_path)


if __name__ == "__main__":
    main()
