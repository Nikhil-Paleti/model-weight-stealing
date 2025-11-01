#!/usr/bin/env python3
# Generic multi-class text classification scorer for any HF dataset/model.
# - Reads dataset + model from CLI args.
# - Adds: bert_logits (List[float]), bert_probs (List[float]), bert_pred (int), bert_label (str).
# - Saves with save_to_disk and (optionally) pushes to the Hub if --push_repo is provided.
# - Auth token is taken from environment ONLY (e.g., HUGGINGFACE_HUB_TOKEN or HF_TOKEN).
#
# Example:
#   python classy.py \
#     --dataset yelp_review_full --splits train test \
#     --model nlptown/bert-base-multilingual-uncased-sentiment \
#     --batch_size 32 --max_length 256 \
#     --save_dir out-yelp \
#     --push_repo YOUR_USERNAME/yelp-bert-5class
#
# Notes:
# - If you put your token in a .env file, set HUGGINGFACE_HUB_TOKEN=hf_xxx (or HF_TOKEN=hf_xxx)
#   and install python-dotenv; otherwise export it in your shell. We do not read any other
#   settings from env—only the token is used implicitly by the hub.

import argparse
import os
from typing import Dict

import torch
import torch.nn.functional as F
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Optional: load .env only to pick up tokens if present.
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass


def parse_args():
    p = argparse.ArgumentParser(description="Score HF dataset with HF model; write probs & optionally push.")

    p.add_argument("--dataset", type=str, required=True, help="HF dataset id (e.g., yelp_review_full)")
    p.add_argument("--model", type=str, required=True, help="HF model id (num_labels >= 2)")
    p.add_argument("--revision", type=str, default=None, help="Optional model revision/branch")

    # Data handling
    p.add_argument("--splits", nargs="+", default=["train"], help="Dataset splits to process, e.g. train test")
    p.add_argument("--column", type=str, default="text", help="Text column name")
    p.add_argument("--batch_size", type=int, default=32, help="Batch size for forward pass")
    p.add_argument("--max_length", type=int, default=256, help="Tokenizer max_length (trunc/pad)")
    p.add_argument("--device", type=int, default=None, help="GPU id (0,1,...) or -1 for CPU; default auto")
    p.add_argument("--save_dir", type=str, default=None, help="Output directory for save_to_disk")

    # Optional push target (token must be in env: HUGGINGFACE_HUB_TOKEN or HF_TOKEN)
    p.add_argument("--push_repo", type=str, default=None, help="HF repo to push to, e.g. username/my-dataset")
    return p.parse_args()


def main():
    args = parse_args()

    # Device
    if args.device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu" if args.device < 0 else f"cuda:{args.device}")

    print(f"[config] dataset={args.dataset}  splits={args.splits}")
    print(f"[config] model={args.model}  revision={args.revision or '(default)'}")
    print(f"[config] device={device}")
    if args.push_repo:
        print(f"[config] push_repo={args.push_repo} (token will be read from env)")
    else:
        print("[config] push_repo=unset (skip push)")

    # Load dataset splits
    ds_dict = {
        split: load_dataset(args.dataset, split=split) for split in args.splits
    }

    # Load model/tokenizer
    if args.revision:
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
        model = AutoModelForSequenceClassification.from_pretrained(args.model, revision=args.revision)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForSequenceClassification.from_pretrained(args.model)

    num_labels = int(getattr(model.config, "num_labels", 0))
    if num_labels < 2:
        raise ValueError(f"Model must have num_labels >= 2; got {num_labels}")

    # Friendly labels from config if present; else class_0..class_{n-1}
    id2label = getattr(model.config, "id2label", None)
    if not isinstance(id2label, dict) or len(id2label) != num_labels:
        id2label = {i: f"class_{i}" for i in range(num_labels)}

    model.to(device).eval()

    def tokenize_fn(batch):
        return tokenizer(
            batch[args.column],
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
        )

    @torch.no_grad()
    def infer_fn(batch):
        input_ids = torch.tensor(batch["input_ids"], device=device)
        attention_mask = torch.tensor(batch["attention_mask"], device=device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [B, C]
        probs = F.softmax(logits, dim=-1)

        logits_list = logits.cpu().tolist()
        probs_list = probs.cpu().tolist()
        pred_ids = torch.argmax(probs, dim=-1).cpu().tolist()
        pred_labels = [id2label[i] for i in pred_ids]

        return {
            "model_logits": logits_list,   # raw scores before softmax
            "model_probs": probs_list,     # probability distribution over classes
            "model_pred": pred_ids,        # predicted class IDs
            "model_label": pred_labels,    # human-readable label
        }

    updated = {}
    for split, ds in ds_dict.items():
        if args.column not in ds.column_names:
            raise ValueError(f"Column '{args.column}' not in {split} columns: {ds.column_names}")

        print(f"[{split}] tokenizing...")
        ds_tok = ds.map(tokenize_fn, batched=True)

        print(f"[{split}] inference (batch_size={args.batch_size})...")
        ds_out = ds_tok.map(infer_fn, batched=True, batch_size=args.batch_size, desc=f"Inference on {split}")

        keep = [args.column]
        if "label" in ds_out.column_names:
            keep.append("label")
        keep += ["model_logits", "model_probs", "model_pred", "model_label"]
        drop = [c for c in ds_out.column_names if c not in keep]
        ds_out = ds_out.remove_columns(drop)

        print(f"[{split}] sample:", {k: ds_out[0][k] for k in keep if len(ds_out) > 0})
        updated[split] = ds_out

    dsd = DatasetDict(updated)
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        print(f"[save] save_to_disk → {args.save_dir}")
        dsd.save_to_disk(args.save_dir)
        print("[save] done.")

    if args.push_repo:
        print(f"[push] pushing to {args.push_repo} ...")
        # Token is pulled automatically from env (HUGGINGFACE_HUB_TOKEN/HF_TOKEN)
        dsd.push_to_hub(args.push_repo)
        print("[push] done.")


if __name__ == "__main__":
    main()