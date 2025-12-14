# Scripts

This folder contains utility scripts used to prepare datasets and run teacher inference.

## `classy.py`

Runs a Hugging Face text-classification model on a Hugging Face dataset and appends teacher outputs.

### What it produces

For each example, it adds:

- `bert_logits`: raw logits from the teacher
- `bert_probs`: softmax probabilities
- `bert_pred`: top-1 prediction

Depending on flags, it can save locally and/or push the augmented dataset to the Hugging Face Hub.

### Example (baseline teacher inference)

```bash
uv run python scripts/classy.py \
  --dataset yelp_review_full \
  --splits train test \
  --model nlptown/bert-base-multilingual-uncased-sentiment \
  --batch_size 32 \
  --max_length 256 \
  --push_repo LightFury9/yelp-5star-probs
```

Notes:

- Pushing requires `HUGGINGFACE_HUB_TOKEN`.
- A GPU is recommended for speed but not required.
