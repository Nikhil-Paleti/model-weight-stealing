uv run python scripts/classy.py \
  --dataset yelp_review_full \
  --splits train test \
  --model nlptown/bert-base-multilingual-uncased-sentiment \
  --batch_size 32 \
  --max_length 256 \
  --push_repo LightFury9/yelp-5star-probs