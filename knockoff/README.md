```bash
uv run knockoff/steal_kd_sklearn_deterministic.py \
  --dataset_hub LightFury9/yelp-5star-probs \
  --split test \
  --val_size 0.2 \
  --random_state 42 \
  --max_features 200000 \
  --save_model student_ridge.joblib \
  --save_manifest manifest.json
  ```


```bash
uv run knockoff/steal_labels_sklearn_deterministic.py \
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