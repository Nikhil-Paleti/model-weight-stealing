#!/usr/bin/env python3
"""
Deterministic model stealing (KD) with scikit-learn.

- Loads a HF dataset split that contains teacher outputs:
    - 'model_probs' (preferred), or
    - 'model_logits' (we softmax once to get probs)
- Builds a simple student: TF-IDF (1–2 grams) -> MultiOutput Ridge
- 80/20 split (configurable), stratified by teacher top-1
- Fully reproducible: seeds + single-thread BLAS + stable vectorizer settings
- Saves:
    * student .joblib
    * manifest.json (versions, seeds, dataset fingerprint, split indices, etc.)

Example:
python steal_kd_sklearn_deterministic.py \
  --dataset_hub LightFury9/yelp-5star-probs \
  --split test \
  --val_size 0.2 \
  --random_state 42 \
  --save_model student_ridge.joblib \
  --save_manifest manifest.json
"""

import argparse, json, os, platform, random, sys
from typing import Dict, Any
import numpy as np

# ==== FORCE SINGLE-THREAD BLAS/NP (before importing sklearn) ====
os.environ.setdefault("PYTHONHASHSEED", "0")
for var in [
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"
]:
    os.environ.setdefault(var, "1")

import sklearn
from datasets import load_dataset, load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from joblib import dump


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / (e.sum(axis=-1, keepdims=True) + 1e-12)


def kl_div(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """KL(P||Q) averaged over rows; both arrays shape [N, C]."""
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.mean(np.sum(p * (np.log(p) - np.log(q)), axis=1)))


def load_split(dataset_hub: str | None, dataset_disk: str | None, split: str):
    if dataset_disk:
        dsd = load_from_disk(dataset_disk)
        return dsd[split], getattr(dsd[split], "_fingerprint", None)
    else:
        dsd = load_dataset(dataset_hub)
        return dsd[split], getattr(dsd[split], "_fingerprint", None)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_hub", type=str, default=None, help="HF repo id (e.g., LightFury9/yelp-5star-probs)")
    ap.add_argument("--dataset_disk", type=str, default=None, help="Local path from save_to_disk()")
    ap.add_argument("--split", type=str, default="test", help="Which split to use (we do 80/20 inside)")
    ap.add_argument("--text_col", type=str, default="text")
    ap.add_argument("--probs_col", type=str, default="model_probs")
    ap.add_argument("--logits_col", type=str, default="model_logits")
    ap.add_argument("--val_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--max_features", type=int, default=200_000)
    ap.add_argument("--save_model", type=str, default="student_ridge.joblib")
    ap.add_argument("--save_manifest", type=str, default="manifest.json")
    return ap.parse_args()


def main():
    args = parse_args()
    set_all_seeds(args.random_state)

    if not args.dataset_hub and not args.dataset_disk:
        raise ValueError("Provide --dataset_hub or --dataset_disk")

    ds, ds_fingerprint = load_split(args.dataset_hub, args.dataset_disk, args.split)

    if args.text_col not in ds.column_names:
        raise ValueError(f"Text column '{args.text_col}' not found; columns: {ds.column_names}")

    # Targets: teacher probs
    if args.probs_col in ds.column_names:
        Y = np.array(ds[args.probs_col], dtype=np.float64)
    elif args.logits_col in ds.column_names:
        logits = np.array(ds[args.logits_col], dtype=np.float64)
        Y = softmax_np(logits)
    else:
        raise ValueError("Need either 'model_probs' or 'model_logits' in the dataset.")

    texts = ds[args.text_col]
    true_labels = np.array(ds["label"], dtype=int) if "label" in ds.column_names else None

    # Stable index array to record the split
    idx = np.arange(len(texts), dtype=np.int64)
    teacher_top1 = Y.argmax(axis=1)

    X_train, X_val, Y_train, Y_val, top1_train, top1_val, idx_train, idx_val = train_test_split(
        texts, Y, teacher_top1, idx,
        test_size=args.val_size, random_state=args.random_state, stratify=teacher_top1
    )

    if true_labels is not None:
        # align true labels with the same split deterministically
        _, ytrue_val = train_test_split(
            true_labels, test_size=args.val_size, random_state=args.random_state, stratify=teacher_top1
        )
    else:
        ytrue_val = None

    # Deterministic TF-IDF + Ridge:
    # - lowercase True, strip_accents='ascii', fixed token_pattern
    # - dtype float64 for numerical stability
    # - no hashing; vocabulary built in deterministic order (single-thread)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=args.max_features,
            min_df=2,
            lowercase=True,
            strip_accents="ascii",
            token_pattern=r"(?u)\b\w\w+\b",
            norm="l2",
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=False,
            dtype=np.float64,
        )),
        ("reg", MultiOutputRegressor(Ridge(alpha=1.0, random_state=args.random_state)))
    ])

    print("[fit] Training student (TF-IDF + Ridge) on teacher probabilities...")
    pipe.fit(X_train, Y_train)

    print("[eval] Predicting on validation set...")
    Y_pred = pipe.predict(X_val)  # [N, C]
    # ensure proper probability simplex
    Y_pred = np.clip(Y_pred, 1e-12, None)
    Y_pred = Y_pred / (Y_pred.sum(axis=1, keepdims=True) + 1e-12)

    student_top1 = Y_pred.argmax(axis=1)
    acc_vs_teacher = accuracy_score(top1_val, student_top1)
    kl_stu_teacher = kl_div(Y_pred, Y_val)   # KL(student || teacher)
    kl_teacher_stu = kl_div(Y_val, Y_pred)   # KL(teacher || student)

    print(f"[metrics] acc_vs_teacher:          {acc_vs_teacher:.6f}")
    print(f"[metrics] KL(student || teacher):  {kl_stu_teacher:.6f}")
    print(f"[metrics] KL(teacher || student):  {kl_teacher_stu:.6f}")

    if ytrue_val is not None:
        acc_vs_true = accuracy_score(ytrue_val, student_top1)
        print(f"[metrics] acc_vs_true_labels:     {acc_vs_true:.6f}")
    else:
        acc_vs_true = None
        print("[metrics] acc_vs_true_labels:      n/a")

    # --- Create outputs/ folder if not exists ---
    output_dir = os.path.join(os.getcwd(), "outputs")
    os.makedirs(output_dir, exist_ok=True)


    # --- Save Model ---
    if args.save_model:
        model_path = os.path.join(output_dir, args.save_model)
        dump(pipe, model_path)
        print(f"[save] student model → {model_path}")


    if args.save_manifest:
        manifest: Dict[str, Any] = {
            "script": os.path.basename(sys.argv[0]),
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "sklearn_version": sklearn.__version__,
            "random_state": args.random_state,
            "blas_threads": {
                k: os.environ.get(k, None) for k in [
                    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"
                ]
            },
            "dataset": {
                "source": args.dataset_hub if args.dataset_hub else args.dataset_disk,
                "split": args.split,
                "fingerprint": ds_fingerprint,
                "size": len(ds),
                "val_size": args.val_size,
                "idx_val": idx_val.tolist(),  # exact rows used for validation
            },
            "model": {
                "type": "TFIDF+Ridge(MultiOutput)",
                "ngram_range": [1, 2],
                "max_features": args.max_features,
                "ridge_alpha": 1.0,
            },
            "metrics": {
                "acc_vs_teacher": acc_vs_teacher,
                "kl_student_teacher": kl_stu_teacher,
                "kl_teacher_student": kl_teacher_stu,
                "acc_vs_true_labels": acc_vs_true,
            },
        }
        manifest_path = os.path.join(output_dir, args.save_manifest)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"[save] manifest → {manifest_path}")


if __name__ == "__main__":
    main()