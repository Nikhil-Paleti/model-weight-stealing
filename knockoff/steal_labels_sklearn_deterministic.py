#!/usr/bin/env python3
"""
Deterministic hard-label training (no teacher probs):
- Uses TF-IDF (1–2 grams) + a simple classifier (logreg or LinearSVC)
- Trains ONLY on the dataset's 'label' column.
- 80/20 stratified split for reproducibility.
- Optionally reports acc vs teacher top-1 if model_probs present (for comparison to KD).

Example:
python knockoff/steal_labels_sklearn_deterministic.py \
  --dataset_hub LightFury9/yelp-5star-probs \
  --split test \
  --val_size 0.2 \
  --random_state 42 \
  --model_type logreg \
  --save_model student_labels_logreg.joblib \
  --save_manifest manifest_labels.json
"""

import argparse, json, os, platform, random, sys
from typing import Dict, Any
import numpy as np

# ===== Single-thread + deterministic env BEFORE sklearn import =====
os.environ.setdefault("PYTHONHASHSEED", "0")
for var in [
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"
]:
    os.environ.setdefault(var, "1")

import sklearn
from datasets import load_dataset, load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)


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
    ap.add_argument("--split", type=str, default="test", help="Split to use (we split 80/20 inside)")
    ap.add_argument("--text_col", type=str, default="text")
    ap.add_argument("--label_col", type=str, default="label")
    ap.add_argument("--val_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--max_features", type=int, default=200_000)
    ap.add_argument("--model_type", type=str, default="logreg", choices=["logreg", "linearsvc"],
                    help="Simple classifier: multinomial logistic regression or LinearSVC")
    ap.add_argument("--C", type=float, default=1.0, help="Inverse regularization strength for classifier")
    ap.add_argument("--save_model", type=str, default="student_labels.joblib")
    ap.add_argument("--save_manifest", type=str, default="manifest_labels.json")
    return ap.parse_args()


def main():
    args = parse_args()
    set_all_seeds(args.random_state)

    if not args.dataset_hub and not args.dataset_disk:
        raise ValueError("Provide --dataset_hub or --dataset_disk")

    ds, ds_fingerprint = load_split(args.dataset_hub, args.dataset_disk, args.split)

    # Check columns
    if args.text_col not in ds.column_names:
        raise ValueError(f"Text column '{args.text_col}' not found; columns: {ds.column_names}")
    if args.label_col not in ds.column_names:
        raise ValueError(f"Label column '{args.label_col}' not found; columns: {ds.column_names}")

    texts = ds[args.text_col]
    labels = np.array(ds[args.label_col], dtype=int)

    # Optional teacher for comparison (not used in training)
    teacher_probs = np.array(ds["model_probs"]) if "model_probs" in ds.column_names else None
    teacher_top1 = teacher_probs.argmax(axis=1) if teacher_probs is not None else None

    # Stratified 80/20 split on true labels
    idx = np.arange(len(texts), dtype=np.int64)
    X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
        texts, labels, idx, test_size=args.val_size,
        random_state=args.random_state, stratify=labels
    )

    # Build classifier
    if args.model_type == "logreg":
        clf = LogisticRegression(
            penalty="l2",
            C=args.C,
            solver="lbfgs",
            multi_class="multinomial",
            max_iter=1000,
            n_jobs=1,
            random_state=args.random_state,
        )
        clf_name = f"LogisticRegression(l2,C={args.C},lbfgs,multinomial)"
    else:
        clf = LinearSVC(
            C=args.C,
            loss="squared_hinge",
            max_iter=5000,
            random_state=args.random_state,
        )
        clf_name = f"LinearSVC(C={args.C})"

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
        ("clf", clf),
    ])

    print(f"[fit] Training student ({clf_name}) on HARD labels...")
    pipe.fit(X_train, y_train)

    print("[eval] Validation...")
    y_pred = pipe.predict(X_val)
    acc_vs_true = accuracy_score(y_val, y_pred)
    print(f"[metrics] acc_vs_true_labels:     {acc_vs_true:.6f}")

    if teacher_top1 is not None:
        # Align teacher_top1 with val indices
        acc_vs_teacher = accuracy_score(teacher_top1[idx_val], y_pred)
        print(f"[metrics] acc_vs_teacher_top1:   {acc_vs_teacher:.6f}")
    else:
        acc_vs_teacher = None
        print("[metrics] acc_vs_teacher_top1:   n/a (no teacher in dataset)")

    # Prepare outputs dir
    output_dir = os.path.join(os.getcwd(), "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(output_dir, args.save_model)
    dump(pipe, model_path)
    print(f"[save] student model → {model_path}")

    # Save manifest
    manifest = {
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
            "idx_val": idx_val.tolist(),
        },
        "model": {
            "type": "TFIDF+" + ("LogReg" if args.model_type == "logreg" else "LinearSVC"),
            "ngram_range": [1, 2],
            "max_features": args.max_features,
            "C": args.C,
        },
        "metrics": {
            "acc_vs_true_labels": acc_vs_true,
            "acc_vs_teacher_top1": acc_vs_teacher,
        },
    }
    manifest_path = os.path.join(output_dir, args.save_manifest)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[save] manifest → {manifest_path}")


if __name__ == "__main__":
    main()