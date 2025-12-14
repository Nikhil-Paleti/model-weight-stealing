# DisGUIDE (Disagreement-Guided Active Querying)

This directory contains a DisGUIDE-style (query-by-committee) **live extraction** experiment. The attacker queries a black-box teacher on a limited budget and trains a lightweight surrogate by prioritizing samples where two student models disagree.

## Attack Setup

- Victim (teacher): `textattack/bert-base-uncased-yelp-polarity`
  - Access: black-box probabilities (input text -> probability distribution)
- Thief (student committee):
  - Student 1: TF-IDF + Ridge regression
  - Student 2: TF-IDF + SGD regression
- Strategy: iteratively query the teacher on inputs with highest student disagreement.

## Results (1,000 query budget)

| Stage   | Queries | Student Accuracy | Avg Disagreement |
| ------- | ------: | ---------------: | ---------------: |
| Initial |     100 |           74.10% |           0.2264 |
| Round 3 |     400 |           77.40% |           0.1449 |
| Round 6 |     700 |           80.50% |           0.0885 |
| Final   |   1,000 |           82.50% |           0.0676 |

## How to Run

Implementation:

- `disguide/disguide.py`

The script is currently a Colab export and uses top-level configuration variables near the top of the file (teacher model, budget, step size, etc.). Adjust those if needed, then run from the repo root:

```bash
uv run python disguide/disguide.py
```

Outputs:

- Prints per-round metrics to stdout.
- Saves `student_live_disguide.joblib` in the current working directory.

Notes:

- A GPU is recommended for fast teacher inference
- CPU works but is slower.
