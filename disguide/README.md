# 🚀 Real-World Attack: Live Extraction of BERT

In addition to our offline baselines, we performed a realistic **"Live Extraction"** attack against a fine-tuned Transformer model. This experiment simulates a real-world scenario where the attacker queries a black-box API (e.g., an LLM or commercial classifier) to train a lightweight surrogate.

## ⚔️ Attack Setup

* **Victim (Teacher):** `textattack/bert-base-uncased-yelp-polarity` (Fine-tuned BERT)
    * *Access:* Black-box API (Input text $\to$ Probability distribution)
* **Thief (Student Committee):**
    * **Student 1:** TF-IDF + Ridge Regression (Stable solver)
    * **Student 2:** TF-IDF + SGD Regression (Adversarial solver)
* **Strategy:** **DisGUIDE (Query-By-Committee)**
    * We maintain a pool of unlabeled Yelp reviews.
    * In each round, we identify samples where the two students **disagree the most** (proxy for uncertainty).
    * We query the Victim API *only* for these high-value samples.

## 📊 Results: 1,000 Query Budget

We restricted the attack to a budget of **1,000 queries** (approx. 0.18% of the full Yelp dataset).

| Round | Queries | Student Accuracy | Avg Disagreement (Uncertainty) |
| :--- | :--- | :--- | :--- |
| **Initial** | 100 | 74.10% | 0.2264 |
| **Round 3** | 400 | 77.40% | 0.1449 |
| **Round 6** | 700 | 80.50% | 0.0885 |
| **Final** | **1,000** | **82.50%** | **0.0676** |

### 💡 Key Findings
1.  **High Fidelity:** With only **1,000 samples**, our linear student achieved **82.5% accuracy** against the BERT teacher. This represents a **+8.4% improvement** over the initial random seed.
2.  **Uncertainty Reduction:** The average disagreement between students dropped exponentially from **0.23** to **0.07**. This confirms that the Active Learning strategy successfully identified and resolved the "blind spots" in the student's knowledge.
3.  **Efficiency:** The method converged rapidly; by 700 queries, the students had largely aligned, and marginal gains decreased, suggesting extremely high data efficiency.

## 🏃‍♀️ How to Run (Colab / GPU)

This experiment requires a GPU to run the Teacher (BERT) inference efficiently.

```bash
# Install dependencies
pip install transformers datasets scikit-learn numpy torch

# Run the live extraction script
python steal_disguide_live_teacher.py \
  --teacher_model "textattack/bert-base-uncased-yelp-polarity" \
  --dataset_hub "yelp_polarity" \
  --query_budget 1000