# Model Weight Stealing

This repository explores **model/task stealing** using various methods.  
This project is conducted as part of the **DSC261 course**.

Repository: https://github.com/LightFury9/model-weight-stealing


## Setup

```bash
git clone https://github.com/LightFury9/model-weight-stealing
cd model-weight-stealing
uv sync
```

(Optional) Add a `.env` file if pushing to Hugging Face:

```
HUGGINGFACE_HUB_TOKEN=hf_xxxxx
```


## Initial Experiment — Teacher & Knockoff (Student) Model

We use the **`yelp_review_full`** dataset and run inference using the teacher model:

- **Teacher Model:** `nlptown/bert-base-multilingual-uncased-sentiment`  
- **Outputs extracted per sample:**
  - `model_probs` → softmax probabilities (size 5)  
  - `model_pred` → predicted label (1–5 stars)  
- These predictions are saved and pushed to the HF Hub as:  
  **`LightFury9/yelp-5star-probs`**

Next, we train two student (knockoff) models using these outputs. Both experiments are in the `knockoff/` folder.

### Experiment 1: Stealing Using Soft Labels (Probabilities)

Student is trained on **full probability vectors (`model_probs`)** using TF-IDF + Ridge Regression.

**Results:**
- Accuracy vs Teacher: **0.6465**
- Accuracy vs True Labels: **0.5585**

### Experiment 2: Stealing Using Hard Labels Only

Student is trained **only on the teacher’s predicted labels (`model_pred`)**, no probability info.

**Results:**
- Accuracy vs True Labels: **0.5862**
- Accuracy vs Teacher: **0.5854**

### Initial Analysis

| Scenario        | API Output        | Accuracy       | Difficulty |
|-----------------|--------------------|----------------|------------|
| Soft-label KD   | Probabilities      | 0.6465 vs teacher | Easier to steal |
| Hard-label only | Class label only   | 0.5862 vs true    | Harder, but still possible |

Providing **probabilities/logits makes model extraction significantly easier**  
Even with **only class labels**, a student can still approximate the teacher model

---


## License

This project is intended for **research and educational purposes only**.