# NLP Learning Cycle

A practical NLP project to understand sequence models step by step
by experimenting with real data and simple architectures.

The focus is on **learning and observation**, not achieving the best accuracy.

---


---

## Dataset

**IMDb Movie Reviews Dataset (CSV)**

- Real movie reviews
- Binary sentiment labels:
  - positive
  - negative
- Loaded from a CSV file

Basic preprocessing is applied:
- lowercase
- word splitting
- sequence truncation

The preprocessing is intentionally simple.

---

## Experiment 1 – RNN with Small Dataset

**File:** `experiments/exp1_rnn_small.py`

### Setup
- Model: Vanilla RNN
- Dataset size: 2000 samples
- Sequence length: 50 words
- Task: Sentiment classification

### Goal
To show that a simple RNN can achieve **reasonable accuracy**
when:
- the dataset is small
- sequences are short

This setup hides the weaknesses of RNNs on purpose.

### Result (example run)

Epoch [1/5] | Loss: 0.7165 | Accuracy: 0.4435
Epoch [2/5] | Loss: 0.6693 | Accuracy: 0.6225
Epoch [3/5] | Loss: 0.6231 | Accuracy: 0.7510
Epoch [4/5] | Loss: 0.5759 | Accuracy: 0.8495
Epoch [5/5] | Loss: 0.5256 | Accuracy: 0.9100

### Observation
- Accuracy increases quickly
- The model appears to work well
- This creates a **false sense of success**, which is intentional

---