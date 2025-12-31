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
- Dataset size: 5000 samples
- Sequence length: 100 words
- Task: Sentiment classification

### Goal
To show that a simple RNN can achieve **reasonable accuracy**
when:
- the dataset is small
- sequences are short

This setup hides the weaknesses of RNNs on purpose.

### Result (example run)

Epoch [1/5] | Loss: 0.7528 | Accuracy: 0.2748
Epoch [2/5] | Loss: 0.7038 | Accuracy: 0.4934
Epoch [3/5] | Loss: 0.6599 | Accuracy: 0.6198
Epoch [4/5] | Loss: 0.6191 | Accuracy: 0.7186
Epoch [5/5] | Loss: 0.5794 | Accuracy: 0.7900

### Observation
- Accuracy increases quickly
- The model appears to work well
- This creates a **false sense of success**, which is intentional

---
## Experiment 2 – RNN with Long Sequences

**Change:** Sequence length increased to 300 words  
**Model:** Same RNN as Experiment 1

### Observation

RNN did not fail immediately on long sequences.
This is expected because:
- IMDB sentiment classification relies on local cues
- The task does not require deep long-term dependency modeling

To expose RNN limitations, longer sequences and more complex tasks are required.
