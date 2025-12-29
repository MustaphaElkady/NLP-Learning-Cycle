# NLP Learning Cycle 🚀  
A Practical Project to Understand RNN, LSTM, and Transformer Limitations

## 📌 Project Overview
This project is a **learning-oriented NLP project** designed to demonstrate **why different sequence models exist** and **when each one fails or succeeds**.

Instead of directly jumping to advanced models (like Transformers), we follow a **learning cycle**:
- Start with simple models
- Observe their limitations
- Introduce improved architectures step by step

The goal is **understanding through practice**, not just achieving high accuracy.

---

## 🧠 Learning Cycle Concept

Small Dataset + Short Sequences

↓

RNN works (appears good)

↓
Longer Sequences / Larger Data

↓
RNN fails

↓

LSTM improves

↓

Very Large Data / Long Dependencies

↓

LSTM struggles

↓

Transformer succeeds


Each experiment in this repository represents **one step in this cycle**.

---

## 📁 Project Structure

NLP-Learning-Cycle/

│
├── data/
│ └── smalldataset/
│ └── load_imdb.py
│
├── models/
│ ├── rnn.py
│ ├── lstm.py
│ └── transformer.py
│
├── experiments/
│ ├── exp1_rnn_small.py
│ ├── exp2_rnn_long.py
│ ├── exp3_lstm.py
│ └── exp4_transformer.py
│
├── main.py
└── README.md


---

## 📊 Dataset

**IMDb Movie Reviews Dataset**  
- Binary sentiment classification (positive / negative)
- Real-world text data
- Variable-length reviews

Why IMDb?
- Standard NLP dataset
- Suitable for sequence modeling
- Allows controlled experiments with short and long sequences

The dataset is loaded using `torchtext` to keep preprocessing simple and reproducible.

---

## 🧪 Experiments Description

### 🔹 Experiment 1: RNN with Small Dataset
**File:** `exp1_rnn_small.py`

- Small subset of IMDb
- Short sequence length
- Purpose: Show that RNN can appear to work well

Expected outcome:
- Training is stable
- Loss decreases
- Model seems successful

> ⚠️ This success is intentionally misleading.

---

### 🔹 Experiment 2: RNN with Long Sequences
**File:** `exp2_rnn_long.py`

- Same dataset
- Longer sequences
- No architectural changes

Expected outcome:
- Training instability
- Poor performance

This experiment demonstrates the **vanishing gradient problem** in RNNs.

---

### 🔹 Experiment 3: LSTM on the Same Data
**File:** `exp3_lstm.py`

- Same data as Experiment 2
- Replace RNN with LSTM

Expected outcome:
- Improved performance
- More stable training

This shows how **gated mechanisms** help model long-term dependencies.

---

### 🔹 Experiment 4: Transformer Model
**File:** `exp4_transformer.py`

- Large dataset
- Long sequences
- Self-attention instead of recurrence

Expected outcome:
- Better scalability
- Faster training
- Strong performance

This explains why Transformers dominate modern NLP.

---

## 🛠 Technologies Used
- Python
- PyTorch
- torchtext
- Git & GitHub

---

## 🎯 Key Takeaways
- Model choice depends on data characteristics
- RNNs are limited by sequential processing
- LSTMs mitigate but do not eliminate these issues
- Transformers solve long-range dependency problems efficiently

---

## 📌 Final Note
This repository is designed as a **learning journey**, not a benchmark competition.
Understanding *why models fail* is more important than achieving perfect accuracy.