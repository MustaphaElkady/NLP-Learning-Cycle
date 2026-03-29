import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from inference import load_model_and_vocab, predict_next_word
from experiments.experiments_registry import EXPERIMENT_REGISTRY

st.header("NLP Learning Sycle")
model_type = st.selectbox("Choose Model", ["RNN","LSTM","TRANSFORMER"])
sentence = st.text_input("Enter sentence")

DATASET_DIR = "data/datasets"
os.makedirs(DATASET_DIR, exist_ok=True)
uploaded_file = st.sidebar.file_uploader(
    "Upload Dataset",
    type=["txt"]
)
if uploaded_file is not None:
    save_path = os.path.join(DATASET_DIR, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.success("Dataset uploaded successfully")

dataset_files = [ # 001
    f for f in os.listdir(DATASET_DIR)
    if f.endswith(".txt")
]

dataset_selected = st.sidebar.selectbox(
    "Choose Dataset",
    dataset_files
) 
# dataset_path = os.path.join(DATASET_DIR, dataset_selected)

#TRAIN MODEL
st.sidebar.header("Training Hyperparameters")

seq_len = st.sidebar.number_input("Sequence Length", 5, 200, 50)

epochs = st.sidebar.slider("Epochs", 1, 100, 5)

max_tokens = st.sidebar.number_input("Max Tokens", 1000, 100000, 20000)

lr = st.sidebar.number_input("Learning Rate", value=0.001)

max_tokens = st.sidebar.number_input("Max Tokens", 8, 512, 32)

embedding_size = st.sidebar.number_input("Embedding Size", 32, 512, 128)

hidden_size = st.sidebar.number_input("Hidden Size", 32, 1024, 256)

dropout = st.sidebar.slider("Dropout", 0.0, 0.9, 0.2)


# def run_experiment(dataset_path, dataset_selected,seq_len, , epochs, lr):

if st.sidebar.button("Train Model"):
    dataset_path = os.path.join(DATASET_DIR, dataset_selected)
    train_fn = EXPERIMENT_REGISTRY[model_type]
    model, vocab = train_fn(
    dataset_path=dataset_path,
    dataset_selected=dataset_selected,
    seq_len=seq_len,
    max_tokens=max_tokens,
    epochs=epochs,
    lr=lr
)
    st.success("Training finished")




# TESt model 
if st.button("Test Model"):
    with st.spinner("computing..."):
        model, vocab = load_model_and_vocab(model_type, dataset_selected)
        word = predict_next_word(model,vocab,sentence)
        st.success(f"Output: {word}")




# UI
# ↓
# call inference
# ↓
# display results