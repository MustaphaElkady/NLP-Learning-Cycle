import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from inference import load_model_and_vocab, predict_next_word

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

dataset_path = os.path.join(DATASET_DIR, dataset_selected)

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