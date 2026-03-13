import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from inference import load_model_and_vocab, predict_next_word

# select model and dataset
st.header("NLP Learning Sycle")
model_type = st.selectbox("Choose Model",["RNN", "LSTM", "TRANSFORMER"])
dataset_type = st.selectbox("Select dataset size", ["small", "BIG dataset"])
sentence = st.text_input("Enter sentence to test the model")


# TESt model 
if st.button("Test Model"):
    model, vocab = load_model_and_vocab(model_type)
    word = predict_next_word(model,vocab,sentence)
    st.success("Output: {word}")