from config.model_config import MODEL_CONFIG
from models.model_registry import MODEL_REGISTRY

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_and_vocab(model_type, dataset_path):
    config = MODEL_CONFIG[model_type]
    vocab = torch.load(config["vocab_path"],map_location=device )
    model_class = MODEL_REGISTRY[model_type]
    model = model_class(len(vocab))

    state_dict = torch.load(config["model_path"],map_location=device )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model , vocab

def predict_next_word(model, vocab, sentence):
    tokens = sentence.lower().split()
    indices = [vocab[token] for token in tokens if token in vocab]
    x = torch.tensor(indices).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(x)
    
    pred = torch.argmax(output, dim=1).item()
    idx_to_word = {v:k for k,v in vocab.items()}
    return idx_to_word.get(pred, "UNK")

'''
"The cat is"
        │
        ▼
["the","cat","is"]
        │
        ▼
[1,45,10]
        │
        ▼
Tensor [[1,45,10]]
        │
        ▼
Model
        │
        ▼
Prediction index = 200
        │
        ▼
" sitting "
'''