import torch
from fastapi import FastAPI

from api.schema import PredictRequest, PredictResponse
from models.transformer import TransformerLanguageModel
from data.wikitext.load_wikitext import encode

# App
app = FastAPI(title="NLP Language Model API")


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("API running on:", device)


# Load model & vocab

vocab = torch.load("checkpoints/vocab_trans.pt")
idx_to_word = {i: w for w, i in vocab.items()}

model = TransformerLanguageModel(len(vocab)).to(device)
model.load_state_dict(torch.load("checkpoints/trans_wikitext.pt", map_location=device))
model.eval()


def predict_next(text, max_tokens):
    tokens = text.lower().split()
    encoded = encode(tokens, vocab).unsqueeze(0).to(device)

    results = []

    for _ in range(max_tokens):
        with torch.no_grad():
            logits = model(encoded)

        next_id = torch.argmax(logits, dim=-1).item()
        next_word = idx_to_word.get(next_id, "<UNK>")

        results.append(next_word)

        encoded = torch.cat(
            [encoded, torch.tensor([[next_id]], device=device)],
            dim=1
        )

    return results


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    preds = predict_next(req.text, req.max_tokens)

    return PredictResponse(
        input_text=req.text,
        predicted_tokens=preds
    )
