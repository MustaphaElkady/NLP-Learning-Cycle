from models.rnn import RNNLanguageModel
from models.lstm import LSTMLanguageModel
from models.transformer import TransformerLanguageModel

MODEL_REGISTRY = {
    "RNN": RNN,
    "LSTM": LSTM,
    "TRANSFORMER": Transformer
}