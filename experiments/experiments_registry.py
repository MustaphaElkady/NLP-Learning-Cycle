from .exp1_rnn import run_experiment as rnn_train
from .exp2_lstm import run_experiment as lstm_train
from .exp3_transformer import run_experiment as transformer_train

EXPERIMENT_REGISTRY = {
    'RNN' : rnn_train,
    'LSTM' : lstm_train,
    'TRANSFORMER' : transformer_train
}