
# mapping to link between model and vocab
MODEL_CONFIG = {
    "RNN" : 
    {
        "model_path" : "checkpoints/rnn_wikitext.pt",
        "vocab_path" : "checkpoints/vocab.pt",
         "model_params": {
            "vocab_size": 10000,
            "embed_dim": 128,
            "hidden_dim": 256,
            "num_layers": 2
        }
    },
    "LSTM" : 
    {
        "model_path" : "checkpoints/lstm_wikitext.pt",
        "vocab_path" : "checkpoints/vocab_lstm.pt",
         "model_params": {
            "vocab_size": 10000,
            "embed_dim": 128,
            "hidden_dim": 256,
            "num_layers": 2
        }
    },
    "TRANSFORMER" : 
    {
        "model_path" : "checkpoints/trans_wikitext.pt",
        "vocab_path" : "checkpoints/vocab_trans.pt",
         "model_params": {
            "vocab_size": 2000,
            "embed_dim": 64,
            "hidden_dim": 128,
            "num_layers": 2
        }
    }
}
