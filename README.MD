### WikiText-2 Data Loader

A new data loader was added to support language modeling experiments
using the WikiText-2 dataset.

Unlike the IMDb dataset, WikiText-2 does not contain labels.
Instead, the task is to predict the next word given a sequence of
previous words.

The loader performs the following steps:
- Reads raw text files
- Splits text into tokens (words)
- Builds a vocabulary mapping words to integer IDs
- Encodes the token stream into numerical form
- Creates input–target pairs using a sliding window

This change allows the same learning cycle
(RNN → LSTM → long sequences)
to be applied to a more challenging and realistic NLP task,
preparing the project for future work with Transformer-based models.
