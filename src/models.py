"""Intent classifier models (trained from scratch): TextCNN and BiLSTM."""
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, UNK = "<pad>", "<unk>"


class Vocab:
    def __init__(self, itos):
        self.itos = list(itos)
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    @classmethod
    def build(cls, token_lists, min_freq=1):
        counts = Counter(t for toks in token_lists for t in toks)
        words = sorted(w for w, c in counts.items() if c >= min_freq)
        return cls([PAD, UNK] + words)

    def encode(self, tokens, max_len):
        ids = [self.stoi.get(t, 1) for t in tokens][:max_len]
        return ids or [1]

    def __len__(self):
        return len(self.itos)


def pad_batch(id_lists, min_len=1):
    """Right-pad to the longest sequence (at least `min_len`). Returns ids, lengths."""
    lengths = torch.tensor([len(x) for x in id_lists])
    width = max(int(lengths.max()), min_len)
    ids = torch.zeros(len(id_lists), width, dtype=torch.long)
    for i, x in enumerate(id_lists):
        ids[i, : len(x)] = torch.tensor(x)
    return ids, lengths


class TextCNN(nn.Module):
    """Kim (2014): parallel 1-D convolutions = n-gram detectors, then max-over-time pooling."""

    def __init__(self, vocab_size, num_classes, emb_dim=128, num_filters=100,
                 kernel_sizes=(2, 3, 4), dropout=0.5):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList(nn.Conv1d(emb_dim, num_filters, k) for k in kernel_sizes)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, ids, lengths=None):
        x = self.embedding(ids).transpose(1, 2)            # (B, E, T)
        pooled = [F.relu(conv(x)).max(dim=2).values for conv in self.convs]
        return self.fc(self.dropout(torch.cat(pooled, dim=1)))


class LSTMClassifier(nn.Module):
    """Bidirectional LSTM; final forward + backward hidden states -> linear."""

    def __init__(self, vocab_size, num_classes, emb_dim=128, hidden_dim=128,
                 num_layers=1, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                            bidirectional=True, dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, ids, lengths):
        x = self.dropout(self.embedding(ids))
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                                   enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        h = torch.cat([h[-2], h[-1]], dim=1)               # last layer, both directions
        return self.fc(self.dropout(h))


MODELS = {"cnn": TextCNN, "lstm": LSTMClassifier}
