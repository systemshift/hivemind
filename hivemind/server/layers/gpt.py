import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GPTIntermediateBlock(nn.Module):
    """
    A simple transformer block with autoregressive mask

    TODO: enforce dropout layer consistency (see dropout.py) for actual training, same as in torch.utils.checkpoint
    TODO: use post-normalization in transformer
    TODO: for logits layer: move logits to a separate block to avoid bottlenecking the entire pipeline
    """

    def __init__(self, d_model, nhead, num_layers):
        super().__init__()

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model * 4, dropout=0),
            num_layers=num_layers)
        self._autoregressive_mask = None

    def get_mask_for(self, sequence_length: int, device: torch.device):
        if self._autoregressive_mask is None or self._autoregressive_mask.shape[0] != sequence_length\
                or self._autoregressive_mask.device != device:
            mask = (torch.triu(torch.ones(sequence_length, sequence_length)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            self._autoregressive_mask = mask.to(device)
        return self._autoregressive_mask

    def forward(self, activations):
        # activations shape: batch_size x seq_length x hid_size
        mask = self.get_mask_for(sequence_length=activations.shape[1], device=activations.device)
        activations_transposed = activations.transpose(0, 1)  # seq_length x batch_size x hid_size
        outputs_transposed = self.transformer(activations_transposed, mask)
        return outputs_transposed.transpose(0, 1)


class GPTInitialBlock(GPTIntermediateBlock):
    def __init__(self, d_model, nhead, num_layers, vocab_size):
        super().__init__(d_model, nhead, num_layers)
        self.embedding = nn.Embedding(vocab_size, embedding_dim=d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=2048)

    def forward(self, tokens):
        # tokens shape = [batch_size x seq_length]
        tokens_transposed = tokens.transpose(0, 1)  # [seq_length x batch_size]
        embeddings_transposed = self.positional_encoding(self.embedding(tokens_transposed))
        mask = self.get_mask_for(tokens.shape[1], embeddings_transposed.device)
        outputs_transposed = self.transformer(embeddings_transposed, mask)
        return outputs_transposed.transpose(0, 1)


class GPTFinalBlock(GPTIntermediateBlock):
    def __init__(self, d_model, nhead, num_layers, vocab_size):
        super().__init__(d_model, nhead, num_layers)
        self.rdo_to_logits = nn.Linear(d_model, vocab_size)

    def forward(self, activations, targets):
        # activations shape: batch_size x seq_length x hid_size of float32
        # targets shape: batch_size x seq_length of int64 reference answers
        readouts = super().forward(activations)  # batch_size x seq_length x hid_size
        logits = self.rdo_to_logits(readouts)  # batch_size x seq_length x vocab_size
        return F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction='none').view(*targets.shape)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0, max_len=2048):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # assumes x of shape [seq_length, batch_size, num_units]
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)