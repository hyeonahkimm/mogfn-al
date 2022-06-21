import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class CondGFNTransformer(nn.Module):
    def __init__(self, num_hid, cond_dim, max_len, vocab_size, num_actions, dropout, bidirectional, num_layers,
                num_head, batch_size):
        super().__init__()
        self.pos = PositionalEncoding(num_hid, dropout=dropout, max_len=max_len + 1)
        self.cond_embed = nn.Linear(cond_dim, num_hid)
        self.embedding = nn.Embedding(vocab_size, num_hid)
        encoder_layers = nn.TransformerEncoderLayer(num_hid, num_head, num_hid, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.output = nn.Linear(num_hid, num_actions)
        self.causal = not bidirectional
        self.Z_mod = nn.Linear(cond_dim, 64)
        self.logsoftmax2 = torch.nn.LogSoftmax(2)

    def Z(self, cond_var):
        return self.Z_mod(cond_var).sum()

    def model_params(self):
        return list(self.pos.parameters()) + list(self.embedding.parameters()) + list(self.encoder.parameters()) + \
            list(self.output.parameters())

    def Z_param(self):
        return self.Z_mod.parameters()

    def forward(self, x, cond, mask, return_all=False, lens=None, logsoftmax=False):
        cond_var = self.cond_embed(cond)
        x = self.embedding(x)
        x = self.pos(x)
        if self.causal:
            x = self.encoder(torch.cat([cond_var, x], axis=0), src_key_padding_mask=mask,
                             mask=generate_square_subsequent_mask(x.shape[0]+1).to(x.device))
            pooled_x = x[lens+1, torch.arange(x.shape[1])]
        else:
            x = self.encoder(x, src_key_padding_mask=mask)
            pooled_x = x[0, :]
        if return_all:
            if logsoftmax:
                return self.logsoftmax2(self.output(x)[1:])
        y = self.output(pooled_x)
        return y


def generate_square_subsequent_mask(sz: int):
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
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
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)