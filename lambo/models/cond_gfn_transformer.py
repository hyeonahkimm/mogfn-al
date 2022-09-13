import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_layers, dropout_prob, init_drop=False):
        super(MLP, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim    
        layers = [nn.Linear(in_dim, hidden_layers[0]), nn.ReLU()] 
        layers += [nn.Dropout(dropout_prob)] if init_drop else []
        for i in range(1, len(hidden_layers)):
            layers.extend([nn.Linear(hidden_layers[i-1], hidden_layers[i]), nn.ReLU(), nn.Dropout(dropout_prob)])
        layers.append(nn.Linear(hidden_layers[-1], out_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x, with_uncertainty=False):
        return self.model(x)

class CondGFNTransformer(nn.Module):
    def __init__(self, num_hid, cond_dim, max_len, vocab_size, num_actions, dropout, num_layers,
                num_head, use_cond, encoder=None, encoder_config=None, **kwargs):
        super().__init__()
        self.pos = PositionalEncoding(num_hid, dropout=dropout, max_len=max_len + 2)
        self.use_cond = use_cond
        self.embedding = nn.Embedding(vocab_size, num_hid)
        if encoder is None:
            encoder_layers = nn.TransformerEncoderLayer(num_hid, num_head, num_hid, dropout=dropout)
            self.encoder = nn.TransformerEncoder(encoder_layers, num_layers)
            self.use_pt_encoder = False
            if self.use_cond:
                self.output = MLP(num_hid + num_hid, num_actions, [4 * num_hid, 4 * num_hid], dropout)
                self.cond_embed = nn.Linear(cond_dim, num_hid)
                self.Z_mod = nn.Linear(cond_dim, num_hid)
            else:
                self.output = MLP(num_hid, num_actions, [2 * num_hid, 2 * num_hid], dropout)
                self.Z_mod = nn.Parameter(torch.ones(num_hid) * 30 / num_hid)
        
        else:
            self.encoder = encoder
            self.use_pt_encoder = True
            if self.use_cond:
                self.output = MLP(encoder_config.model.latent_dim + num_hid, num_actions, [4 * num_hid, 4 * num_hid], dropout)
                self.cond_embed = nn.Linear(cond_dim, num_hid)
                self.Z_mod = nn.Linear(cond_dim, num_hid)
            else:
                self.output = MLP(encoder_config.model.latent_dim, num_actions, [2 * num_hid, 2 * num_hid], dropout)
                self.Z_mod = nn.Parameter(torch.ones(num_hid) * 30 / num_hid)

        # self.Z_mod = MLP(cond_dim, num_hid, [num_hid, num_hid], 0.05)
        self.logsoftmax2 = torch.nn.LogSoftmax(2)
        self.num_hid = num_hid

    def Z(self, cond_var):
        return self.Z_mod(cond_var).sum(1) if self.use_cond else self.Z_mod.sum()

    def model_params(self, freeze_encoder=False):
        return list(self.pos.parameters()) + list(self.embedding.parameters()) + list(self.encoder.parameters()) + \
            list(self.output.parameters()) if not freeze_encoder else list(self.pos.parameters()) + list(self.embedding.parameters()) + \
            list(self.output.parameters())

    def Z_param(self):
        return self.Z_mod.parameters() if self.use_cond else [self.Z_mod]

    def forward(self, x, cond, mask, return_all=False, lens=None, logsoftmax=False):
        """
        Assumes batch_first=False
        cond is separate cond for each x, same batch dim as x
        """
        if self.use_pt_encoder:
            # here batch_first is True so x.t()
            pooled_x = self.encoder(x.t())
        else:    
            x = self.embedding(x)
            x = self.pos(x)            
            x = self.encoder(x, src_key_padding_mask=mask,
                                mask=generate_square_subsequent_mask(x.shape[0]).to(x.device))
            pooled_x = x[lens-1, torch.arange(x.shape[1])]

        if self.use_cond:
            cond_var = self.cond_embed(cond) # batch x hidden_dim
            cond_var = torch.tile(cond_var, (x.shape[0], 1, 1)) if return_all else cond_var
            final_rep = torch.cat((x, cond_var), axis=-1) if return_all else torch.cat((pooled_x, cond_var), axis=-1)
        else:
            final_rep = x if return_all else pooled_x
        
        if return_all:
            out = self.output(final_rep)
            return self.logsoftmax2(out) if logsoftmax else out
        
        y = self.output(final_rep)
        return y


class CondSeqTransformer(nn.Module):
    def __init__(self, num_hid, cond_dim, max_len, vocab_size, num_actions, dropout, num_layers,
                num_head, use_cond, encoder=None, tie_embedding=False, **kwargs):
        super().__init__()
        self.pos = PositionalEncoding(num_hid, dropout=dropout, max_len=max_len + 2)
        self.use_cond = use_cond
        self.embedding = nn.Embedding(vocab_size, num_hid)
        self.tie_embedding = tie_embedding
        if encoder is None:
            encoder_layers = nn.TransformerEncoderLayer(num_hid, num_head, num_hid, dropout=dropout)
            self.encoder = nn.TransformerEncoder(encoder_layers, num_layers)
            self.use_pt_encoder = False
        else:
            self.encoder = encoder
            self.use_pt_encoder = True
        # self.output = nn.Linear(num_hid + num_hid, num_actions)
        if self.use_cond:
            self.output_pos = MLP(num_hid + num_hid, 1, [4 * num_hid, 4 * num_hid], dropout)
            if tie_embedding:
                self.output_tok = nn.Sequential(nn.LayerNorm(num_hid+num_hid, eps=0.5), nn.Linear(num_hid + num_hid, num_hid))
            else:
                self.output_tok = MLP(num_hid + num_hid, num_actions, [4 * num_hid, 4 * num_hid], dropout)
            self.cond_embed = nn.Linear(cond_dim, num_hid)
            self.Z_mod = nn.Linear(cond_dim, num_hid)
        else:
            self.output_pos = MLP(num_hid, 1, [4 * num_hid, 4 * num_hid], dropout)
            if tie_embedding:
                self.output_tok = nn.Sequential(nn.LayerNorm(num_hid, eps=0.5))
            else:
                self.output_tok = MLP(num_hid, num_actions, [4 * num_hid, 4 * num_hid], dropout)
            # self.output = MLP(num_hid, num_actions, [2 * num_hid, 2 * num_hid], dropout)
            self.Z_mod = nn.Parameter(torch.ones(num_hid) * 30 / num_hid)
        # self.Z_mod = MLP(cond_dim, num_hid, [num_hid, num_hid], 0.05)
        self.logsoftmax2 = torch.nn.LogSoftmax(2)
        self.num_hid = num_hid

    def Z(self, cond_var):
        return self.Z_mod(cond_var).sum(1) if self.use_cond else self.Z_mod.sum()

    def model_params(self, freeze_encoder=False):
        return list(self.pos.parameters()) + list(self.embedding.parameters()) + list(self.encoder.parameters()) + \
            list(self.output_pos.parameters()) + list(self.output_tok.parameters()) if not freeze_encoder else list(self.pos.parameters()) + list(self.embedding.parameters()) + \
            list(self.output_pos.parameters()) + list(self.output_tok.parameters())

    def Z_param(self):
        return self.Z_mod.parameters() if self.use_cond else [self.Z_mod]

    def forward(self, x, cond, mask, lens=None, logsoftmax=False):
        """
        cond is separate cond for each x, same batch dim as x
        """        

        if self.use_pt_encoder:
            x = self.encoder.get_token_features(x.t()).transpose(1, 0)
        else:
            x = self.embedding(x)
            x = self.pos(x)            
            x = self.encoder(x, src_key_padding_mask=mask)

        if self.use_cond:
            cond_var = self.cond_embed(cond) # batch x hidden_dim
            cond_var = torch.tile(cond_var, (x.shape[0], 1, 1))
            final_rep = torch.cat((x, cond_var), axis=-1)
        else:
            final_rep = x
        
        pos_logits = self.output_pos(final_rep).squeeze().transpose(1, 0)
        tok_logits = self.output_tok(final_rep).squeeze().transpose(1, 0)
        if self.tie_embedding:
            tok_logits = tok_logits @ self.embedding.weight.t()
        return (pos_logits, tok_logits)

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