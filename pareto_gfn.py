import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from lambo.utils import ResidueTokenizer
from torch.distributions import Categorical
import numpy as np
from tqdm import tqdm

LOGINF = 1000

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


CONFIG = {
    "gfn": {
        "random_action_prob": 0.01,
        "max_len": 60,
        "batch_size": 32,
        "reward_exp": 4,
        "reward_min": 1e-10,
        "reward_max": 100,
        "sampling_temp": 1,
        "train_steps": 10000,
        "temp_use_therm": True,
        "pref_use_therm": True,
        "pi_lr": 0.0001,
        "z_lr": 0.001,
        "wd": 0.0001,
        "therm_n_bins": 50,
        "gen_clip": 10
    },
    "model": {
        "vocab_size": 26,
        "num_actions": 21,
        "num_hid": 64,
        "num_layers": 5,
        "num_head": 8,
        "bidirectional": False,
        "dropout": 0,
        "max_len": 60
    }
}

lists = lambda n: [list() for i in range(n)]

def thermometer(v, n_bins=50, vmin=0, vmax=32):
    bins = torch.linspace(vmin, vmax, n_bins)
    gap = bins[1] - bins[0]
    return (v[..., None] - bins.reshape((1,) * v.ndim + (-1,))).clamp(0, gap.item()) / gap


class CondGFNTransformer(nn.Module):
    def __init__(self, cfg, cond_dim):
        super().__init__()
        self.pos = PositionalEncoding(cfg.num_hid, dropout=cfg.dropout, max_len=cfg.max_len + 1)
        self.cond_embed = nn.Linear(cond_dim, cfg.num_hid)
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.num_hid)
        encoder_layers = nn.TransformerEncoderLayer(cfg.num_hid, cfg.num_head, cfg.num_hid, dropout=cfg.dropout)
        self.encoder = nn.TransformerEncoder(encoder_layers, cfg.num_layers)
        self.output = nn.Linear(cfg.num_hid, cfg.num_actions)
        self.causal = not cfg.bidirectional
        self.Z_mod = nn.Linear(cond_dim, 64)

    def Z(self, cond_var):
        return self.Z_mod(cond_var).sum()

    def model_params(self):
        return list(self.pos.parameters()) + list(self.embedding.parameters()) + list(self.encoder.parameters()) + \
            list(self.output.parameters())

    def Z_param(self):
        return self.Z_mod.parameters()

    def forward(self, x, cond, mask, return_all=False, lens=None):
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
            return self.output(x)
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

class MbStack:
    def __init__(self, f):
        self.stack = []
        self.f = f

    def push(self, x, i):
        self.stack.append((x, i))

    def pop_all(self):
        if not len(self.stack):
            return []
        with torch.no_grad():
            ys = self.f([i[0] for i in self.stack]) # eos_tok == 2
        idxs = [i[1] for i in self.stack]
        self.stack = []
        return zip(ys, idxs)

class ParetoGFN:
    def __init__(self, oracle, num_fs):
        self.tokenizer = ResidueTokenizer()
        self.args = dotdict(CONFIG["gfn"])
        self.model_args = dotdict(CONFIG["model"])
        pref_dim = self.args.therm_n_bins * num_fs if self.args.pref_use_therm else num_fs
        temp_dim = self.args.therm_n_bins if self.args.temp_use_therm else 1
        self.model = CondGFNTransformer(self.model_args, pref_dim + temp_dim)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.num_props = num_fs
        self.workers = MbStack(oracle)
        self.model.to(self.device)
        self.opt = torch.optim.Adam(self.model.model_params(), self.args.pi_lr, weight_decay=self.args.wd,
                            betas=(0.9, 0.999))
        self.opt_Z = torch.optim.Adam(self.model.Z_param(), self.args.z_lr, weight_decay=self.args.wd,
                            betas=(0.9, 0.999))
        self.eos_char = "[SEP]"
        self.pad_tok = self.tokenizer.convert_token_to_id("[PAD]")

    def train(self):
        losses = []
        for i in tqdm(range(self.args.train_steps)):
            loss = self.train_step(self.args.batch_size)
            losses.append(loss)
            if i % 10 == 0:
                print(sum(losses[-10:]) / 10)
        return losses

    def sample(self, episodes, prefs, temp):
        states = [''] * episodes
        traj_dones = lists(episodes)

        traj_logprob = torch.zeros(episodes).to(self.device)
        cond_var = torch.cat((prefs.view(-1), temp.view(-1))).float().to(self.device)
        
        for t in (range(self.args.max_len) if episodes > 0 else []):
            active_indices = np.int32([i for i in range(episodes)
                                       if not states[i].endswith(self.eos_char)])
            x = torch.tensor([self.tokenizer.encode(states[i], use_sep=False) for i in active_indices]).to(self.device, torch.long)
            x = x.transpose(1,0)
            lens = torch.tensor([len(i) for i in states
                                 if not i.endswith(self.eos_char)]).long().to(self.device)
            logits = self.model(x, torch.tile(cond_var, (1, x.shape[1], 1)), mask=None, lens=lens)
            
            if t == 0:
                logits[:, 0] = -1000 # Prevent model from stopping
                                     # without having output anything
                traj_logprob += self.model.Z(cond_var)

            cat = Categorical(logits=logits / self.args.sampling_temp)

            actions = cat.sample()
            if self.args.random_action_prob > 0:
                for i in range(actions.shape[0]):
                    if np.random.uniform(0,1) < self.args.random_action_prob:
                        actions[i] = torch.tensor(np.random.randint(t == 0, logits.shape[1])).to(self.device)
            log_prob = cat.log_prob(actions)
            chars = [self.tokenizer.convert_id_to_token(i) for i in actions + 4]

            # Append predicted characters for active trajectories
            for b_i, i, c, a in zip(range(len(actions)), active_indices, chars, actions):
                traj_logprob[i] += log_prob[b_i]
                if c == self.eos_char or t == self.args.max_len - 1:
                    self.workers.push(states[i] + (c if c != self.eos_char else ''), i)
                    r = 0
                    d = 1
                else:
                    r = 0
                    d = 0
                traj_dones[i].append(d)
                states[i] += c
            if all(i.endswith(self.eos_char) for i in states):
                break

        return states, traj_logprob

    def train_step(self, batch_size):
        # generate cond_var randomly
        prefs = np.random.dirichlet([1.5]*self.num_props)
        temp = np.random.gamma(2,1)
        if self.args.pref_use_therm:
            prefs_enc = thermometer(torch.from_numpy(prefs), self.args.therm_n_bins, 0, 1)
        if self.args.temp_use_therm:
            temp_enc = thermometer(torch.from_numpy(np.array([temp])), self.args.therm_n_bins, 0, 32)
        
        states, logprobs = self.sample(batch_size, prefs_enc, temp_enc)
        
        rs = []
        for (r, mbidx) in self.workers.pop_all():
            rs.append(self.process_reward(r, prefs, temp))
        r = torch.tensor(rs).to(self.device)
        self.opt.zero_grad()
        self.opt_Z.zero_grad()        
        # TB Loss
        loss = (logprobs - r.clamp(min=self.args.reward_min).log()).pow(2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.gen_clip)
        self.opt.step()
        self.opt_Z.step()
        return loss.item()
    
    def process_reward(self, r, prefs, temp):
        reward = (prefs * r).sum()
        return reward ** temp


def get_samples(args, generator, model, tokenizer, dataset):
    # instantiate ParetoGFN
    # Train ParetoGFN
    # Generate Sample
    return samples

def test_gfn():
    gfn = ParetoGFN(oracle=lambda x: np.array([[1, 1, 1]] * len(x)), num_fs=3)
    gfn.train()

if __name__ == "__main__":
    test_gfn()
    