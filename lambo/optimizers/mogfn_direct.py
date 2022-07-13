import hydra
import wandb
import pandas as pd
import time
import numpy as np
import torch
import random
import matplotlib.pyplot as plt

from torch.nn import functional as F
from itertools import product
import itertools
from polyleven import levenshtein

from pymoo.factory import get_performance_indicator

from botorch.utils.multi_objective import infer_reference_point

from lambo.models.mlm import sample_tokens, evaluate_windows
from lambo.optimizers.pymoo import pareto_frontier, Normalizer
from lambo.models.shared_elements import check_early_stopping
from lambo.utils import weighted_resampling, DataSplit, update_splits, str_to_tokens, tokens_to_str, safe_np_cat
from lambo.models.lanmt import corrupt_tok_idxs
from lambo.metrics.r2 import r2_indicator_set
from lambo.metrics.hsr_indicator import HSR_Calculator

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from lambo.utils import ResidueTokenizer
from torch.distributions import Categorical
import numpy as np
from tqdm import tqdm

def mean_pairwise_distances(seqs):
    dists = []
    for pair in itertools.combinations(seqs, 2):
        dists.append(levenshtein(*pair))
    return np.mean(dists)

def generate_simplex(dims, n_per_dim):
    spaces = [np.linspace(0.0, 1.0, n_per_dim) for _ in range(dims)]
    return np.array([comb for comb in product(*spaces) 
                     if np.allclose(sum(comb), 1.0)])

def thermometer(v, n_bins=50, vmin=0, vmax=32):
    bins = torch.linspace(vmin, vmax, n_bins)
    gap = bins[1] - bins[0]
    return (v[..., None] - bins.reshape((1,) * v.ndim + (-1,))).clamp(0, gap.item()) / gap

class MOGFN(object):
    def __init__(self, bb_task, model, tokenizer, encoder, surrogate, acquisition, num_rounds, num_gens,
                 pi_lr, z_lr, train_steps, random_action_prob, max_len, min_len, batch_size, reward_min, sampling_temp,
                 wd, therm_n_bins, gen_clip, beta_use_therm, pref_use_therm, encoder_obj, sample_beta,
                 beta_cond, pref_cond, beta_scale, beta_shape, pref_alpha, beta_max, simplex_bins, eval_freq, **kwargs):
        self.tokenizer = tokenizer
        self.num_rounds = num_rounds
        self.num_gens = num_gens
        self.train_steps = train_steps
        self._hv_ref = None
        self._ref_point = np.array([1] * bb_task.obj_dim)
        self.max_len = max_len
        self.min_len = min_len
        self.random_action_prob = random_action_prob
        self.batch_size = batch_size
        self.reward_min = reward_min
        self.therm_n_bins = therm_n_bins
        self.beta_use_therm = beta_use_therm
        self.pref_use_therm = pref_use_therm
        self.gen_clip = gen_clip
        self.sampling_temp = sampling_temp
        self.sample_beta = sample_beta
        self.beta_cond = beta_cond
        self.pref_cond = pref_cond
        self.beta_scale = beta_scale
        self.beta_shape = beta_shape
        self.pref_alpha = pref_alpha
        self.beta_max = beta_max
        self.eval_freq = eval_freq
        self.k = kwargs["k"]
        self.num_samples = kwargs["num_samples"]
        self.bb_task = hydra.utils.instantiate(bb_task, tokenizer=tokenizer, candidate_pool=[])

        self.encoder_config = encoder
        self.encoder = hydra.utils.instantiate(encoder, tokenizer=tokenizer)
        
        # self.encoder_obj = encoder_obj
        pref_dim = self.therm_n_bins * self.bb_task.obj_dim if self.pref_use_therm else self.bb_task.obj_dim
        beta_dim = self.therm_n_bins if self.beta_use_therm else 1
        cond_dim = pref_dim+beta_dim if self.beta_cond else pref_dim
        self.model = hydra.utils.instantiate(model, cond_dim=cond_dim)


        self.surrogate_config = surrogate
        self.surrogate_model = hydra.utils.instantiate(surrogate, tokenizer=self.encoder.tokenizer,
                                                       encoder=self.encoder)
        self.acquisition = acquisition

        self.tokenizer = tokenizer
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.num_props = self.bb_task.obj_dim
        self.model.to(self.device)
        self.opt = torch.optim.Adam(self.model.model_params(), pi_lr, weight_decay=wd,
                            betas=(0.9, 0.999))
        self.opt_Z = torch.optim.Adam(self.model.Z_param(), z_lr, weight_decay=wd,
                            betas=(0.9, 0.999))
        self.eos_char = "[SEP]"
        self.pad_tok = self.tokenizer.convert_token_to_id("[PAD]")
        self.encoder_obj = encoder_obj
        self.simplex = generate_simplex(self.num_props, simplex_bins)

    def optimize(self, candidate_pool, pool_targets, all_seqs, all_targets, log_prefix=''):
        batch_size = self.bb_task.batch_size
        target_min = all_targets.min(axis=0).copy()
        target_range = all_targets.max(axis=0).copy() - target_min
        hypercube_transform = Normalizer(
            loc=target_min + 0.5 * target_range,
            scale=target_range / 2.,
        )
        new_seqs = all_seqs.copy()
        new_targets = all_targets.copy()
        self._ref_point = torch.zeros(self.num_props).numpy()
        print(self._ref_point)

        print('\n---- optimizing candidates ----')
        gfn_records = self.train()

        return gfn_records
    
    def compute_mo_metrics(self, solutions):
        hv_indicator = get_performance_indicator('hv', ref_point=self._ref_point)
        # print(pareto_targets)
        hv = hv_indicator.do(-solutions)
        
        r2 = r2_indicator_set(self.simplex, solutions, np.ones(self.num_props))
        hsr_class = HSR_Calculator(lower_bound=-np.ones(self.num_props) - 0.1, upper_bound=np.zeros(self.num_props) + 0.1)
        try:
            hsri, x = hsr_class.calculate_hsr(-solutions)
        except:
            hsri = 0.

        return hv, r2, hsri

    def sample_eval(self, plot=False):
        new_candidates = []
        r_scores = [] 
        all_rewards = []
        topk_rs = []
        topk_div = []
        for prefs in self.simplex:
            cond_var, (_, beta) = self.get_condition_var(prefs=prefs, train=False, bs=self.num_samples)
            samples, _ = self.sample(self.num_samples, cond_var, train=False)
            rewards = -self.bb_task.score(samples)
            r = self.process_reward(samples, prefs, rewards=rewards)
            
            # topk metrics
            topk_r, topk_idx = torch.topk(r, self.k)
            samples = np.array(samples)
            topk_seq = samples[topk_idx].tolist()
            edit_dist = mean_pairwise_distances(topk_seq)
            topk_rs.append(topk_r.mean().item())
            topk_div.append(edit_dist)
            
            # top 1 metrics
            max_idx = r.argmax()
            new_candidates.append(samples[max_idx])
            all_rewards.append(rewards[max_idx])
            r_scores.append(r.max().item())

        r_scores = np.array(r_scores)
        all_rewards = np.array(all_rewards)
        new_candidates = np.array(new_candidates)
        pareto_candidates, pareto_targets = pareto_frontier(new_candidates, all_rewards, maximize=True)
        
        mo_metrics = self.compute_mo_metrics(pareto_targets)

        fig = self.plot_pareto(all_rewards) if plot else None
        
        return new_candidates, all_rewards, r_scores, mo_metrics, (np.array(topk_rs), np.array(topk_div)), fig

    def plot_pareto(self, obj_vals):
        if self.num_props <= 3:
            fig = plt.figure()
            ax = fig.add_subplot(111) if self.num_props == 2 else fig.add_subplot(111, projection='3d')
            ax.scatter(*np.hsplit(obj_vals, obj_vals.shape[-1]))
            ax.set_xlabel("Reward 1")
            ax.set_ylabel("Reward 2")
            if self.num_props == 3:
                ax.set_xlabel("Reward 3")
            return fig

    def get_condition_var(self, prefs=None, beta=None, train=True, bs=None):
        if prefs is None:
            if not train:
                prefs = self.simplex[0]
            else:
                prefs = np.random.dirichlet([self.pref_alpha]*self.num_props)
        if beta is None:
            if train:
                beta = float(np.random.randint(1, self.beta_max+1)) if self.beta_cond else self.sample_beta
            else:
                beta = self.sample_beta

        if self.pref_use_therm:
            prefs_enc = thermometer(torch.from_numpy(prefs), self.therm_n_bins, 0, 1) 
        else: 
            prefs_enc = torch.from_numpy(prefs)
        
        if self.beta_use_therm:
            beta_enc = thermometer(torch.from_numpy(np.array([beta])), self.therm_n_bins, 0, self.beta_max) 
        else:
            beta_enc = torch.from_numpy(np.array([beta]))
        if self.beta_cond:
            cond_var = torch.cat((prefs_enc.view(-1), beta_enc.view(-1))).float().to(self.device)
        else:
            cond_var = prefs_enc.view(-1).float().to(self.device)
        if bs:
            cond_var = torch.tile(cond_var.unsqueeze(0), (bs, 1))
        return cond_var, (prefs, beta)

    def train(self):
        losses, rewards = [], []
        hv, r2, hsri, rs = 0., 0., 0., np.zeros(self.num_props)
        pb = tqdm(range(self.train_steps))
        for i in pb:
            loss, r = self.train_step(self.batch_size)
            losses.append(loss)
            rewards.append(r)
            
            if i != 0 and i % self.eval_freq == 0:
                with torch.no_grad():
                    samples, all_rews, rs, mo_metrics, topk_metrics, fig = self.sample_eval(plot=True)
                hv, r2, hsri = mo_metrics[0], mo_metrics[1], mo_metrics[2]
                try:
                    hsri = hsri if type(hsri) is float else hsri[0]
                except:
                    hsri = 0.
                wandb.log(dict(
                    topk_rewards=topk_metrics[0].mean(),
                    topk_diversity=topk_metrics[1].mean(),
                    hv=mo_metrics[0],
                    r2=mo_metrics[1],
                    hsri=mo_metrics[2],
                    sample_r=rs.mean()
                ),commit=False)
                if fig is not None:
                    wandb.log(dict(
                        pareto_front=wandb.Image(fig)
                    ),commit=False)
                table = wandb.Table(columns = ["Sequence", "Rewards", "Prefs"])
                for sample, rew, pref in zip(samples, all_rews, self.simplex):
                    table.add_data(str(sample), str(rew), str(pref))
                wandb.log({"generated_seqs": table})
            wandb.log(dict(
                train_loss=loss,
                train_rewards=r,
            ))
            pb.set_description("Sample Rew: {:.3f}, HV: {:.3f}, R2: {:.3f}, HSRI: {:.3f}, Train Loss: {:.3f}, Train Rewards: {:.3f}".format(rs.mean(), hv, r2, hsri, sum(losses[-10:]) / 10, sum(rewards[-10:]) / 10))
        return {
            'losses': losses,
            'train_rs': rewards,
            'hypervol_rel': hv
        }
    
    def train_step(self, batch_size):
        cond_var, (prefs, beta) = self.get_condition_var(train=True, bs=batch_size)
        states, logprobs = self.sample(batch_size, cond_var)

        r = self.process_reward(states, prefs).to(self.device)
        self.opt.zero_grad()
        self.opt_Z.zero_grad()
        
        # TB Loss
        loss = (logprobs - beta * r.clamp(min=self.reward_min).log()).pow(2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gen_clip)
        self.opt.step()
        self.opt_Z.step()
        return loss.item(), r.mean()

    def get_log_prob(self, states, cond_var, batch_cond):
        lens = torch.tensor([len(z) + 2 for z in states]).long().to(self.device)
        x = str_to_tokens(states, self.encoder.tokenizer).to(self.device).t()
        mask = x.eq(self.encoder.tokenizer.padding_idx)
        logits = self.model(x, cond_var, batch_cond, mask=mask.transpose(1,0), return_all=True, lens=lens, logsoftmax=True)
        seq_logits = (logits.reshape(-1, 21)[torch.arange(x.shape[0] * x.shape[1], device=self.device), (x.reshape(-1)-4).clamp(0)].reshape(x.shape) * mask.logical_not().float()).sum(0)
        seq_logits += self.model.Z(cond_var)
        return seq_logits

    def val_step(self, batch_size):
        overall_loss = 0.
        for pref in self.simplex:
            cond_var, (prefs, beta) = self.get_condition_var(prefs=pref, train=False, bs=batch_size)
            num_batches = len(self.val_split.inputs) // self.batch_size
            losses = 0
            for i in range(num_batches):
                states = self.val_split.inputs[i * self.batch_size:(i+1) * self.batch_size]
                logprobs = self.get_log_prob(states, cond_var, batch_cond=None)
                r = self.process_reward(self.val_split.inputs[i * self.batch_size:(i+1) * self.batch_size], prefs).to(seq_logits.device)
                loss = (seq_logits - beta * r.clamp(min=self.reward_min).log()).pow(2).mean()

                losses += loss.item()
            overall_loss += (losses / num_batches)
        return overall_loss / len(self.simplex)

    def sample(self, episodes, cond_var=None, train=True):
        states = [''] * episodes
        traj_logprob = torch.zeros(episodes).to(self.device)
        if cond_var is None:
            cond_var, _ = self.get_condition_var(train=train, bs=episodes)
        active_mask = torch.ones(episodes).bool().to(self.device)
        x = str_to_tokens(states, self.encoder.tokenizer).to(self.device).t()[:1]
        lens = torch.zeros(episodes).long().to(self.device)
        uniform_pol = torch.empty(episodes).fill_(self.random_action_prob).to(self.device)

        for t in (range(self.max_len) if episodes > 0 else []):
            logits = self.model(x, cond_var, lens=lens, mask=None)
            
            if t <= self.min_len:
                logits[:, 0] = -1000 # Prevent model from stopping
                                     # without having output anything
                if t == 0:
                    traj_logprob += self.model.Z(cond_var)

            cat = Categorical(logits=logits / self.sampling_temp)
            actions = cat.sample()
            if train and self.random_action_prob > 0:
                uniform_mix = torch.bernoulli(uniform_pol).bool()
                actions = torch.where(uniform_mix, torch.randint(int(t <= self.min_len), logits.shape[1], (episodes, )).to(self.device), actions)
            
            log_prob = cat.log_prob(actions) * active_mask
            traj_logprob += log_prob

            actions_apply = torch.where(torch.logical_not(active_mask), torch.zeros(episodes).to(self.device).long(), actions + 4)
            active_mask = torch.where(active_mask, actions != 0, active_mask)

            x = torch.cat((x, actions_apply.unsqueeze(0)), axis=0)
            if active_mask.sum() == 0:
                break
        states = tokens_to_str(x.t(), self.encoder.tokenizer)
        return states, traj_logprob
    

    def process_reward(self, seqs, prefs, rewards=None):
        if rewards is None:
            rewards = -self.bb_task.score(seqs)
        reward = (torch.tensor(prefs) * (rewards)).sum(axis=1)
        return reward

    def _log_candidates(self, candidates, targets, round_idx, log_prefix):
        table_cols = ['round_idx', 'cand']
        table_cols.extend([f'obj_val_{idx}' for idx in range(self.bb_task.obj_dim)])
        for cand, obj in zip(candidates, targets):
            new_row = [round_idx, cand]
            new_row.extend([elem for elem in obj])
            record = {'/'.join((log_prefix, 'candidates', key)): val for key, val in zip(table_cols, new_row)}
            wandb.log(record)


    def _log_optimizer_metrics(self, normed_targets, round_idx, num_bb_evals, start_time, log_prefix):
        hv_indicator = get_performance_indicator('hv', ref_point=self._ref_point)
        new_hypervol = hv_indicator.do(normed_targets)
        self._hv_ref = new_hypervol if self._hv_ref is None else self._hv_ref
        metrics = dict(
            round_idx=round_idx,
            hypervol_abs=new_hypervol,
            hypervol_rel=new_hypervol / max(1e-6, self._hv_ref),
            num_bb_evals=num_bb_evals,
            time_elapsed=time.time() - start_time,
        )
        print(pd.DataFrame([metrics]).to_markdown(floatfmt='.4f'))
        metrics = {'/'.join((log_prefix, 'opt_metrics', key)): val for key, val in metrics.items()}
        wandb.log(metrics)
        return metrics
