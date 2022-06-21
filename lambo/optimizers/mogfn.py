import hydra
import wandb
import pandas as pd
import time
import numpy as np
import torch
import random

from torch.nn import functional as F
from itertools import product
from pymoo.factory import get_performance_indicator

from botorch.utils.multi_objective import infer_reference_point

from lambo.models.mlm import sample_tokens, evaluate_windows
from lambo.optimizers.pymoo import pareto_frontier, Normalizer
from lambo.models.shared_elements import check_early_stopping
from lambo.utils import weighted_resampling, DataSplit, update_splits, str_to_tokens, tokens_to_str, safe_np_cat
from lambo.models.lanmt import corrupt_tok_idxs

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from lambo.utils import ResidueTokenizer
from torch.distributions import Categorical
import numpy as np
from tqdm import tqdm

def generate_simplex(dims, n_per_dim):
    spaces = [np.linspace(0.0, 1.0, n_per_dim) for _ in range(dims)]
    return np.array([comb for comb in product(*spaces) 
                     if np.allclose(sum(comb), 1.0)])

def thermometer(v, n_bins=50, vmin=0, vmax=32):
    bins = torch.linspace(vmin, vmax, n_bins)
    gap = bins[1] - bins[0]
    return (v[..., None] - bins.reshape((1,) * v.ndim + (-1,))).clamp(0, gap.item()) / gap

lists = lambda n: [list() for i in range(n)]

class MOGFN(object):
    def __init__(self, bb_task, model, tokenizer, encoder, surrogate, acquisition, num_rounds, num_gens,
                 pi_lr, z_lr, train_steps, random_action_prob, max_len, batch_size, reward_min, sampling_temp,
                 wd, therm_n_bins, gen_clip, temp_use_therm, pref_use_therm, encoder_obj, sample_beta,
                 **kwargs):
        self.tokenizer = tokenizer
        self.num_rounds = num_rounds
        self.num_gens = num_gens
        self.train_steps = train_steps
        self._hv_ref = None
        self._ref_point = np.array([1] * bb_task.obj_dim)
        self.max_len = max_len
        self.random_action_prob = random_action_prob
        self.batch_size = batch_size
        self.reward_min = reward_min
        self.therm_n_bins = therm_n_bins
        self.temp_use_therm = temp_use_therm
        self.pref_use_therm = pref_use_therm
        self.gen_clip = gen_clip
        self.sampling_temp = sampling_temp
        self.sample_beta = sample_beta

        self.bb_task = hydra.utils.instantiate(bb_task, tokenizer=tokenizer, candidate_pool=[])

        self.encoder_config = encoder
        self.encoder = hydra.utils.instantiate(encoder, tokenizer=tokenizer)
        
        # self.encoder_obj = encoder_obj
        pref_dim = self.therm_n_bins * self.bb_task.obj_dim if self.pref_use_therm else self.bb_task.obj_dim
        temp_dim = self.therm_n_bins if self.temp_use_therm else 1
        self.model = hydra.utils.instantiate(model, cond_dim=pref_dim+temp_dim)


        self.surrogate_config = surrogate
        self.surrogate_model = hydra.utils.instantiate(surrogate, tokenizer=self.encoder.tokenizer,
                                                       encoder=self.encoder)
        self.acquisition = acquisition

        self.tokenizer = tokenizer
        # self = dotdict(CONFIG["gfn"])
        # self.model_args = dotdict(CONFIG["model"])
        # pref_dim = self.therm_n_bins * num_fs if self.pref_use_therm else num_fs
        # temp_dim = self.therm_n_bins if self.temp_use_therm else 1
        # self.model = CondGFNTransformer(self.model_args, pref_dim + temp_dim)
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
        self.simplex = generate_simplex(self.bb_task.obj_dim, 10)
        self.active_candidates = None
        self.active_targets = None
        self.train_split = DataSplit()
        self.val_split = DataSplit()
        self.test_split = DataSplit()

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
        is_feasible = self.bb_task.is_feasible(candidate_pool)
        pool_candidates = candidate_pool[is_feasible]
        pool_targets = pool_targets[is_feasible]
        pool_seqs = np.array([p_cand.mutant_residue_seq for p_cand in pool_candidates])

        self.active_candidates, self.active_targets = pool_candidates, pool_targets
        self.active_seqs = pool_seqs

        pareto_candidates, pareto_targets = pareto_frontier(new_seqs, new_targets)
        pareto_seqs = np.array([p_cand for p_cand in pareto_candidates])
        pareto_cand_history = pareto_candidates.copy()
        pareto_seq_history = pareto_seqs.copy()
        pareto_target_history = pareto_targets.copy()
        norm_pareto_targets = hypercube_transform(pareto_targets)
        self._ref_point = -infer_reference_point(-torch.tensor(norm_pareto_targets)).numpy()
        print(self._ref_point)
        rescaled_ref_point = hypercube_transform.inv_transform(self._ref_point.copy())

        # logging setup
        total_bb_evals = 0
        start_time = time.time()
        round_idx = 0
        self._log_candidates(pareto_candidates, pareto_targets, round_idx, log_prefix)
        metrics = self._log_optimizer_metrics(norm_pareto_targets, round_idx, total_bb_evals, start_time, log_prefix)

        print('\n best candidates')
        obj_vals = {f'obj_val_{i}': pareto_targets[:, i].min() for i in range(self.bb_task.obj_dim)}
        print(pd.DataFrame([obj_vals]).to_markdown(floatfmt='.4f'))

        for round_idx in range(1, self.num_rounds + 1):
            metrics = {}

            print(rescaled_ref_point)

            print('\n---- fitting surrogate model ----')
            # acquisition fns assume maximization so we normalize and negate targets here
            z_score_transform = Normalizer(all_targets.mean(0), all_targets.std(0))

            tgt_transform = lambda x: -z_score_transform(x)
            transformed_ref_point = tgt_transform(rescaled_ref_point)

            new_split = DataSplit(new_seqs, new_targets)
            holdout_ratio = self.surrogate_model.holdout_ratio
            all_splits = update_splits(
                self.train_split, self.val_split, self.test_split, new_split, holdout_ratio,
            )
            self.train_split, self.val_split, self.test_split = all_splits

            X_train, Y_train = self.train_split.inputs, tgt_transform(self.train_split.targets)
            X_val, Y_val = self.val_split.inputs, tgt_transform(self.val_split.targets)
            X_test, Y_test = self.test_split.inputs, tgt_transform(self.test_split.targets)
            self.tgt_transform = tgt_transform
            records = self.surrogate_model.fit(
                X_train, Y_train, X_val, Y_val, X_test, Y_test,
                encoder_obj=self.encoder_obj, resampling_temp=None
            )

            # log result
            last_entry = {key.split('/')[-1]: val for key, val in records[-1].items()}
            best_idx = last_entry['best_epoch']
            best_entry = {key.split('/')[-1]: val for key, val in records[best_idx].items()}
            print(pd.DataFrame([best_entry]).to_markdown(floatfmt='.4f'))
            metrics.update(dict(
                test_rmse=best_entry['test_rmse'],
                test_nll=best_entry['test_nll'],
                test_s_rho=best_entry['test_s_rho'],
                test_ece=best_entry['test_ece'],
                test_post_var=best_entry['test_post_var'],
                test_perplexity=best_entry['test_perplexity'],
                round_idx=round_idx,
                num_bb_evals=total_bb_evals,
                num_train=X_train.shape[0],
                time_elapsed=time.time() - start_time,
            ))
            metrics = {
                '/'.join((log_prefix, 'opt_metrics', key)): val for key, val in metrics.items()
            }
            wandb.log(metrics)

            print('\n---- optimizing candidates ----')
            gfn_records = self.train()
            print(gfn_records[-10])
            new_candidates = []
            r_scores = [] 
            for prefs in self.simplex:
                prefs_enc = thermometer(torch.from_numpy(prefs), self.therm_n_bins, 0, 1) if self.pref_use_therm else torch.from_numpy(pref)
                temp_enc = thermometer(torch.from_numpy(np.array([self.sample_beta])), self.therm_n_bins, 0, 1) if self.temp_use_therm else torch.from_numpy(np.array([self.sample_beta]))
                samples, _ = self.sample(self.batch_size // 2, prefs_enc, temp_enc)
                r = self.process_reward(samples, prefs, self.sample_beta)
                idx = r.argmax()
                new_candidates.append(samples[idx])
                r_scores.append(r.max().item())
            r_scores = np.array(r_scores)
            idx = np.random.choice(len(new_candidates), size=self.batch_size)
            new_candidates = np.array(new_candidates)[idx]
            r_scores = r_scores[idx]

            # logging
            metrics = dict(
                acq_val=r_scores.mean(),
                round_idx=round_idx,
                num_bb_evals=total_bb_evals,
                time_elapsed=time.time() - start_time,
            )
            print(pd.DataFrame([metrics]).to_markdown(floatfmt='.4f'))
            metrics = {'/'.join((log_prefix, 'opt_metrics', key)): val for key, val in metrics.items()}
            wandb.log(metrics)

            print('\n---- querying objective function ----')
            # new_candidates = self.bb_task.make_new_candidates(base_candidates, new_seqs)

            new_seqs = new_candidates.copy()
            new_targets = self.bb_task.score(new_candidates)
            all_targets = np.concatenate((all_targets, new_targets))
            all_seqs = np.concatenate((all_seqs, new_seqs))

            # overall Pareto frontier including terminal candidates
            pareto_candidates, pareto_targets = pareto_frontier(
                np.concatenate((pareto_candidates, new_candidates)),
                np.concatenate((pareto_targets, new_targets)),
            )
            pareto_seqs = np.array([p_cand for p_cand in pareto_candidates])

            print('\n new candidates')
            obj_vals = {f'obj_val_{i}': new_targets[:, i].min() for i in range(self.bb_task.obj_dim)}
            print(pd.DataFrame([obj_vals]).to_markdown(floatfmt='.4f'))

            print('\n best candidates')
            obj_vals = {f'obj_val_{i}': pareto_targets[:, i].min() for i in range(self.bb_task.obj_dim)}
            print(pd.DataFrame([obj_vals]).to_markdown(floatfmt='.4f'))

            # store good candidates for backtracking
            par_is_new = np.in1d(pareto_seqs, pareto_seq_history, invert=True)
            pareto_cand_history = safe_np_cat([pareto_cand_history, pareto_candidates[par_is_new]])
            pareto_seq_history = safe_np_cat([pareto_seq_history, pareto_seqs[par_is_new]])
            pareto_target_history = safe_np_cat([pareto_target_history, pareto_targets[par_is_new]])

            # logging
            norm_pareto_targets = hypercube_transform(pareto_targets)
            total_bb_evals += batch_size
            self._log_candidates(new_candidates, new_targets, round_idx, log_prefix)
            metrics = self._log_optimizer_metrics(
                norm_pareto_targets, round_idx, total_bb_evals, start_time, log_prefix
            )

        return metrics

    def train(self):
        losses = []
        pref_alpha, temp_shape, temp_scale = 1.5, 2, 1
        pb = tqdm(range(self.train_steps))
        for i in pb:
            loss = self.train_step(self.batch_size, pref_alpha, temp_shape, temp_scale)
            losses.append(loss)
            if i % 25 == 0:
                val_loss = self.val_step(self.batch_size, pref_alpha, temp_shape, temp_scale)
                # print(sum(losses[-10:]) / 10)
            pb.set_description("Val Loss: {}, Train Loss: {}".format(val_loss, sum(losses[-10:]) / 10))
        return losses

    def val_step(self, batch_size, pref_alpha, temp_shape, temp_scale):
        # import pdb; pdb.set_trace();
        prefs = np.random.dirichlet([pref_alpha]*self.num_props)
        temp = np.random.gamma(temp_shape, temp_scale)
        if self.pref_use_therm:
            prefs_enc = thermometer(torch.from_numpy(prefs), self.therm_n_bins, 0, 1)
        if self.temp_use_therm:
            temp_enc = thermometer(torch.from_numpy(np.array([temp])), self.therm_n_bins, 0, 32)
        cond_var = torch.cat((prefs_enc.view(-1), temp_enc.view(-1))).float().to(self.device)
        num_batches = len(self.val_split.inputs) // self.batch_size
        losses = 0
        for i in range(num_batches):
            x = self.val_split.inputs[i * self.batch_size:(i+1) * self.batch_size]
            # rs = self.tgt_transform(self.val_split.targets[i * self.batch_size:(i+1) * self.batch_size])
            lens = torch.tensor([len(z) for z in x]).long().to(self.device)
            # lps = torch.zeros(len(x))
            x = str_to_tokens(x, self.encoder.tokenizer).to(self.device)
            x = x.transpose(1, 0)
            mask = x.eq(self.encoder.tokenizer.padding_idx)
            mask = torch.cat((torch.zeros((1, mask.shape[1])).to(mask.device), mask), axis=0).bool()
            with torch.no_grad():
                logits = self.model(x, torch.tile(cond_var, (1, x.shape[1], 1)), mask=mask.transpose(1,0), return_all=True, lens=lens, logsoftmax=True)
            seq_logits = (logits.reshape(-1, 21)[torch.arange(x.shape[0] * x.shape[1], device=self.device), (x.reshape(-1)-4).clamp(0)].reshape(x.shape) * mask[1:, :].logical_not().float()).sum(0)
            r = self.process_reward(self.val_split.inputs[i * self.batch_size:(i+1) * self.batch_size], prefs, temp).to(seq_logits.device)
            loss = (seq_logits - temp * r.clamp(min=self.reward_min).log()).pow(2).mean()

            losses += loss.item()
        return losses / num_batches

    def sample(self, episodes, prefs, temp):
        states = [''] * episodes
        traj_dones = lists(episodes)

        traj_logprob = torch.zeros(episodes).to(self.device)
        cond_var = torch.cat((prefs.view(-1), temp.view(-1))).float().to(self.device)
        
        for t in (range(self.max_len) if episodes > 0 else []):
            active_indices = np.int32([i for i in range(episodes)
                                       if not states[i].endswith(self.eos_char)])
            # x = torch.tensor([self.tokenizer.encode(states[i], use_sep=False) for i in active_indices]).to(self.device, torch.long)
            x = str_to_tokens([states[i] for i in active_indices], self.encoder.tokenizer).to(self.encoder.device)
            x = x.transpose(1,0)
            lens = torch.tensor([len(i) for i in states
                                 if not i.endswith(self.eos_char)]).long().to(self.device)
            logits = self.model(x, torch.tile(cond_var, (1, x.shape[1], 1)), mask=None, lens=lens)
            
            if t == 0:
                logits[:, 0] = -1000 # Prevent model from stopping
                                     # without having output anything
                traj_logprob += self.model.Z(cond_var)

            cat = Categorical(logits=logits / self.sampling_temp)

            actions = cat.sample()
            if self.random_action_prob > 0:
                for i in range(actions.shape[0]):
                    if np.random.uniform(0,1) < self.random_action_prob:
                        actions[i] = torch.tensor(np.random.randint(t == 0, logits.shape[1])).to(self.device)
            log_prob = cat.log_prob(actions)
            chars = [self.tokenizer.convert_id_to_token(i) for i in actions + 4]

            # Append predicted characters for active trajectories
            for b_i, i, c, a in zip(range(len(actions)), active_indices, chars, actions):
                traj_logprob[i] += log_prob[b_i]
                if c == self.eos_char or t == self.max_len - 1:
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

    def train_step(self, batch_size, pref_alpha, temp_shape, temp_scale):
        # generate cond_var randomly
        prefs = np.random.dirichlet([pref_alpha]*self.num_props)
        temp = np.random.gamma(temp_shape, temp_scale)
        if self.pref_use_therm:
            prefs_enc = thermometer(torch.from_numpy(prefs), self.therm_n_bins, 0, 1)
        if self.temp_use_therm:
            temp_enc = thermometer(torch.from_numpy(np.array([temp])), self.therm_n_bins, 0, 32)
        
        states, logprobs = self.sample(batch_size, prefs_enc, temp_enc)
        # remove [SEP] from the end
        for i in range(len(states)):
            if states[i].endswith(self.eos_char):
                states[i] = states[i][:-5]
        
        r = self.process_reward(states, prefs, temp).to(self.device)
        self.opt.zero_grad()
        self.opt_Z.zero_grad()        
        # TB Loss
        loss = (logprobs - temp * r.clamp(min=self.reward_min).log()).pow(2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gen_clip)
        self.opt.step()
        self.opt_Z.step()
        return loss.item()
    
    def process_reward(self, seqs, prefs, temp):
        toks = str_to_tokens(seqs, self.encoder.tokenizer).to(self.encoder.device)
        feats, src_mask = self.encoder.get_token_features(toks)
        pooled_features = self.encoder.pool_features(feats, src_mask)

        with torch.no_grad():
            ys = self.surrogate_model.predict(pooled_features[1])
        
        reward = (torch.tensor(prefs) * (ys[1].cpu() + 1)).sum(axis=1)
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
