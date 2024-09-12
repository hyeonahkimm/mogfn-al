import hydra
import wandb
import pandas as pd
import time
import numpy as np
import torch
import random
from tqdm import tqdm
from torch.nn import functional as F
from torch.distributions import Categorical
# import ray

from pymoo.factory import get_performance_indicator

from botorch.utils.multi_objective import infer_reference_point

from lambo.models.mlm import sample_tokens, evaluate_windows
from lambo.optimizers.pymoo import pareto_frontier, Normalizer
from lambo.models.shared_elements import check_early_stopping
from lambo.utils import weighted_resampling, DataSplit, update_splits, str_to_tokens, tokens_to_str, safe_np_cat, generate_simplex, thermometer
from lambo.models.lanmt import corrupt_tok_idxs
from lambo.candidate import StringCandidate


# @ray.remote
# def create_new_cand(task, base_cands, new_seq):
#     return task.make_new_candidates(base_cands, new_seq)


class MOGFNSeq(object):
    '''
    Here instead of generating sequence from scratch, we generate modifications for the current pareto_front
    '''
    def __init__(self, bb_task, tokenizer, encoder, surrogate, acquisition, num_rounds, 
                 num_opt_steps, concentrate_pool, resampling_weight, encoder_obj, model, **kwargs):

        self.tokenizer = tokenizer
        self.num_rounds = num_rounds
        self.concentrate_pool = concentrate_pool
        self._hv_ref = None
        self._ref_point = np.array([1] * bb_task.obj_dim)
        self.obj_dim = bb_task.obj_dim
        self.max_num_edits = bb_task.max_num_edits

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


        self.bb_task = hydra.utils.instantiate(bb_task, tokenizer=tokenizer, candidate_pool=[])

        self.encoder_config = encoder
        self.encoder = hydra.utils.instantiate(encoder, tokenizer=tokenizer)
        self.encoder_obj = encoder_obj

        self.surrogate_config = surrogate
        self.surrogate_model = hydra.utils.instantiate(surrogate, tokenizer=self.encoder.tokenizer,
                                                       encoder=self.encoder)
        self.acquisition = acquisition
        self.num_opt_steps = num_opt_steps
        self.resampling_weight = resampling_weight
        self.load_gfn_params(kwargs, model)

        self.active_candidates = None
        self.active_targets = None
        self.train_split = DataSplit()
        self.val_split = DataSplit()
        self.test_split = DataSplit()

    def load_gfn_params(self, kwargs, model):
        self.val_batch_size = kwargs["val_batch_size"]        
        self.random_action_prob = kwargs["random_action_prob"]
        self.train_batch_size = kwargs["train_batch_size"]
        self.reward_min = kwargs["reward_min"]
        self.therm_n_bins = kwargs["therm_n_bins"]
        self.beta_use_therm = kwargs["beta_use_therm"]
        self.pref_use_therm = kwargs["pref_use_therm"]
        self.gen_clip = kwargs["gen_clip"]
        self.sampling_temp = kwargs["sampling_temp"]
        self.sample_beta = kwargs["sample_beta"]
        self.beta_cond = kwargs["beta_cond"]
        self.pref_cond = kwargs["pref_cond"]
        self.beta_scale = kwargs["beta_scale"]
        self.beta_shape = kwargs["beta_shape"]
        self.pref_alpha = kwargs["pref_alpha"]
        self.beta_max = kwargs["beta_max"]
        self.beta_sched = kwargs["beta_sched"]
        self.reward_type = kwargs["reward_type"]
        self.eval_freq = kwargs["eval_freq"]
        self.offline_gamma = kwargs["offline_gamma"]
        self.k = kwargs["k"]
        self.num_samples = kwargs["num_eval_samples"]
        self.eos_char = "[SEP]"
        self.pad_tok = self.tokenizer.convert_token_to_id("[PAD]")
        self.simplex = generate_simplex(self.obj_dim, kwargs["simplex_bins"])
        self.max_len = kwargs["max_len"] - 2 # -2 because the lambo tasks count BOS and EOS tokens as well
        self.min_len = kwargs["min_len"] if kwargs["min_len"] else 2
        pref_dim = self.therm_n_bins * self.obj_dim if self.pref_use_therm else self.obj_dim
        beta_dim = self.therm_n_bins if self.beta_use_therm else 1
        cond_dim = pref_dim + beta_dim if self.beta_cond else pref_dim
        share_encoder = kwargs.get("share_encoder", False)
        freeze_encoder = kwargs.get("freeze_encoder", False)
        self.use_acqf = kwargs.get("use_acqf", False)
        
        ## Hyeonah
        self.use_trust_region = kwargs.get("use_trust_region", False)
        self.min_radius = kwargs.get("min_radius", 0.1)
        self.region_decay = kwargs.get("region_decay", 1.1)
        
        self.model_cfg = model
        self.model = hydra.utils.instantiate(model, cond_dim=cond_dim, use_cond=(self.beta_cond or self.pref_cond),
                                             encoder=self.encoder if share_encoder else None)

        self.model.to(self.device)
        self.opt = torch.optim.Adam(self.model.model_params(), kwargs["pi_lr"], weight_decay=kwargs["wd"],
                            betas=(0.9, 0.999))
        self.opt_Z = torch.optim.Adam(self.model.Z_param(), kwargs["z_lr"], weight_decay=kwargs["wd"],
                            betas=(0.9, 0.999))


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

        pareto_candidates, pareto_targets = pareto_frontier(self.active_candidates, self.active_targets)
        pareto_seqs = np.array([p_cand.mutant_residue_seq for p_cand in pareto_candidates])
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
            self.sample_beta -= max(self.beta_sched, 2)
            # import pdb; pdb.set_trace();
            # contract active pool to current Pareto frontier
            if (self.concentrate_pool > 0 and round_idx % self.concentrate_pool == 0) or self.latent_init == 'perturb_pareto':
                self.active_candidates, self.active_targets = pareto_frontier(
                    self.active_candidates, self.active_targets
                )
                self.active_seqs = np.array([a_cand.mutant_residue_seq for a_cand in self.active_candidates])
                print(f'\nactive set contracted to {self.active_candidates.shape[0]} pareto points')
            # augment active set with old pareto points
            if self.active_candidates.shape[0] < batch_size:
                num_samples = min(batch_size, pareto_cand_history.shape[0])
                num_backtrack = min(num_samples, batch_size - self.active_candidates.shape[0])
                _, weights, _ = weighted_resampling(pareto_target_history, k=self.resampling_weight)
                hist_idxs = np.random.choice(
                    np.arange(pareto_cand_history.shape[0]), num_samples, p=weights, replace=False
                )
                is_active = np.in1d(pareto_seq_history[hist_idxs], self.active_seqs)
                hist_idxs = hist_idxs[~is_active]
                if hist_idxs.size > 0:
                    hist_idxs = hist_idxs[:num_backtrack]
                    backtrack_candidates = pareto_cand_history[hist_idxs]
                    backtrack_targets = pareto_target_history[hist_idxs]
                    backtrack_seqs = pareto_seq_history[hist_idxs]
                    self.active_candidates = np.concatenate((self.active_candidates, backtrack_candidates))
                    self.active_targets = np.concatenate((self.active_targets, backtrack_targets))
                    self.active_seqs = np.concatenate((self.active_seqs, backtrack_seqs))
                    print(f'active set augmented with {backtrack_candidates.shape[0]} backtrack points')
            # augment active set with random points
            if self.active_candidates.shape[0] < batch_size:
                num_samples = min(batch_size, pool_candidates.shape[0])
                num_rand = min(num_samples, batch_size - self.active_candidates.shape[0])
                _, weights, _ = weighted_resampling(pool_targets, k=self.resampling_weight)
                rand_idxs = np.random.choice(
                    np.arange(pool_candidates.shape[0]), num_samples, p=weights, replace=False
                )
                is_active = np.in1d(pool_seqs[rand_idxs], self.active_seqs)
                rand_idxs = rand_idxs[~is_active][:num_rand]
                rand_candidates = pool_candidates[rand_idxs]
                rand_targets = pool_targets[rand_idxs]
                rand_seqs = pool_seqs[rand_idxs]
                self.active_candidates = np.concatenate((self.active_candidates, rand_candidates))
                self.active_targets = np.concatenate((self.active_targets, rand_targets))
                self.active_seqs = np.concatenate((self.active_seqs, rand_seqs))
                print(f'active set augmented with {rand_candidates.shape[0]} random points')

            print(rescaled_ref_point)
            print(self.active_targets)
            for seq in self.active_seqs:
                if hasattr(self.tokenizer, 'to_smiles'):
                    print(self.tokenizer.to_smiles(seq))
                else:
                    print(seq)

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

            baseline_seqs = np.array([cand.mutant_residue_seq for cand in self.active_candidates])
            baseline_targets = self.active_targets
            baseline_seqs, baseline_targets = pareto_frontier(baseline_seqs, baseline_targets)
            baseline_targets = tgt_transform(baseline_targets)

            acq_fn = hydra.utils.instantiate(
                self.acquisition,
                X_baseline=baseline_seqs,
                known_targets=torch.tensor(baseline_targets).to(self.surrogate_model.device),
                surrogate=self.surrogate_model,
                ref_point=torch.tensor(transformed_ref_point).to(self.surrogate_model.device),
                obj_dim=self.bb_task.obj_dim,
            )

            if self.use_acqf:
                task = AcqFnTask(acq_fn)
            else:
                task = Task(self.surrogate_model, max_val=Y_train.max(0))
            
            print('\n---- optimizing candidates ----')
            train_losses, train_rewards, val_losses = [], [], []
            start_candidates = np.random.choice(self.active_candidates, size=self.train_batch_size, replace=True)
            eval_start_candidates = self.active_candidates

            eval_start_states = np.array([c.mutant_residue_seq for c in eval_start_candidates])
            start_states = np.array([c.mutant_residue_seq for c in start_candidates])
            hv, rs, val_loss = 0., np.zeros(self.obj_dim), 0.
            pb = tqdm(range(self.num_opt_steps))
            desc_str = "Evaluation := Reward: {:.3f} HV: {:.3f} | Validation:= Loss {:.3f} | Train := Loss: {:.3f} Rewards: {:.3f}"
            pb.set_description(desc_str.format(0,0,0,0,0))
            # import pdb; pdb.set_trace();
            
            for i in pb:
                # import pdb;pdb.set_trace();
                loss, r = self.train_step(start_states, task, self.train_batch_size)
                train_losses.append(loss)
                train_rewards.append(r)
                
                # if i != 0 and i % self.eval_freq == 0:
                # if i % self.eval_freq == 0:
                #     with torch.no_grad():
                #         # samples, all_rews, rs, mo_metrics, topk_metrics = self.evaluation(task, plot=True)
                #         val_loss = self.val_step(task, self.val_batch_size)

                    # add early stopping logic? 

                pb.set_description(desc_str.format(rs.mean(),hv,val_loss,sum(train_losses[-10:]) / 10 , sum(train_rewards[-10:]) / 10))
            
            # import pdb; pdb.set_trace();
            new_seqs, r_scores = self.sample_new_pareto_front(eval_start_states, task, batch_size)

            # generate new sequences
            # new_seq_batches = np.stack(new_seq_batches)
            # new_seq_scores = np.stack(new_seq_scores)
            # best_batch_idx = new_seq_scores.argmin()

            # base_candidates = base_cand_batches[best_batch_idx]
            # base_seqs = np.array([b_cand.mutant_residue_seq for b_cand in base_candidates])
            # new_seqs = new_seq_batches[best_batch_idx]
            

            # logging
            metrics = dict(
                acq_val=r_scores.mean().item(),
                round_idx=round_idx,
                num_bb_evals=total_bb_evals,
                time_elapsed=time.time() - start_time,
            )
            print(pd.DataFrame([metrics]).to_markdown(floatfmt='.4f'))
            metrics = {'/'.join((log_prefix, 'opt_metrics', key)): val for key, val in metrics.items()}
            wandb.log(metrics)

            print('\n---- querying objective function ----')
            # import pdb; pdb.set_trace();
            # new_candidates = np.array([type(self.active_candidates[0])(seq, [], self.tokenizer) for seq in new_seqs])
            
            new_candidates = self.bb_task.make_new_candidates(eval_start_candidates, new_seqs)
            # new_candidates = [create_new_cand.remote(self.bb_task, np.array([eval_start_candidates[i]]), np.array([new_seqs[i]])) for i in range(len(eval_start_candidates))]
            # new_candidates = ray.get(new_candidates)
            # self.bb_task.make_new_candidates(eval_start_candidates, new_seqs)

            # filter infeasible candidates
            is_feasible = self.bb_task.is_feasible(new_candidates)
            # base_candidates = base_candidates[is_feasible]
            # base_seqs = base_seqs[is_feasible]
            new_seqs = new_seqs[is_feasible]
            new_candidates = new_candidates[is_feasible]
            # new_tokens = new_tokens[is_feasible]
            if new_candidates.size == 0:
                print('no new candidates')
                continue

            # filter duplicate candidates
            new_seqs, unique_idxs = np.unique(new_seqs, return_index=True)
            # base_candidates = base_candidates[unique_idxs]
            # base_seqs = base_seqs[unique_idxs]
            new_candidates = new_candidates[unique_idxs]

            # filter redundant candidates
            is_new = np.in1d(new_seqs, all_seqs, invert=True)
            # base_candidates = base_candidates[is_new]
            # base_seqs = base_seqs[is_new]
            new_seqs = new_seqs[is_new]
            new_candidates = new_candidates[is_new]
            if new_candidates.size == 0:
                print('no new candidates')
                continue

            new_targets = self.bb_task.score(new_candidates)
            all_targets = np.concatenate((all_targets, new_targets))
            all_seqs = np.concatenate((all_seqs, new_seqs))

            for seq in new_seqs:
                if hasattr(self.tokenizer, 'to_smiles'):
                    print(self.tokenizer.to_smiles(seq))
                else:
                    print(seq)
            # import pdb; pdb.set_trace();
            # assert base_seqs.shape[0] == new_seqs.shape[0] and new_seqs.shape[0] == new_targets.shape[0]
            # for b_cand, n_cand, f_val in zip(base_candidates, new_candidates, new_targets):
            #     print(f'{len(b_cand)} --> {len(n_cand)}: {f_val}')

            pool_candidates = np.concatenate((pool_candidates, new_candidates))
            pool_targets = np.concatenate((pool_targets, new_targets))
            pool_seqs = np.concatenate((pool_seqs, new_seqs))

            # augment active pool with candidates that can be mutated again
            self.active_candidates = np.concatenate((self.active_candidates, new_candidates))
            self.active_targets = np.concatenate((self.active_targets, new_targets))
            self.active_seqs = np.concatenate((self.active_seqs, new_seqs))

            # overall Pareto frontier including terminal candidates
            pareto_candidates, pareto_targets = pareto_frontier(
                np.concatenate((pareto_candidates, new_candidates)),
                np.concatenate((pareto_targets, new_targets)),
            )
            pareto_seqs = np.array([p_cand.mutant_residue_seq for p_cand in pareto_candidates])

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
    
    def sample_new_pareto_front(self, start_states, task, batch_size):
        if self.pref_cond:
            new_candidates = [[] for _ in range(len(start_states))]
            r_scores = [[] for _ in range(len(start_states))] 
            all_rewards = [[] for _ in range(len(start_states))]
            for prefs in self.simplex:
                cond_var, (_, beta) = self._get_condition_var(prefs=prefs, train=False, bs=len(start_states))
                samples, _ = self.sample(start_states, cond_var,len(start_states), train=False)
                rewards = task.score(samples)
                r = self.process_reward(samples, prefs, task, rewards=rewards)
                # topk metrics
                # topk_r, topk_idx = torch.topk(r, self.k)
                # samples = np.array(samples)
                # topk_seq = samples[topk_idx].tolist()
                # edit_dist = mean_pairwise_distances(topk_seq)
                # topk_rs.append(topk_r.mean().item())
                # topk_div.append(edit_dist)
                
                for i in range(len(samples)):
                    new_candidates[i].append(samples[i])
                    all_rewards[i].append(rewards[i])
                    r_scores[i].append(r[i].item())

                # top 1 metrics
                # max_idx = r.argmax()
                # new_candidates.append(samples[max_idx])
                # all_rewards.append(rewards[max_idx])
                # r_scores.append(r.max().item())

            r_scores = np.array(r_scores)
            all_rewards = np.array(all_rewards)
            new_candidates = np.array(new_candidates)
            gen_samples = []
            gen_rew = []
            for i in range(len(start_states)):
                idx = np.argmax(r_scores[i])
                gen_samples.append(new_candidates[i][idx])
                gen_rew.append(r_scores[i][idx])
            return np.array(gen_samples), np.array(gen_rew)
        else:
            new_candidates = [[] for _ in range(len(start_states))]
            r_scores = [[] for _ in range(len(start_states))] 
            all_rewards = [[] for _ in range(len(start_states))]
            for _ in range(self.num_samples):
                cond_var, (_, beta) = self._get_condition_var(prefs=None, train=False, bs=len(start_states))
                samples, _ = self.sample(start_states, cond_var,len(start_states), train=False)
                rewards = task.score(samples).cpu()
                r = self.process_reward(samples, None, task, rewards=rewards)

                for i in range(len(samples)):
                    new_candidates[i].append(samples[i])
                    all_rewards[i].append(rewards[i])
                    r_scores[i].append(r[i].item())
            
            r_scores = np.array(r_scores)
            all_rewards = np.array(all_rewards)
            new_candidates = np.array(new_candidates)
            gen_samples = []
            gen_rew = []
            for i in range(len(start_states)):
                idx = np.argmax(r_scores[i])
                gen_samples.append(new_candidates[i][idx])
                gen_rew.append(r_scores[i][idx])
            # idx = np.argsort(r_scores)[-batch_size:]
            return np.array(gen_samples), np.array(gen_rew)


    def sample_offline_data(self, size, prefs):
        w = -np.sum(prefs[None, :] * self.train_split.targets, axis=-1)
        return np.random.choice(self.train_split.inputs, size=size, replace=False, p = np.exp(w) / np.exp(w).sum(0))

    def train_step(self, start_states, task, batch_size):
        cond_var, (prefs, beta) = self._get_condition_var(train=True, bs=batch_size)
        states, logprobs = self.sample(start_states, cond_var, batch_size)
        if self.offline_gamma > 0 and int(self.offline_gamma * batch_size) > 0:
            offline_batch = self.sample_offline_data(int(self.offline_gamma * batch_size), prefs=prefs)
            cond_var, _ = self._get_condition_var(prefs=prefs, beta=beta, train=True, bs=int(self.offline_gamma * batch_size))
            offline_logprobs = self._get_log_prob(offline_batch, cond_var)
            logprobs = torch.cat((logprobs, offline_logprobs), axis=0)
            states = np.concatenate((states, offline_batch), axis=0)
        log_r = self.process_reward(states, prefs, task).to(self.device)

        self.opt.zero_grad()
        self.opt_Z.zero_grad()
        
        # TB Loss
        loss = (logprobs - beta * log_r).pow(2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gen_clip)
        self.opt.step()
        self.opt_Z.step()
        return loss.item(), log_r.mean()
    
    def val_step(self, task, batch_size):
        overall_loss = 0.
        num_batches = (len(self.val_split.inputs) // batch_size) + 1
        for i in range(num_batches):
            if i * batch_size >= len(self.val_split.inputs):
                break
            states = self.val_split.inputs[i * batch_size:(i+1) * batch_size]
            rews = task.score(self.val_split.inputs[i * batch_size:(i+1) * batch_size])
            losses = 0
            for pref in self.simplex:
                cond_var, (prefs, beta) = self._get_condition_var(prefs=pref, train=False, bs=len(states))
                logprobs = self._get_log_prob(states, cond_var)
                log_r = self.process_reward(None, prefs, task, rewards=rews).to(logprobs.device)
                loss = (logprobs - beta * log_r).pow(2).mean()

                losses += loss.item()
            overall_loss += (losses / num_batches)
        return overall_loss / len(self.simplex)

    def sample(self, start_states, cond_var=None, episodes=16, train=True):
        # import pdb; pdb.set_trace();
        states = start_states
        traj_logprob = torch.zeros(episodes).to(self.device)
        if cond_var is None:
            cond_var, _ = self._get_condition_var(train=train, bs=episodes)
        active_mask = torch.ones(episodes).bool().to(self.device)
        x = str_to_tokens(states, self.tokenizer).to(self.device).t()
        traj_lens = torch.zeros(episodes).long().to(self.device)
        lens = torch.tensor(np.array([len(s) for s in states])).long().to(self.device)
        uniform_pol = torch.empty(episodes).fill_(self.random_action_prob).to(self.device)
        updated = torch.empty(episodes).long().to(self.device)
        for t in (range(self.max_len-2) if episodes > 0 else []):
            # import pdb; pdb.set_trace();
            (pos_logits, tok_logits), (back_pos_logits, back_tok_logits) = self.model(x, cond_var, lens=lens, mask=None)
            
            # tokens in the sequence cannot 
            # tok_logits[:, 1:, 0] = -1000 # block all special tokens
            # tok_logits[:, 0, 1:] = -1000 # can't change cls to anything
            # tok_logits[:, 0, -] = -1000 # can't change cls to anything
            # import pdb; pdb.set_trace();
            pos_logits[:, lens+1] = -1000 # can't change last token
            pos_logits[x.t() == 0] = -1000
            log_pb = torch.zeros(episodes).to(pos_logits.device)
            if t > 0:
                if t == 1:
                    pos_logits = pos_logits.scatter(1, updated.unsqueeze(1),-1000)    
                    log_pb_pos = torch.zeros(episodes).to(pos_logits.device)
                else:
                    pos_logits = pos_logits.scatter(1, updated, -1000)
                    log_pb_pos = torch.log(torch.ones(episodes).to(pos_logits.device) / (updated != 0).sum(1))

                log_pb = log_pb_pos + torch.log(torch.ones_like(log_pb_pos) / self.model_cfg.num_actions)
                
            if t <= self.min_len:
                # pos 0 (cls token) indicates stop
                pos_logits[:, 0] = -1000 # Prevent model from stopping
                                     # without having output anything
                if t == 0:
                    traj_logprob += self.model.Z(cond_var)

            pos_dist = Categorical(logits=pos_logits / self.sampling_temp)
            pos_actions = pos_dist.sample()

            if train and self.random_action_prob > 0:
                pos_logits_temp = pos_logits.detach().clone() / 100
                pos_logits_temp[:, lens+1] = -1000 # can't change last token
                pos_logits_temp[x.t() == 0] = -1000
                if t > 0:
                    if t == 1:
                        pos_logits_temp = pos_logits_temp.scatter(1, updated.unsqueeze(1), -1000)    
                    else:
                        pos_logits_temp = pos_logits_temp.scatter(1, updated, -1000)
                pos_dist_temp = Categorical(logits=pos_logits_temp)
                pos_actions_temp = pos_dist_temp.sample()
                uniform_mix = torch.bernoulli(uniform_pol).bool()
                pos_actions = torch.where(uniform_mix, pos_actions_temp, pos_actions)
                # pos_actions = torch.where(uniform_mix, torch.randint(int(t <= self.min_len), pos_logits.shape[1]-1, (episodes, )).to(self.device), pos_actions)
            if (pos_actions > lens).any():
                import pdb; pdb.set_trace();
            tok_logits = tok_logits[torch.arange(tok_logits.shape[0]), pos_actions, :]
            tok_logits[pos_actions != 0, :1] = -1000 # block all special tokens
            tok_logits[pos_actions == 0, 1:] = -1000
            tok_logits[torch.arange(tok_logits.shape[0]), x.t()[torch.arange(x.shape[1]), pos_actions] - 4] = -1000
            tok_dist = Categorical(logits=tok_logits / self.sampling_temp)
            tok_actions = tok_dist.sample()

            if train and self.random_action_prob > 0:
                tok_logits_temp = tok_logits.detach().clone() / 100
                tok_logits_temp[pos_actions != 0, :1] = -1000 # block all special tokens
                tok_logits_temp[pos_actions == 0, 1:] = -1000
                tok_logits_temp[torch.arange(tok_logits.shape[0]), x.t()[torch.arange(x.shape[1]), pos_actions] - 4] = -1000
                tok_dist_temp = Categorical(logits=tok_logits_temp)
                tok_actions_temp = tok_dist_temp.sample()
                uniform_mix = torch.bernoulli(uniform_pol).bool()
                tok_actions_bck = tok_actions.clone()
                tok_actions = torch.where(uniform_mix, tok_actions_temp, tok_actions)
            if (tok_dist.log_prob(tok_actions) < -1000).any():
                # import pdb; pdb.set_trace();
                uniform_mix = torch.bernoulli(uniform_pol).bool()
                tok_actions = torch.where(uniform_mix, tok_actions_temp, tok_actions_bck)
                
            log_prob = (pos_dist.log_prob(pos_actions) + tok_dist.log_prob(tok_actions)) * active_mask
            
            #     import pdb; pdb.set_trace();

            traj_logprob += (log_prob - log_pb)
            traj_lens += torch.where(active_mask, torch.ones_like(lens), torch.zeros_like(lens))

            active_mask = torch.where(active_mask, pos_actions != 0, active_mask)
            # Apply action function
            # if active_mask.sum() < 128:
            #     import pdb;pdb.set_trace();
            tok_actions = torch.where(active_mask, tok_actions+4, x.t()[torch.arange(x.shape[1]), pos_actions])
            tok_actions = torch.where(pos_actions==0, x.t()[torch.arange(x.shape[1]), 0], tok_actions)
            x = x.t().scatter(1, pos_actions.unsqueeze(1), tok_actions.unsqueeze(1)).t()
            # print((x > 4).sum())
            # x = torch.where()
            # x[pos_actions, active_mask] = tok_actions[active_mask]
            if t > 0:
                updated = torch.column_stack((updated, pos_actions))
            else:
                updated = pos_actions
            # actions_apply = torch.where(torch.logical_not(active_mask), torch.zeros(episodes).to(self.device).long(), actions + 4)
            # x = torch.cat((x, actions_apply.unsqueeze(0)), axis=0)
            # x = self.apply_action(x, actions_apply)

            if active_mask.sum() == 0:
                break
        states = tokens_to_str(x.t(), self.tokenizer)
        # print(states, traj_lens, traj_logprob)
        if [len(s) for s in start_states] != [len(s) for s in states]:
            import pdb; pdb.set_trace();
        return states, traj_logprob   

    def process_reward(self, seqs, prefs, task, rewards=None):
        if rewards is None:
            rewards = task.score(np.array(seqs)).clip(min=self.reward_min)
        # print(seqs)
        if self.pref_cond:
            if self.reward_type == "convex":
                log_r = (torch.tensor(prefs) * (rewards)).sum(axis=1).clamp(min=self.reward_min).log()
            elif self.reward_type == "logconvex":
                log_r = (torch.tensor(prefs) * torch.tensor(rewards).clamp(min=self.reward_min).log()).sum(axis=1)
        else:
            # import pdb; pdb.set_trace();
            log_r = torch.tensor(rewards).clamp(min=self.reward_min).log()
        return log_r

    def _log_candidates(self, candidates, targets, round_idx, log_prefix):
        table_cols = ['round_idx', 'cand_uuid', 'cand_ancestor', 'cand_seq']
        table_cols.extend([f'obj_val_{idx}' for idx in range(self.bb_task.obj_dim)])
        for cand, obj in zip(candidates, targets):
            new_row = [round_idx, cand.uuid, cand.wild_name, cand.mutant_residue_seq]
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

    def _get_condition_var(self, prefs=None, beta=None, train=True, bs=None):
        if prefs is None:
            if not train:
                prefs = self.simplex[0]
            else:
                prefs = np.random.dirichlet([self.pref_alpha]*self.obj_dim)
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

    def _get_log_prob(self, states, cond_var):
        lens = torch.tensor([len(z) + 2 for z in states]).long().to(self.device)
        x = str_to_tokens(states, self.tokenizer).to(self.device).t()
        mask = x.eq(self.tokenizer.padding_idx)
        logits = self.model(x, cond_var, mask=mask.transpose(1,0), return_all=True, lens=lens, logsoftmax=True)
        seq_logits = (logits.reshape(-1, self.model_cfg.num_actions)[torch.arange(x.shape[0] * x.shape[1], device=self.device), (x.reshape(-1)-4).clamp(0)].reshape(x.shape) * mask.logical_not().float()).sum(0)
        seq_logits += self.model.Z(cond_var)
        return seq_logits

class Task():
    def __init__(self, model, max_val):
        self.model = model
        self.offset = max_val 
    
    def score(self, x):
        return -self.model.posterior(x).mean.cpu().numpy() + self.offset[None, :]


class AcqFnTask():
    def __init__(self, acq_fn):
        self.acq_fn = acq_fn
    
    def score(self, x):
        return self.acq_fn(x[:, None])
