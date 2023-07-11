import re

import numpy as np

import design_bench
from lambo.candidate import StringCandidate
from lambo.tasks.base_task import BaseTask
from lambo.utils import random_proteins, apply_mutation, mutation_list, str_to_tokens

AMINO_ACIDS = ["A","R","N","D","C","E","Q","G","H","I","L","K","M","F","P","S","T","W","Y","V"]

class GFPTask(BaseTask):
    def __init__(self, min_len, num_start_examples, tokenizer,
                 candidate_pool, obj_dim, transform=lambda x: x, **kwargs):
        super().__init__(tokenizer, candidate_pool, obj_dim, transform, **kwargs)
        self.task = design_bench.make('GFP-Transformer-v0')
        self.min_len = min_len
        self.num_start_examples = num_start_examples
        self.max_reward_per_dim = kwargs['max_score_per_dim']

    def _process_x(self, data):
        processed_data = []
        for i in range(len(data)):
            processed_data.append("".join([AMINO_ACIDS[c] for c in data[i]]))
        return processed_data

    def task_setup(self, *args, **kwargs):
        x = np.array(self._process_x(self.task.x))

        y = self.task.y.reshape(-1, 1)
        # self.y_min = y.min()
        # self.y_max = y.max() 
        self.y_min = 1.2834193
        self.y_max = 4.123109
        y = (y - self.y_min) / (self.y_max - self.y_min)
        all_seqs = x[:self.num_start_examples]
        all_targets = y[:self.num_start_examples]

        base_candidates = np.array([
            StringCandidate(seq, mutation_list=[], tokenizer=self.tokenizer) for seq in all_seqs
        ]).reshape(-1)
        base_targets = all_targets.copy()

        return base_candidates, base_targets, all_seqs, all_targets

    def _evaluate(self, x, out, *args, **kwargs):
        assert x.ndim == 2
        x_cands, x_seqs, f_vals = [], [], []
        for query_pt in x:
            cand_idx, mut_pos, mut_res_idx, op_idx = query_pt
            op_type = self.op_types[op_idx]
            base_candidate = self.candidate_pool[cand_idx]
            base_seq = base_candidate.mutant_residue_seq
            mut_res = self.tokenizer.sampling_vocab[mut_res_idx]
            # TODO add support for insertion and deletion here
            # mutation_ops = [base_candidate.new_mutation(mut_pos, mut_res, mutation_type='sub')]
            mut_seq = apply_mutation(base_seq, mut_pos, mut_res, op_type, self.tokenizer)
            mutation_ops = mutation_list(base_seq, mut_seq, self.tokenizer)
            candidate = base_candidate.new_candidate(mutation_ops, self.tokenizer)
            x_cands.append(candidate)
            x_seqs.append(candidate.mutant_residue_seq)
        x_seqs = np.array(x_seqs).reshape(-1)
        x_cands = np.array(x_cands).reshape(-1)

        out["X_cand"] = x_cands
        out["X_seq"] = x_seqs
        out["F"] = self.transform(self.score(x_cands))

    def score(self, candidates):
        import pdb; pdb.set_trace();
        if type(candidates[0]) == StringCandidate:
            str_array = np.array([cand.mutant_residue_seq for cand in candidates])
        else:
            str_array = np.array(candidates)
        scores = []
        # for i in range(int(np.ceil(len(x) / batch_size))):
        s = self.task.predict(np.array(str_to_tokens(str_array, self.tokenizer)[:, 1:-1]) - 5).reshape(-1)
            # s = (s-self.y_min) / (self.y_max - self.y_min)
            # scores += s.tolist()
        return np.float32(s)
        return scores