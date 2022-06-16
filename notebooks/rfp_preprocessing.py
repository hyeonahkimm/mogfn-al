#!/usr/bin/env python
# coding: utf-8

# # RFP Data Preprocessing
# 
# This notebook prepares a pool of starting RFP wild variants with known structure for optimization.

# In[1]:


import pandas as pd
import pypdb
import requests
import json
import time
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from botorch.utils.multi_objective import pareto
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

from Bio.PDB.PDBList import PDBList
from Bio import PDB
from Bio.SeqUtils import seq1

from lambo.candidate import pdb_to_residues, FoldedCandidate
from lambo.tasks.proxy_rfp.foldx import FoldxManager, extract_chain
from lambo.utils import ResidueTokenizer

import warnings
warnings.filterwarnings("ignore")

sns.set(style='whitegrid', font_scale=1.75)

FPBASE_ENDPOINT = "https://www.fpbase.org/api/proteins/?"


# ## Merge name and sequence csv files
# Keep only 'red' proteins (at least 580 nm emission wavelength)

# In[2]:


fpbase_names = pd.read_csv('../lambo/assets/fpbase/fpbase_names.csv')
fpbase_seqs =  pd.read_csv('../lambo/assets/fpbase/fpbase_sequences.csv')
select_cols = ['Name', 'Seq', 'num_mutations']
fpbase_seqs = fpbase_seqs[select_cols]

fpbase_data = fpbase_names.merge(fpbase_seqs, on='Name', how='inner')  # combine csv files
rfp_data = fpbase_data[fpbase_data['Em max (nm)'] >= 580]  # only keep "red" proteins
rfp_data.reset_index(drop=True, inplace=True)

rfp_data.head()


# ## Get PDB ids for proteins with known structure from FPBase
# If multiple structures are available use the highest-resolution structure.
# 
# Record crystal pH value for FoldX repair.

# In[3]:


def fpbase_name_to_pdb_ids(name):
    name = name.split(' ')[0]
    print(f'---- {name} ----')
    query = f'{FPBASE_ENDPOINT}name__iexact={name}&format=json'
    response = requests.get(query)
    if not response.status_code == 200:
        print('query failed')
        return []
    
    query_results = json.loads(response.text)[0]['pdb']
    
    def get_resolution(info):
        try:
            return info['rcsb_entry_info']['resolution_combined'][0]
        except KeyError:
            return float('NaN')
        
    def get_pH(info):
        try:
            return info['expt1_crystal_grow']['ph']
        except KeyError:
            return float('NaN')
    
    if len(query_results) > 0:
        entry_info = [pypdb.get_info(pdb_id) for pdb_id in query_results]
        resolution = [get_resolution(info) for info in entry_info]
        pH = [get_pH(info) for info in entry_info]
        query_results = [
            (pdb_id, res, ph) for pdb_id, res, ph in zip(query_results, resolution, pH)
        ]
        query_results.sort(key=lambda datum: datum[1])
    
    return query_results


for row_idx, datum in rfp_data.iterrows():
    query_results = fpbase_name_to_pdb_ids(datum.Name)
    time.sleep(0.1)
    if len(query_results) == 0:
        continue
        
    rfp_data.loc[row_idx, 'pdb_id'] = query_results[0][0]
    rfp_data.loc[row_idx, 'pdb_resolution'] = query_results[0][1]
    rfp_data.loc[row_idx, 'crystal_pH'] = query_results[0][2]


# ## Clean up dataframe

# In[4]:


rfp_data.crystal_pH.fillna(7.0, inplace=True)
rfp_known_structures = rfp_data.dropna(subset=['pdb_id'])
rfp_known_structures.drop_duplicates(subset=['pdb_id'], inplace=True, keep=False)
rfp_known_structures.sort_values('Name', inplace=True)
rfp_known_structures.rename(columns={'Seq': 'fpbase_seq'}, inplace=True)

for _, datum in rfp_known_structures.iterrows():
    print(f'{datum.Name}: {datum.pdb_id} ({datum.pdb_resolution} nm)')


# ## Download structures from RCSB PDB

# In[5]:


pdb_list = PDBList()
pdb_list.download_pdb_files(
    rfp_known_structures.pdb_id,
    pdir='../lambo/assets/pdb',
    file_format="pdb"
)


# ## Extract longest chain in the structure
# Break ties with chain ID (i.e. use chain A by default)

# In[6]:


for row_idx, datum in rfp_known_structures.iterrows():
    pdb_path = f'../lambo/assets/pdb/pdb{datum.pdb_id.lower()}.ent'
    parser = PDB.PDBParser()
    pdb_path = Path(pdb_path)
    struct = parser.get_structure(pdb_path.stem, pdb_path)
    chain_residues = {
        chain.get_id(): seq1(''.join(x.resname for x in chain)) for chain in struct.get_chains()
    }
    chain_lengths = [
        (-len(seq.replace('X', '')), chain_id) for chain_id, seq in chain_residues.items()
    ]
    chain_lengths.sort()
    longest_chain = chain_lengths[0][1]
    rfp_known_structures.loc[row_idx, 'longest_chain'] = longest_chain
    extract_chain(pdb_path, longest_chain)

rfp_known_structures.longest_chain.value_counts()


# ## Repair single-chain structures with FoldX

# In[7]:


for _, datum in rfp_known_structures.iterrows():
    pdb_path = f'../lambo/assets/pdb/pdb{datum.pdb_id.lower()}_{datum.longest_chain}.pdb'
    work_dir = f'../lambo/assets/foldx/{datum.pdb_id.lower()}_{datum.longest_chain}/'
    print(f'---- {datum.Name}-{datum.longest_chain} ----')
    if os.path.exists(Path(work_dir) / 'wt_input_Repair.pdb'):
        print('file exists, skipping')
        continue
    else:
        os.makedirs(work_dir, exist_ok=True)
        FoldxManager(work_dir=work_dir, wt_pdb=pdb_path, skip_minimization=False,
                     ph=datum.crystal_pH)


# ## Validate repaired FoldX sequences
# Compare against FPBase and corresponding NCSB PDB chain. They won't match exactly.

# In[8]:


for row_idx, datum in rfp_known_structures.iterrows():
    fpbase_seq = datum.fpbase_seq
    
    pdb_path = f'../lambo/assets/pdb/pdb{datum.pdb_id.lower()}.ent'
    rcsb_seq, _ = pdb_to_residues(pdb_path, datum.longest_chain)
    rcsb_seq = rcsb_seq.replace('X', '')
    
    pdb_path = f'../lambo/assets/foldx/{datum.pdb_id.lower()}_{datum.longest_chain}/wt_input_Repair.pdb'
    foldx_seq, _ = pdb_to_residues(pdb_path, datum.longest_chain)
    rfp_known_structures.loc[row_idx, 'foldx_seq'] = str(foldx_seq)
    
    print(f'\n---- {datum.Name}-{datum.pdb_id}-{datum.longest_chain} ----')
    print(f'FPBase ({len(fpbase_seq)} residues): {fpbase_seq}\n')
    print(f'FoldX ({len(foldx_seq)} residues): {foldx_seq}\n')


# In[10]:


fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(1, 1, 1)

plt.hist(rfp_known_structures.fpbase_seq.apply(lambda seq: len(seq)), label='FPBase',
         zorder=1, bins=4)
plt.hist(rfp_known_structures.foldx_seq.apply(lambda seq: len(seq)), label='FoldX',
         alpha=0.6, zorder=2, bins=4)
plt.xticks(rotation='15')
ax.legend()
ax.set_xlabel('# residues')
ax.set_ylabel('count')
ax.set_title('RFP sequences w/ known structure')
plt.tight_layout()
# plt.savefig('./figures/rfp_known_structures_seq_len_plot.pdf')


# ## Estimate total energy and SASA

# In[13]:


for row_idx, datum in rfp_known_structures.iterrows():
    print(f'---- {datum.Name}-{datum.longest_chain} ----')
    work_dir = './tmp'
    pdb_path = f'../lambo/assets/foldx/{datum.pdb_id.lower()}_{datum.longest_chain}/wt_input_Repair.pdb'
    tokenizer = ResidueTokenizer()
    base_candidate = FoldedCandidate(work_dir, pdb_path, [], tokenizer=tokenizer,
                                     skip_minimization=True, chain=datum.longest_chain)
    rfp_known_structures.loc[row_idx, 'foldx_total_energy'] = base_candidate.wild_total_energy
    rfp_known_structures.loc[row_idx, 'SASA'] = base_candidate.wild_surface_area
    print(f'stability: {-base_candidate.wild_total_energy}')
    print(f'surface area: {base_candidate.wild_surface_area}\n')


# ## Evaluate fitness landscape

# In[14]:


# post-processing
mutant_surface_area = torch.tensor(rfp_known_structures.SASA.values)
sa_min = mutant_surface_area.min()
sa_range = mutant_surface_area.max() - sa_min
x_data = 2 * (mutant_surface_area - sa_min) / sa_range - 1

mutant_stability = -torch.tensor(rfp_known_structures.foldx_total_energy.values)
stab_min = mutant_stability.min()
stab_range = mutant_stability.max() - stab_min
y_data = 2 * (mutant_stability - stab_min) / stab_range - 1

obj = torch.stack([x_data, y_data], dim=-1)
pareto_mask = pareto.is_non_dominated(obj)

print('---- Non-dominated RFP variants ----')
select_cols = ['Name', 'SASA', 'foldx_total_energy']
select_df = rfp_known_structures.loc[pareto_mask.numpy(), select_cols]
select_df['stability'] = -select_df.foldx_total_energy
select_df = select_df.drop(columns='foldx_total_energy')
select_df = select_df.sort_values('SASA')
print(select_df.to_markdown())
print('\n')

ref_x = 2 * (
    rfp_known_structures[rfp_known_structures.Name == 'DsRed'].SASA.item() - sa_min
) / sa_range - 1
ref_y = 2 * (
    -rfp_known_structures[rfp_known_structures.Name == 'DsRed'].foldx_total_energy.item() - stab_min
) / stab_range - 1

# fit OLS model
lin_reg = LinearRegression().fit(x_data.view(-1, 1), y_data)
r_squared = lin_reg.score(x_data.view(-1, 1), y_data)
reg_x = np.linspace(-1, 1, 100)
reg_y = lin_reg.coef_ * reg_x + lin_reg.intercept_

# detailed OLS results
X2 = sm.add_constant(x_data.view(-1, 1).numpy())
est = sm.OLS(y_data.numpy(), X2)
est2 = est.fit()
print(est2.summary())


# In[19]:


fig = plt.figure(figsize=(13, 8))
ax = fig.add_subplot(1, 1, 1)

# trend not shown bc not statistically significant
# ax.plot(reg_x, reg_y, color='black', linestyle='--', linewidth=2, zorder=1)
ax.scatter(x_data[~pareto_mask], y_data[~pareto_mask], 
           s=64, facecolors='none', edgecolors='cornflowerblue', zorder=2, label='dominated')
ax.scatter(x_data[pareto_mask], y_data[pareto_mask], 
           s=64, label='non-dominated', color='darkorange', zorder=3)
ax.scatter(ref_x, ref_y, color='red', s=64, marker='x', label='DSRed',
           linewidth=4, zorder=2)

ax.set_xlabel('SASA')
ax.set_ylabel('Stability')
ax.legend(loc='lower left', ncol=3)
ax.set_xlim((-1.1, 1.1))
ax.set_ylim((-2, 1.1))

ax.set_title('RFP sequences w/ known structure')
plt.tight_layout()
# plt.savefig('./figures/rfp_known_structures_fitness_landscape_plot.pdf')


# ## Save results
# 
# Running this cell will overwrite the dataframe included in the repo.

# In[ ]:


# rfp_known_structures.to_csv('../lambo/assets/fpbase/rfp_known_structures.csv', index=None)


# In[ ]:




