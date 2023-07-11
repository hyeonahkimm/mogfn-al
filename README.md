# MOGFN-AL: Multi-Objective GFlowNets for Active Learning
This code is built on top of the LaMBO repo from [Stanton et al., 2022](https://arxiv.org/abs/2203.12742). We add the multi-objective GFlowNet to generate diverse candidates within the multi-objective Bayesian optimization setting. Further details can be found in our paper [paper](https://arxiv.org/abs/2210.12765). 

## Installation

#### FoldX
[FoldX](https://foldxsuite.crg.eu/academic-license-info) is available under a free academic license. 
After creating an account you will be emailed a link to download the FoldX executable and supporting assets.
Copy the contents of the downloaded archive to `~/foldx`.
You may also need to rename the FoldX executable (e.g. `mv -v ~/foldx/foldx_20221231 ~/foldx/foldx`).


```bash
conda create --name lambo-env python=3.7 -y && conda activate lambo-env
conda install -c conda-forge pdbfixer -y
pip install -r requirements.txt --upgrade
pip install -e .
```


You can run the `regex` task to test your installation.

```bash
python scripts/black_box_opt.py optimizer=mogfn_seq task=regex tokenizer=protein optimizer.encoder_obj=mlm surrogate=multi_task_exact_gp acquisition=nehvi optimizer.use_acqf=True optimizer.pref_cond=False optimizer.num_opt_steps=1500 optimizer.max_len=30 optimizer.sample_beta=32 optimizer.beta_sched=1 optimizer.pref_alpha=1 wandb_mode=disabled
```

For the full LaMBO algorithm on the RFP task, run
```bash
python scripts/black_box_opt.py optimizer=lambo optimizer.encoder_obj=mlm task=proxy_rfp tokenizer=protein surrogate=multi_task_exact_gp acquisition=nehvi
```

For MOGFN-AL on the RFP task run 
```bash
python scripts/black_box_opt.py optimizer=mogfn_seq optimizer.encoder_obj=mlm task=proxy_rfp tokenizer=protein surrogate=multi_task_exact_gp acquisition=nehvi optimizer.use_acqf=True optimizer.pref_cond=False optimizer.num_opt_steps=1500 optimizer.max_len=10 optimizer.sample_beta=32 optimizer.beta_sched=1 wandb_mode=online exp_name="<exp_name_here>" group_name="<group_name_here>" exp_tags="<tags_here>" trial_id=<id_here>
```

(Takes around 96 hours with a RTX8000 for the full run - the time is mostly spent on evaluation of candidates with FoldX)


#### Key GFlowNet Options
- `sample_beta` 
- `random_action_prob`
- `beta_sched`
- `num_opt_step`
- `min_len`
- `max_len`

There are several other hyperparameters can be varied, see `./hydra_config/optimizer/mogfn_seq.yaml`

An example SLURM script (`job.sh`) is included for reference.

## Citation

If you use any part of this code for your own work, please cite the original paper, as well as our paper.

```
@inproceedings{jain2023multi,
  title={Multi-objective gflownets},
  author={Jain, Moksh and Raparthy, Sharath Chandra and Hern{\'a}ndez-Garc{\i}́a, Alex and Rector-Brooks, Jarrid and Bengio, Yoshua and Miret, Santiago and Bengio, Emmanuel},
  booktitle={International Conference on Machine Learning},
  pages={14631--14653},
  year={2023},
  organization={PMLR}
}

@inproceedings{stanton2022accelerating,
  title={Accelerating bayesian optimization for biological sequence design with denoising autoencoders},
  author={Stanton, Samuel and Maddox, Wesley and Gruver, Nate and Maffettone, Phillip and Delaney, Emily and Greenside, Peyton and Wilson, Andrew Gordon},
  booktitle={International Conference on Machine Learning},
  pages={20459--20478},
  year={2022},
  organization={PMLR}
}
```