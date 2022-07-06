# sbatch job.sh optimizer=mogfn_direct optimizer.encoder_obj=mlm task=regex_easy tokenizer=protein surrogate=multi_task_exact_gp acquisition=nehvi optimizer.train_steps=10000 task.min_len=12 task.max_len=16 wandb_mode=online optimizer.beta_cond=False optimizer.sample_beta=48 optimizer.beta_max=64 exp_name=mogfn_v2_regex_easy_12_16_no_beta_fixed_48_64

# sbatch job.sh optimizer=mogfn_direct optimizer.encoder_obj=mlm task=regex_easy tokenizer=protein surrogate=multi_task_exact_gp acquisition=nehvi optimizer.train_steps=10000 task.min_len=12 task.max_len=16 wandb_mode=online optimizer.beta_cond=False optimizer.sample_beta=48 optimizer.beta_max=48 exp_name=mogfn_v2_regex_easy_12_16_no_beta_fixed_48_48

# sbatch job.sh optimizer=mogfn_direct optimizer.encoder_obj=mlm task=regex_easy tokenizer=protein surrogate=multi_task_exact_gp acquisition=nehvi optimizer.train_steps=10000 task.min_len=12 task.max_len=16 wandb_mode=online optimizer.beta_cond=True optimizer.sample_beta=48 optimizer.beta_max=48 exp_name=mogfn_v2_regex_easy_12_16_beta_unif_48_48

# sbatch job.sh optimizer=mogfn_direct optimizer.encoder_obj=mlm task=regex_easy tokenizer=protein surrogate=multi_task_exact_gp acquisition=nehvi optimizer.train_steps=10000 task.min_len=12 task.max_len=16 wandb_mode=online optimizer.beta_cond=True optimizer.sample_beta=48 optimizer.beta_max=64 exp_name=mogfn_v2_regex_easy_12_16_beta_unif_48_64



# sbatch job.sh optimizer=mogfn_direct optimizer.encoder_obj=mlm task=regex_easy tokenizer=protein surrogate=multi_task_exact_gp acquisition=nehvi optimizer.train_steps=10000 task.min_len=12 task.max_len=16 wandb_mode=online optimizer.beta_cond=False optimizer.sample_beta=16 optimizer.beta_max=32 exp_name=mogfn_v2_regex_easy_12_16_no_beta_fixed_16_32

# sbatch job.sh optimizer=mogfn_direct optimizer.encoder_obj=mlm task=regex_easy tokenizer=protein surrogate=multi_task_exact_gp acquisition=nehvi optimizer.train_steps=10000 task.min_len=12 task.max_len=16 wandb_mode=online optimizer.beta_cond=False optimizer.sample_beta=16 optimizer.beta_max=16 exp_name=mogfn_v2_regex_easy_12_16_no_beta_fixed_16_16

# sbatch job.sh optimizer=mogfn_direct optimizer.encoder_obj=mlm task=regex_easy tokenizer=protein surrogate=multi_task_exact_gp acquisition=nehvi optimizer.train_steps=10000 task.min_len=12 task.max_len=16 wandb_mode=online optimizer.beta_cond=True optimizer.sample_beta=16 optimizer.beta_max=16 exp_name=mogfn_v2_regex_easy_12_16_beta_unif_16_16

# sbatch job.sh optimizer=mogfn_direct optimizer.encoder_obj=mlm task=regex_easy tokenizer=protein surrogate=multi_task_exact_gp acquisition=nehvi optimizer.train_steps=10000 task.min_len=12 task.max_len=16 wandb_mode=online optimizer.beta_cond=True optimizer.sample_beta=16 optimizer.beta_max=32 exp_name=mogfn_v2_regex_easy_12_16_beta_unif_16_32


# sbatch job.sh optimizer=mogfn_direct optimizer.encoder_obj=mlm task=regex_easy tokenizer=protein surrogate=multi_task_exact_gp acquisition=nehvi optimizer.train_steps=10000 task.min_len=12 task.max_len=16 wandb_mode=online optimizer.beta_cond=False optimizer.sample_beta=48 optimizer.beta_max=64 exp_name=mogfn_v2_regex_easy_12_16_no_beta_fixed_48_64_20bins optimizer.simplex_bins=20

# sbatch job.sh optimizer=mogfn_direct optimizer.encoder_obj=mlm task=regex_easy tokenizer=protein surrogate=multi_task_exact_gp acquisition=nehvi optimizer.train_steps=10000 task.min_len=12 task.max_len=16 wandb_mode=online optimizer.beta_cond=False optimizer.sample_beta=48 optimizer.beta_max=48 exp_name=mogfn_v2_regex_easy_12_16_no_beta_fixed_48_48_20bins optimizer.simplex_bins=20

# sbatch job.sh optimizer=mogfn_direct optimizer.encoder_obj=mlm task=regex_easy tokenizer=protein surrogate=multi_task_exact_gp acquisition=nehvi optimizer.train_steps=10000 task.min_len=12 task.max_len=16 wandb_mode=online optimizer.beta_cond=True optimizer.sample_beta=48 optimizer.beta_max=48 exp_name=mogfn_v2_regex_easy_12_16_beta_unif_48_48_20bins optimizer.simplex_bins=20

# sbatch job.sh optimizer=mogfn_direct optimizer.encoder_obj=mlm task=regex_easy tokenizer=protein surrogate=multi_task_exact_gp acquisition=nehvi optimizer.train_steps=10000 task.min_len=12 task.max_len=16 wandb_mode=online optimizer.beta_cond=True optimizer.sample_beta=48 optimizer.beta_max=64 exp_name=mogfn_v2_regex_easy_12_16_beta_unif_48_64_20bins optimizer.simplex_bins=20



# sbatch job.sh optimizer=mogfn_direct optimizer.encoder_obj=mlm task=regex_easy tokenizer=protein surrogate=multi_task_exact_gp acquisition=nehvi optimizer.train_steps=10000 task.min_len=12 task.max_len=16 wandb_mode=online optimizer.beta_cond=False optimizer.sample_beta=16 optimizer.beta_max=32 exp_name=mogfn_v2_regex_easy_12_16_no_beta_fixed_16_32_20bins optimizer.simplex_bins=20

# sbatch job.sh optimizer=mogfn_direct optimizer.encoder_obj=mlm task=regex_easy tokenizer=protein surrogate=multi_task_exact_gp acquisition=nehvi optimizer.train_steps=10000 task.min_len=12 task.max_len=16 wandb_mode=online optimizer.beta_cond=False optimizer.sample_beta=16 optimizer.beta_max=16 exp_name=mogfn_v2_regex_easy_12_16_no_beta_fixed_16_16_20bins optimizer.simplex_bins=20

# sbatch job.sh optimizer=mogfn_direct optimizer.encoder_obj=mlm task=regex_easy tokenizer=protein surrogate=multi_task_exact_gp acquisition=nehvi optimizer.train_steps=10000 task.min_len=12 task.max_len=16 wandb_mode=online optimizer.beta_cond=True optimizer.sample_beta=16 optimizer.beta_max=16 exp_name=mogfn_v2_regex_easy_12_16_beta_unif_16_16_20bins optimizer.simplex_bins=20

# sbatch job.sh optimizer=mogfn_direct optimizer.encoder_obj=mlm task=regex_easy tokenizer=protein surrogate=multi_task_exact_gp acquisition=nehvi optimizer.train_steps=10000 task.min_len=12 task.max_len=16 wandb_mode=online optimizer.beta_cond=True optimizer.sample_beta=16 optimizer.beta_max=32 exp_name=mogfn_v2_regex_easy_12_16_beta_unif_16_32_20bins optimizer.simplex_bins=20




for task in regex_easy regex_easy_3 regex_easy_4 regex
do
    for simplex_bins in 20
    do
        for beta_cond in True False
        do
            for sample_beta in 16 48
            do
                if test ${sample_beta} -eq 16 
                then
                    for beta_max in 16 32
                    do
                        echo "${task}, ${simplex_bins}, ${beta_cond}, ${sample_beta}, ${beta_max}"
                        sbatch job.sh optimizer=mogfn_direct optimizer.encoder_obj=mlm task=${task} tokenizer=protein surrogate=multi_task_exact_gp acquisition=nehvi optimizer.train_steps=20000 task.min_len=12 task.max_len=16 wandb_mode=online optimizer.beta_cond=${beta_cond} optimizer.sample_beta=${sample_beta} optimizer.beta_max=${beta_max} optimizer.simplex_bins=${simplex_bins} group_name=mogfn_v2 exp_name=${beta_cond}_${simplex_bins}_${sample_beta}_${beta_max} exp_tags="[${task},16-18]"
                        sleep 5
                    done
                fi

                if test ${sample_beta} -eq 48
                then
                    for beta_max in 48 64
                    do
                        echo "${task}, ${simplex_bins}, ${beta_cond}, ${sample_beta}, ${beta_max}"
                        sbatch job.sh optimizer=mogfn_direct optimizer.encoder_obj=mlm task=${task} tokenizer=protein surrogate=multi_task_exact_gp acquisition=nehvi optimizer.train_steps=20000 task.min_len=12 task.max_len=16 wandb_mode=online optimizer.beta_cond=${beta_cond} optimizer.sample_beta=${sample_beta} optimizer.beta_max=${beta_max} optimizer.simplex_bins=${simplex_bins} group_name=mogfn_v2 exp_name=${beta_cond}_${simplex_bins}_${sample_beta}_${beta_max} exp_tags="[${task},16-18]"
                        sleep 5
                    done
                fi
                
            done
        done  
    done
done