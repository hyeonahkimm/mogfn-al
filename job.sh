#!/bin/bash
#SBATCH --partition=long                      # Ask for unkillable job
#SBATCH --cpus-per-task=6                    # Ask for 2 CPUs
#SBATCH --gres=gpu:1                         # Ask for 1 GPU
#SBATCH --mem=12G                             # Ask for 10 GB of RAM
#SBATCH --time=96:00:00                        # The job will run for 3 hours
#SBATCH -o /network/scratch/m/moksh.jain/logs/gfnlambo-%j.out  # Write the log on tmp1

module load python/3.9 cuda/11.1
export PYTHONUNBUFFERED=1

cd $SLURM_TMPDIR/
virtualenv env
source env/bin/activate

cd ~/lambo
pip install -r requirements.txt
pip install -e .

python scripts/black_box_opt.py "$@"