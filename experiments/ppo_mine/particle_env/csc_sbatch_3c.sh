#!/bin/bash
#SBATCH --job-name=particle
#SBATCH --output=particle_out_%A_%a.txt
#SBATCH --error=particle_err_%A_%a.txt
#SBATCH --account=project_2003365
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=30
#SBATCH --mem=130G
#SBATCH --partition=small
#SBATCH --array=1-5

# export OMP_PROC_BIND=true

srun python ppo_lstm_sep_comb_nsp_c3c.py --seed $SLURM_ARRAY_TASK_ID