#!/bin/bash
#SBATCH --job-name=dd0
#SBATCH --array=0-7
#SBATCH --time=11:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=35
# Note that all jobs will write to the same file.  This makes less
# files, but will be hard to tell the outputs apart.

case $SLURM_ARRAY_TASK_ID in

    0)  SEED=2 ;;
    1)  SEED=2 ;;
    2)  SEED=3 ;;
    3)  SEED=4 ;;
    4)  SEED=0 ;;
    5)  SEED=1 ;;
    6)  SEED=2 ;;
esac

# Job step
srun python ppo_lstm_sep_comb_trajs_c3b.py --seed $SEED