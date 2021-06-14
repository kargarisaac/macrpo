#!/bin/bash
#SBATCH --job-name=mw
#SBATCH --array=0-4
#SBATCH --time=8:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=10
# Note that all jobs will write to the same file.  This makes less
# files, but will be hard to tell the outputs apart.

case $SLURM_ARRAY_TASK_ID in

    0)  SEED=1 ;;
    1)  SEED=2 ;;
    2)  SEED=3 ;;
    3)  SEED=4 ;;
    4)  SEED=5 ;;
esac

# Job step
srun python ppo_lstm_sep_comb_trajs_c3b.py --seed $SEED