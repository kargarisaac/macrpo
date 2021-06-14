#!/bin/bash
#SBATCH --job-name=mw
#SBATCH --array=0-4
#SBATCH --time=8:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=10
# Note that all jobs will write to the same file.  This makes less
# files, but will be hard to tell the outputs apart.

case $SLURM_ARRAY_TASK_ID in

    0)  SEED=6 ;;
    1)  SEED=7 ;;
    2)  SEED=8 ;;
    3)  SEED=9 ;;
    4)  SEED=10 ;;
esac

# Job step
srun python ppo_lstm_sep_trajs.py --seed $SEED