#!/bin/bash
#SBATCH --job-name=dd0
#SBATCH --array=0-6
#SBATCH --time=10:00:00
#SBATCH --mem=60G
#SBATCH --cpus-per-task=30
# Note that all jobs will write to the same file.  This makes less
# files, but will be hard to tell the outputs apart.

case $SLURM_ARRAY_TASK_ID in

    0)  SEED=0 ;;
    1)  SEED=1 ;;
    2)  SEED=2 ;;
    3)  SEED=3 ;;
    4)  SEED=4 ;;
    5)  SEED=5 ;;
esac

# Job step
srun python ppo_lstm_sep_trajs.py --seed $SEED