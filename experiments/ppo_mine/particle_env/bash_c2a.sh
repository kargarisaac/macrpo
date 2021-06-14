#!/bin/bash
#SBATCH --job-name=particle
#SBATCH --array=0-5
#SBATCH --time=12:00:00
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
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
srun python ppo_lstm_sep_trajs_nsp_c2a.py --seed $SEED --lr 0.005 --beta 0.0
