#!/bin/bash
#SBATCH --job-name=dd0
#SBATCH --array=0-4
#SBATCH --time=03:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=20
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
srun python parameterSharingDD0.py --seed $SEED --method PPO --episodes-total 300000
