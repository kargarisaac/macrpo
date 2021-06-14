
for seed in 2, 3, 4
do
    srun -t 8:00:00 -c 10 --mem=40G python ppo_lstm_sep_comb_trajs.py --seed $seed
done