# for seed in {3..4}
# do
#     python ppo_ff_sep_trajs.py --seed $seed
# done

# for seed in {3..4}
# do
#     python ppo_ff_comb_trajs.py --seed $seed
# done

# for seed in {3..4}
# do
#     python ppo_lstm_sep_trajs.py --seed $seed
# done

# for seed in {1..2}
# do
#     python ppo_lstm_comb_trajs.py --seed $seed
# done

for seed in {2..5}
do
    python ppo_ff_sep_trajs.py --seed $seed
done



# nohup srun -t 5:00:00 -c 20 --mem=30G python ppo_ff_comb_trajs.py --seed 3 >/dev/null 2>&1 &