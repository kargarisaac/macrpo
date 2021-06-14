export PYTHONPATH=$PYTHONPATH:/home/isaac/codes/autonomous_driving/multi-agent/
for seed in 2 3 4
do
    python particle_ppo_lstm_sep_comb_trajs_nosharedw.py --seed $seed --lr 6e-4
done
