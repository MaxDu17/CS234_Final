import os

levels = [3, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 17]
seeds = [1, 2, 3]
for level in levels:
    for seed in seeds:
        os.system(f'python policy_gradient.py --seed {seed}  '
                  f'--level {level} --name "full_algorithm_noanneal" --eval_trials 20 '
                  '--eval_frq 20 --batch_size 1 --epochs 200  '
                  '--exp_dir "experiments/pg_seeds" --shaped_reward --counterfactual --object_prior'
        )
