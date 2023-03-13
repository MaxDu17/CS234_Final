import os

# levels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 17]
levels = [10, 11, 14, 15, 16, 17]
seeds = [1, 2, 3]

# shafts 12 and 13 require a larger sigma of 1.5. Let's run these manually
for level in levels:
    for seed in seeds:
        os.system(f'python policy_gradient.py --seed {seed}  '
                  f'--level {level} --name "full_algorithm_noanneal" --eval_trials 20 '
                  '--eval_frq 20 --batch_size 1 --epochs 800  '
                  '--exp_dir "experiments/800_step" --shaped_reward --counterfactual --object_prior'
        )
