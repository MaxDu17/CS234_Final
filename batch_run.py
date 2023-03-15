import os

# levels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 17]
# levels = [10, 11, 14, 15, 16, 17]
# levels = [12,13] #REQUIRES SPECIAL SIGMAY
# levels = [5, 8, 9] # HARD LEVELS 1
levels = [11, 13, 15] #HARD LEVELS 2
seeds = [4,5,6]

# for level in levels:
#     for seed in seeds:
#         os.system(f'python policy_gradient.py --seed {seed}  '
#                   f'--level {level} --name "full_algorithm_noanneal" --eval_trials 20 '
#                   '--eval_frq 40 --batch_size 1 --epochs 1600  '
#                   '--exp_dir "experiments/1600_step" --shaped_reward --counterfactual --object_prior'
#         )

## ABBLATIONS
for seed in seeds:
    # os.system(f'python policy_gradient.py --seed {seed}  '
    #           f'--level {1} --name "full_algorithm" --eval_trials 20 '
    #           '--eval_frq 20 --batch_size 1 --epochs 400  '
    #           '--exp_dir "experiments/ablations" --shaped_reward --counterfactual --object_prior'
    # )
    # os.system(f'python policy_gradient.py --seed {seed}  '
    #           f'--level {1} --name "no_prior" --eval_trials 20 '
    #           '--eval_frq 20 --batch_size 1 --epochs 400  '
    #           '--exp_dir "experiments/ablations" --shaped_reward --counterfactual'
    #           )
    # os.system(f'python policy_gradient.py --seed {seed}  '
    #           f'--level {1} --name "no_counterfactual" --eval_trials 20 '
    #           '--eval_frq 20 --batch_size 1 --epochs 400  '
    #           '--exp_dir "experiments/ablations" --shaped_reward  --object_prior'
    #           )
    # os.system(f'python policy_gradient.py --seed {seed}  '
    #           f'--level {1} --name "no_baseline" --eval_trials 20 '
    #           '--eval_frq 20 --batch_size 1 --epochs 400  '
    #           '--exp_dir "experiments/ablations" --shaped_reward --counterfactual --object_prior'
    # )
    # os.system(f'python policy_gradient.py --seed {seed}  '
    #           f'--level {1} --name "no_burn_in" --eval_trials 20 '
    #           '--eval_frq 20 --batch_size 1 --epochs 400  '
    #           '--exp_dir "experiments/ablations" --shaped_reward --counterfactual --object_prior'
    #           )
    os.system(f'python policy_gradient.py --seed {seed}  '
              f'--level {1} --name "fixed_epsilon" --eval_trials 20 '
              '--eval_frq 20 --batch_size 1 --epochs 400  '
              '--exp_dir "experiments/ablations" --shaped_reward --counterfactual --object_prior'
              )

