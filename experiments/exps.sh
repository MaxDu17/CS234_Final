python policy_gradient.py --seed 20  --level 1 --name "full_algorithm" --eval_trials 20 --eval_frq 5 --batch_size 10 --epochs 100  --exp_dir "experiments/initial_tests" --shaped_reward --counterfactual --object_prior
python policy_gradient.py --seed 2  --name "ablate_prior" --eval_trials 20 --eval_frq 5 --batch_size 10 --epochs 100  --exp_dir "experiments/initial_tests" --shaped_reward --counterfactual
python policy_gradient.py --seed 2  --name "ablate_reward" --eval_trials 20 --eval_frq 5 --batch_size 10 --epochs 100  --exp_dir "experiments/initial_tests" --counterfactual --object_prior
python policy_gradient.py --seed 2  --name "ablate_counterfactual" --eval_trials 20 --eval_frq 5 --batch_size 10 --epochs 100  --exp_dir "experiments/initial_tests" --object_prior --shaped_reward
