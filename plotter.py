import matplotlib.pyplot as plt
import os
import numpy as np
import scipy.stats as stats


root_directory = "experiments/initial_tests"
seeds = [1, 2, 3]
runs = ["ablate_reward", "ablate_counterfactual", "ablate_prior", "full_algorithm"]

fig, ax = plt.subplots()

#TODO: plot multiple seeds
# plot_dict = {}
line_list = list()
name_list = list()

x_axis = np.arange(0, 100, 5)

for run in runs:
    run_data = list()
    for seed in seeds:
        data = np.load(root_directory + "/" + run + "_" + str(seed) + ".npy")
        run_data.append(data)
    run_arr = np.array(run_data)
    means = np.mean(run_arr, axis = 0)

    try:
        errs = stats.sem(run_arr, axis=0)
    except:
        errs = np.zeros_like(means)

    plt.fill_between(x_axis, means - errs, means + errs, alpha=0.25)
    plt.plot(x_axis, means, label=run)

ax.set_ylabel("Success Rate")
ax.set_xlabel("Epochs")
ax.set_title("Level 0 Performance (3 seeds)")
plt.legend()

fig.savefig("training.png")
plt.show()
