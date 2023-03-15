import matplotlib.pyplot as plt
import os
import numpy as np
import scipy.stats as stats


root_directory = "experiments/800_step"
seeds = [1, 2, 3]
# runs = ["ablate_reward", "ablate_counterfactual", "ablate_prior", "full_algorithm"]
levels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
worlds = [x.split(".")[0].replace("_", " ") for x in os.listdir("environment/Trials/Original")]

hard_levels = [5, 8, 9, 11, 13, 15]

rows = 3
cols = 6
fig, axes = plt.subplots(rows,cols, figsize=(11, 6))

for i, level in enumerate(levels):
    # print(level)

    runs = [f"full_algorithm_noanneal_level{level}"]

    print(i)
    ax = axes[i // cols][i % cols]

    line_list = list()
    name_list = list()

    x_axis = np.arange(0, 801, 20)

    for run in runs:
        run_data = list()
        for seed in seeds:
            if i in hard_levels:
                data = np.load("experiments/1600_step" + "/" + run + "_" + str(seed) + ".npy")
            else:
                data = np.load(root_directory + "/" + run + "_" + str(seed) + ".npy")
            run_data.append(data)
        run_arr = np.array(run_data)
        means = np.mean(run_arr, axis = 0)
        try:
            errs = stats.sem(run_arr, axis=0)
        except:
            errs = np.zeros_like(means)

        ax.fill_between(x_axis, means - errs, means + errs, alpha=0.25, color = "green")
        ax.plot(x_axis, means, label=run, color = "green")
        ax.tick_params(axis='x', labelrotation = 45)
        ax.label_outer() #critical for labeling outer grid only
        ax.set_xlim(-20, 800)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"{worlds[level]}")

for i in range(len(levels), rows * cols):
    ax = axes[i // cols][i % cols]
    ax.label_outer()  # critical for labeling outer grid only
    ax.set_xlim(-20, 800)
    ax.set_ylim(-0.05, 1.05)

fig.suptitle("Performance on All Environments")
fig.supylabel('Success Rate')
fig.tight_layout()
plt.show()
fig.savefig(f"ALLRESULTS.png")
