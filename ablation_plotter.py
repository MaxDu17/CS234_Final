import matplotlib.pyplot as plt
import os
import numpy as np
import scipy.stats as stats


root_directory = "experiments/ablations"
seeds = [1, 2, 3, 4, 5, 6]

fig, ax= plt.subplots()
level = 1

runs = [f"full_algorithm_level{level}",
        f"no_prior_level{level}",
        f"no_counterfactual_level{level}",
        f"no_baseline_level{level}",
        f"no_burn_in_level{level}",
        f"fixed_epsilon_level{level}"]

line_list = list()
name_list = list()

x_axis = np.arange(0, 401, 20)

color_list = ["#ff4c38", "#3899a8", "#daa049", "#99ae32", "#368745", "#b38ec7", "#8b3c6d"]

for i, run in enumerate(runs):
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

    ax.fill_between(x_axis, means - errs, means + errs, alpha=0.15, color = color_list[i])
    label = run.split("level")[0]
    label = label.replace("_", " ")
    ax.plot(x_axis, means, label=label, linewidth = 3, color = color_list[i])
    # ax.tick_params(axis='x', labelrotation = 45)
    # ax.label_outer() #critical for labeling outer grid only
    # ax.set_xlim(, 800)
    # ax.set_ylim(-0.05, 1.05)
    # ax.set_title(f"{worlds[level]}")


fig.suptitle("Ablations")
# fig.supylabel('Success Rate')
# fig.supxlabel('Trials')
fig.tight_layout()
plt.legend(loc = "lower center", bbox_to_anchor=(0.8, 0.12), ncol = 1, framealpha=.9, facecolor='white')
plt.show()


fig.savefig(f"ABLATIONS.png")
