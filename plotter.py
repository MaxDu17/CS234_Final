import matplotlib.pyplot as plt
import os
import numpy as np
import scipy.stats as stats


root_directory = "experiments/initial_tests"
seeds = [1]
runs = ["ablate_reward", "ablate_counterfactual", "ablate_prior", "full_algorithm"]

files = [file for file in os.listdir(root_directory) if ".npy" in file]

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

# for file in files:
#     data = np.load(root_directory + "/" + file)
#     line = ax.plot(data)
#     line_list.append(line)
#     # plot_dict[file.split(".")[0]] = plots

plt.legend()

# plt.legend(line, [file.split(".")[0]])

plt.show()

print(files)
