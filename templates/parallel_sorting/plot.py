import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import json
import os
import os.path as osp

# LOAD FINAL RESULTS:
algorithms = ["merge", "quick"]
folders = os.listdir("./")
final_results = {}

for folder in folders:
    if folder.startswith("run") and osp.isdir(folder):
        with open(osp.join(folder, "results.json"), "r") as f:
            final_results[folder] = json.load(f)

# CREATE LEGEND -- PLEASE FILL IN YOUR RUN NAMES HERE
# Keep the names short, as these will be in the legend.
labels = {
    "run_0": "Baseline",
}

# Use the run key as the default label if not specified
runs = list(final_results.keys())
for run in runs:
    if run not in labels:
        labels[run] = run

# CREATE PLOTS

# Create a programmatic color palette
def generate_color_palette(n):
    cmap = plt.get_cmap('tab20')  # You can change 'tab20' to other colormaps like 'Set1', 'Set2', 'Set3', etc.
    return [mcolors.rgb2hex(cmap(i)) for i in np.linspace(0, 1, n)]

# Get the list of runs and generate the color palette
runs = list(final_results.keys())
colors = generate_color_palette(len(runs))

# Plot 1: Line plot of execution time vs number of cores for each algorithm and data size
for run in runs:
    results = final_results[run]
    data_sizes = list(results[algorithms[0]].keys())
    
    fig, axs = plt.subplots(len(data_sizes), len(algorithms), figsize=(14, 4*len(data_sizes)), squeeze=False)
    fig.suptitle(f"Execution Time vs Number of Cores - {labels[run]}")
    
    for i, size in enumerate(data_sizes):
        for j, algo in enumerate(algorithms):
            cores = list(results[algo][size].keys())
            times = [results[algo][size][core]["mean_time"] for core in cores]
            std_times = [results[algo][size][core]["std_time"] for core in cores]
            
            axs[i, j].errorbar(cores, times, yerr=std_times, fmt='-o', capsize=5, color=colors[runs.index(run)])
            axs[i, j].set_title(f"{algo.capitalize()} Sort - Data Size: {size}")
            axs[i, j].set_xlabel("Number of Cores")
            axs[i, j].set_ylabel("Execution Time (s)")
            axs[i, j].set_xscale('log', base=2)
            axs[i, j].set_yscale('log')
            axs[i, j].grid(True)

    plt.tight_layout()
    plt.savefig(f"execution_time_vs_cores_{run}.png")
    plt.close()

# Plot 2: Bar plot comparing algorithms for each data size and number of cores
for run in runs:
    results = final_results[run]
    data_sizes = list(results[algorithms[0]].keys())
    cores = list(results[algorithms[0]][data_sizes[0]].keys())
    
    fig, axs = plt.subplots(len(data_sizes), 1, figsize=(14, 6*len(data_sizes)), squeeze=False)
    fig.suptitle(f"Algorithm Comparison - {labels[run]}")
    
    x = np.arange(len(cores))
    width = 0.35
    
    for i, size in enumerate(data_sizes):
        merge_times = [results["merge"][size][core]["mean_time"] for core in cores]
        quick_times = [results["quick"][size][core]["mean_time"] for core in cores]
        
        axs[i, 0].bar(x - width/2, merge_times, width, label='Merge Sort', color=colors[0])
        axs[i, 0].bar(x + width/2, quick_times, width, label='Quick Sort', color=colors[1])
        
        axs[i, 0].set_title(f"Data Size: {size}")
        axs[i, 0].set_xlabel("Number of Cores")
        axs[i, 0].set_ylabel("Execution Time (s)")
        axs[i, 0].set_xticks(x)
        axs[i, 0].set_xticklabels(cores)
        axs[i, 0].legend()
        axs[i, 0].grid(True)

    plt.tight_layout()
    plt.savefig(f"algorithm_comparison_{run}.png")
    plt.close()

print("Plots have been generated and saved.")
