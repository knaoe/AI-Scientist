import argparse
import json
import time
import os.path as osp
import numpy as np
from tqdm.auto import tqdm
import pickle
import pathlib
import multiprocessing as mp

def parallel_merge_sort(arr, num_processes):
    chunk_size = len(arr) // num_processes
    chunks = [arr[i:i+chunk_size] for i in range(0, len(arr), chunk_size)]
    
    with mp.Pool(processes=num_processes) as pool:
        sorted_chunks = pool.map(sorted, chunks)
    
    return merge_sorted_arrays(sorted_chunks)

def merge_sorted_arrays(arrays):
    while len(arrays) > 1:
        merged = []
        for i in range(0, len(arrays), 2):
            if i + 1 < len(arrays):
                merged.append(merge(arrays[i], arrays[i+1]))
            else:
                merged.append(arrays[i])
        arrays = merged
    return arrays[0]

def merge(left, right):
    result = []
    i, j = 0, 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

def parallel_quicksort(arr, num_processes):
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    with mp.Pool(processes=num_processes) as pool:
        left_sorted, right_sorted = pool.map(sorted, [left, right])
    
    return left_sorted + middle + right_sorted

def run_experiment(algorithm, data_size, num_cores):
    data = np.random.randint(0, 1000000, size=data_size).tolist()
    start_time = time.time()
    if algorithm == "merge":
        sorted_data = parallel_merge_sort(data, num_cores)
    elif algorithm == "quick":
        sorted_data = parallel_quicksort(data, num_cores)
    end_time = time.time()
    return end_time - start_time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_sizes", nargs="+", type=int, default=[10000, 100000, 1000000])
    parser.add_argument("--max_cores", type=int, default=mp.cpu_count())
    parser.add_argument("--num_runs", type=int, default=5)
    parser.add_argument("--out_dir", type=str, default="run_0")
    config = parser.parse_args()

    pathlib.Path(config.out_dir).mkdir(parents=True, exist_ok=True)

    algorithms = ["merge", "quick"]
    results = {}

    for algorithm in algorithms:
        results[algorithm] = {}
        for data_size in config.data_sizes:
            results[algorithm][data_size] = {}
            for num_cores in range(1, config.max_cores + 1):
                times = []
                for _ in range(config.num_runs):
                    execution_time = run_experiment(algorithm, data_size, num_cores)
                    times.append(execution_time)
                results[algorithm][data_size][num_cores] = {
                    "mean_time": np.mean(times),
                    "std_time": np.std(times)
                }

    with open(osp.join(config.out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("Experiment completed. Results saved in", config.out_dir)
