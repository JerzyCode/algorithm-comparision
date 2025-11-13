import json

import numpy as np
from matplotlib import pyplot as plt


def plot_comparision(results_path: str, figsize=(8, 5)):
    with open(results_path, "r") as f:
        results = json.load(f)

    optimizers = results["optimizers"]
    problem_name = results["problem"]
    dim = results["problem_dim"]
    lower_bound = results["problem_lower_bound"]
    upper_bound = results["problem_upper_bound"]
    title = f"{problem_name}  (dim={dim}, lb={lower_bound}, ub={upper_bound})\n Mean and Std values for each optimizer"

    plt.figure(figsize=figsize)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Mean Value")

    for optim_name, optim_data in optimizers.items():
        values = np.array(optim_data["values"])
        std = values.std(axis=0)
        mean = values.mean(axis=0)

        label = f"{optim_name} {optim_data['params']}"
        plt.plot(mean, label=label)
        plt.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.2)

    plt.legend(loc="upper right")
    plt.grid()
    plt.show()
