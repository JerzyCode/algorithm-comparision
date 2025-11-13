import json
from pathlib import Path
from typing import List

import numpy as np
import polars as pl
from matplotlib import pyplot as plt
from sqlalchemy import create_engine


def plot_comparision_ax(results_path: str, ax, title: str = None):
    with open(results_path, "r") as f:
        results = json.load(f)

    optimizers = results["optimizers"]
    problem_name = results["problem"]
    dim = results["problem_dim"]
    lower_bound = results["problem_lower_bound"]
    upper_bound = results["problem_upper_bound"]

    if title is None:
        title = f"{problem_name}  (dim={dim}, lb={lower_bound}, ub={upper_bound})"

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Value")

    for optim_name, optim_data in optimizers.items():
        values = np.array(optim_data["values"])
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        ax.plot(mean, label=optim_name)
        ax.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.2)

    ax.grid()
    ax.legend(fontsize=8)


def plot_multiple_json(json_files: List[str], nrows=2, ncols=3, figsize=(15, 8)):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.flatten()

    for i, json_path in enumerate(json_files):
        if i >= nrows * ncols:
            break
        plot_comparision_ax(json_path, axes[i])

    for j in range(i + 1, nrows * ncols):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def plot_comparision(results_path: str, figsize=(8, 5), save_file: str = None):
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

        plt.plot(mean, label=optim_name)
        plt.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.2)

    plt.legend(loc="upper right")
    plt.grid()

    if save_file is not None:
        save_path = Path(save_file)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def load_top_k_algorithms(
    results_path: str, experiment_id: int, k: int = 5
) -> pl.DataFrame:
    engine = create_engine(f"sqlite:///{results_path}")
    query = """
        SELECT CAST(fitness AS TEXT) AS fitness,
               evaluation.algorithm_id
        FROM evaluation AS evaluation
    """
    evaluations_df = (
        pl.read_database(query, connection=engine)
        .filter(pl.col("fitness") != "ERROR")
        .with_columns(pl.col("fitness").cast(pl.Float64))
    )

    algorithms_df = pl.read_database("SELECT * FROM algorithm", connection=engine)

    merged_df = (
        evaluations_df.join(
            algorithms_df, left_on="algorithm_id", right_on="id", how="inner"
        )
        .filter(pl.col("experiment_id") == experiment_id)
        .unique(subset=["execution_hash"])
    )

    top_k = merged_df.sort("fitness", descending=True).head(k)
    return top_k
