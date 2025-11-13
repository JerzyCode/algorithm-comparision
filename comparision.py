import json
import time
import uuid
from pathlib import Path
from typing import List

import torch
from tqdm import tqdm

from problem import Problem


class Result:
    def __init__(self, repetitions: int):
        self.values = [[] for _ in range(repetitions)]

    def insert(self, repetition: int, value: float) -> None:
        self.values[repetition].append(value)

    def __hash__(self) -> int:
        return hash(tuple(tuple(row) for row in self.values))

    def __eq__(self, other) -> bool:
        if not isinstance(other, Result):
            return False
        return self.values == other.values


class OptimizerSchema:
    def __init__(self, optimizer_class: type, params: dict):
        self.optimizer_class = optimizer_class
        self.params = dict(params)

    def __hash__(self):
        return hash((self.optimizer_class, tuple(sorted(self.params.items()))))

    def __eq__(self, other):
        if not isinstance(other, OptimizerSchema):
            return False
        return (
            self.optimizer_class == other.optimizer_class
            and self.params == other.params
        )


class Comparision:
    def __init__(
        self,
        problem: Problem,
        optimizers: List[OptimizerSchema],
        epochs: int = 100,
        repetitions: int = 10,
        seed: int = 42,
        results_dir: str = "experiment_result",
        log_freq: int = None,
    ):
        self.comparision_id = uuid.uuid4()
        self.problem = problem
        self.optimizers = optimizers
        self.epochs = epochs
        self.repetitions = repetitions
        self.seed = seed
        self.date = time.strftime("%Y-%m-%d")
        self.log_freq = round(0.1 * epochs) if not log_freq else log_freq
        self.results_path = Path(results_dir).joinpath(
            f"{self.problem.name}_D{self.problem.dim}_{self.date}.json"
        )

        self.results = {optim: Result(self.repetitions) for optim in self.optimizers}

    def run(self):
        for optim in self.optimizers:
            torch.manual_seed(self.seed)
            print(f"[{self.comparision_id}] Running {optim.optimizer_class.__name__}")

            for repetition in range(self.repetitions):
                start_time = time.time()
                self._run_once(repetition, optim)
                end_time = time.time()
                print(f"Done repetition {repetition} in {end_time - start_time:.2f}s")

        self._save_results()

    def _save_results(self):
        if self.problem.shift is not None and self.problem.shift.numel() > 0:
            shift_to_save = self.problem.shift.tolist()
        else:
            shift_to_save = None

        data_to_save = {
            "problem": self.problem.name,
            "problem_dim": self.problem.dim,
            "problem_lower_bound": self.problem.lower_bound,
            "problem_upper_bound": self.problem.upper_bound,
            "problem_shift": shift_to_save,
            "seed": self.seed,
            "date": self.date,
            "comparision_id": str(self.comparision_id),
            "epochs": self.epochs,
            "repetitions": self.repetitions,
            "epochs_per_repetition": self.epochs,
            "optimizers": {},
        }

        for optim in self.optimizers:
            param_suffix = "_".join(f"{k}={v}" for k, v in optim.params.items())
            unique_name = f"{optim.optimizer_class.__name__}({param_suffix})"
            data_to_save["optimizers"][unique_name] = {
                "params": optim.params,
                "values": self.results[optim].values,
            }

        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.results_path, "w") as f:
            json.dump(data_to_save, f, indent=4)

        print(f"Results saved to {self.results_path}")

    def _run_once(self, repetition: int, schema: OptimizerSchema):
        low = self.problem.lower_bound
        high = self.problem.upper_bound
        x = torch.empty(self.problem.dim).uniform_(low, high)
        x.requires_grad_(True)
        opt = schema.optimizer_class([x], **schema.params)

        for i in tqdm(
            range(self.epochs),
            desc=f"Epochs for {schema.optimizer_class.__name__} rep {repetition}",
        ):
            opt.zero_grad()
            loss = self.problem(x)
            loss.backward()
            opt.step()

            loss_val = loss.item()
            self.results[schema].insert(repetition, loss_val)

            if i % self.log_freq == 0:
                print(
                    f"[{schema.optimizer_class.__name__}] epoch {i}: {loss_val:.6f}, repetition {repetition}"
                )
