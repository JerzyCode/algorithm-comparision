import time
import uuid
from pathlib import Path
from typing import List
import json

import torch

from problem import Problem


class Result:
    def __init__(self, repetitions: int):
        self.values = [[] for _ in range(repetitions)]

    def insert(self, repetition: int, value: float) -> None:
        self.values[repetition].append(value)


class Comparision:
    def __init__(
        self,
        problem: Problem,
        optimizers: List[torch.optim.Optimizer],
        epochs: int = 100,
        repetitions: int = 10,
        seed: int = 42,
        results_dir: str = "experiment_result",
    ):
        self.comparision_id = uuid.uuid4()
        self.problem = problem
        self.optimizers = optimizers
        self.epochs = epochs
        self.repetitions = repetitions
        self.seed = seed
        self.date = time.strftime("%Y-%m-%d")
        self.log_freq = round(0.1 * epochs)
        self.results_path = Path(results_dir).joinpath(
            f"{self.problem.name}_{self.date}.json"
        )

        self.results = {
            optim.__class__.__name__: Result(self.repetitions)
            for optim in self.optimizers
        }

    def run(self):
        for optim in self.optimizers:
            torch.manual_seed(self.seed)
            print(f"[{self.comparision_id}] Running {optim.__class__.__name__}")

            for repetition in range(self.repetitions):
                start_time = time.time()
                self._run_once(repetition, optim)
                end_time = time.time()
                print(f"Done repetition {repetition} in {end_time - start_time:.2f}s")

        self._save_results()

    def _save_results(self):
        data_to_save = {
            "problem": self.problem.name,
            "problem_dim": self.problem.dim,
            "problem_lower_bound": self.problem.lower_bound,
            "problem_upper_bound": self.problem.upper_bound,
            "problem_shift": self.problem.shift.tolist()
            if self.problem.shift
            else None,
            "seed": self.seed,
            "date": self.date,
            "comparision_id": str(self.comparision_id),
            "epochs": self.epochs,
            "repetitions": self.repetitions,
            "optimizers": {},
        }

        for optim in self.optimizers:
            optim_name = optim.__class__.__name__
            params = {
                k: (v.tolist() if isinstance(v, torch.Tensor) else v)
                for k, v in optim.defaults.items()
            }

            data_to_save["optimizers"][optim_name] = {
                "params": params,
                "values": self.results[optim_name].values,
            }

        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.results_path, "w") as f:
            json.dump(data_to_save, f, indent=4)

        print(f"Results saved to {self.results_path}")

    def _run_once(self, repetition: int, optim_prototype: torch.optim.Optimizer):
        low = self.problem.lower_bound
        high = self.problem.upper_bound
        x = (high - low) * torch.rand(self.problem.dim, requires_grad=True) + low

        opt = optim_prototype.__class__([x], **optim_prototype.defaults)

        for i in range(self.epochs):
            opt.zero_grad()
            loss = self.problem(x)
            loss.backward()
            opt.step()

            loss_val = loss.item()
            self.results[optim_prototype.__class__.__name__].insert(
                repetition, loss_val
            )

            if i % self.log_freq == 0:
                print(
                    f"[{self.comparision_id}] Repetition {repetition} epoch {i}: {loss_val:.6f}"
                )
