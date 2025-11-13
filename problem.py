import abc
from abc import abstractmethod

import torch


class Problem(abc.ABC):
    def __init__(
        self,
        name: str,
        dim: int,
        shift: torch.Tensor,
        lower_bound: float,
        upper_bound: float,
    ):
        self.name = name
        self.dim = dim
        self.shift = shift
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    @abstractmethod
    def __call__(self, x: torch.Tensor):
        pass


class Sphere(Problem):
    def __init__(self, dim: int, shift: torch.Tensor):
        super().__init__("Sphere", dim, shift, -100.0, 100.0)

    def __call__(self, x: torch.Tensor):
        z = x - self.shift
        return torch.sum(z * z)


class Rastrigin(Problem):
    def __init__(self, dim: int, shift: torch.Tensor):
        super().__init__("Rastrigin", dim, shift, -5.12, 5.12)

    def __call__(self, x: torch.Tensor):
        z = x - self.shift
        return 10 * self.dim + torch.sum(z * z - 10 * torch.cos(2 * torch.pi * z))


class Rosenbrock(Problem):
    def __init__(self, dim: int, shift: torch.Tensor):
        super().__init__("Rosenbrock", dim, shift, -30.0, 30.0)

    def __call__(self, x: torch.Tensor):
        z = x - self.shift
        return torch.sum(100 * (z[1:] - z[:-1] ** 2) ** 2 + (z[:-1] - 1) ** 2)
