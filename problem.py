import abc
import math
from abc import abstractmethod
from typing import Optional

import torch


class Problem(abc.ABC):
    def __init__(
        self,
        name: str,
        dim: int,
        shift: Optional[torch.Tensor] = None,
        lower_bound: float = -100.0,
        upper_bound: float = 100.0,
    ):
        self.name = name
        self.dim = dim
        self.shift = shift if shift is not None else torch.zeros(dim)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    @abstractmethod
    def __call__(self, x: torch.Tensor):
        pass


class Sphere(Problem):
    def __init__(self, dim: int, shift: Optional[torch.Tensor] = None):
        super().__init__("Sphere", dim, shift, -5.0, 5.0)

    def __call__(self, x: torch.Tensor):
        z = x - self.shift
        return torch.sum(z * z)


class Rastrigin(Problem):
    def __init__(self, dim: int, shift: Optional[torch.Tensor] = None):
        super().__init__("Rastrigin", dim, shift, -5.12, 5.12)

    def __call__(self, x: torch.Tensor):
        z = x - self.shift
        return 10 * self.dim + torch.sum(z * z - 10 * torch.cos(2 * torch.pi * z))


class Rosenbrock(Problem):
    def __init__(self, dim: int, shift: Optional[torch.Tensor] = None):
        super().__init__("Rosenbrock", dim, shift, -30.0, 30.0)

    def __call__(self, x: torch.Tensor):
        z = x - self.shift
        return torch.sum(100 * (z[1:] - z[:-1] ** 2) ** 2 + (z[:-1] - 1) ** 2)


class Ackley(Problem):
    def __init__(self, dim: int, shift: Optional[torch.Tensor] = None):
        super().__init__("Ackley", dim, shift, -32.0, 32.0)

    def __call__(self, x: torch.Tensor):
        z = x - self.shift
        a = 20
        b = 0.2
        c = 2 * math.pi
        sum_sq = torch.sum(z**2)
        sum_cos = torch.sum(torch.cos(c * z))
        term1 = -a * torch.exp(-b * torch.sqrt(sum_sq / self.dim))
        term2 = -torch.exp(sum_cos / self.dim)
        return term1 + term2 + a + math.e
