# +
from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor


class SoftZero_fn(ABC):
    """
    A "SoftZero" is a function which satisfies:
    1. f(0) = 0
    2. f(1) = 1
    3. 0 < x < 1 --> 0 < f(x) < 1
    4. f'(0) = 0

    The main property is that both its value f(x) and
    its derivative f'(x) should be zero at x = 0.

    """

    @abstractmethod
    def function(self, x: Tensor) -> Tensor:
        ...
