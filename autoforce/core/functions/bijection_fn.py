# +
from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor


class Bijection_fn(ABC):
    """
    A "Bijection" is a "Function" which, in
    addition to the "function" method, has
    the "inverse" method such that

        x = inverse(function(x))

    """

    @abstractmethod
    def function(self, x: Tensor) -> Tensor:
        ...

    @abstractmethod
    def inverse(self, x: Tensor) -> Tensor:
        ...
