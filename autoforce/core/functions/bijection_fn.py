# +
from __future__ import annotations

import abc

from torch import Tensor


class Bijection_fn(abc.ABC):
    """
    A "Bijection" is a "Function" which, in
    addition to the "function" method, has
    the "inverse" method such that

        x = inverse(function(x))

    """

    @abc.abstractmethod
    def function(self, x: Tensor) -> Tensor:
        ...

    @abc.abstractmethod
    def inverse(self, x: Tensor) -> Tensor:
        ...
