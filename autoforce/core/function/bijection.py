# +
from abc import abstractmethod

from torch import Tensor

from .function import Function


class Bijection(Function):
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
