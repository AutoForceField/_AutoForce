# +
from __future__ import annotations

from torch import Tensor

from autoforce.core.functions import SoftZero_fn


class Polynomial_sz(SoftZero_fn):
    def __init__(self, n: int = 2):
        assert n > 1
        self.n = n

    def function(self, x: Tensor) -> Tensor:
        return x**self.n
