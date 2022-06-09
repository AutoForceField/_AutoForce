# +
from __future__ import annotations

from torch import Tensor

from autoforce.core.functions import Bijection_fn


class Identity_bi(Bijection_fn):
    def function(self, x: Tensor) -> Tensor:
        return x

    def inverse(self, x: Tensor) -> Tensor:
        return x
