# +
from __future__ import annotations

from torch import Tensor

from autoforce.core.functions import Kernel_fn


class DotProductKernel(Kernel_fn):
    def function(self, uv: Tensor, u: Tensor, v: Tensor) -> Tensor:
        return uv / (u * v)
