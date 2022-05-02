# +
from torch import Tensor

import autoforce.core as core


class DotProductKernel(core.Kernel_fn):
    def function(self, uv: Tensor, u: Tensor, v: Tensor) -> Tensor:
        return uv / (u * v)
