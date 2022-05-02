# +
from abc import ABC, abstractmethod

from torch import Tensor


class Kernel_fn(ABC):
    @abstractmethod
    def function(self, uv: Tensor, u: Tensor, v: Tensor) -> Tensor:
        """
        uv:   scalar products matrix <u_i,v_j> with shape (m, n)
        u:    norms sqrt(<u_i,u_i>) with shape (m, 1)
        v:    norms sqrt(<v_j,v_j>) with shape (1, n)

        output:
              a tensor with the same shape as uv
        """
        ...
