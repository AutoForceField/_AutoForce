# +
from __future__ import annotations

import abc

from torch import Tensor


class Descriptor_fn(abc.ABC):
    @abc.abstractmethod
    def function(
        self, rij: Tensor, wj: Tensor, numbers: Tensor, unique: set[int]
    ) -> dict[tuple[int, ...], Tensor]:
        ...
