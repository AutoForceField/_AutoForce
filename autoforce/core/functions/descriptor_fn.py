# +
from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor


class Descriptor_fn(ABC):
    @abstractmethod
    def function(
        self, rij: Tensor, wj: Tensor, numbers: Tensor, unique: set[int]
    ) -> dict[tuple[int, ...], Tensor]:
        ...
