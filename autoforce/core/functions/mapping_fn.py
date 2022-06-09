# +
from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor


class Mapping_fn(ABC):
    @abstractmethod
    def function(self, rij: Tensor) -> Tensor:
        ...
