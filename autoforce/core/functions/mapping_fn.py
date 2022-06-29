# +
from __future__ import annotations

import abc

from torch import Tensor


class Mapping_fn(abc.ABC):
    @abc.abstractmethod
    def function(self, rij: Tensor) -> Tensor:
        ...
