# +
from abc import ABC, abstractmethod
from typing import Dict, Set, Tuple

from torch import Tensor


class Descriptor_fn(ABC):
    @abstractmethod
    def function(
        self, rij: Tensor, wj: Tensor, numbers: Tensor, unique: Set[int]
    ) -> Dict[Tuple[int, ...], Tensor]:
        ...
