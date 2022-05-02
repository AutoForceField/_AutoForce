# +
from abc import ABC, abstractmethod

from torch import Tensor


class Map_fn(ABC):
    @abstractmethod
    def function(self, rij: Tensor) -> Tensor:
        ...
