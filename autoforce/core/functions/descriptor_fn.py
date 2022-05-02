# +
from abc import ABC, abstractmethod
from typing import Dict, Set, Tuple, Union

from torch import Tensor


class Descriptor_fn(ABC):
    @abstractmethod
    def function(
        self, rij: Tensor, wj: Tensor, numbers: Tensor, unique: Set[int]
    ) -> Dict[Union[int, Tuple[int, ...]], Tensor]:
        ...
