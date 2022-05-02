# +
from abc import ABC, abstractmethod

import torch
from torch import Tensor

import autoforce.cfg as cfg


class Cutoff_fn(ABC):
    """
    Smooth cutoff functions.

    """

    def function(self, dij: Tensor, cutoff: Tensor) -> Tensor:
        """
        dij        distances
        cutoff     can be either a scalar or a tensor
                   with the same length as dij
        """

        beyond = dij > cutoff
        result = torch.where(beyond, cfg.zero, self.smooth(dij / cutoff))
        return result

    @abstractmethod
    def smooth(self, dij: Tensor) -> Tensor:
        ...
