# +
from abc import abstractmethod

import torch
from torch import Tensor

import autoforce.cfg as cfg

from .function import Function


class Cutoff_fn(Function):
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
