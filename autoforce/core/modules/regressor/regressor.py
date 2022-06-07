# +
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

from torch import Tensor

from autoforce.core.dataclasses import Conf, Target
from autoforce.core.parameters import ParameterMapping


class Regressor(ABC):
    """
    TODO:

    """

    @property
    @abstractmethod
    def cutoff(self) -> ParameterMapping | None:
        """
        TODO:

        """
        ...

    @abstractmethod
    def get_design_matrix(self, confs: Sequence[Conf]) -> tuple[Tensor, Tensor, Any]:
        """
        TODO:

        """
        ...

    @abstractmethod
    def set_weights(self, weights: Tensor, sections: Any) -> None:
        """
        TODO:

        """
        ...

    @abstractmethod
    def get_target(self, conf: Conf) -> Target:
        """
        TODO:

        """
        ...
