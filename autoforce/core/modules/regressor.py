# +
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

from torch import Tensor

from autoforce.core.dataclasses import Properties, Structure
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
    def get_design_matrix(
        self, structures: Sequence[Structure]
    ) -> tuple[Tensor, Tensor, Any]:
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
    def get_properties(self, struc: Structure) -> Properties:
        """
        TODO:

        """
        ...
