# +
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

from torch import Tensor

from autoforce.core.dataclasses import Structure, Target
from autoforce.core.parameters import ParameterMapping


class Regressor_md(ABC):
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
    def get_target(self, struc: Structure) -> Target:
        """
        TODO:

        """
        ...
