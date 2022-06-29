# +
from __future__ import annotations

import abc
from typing import Any, Sequence

from torch import Tensor

from autoforce.core.dataclasses import Properties, Structure
from autoforce.core.parameters import ParameterMapping


class Regressor(abc.ABC):
    """
    TODO:

    """

    @property
    @abc.abstractmethod
    def cutoff(self) -> ParameterMapping | None:
        """
        TODO:

        """
        ...

    @abc.abstractmethod
    def get_design_matrix(
        self, structures: Sequence[Structure]
    ) -> tuple[Tensor, Tensor, Any]:
        """
        TODO:

        """
        ...

    @abc.abstractmethod
    def set_weights(self, weights: Tensor, sections: Any) -> None:
        """
        TODO:

        """
        ...

    @abc.abstractmethod
    def predict(self, struc: Structure) -> Properties:
        """
        TODO:

        """
        ...
