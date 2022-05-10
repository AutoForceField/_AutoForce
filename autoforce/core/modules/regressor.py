# +
from abc import ABC, abstractmethod
from typing import Any, Sequence, Union

from torch import Tensor

from ..dataclasses import Conf, Target
from ..parameters import ParameterMapping


class Regressor(ABC):
    """
    TODO:

    """

    @property
    @abstractmethod
    def cutoff(self) -> Union[ParameterMapping, None]:
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
