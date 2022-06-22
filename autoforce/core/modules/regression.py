# +
import abc

from torch import Tensor


class Regression(abc.ABC):
    """
    Abstract base class for "Regression".

    A "Regression" is an algorithm for minimizing
        L(X) = ||A @ X - Y|| + R(X)
    where A is the design matrix,
    Y is the vector of targets,
    and X in the solution vector.
    R(X) is a regularization term which is the
    defining feature of the "Regression".

    A "Regression" should have a "fit" method
    which takes the design matrix and targets
    as input and returns the solution.

    """

    @abc.abstractmethod
    def fit(self, design_matrix: Tensor, targets: Tensor) -> Tensor:
        """
        Agruments:
            design_matrix       a 2d tensor of size (m, n)
            targets             a 1d tensor of size m

        Returns:
            solution            a 1d tensor of size n

        """
        ...

    @property
    @abc.abstractmethod
    def is_deterministic(self) -> bool:
        ...
