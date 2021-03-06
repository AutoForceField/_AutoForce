# +
from __future__ import annotations

from torch import Tensor

from .descriptor import Descriptor


class Environment:
    """
    Local chemical environment (LCE) of an atom.

    Keywords:
    central atom    the atom for which the Environment
                    is created
    neighborhood    atoms in the neighborhood of
                    the central atom (|rij| < cutoff)

    """

    __slots__ = (
        "index",
        "number",
        "neighbors",
        "numbers",
        "rij",
        "_cache_d",
    )

    def __init__(
        self,
        index: Tensor,
        number: Tensor,
        neighbors: Tensor,
        numbers: Tensor,
        rij: Tensor,
    ) -> None:
        """
        index    index of the central atom in the
                 Structure it belongs to.
        number   atomic number of the central atom
        numbers  atomic numbers of the neighborhood atoms
        rij      coordinates of the neighborhood atoms
                 relative to the central atom

        """

        self.index = index
        self.number = number
        self.neighbors = neighbors
        self.numbers = numbers
        self.rij = rij

        # cache
        self._cache_d: list[Descriptor | None] = []
