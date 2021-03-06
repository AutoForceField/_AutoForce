# +
from __future__ import annotations

from typing import Optional

from torch import Tensor

from .environment import Environment
from .properties import Properties


class Structure:
    """
    An atomic structure.

    Same role as ase.Atoms, except the data
    are stored as torch.Tensor instead of
    np.ndarray, for using the autograd.


    Keywords:
    numbers      atomic numbers
    positions    self explanatory
    cell         self explanatory
    pbc          periodic boundary conditions
    properties   energy & forces

    """

    __slots__ = (
        "numbers",
        "positions",
        "cell",
        "pbc",
        "properties",
        "number_of_atoms",
        "unique_counts",
        "_cache_e",
        "_cached_isolated_atoms",
    )

    def __init__(
        self,
        numbers: Tensor,
        positions: Tensor,
        cell: Tensor,
        pbc: list[bool],
        properties: Optional[Properties] = None,
    ) -> None:
        """
        Self explanatory.

        """

        if positions.requires_grad != cell.requires_grad:
            raise RuntimeError(
                "requires_grad should be the " "same for positions and cell!"
            )

        self.numbers = numbers
        self.number_of_atoms = int(numbers.numel())
        u, c = numbers.unique(return_counts=True)
        self.unique_counts = {int(a): int(b) for a, b in zip(u, c)}
        self.positions = positions
        self.cell = cell
        self.pbc = pbc

        if properties is None:
            properties = Properties()
        self.properties = properties

        # cache
        self._cache_e: Optional[list[Environment]] = None

    @property
    def requires_grad(self) -> bool:
        return self.positions.requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool) -> None:
        self.positions.requires_grad = value
        self.cell.requires_grad = value
