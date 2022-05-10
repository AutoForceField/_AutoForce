# +
from __future__ import annotations

from ase.neighborlist import primitive_neighbor_list
from numpy import ndarray

from autoforce.core.neighborlist import NeighborList


class ASE_NeighborList(NeighborList):
    def get_neighborlist(
        self,
        cutoff: dict,
        pbc: list[bool],
        cell: ndarray,
        positions: ndarray,
        atomic_numbers: ndarray,
    ) -> tuple[ndarray, ndarray, ndarray]:

        i, j, sij = primitive_neighbor_list(
            "ijS", pbc, cell, positions, cutoff, numbers=atomic_numbers
        )

        return i, j, sij
