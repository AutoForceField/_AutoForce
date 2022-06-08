# +
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter

import numpy as np
import torch
from numpy import ndarray

from .dataclasses import Environment, Structure
from .parameters import ParameterMapping


class NeighborList(ABC):
    def __init__(self, cutoff: ParameterMapping) -> None:
        self.cutoff = cutoff

    @abstractmethod
    def get_neighborlist(
        self,
        cutoff: dict[tuple[int, int], float],
        pbc: list[bool],
        cell: ndarray,
        positions: ndarray,
        atomic_numbers: ndarray,
    ) -> tuple[ndarray, ndarray, ndarray]:
        ...

    def get_local_envs(self, struc: Structure) -> list[Environment]:
        if struc._cache_e is None:
            struc._cache_e = self.build_local_envs(struc)
        return struc._cache_e

    def build_local_envs(self, struc: Structure) -> list[Environment]:
        # 1. Get neighborlist:
        unique = struc.unique_counts.keys()
        _i, _j, _sij = self.get_neighborlist(
            self.cutoff.product(unique, repeat=2),
            struc.pbc,
            struc.cell.detach().numpy(),
            struc.positions.detach().numpy(),
            struc.numbers.numpy(),
        )
        # 2. Displacements rij:
        i = torch.from_numpy(_i)
        j = torch.from_numpy(_j)
        sij = torch.from_numpy(_sij)
        shifts = (sij[..., None] * struc.cell).sum(dim=1)
        rij = struc.positions[j] - struc.positions[i] + shifts

        # 3. Split:
        # Note that neighborlist is already sorted wrt i
        sizes = np.bincount(_i, minlength=struc.number_of_atoms).tolist()
        i = i.split(sizes)
        j = j.split(sizes)
        rij = rij.split(sizes)

        # 4. Cache
        _cache_e: list[Environment] = []
        _cached_isolated_atoms: dict[int, int] = Counter()
        for k in range(struc.number_of_atoms):
            if sizes[k] == 0:
                _cached_isolated_atoms[int(struc.numbers[k])] += 1
            else:
                _ii = i[k][0]
                env = Environment(
                    _ii, struc.numbers[_ii], j[k], struc.numbers[j[k]], rij[k]
                )
                _cache_e.append(env)
        return _cache_e
