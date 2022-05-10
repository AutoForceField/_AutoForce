# +
from abc import ABC, abstractmethod
from collections import Counter

import numpy as np
import torch
from numpy import ndarray

from .dataclasses import Conf, LocalEnv
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

    def get_local_envs(self, conf: Conf) -> list[LocalEnv]:
        if conf._cache_e is None:
            conf._cache_e = self.build_local_envs(conf)
        return conf._cache_e

    def build_local_envs(self, conf: Conf) -> list[LocalEnv]:
        # 1. Get neighborlist:
        unique = conf.unique_counts.keys()
        _i, _j, _sij = self.get_neighborlist(
            self.cutoff.product(unique, repeat=2),
            conf.pbc,
            conf.cell.detach().numpy(),
            conf.positions.detach().numpy(),
            conf.numbers.numpy(),
        )
        # 2. Displacements rij:
        i = torch.from_numpy(_i)
        j = torch.from_numpy(_j)
        sij = torch.from_numpy(_sij)
        shifts = (sij[..., None] * conf.cell).sum(dim=1)
        rij = conf.positions[j] - conf.positions[i] + shifts

        # 3. Split:
        # Note that neighborlist is already sorted wrt i
        sizes = np.bincount(_i, minlength=conf.number_of_atoms).tolist()
        i = i.split(sizes)
        j = j.split(sizes)
        rij = rij.split(sizes)

        # 4. Cache
        _cache_e: list[LocalEnv] = []
        _cached_isolated_atoms: dict[int, int] = Counter()
        for k in range(conf.number_of_atoms):
            if sizes[k] == 0:
                _cached_isolated_atoms[int(conf.numbers[k])] += 1
            else:
                _ii = i[k][0]
                env = LocalEnv(_ii, conf.numbers[_ii], j[k], conf.numbers[j[k]], rij[k])
                _cache_e.append(env)
        return _cache_e
