# +
from abc import ABC, abstractmethod
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import torch
from numpy import ndarray

from .dataclasses import Conf, LocalEnv
from .parameters import Cutoff


class NeighborList(ABC):
    def __init__(self, cutoff: Optional[Cutoff] = None) -> None:
        self.set_cutoff(cutoff)

    def set_cutoff(self, cutoff: Cutoff) -> None:
        self.cutoff = cutoff

    @abstractmethod
    def get_neighborlist(
        self,
        cutoff: Dict,
        pbc: List[bool],
        cell: ndarray,
        positions: ndarray,
        atomic_numbers: ndarray,
    ) -> (ndarray, ndarray, ndarray):
        ...

    def get_local_envs(self, conf: Conf) -> List[LocalEnv]:
        if conf._cache_e is None:
            self.build_local_envs(conf)
        return conf._cache_e

    def build_local_envs(self, conf: Conf) -> None:
        # 1. Get neighborlist:
        unique = conf.unique_counts.keys()
        i, j, sij = self.get_neighborlist(
            self.cutoff.as_dict(unique, float),
            conf.pbc,
            conf.cell.detach().numpy(),
            conf.positions.detach().numpy(),
            conf.numbers.numpy(),
        )
        # 2. Displacements rij:
        sij = torch.from_numpy(sij)
        shifts = (sij[..., None] * conf.cell).sum(dim=1)
        rij = conf.positions[j] - conf.positions[i] + shifts

        # 3. Split:
        # Note that neighborlist is already sorted wrt i
        sizes = np.bincount(i, minlength=conf.number_of_atoms).tolist()
        i = torch.from_numpy(i).split(sizes)
        j = torch.from_numpy(j).split(sizes)
        rij = rij.split(sizes)

        # 4. Cache
        conf._cache_e = []
        _cached_isolated_atoms = Counter()
        for k in range(conf.number_of_atoms):
            if sizes[k] == 0:
                _cached_isolated_atoms[int(conf.numbers[k])] += 1
            else:
                _i = i[k][0]
                env = LocalEnv(_i, conf.numbers[_i], j[k], conf.numbers[j[k]], rij[k])
                conf._cache_e.append(env)
