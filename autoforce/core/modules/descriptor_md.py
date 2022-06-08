# +
from __future__ import annotations

import itertools

import torch
from torch import Tensor

import autoforce.cfg as cfg

from ..dataclasses import Descriptor, Environment, Structure
from ..functions import Descriptor_fn, SoftZero_fn
from ..parameters import ParameterMapping


class Descriptor_md:

    instances = 0

    def __init__(
        self,
        cutoff: ParameterMapping,
        softzero_fn: SoftZero_fn,
        descriptor_fn: Descriptor_fn,
    ) -> None:

        self.cutoff = cutoff
        self.softzero_fn = softzero_fn
        self.descriptor_fn = descriptor_fn

        # Assign a global index for this instance
        self.index = Descriptor_md.instances
        Descriptor_md.instances += 1

    def descriptor(
        self, number: int, numbers: list[int], rij: Tensor, cij: Tensor
    ) -> dict[tuple[int, ...], Tensor]:
        dij = rij.norm(dim=1)
        m = dij < cij
        wij = self.softzero_fn.function(1 - dij[m] / cij[m])
        within = list(itertools.compress(numbers, m))
        unique = set(within)
        _numbers = torch.tensor(within)
        d = self.descriptor_fn.function(rij[m], wij, _numbers, unique)
        return d

    def get_descriptor(self, e: Environment) -> Descriptor:
        while len(e._cache_d) < Descriptor_md.instances:
            e._cache_d.append(None)
        d = e._cache_d[self.index]
        if d is None:
            number = int(e.number)
            numbers = e.numbers.tolist()
            cij = torch.as_tensor(self.cutoff.broadcast((number, numbers)))
            _d = self.descriptor(number, numbers, e.rij, cij)
            _norm = self.scalar_product(_d, _d).sqrt().view([])
            d = Descriptor(number, _d, _norm)
            e._cache_d[self.index] = d
        return d

    def get_descriptors(self, struc: Structure) -> list[Descriptor]:
        if struc._cache_e is None:
            raise RuntimeError(f"{struc._cache_e = }")
        return [self.get_descriptor(l) for l in struc._cache_e]

    @staticmethod
    def scalar_product(
        x: dict[tuple[int, ...], Tensor],
        y: dict[tuple[int, ...], Tensor],
    ) -> Tensor:
        keys = set(x.keys()).intersection(set(y.keys()))
        p = cfg.zero
        for k in keys:
            p = p + (x[k] * y[k]).sum()
        return p

    @staticmethod
    def vjp(
        v: dict[tuple[int, ...], Tensor],
        j: dict[tuple[int, ...], Tensor],
    ) -> Tensor:
        keys = set(v.keys()).intersection(set(j.keys()))
        p = cfg.zero
        for k in keys:
            p = p + (v[k][:, None, None] * j[k]).sum(dim=0)
        return p
