# +
import itertools
from collections import defaultdict
from typing import Dict, List, Tuple, Union

import torch
from torch import Tensor

import autoforce.cfg as cfg

from ..dataclasses import Basis, Conf, LocalDes, LocalEnv
from ..functions import Cutoff_fn, Descriptor_fn
from ..parameters import Cutoff


def scalar_product(
    x: Dict[Union[int, Tuple[int, ...]], Tensor],
    y: Dict[Union[int, Tuple[int, ...]], Tensor],
) -> Tensor:
    keys = set(x.keys()).intersection(set(y.keys()))
    p = cfg.zero
    for k in keys:
        p = p + (x[k] * y[k]).sum()
    return p


class Descriptor:
    """
    A Descriptor converts a "LocalEnv" object
    into a "LocalDes" object:

        Descriptor: LocalEnv -> LocalDes

    See LocalEnv and LocalDes in core.dataclass.


    Required methods:
        1) descriptor
        2) scalar_product


    The "descriptor" method:
    The body of a Descriptor should be implemented
    as the "descriptor" method. The main component of
    a LocalDes object is the "tensors" attribute.
    "meta" are arbirtary data for index-keeping,
    etc which are handled opaquely by the Descriptor.
    Other attributes are "index", "species", and "norm"
    which are automatically handled and, generally,
    should not be specified within this method.


    The "scalar_product" method:
    Returns the scalar product of two LocalDes objects.
    In particlar, the "norm" of a LocalDes object is
    defined as the square root of its scalar product
    by itself.

    """

    instances = 0

    def __init__(
        self, cutoff: Cutoff, cutoff_fn: Cutoff_fn, descriptor_fn: Descriptor_fn
    ) -> None:

        self.cutoff = cutoff
        self.cutoff_fn = cutoff_fn
        self.descriptor_fn = descriptor_fn

        # Assign a global index for this instance
        self.index = Descriptor.instances
        Descriptor.instances += 1
        self.basis = tuple()

    def descriptor(
        self, number: Tensor, numbers: Tensor, rij: Tensor, cij: Tensor
    ) -> Dict[Union[int, Tuple[int, ...]], Tensor]:
        dij = rij.norm(dim=1)
        m = dij < cij
        wij = self.cutoff_fn.function(dij[m], cij[m])
        unique = set(numbers[m].tolist())
        d = self.descriptor_fn.function(rij[m], wij, numbers[m], unique)
        return d

    def get_descriptor(self, e: LocalEnv) -> LocalDes:
        while len(e._cache_d) < Descriptor.instances:
            e._cache_d.append(None)
        if e._cache_d[self.index] is None:
            cij = self.cutoff(e.number, e.numbers)
            _species = int(e.number)
            _d = self.descriptor(e.number, e.numbers, e.rij, cij)
            _norm = scalar_product(_d, _d).sqrt().view([])
            d = LocalDes(_species, _d, _norm)
            e._cache_d[self.index] = d
        return e._cache_d[self.index]

    def get_descriptors(self, conf: Conf) -> List[LocalDes]:
        if conf._cache_e is None:
            raise RuntimeError(f"{conf._cache_e = }")
        return [self.get_descriptor(l) for l in conf._cache_e]

    def get_scalar_products(self, d: LocalDes, basis: Basis) -> List[Tensor]:
        # 1. update cache: d._cache_p
        while len(d._cache_p) <= basis.index:
            d._cache_p.append([])
        m = len(d._cache_p[basis.index])
        new = [
            scalar_product(base.tensors, d.tensors) if active else None
            for base, active in zip(
                basis.descriptors[d.species][m:], basis.active[d.species][m:]
            )
        ]
        d._cache_p[basis.index].extend(new)

        # 2. retrieve from cache
        out = itertools.compress(d._cache_p[basis.index], basis.active[d.species])
        return list(out)

    def get_scalar_products_dict(self, conf: Conf, basis: Basis) -> (Dict, Dict):
        prod = defaultdict(list)
        norms = defaultdict(list)
        for d in self.get_descriptors(conf):
            k = self.get_scalar_products(d, basis)
            prod[d.species].append(k)
            norms[d.species].append(d.norm)
        return prod, norms

    def get_gram_dict(self, basis: Basis) -> Dict:
        gram = {}
        for species, descriptors in basis.descriptors.items():
            z = itertools.compress(descriptors, basis.active[species])
            gram[species] = torch.stack(
                [torch.stack(self.get_scalar_products(b, basis)) for b in z]
            )
        return gram

    def new_basis(self) -> int:
        new = Basis()
        new.index = len(self.basis)
        self.basis = (*self.basis, new)
        return new