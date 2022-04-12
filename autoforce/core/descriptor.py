# +
import itertools
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List

import torch

import autoforce.cfg as cfg
from autoforce._typing import Tensor, TensorDict

from .dataclass import Basis, Conf, LocalDes, LocalEnv
from .function import Cutoff_fn
from .parameter import Cutoff


class Descriptor(ABC):
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

    def __init__(self, cutoff: Cutoff, cutoff_fn: Cutoff_fn) -> None:

        self.cutoff = cutoff
        self.cutoff_fn = cutoff_fn

        # Assign a global index for this instance
        self.index = Descriptor.instances
        Descriptor.instances += 1
        self.basis = tuple()

    @abstractmethod
    def descriptor(
        self, number: Tensor, numbers: Tensor, rij: Tensor, wij: Tensor
    ) -> TensorDict:
        """
        Should be implemented in a subclass.

        """
        ...

    def _descriptor(
        self, number: Tensor, numbers: Tensor, rij: Tensor, cij: Tensor
    ) -> TensorDict:
        dij = rij.norm(dim=1)
        m = dij < cij
        wij = self.cutoff_fn.function(dij[m], cij[m])
        d = self.descriptor(number, numbers[m], rij[m], wij)
        return d

    def get_descriptor(self, e: LocalEnv) -> LocalDes:
        while len(e._cached_descriptors) < Descriptor.instances:
            e._cached_descriptors.append(None)
        if e._cached_descriptors[self.index] is None:
            cij = self.cutoff(e.number, e.numbers)
            _d = self._descriptor(e.number, e.numbers, e.rij, cij)
            d = LocalDes(_d)
            d.norm = self.scalar_product(d, d).sqrt().view([])
            d.species = int(e.number)
            e._cached_descriptors[self.index] = d
        return e._cached_descriptors[self.index]

    def get_descriptors(self, conf: Conf) -> List[LocalDes]:
        if conf._cached_local_envs is None:
            raise RuntimeError(f"{conf._cached_local_envs = }")
        return [self.get_descriptor(l) for l in conf._cached_local_envs]

    def scalar_product(self, x: LocalDes, y: LocalDes) -> Tensor:
        kx = set(x.descriptor.keys())
        ky = set(y.descriptor.keys())
        product = cfg.zero
        for k in kx.intersection(ky):
            product = product + (x.descriptor[k] * y.descriptor[k]).sum()
        return product

    def get_scalar_products(self, d: LocalDes, basis: Basis) -> List[Tensor]:
        # 1. update cache: d._cached_scalar_products
        while len(d._cached_scalar_products) <= basis.index:
            d._cached_scalar_products.append([])
        m = len(d._cached_scalar_products[basis.index])
        new = [
            self.scalar_product(base, d) if active else None
            for base, active in zip(
                basis.descriptors[d.species][m:], basis.active[d.species][m:]
            )
        ]
        d._cached_scalar_products[basis.index].extend(new)

        # 2. retrieve from cache
        out = itertools.compress(
            d._cached_scalar_products[basis.index], basis.active[d.species]
        )
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
