# +
from __future__ import annotations

import itertools
from collections import defaultdict
from typing import Sequence

import torch
from torch import Tensor

import autoforce.cfg as cfg
from autoforce.core.dataclasses import Descriptor, Properties, Structure
from autoforce.core.functions import Kernel_fn
from autoforce.core.modules import Geometry, Regressor
from autoforce.core.parameters import ParameterMapping

from .basis import Basis


class Kernel_reg(Regressor):
    """
    TODO:

    """

    def __init__(
        self, descriptor: Geometry, kernel: Kernel_fn, exponent: ParameterMapping
    ) -> None:
        """
        TODO:

        """
        self.descriptor = descriptor
        self.basis = Basis()
        self.basis.index = 0
        self.kernel = kernel
        self.exponent = exponent

    @property
    def cutoff(self) -> ParameterMapping:
        return self.descriptor.cutoff

    def get_design_matrix(
        self,
        structures: Sequence[Structure],
    ) -> tuple[Tensor, Tensor, tuple[tuple[int, int], ...]]:
        Ke = []
        Kf = []
        sections = tuple((s, c) for s, c in self.basis.count().items())
        for struc in structures:
            design_dict = self.get_design_dict(struc)
            ke = []
            kf = []
            for species, count in sections:
                if species in design_dict:
                    e, f = design_dict[species]
                else:
                    e = torch.zeros(1, count, dtype=cfg.float_t)
                    f = torch.zeros(struc.positions.numel(), count, dtype=cfg.float_t)
                ke.append(e)
                kf.append(f)
            Ke.append(torch.cat(ke, dim=1))
            Kf.append(torch.cat(kf, dim=1))
        Ke_cat = torch.cat(Ke)
        Kf_cat = torch.cat(Kf)
        return Ke_cat, Kf_cat, sections

    def get_design_dict(
        self,
        struc: Structure,
    ) -> dict:
        design_dict = {}
        basis_norms = self.basis.norms()
        products, norms = self.get_scalar_products_dict(struc)
        for species in products.keys():
            kern = []
            kern_grad = []
            species_norms = torch.stack(norms[species]).view(1, -1)
            for a in zip(*products[species], basis_norms[species]):
                k = (
                    self.kernel.function(
                        torch.stack(a[:-1]).view(1, -1),
                        a[-1].view(1, 1),
                        species_norms,
                    )
                    .pow(self.exponent[species])
                    .sum()
                )
                (dk,) = torch.autograd.grad(k, struc.positions, retain_graph=True)
                kern.append(k.detach())
                kern_grad.append(dk.view(-1, 1))
            kern_cat = torch.stack(kern).view(1, -1)
            kern_grad_cat = torch.cat(kern_grad, dim=1)
            design_dict[species] = (kern_cat, -kern_grad_cat)
        return design_dict

    def set_weights(
        self, weights: Tensor, sections: tuple[tuple[int, int], ...]
    ) -> None:
        """
        TODO:

        """
        species, count = zip(*sections)
        weights = torch.split(weights, count)
        self.weights = {s: w for s, w in zip(species, weights)}

    def predict(self, struc: Structure) -> Properties:
        """
        TODO:

        """
        basis_norms = self.basis.norms()
        products, norms = self.get_scalar_products_dict(struc)
        energy = cfg.zero
        for species, prod in products.items():
            k = self.kernel.function(
                torch.stack([torch.stack(a) for a in prod]),
                torch.stack(norms[species]).view(-1, 1),
                torch.stack(basis_norms[species]).view(1, -1),
            ).pow(self.exponent[species])
            energy = energy + (k @ self.weights[species]).sum()
        if energy.grad_fn:
            (g,) = torch.autograd.grad(energy, struc.positions, retain_graph=True)
        else:
            g = cfg.zero
        return Properties(energy=energy.detach(), forces=-g)

    # ------------------------
    def get_scalar_products_dict(self, struc: Structure) -> tuple[dict, dict]:
        prod = defaultdict(list)
        norms = defaultdict(list)
        for d in self.descriptor.get_descriptors(struc):
            k = self.get_scalar_products(d)
            prod[d.species].append(k)
            norms[d.species].append(d.norm)
        return prod, norms

    def get_scalar_products(self, d: Descriptor) -> list[Tensor]:
        basis = self.basis
        # 1. update cache: d._cache_p
        while len(d._cache_p) <= basis.index:
            d._cache_p.append([])
        m = len(d._cache_p[basis.index])
        new = [
            self.descriptor.scalar_product(base.tensors, d.tensors) if active else None
            for base, active in zip(
                basis.descriptors[d.species][m:], basis.active[d.species][m:]
            )
        ]
        d._cache_p[basis.index].extend(new)

        # 2. retrieve from cache
        out = itertools.compress(d._cache_p[basis.index], basis.active[d.species])
        return list(out)
