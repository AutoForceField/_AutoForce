# +
from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor

import autoforce.cfg as cfg

from ..dataclasses import Conf, Target
from ..functions import Kernel_fn
from ..parameters import ParameterMapping
from .descriptor import Descriptor
from .regressor import Regressor


class KernelRegressor(Regressor):
    """
    TODO:

    """

    def __init__(
        self, descriptor: Descriptor, kernel: Kernel_fn, exponent: ParameterMapping
    ) -> None:
        """
        TODO:

        """
        self.descriptor = descriptor
        self.basis = descriptor.new_basis()
        self.kernel = kernel
        self.exponent = exponent

    @property
    def cutoff(self) -> ParameterMapping:
        return self.descriptor.cutoff

    def get_design_matrix(
        self,
        confs: Sequence[Conf],
    ) -> tuple[Tensor, Tensor, tuple[tuple[int, int], ...]]:
        Ke = []
        Kf = []
        sections = tuple((s, c) for s, c in self.basis.count().items())
        for conf in confs:
            design_dict = self.get_design_dict(conf)
            ke = []
            kf = []
            for species, count in sections:
                if species in design_dict:
                    e, f = design_dict[species]
                else:
                    e = torch.zeros(1, count, dtype=cfg.float_t)
                    f = torch.zeros(conf.positions.numel(), count, dtype=cfg.float_t)
                ke.append(e)
                kf.append(f)
            Ke.append(torch.cat(ke, dim=1))
            Kf.append(torch.cat(kf, dim=1))
        Ke_cat = torch.cat(Ke)
        Kf_cat = torch.cat(Kf)
        return Ke_cat, Kf_cat, sections

    def get_design_dict(
        self,
        conf: Conf,
    ) -> dict:
        design_dict = {}
        basis_norms = self.basis.norms()
        products, norms = self.descriptor.get_scalar_products_dict(conf, self.basis)
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
                (dk,) = torch.autograd.grad(k, conf.positions, retain_graph=True)
                kern.append(k.detach())
                kern_grad.append(dk.view(-1, 1))
            kern_cat = torch.stack(kern).view(1, -1)
            kern_grad_cat = torch.cat(kern_grad, dim=1)
            design_dict[species] = (kern_cat, -kern_grad_cat)
        return design_dict

    def get_basis_overlaps(self) -> dict:
        gram_dict = self.descriptor.get_gram_dict(self.basis)
        basis_norms = self.basis.norms()
        for species, gram in gram_dict.items():
            norms = torch.stack(basis_norms[species])
            gram_dict[species] = self.kernel.function(
                gram, norms.view(1, -1), norms.view(-1, 1)
            ).pow(self.exponent[species])
        return gram_dict

    def set_weights(
        self, weights: Tensor, sections: tuple[tuple[int, int], ...]
    ) -> None:
        """
        TODO:

        """
        species, count = zip(*sections)
        weights = torch.split(weights, count)
        self.weights = {s: w for s, w in zip(species, weights)}

    def get_target(self, conf: Conf) -> Target:
        """
        TODO:

        """
        basis_norms = self.basis.norms()
        products, norms = self.descriptor.get_scalar_products_dict(conf, self.basis)
        energy = cfg.zero
        for species, prod in products.items():
            k = self.kernel.function(
                torch.stack([torch.stack(a) for a in prod]),
                torch.stack(norms[species]).view(-1, 1),
                torch.stack(basis_norms[species]).view(1, -1),
            ).pow(self.exponent[species])
            energy = energy + (k @ self.weights[species]).sum()
        if energy.grad_fn:
            (g,) = torch.autograd.grad(energy, conf.positions, retain_graph=True)
        else:
            g = cfg.zero
        return Target(energy=energy.detach(), forces=-g)
