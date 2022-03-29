# +
import autoforce.cfg as cfg
from .dataclasses import Conf, Basis
from .descriptor import Descriptor
from .parameter import ChemPar
import torch
from torch import Tensor
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional


class Kernel(ABC):

    def __init__(self,
                 descriptor: Descriptor,
                 exponent: ChemPar
                 ) -> None:
        self.descriptor = descriptor
        self.exponent = exponent

    @abstractmethod
    def forward(self,
                uv: Tensor,
                u: Tensor,
                v: Tensor
                ) -> Tensor:
        """
        uv:   scalar products matrix <u_i,v_j> with shape (m, n)
        u:    norms sqrt(<u_i,u_i>) with shape (m, 1)
        v:    norms sqrt(<v_j,v_j>) with shape (1, n)

        output:
              a tensor with the same shape as uv
        """
        ...

    def kernel(self,
               s: int,
               uv: Tensor,
               u: Tensor,
               v: Tensor
               ) -> Tensor:
        return self.forward(uv, u, v)**self.exponent[s]

    def get_potential_energy(self,
                             conf: Conf,
                             basis: Basis,
                             weights: Dict
                             ) -> Tensor:
        basis_norms = basis.norms()
        products, norms = self.descriptor.get_scalar_products_dict(conf, basis)
        energy = 0
        for species, prod in products.items():
            k = self.kernel(species,
                            torch.stack([torch.stack(a) for a in prod]),
                            torch.stack(norms[species]).view(-1, 1),
                            torch.stack(basis_norms[species]).view(1, -1),
                            )
            energy = energy + (k @ weights[species]).sum()
        return energy

    def get_design_matrix(self,
                          confslist: List[Conf],
                          basis: Basis
                          ) -> (Tuple, Tensor, Tensor):
        Ke = []
        Kf = []
        basis_count = tuple((s, c) for s, c in basis.count().items())
        for conf in confslist:
            species_matrix = self.get_design_dict(conf, basis)
            ke = []
            kf = []
            for species, count in basis_count:
                if species in species_matrix:
                    e, f = species_matrix[species]
                else:
                    e = torch.zeros(1, count, dtype=cfg.float_t)
                    f = torch.zeros(conf.positions.numel(),
                                    count, dtype=cfg.float_t)
                ke.append(e)
                kf.append(f)
            Ke.append(torch.cat(ke, dim=1))
            Kf.append(torch.cat(kf, dim=1))
        Ke = torch.cat(Ke)
        Kf = torch.cat(Kf)
        return basis_count, Ke, Kf

    def get_design_dict(self,
                        conf: Conf,
                        basis: Basis
                        ) -> Dict:
        species_matrix = {}
        basis_norms = basis.norms()
        products, norms = self.descriptor.get_scalar_products_dict(conf, basis)
        for species in products.keys():
            kern = []
            kern_grad = []
            species_norms = torch.stack(norms[species]).view(1, -1)
            for a in zip(*products[species], basis_norms[species]):
                k = self.kernel(species,
                                torch.stack(a[:-1]).view(1, -1),
                                a[-1].view(1, 1),
                                species_norms,
                                ).sum()
                dk, = torch.autograd.grad(k,
                                          conf.positions,
                                          retain_graph=True)
                kern.append(k.detach())
                kern_grad.append(dk.view(-1, 1))
            kern = torch.stack(kern).view(1, -1)
            kern_grad = torch.cat(kern_grad, dim=1)
            species_matrix[species] = (kern, -kern_grad)
        return species_matrix

    def get_basis_overlaps_dict(self,
                                basis: Basis
                                ) -> Dict:
        gram_dict = self.descriptor.get_gram_dict(basis)
        basis_norms = basis.norms()
        for species, gram in gram_dict.items():
            norms = torch.stack(basis_norms[species])
            gram_dict[species] = self.kernel(species,
                                             gram,
                                             norms.view(1, -1),
                                             norms.view(-1, 1))
        return gram_dict
