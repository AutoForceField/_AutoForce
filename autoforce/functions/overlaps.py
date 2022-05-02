# +
from math import factorial as fac
from math import sqrt
from typing import Dict, Set, Tuple

import torch
from torch import Tensor

import autoforce.cfg as cfg
from autoforce.core import Descriptor_fn
from autoforce.functions.harmonics import Harmonics


class Overlaps(Descriptor_fn):
    def __init__(self, lmax: int, nmax: int) -> None:
        super().__init__()
        self.harmonics = Harmonics(lmax)
        self.nmax = nmax

        # Auxiliary Tensors
        self._nnl = _nnl(lmax, nmax)
        self._pow_2n = 2 * torch.arange(nmax + 1).to(torch.int)
        self._l = self.harmonics.l.reshape(-1).to(torch.int64)
        self._c_m = torch.eye(lmax + 1).neg().add(2).to(torch.int)
        self.ui, self.uj, self._2sq = _2sq(nmax)
        self.compress = True

    @property
    def lmax(self) -> int:
        return self.harmonics.lmax

    def function(
        self, rij: Tensor, wj: Tensor, numbers: Tensor, unique: Set[int]
    ) -> Dict[Tuple[int, int], Tensor]:

        # Dimensions:
        # _[s][n][l][m][j] -> [ns][nmax+1][lmax+1][lmax+1][nj]

        # 1. Mappings
        d_j = rij.norm(dim=1)
        q_j = (-0.5 * d_j**2).exp()
        if wj is not None:
            q_j = q_j * wj
        y_lmj = self.harmonics.function(rij)

        # 2. Radial & Angular Coupling
        r_nj = q_j * d_j[None] ** self._pow_2n[:, None]
        f_nlmj = r_nj[:, None, None] * y_lmj[None]

        # 3. Density per species
        c_snlm = []
        for k, z in enumerate(unique):
            c_snlm.append(f_nlmj[..., numbers == z].sum(dim=-1))
        c_snlm = torch.stack(c_snlm)

        # 4. Sum over m (c_snlm & c^*_snlm product)
        result = {}
        nnl = torch.zeros(
            self.nmax + 1, self.nmax + 1, self.lmax + 1, dtype=cfg.float_t
        )
        for a, a_nlm in zip(unique, c_snlm):
            for b, b_nlm in zip(unique, c_snlm):
                nnlm = (a_nlm[None] * b_nlm[:, None] * self._c_m).flatten(-2, -1)
                res = nnl.index_add(2, self._l, nnlm) * self._nnl
                if self.compress:
                    res = (res[self.ui, self.uj] * self._2sq).flatten()
                result[(a, b)] = res
        return result


def _nnl(lmax: int, nmax: int) -> Tensor:
    a = torch.tensor(
        [
            [
                1 / (sqrt(2 * l + 1) * 2 ** (2 * n + l) * fac(n) * fac(n + l))
                for l in range(lmax + 1)
            ]
            for n in range(nmax + 1)
        ]
    )
    return (a[None] * a[:, None]).sqrt().to(cfg.float_t)


def _2sq(nmax: int) -> (Tensor, Tensor, Tensor):
    ui, uj = torch.triu_indices(nmax + 1, nmax + 1)
    sq = (2.0 - torch.eye(nmax + 1, dtype=cfg.float_t)).sqrt()
    return ui, uj, sq[ui, uj, None]
