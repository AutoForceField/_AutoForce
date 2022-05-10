# +
from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor

import autoforce.cfg as cfg
from autoforce.core.dataclasses import Conf, Target
from autoforce.core.modules import Regressor


class Shift(Regressor):
    """
    TODO:

    """

    @property
    def cutoff(self) -> None:
        return None

    def set_weights(self, weights: Tensor, sections: tuple[int, ...]) -> None:
        self.weights = {s: a for s, a in zip(sections, weights)}

    def get_design_matrix(
        self, confs: Sequence[Conf]
    ) -> tuple[Tensor, Tensor, tuple[int, ...]]:
        _sections: set[int] = set()
        for conf in confs:
            _sections.update(conf.unique_counts.keys())
        sections = tuple(_sections)
        dim = len(sections)
        index = {s: i for i, s in enumerate(sections)}
        _e = []
        f_len = 0
        for conf in confs:
            v = dim * [0]
            for z, c in conf.unique_counts.items():
                v[index[z]] = c
            _e.append(v)
            f_len += conf.number_of_atoms
        e = torch.tensor(_e, dtype=cfg.float_t)
        f = torch.zeros(3 * f_len, dim, dtype=cfg.float_t)
        return e, f, sections

    def get_target(self, conf):
        e = 0
        for number, count in conf.unique_counts.items():
            e = e + self.weights[number] * count
        return Target(energy=e, forces=0)
