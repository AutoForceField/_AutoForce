# +
from __future__ import annotations

import abc
from typing import Sequence

import torch

import autoforce.cfg as cfg

from ..dataclasses import Properties, Structure
from ..parameters import ParameterMapping
from .regression import Regression
from .regressor import Regressor

__all__ = ["Model", "RegressionModel"]


class Model(abc.ABC):
    @property
    @abc.abstractmethod
    def cutoff(self) -> ParameterMapping | None:
        ...

    @abc.abstractmethod
    def fit(self, structures: Sequence[Structure]) -> None:
        ...

    @abc.abstractmethod
    def predict(self, struc: Structure) -> Properties:
        ...


class RegressionModel(Model):
    """
    TODO:

    """

    def __init__(self, regressors: Sequence[Regressor], regression: Regression) -> None:
        """
        TODO:

        """
        self.regressors = regressors
        self.regression = regression

    @property
    def cutoff(self) -> ParameterMapping:
        raise NotImplementedError("!")

    def fit(self, structures: Sequence[Structure]) -> None:
        """
        TODO:

        """
        # 1. targets
        _energies = []
        _forces = []
        for struc in structures:
            # TODO:
            if struc.properties.energy is not None:
                _energies.append(struc.properties.energy)
            if struc.properties.forces is not None:
                _forces.append(struc.properties.forces)
        energies = torch.stack(_energies)
        forces = torch.stack(_forces).view(-1)
        targets = torch.cat([energies, forces])

        # 2. design matrix
        matrices = []
        dims = []
        sections = []
        for reg in self.regressors:
            e, f, sec = reg.get_design_matrix(structures)
            matrices.append((e, f))
            dims.append(int(e.size(1)))
            sections.append(sec)
        design_matrix = torch.cat([torch.cat(m, dim=1) for m in zip(*matrices)])

        # 3. solve
        opt_mae = None
        for _ in range(7):
            sol = self.regression.fit(design_matrix, targets)
            mae = (design_matrix @ sol - targets).abs().mean()
            if opt_mae is None or mae < opt_mae:
                opt_mae = mae
                solution = sol
            if self.regression.is_deterministic:
                break

        # 4. weights
        weights = torch.split(solution, dims)
        for reg, w, sec in zip(self.regressors, weights, sections):
            reg.set_weights(w, sec)

    def predict(self, struc: Structure) -> Properties:
        """
        TODO:

        """
        t = Properties(
            energy=torch.tensor(0.0, dtype=cfg.float_t),
            forces=torch.zeros_like(struc.positions),
        )
        for reg in self.regressors:
            _t = reg.predict(struc)
            t.energy += _t.energy  # type: ignore
            t.forces += _t.forces  # type: ignore
        return t
