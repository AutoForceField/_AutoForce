# +
from typing import Any, Sequence

import torch

from .dataclass import Conf, Target
from .parameter import ReducedPar


class Model:
    """
    TODO:

    """

    def __init__(self, *regressors: Any) -> None:
        """
        TODO:

        """
        self.regressors = regressors
        self._cutoff = None

    @property
    def cutoff(self) -> ReducedPar:
        if self._cutoff is None:
            self._cutoff = ReducedPar(op=max)
            for reg in self.regressors:
                if reg.cutoff:
                    self._cutoff.include(reg.cutoff)
        return self._cutoff

    def fit(self, confs: Sequence[Conf]) -> None:
        """
        TODO:

        """
        # 1. Targets
        energies = []
        forces = []
        for conf in confs:
            energies.append(conf.target.energy)
            forces.append(conf.target.forces)
        energies = torch.stack(energies)
        forces = torch.stack(forces).view(-1)
        targets = torch.cat([energies, forces])

        # 2.
        matrices = []
        dims = []
        sections = []
        for reg in self.regressors:
            e, f, sec = reg.get_design_matrix(confs)
            matrices.append((e, f))
            dims.append(int(e.size(1)))
            sections.append(sec)

        matrix = torch.cat([torch.cat(m, dim=1) for m in zip(*matrices)])

        # 3. torch.lstsq is random -> best of 7 maybe semi-deterministic
        opt_mae = None
        for _ in range(7):
            sol = torch.linalg.lstsq(matrix, targets).solution
            mae = (matrix @ sol - targets).abs().mean()
            if opt_mae is None or mae < opt_mae:
                opt_mae = mae
                solution = sol

        weights = torch.split(solution, dims)
        for reg, w, sec in zip(self.regressors, weights, sections):
            reg.set_weights(w, sec)

    def get_target(self, conf: Conf) -> Target:
        """
        TODO:

        """
        t = Target(energy=0, forces=0)
        for reg in self.regressors:
            _t = reg.get_target(conf)
            t.energy += _t.energy
            t.forces += _t.forces
        return t
