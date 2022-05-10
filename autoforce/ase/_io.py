# +
from typing import Any, Union

import ase
import torch
from ase.io import read as _read
from ase.neighborlist import wrap_positions

import autoforce.cfg as cfg
from autoforce.core.dataclasses import Conf, Target


def from_atoms(atoms: ase.Atoms) -> Conf:
    """
    Generates a data.Conf object from a ase.Atoms object.

    """

    # 1.
    numbers = torch.from_numpy(atoms.numbers)
    wrapped = wrap_positions(atoms.positions, atoms.cell, atoms.pbc)
    positions = torch.from_numpy(wrapped).to(cfg.float_t)
    cell = torch.from_numpy(atoms.cell.array).to(cfg.float_t)

    # 2.
    e, f = None, None
    if atoms.calc:
        if "energy" in atoms.calc.results:
            e = atoms.get_potential_energy()
            e = torch.tensor(e).to(cfg.float_t)
        if "forces" in atoms.calc.results:
            f = atoms.get_forces()
            f = torch.from_numpy(f).to(cfg.float_t)
    target = Target(energy=e, forces=f)

    # 3.
    conf = Conf(numbers, positions, cell, atoms.pbc, target=target)

    return conf


def read(*args: Any, **kwargs: Any) -> Union[Conf, list[Conf]]:
    """
    Reads Atoms and converts them to Conf.
    """
    data: Union[ase.Atoms, list[ase.Atoms]] = _read(*args, **kwargs)
    if type(data) == list:
        return [from_atoms(x) for x in data]
    elif type(data) == ase.Atoms:
        return from_atoms(data)
    else:
        raise RuntimeError("!")
