# +
from __future__ import annotations

from typing import Any

import ase
import torch
from ase.io import read as _read
from ase.neighborlist import wrap_positions

import autoforce.cfg as cfg
from autoforce.core.dataclasses import Properties, Structure


def from_atoms(atoms: ase.Atoms) -> Structure:
    """
    Generates a data.Structure object from a ase.Atoms object.

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
    properties = Properties(energy=e, forces=f)

    # 3.
    struc = Structure(numbers, positions, cell, atoms.pbc, properties=properties)

    return struc


def read(*args: Any, **kwargs: Any) -> Structure | list[Structure]:
    """
    Reads Atoms and converts them to Structure.
    """
    data: ase.Atoms | list[ase.Atoms] = _read(*args, **kwargs)
    if type(data) == list:
        return [from_atoms(x) for x in data]
    elif type(data) == ase.Atoms:
        return from_atoms(data)
    else:
        raise RuntimeError("!")
