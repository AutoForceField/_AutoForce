# +
from __future__ import annotations

from typing import Optional

from torch import Tensor


class Target:

    __slots__ = ("energy", "forces")

    def __init__(
        self, *, energy: Optional[Tensor] = None, forces: Optional[Tensor] = None
    ) -> None:
        """
        Self explanatory.

        """
        self.energy = energy
        self.forces = forces
