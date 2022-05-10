# +
from __future__ import annotations

from torch import Tensor


class LocalDes:
    """
    TODO: doc

    """

    __slots__ = ("species", "tensors", "norm", "_cache_p")

    def __init__(
        self,
        species: int,
        tensors: dict[tuple[int, ...], Tensor],
        norm: Tensor,
    ) -> None:
        """
        species      species of the descriptor
        tensors      a dict of {key: tensor} (main data)
        norm         norm of the descriptor

        """

        self.species = species
        self.tensors = tensors
        self.norm = norm

        # cache
        self._cache_p: list[list[Tensor | None]] = []

    def detach(self) -> "LocalDes":
        tensors = {k: t.detach() for k, t in self.tensors.items()}
        detached = LocalDes(self.species, tensors, self.norm.detach())
        detached._cache_p = [
            [None if t is None else t.detach() for t in wrt] for wrt in self._cache_p
        ]
        return detached
