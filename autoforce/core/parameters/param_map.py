# +
from collections import defaultdict
from itertools import product, repeat
from typing import Iterable, Iterator, Mapping, Optional, Sized, Union

__all__ = ["ParamMap", "SymParamMap"]


# typing aliases:
SpeciesType = int
ParamType = float
KeyType = Union[SpeciesType, tuple[SpeciesType, ...]]
MappingType = dict[KeyType, ParamType]
TupleKeyType = tuple[SpeciesType, ...]
TupleMappingType = dict[TupleKeyType, ParamType]


class ParamMap(Mapping):
    def __init__(
        self,
        default: ParamType,
        defined: Optional[MappingType] = None,
        indices: Optional[int] = None,
    ) -> None:
        if defined is None:
            defined = {}
        if indices is None:
            try:
                (indices,) = {len(_tuple(k)) for k in defined}
            except ValueError:
                raise RuntimeError("Num indices is not unique or can't be inferred!")

        def callable_() -> ParamType:
            return default

        self.indices: int = indices
        dict_: TupleMappingType = {self._key(k): v for k, v in defined.items()}
        self.dict = defaultdict(callable_, dict_)

    # public methods:
    def broadcast(
        self, *args: Union[SpeciesType, Iterable[SpeciesType]]
    ) -> tuple[ParamType, ...]:
        assert len(args) == self.indices, f"Num args should be {self.indices}!"
        assert any(
            tuple(isinstance(arg, Sized) for arg in args)
        ), "At least one of args should be Sized!"
        bcast = tuple(arg if isinstance(arg, Iterable) else repeat(arg) for arg in args)
        ret = tuple(self.dict[self._key(k)] for k in zip(*bcast))
        # timings: ~ 100 us
        return ret

    def product(self, species: set[SpeciesType]) -> MappingType:
        return {
            _int(self._key(k)): self[k] for k in product(species, repeat=self.indices)
        }

    # Mapping methods:
    def __getitem__(self, k: KeyType) -> ParamType:
        return self.dict[self._key(k)]

    def __len__(self) -> int:
        return len(self.dict)

    def __iter__(self) -> Iterator[TupleKeyType]:
        return iter(self.dict)

    # private methods:
    def _key(self, k: KeyType) -> TupleKeyType:
        key = (k,) if isinstance(k, int) else k
        # TODO: remove assert
        assert len(key) == self.indices, f"Length of key should be {self.indices}!"
        return key


class SymParamMap(ParamMap):

    # override:
    def _key(self, k: KeyType) -> TupleKeyType:
        return tuple(sorted(super()._key(k)))


def _tuple(k: KeyType) -> TupleKeyType:
    return (k,) if isinstance(k, int) else k


def _int(k: TupleKeyType) -> KeyType:
    return k if len(k) > 1 else k[0]
