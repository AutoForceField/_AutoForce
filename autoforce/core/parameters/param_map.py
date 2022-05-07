# +
from collections import defaultdict
from itertools import repeat
from typing import Dict, Iterable, Iterator, Mapping, Optional, Tuple, Union

__all__ = ["ParamMap", "SymParamMap"]


# typing aliases:
SpeciesType = int
KeyType = Union[SpeciesType, Tuple[SpeciesType, ...]]
ParamType = float
MappingType = Dict[KeyType, ParamType]
TupleKeyType = Tuple[SpeciesType, ...]
TupleMappingType = Dict[TupleKeyType, ParamType]


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
                (indices,) = {len(int_to_tuple(k)) for k in defined}
            except ValueError:
                raise RuntimeError("Num indices is not unique or can't be inferred!")

        def callable_() -> ParamType:
            return default

        self.indices: int = indices
        dict_: TupleMappingType = {self._key(k): v for k, v in defined.items()}
        self.dict = defaultdict(callable_, dict_)

    # Mapping methods:
    def __getitem__(self, k: KeyType) -> ParamType:
        return self.dict[self._key(k)]

    def __len__(self) -> int:
        return len(self.dict)

    def __iter__(self) -> Iterator[TupleKeyType]:
        return iter(self.dict)

    # public methods:
    ...

    # private methods
    def _key(self, k: KeyType) -> TupleKeyType:
        key = (k,) if isinstance(k, int) else k
        assert len(key) == self.indices, f"len(key) should be {self.indices}!"
        return key

    def _broadcast(
        self, *args: Union[SpeciesType, Iterable[SpeciesType]]
    ) -> Tuple[ParamType, ...]:
        assert len(args) == self.indices, f"Num args should be {self.indices}!"
        # assert any(
        #    tuple(isinstance(arg, Sized) for arg in args)
        # ), "At least one of args should be Sized"
        bcast = tuple(arg if isinstance(arg, Iterable) else repeat(arg) for arg in args)
        ret = tuple(self.dict[self._key(k)] for k in zip(*bcast))
        return ret


class SymParamMap(ParamMap):

    # override:
    def _key(self, k: KeyType) -> TupleKeyType:
        return tuple(sorted(super()._key(k)))


def int_to_tuple(k: KeyType) -> TupleKeyType:
    return (k,) if isinstance(k, int) else k
