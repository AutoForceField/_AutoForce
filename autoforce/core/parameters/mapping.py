# +
from __future__ import annotations

from abc import abstractmethod
from collections import defaultdict
from collections.abc import MutableMapping
from itertools import product, repeat
from typing import Any, Hashable, Iterable, Iterator, Optional, Sized


class ParameterMapping(MutableMapping):
    def __init__(self, dict_: dict | defaultdict) -> None:
        self._dict = dict_.copy()
        self._dict.clear()
        for k, v in dict_.items():
            self._dict[self._key(k)] = v

    @abstractmethod
    def _key(self, key: Any) -> Hashable:
        ...

    def __getitem__(self, key: Any) -> Any:
        return self._dict[self._key(key)]

    def __setitem__(self, key: Any, value: Any) -> None:
        self._dict[self._key(key)] = value

    def __delitem__(self, key: Any) -> None:
        del self._dict[self._key(key)]

    def __len__(self) -> int:
        return len(self._dict)

    def __iter__(self) -> Iterator:
        return iter(self._dict)

    def __repr__(self) -> str:
        return repr(self._dict)

    def broadcast(self, args: Iterable) -> tuple[Any, ...]:
        """
        args = (1, (1, 2)) --> values for (1, 1), (1, 2)
        """
        # For avoiding infinite loops:
        sized = tuple(isinstance(arg, Sized) for arg in args)
        assert any(sized), "At least one of args should be Sized!"
        #
        bcast = tuple(arg if isinstance(arg, Iterable) else repeat(arg) for arg in args)
        return tuple(self[key] for key in zip(*bcast))

    def product(self, args: Iterable, repeat: int = 1) -> dict:
        """
        args = (1, 2), repeat = 2 --> a dict from keys (1, 1), (1, 2), (2, 1), (2, 2)
        """
        keys = (self._key(k) for k in product(args, repeat=repeat))
        return {key: self._dict[key] for key in keys}


class DefaultParameterMapping(ParameterMapping):
    def __init__(self, default: Any, dict_: Optional[dict] = None) -> None:
        def callable_():
            return default

        _dict = defaultdict(callable_, {} if dict_ is None else dict_)
        super().__init__(_dict)


class TupleKey:
    def _key(self, key: Any) -> tuple:
        return key if isinstance(key, tuple) else (key,)


class SortedKey:
    def _key(self, key: Any) -> tuple:
        return tuple(sorted(key)) if isinstance(key, tuple) else (key,)


class AsymmetricMapping(TupleKey, DefaultParameterMapping):
    pass


class SymmetricMapping(SortedKey, DefaultParameterMapping):
    pass
