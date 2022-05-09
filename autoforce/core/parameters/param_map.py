# +
from typing import Optional, Union

from .proxy_map import DefaultProxyMap, SortedKey, TupleKey

__all__ = ["ParamMap", "SymmetricParamMap"]


# typing aliases:
ParamType = float
KeyType = Union[int, tuple[int, ...]]
MappingType = dict[KeyType, ParamType]


class ParamMap(TupleKey, DefaultProxyMap):
    def __init__(
        self,
        default: ParamType,
        defined: Optional[MappingType] = None,
    ) -> None:
        if defined is None:
            defined = {}
        super().__init__(default, defined)


class SymmetricParamMap(SortedKey, ParamMap):
    pass
