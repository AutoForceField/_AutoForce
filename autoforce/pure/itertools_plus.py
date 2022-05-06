# +
import itertools
from typing import Any, Iterable, Optional, Tuple


def broadcast(
    *args: Tuple[Any, ...], times: Optional[int] = None
) -> Tuple[Iterable, ...]:
    """
    Replaces non-Iterable args with itertools.repeat(arg [,times]).

    Warning:
    The common usage is zip(*broadcast(*args)) which may result
    in an infinite Iterable if times == None and all args are
    non-Iterable.

    """
    if times is None:
        return tuple(
            arg if isinstance(arg, Iterable) else itertools.repeat(arg) for arg in args
        )
    else:
        return tuple(
            arg if isinstance(arg, Iterable) else itertools.repeat(arg, times)
            for arg in args
        )
