# +
from typing import Callable, Dict, Set, Tuple, Union

from torch import Tensor

Key = Union[int, Tuple[int, ...]]
TensorDict = Dict[Key, Tensor]

#
Descriptor_fn = Callable[[Tensor, Tensor, Tensor, Set[int]], TensorDict]
