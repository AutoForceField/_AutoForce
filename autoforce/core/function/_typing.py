# +
from typing import Callable, Set

from ..dataclass import Tensor, TensorDict

Descriptor_fn = Callable[[Tensor, Tensor, Tensor, Set[int]], TensorDict]
