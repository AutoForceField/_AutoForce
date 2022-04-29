# +
from typing import Dict, Tuple, Union

from torch import Tensor

Key = Union[int, Tuple[int, ...]]
TensorDict = Dict[Key, Tensor]
