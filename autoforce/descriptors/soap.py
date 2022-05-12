# +
from __future__ import annotations

from autoforce.core.functions import SoftZero_fn
from autoforce.core.modules import Descriptor
from autoforce.core.parameters import ParameterMapping
from autoforce.functions.overlaps import Overlaps


class SOAP(Descriptor):
    def __init__(
        self, cutoff: ParameterMapping, softzero_fn: SoftZero_fn, lmax: int, nmax: int
    ) -> None:
        super().__init__(cutoff, softzero_fn, Overlaps(lmax, nmax))
