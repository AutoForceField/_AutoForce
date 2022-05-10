# +
import autoforce.core as core
from autoforce.functions import Overlaps


class SOAP(core.Descriptor):
    def __init__(
        self, cutoff: core.Cutoff, cutoff_fn: core.Cutoff_fn, lmax: int, nmax: int
    ) -> None:
        super().__init__(cutoff, cutoff_fn, Overlaps(lmax, nmax))
