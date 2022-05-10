# +
from autoforce.core.functions import Cutoff_fn
from autoforce.core.modules import Descriptor
from autoforce.core.parameters import ParameterMapping
from autoforce.functions.overlaps import Overlaps


class SOAP(Descriptor):
    def __init__(
        self, cutoff: ParameterMapping, cutoff_fn: Cutoff_fn, lmax: int, nmax: int
    ) -> None:
        super().__init__(cutoff, cutoff_fn, Overlaps(lmax, nmax))
