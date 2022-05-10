# +
import torch
from torch import Tensor

from autoforce.core.functions import Bijection_fn


class Identity_bi(Bijection_fn):
    def function(self, x: Tensor) -> Tensor:
        return x

    def inverse(self, x: Tensor) -> Tensor:
        return x


class Range_bi(Bijection_fn):
    def __init__(self, a: float, b: float) -> None:
        assert b > a
        self.a = a
        self.l = b - a

    def function(self, x: Tensor) -> Tensor:
        y = self.a + self.l * torch.special.expit(torch.as_tensor(x))
        return y

    def inverse(self, y: Tensor) -> Tensor:
        x = (y - self.a) / self.l
        return torch.special.logit(torch.as_tensor(x))
