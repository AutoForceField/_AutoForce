# +
from __future__ import annotations

import torch
from torch import Tensor

import autoforce.cfg as cfg
from autoforce.core.functions import SoftZero_fn


class Polynomial_sz(SoftZero_fn):
    def __init__(self, n: int = 2):
        assert n > 1
        self.n = n

    def function(self, x: Tensor) -> Tensor:
        return x**self.n


class Sine_sz(SoftZero_fn):
    def function(self, x: Tensor) -> Tensor:
        return torch.sin(x * cfg.pi / 2)
